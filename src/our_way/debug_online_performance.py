# -*- coding: utf-8 -*-
"""
Debug script to identify why real-time performance is poor compared to offline.
This script compares offline vs online preprocessing and model predictions.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import Preprocessor, preprocess, create_windows_from_events, exponential_moving_standardize
from braindecode.models import ShallowFBCSPNet
from online_standardizer import OnlineExponentialStandardizer
import mne
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd

def load_offline_data():
    """Load and preprocess data exactly as in training."""
    print("Loading offline data...")

    dataset = MOABBDataset("BNCI2014_001", subject_ids=[3])

    included_channels = [
        'C3', 'C4', 'Cz', 'FC1', 'FC2', 'FCz', 'CP1', 'CP2', 'CPz',
        'P1', 'P2', 'Pz', 'C1', 'C2', 'CP3', 'CP4'
    ]

    # Exact training preprocessing
    preprocessors = [
        Preprocessor('pick_channels', ch_names=included_channels, ordered=True),
        Preprocessor(lambda data: data * 1e6),
        Preprocessor('resample', sfreq=125),
        Preprocessor('filter', l_freq=4, h_freq=38),
        Preprocessor(exponential_moving_standardize, factor_new=1e-3, init_block_size=250)
    ]

    preprocess(dataset, preprocessors, n_jobs=1)

    # Create windows
    input_window_samples = 250
    sfreq = dataset.datasets[0].raw.info['sfreq']
    trial_start_offset_samples = int(-0.5 * sfreq)

    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=0,
        window_size_samples=input_window_samples,
        window_stride_samples=1,
        drop_last_window=False,
        preload=True
    )

    splitted = windows_dataset.split('session')
    test_set = splitted['1test']

    return test_set, dataset

def simulate_online_preprocessing(test_set, dataset):
    """Simulate online preprocessing on the same data."""
    print("Simulating online preprocessing...")

    # Get raw data before preprocessing
    raw_dataset = MOABBDataset("BNCI2014_001", subject_ids=[3])
    raw = raw_dataset.datasets[0].raw

    included_channels = [
        'C3', 'C4', 'Cz', 'FC1', 'FC2', 'FCz', 'CP1', 'CP2', 'CPz',
        'P1', 'P2', 'Pz', 'C1', 'C2', 'CP3', 'CP4'
    ]

    # Pick channels and resample
    raw.pick_channels(included_channels)
    if raw.info['sfreq'] != 125:
        raw.resample(125)

    raw_data = raw.get_data()

    # Initialize online standardizer
    standardizer = OnlineExponentialStandardizer(
        n_channels=len(included_channels),
        factor_new=1e-3,
        init_block_size=250
    )

    # Feed first 1000 samples for calibration
    for i in range(1000):
        if i < raw_data.shape[1]:
            sample = raw_data[:, i] * 1e6  # Scale to microvolts
            standardizer.feed_sample(sample)

    print(f"Online standardizer initialized: {standardizer.initialized}")

    # Process windows using online preprocessing
    online_results = []
    input_window_samples = 250

    for i in range(len(test_set)):
        # Get the same time window from raw data
        window_start = i * input_window_samples
        window_end = window_start + input_window_samples

        if window_end > raw_data.shape[1]:
            break

        # Extract raw window
        raw_window = raw_data[:, window_start:window_end]

        # Apply online preprocessing
        # 1. Scale to microvolts
        scaled_window = raw_window * 1e6

        # 2. Filter (simplified - in real implementation would be more complex)
        filtered_window = mne.filter.filter_data(
            scaled_window,
            sfreq=125,
            l_freq=4,
            h_freq=38,
            method='iir',
            picks=None,
            verbose=False
        )

        # 3. Standardize
        if standardizer.initialized:
            standardized_window = standardizer.standardize_window(filtered_window)
        else:
            standardized_window = filtered_window

        online_results.append(standardized_window)

    return online_results

def compare_preprocessing(offline_windows, online_windows):
    """Compare offline vs online preprocessing."""
    print("\nComparing preprocessing...")

    differences = []
    correlations = []

    for i in range(min(len(offline_windows), len(online_windows))):
        offline = offline_windows[i]
        online = online_windows[i]

        # Calculate difference
        diff = np.abs(offline - online)
        differences.append(np.mean(diff))

        # Calculate correlation
        corr = np.corrcoef(offline.flatten(), online.flatten())[0, 1]
        correlations.append(corr)

    print(f"Mean absolute difference: {np.mean(differences):.6f}")
    print(f"Mean correlation: {np.mean(correlations):.6f}")
    print(f"Min correlation: {np.min(correlations):.6f}")
    print(f"Max correlation: {np.max(correlations):.6f}")

    return differences, correlations

def test_model_predictions(model, offline_windows, online_windows, test_set):
    """Test model predictions on both offline and online preprocessed data."""
    print("\nTesting model predictions...")

    device = next(model.parameters()).device

    offline_predictions = []
    online_predictions = []
    true_labels = []

    model.eval()
    with torch.no_grad():
        for i in range(min(len(offline_windows), len(online_windows))):
            # Get true label
            _, y, _ = test_set[i]
            true_labels.append(y)

            # Offline prediction
            offline_tensor = torch.tensor(offline_windows[i], dtype=torch.float32, device=device).unsqueeze(0)
            offline_logits = model(offline_tensor)
            if offline_logits.ndim == 3:
                offline_logits = offline_logits.mean(dim=2)
            offline_probs = torch.softmax(offline_logits, dim=1)
            offline_pred = torch.argmax(offline_probs, dim=1).cpu().numpy()[0]
            offline_predictions.append(offline_pred)

            # Online prediction
            online_tensor = torch.tensor(online_windows[i], dtype=torch.float32, device=device).unsqueeze(0)
            online_logits = model(online_tensor)
            if online_logits.ndim == 3:
                online_logits = online_logits.mean(dim=2)
            online_probs = torch.softmax(online_logits, dim=1)
            online_pred = torch.argmax(online_probs, dim=1).cpu().numpy()[0]
            online_predictions.append(online_pred)

    # Calculate accuracies
    offline_acc = accuracy_score(true_labels, offline_predictions)
    online_acc = accuracy_score(true_labels, online_predictions)

    print(f"Offline accuracy: {offline_acc*100:.2f}%")
    print(f"Online accuracy: {online_acc*100:.2f}%")
    print(f"Accuracy drop: {(offline_acc - online_acc)*100:.2f}%")

    # Check prediction agreement
    agreement = np.mean(np.array(offline_predictions) == np.array(online_predictions))
    print(f"Prediction agreement: {agreement*100:.2f}%")

    return offline_predictions, online_predictions, true_labels

def analyze_errors(offline_preds, online_preds, true_labels):
    """Analyze where the errors occur."""
    print("\nAnalyzing errors...")

    # Find where predictions disagree
    disagreements = np.where(np.array(offline_preds) != np.array(online_preds))[0]

    print(f"Number of prediction disagreements: {len(disagreements)}")
    print(f"Disagreement rate: {len(disagreements)/len(offline_preds)*100:.2f}%")

    if len(disagreements) > 0:
        print("\nSample disagreements:")
        for i in disagreements[:10]:  # Show first 10
            print(f"  Sample {i}: True={true_labels[i]}, Offline={offline_preds[i]}, Online={online_preds[i]}")

    # Analyze by class
    labels = ['feet', 'left_hand', 'right_hand', 'tongue']
    for i, label in enumerate(labels):
        class_mask = np.array(true_labels) == i
        if np.any(class_mask):
            offline_class_acc = accuracy_score(
                np.array(true_labels)[class_mask],
                np.array(offline_preds)[class_mask]
            )
            online_class_acc = accuracy_score(
                np.array(true_labels)[class_mask],
                np.array(online_preds)[class_mask]
            )
            print(f"  {label}: Offline={offline_class_acc*100:.1f}%, Online={online_class_acc*100:.1f}%")

def create_visualizations(offline_windows, online_windows, differences, correlations):
    """Create visualizations to help debug."""
    print("\nCreating visualizations...")

    # Create output directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    save_dir = os.path.join(project_root, 'log', 'debug_online')
    os.makedirs(save_dir, exist_ok=True)

    # 1. Preprocessing comparison for first few windows
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for i in range(min(3, len(offline_windows))):
        # Offline
        axes[0, i].imshow(offline_windows[i], aspect='auto', cmap='viridis')
        axes[0, i].set_title(f'Offline Window {i+1}')
        axes[0, i].set_ylabel('Channels')

        # Online
        axes[1, i].imshow(online_windows[i], aspect='auto', cmap='viridis')
        axes[1, i].set_title(f'Online Window {i+1}')
        axes[1, i].set_ylabel('Channels')
        axes[1, i].set_xlabel('Time')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'preprocessing_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Difference and correlation histograms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.hist(differences, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('Mean Absolute Difference')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Preprocessing Differences')
    ax1.grid(True, alpha=0.3)

    ax2.hist(correlations, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('Correlation')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Preprocessing Correlations')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'preprocessing_stats.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Visualizations saved to {save_dir}")

def main():
    """Main debugging function."""
    print("Debugging Online Performance Issues")
    print("="*50)

    # Load model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    model_path = os.path.join(project_root, 'models', 'subj3_model_250_full.pth')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.serialization.add_safe_globals([ShallowFBCSPNet])
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device).eval()

    # Load data
    test_set, dataset = load_offline_data()

    # Get offline windows
    offline_windows = []
    for i in range(len(test_set)):
        x, y, _ = test_set[i]
        offline_windows.append(x)

    # Simulate online preprocessing
    online_windows = simulate_online_preprocessing(test_set, dataset)

    # Compare preprocessing
    differences, correlations = compare_preprocessing(offline_windows, online_windows)

    # Test predictions
    offline_preds, online_preds, true_labels = test_model_predictions(
        model, offline_windows, online_windows, test_set
    )

    # Analyze errors
    analyze_errors(offline_preds, online_preds, true_labels)

    # Create visualizations
    create_visualizations(offline_windows, online_windows, differences, correlations)

    print("\n✅ Debugging completed!")
    print("Check the visualizations in log/debug_online/ for insights.")

if __name__ == "__main__":
    main()