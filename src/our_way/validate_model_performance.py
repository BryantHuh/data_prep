# -*- coding: utf-8 -*-
"""
Simple validation script to test model performance on test set.
This uses the exact same preprocessing as training to verify the model works correctly.
"""

import os
import numpy as np
import torch
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import Preprocessor, preprocess, create_windows_from_events, exponential_moving_standardize
from braindecode.models import ShallowFBCSPNet
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

def load_and_preprocess_data():
    """Load and preprocess data exactly as in training."""
    print("Loading and preprocessing data...")

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
        Preprocessor(exponential_moving_standardize, factor_new=1e-3, init_block_size=1000)
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

    return test_set

def load_model(model_path, device):
    """Load the trained model."""
    print(f"Loading model from {model_path}...")

    torch.serialization.add_safe_globals([ShallowFBCSPNet])
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device).eval()

    return model

def predict_on_dataset(model, test_set, device):
    """Make predictions on the test set."""
    print("Making predictions...")

    predictions = []
    true_labels = []
    confidences = []

    model.eval()
    with torch.no_grad():
        for i in range(len(test_set)):
            x, y, _ = test_set[i]

            # Convert to tensor
            x_tensor = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)

            # Get predictions
            logits = model(x_tensor)
            if logits.ndim == 3:
                logits = logits.mean(dim=2)

            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).cpu().numpy()[0]
            conf = torch.max(probs, dim=1)[0].cpu().numpy()[0]

            predictions.append(pred)
            true_labels.append(y)
            confidences.append(conf)

    return predictions, true_labels, confidences

def analyze_results(predictions, true_labels, confidences):
    """Analyze the results."""
    print("\nResults Analysis:")
    print("="*50)

    # Overall accuracy
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Overall Accuracy: {accuracy*100:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    print(f"\nConfusion Matrix:")
    print(cm)

    # Per-class accuracy
    labels = ['feet', 'left_hand', 'right_hand', 'tongue']
    for i, label in enumerate(labels):
        class_mask = np.array(true_labels) == i
        if np.any(class_mask):
            class_acc = accuracy_score(
                np.array(true_labels)[class_mask],
                np.array(predictions)[class_mask]
            )
            print(f"{label}: {class_acc*100:.1f}%")

    # Confidence statistics
    print(f"\nConfidence Statistics:")
    print(f"Mean confidence: {np.mean(confidences):.3f}")
    print(f"Std confidence: {np.std(confidences):.3f}")
    print(f"Min confidence: {np.min(confidences):.3f}")
    print(f"Max confidence: {np.max(confidences):.3f}")

    return accuracy, cm

def main():
    """Main validation function."""
    print("Model Performance Validation")
    print("="*40)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    test_set = load_and_preprocess_data()
    print(f"Test set size: {len(test_set)}")

    # Load model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    model_path = os.path.join(project_root, 'models', 'subj3_model_250_full.pth')

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    model = load_model(model_path, device)

    # Make predictions
    predictions, true_labels, confidences = predict_on_dataset(model, test_set, device)

    # Analyze results
    accuracy, cm = analyze_results(predictions, true_labels, confidences)

    print(f"\n✅ Validation completed!")
    print(f"Model achieves {accuracy*100:.1f}% accuracy on test set.")

if __name__ == "__main__":
    main()