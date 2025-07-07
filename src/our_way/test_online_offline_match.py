# -*- coding: utf-8 -*-
"""
Test script to validate that online standardization matches offline preprocessing.
This ensures that training and inference use identical preprocessing steps.
"""

import numpy as np
import torch
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import Preprocessor, preprocess, create_windows_from_events, exponential_moving_standardize
from braindecode.models import ShallowFBCSPNet
from online_standardizer import OnlineExponentialStandardizer
import matplotlib.pyplot as plt
import os

def test_online_offline_match():
    """Test that online standardization matches offline preprocessing."""

    print("🧪 Testing Online vs Offline Standardization Match")
    print("=" * 60)

    # Parameters
    subject_id = 3
    sfreq = 125
    input_window_samples = 250
    included_channels = [
        'C3', 'C4', 'Cz',
        'FC1', 'FC2', 'FCz',
        'CP1', 'CP2', 'CPz',
        'P1', 'P2', 'Pz',
        'C1', 'C2',
        'CP3', 'CP4'
    ]

    # Load raw data
    print("📊 Loading MOABB dataset...")
    dataset = MOABBDataset("BNCI2014_001", subject_ids=[subject_id])

    # Get raw data before preprocessing
    raw = getattr(dataset.datasets[0], '_raw', None)
    if raw is None:
        raw = getattr(dataset.datasets[0], 'raw', None)
    if raw is None:
        raise AttributeError('Could not find raw or _raw attribute in dataset.datasets[0]')

    raw.pick_channels(included_channels)
    if raw.info['sfreq'] != sfreq:
        raw.resample(sfreq)

    raw_data = raw.get_data(picks=included_channels)
    print(f"Raw data shape: {raw_data.shape}")

    # Step 1: Apply offline preprocessing
    print("\n🔄 Step 1: Applying offline preprocessing...")
    offline_preprocessors = [
        Preprocessor('pick_channels', ch_names=included_channels, ordered=True),
        Preprocessor(lambda data: data * 1e6),  # Scale to microvolts
        Preprocessor('resample', sfreq=sfreq),
        Preprocessor('filter', l_freq=4, h_freq=38),
        Preprocessor(
            exponential_moving_standardize,
            apply_on_array=False,  # Online-compatible
            factor_new=1e-3,
            init_block_size=1000
        )
    ]

    offline_dataset = MOABBDataset("BNCI2014_001", subject_ids=[subject_id])
    preprocess(offline_dataset, offline_preprocessors, n_jobs=1)

    # Get a window from offline preprocessed data
    trial_start_offset_samples = int(-0.5 * sfreq)
    offline_windows = create_windows_from_events(
        offline_dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=0,
        window_size_samples=input_window_samples,
        window_stride_samples=input_window_samples,
        drop_last_window=False,
        preload=True
    )

    # Get the first window
    offline_window = offline_windows[0][0].numpy()
    print(f"Offline window shape: {offline_window.shape}")
    print(f"Offline window stats - min: {offline_window.min():.6f}, max: {offline_window.max():.6f}, mean: {offline_window.mean():.6f}, std: {offline_window.std():.6f}")

    # Step 2: Apply online preprocessing to the same data
    print("\n🔄 Step 2: Applying online preprocessing...")

    # Initialize online standardizer
    online_standardizer = OnlineExponentialStandardizer(
        n_channels=len(included_channels),
        factor_new=1e-3,
        init_block_size=1000
    )

    # Get the same window from raw data
    window_start = 0
    window_end = window_start + input_window_samples
    raw_window = raw_data[:, window_start:window_end]

    # Apply online preprocessing step by step
    # Step 2a: Scale to microvolts
    scaled_window = raw_window * 1e6

    # Step 2b: Apply bandpass filtering (simplified - in real implementation use proper filtering)
    # For this test, we'll skip filtering to focus on standardization
    filtered_window = scaled_window

    # Step 2c: Feed samples to online standardizer for calibration
    print("   Calibrating online standardizer...")
    for i in range(1000):  # Use first 1000 samples for calibration
        if i < raw_data.shape[1]:
            sample = raw_data[:, i] * 1e6  # Scale sample
            online_standardizer.feed_sample(sample)

    print(f"   Calibration progress: {online_standardizer.get_calibration_progress():.1f}%")

    # Step 2d: Apply online standardization to the window
    online_window = online_standardizer.standardize_window(filtered_window)
    print(f"Online window shape: {online_window.shape}")
    print(f"Online window stats - min: {online_window.min():.6f}, max: {online_window.max():.6f}, mean: {online_window.mean():.6f}, std: {online_window.std():.6f}")

    # Step 3: Compare results
    print("\n🔍 Step 3: Comparing results...")

    # Calculate differences
    max_diff = np.max(np.abs(offline_window - online_window))
    mean_diff = np.mean(np.abs(offline_window - online_window))
    correlation = np.corrcoef(offline_window.flatten(), online_window.flatten())[0, 1]

    print(f"Maximum absolute difference: {max_diff:.6f}")
    print(f"Mean absolute difference: {mean_diff:.6f}")
    print(f"Correlation coefficient: {correlation:.6f}")

    # Step 4: Test model predictions
    print("\n🤖 Step 4: Testing model predictions...")

    # Load model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    model_path = os.path.join(project_root, 'models', 'subj3_model_250_full.pth')

    if os.path.exists(model_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.serialization.add_safe_globals([ShallowFBCSPNet])
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.to(device).eval()

        # Test predictions
        with torch.no_grad():
            # Offline window prediction
            x_offline = torch.tensor(offline_window, dtype=torch.float32, device=device).unsqueeze(0)
            output_offline = model(x_offline)
            if output_offline.ndim == 3:
                output_offline = output_offline.mean(dim=2)
            # The model already has a softmax layer, so output contains probabilities
            probs_offline = output_offline.cpu().numpy().squeeze()
            pred_offline = int(np.argmax(probs_offline))
            conf_offline = float(np.max(probs_offline))

            # Online window prediction
            x_online = torch.tensor(online_window, dtype=torch.float32, device=device).unsqueeze(0)
            output_online = model(x_online)
            if output_online.ndim == 3:
                output_online = output_online.mean(dim=2)
            # The model already has a softmax layer, so output contains probabilities
            probs_online = output_online.cpu().numpy().squeeze()
            pred_online = int(np.argmax(probs_online))
            conf_online = float(np.max(probs_online))

        class_names = ['feet', 'left_hand', 'right_hand', 'tongue']

        print(f"Offline prediction: {class_names[pred_offline]} (confidence: {conf_offline:.3f})")
        print(f"Online prediction:  {class_names[pred_online]} (confidence: {conf_online:.3f})")
        print(f"Prediction match: {'✅' if pred_offline == pred_online else '❌'}")
        print(f"Confidence difference: {abs(conf_offline - conf_online):.6f}")

    else:
        print("⚠️  Model not found, skipping prediction test")

    # Step 5: Summary
    print("\n📋 Summary:")
    print("=" * 60)

    if max_diff < 1e-3 and correlation > 0.99:
        print("✅ SUCCESS: Online and offline preprocessing match!")
        print("   The online standardization implementation is correct.")
        print("   Training and inference will use identical preprocessing.")
    else:
        print("❌ FAILURE: Online and offline preprocessing do not match!")
        print("   There may be an issue with the online implementation.")
        print("   Check the standardization algorithm.")

    print(f"\n📊 Metrics:")
    print(f"   Max difference: {max_diff:.6f} {'✅' if max_diff < 1e-3 else '❌'}")
    print(f"   Correlation: {correlation:.6f} {'✅' if correlation > 0.99 else '❌'}")

    # Plot comparison
    print("\n📈 Generating comparison plot...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Plot 1: Raw data
    axes[0, 0].plot(raw_window[0, :])
    axes[0, 0].set_title('Raw EEG (Channel 0)')
    axes[0, 0].set_ylabel('Amplitude (V)')

    # Plot 2: Scaled data
    axes[0, 1].plot(scaled_window[0, :])
    axes[0, 1].set_title('Scaled EEG (Channel 0)')
    axes[0, 1].set_ylabel('Amplitude (μV)')

    # Plot 3: Offline vs Online
    axes[1, 0].plot(offline_window[0, :], label='Offline', alpha=0.8)
    axes[1, 0].plot(online_window[0, :], label='Online', alpha=0.8)
    axes[1, 0].set_title('Standardized EEG (Channel 0)')
    axes[1, 0].set_ylabel('Standardized Amplitude')
    axes[1, 0].legend()

    # Plot 4: Difference
    axes[1, 1].plot(offline_window[0, :] - online_window[0, :])
    axes[1, 1].set_title('Difference (Offline - Online)')
    axes[1, 1].set_ylabel('Difference')
    axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(project_root, 'log', 'online_offline_comparison.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()

    print(f"📊 Comparison plot saved to: {plot_path}")

    return max_diff < 1e-3 and correlation > 0.99

if __name__ == "__main__":
    success = test_online_offline_match()
    if success:
        print("\n🎉 All tests passed! Online standardization is working correctly.")
    else:
        print("\n⚠️  Tests failed. Please check the implementation.")