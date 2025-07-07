# -*- coding: utf-8 -*-
"""
Debug script to compare offline vs real-time preprocessing.
"""

import numpy as np
import torch
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import Preprocessor, preprocess, create_windows_from_events, exponential_moving_standardize
from braindecode.models import ShallowFBCSPNet
import mne
from online_standardizer import OnlineExponentialStandardizer
import matplotlib.pyplot as plt

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

# Load model
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
model_path = os.path.join(project_root, 'models', 'subj3_model_250_full.pth')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.serialization.add_safe_globals([ShallowFBCSPNet])
model = torch.load(model_path, map_location=device, weights_only=False)
model.to(device).eval()

# Load dataset
dataset = MOABBDataset("BNCI2014_001", subject_ids=[subject_id])

# OFFLINE PREPROCESSING
print("=== OFFLINE PREPROCESSING ===")
preprocessors = [
    Preprocessor('pick_channels', ch_names=included_channels, ordered=True),
    Preprocessor(lambda data: data * 1e6),
    Preprocessor('resample', sfreq=sfreq),
    Preprocessor('filter', l_freq=4, h_freq=38),
    Preprocessor(
        exponential_moving_standardize,
        apply_on_array=True,
        factor_new=1e-3,
        init_block_size=250
    )
]
preprocess(dataset, preprocessors, n_jobs=1)

# Create windows
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

# Get first window from offline
first = test_set[0]
if isinstance(first, tuple) and len(first) == 3:
    offline_window, y_true, meta = first
elif isinstance(first, tuple) and len(first) == 2:
    offline_window, y_true = first
    meta = None
else:
    raise ValueError('Unexpected return value from test_set[0]')
# Ensure offline_window is a numpy array
if not isinstance(offline_window, np.ndarray):
    offline_window = np.array(offline_window)
print(f"Offline window shape: {offline_window.shape}")
print(f"Offline window stats - min: {offline_window.min():.6f}, max: {offline_window.max():.6f}, mean: {offline_window.mean():.6f}, std: {offline_window.std():.6f}")

# Print label mapping if available
window_kwargs = getattr(getattr(test_set.datasets[0], 'window_kwargs', None), '__getitem__', lambda x: None)(0)
if window_kwargs and isinstance(window_kwargs, tuple) and len(window_kwargs) > 1 and 'mapping' in window_kwargs[1]:
    mapping = window_kwargs[1]['mapping']
    print(f"Offline label mapping: {mapping}")
else:
    print("No label mapping found in window_kwargs.")

# Test offline prediction
x_tensor = torch.tensor(offline_window, dtype=torch.float32, device=device).unsqueeze(0)
with torch.no_grad():
    logits = model(x_tensor)
    if logits.ndim == 3:
        logits = logits.mean(dim=2)
    # The model already has a softmax layer, so output contains probabilities
    probs = logits.cpu().numpy().squeeze()
    pred = int(np.argmax(probs))
    conf = float(np.max(probs))
print(f"Offline prediction: class {pred}, confidence {conf:.3f}")
print(f"Offline probabilities: {probs}")

# REALTIME PREPROCESSING SIMULATION
print("\n=== REALTIME PREPROCESSING SIMULATION ===")

# Get raw data (before any preprocessing)
raw_dataset = MOABBDataset("BNCI2014_001", subject_ids=[subject_id])
raw = getattr(raw_dataset.datasets[0], '_raw', None)
if raw is None:
    raw = getattr(raw_dataset.datasets[0], 'raw', None)
if raw is None:
    raise AttributeError('Could not find raw or _raw attribute in dataset.datasets[0]')
raw.pick_channels(included_channels)
if raw.info['sfreq'] != sfreq:
    raw.resample(sfreq)

# Get the same window from raw data
raw_data = raw.get_data(picks=included_channels)
print(f"Raw data shape: {raw_data.shape}")

# Simulate real-time preprocessing on the same window
# 1. Get window from raw data (same time period as offline)
window_start = 0  # Start from beginning
window_end = window_start + input_window_samples
raw_window = raw_data[:, window_start:window_end]
print(f"Raw window shape: {raw_window.shape}")
print(f"Raw window stats - min: {raw_window.min():.6f}, max: {raw_window.max():.6f}, mean: {raw_window.mean():.6f}, std: {raw_window.std():.6f}")

# 2. Scale to microvolts
scaled_window = raw_window * 1e6
print(f"Scaled window stats - min: {scaled_window.min():.6f}, max: {scaled_window.max():.6f}, mean: {scaled_window.mean():.6f}, std: {scaled_window.std():.6f}")

# 3. Filter (like real-time)
filtered_window = mne.filter.filter_data(
    scaled_window,
    sfreq=125,
    l_freq=4,
    h_freq=38,
    method='iir',
    picks=None,
    verbose=False
)
print(f"Filtered window stats - min: {filtered_window.min():.6f}, max: {filtered_window.max():.6f}, mean: {filtered_window.mean():.6f}, std: {filtered_window.std():.6f}")

# 4. Initialize standardizer with first 1000 samples
standardizer = OnlineExponentialStandardizer(n_channels=len(included_channels), factor_new=1e-3, init_block_size=250)
for i in range(1000):
    if i < raw_data.shape[1]:
        sample = raw_data[:, i] * 1e6  # Scale sample
        standardizer.feed_sample(sample)

# 5. Standardize window
if standardizer.initialized:
    standardized_window = standardizer.standardize_window(filtered_window)
    print(f"Standardized window stats - min: {standardized_window.min():.6f}, max: {standardized_window.max():.6f}, mean: {standardized_window.mean():.6f}, std: {standardized_window.std():.6f}")
else:
    print("Standardizer not initialized yet")
    standardized_window = filtered_window

# Test real-time prediction
x_tensor_rt = torch.tensor(standardized_window, dtype=torch.float32, device=device).unsqueeze(0)
with torch.no_grad():
    logits_rt = model(x_tensor_rt)
    if logits_rt.ndim == 3:
        logits_rt = logits_rt.mean(dim=2)
    # The model already has a softmax layer, so output contains probabilities
    probs_rt = logits_rt.cpu().numpy().squeeze()
    pred_rt = int(np.argmax(probs_rt))
    conf_rt = float(np.max(probs_rt))
print(f"Real-time prediction: class {pred_rt}, confidence {conf_rt:.3f}")
print(f"Real-time probabilities: {probs_rt}")

# Compare
print(f"\n=== COMPARISON ===")
if isinstance(offline_window, np.ndarray) and isinstance(standardized_window, np.ndarray):
    print(f"Offline vs Real-time prediction: {pred} vs {pred_rt}")
    print(f"Offline vs Real-time confidence: {conf:.3f} vs {conf_rt:.3f}")
    print(f"Window difference (max abs): {np.max(np.abs(offline_window - standardized_window)):.6f}")
    print(f"Window correlation: {np.corrcoef(offline_window.flatten(), standardized_window.flatten())[0,1]:.6f}")
    # Side-by-side plot
    plt.figure(figsize=(14, 6))
    for ch in range(len(included_channels)):
        plt.subplot(4, 4, ch+1)
        plt.plot(offline_window[ch], label='Offline', alpha=0.7)
        plt.plot(standardized_window[ch], label='Real-time', alpha=0.7)
        plt.title(included_channels[ch])
        plt.xticks([])
        if ch == 0:
            plt.legend()
    plt.suptitle('Offline vs Real-time Preprocessed Window (First Window)')
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()
else:
    print("Error: offline_window or standardized_window is not a numpy array. Cannot compare or plot.")