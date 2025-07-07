# -*- coding: utf-8 -*-
"""
Simple test to compare offline vs real-time preprocessing using identical methods.
"""

import numpy as np
import torch
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import Preprocessor, preprocess, create_windows_from_events, exponential_moving_standardize
from braindecode.models import ShallowFBCSPNet
import mne

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

# Load raw data
dataset = MOABBDataset("BNCI2014_001", subject_ids=[subject_id])
raw = getattr(dataset.datasets[0], '_raw', None)
if raw is None:
    raw = getattr(dataset.datasets[0], 'raw', None)
raw.pick_channels(included_channels)
if raw.info['sfreq'] != sfreq:
    raw.resample(sfreq)

raw_data = raw.get_data(picks=included_channels)
print(f"Raw data shape: {raw_data.shape}")

# Test 1: Use EXACTLY the same preprocessing as offline
print("\n=== TEST 1: IDENTICAL PREPROCESSING ===")

# Apply offline preprocessing to entire dataset
preprocessors = [
    Preprocessor('pick_channels', ch_names=included_channels, ordered=True),
    Preprocessor(lambda data: data * 1e6),
    Preprocessor('resample', sfreq=sfreq),
    Preprocessor('filter', l_freq=4, h_freq=38),
    Preprocessor(
        exponential_moving_standardize,
        apply_on_array=True,
        factor_new=1e-3,
        init_block_size=1000
    )
]
preprocess(dataset, preprocessors, n_jobs=1)

# Get first window from preprocessed data
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

first = test_set[0]
if isinstance(first, tuple) and len(first) == 3:
    offline_window, y_true, meta = first
elif isinstance(first, tuple) and len(first) == 2:
    offline_window, y_true = first
    meta = None

if not isinstance(offline_window, np.ndarray):
    offline_window = np.array(offline_window)

print(f"Offline window shape: {offline_window.shape}")
print(f"Offline window stats - min: {offline_window.min():.6f}, max: {offline_window.max():.6f}, mean: {offline_window.mean():.6f}, std: {offline_window.std():.6f}")

# Test 2: Apply same preprocessing to raw data manually
print("\n=== TEST 2: MANUAL PREPROCESSING ===")

# Get the same window from raw data
window_start = 0
window_end = window_start + input_window_samples
raw_window = raw_data[:, window_start:window_end]

print(f"Raw window shape: {raw_window.shape}")
print(f"Raw window stats - min: {raw_window.min():.6f}, max: {raw_window.max():.6f}, mean: {raw_window.mean():.6f}, std: {raw_window.std():.6f}")

# Step 1: Scale
scaled_window = raw_window * 1e6
print(f"After scaling - min: {scaled_window.min():.6f}, max: {scaled_window.max():.6f}, mean: {scaled_window.mean():.6f}, std: {scaled_window.std():.6f}")

# Step 2: Filter using MNE (same as real-time)
filtered_window = mne.filter.filter_data(
    scaled_window,
    sfreq=125,
    l_freq=4,
    h_freq=38,
    method='iir',
    picks=None,
    verbose=False
)
print(f"After filtering - min: {filtered_window.min():.6f}, max: {filtered_window.max():.6f}, mean: {filtered_window.mean():.6f}, std: {filtered_window.std():.6f}")

# Step 3: Standardize using braindecode's function
# First, get the first 1000 samples for initialization
init_data = raw_data[:, :1000] * 1e6
init_data = mne.filter.filter_data(
    init_data,
    sfreq=125,
    l_freq=4,
    h_freq=38,
    method='iir',
    picks=None,
    verbose=False
)

# Apply exponential moving standardize
standardized_window = exponential_moving_standardize(
    filtered_window,
    factor_new=1e-3,
    init_block_size=1000,
    eps=1e-4
)
print(f"After standardization - min: {standardized_window.min():.6f}, max: {standardized_window.max():.6f}, mean: {standardized_window.mean():.6f}, std: {standardized_window.std():.6f}")

# Compare
print(f"\n=== COMPARISON ===")
print(f"Window difference (max abs): {np.max(np.abs(offline_window - standardized_window)):.6f}")
print(f"Window correlation: {np.corrcoef(offline_window.flatten(), standardized_window.flatten())[0,1]:.6f}")

# Test predictions
x_tensor_offline = torch.tensor(offline_window, dtype=torch.float32, device=device).unsqueeze(0)
x_tensor_manual = torch.tensor(standardized_window, dtype=torch.float32, device=device).unsqueeze(0)

with torch.no_grad():
    logits_offline = model(x_tensor_offline)
    if logits_offline.ndim == 3:
        logits_offline = logits_offline.mean(dim=2)
    # The model already has a softmax layer, so output contains probabilities
    probs_offline = logits_offline.cpu().numpy().squeeze()
    pred_offline = int(np.argmax(probs_offline))
    conf_offline = float(np.max(probs_offline))

    logits_manual = model(x_tensor_manual)
    if logits_manual.ndim == 3:
        logits_manual = logits_manual.mean(dim=2)
    # The model already has a softmax layer, so output contains probabilities
    probs_manual = logits_manual.cpu().numpy().squeeze()
    pred_manual = int(np.argmax(probs_manual))
    conf_manual = float(np.max(probs_manual))

print(f"Offline prediction: class {pred_offline}, confidence {conf_offline:.3f}")
print(f"Manual prediction: class {pred_manual}, confidence {conf_manual:.3f}")
print(f"Prediction match: {pred_offline == pred_manual}")
print(f"Confidence difference: {abs(conf_offline - conf_manual):.3f}")