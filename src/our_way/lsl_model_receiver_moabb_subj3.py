# -*- coding: utf-8 -*-
"""
LSL model receiver: Receives raw EEG and marker streams, preprocesses, windows, and classifies in real time.
Prints true label (from most recent marker), predicted label, and confidence for each window.
Runs until interrupted.
"""

import os
import time
import numpy as np
import torch
from torch.nn.functional import softmax
from pylsl import StreamInlet, resolve_byprop
from braindecode.models import ShallowFBCSPNet
from braindecode.preprocessing import exponential_moving_standardize
import pandas as pd
from online_standardizer import OnlineExponentialStandardizer
import mne
from scipy import signal
from scipy.signal import butter, filtfilt, resample_poly

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

# Model path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
model_path = os.path.join(project_root, 'models', 'subj3_model_250_full.pth')

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Using device: {device}")
torch.serialization.add_safe_globals([ShallowFBCSPNet])
model = torch.load(model_path, map_location=device, weights_only=False)
model.to(device).eval()
print(f"Model device: {next(model.parameters()).device}")
n_preds_per_input = model.get_output_shape()[2]

# LSL stream setup
print("Looking for EEG and marker streams...")
eeg_streams = resolve_byprop('type', 'EEG', timeout=60)
marker_streams = resolve_byprop('type', 'Markers', timeout=60)
eeg_inlet = StreamInlet(eeg_streams[0])
marker_inlet = StreamInlet(marker_streams[0])

print("Connected to LSL streams. Waiting for data...")

# Preprocessing state
standardizer = OnlineExponentialStandardizer(n_channels=len(included_channels), factor_new=1e-3, init_block_size=250)
buffer = []  # Simple buffer for raw samples
marker_buffer = []
marker_times = []
current_label = None
sample_idx = 0

# Preprocessing parameters
filter_low = 4        # Low frequency cutoff
filter_high = 38      # High frequency cutoff

# Preprocessing function to match offline exactly
def preprocess_window(window_data):
    """Apply preprocessing to a window of data (filtering only, scaling already done)"""
    # window_data is already scaled to microvolts from the sample processing

    # Bandpass filter (4-38 Hz) - matches offline Preprocessor('filter', l_freq=4, h_freq=38)
    window_data = mne.filter.filter_data(
        window_data,
        sfreq=125,  # Already at 125 Hz from sender
        l_freq=filter_low,
        h_freq=filter_high,
        method='iir',
        picks=None,
        verbose=False
    )

    return window_data

# Wait for first real marker before initializing standardizer
waiting_for_first_marker = True
first_marker_sample_idx = None
results = []

# Debug: Test both window alignments
use_offset = False  # Set to False to test original alignment

# Label mapping (update as needed)
# The model outputs 0-3, but we need to check what these actually correspond to
# Actual mapping from MOABB BNCI2014_001: 0=feet, 1=left_hand, 2=right_hand, 3=tongue
label_dict = {'feet': 0, 'left_hand': 1, 'right_hand': 2, 'tongue': 3}
inv_label_dict = {v: k for k, v in label_dict.items()}

try:
    while True:
        # Pull EEG sample
        sample, ts = eeg_inlet.pull_sample(timeout=0.1)
        if sample is not None:
            # Convert sample to numpy array and apply scaling
            sample_array = np.array(sample)
            scaled_sample = sample_array * 1e6  # Scale to microvolts
            buffer.append(scaled_sample)  # Store scaled sample, not raw sample
            sample_idx += 1
            # Feed sample to standardizer for initialization
            if not waiting_for_first_marker:
                standardizer.feed_sample(scaled_sample)
        # Pull marker
        marker, mts = marker_inlet.pull_sample(timeout=0.0)
        if marker is not None and marker[0] and marker[0] != 'start':
            marker_buffer.append(marker[0])
            marker_times.append(sample_idx)
            current_label = marker[0]
            if waiting_for_first_marker:
                waiting_for_first_marker = False
                first_marker_sample_idx = sample_idx
                print(f"First real marker '{current_label}' received at sample {sample_idx}. Starting standardizer initialization.")
        # Only start inference after standardizer is initialized
        if not waiting_for_first_marker and standardizer.initialized and len(buffer) >= input_window_samples:
            # Get window from buffer
            window_data = np.array(buffer[-input_window_samples:]).T  # shape: (n_channels, window)

            # Apply preprocessing
            window_data = preprocess_window(window_data)

            # Apply running standardization (matches offline exponential_moving_standardize)
            window_data = standardizer.standardize_window(window_data)

            # Model expects shape (batch, channels, time)
            x_tensor = torch.tensor(window_data, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits = model(x_tensor)
                if logits.ndim == 3:
                    logits = logits.mean(dim=2)
                # The model already has a softmax layer, so output contains probabilities
                probs = logits.cpu().numpy().squeeze()
                pred = int(np.argmax(probs))
                conf = float(np.max(probs))
            # Find the most recent marker for this window
            true_label = current_label if current_label is not None else 'unknown'
            print(f"Window ending at sample {sample_idx:5d} | True: {true_label:10s} | Pred: {inv_label_dict.get(pred, pred):10s} | Conf: {conf:.2f}")
            # Store results for CSV
            row = {
                'sample_idx': sample_idx,
                'true_label': true_label,
                'pred_label': inv_label_dict.get(pred, pred),
                'conf_0': probs[0],
                'conf_1': probs[1],
                'conf_2': probs[2],
                'conf_3': probs[3],
            }
            results.append(row)
        # Sleep a bit to avoid busy waiting
        time.sleep(0.001)
except KeyboardInterrupt:
    print("Stopped by user.")
    # Save results to CSV
    df = pd.DataFrame(results)
    out_path = os.path.join(current_dir, f'lsl_model_results_{subject_id}.csv')
    df.to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")