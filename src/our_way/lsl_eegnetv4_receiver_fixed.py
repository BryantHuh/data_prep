# -*- coding: utf-8 -*-
"""
FIXED LSL EEGNetv4 receiver: Uses trial-aligned windows for proper timing.
Receives separate EEG and marker LSL streams, preprocesses, and classifies using EEGNetv4.
Prints true label (from trial start marker), predicted label, and confidence for each trial.
Runs until interrupted.
"""

import os
import time
import numpy as np
import torch
from torch.nn.functional import softmax
from pylsl import StreamInlet, resolve_byprop
from braindecode.models import EEGNetv4
import pandas as pd

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
model_path = os.path.join(project_root, 'models', 'eegnetv4_subj3_model_250_full.pth')

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Using device: {device}")

# Load EEGNetv4 model
torch.serialization.add_safe_globals([EEGNetv4])
model = torch.load(model_path, map_location=device, weights_only=False)
model.to(device).eval()
print(f"Model device: {next(model.parameters()).device}")

# LSL stream setup
print("Looking for EEG and marker streams...")
eeg_streams = resolve_byprop('type', 'EEG', timeout=60)
marker_streams = resolve_byprop('type', 'Markers', timeout=60)
eeg_inlet = StreamInlet(eeg_streams[0])
marker_inlet = StreamInlet(marker_streams[0])

print("Connected to LSL streams. Waiting for data...")

# Trial-aligned buffer system
buffer = []
trial_starts = []  # List of (sample_idx, label) tuples for trial starts
current_trial_label = None
sample_idx = 0
processed_trials = set()  # Track which trials we've already processed

# Label mapping (update as needed)
# Actual mapping from MOABB BNCI2014_001: 0=feet, 1=left_hand, 2=right_hand, 3=tongue
label_dict = {'feet': 0, 'left_hand': 1, 'right_hand': 2, 'tongue': 3}
inv_label_dict = {v: k for k, v in label_dict.items()}

# Wait for first real marker before starting
waiting_for_first_marker = True
results = []

def get_trial_window(trial_start_idx, buffer, window_size=250):
    """
    Get window data aligned with trial start.

    Args:
        trial_start_idx: Sample index where trial started
        buffer: Full buffer of samples
        window_size: Number of samples in window

    Returns:
        window_data: Window data if available, None otherwise
    """
    window_end_idx = trial_start_idx + window_size

    # Check if we have enough data
    if window_end_idx > len(buffer):
        return None

    # Extract window from trial start
    window_data = np.array(buffer[trial_start_idx:window_end_idx]).T  # shape: (n_channels, window)
    return window_data

try:
    while True:
        # Pull EEG sample
        sample, ts = eeg_inlet.pull_sample(timeout=0.1)
        if sample is not None:
            # Convert sample to numpy array and apply scaling
            sample_array = np.array(sample)
            scaled_sample = sample_array * 1e6  # Scale to microvolts
            buffer.append(scaled_sample)  # Store scaled sample
            sample_idx += 1

        # Pull marker
        marker, mts = marker_inlet.pull_sample(timeout=0.0)
        if marker is not None and marker[0] and marker[0] != 'start':
            # New trial started
            trial_label = marker[0]
            trial_starts.append((sample_idx, trial_label))
            current_trial_label = trial_label

            if waiting_for_first_marker:
                waiting_for_first_marker = False
                print(f"First real marker '{trial_label}' received at sample {sample_idx}. Starting EEGNetv4 inference.")

        # Process completed trials
        if not waiting_for_first_marker and len(trial_starts) > 0:
            for trial_start_idx, trial_label in trial_starts:
                # Skip if we've already processed this trial
                if trial_start_idx in processed_trials:
                    continue

                # Check if we have enough data for this trial
                window_data = get_trial_window(trial_start_idx, buffer, input_window_samples)
                if window_data is None:
                    continue  # Not enough data yet

                # Mark this trial as processed
                processed_trials.add(trial_start_idx)

                # EEGNetv4 expects shape (batch, channels, time)
                x_tensor = torch.tensor(window_data, dtype=torch.float32, device=device).unsqueeze(0)

                with torch.no_grad():
                    logits = model(x_tensor)
                    probabilities = softmax(logits, dim=1)

                    # Get predicted class and confidence
                    confidence, predicted_class = torch.max(probabilities, 1)
                    predicted_class = predicted_class.cpu().numpy()[0]
                    confidence = confidence.cpu().numpy()[0]
                    probabilities = probabilities.cpu().numpy()[0]

                # Calculate window end for display
                window_end_idx = trial_start_idx + input_window_samples

                print(f"Trial {trial_start_idx:5d}-{window_end_idx:5d} | True: {trial_label:10s} | Pred: {inv_label_dict.get(predicted_class, predicted_class):10s} | Conf: {confidence:.3f}")

                # Store results for CSV
                row = {
                    'trial_start': trial_start_idx,
                    'trial_end': window_end_idx,
                    'true_label': trial_label,
                    'pred_label': inv_label_dict.get(predicted_class, predicted_class),
                    'confidence': confidence,
                    'conf_0': probabilities[0],
                    'conf_1': probabilities[1],
                    'conf_2': probabilities[2],
                    'conf_3': probabilities[3],
                }
                results.append(row)

        # Sleep a bit to avoid busy waiting
        time.sleep(0.001)

except KeyboardInterrupt:
    print("Stopped by user.")
    # Save results to CSV
    df = pd.DataFrame(results)
    out_path = os.path.join(current_dir, f'lsl_eegnetv4_results_fixed_{subject_id}.csv')
    df.to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")

    # Calculate accuracy if we have results
    if results:
        correct = 0
        total = 0
        for row in results:
            if row['true_label'] in label_dict:
                true_idx = label_dict[row['true_label']]
                pred_idx = label_dict[row['pred_label']]
                if true_idx == pred_idx:
                    correct += 1
                total += 1

        if total > 0:
            accuracy = correct / total
            print(f"Real-time accuracy (trial-aligned): {accuracy*100:.2f}% ({correct}/{total})")
            print(f"Total trials processed: {len(results)}")