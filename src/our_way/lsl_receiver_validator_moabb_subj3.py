# -*- coding: utf-8 -*-
"""
LSL receiver/validator: Receives raw EEG and marker streams from LSL, buffers them, and compares to original MOABB subject 3 data.
Checks channel order, sample values, and marker timing/labels.
"""

import os
import time
import numpy as np
from pylsl import StreamInlet, resolve_byprop
from braindecode.datasets import MOABBDataset

# Parameters
subject_id = 3
sfreq = 125  # Must match sender
included_channels = [
    'C3', 'C4', 'Cz',
    'FC1', 'FC2', 'FCz',
    'CP1', 'CP2', 'CPz',
    'P1', 'P2', 'Pz',
    'C1', 'C2',
    'CP3', 'CP4'
]

# Load original data for comparison
dataset = MOABBDataset("BNCI2014_001", subject_ids=[subject_id])
raw = getattr(dataset.datasets[0], '_raw', None)
if raw is None:
    raw = getattr(dataset.datasets[0], 'raw', None)
if raw is None:
    raise AttributeError('Could not find raw or _raw attribute in dataset.datasets[0]')
raw.pick_channels(included_channels)
if sfreq is not None and raw.info['sfreq'] != sfreq:
    raw.resample(sfreq)
orig_data = raw.get_data(picks=included_channels)
orig_annotations = raw.annotations

# Resolve LSL streams
print("Looking for EEG and marker streams...")
eeg_streams = resolve_byprop('type', 'EEG', timeout=60)
marker_streams = resolve_byprop('type', 'Markers', timeout=60)

eeg_inlet = StreamInlet(eeg_streams[0])
marker_inlet = StreamInlet(marker_streams[0])

# Buffer for received data
received_eeg = []
received_markers = []
received_marker_times = []

print("Receiving EEG and markers. Press Ctrl+C to stop...")
try:
    while True:
        sample, ts = eeg_inlet.pull_sample(timeout=0.0)
        if sample is not None:
            received_eeg.append(sample)
        marker, mts = marker_inlet.pull_sample(timeout=0.0)
        if marker is not None:
            received_markers.append(marker[0])
            received_marker_times.append(len(received_eeg)-1)  # Approximate sample index
        if len(received_eeg) >= orig_data.shape[1]:
            print("Received all expected samples.")
            break
except KeyboardInterrupt:
    print("Stopped by user.")

received_eeg = np.array(received_eeg).T  # shape: (n_channels, n_samples)

# --- Validation ---
print("\nValidating EEG data...")
if received_eeg.shape != orig_data.shape:
    print(f"Shape mismatch: received {received_eeg.shape}, original {orig_data.shape}")
    # Print first and last 5 sample indices for both
    print("First 5 received samples:", received_eeg[:, :5])
    print("First 5 original samples:", orig_data[:, :5])
    print("Last 5 received samples:", received_eeg[:, -5:])
    print("Last 5 original samples:", orig_data[:, -5:])
else:
    diff = np.abs(received_eeg - orig_data)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    exact_matches = np.sum(received_eeg == orig_data)
    print(f"Max abs diff: {max_diff:.6f}, Mean abs diff: {mean_diff:.6f}")
    print(f"Number of exact matches: {exact_matches} / {received_eeg.size}")
    if max_diff < 1e-5:
        print("EEG data matches exactly.")
    else:
        print("Warning: EEG data does not match exactly.")

print("\nValidating markers...")
orig_marker_events = [(int(onset * raw.info['sfreq']), desc) for onset, desc in zip(orig_annotations.onset, orig_annotations.description)]
print("Original markers (index, label):", orig_marker_events)
print("Received markers (index, label):", list(zip(received_marker_times, received_markers)))

# Handle 'start' marker
if received_markers and received_markers[0] == 'start':
    print("'start' marker detected as first received marker. Skipping for validation.")
    received_markers_valid = received_markers[1:]
    received_marker_times_valid = received_marker_times[1:]
else:
    print("Warning: No 'start' marker detected as first marker!")
    received_markers_valid = received_markers
    received_marker_times_valid = received_marker_times

# Only validate the first 20 real event markers
num_to_validate = min(20, len(received_markers_valid), len(orig_marker_events))
print(f"Validating the first {num_to_validate} markers.")
received_markers_valid = received_markers_valid[:num_to_validate]
received_marker_times_valid = received_marker_times_valid[:num_to_validate]
orig_marker_events = orig_marker_events[:num_to_validate]

if len(received_markers_valid) != len(orig_marker_events):
    print(f"Marker count mismatch: received {len(received_markers_valid)}, original {len(orig_marker_events)}")
else:
    mismatches = 0
    for (recv_idx, recv_desc), (orig_idx, orig_desc) in zip(zip(received_marker_times_valid, received_markers_valid), orig_marker_events):
        if recv_desc != orig_desc or abs(recv_idx - orig_idx) > 1:
            print(f"Mismatch: received ({recv_idx}, {recv_desc}), original ({orig_idx}, {orig_desc})")
            mismatches += 1
    if mismatches == 0:
        print("All markers match.")
    else:
        print(f"{mismatches} marker mismatches found.")

# --- Alignment and comparison of EEG data based on first matching marker label ---
print("\nAligning and comparing EEG data based on first matching marker label...")
# Find the first real event marker label in both lists (skip empty and 'start')
def first_real_marker(markers, times):
    for idx, (label, t) in enumerate(zip(markers, times)):
        if label and label != 'start':
            return idx, label, t
    return None, None, None

recv_idx, recv_label, recv_time = first_real_marker(received_markers, received_marker_times)
orig_idx, orig_label, orig_time = first_real_marker([desc for _, desc in orig_marker_events], [idx for idx, _ in orig_marker_events])

if recv_label == orig_label and recv_label is not None and recv_time is not None and orig_time is not None:
    offset = recv_time - orig_time
    print(f"First matching marker label: '{recv_label}' at received sample {recv_time}, original sample {orig_time}")
    print(f"Sample offset (received - original): {offset}")
    # Align and compare a window of 1000 samples
    window = 1000
    if offset >= 0:
        received_aligned = received_eeg[:, :window]
        original_aligned = orig_data[:, offset:offset+window]
    else:
        received_aligned = received_eeg[:, -offset:-offset+window]
        original_aligned = orig_data[:, :window]
    diff = np.abs(received_aligned - original_aligned)
    print(f"Aligned max abs diff: {np.max(diff):.6g}, mean abs diff: {np.mean(diff):.6g}")
else:
    print("Could not find a matching marker label to align on.")

print("\nValidation complete.")