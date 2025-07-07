# -*- coding: utf-8 -*-
"""
LSL sender: Streams MOABB subject 3 RAW EEG data and event markers over LSL.
Streams only raw EEG (with correct channels and optional resampling), no scaling/filtering/standardization.
All preprocessing should be done at the receiver.
"""

import os
import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock, cf_float32, cf_string
from braindecode.datasets import MOABBDataset

# Parameters
subject_id = 3
sfreq = 125  # Set to None to use original sampling rate, or set to 125 for optional resampling
included_channels = [
    'C3', 'C4', 'Cz',
    'FC1', 'FC2', 'FCz',
    'CP1', 'CP2', 'CPz',
    'P1', 'P2', 'Pz',
    'C1', 'C2',
    'CP3', 'CP4'
]

# Load MOABB data (no preprocessing except channel selection and optional resampling)
dataset = MOABBDataset("BNCI2014_001", subject_ids=[subject_id])
raw = getattr(dataset.datasets[0], '_raw', None)
if raw is None:
    raw = getattr(dataset.datasets[0], 'raw', None)
if raw is None:
    raise AttributeError('Could not find raw or _raw attribute in dataset.datasets[0]')
raw.pick_channels(included_channels)
if sfreq is not None and raw.info['sfreq'] != sfreq:
    raw.resample(sfreq)
annotations = raw.annotations

# LSL stream info for EEG
eeg_info = StreamInfo(
    name='MOABB_EEG_RAW',
    type='EEG',
    channel_count=len(included_channels),
    nominal_srate=raw.info['sfreq'],
    channel_format=cf_float32,
    source_id='moabb_subj3_eeg_raw'
)
# Add channel labels
chns = eeg_info.desc().append_child("channels")
for ch in included_channels:
    chns.append_child("channel").append_child_value("label", ch)

eeg_outlet = StreamOutlet(eeg_info, chunk_size=1, max_buffered=360)

# LSL stream info for Markers
marker_info = StreamInfo(
    name='MOABB_Markers',
    type='Markers',
    channel_count=1,
    nominal_srate=0,
    channel_format=cf_string,
    source_id='moabb_subj3_markers'
)
marker_outlet = StreamOutlet(marker_info)

# Prepare marker events (sample index, description)
marker_events = [(0, 'start')]
for onset, desc in zip(annotations.onset, annotations.description):
    sample_idx = int(onset * raw.info['sfreq'])
    marker_events.append((sample_idx, desc))
marker_events.sort()

# Stream the full session
stop_sample = raw.get_data(picks=included_channels).shape[1]
print(f"Will stream the full session up to sample {stop_sample}")

print(f"Streaming RAW EEG and markers for subject {subject_id} at {raw.info['sfreq']} Hz...")

# Stream EEG and markers
# Send completely raw data - all preprocessing will be done at receiver
data = raw.get_data(picks=included_channels)
total_samples = data.shape[1]
marker_idx = 0

for i in range(total_samples):
    if i >= stop_sample:
        print(f"Reached end of session at sample {i}, stopping stream.")
        break
    sample = data[:, i].astype(np.float32)
    eeg_outlet.push_sample(sample.tolist())
    # Check for marker at this sample
    while marker_idx < len(marker_events) and marker_events[marker_idx][0] == i:
        desc = marker_events[marker_idx][1]
        marker_outlet.push_sample([desc])
        print(f"Sent marker: {desc} at sample {i}")
        marker_idx += 1
    # Simulate real-time
    time.sleep(1.0 / raw.info['sfreq'])
    if i % (int(raw.info['sfreq']) * 5) == 0:
        print(f"Streamed {i/raw.info['sfreq']:.1f} seconds...")

print("Streaming complete.")