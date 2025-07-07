# -*- coding: utf-8 -*-
"""
Simple script to check the actual label mapping from the trained model.
"""

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import Preprocessor, preprocess, create_windows_from_events

# Load dataset
dataset = MOABBDataset("BNCI2014_001", subject_ids=[3])

# Preprocess exactly like training
included_channels = [
    'C3', 'C4', 'Cz',
    'FC1', 'FC2', 'FCz',
    'CP1', 'CP2', 'CPz',
    'P1', 'P2', 'Pz',
    'C1', 'C2',
    'CP3', 'CP4'
]

preprocessors = [
    Preprocessor('pick_channels', ch_names=included_channels, ordered=True),
    Preprocessor(lambda data: data * 1e6),
    Preprocessor('resample', sfreq=125),
    Preprocessor('filter', l_freq=4, h_freq=38),
]

preprocess(dataset, preprocessors, n_jobs=1)

# Create windows exactly like training
input_window_samples = 250
n_preds_per_input = 1  # For simplicity
sfreq = 125
trial_start_offset_samples = int(-0.5 * sfreq)

windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=trial_start_offset_samples,
    trial_stop_offset_samples=0,
    window_size_samples=input_window_samples,
    window_stride_samples=n_preds_per_input,
    drop_last_window=False,
    preload=True
)

# Split like training
splitted = windows_dataset.split('session')
valid_set = splitted['1test']

# Get the actual mapping
if hasattr(valid_set.datasets[0], 'window_kwargs') and valid_set.datasets[0].window_kwargs:
    mapping = valid_set.datasets[0].window_kwargs[0][1]['mapping']
    print(f"Actual label mapping: {mapping}")

    # Sort by numeric value
    sorted_mapping = sorted(mapping.items(), key=lambda x: x[1])
    print("\nClasses in order (0-3):")
    for label, value in sorted_mapping:
        print(f"  {value}: {label}")
else:
    print("Could not find window_kwargs mapping")

    # Check unique targets
    y_true = valid_set.get_metadata().target
    unique_y = sorted(set(y_true))
    print(f"Unique target values: {unique_y}")

    # Check first few samples
    print("\nFirst 10 sample targets:")
    for i in range(min(10, len(valid_set))):
        _, y, _ = valid_set[i]
        print(f"  Sample {i}: target = {y}")