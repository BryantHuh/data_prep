import os
import torch
import numpy as np
import pandas as pd
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    Preprocessor, preprocess, exponential_moving_standardize, create_windows_from_events
)

# Parameter
subject_id = 8
sfreq_target = 125
input_window_samples = 500
n_preds_per_input = 12  # Aus Trainingskonfiguration

included_channels = [
    'C3', 'C4', 'Cz', 'FC1', 'FC2', 'FCz',
    'CP1', 'CP2', 'CPz', 'P1', 'P2', 'Pz',
    'C1', 'C2', 'CP3', 'CP4'
]

# Lade Datensatz
dataset = MOABBDataset("BNCI2014_001", subject_ids=[subject_id])

# Preprocessing
preprocessors = [
    Preprocessor("pick_channels", ch_names=included_channels, ordered=True),
    Preprocessor(lambda x: x * 1e6),
    Preprocessor("resample", sfreq=sfreq_target),
    Preprocessor("filter", l_freq=4, h_freq=38),
    Preprocessor(
        exponential_moving_standardize,
        apply_on_array=True,
        factor_new=1e-3,
        init_block_size=1000,
    ),
]
preprocess(dataset, preprocessors, n_jobs=1)

# Fenster erzeugen
sfreq = dataset.datasets[0].raw.info["sfreq"]
trial_start_offset_samples = int(-0.5 * sfreq)

windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=trial_start_offset_samples,
    trial_stop_offset_samples=0,
    window_size_samples=input_window_samples,
    window_stride_samples=input_window_samples,
    drop_last_window=True,
    preload=True,
)

# Metadaten auslesen
metadata = windows_dataset.get_metadata()
print("Spalten im Metadaten-DataFrame:\n", metadata.columns)
print("\nBeispielzeilen:")
metadata.info()
unique_trials = metadata['i_start_in_trial'].unique()
print(f"Anzahl eindeutiger Trials: {len(unique_trials)}")

# Targets pro Trial anzeigen (zum Debuggen)
trial_info = metadata.groupby('i_start_in_trial')['target'].agg(['first', 'nunique'])
print(trial_info.head(10))

meta = windows_dataset.get_metadata()
trial_groups = meta.groupby('i_start_in_trial')

n_total = 0
n_ok = 0
for trial_id, group in trial_groups:
    unique_targets = group['target'].unique()
    if len(unique_targets) == 1:
        n_ok += 1
    n_total += 1

print(f"Eindeutige Label pro Trial: {n_ok}/{n_total} = {n_ok/n_total:.2%}")
