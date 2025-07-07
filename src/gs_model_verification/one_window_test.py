# -*- coding: utf-8 -*-
"""
Performs inference on windows from the MOABB dataset (BNCI2014_001)
using a pre-trained ShallowFBCSPNet model.
"""

import os
import torch
import numpy as np
import pandas as pd
from collections import Counter

from braindecode.datasets import MOABBDataset
from braindecode.datasets.base import BaseConcatDataset
from braindecode.preprocessing import (
    Preprocessor,
    preprocess,
    create_windows_from_events,
    exponential_moving_standardize,
)
from braindecode.models import ShallowFBCSPNet

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Paths and Parameters ---
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
except NameError:
    project_root = os.path.abspath('.')

model_path = os.path.join(project_root, 'models', 'moabb_downsampled_good_subjects_model_full.pth')

SUBJECT_ID = 3
TARGET_SFREQ = 125
WINDOW_SIZE_SAMPLES = 500
N_CHANNELS = 16
N_CLASSES = 4

# --- 1. Load and Preprocess Data ---
print(f"Loading data for subject {SUBJECT_ID}...")
dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[SUBJECT_ID])
dataset = BaseConcatDataset([ds for ds in dataset.datasets if ds.description["session"] == "1test"])
included_channels = [
    'C3', 'C4', 'Cz', 'FC1', 'FC2', 'FCz',
    'CP1', 'CP2', 'CPz', 'P1', 'P2', 'Pz',
    'C1', 'C2', 'CP3', 'CP4'
]

preprocessors = [
    Preprocessor('pick_channels', ch_names=included_channels, ordered=True),
    Preprocessor(lambda data: data * 1e6),
    Preprocessor('resample', sfreq=TARGET_SFREQ),
    Preprocessor('filter', l_freq=4, h_freq=38),
    Preprocessor(
        exponential_moving_standardize,
        apply_on_array=True,
        factor_new=1e-3,
        init_block_size=1000
    )
]

preprocess(dataset, preprocessors, n_jobs=-1)

# --- 2. Load the Pre-trained Model ---
print(f"Loading model from: {model_path}")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Please check the path.")

torch.serialization.add_safe_globals([ShallowFBCSPNet])
model = torch.load(model_path, map_location=device, weights_only=False)
model.to(device)
model.eval()
print("Model loaded successfully.")

# --- 3. Create Windows from Data ---
n_preds_per_input = model.get_output_shape()[2]

print("Creating windows from events...")
windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=int(-0.5 * TARGET_SFREQ),
    trial_stop_offset_samples=0,
    window_size_samples=WINDOW_SIZE_SAMPLES,
    window_stride_samples=n_preds_per_input,
    drop_last_window=False,
    preload=True
)

# --- 4. Select a Trial and Predict Each Window ---
metadata = windows_dataset.get_metadata()
print("\nAvailable metadata columns:", metadata.columns.tolist())


if metadata.empty:
    raise ValueError("No windows were created from the dataset. Check parameters.")

# Get all windows belonging to the first trial.
# We use 'i_start_in_trial' as the unique identifier for a trial.
from collections import Counter

results = []  # Liste für (prediction, ground truth)

for trial_start_sample in metadata['i_start_in_trial'].unique():
    trial_df = metadata[metadata['i_start_in_trial'] == trial_start_sample]
    true_label = trial_df['target'].iloc[0]
    window_indices = trial_df.index

    predictions = []
    with torch.no_grad():
        for i in window_indices:
            X_window, _, _ = windows_dataset[i]
            X_tensor = torch.tensor(X_window, dtype=torch.float32, device=device).unsqueeze(0)
            output = model(X_tensor)

            if output.ndim == 3:
                output = output.mean(dim=2)

            pred_class = output.argmax(dim=1).item()
            predictions.append(pred_class)

    majority_vote = Counter(predictions).most_common(1)[0][0]
    results.append((majority_vote, true_label))


# Auswertung
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df = pd.DataFrame(results, columns=["predicted", "true"])
accuracy = accuracy_score(df["true"], df["predicted"])
print(f"\n✅ Gesamt-Accuracy: {accuracy:.2%}")

# Confusion Matrix anzeigen
cm = confusion_matrix(df["true"], df["predicted"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()
