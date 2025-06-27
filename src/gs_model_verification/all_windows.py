import os
import torch
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    Preprocessor,
    preprocess,
    create_windows_from_events,
    exponential_moving_standardize,
)
from braindecode.models import ShallowFBCSPNet
from braindecode.visualization import plot_confusion_matrix
import matplotlib.pyplot as plt

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Paths and Parameters ---
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
except NameError:
    project_root = os.path.abspath('.')


# Model path und Subjekt ID setzen zum testen.

model_path = os.path.join(project_root, 'models', 'test8lo.pth')

SUBJECT_ID = 8
TARGET_SFREQ = 125
WINDOW_SIZE_SAMPLES = 500

# --- 1. Load and Preprocess Data ---
print(f"Loading data for subject {SUBJECT_ID}...")
dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[SUBJECT_ID])

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

# --- 4. Group Windows by Trial and Predict ---
metadata = windows_dataset.get_metadata()
if metadata.empty:
    raise ValueError("No windows were created from the dataset. Check parameters.")

# A unique trial is identified by its session, run, and start time
trial_groups = metadata.groupby(['session', 'run', 'i_start_in_trial'])

all_true_labels = []
all_majority_predictions = []

print(f"\nProcessing {len(trial_groups)} trials for subject {SUBJECT_ID}...")

for i, (trial_identifier, trial_df) in enumerate(trial_groups):
    window_indices = trial_df.index
    true_label = trial_df['target'].iloc[0]

    predictions_for_trial = []
    with torch.no_grad():
        for window_idx in window_indices:
            X_window, _, _ = windows_dataset[window_idx]
            X_tensor = torch.tensor(X_window, dtype=torch.float32, device=device).unsqueeze(0)
            output = model(X_tensor)

            if output.ndim == 3:
                output = output.mean(dim=2)

            predicted_class = output.argmax(dim=1).item()
            predictions_for_trial.append(predicted_class)

    # Determine majority vote for the current trial
    majority_vote_class = Counter(predictions_for_trial).most_common(1)[0][0]

    # Store results
    all_true_labels.append(true_label)
    all_majority_predictions.append(majority_vote_class)

    result = "CORRECT" if majority_vote_class == true_label else "INCORRECT"
    print(f"Trial {i+1:3d}/{len(trial_groups)} | Truth: {true_label}, Prediction: {majority_vote_class} -> {result}")


# --- 5. Display Overall Results ---
print("\n" + "="*40)
print(f"📊 Overall Results for Subject {SUBJECT_ID}")
print("="*40)

# Overall Accuracy
overall_accuracy = accuracy_score(all_true_labels, all_majority_predictions)
print(f"\nTotal Trials: {len(all_true_labels)}")
print(f"Correct Predictions: {np.sum(np.array(all_true_labels) == np.array(all_majority_predictions))}")
print(f"Overall Accuracy: {overall_accuracy:.2%}")

# Classification Report
print("\nClassification Report:")
# The labels in BNCI2014_001 are {left_hand: 0, right_hand: 1, feet: 2, tongue: 3}
class_names = ['left_hand', 'right_hand', 'feet', 'tongue']
report = classification_report(all_true_labels, all_majority_predictions, target_names=class_names)
print(report)

# Confusion Matrix
print("\nConfusion Matrix:")
conf_mat = confusion_matrix(all_true_labels, all_majority_predictions)
fig = plot_confusion_matrix(conf_mat, class_names=class_names)
fig.suptitle(f'Confusion Matrix for Subject {SUBJECT_ID}')
plt.show()

print("\n✅ Processing complete.")