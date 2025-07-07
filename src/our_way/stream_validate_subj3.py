# -*- coding: utf-8 -*-
"""
Simulate streaming MOABB subject 3 data, classify each window, and print live predictions with confidence.
"""

import os
import torch
import numpy as np
from torch.nn.functional import softmax
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import Preprocessor, preprocess, create_windows_from_events, exponential_moving_standardize
from braindecode.models import ShallowFBCSPNet

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

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
model_path = os.path.join(project_root, 'models', 'subj3_model_250_full.pth')

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model
print(f"Loading model from {model_path}")
torch.serialization.add_safe_globals([ShallowFBCSPNet])
model = torch.load(model_path, map_location=device, weights_only=False)
model.to(device).eval()
n_preds_per_input = model.get_output_shape()[2]

# Load and preprocess data
dataset = MOABBDataset("BNCI2014_001", subject_ids=[subject_id])
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

# Windowing
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
splitted = windows_dataset.split('session')
if '1test' not in splitted:
    print(f"No test session found for subject {subject_id}.")
    exit(1)
test_set = splitted['1test']

# Get label mapping
# Use only the fallback, as window_kwargs is not reliably available
y_true_arr = test_set.get_metadata().target
unique_y = np.unique(y_true_arr)
label_dict = {str(i): i for i in unique_y}
inv_label_dict = {v: k for k, v in label_dict.items()}

# Streaming simulation and live prediction
y_true = test_set.get_metadata().target
y_pred = []
confidences = []

print("\nStreaming and classifying windows:")
for idx in range(len(test_set)):
    tup = test_set[idx]
    if isinstance(tup, tuple) and len(tup) == 3:
        x, y, meta = tup
    elif isinstance(tup, tuple) and len(tup) == 2:
        x, y = tup
        meta = None
    else:
        raise ValueError('Unexpected return value from test_set[idx]')
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(x_tensor)
        if logits.ndim == 3:
            logits = logits.mean(dim=2)
        # The model already has a softmax layer, so output contains probabilities
        probs = logits.cpu().numpy().squeeze()
        pred = int(np.argmax(probs))
        conf = float(np.max(probs))
    y_pred.append(pred)
    confidences.append(conf)
    print(f"Window {idx+1:4d} | True: {inv_label_dict.get(y, y)} | Pred: {inv_label_dict.get(pred, pred)} | Conf: {conf:.2f}")

# Accuracy
acc = 100 * np.mean(np.array(y_true) == np.array(y_pred))
print(f"\nFinal accuracy over all windows: {acc:.2f}%")