# -*- coding: utf-8 -*-
"""
Evaluierung des vortrainierten ShallowFBCSPNet auf dem Test-Set der "good" Subjects (1,3,8,9)
Reproduziert die im Trainingsskript gezeigte ~77% Accuracy und Confusion-Matrix.
"""
import os
import torch
import numpy as np
from braindecode.datasets import MOABBDataset
from braindecode.datasets.base import BaseConcatDataset
from braindecode.preprocessing import (
    Preprocessor,
    preprocess,
    create_windows_from_events,
    exponential_moving_standardize,
)
from braindecode.models import ShallowFBCSPNet
from sklearn.metrics import confusion_matrix, accuracy_score

# 0. Gerät auswählen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Verwende Gerät: {device}")

# 1. Pfade & Parameter
model_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "models",
        "moabb_downsampled_good_subjects_model_full.pth"
    )
)
sfreq_target = 125    # Hz, wie beim Training
win_len = 500         # Samples (4s)

# 2. Datensatz laden & preprocess für alle "good" Subjects
subject_ids = [1, 3, 8, 9]
datasets = [MOABBDataset("BNCI2014_001", subject_ids=[sid]) for sid in subject_ids]
dataset = BaseConcatDataset(datasets)

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
    Preprocessor(lambda data: data * 1e6),  # V -> µV
    Preprocessor('resample', sfreq=sfreq_target),
    Preprocessor('filter', l_freq=4, h_freq=38),
    Preprocessor(
        exponential_moving_standardize,
        apply_on_array=True,
        factor_new=1e-3,
        init_block_size=1000
    ),
]
preprocess(dataset, preprocessors, n_jobs=1)
print("Preprocessing abgeschlossen.")

# 3. Modell laden (full, dense prediction mode bereits gespeichert)
print("Lade Modell…")
torch.serialization.add_safe_globals([ShallowFBCSPNet])
model = torch.load(model_path, map_location=device, weights_only=False)
model.to(device).eval()
print(model)

# 4. Fenster erzeugen wie im Training (Cropped-Decoding Stride)
n_preds_per_input = model.get_output_shape()[2]
print(f"Cropped-Stride (n_preds_per_input): {n_preds_per_input}")
windows = create_windows_from_events(
    dataset,
    trial_start_offset_samples=int(-0.5 * sfreq_target),
    trial_stop_offset_samples=0,
    window_size_samples=win_len,
    window_stride_samples=n_preds_per_input,
    drop_last_window=False,
    preload=True,
)
print(f"Erzeugte Fenster insgesamt: {len(windows)}")

# 5. Split in Training/Validation nach Sessions
splitted = windows.split('session')
valid_set = splitted['1test']
print(f"Fenster im Test-Set: {len(valid_set)}")

# 6. Inferenz über alle Test-Fenster
y_true = valid_set.get_metadata().target
preds = []
with torch.no_grad():
    for i in range(len(valid_set)):
        X_w, y_w, meta = valid_set[i]
        x = torch.tensor(X_w, dtype=torch.float32, device=device).unsqueeze(0)
        logits = model(x)
        # Bei Dense-Prediction: [1, n_classes, n_preds]
        if logits.ndim == 3:
            logits = logits.mean(dim=2)
        pred = int(logits.argmax(dim=1).item())
        preds.append(pred)

y_pred = np.array(preds)

# 7. Metriken berechnen und ausgeben
acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print(f"Accuracy auf dem Test-Set: {acc * 100:.2f}%")
print("Confusion-Matrix (Zeilen=true, Spalten=pred):")
print(cm)
