# src/gs_model_verification/one_input_test.py
import os
import torch
import numpy as np
from collections import Counter

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    Preprocessor,
    preprocess,
    create_windows_from_events,
    exponential_moving_standardize,
)
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet

# 0. Gerät auswählen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Verwende Gerät: {device}")

# 1. Pfade & Parameter
model_path = os.path.abspath(
    os.path.join(__file__, "..", "..", "..", "models", "moabb_downsampled_good_subjects_model_full.pth")
)
subject_id = 8
sfreq_target = 125  # wie beim Training
win_len = 500       # wie beim Training

# 2. Datensatz laden & preprocess (Subject 1 exemplarisch)
print("Lade Datensatz…")
dataset = MOABBDataset("BNCI2014_001", subject_ids=[subject_id])

# dieselben 16 Kanäle wie im Training
included_channels = [
    'C3', 'C4', 'Cz',
    'FC1', 'FC2', 'FCz',
    'CP1', 'CP2', 'CPz',
    'P1', 'P2', 'Pz',
    'C1', 'C2',
    'CP3', 'CP4'
]

preprocessors = [
    Preprocessor("pick_channels", ch_names=included_channels, ordered=True),
    Preprocessor(lambda x: x * 1e6),  # V -> µV
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

# 5. Modell laden
print("Lade gesamtes Modell…")
torch.serialization.add_safe_globals([ShallowFBCSPNet])
model = torch.load(model_path, map_location=device, weights_only=False)
print(f"Modell geladen: {model}")
model.to(device).eval()

n_preds_per_input = model.get_output_shape()[2]

# 3. Fenster erzeugen: Länge 500, non-overlap
print("Erzeuge Fenster…")
windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=int(-0.5 * sfreq_target),  # optional
    trial_stop_offset_samples=0,
    window_size_samples=win_len,
    window_stride_samples=n_preds_per_input,
    drop_last_window=False,
    preload=True,
)

# 4. Hole das erste Fenster (Trial 0, Fenster 0) und seine Metadaten
X_np, y_true, meta0 = windows_dataset[0]
print(f"Loaded one window: shape={X_np.shape}, label={y_true}, meta={meta0}")

# 5. Modell laden
print("Lade gesamtes Modell…")
torch.serialization.add_safe_globals([ShallowFBCSPNet])
model = torch.load(model_path, map_location=device, weights_only=False)
print(f"Modell geladen: {model}")
model.to(device).eval()

# 6. Sliding-Window-Inferenz über alle 1-Sample-Crops im ersten Trial
#    Wir suchen alle Indizes, bei denen trial_nr == 0
idxs_trial0 = [
    i for i in range(len(windows_dataset))
    if windows_dataset[i][2][0] == meta0[0]
]

preds = []
with torch.no_grad():
    for idx in idxs_trial0:
        X_w, y_w, meta = windows_dataset[idx]
        x = torch.tensor(X_w, dtype=torch.float32, device=device).unsqueeze(0)  # [1, n_chans, win_len]
        logits = model(x)  # Dense-Net liefert [1, n_classes, n_preds]

        # 1) Mittel über alle Prädiktionen (Zeitschnitte):
        if logits.ndim == 3:
            # Mitteln über die Zeit-Achse (dim=2), Ergebnis: [1, n_classes]
            logits = logits.mean(dim=2)

        # 2) Jetzt argmax über die Klassen-Achse (dim=1) und Index extrahieren
        pred = int(logits.argmax(dim=1).item())
        preds.append(pred)

# 3) Verteilung anzeigen
cnt = Counter(preds)
print(f"\nVerteilung der Vorhersagen über alle {len(preds)} Crops im ersten Trial:")
for cls, count in sorted(cnt.items()):
    print(f"  Klasse {cls}: {count:3d} ({count/len(preds)*100:5.1f}%)")

# 4) Mehrheitsvorhersage (Mode)
majority = cnt.most_common(1)[0][0]
print(f"Mehrheitsklasse: {majority}  (Ground-Truth war: {y_true})")