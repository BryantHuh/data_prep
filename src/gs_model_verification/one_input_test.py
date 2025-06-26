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

# 4. Hole Metadaten-DataFrame und analysiere Fenster-Trial-Zuordnung
metadata_df = windows_dataset.get_metadata()
print("\n📄 Spalten im Metadaten-DataFrame:", metadata_df.columns.tolist())

# Wähle einen Beispiel-Trial (z. B. erstes vorkommendes)
trial_start = metadata_df.iloc[0]["i_start_in_trial"]
trial_windows = metadata_df[metadata_df["i_start_in_trial"] == trial_start]
print(f"Gefundene Fenster für Trial-Start {trial_start}: {len(trial_windows)}")
print("Trial-Metadaten-Vorschau:\n", trial_windows.head())

# Richtiges Label (wird für alle Fenster identisch sein)
true_label = trial_windows["target"].iloc[0]

# 5. Inferenz über alle Fenster des Trials
preds = []
with torch.no_grad():
    for i in trial_windows.index:
        X_w, y_w, meta = windows_dataset[i]
        x = torch.tensor(X_w, dtype=torch.float32, device=device).unsqueeze(0)
        logits = model(x)
        if logits.ndim == 3:
            logits = logits.mean(dim=2)
        pred = int(logits.argmax(dim=1).item())
        preds.append(pred)

# 6. Ausgabe
cnt = Counter(preds)
print(f"\n📊 Verteilung der Vorhersagen über alle {len(preds)} Crops im Trial:")
for cls, count in sorted(cnt.items()):
    print(f"  Klasse {cls}: {count:3d} ({count/len(preds)*100:5.1f}%)")

majority = cnt.most_common(1)[0][0]
print(f"🏁 Mehrheitsklasse: {majority}  (Ground-Truth war: {true_label})")
