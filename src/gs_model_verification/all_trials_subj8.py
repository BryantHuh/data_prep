import os
import torch
import numpy as np
import pandas as pd
from collections import Counter
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    Preprocessor,
    preprocess,
    create_windows_from_events,
    exponential_moving_standardize,
)
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Verwende Gerät: {device}")

subject_id = 9
sfreq_target = 125
win_len = 500
model_path = os.path.abspath(
    os.path.join(__file__, "..", "..", "..", "models", "moabb_downsampled_good_subjects_model_full.pth")
)

# Dataset
print("Lade Datensatz…")
dataset = MOABBDataset("BNCI2014_001", subject_ids=[subject_id])
included_channels = [
    'C3', 'C4', 'Cz', 'FC1', 'FC2', 'FCz',
    'CP1', 'CP2', 'CPz', 'P1', 'P2', 'Pz',
    'C1', 'C2', 'CP3', 'CP4'
]
preprocessors = [
    Preprocessor("pick_channels", ch_names=included_channels, ordered=True),
    Preprocessor(lambda x: x * 1e6),
    Preprocessor("resample", sfreq=sfreq_target),
    Preprocessor("filter", l_freq=4, h_freq=38),
    Preprocessor(exponential_moving_standardize, apply_on_array=True,
                 factor_new=1e-3, init_block_size=1000),
]
preprocess(dataset, preprocessors, n_jobs=1)

# Modell
print("Lade Modell…")
torch.serialization.add_safe_globals([ShallowFBCSPNet])
model = torch.load(model_path, map_location=device, weights_only=False)
model.to(device).eval()
n_preds_per_input = model.get_output_shape()[2]

# Fenster
print("Erzeuge Fenster…")
windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=int(-0.5 * sfreq_target),
    trial_stop_offset_samples=0,
    window_size_samples=win_len,
    window_stride_samples=n_preds_per_input,
    drop_last_window=False,
    preload=True,
)

# Metadaten
meta_df = windows_dataset.get_metadata()
trial_starts = sorted(meta_df["i_start_in_trial"].unique())

# Alle Trials testen
print(f"\n📊 Teste alle {len(trial_starts)} Trials…")
correct = 0
results = []

for start in trial_starts:
    mask = meta_df["i_start_in_trial"] == start
    indices = meta_df[mask].index.tolist()

    preds = []
    for idx in indices:
        X, y, _ = windows_dataset[idx]
        x_tensor = torch.tensor(X, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = model(x_tensor)
            if logits.ndim == 3:
                logits = logits.mean(dim=2)
            pred = int(logits.argmax(dim=1).item())
            preds.append(pred)

    majority = Counter(preds).most_common(1)[0][0]
    true_label = int(meta_df.iloc[indices[0]]["target"])

    is_correct = (majority == true_label)
    if is_correct:
        correct += 1

    results.append((start, majority, true_label, is_correct))

# Ausgabe
print(f"\n🎯 Trial-Genauigkeit: {correct}/{len(results)} = {correct / len(results):.2%}\n")
for start, pred, label, correct_flag in results:
    symbol = "✔" if correct_flag else "✘"
    print(f"Trial Start: {start:6} | Vorhersage: {pred} | Label: {label} | {symbol}")
