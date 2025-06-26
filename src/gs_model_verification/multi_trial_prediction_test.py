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
from torch.nn.functional import softmax
import pandas as pd

# 0. Gerät auswählen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Verwende Gerät: {device}")

# 1. Parameter & Modellpfad
subject_id = 8
sfreq_target = 125
win_len = 500

model_path = os.path.abspath(
    os.path.join(__file__, "..", "..", "..", "models", "Test8.pth")
)

# 2. Datensatz laden
print("Lade MOABB-Datensatz…")
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
    Preprocessor(
        exponential_moving_standardize,
        apply_on_array=True,
        factor_new=1e-3,
        init_block_size=1000,
    ),
]
preprocess(dataset, preprocessors, n_jobs=1)

# 3. Fenster erzeugen
print("Erzeuge Fenster aus Events…")
model = torch.load(model_path, map_location=device, weights_only=False)
model.to(device).eval()
n_preds_per_input = model.get_output_shape()[2]

windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=int(-0.5 * sfreq_target),
    trial_stop_offset_samples=0,
    window_size_samples=win_len,
    window_stride_samples=n_preds_per_input,
    drop_last_window=False,
    preload=True,
)

# 4. Metadaten analysieren
metadata = windows_dataset.get_metadata()
grouped = metadata.groupby("i_start_in_trial")

print(f"\n📊 Anzahl erkannter Trials: {len(grouped)}")
correct = 0
results = []

for trial_start, group in grouped:
    true_label = group["target"].mode()[0]
    indices = group.index
    logits_sum = torch.zeros(4, device=device)

    for idx in indices:
        x, _, _ = windows_dataset[idx]
        x = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            if logits.ndim == 3:
                logits = logits.mean(dim=2)
            probs = softmax(logits, dim=1).squeeze()
            logits_sum += probs

    pred = torch.argmax(logits_sum).item()
    results.append((trial_start, pred, true_label))
    if pred == true_label:
        correct += 1

# 5. Ergebnisse ausgeben
print(f"\n🎯 Trial-Accuracy: {correct}/{len(results)} = {correct / len(results):.2%}\n")
for trial_start, pred, true_label in results:
    print(f"Trial Start: {trial_start:5d} | Vorhersage: {pred} | Label: {true_label} | {'✔' if pred == true_label else '✘'}")

print("\n📊 Zusammenfassung:")
print(f"  🧪 Trials insgesamt: {len(results)}")
print(f"  ✅ Richtig klassifiziert: {correct}")
print(f"  ❌ Falsch klassifiziert: {len(results) - correct}")
print(f"  🎯 Gesamtgenauigkeit: {correct / len(results) * 100:.2f}%")
