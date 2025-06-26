import os
import torch
import numpy as np
from torch.nn.functional import softmax
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

# 4. Modell laden
print("Lade trainiertes Modell…")
torch.serialization.add_safe_globals([ShallowFBCSPNet])
model = torch.load(model_path, map_location=device, weights_only=False)
model.to(device).eval()
n_preds_per_input = model.get_output_shape()[2]

# 3. Fenster erzeugen
print("Erzeuge Fenster aus Events…")
windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=int(-0.5 * sfreq_target),
    trial_stop_offset_samples=0,
    window_size_samples=win_len,
    window_stride_samples=n_preds_per_input,
    drop_last_window=False,
    preload=True,
)

# --- Debug: Sanity-Check ---
print("== SANITY CHECK ==")
X_test, y_test, meta_test = windows_dataset[0]
print(f"1. Fenster – Label: {y_test}, Meta: {meta_test}")
print("Modell-Ausgabeform:", model.get_output_shape())
print("Model type:", type(model))
print("Model device:", next(model.parameters()).device)

x_tensor = torch.tensor(X_test, dtype=torch.float32, device=device).unsqueeze(0)
logits = model(x_tensor)
if logits.ndim == 3:
    logits = logits.mean(dim=2)
pred = int(logits.argmax(dim=1).item())
print(f"Modellvorhersage für 1. Fenster: {pred} | Wahres Label: {y_test}")

unique_labels = np.unique([y for _, y, _ in windows_dataset])
print("Eindeutige Labels im Testset:", unique_labels)

try:
    label_dict = windows_dataset.datasets[0].window_kwargs[1]['mapping']
    print("Label Mapping:", label_dict)
except Exception as e:
    print("⚠️ Kein Mapping gefunden:", e)

# 5. Alle Trials automatisch testen mit Softmax-Mittelung
unique_trials = np.unique([meta[0] for _, _, meta in windows_dataset])
correct = 0
results = []

for trial_idx in unique_trials:
    logits_sum = torch.zeros(4, device=device)
    true_label = None
    n_crops = 0

    for X_w, y_w, meta in windows_dataset:
        if meta[0] != trial_idx:
            continue
        if true_label is None:
            true_label = y_w

        x = torch.tensor(X_w, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            if logits.ndim == 3:
                logits = logits.mean(dim=2)
            probs = softmax(logits, dim=1).squeeze()
            logits_sum += probs
            n_crops += 1

    if n_crops == 0:
        print(f"⚠️ Keine Fenster gefunden für Trial {trial_idx}")
        continue

    pred = torch.argmax(logits_sum).item()
    results.append((trial_idx, pred, true_label))
    if pred == true_label:
        correct += 1

# 6. Ergebnisse ausgeben
print(f"\n✅ Getestete Trials: {len(results)}")
print(f"🎯 Trial-Accuracy: {correct}/{len(results)} = {correct / len(results):.2%}\n")

for trial_idx, pred, true_label in results:
    print(f"Trial {trial_idx:2d} | Vorhersage: {pred} | Label: {true_label} | {'✔' if pred == true_label else '✘'}")
