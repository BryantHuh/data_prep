# EEG Preprocessing Pipeline für BCI IV 2a – robust & minimal

from mne.io import read_raw_gdf
from mne import Epochs as MneEpochs
from braindecode.preprocessing import Preprocessor, exponential_moving_standardize, preprocess
from braindecode.datasets import create_from_mne_epochs
from pathlib import Path
import numpy as np
import torch
import logging

logging.basicConfig(level=logging.INFO)

# === Schritt 1: Label-Mapping ===
event_id = {"left_hand": 0, "right_hand": 1, "foot": 2, "tongue": 3}  # Saubere Eventnamen für MNE

# === Schritt 2: GDF laden & Trials erstellen ===
def load_trials_from_gdf(gdf_path: str, tmin: float = -0.5, tmax: float = 4.0):
    logging.info(f"Lade Datei: {gdf_path}")
    raw = read_raw_gdf(gdf_path, preload=True)

    # Annotationen manuell parsen
    desc = raw.annotations.description
    print("Alle gefundenen Annotationen:", set(desc))
    onset = raw.annotations.onset
    sfreq = raw.info["sfreq"]
    samples = (onset * sfreq).astype(int)

    label_map = {"769": "left_hand", "770": "right_hand", "771": "foot", "772": "tongue"}
    events = []
    for s, d in zip(samples, desc):
        if d in label_map:
            events.append([int(s), 0, label_map[d]])
    if not events:
        raise RuntimeError("Keine gültigen Events (769–772) gefunden.")

    events = np.array(events, dtype=object)  # damit str in Spalte 2 erlaubt ist  # MNE erwartet final int32-Array mit shape (N, 3) korrektes Format für MNE

    epochs = MneEpochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        detrend=1
    )
    return create_from_mne_epochs(epochs)

# === Schritt 3: Preprocessing ===
def apply_preprocessing(dataset):
    preprocessors = [
        Preprocessor("pick_types", eeg=True, meg=False),
        Preprocessor(lambda x: x * 1e6),
        Preprocessor("filter", l_freq=4., h_freq=38.),
        Preprocessor(exponential_moving_standardize, factor_new=1e-3, init_block_size=1000)
    ]
    preprocess(dataset, preprocessors, n_jobs=-1)
    return dataset

# === Schritt 4: Speichern ===
def save_to_pt(dataset, out_dir="preprocessed"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for i, ds in enumerate(dataset.datasets):
        data = ds.windows.get_data()
        label = ds.y
        torch.save({"X": torch.tensor(data), "y": torch.tensor(label)}, Path(out_dir) / f"sample_{i}.pt")
        logging.info(f"Gespeichert: sample_{i}.pt")

# === Hauptfunktion ===
def run_pipeline(gdf_path: str):
    dataset = load_trials_from_gdf(gdf_path)
    dataset = apply_preprocessing(dataset)
    save_to_pt(dataset)

if __name__ == "__main__":
    run_pipeline("data/gdf/A01T.gdf")
