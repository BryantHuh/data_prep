# -*- coding: utf-8 -*-
"""
Moabb-Dataset über LSL streamen mit Marker-Kanal im EEG-Stream
- Liest Rohdaten und Annotationen aus BNCI2014_001 Subjekt
- Preprocessing: Kanal-Auswahl, µV-Skalierung, Resampling
- Erstellt einen LSL-Outlet "EEG" mit 17 Kanälen: 16 EEG + 1 Marker
- Marker-Kanal: 0=no event, 1=feet,2=left_hand,3=right_hand,4=tongue
- Streamt Sample für Sample in Echtzeit gemäß Sampling-Rate

Verwendung:
    python stream_moabb_with_marker.py --subject 8
"""
import os
import time
import argparse
import numpy as np
from pylsl import StreamInfo, StreamOutlet
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import Preprocessor, preprocess

# Marker-Mapping
MARKER_MAP = {
    'feet': 1,
    'left_hand': 2,
    'right_hand': 3,
    'tongue': 4
}

# Argumente
parser = argparse.ArgumentParser(description='Stream MOABB dataset via LSL with marker channel')
parser.add_argument('--subject', type=int, default=1, help='Subject ID from BNCI2014_001')
args = parser.parse_args()
subject_id = args.subject

# --- Laden & Preprocessing ---
sfreq_target = 125
channels = [
    'C3','C4','Cz',
    'FC1','FC2','FCz',
    'CP1','CP2','CPz',
    'P1','P2','Pz',
    'C1','C2',
    'CP3','CP4'
]
# Load
dataset = MOABBDataset('BNCI2014_001', subject_ids=[subject_id])
raw = dataset.datasets[0].raw.copy()
# Apply preprocessors
preprocessors = [
    Preprocessor('pick_channels', ch_names=channels, ordered=True),
    Preprocessor(lambda x: x * 1e6),  # V -> µV
    Preprocessor('resample', sfreq=sfreq_target)
]
preprocess(dataset, preprocessors, n_jobs=1)
# After preprocess dataset.datasets[0].raw is modified
raw = dataset.datasets[0].raw

data = raw.get_data()  # shape (n_channels, n_samples)
# transpose for easier push_sample: shape (n_samples, n_channels)
samples = data.T
annotations = raw.annotations  # onset, duration, description arrays
onsets = annotations.onset
descs  = annotations.description
# build marker array of length n_samples
times = raw.times  # in seconds
marker_chan = np.zeros(len(times), dtype=float)
# for each annotation, find nearest sample index and assign code
for onset, desc in zip(onsets, descs):
    if desc in MARKER_MAP:
        idx = np.argmin(np.abs(times - onset))
        marker_chan[idx] = float(MARKER_MAP[desc])

n_channels = len(channels)
print(f"Starte MOABB-LSL-Stream: Subj {subject_id}, EEG-Kanäle={n_channels}, Sampling-Rate={sfreq_target} Hz")

# --- LSL-Stream starten ---
total_ch = n_channels + 1
info = StreamInfo('EEG', 'EEG', channel_count=total_ch,
                  nominal_srate=sfreq_target, channel_format='float32', source_id=f'moabb_subj{subject_id}')
outlet = StreamOutlet(info)

# --- Echtzeit-Streaming ---
# compute inter-sample delay
delay = 1.0 / sfreq_target
for i, sample in enumerate(samples):
    # combine EEG + marker
    combined = np.concatenate([sample, [marker_chan[i]]])
    outlet.push_sample(combined.tolist())
    time.sleep(delay)
print("Stream beendet.")
