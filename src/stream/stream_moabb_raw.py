# stream_moabb_raw_per_sample_marker.py
import time
import argparse
import numpy as np
from pylsl import StreamInfo, StreamOutlet
from braindecode.datasets import MOABBDataset

# Marker-Mapping
MARKER_MAP = {
    'feet': 1,
    'left_hand': 2,
    'right_hand': 3,
    'tongue': 4
}

# Argumente
parser = argparse.ArgumentParser(description='Stream raw MOABB EEG data via LSL with persistent markers')
parser.add_argument('--subject', type=int, default=1, help='Subject ID from BNCI2014_001')
args = parser.parse_args()
subject_id = args.subject

# --- Rohdaten laden ---
dataset = MOABBDataset("BNCI2014_001", subject_ids=[subject_id])
raw = dataset.datasets[0].raw

# Daten vorbereiten
data = raw.get_data().T  # shape (n_samples, n_channels)
annotations = raw.annotations
times = raw.times

# Marker initialisieren
marker_chan = np.zeros(len(times), dtype=float)
current_marker = 0
next_event_idx = 0
event_onsets = annotations.onset
event_descs = annotations.description

# Marker dauerhaft setzen, bis neues Event beginnt
for i, t in enumerate(times):
    # Event erreicht?
    while (next_event_idx < len(event_onsets)) and (t >= event_onsets[next_event_idx]):
        desc = event_descs[next_event_idx]
        if desc in MARKER_MAP:
            current_marker = MARKER_MAP[desc]
        next_event_idx += 1
    marker_chan[i] = current_marker

# --- LSL-Stream vorbereiten ---
sfreq = raw.info['sfreq']
n_channels = data.shape[1]
print(f"📡 Streaming Subject {subject_id} mit {n_channels} EEG-Kanälen + Marker-Kanal @ {sfreq} Hz")

info = StreamInfo('EEG', 'EEG', n_channels + 1, sfreq, 'float32', f'moabb_subj{subject_id}_raw_persistent')
outlet = StreamOutlet(info)

# --- Stream starten ---
delay = 1.0 / sfreq
for i in range(len(data)):
    sample = np.concatenate([data[i], [marker_chan[i]]])
    outlet.push_sample(sample.tolist())
    time.sleep(delay)

print("✅ Stream beendet.")
