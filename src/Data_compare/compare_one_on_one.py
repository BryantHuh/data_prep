

import numpy as np
import mne
from braindecode.preprocessing import (
    exponential_moving_standardize,
    preprocess,
    Preprocessor,
)

# Path to GDF file and channel list
gdf_file = "../../../data/subject1_gdf/A01E.gdf"
included_channels = [
    'C3', 'C4', 'Cz',
    'FC1', 'FC2', 'FCz',
    'CP1', 'CP2', 'CPz',
    'P1', 'P2', 'Pz',
    'C1', 'C2',
    'CP3', 'CP4'
]

# Load raw GDF
raw = mne.io.read_raw_gdf(gdf_file, preload=True)
raw.pick_channels(included_channels)

# Extract left hand events (Cue type 769)
events, event_id = mne.events_from_annotations(raw)
left_events = events[events[:, 2] == 769]

# Extract first left hand trial (4s window)
sfreq = raw.info['sfreq']
start = left_events[0, 0]
stop = start + int(4 * sfreq)
segment_raw = raw.get_data(start=start, stop=stop)

# Comparison 1: Unprocessed
print("=== Vergleich ohne Preprocessing ===")
print("Shape Segment (Raw):", segment_raw.shape)

# Apply Braindecode Preprocessing
def apply_bd_preprocessing(rawdata):
    preprocess(rawdata, [Preprocessor("pick_types", eeg=True, eog=False)])
    preprocess(rawdata, [Preprocessor("filter", l_freq=4.0, h_freq=38.0)])
    preprocess(rawdata, [Preprocessor(exponential_moving_standardize, factor_new=0.001, init_block_size=1000)])
    return rawdata

raw_proc = apply_bd_preprocessing(raw.copy())
segment_proc = raw_proc.get_data(start=start, stop=stop)

print("=== Vergleich MIT Braindecode Preprocessing ===")
print("Shape Segment (Processed):", segment_proc.shape)
print("Max. Absolutdifferenz (Raw vs Processed):", np.max(np.abs(segment_raw - segment_proc)))