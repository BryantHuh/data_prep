from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import Preprocessor, preprocess, create_windows_from_events
from braindecode.datasets.base import BaseConcatDataset
import numpy as np

# 1. Datensatz laden
ds = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[1])

# 2. Preprocessing (optional)
preprocessors = [
    Preprocessor('pick_types', eeg=True),
    Preprocessor(lambda x: x * 1e6),  # V -> µV
    Preprocessor('filter', l_freq=4, h_freq=38),
]
preprocess(ds, preprocessors)

# 3. Fensterung
windows_ds = create_windows_from_events(
    ds,
    trial_start_offset_samples=0,
    trial_stop_offset_samples=0,
    window_size_samples=1000,
    window_stride_samples=1000,
    drop_last_window=True,
    preload=True
)

# 4. Klassenverteilung
labels = windows_ds.get_metadata().target
unique, counts = np.unique(labels, return_counts=True)

print(f"🏷️ Gefensterte Klassenverteilung ({len(labels)} Fenster):")
for label, count in zip(unique, counts):
    print(f"- Klasse {int(label)}: {count}")
