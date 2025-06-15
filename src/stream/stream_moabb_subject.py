import time
from pylsl import StreamInfo, StreamOutlet
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import Preprocessor, preprocess
import numpy as np

# ----------- Konfiguration -----------
subject_id = 8
sfreq_target = 125
channels = [
    'C3', 'C4', 'Cz',
    'FC1', 'FC2', 'FCz',
    'CP1', 'CP2', 'CPz',
    'P1', 'P2', 'Pz',
    'C1', 'C2',
    'CP3', 'CP4'
]

# ----------- Laden & Preprocessing -----------
dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[subject_id])
raw = dataset.datasets[0].raw

preprocessors = [
    Preprocessor('pick_channels', ch_names=channels, ordered=True),
    Preprocessor(lambda x: x * 1e6),
    Preprocessor('resample', sfreq=sfreq_target)
]
preprocess(dataset, preprocessors)

data = raw.get_data().T
sfreq = raw.info['sfreq']
n_channels = data.shape[1]

print(f"Starte MOABB-LSL-Stream: Subject {subject_id}, {n_channels} Kanäle, {sfreq} Hz")

# ----------- LSL-Stream starten -----------
info = StreamInfo(name='EEG', type='EEG', channel_count=n_channels,
                  nominal_srate=sfreq, channel_format='float32', source_id='moabb_subject')
outlet = StreamOutlet(info)

# ----------- Daten "live" streamen -----------
for sample in data:
    outlet.push_sample(sample.tolist())
    time.sleep(1.0 / sfreq)
