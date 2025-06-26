import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from moabb.datasets import BNCI2014001

# --------------- 1) GDF laden und epochieren der 6 Runs ------------------

gdf_path = r"E:\schirri_test_braindecode\data\subject1_gdf\A01T.gdf"

# Raw aus GDF laden
raw_gdf = mne.io.read_raw_gdf(gdf_path, preload=True, verbose='error')

# Kanalnamen umbenennen
RENAMING = {
    'EEG-Fz': 'Fz', 'EEG-0': 'FC3', 'EEG-1': 'FC1', 'EEG-2': 'FCz',
    'EEG-3': 'FC2', 'EEG-4': 'FC4', 'EEG-5': 'C5', 'EEG-C3': 'C3',
    'EEG-6': 'C1', 'EEG-Cz': 'Cz', 'EEG-7': 'C2', 'EEG-C4': 'C4',
    'EEG-8': 'C6', 'EEG-9': 'CP3','EEG-10':'CP1','EEG-11':'CPz',
    'EEG-12':'CP2','EEG-13':'CP4','EEG-14':'P1','EEG-Pz':'Pz',
    'EEG-15':'P2','EEG-16':'POz',
}
raw_gdf.rename_channels(RENAMING)

# nur EEG-Kanäle behalten
EEG_CH = list(RENAMING.values())
raw_gdf.pick_channels(EEG_CH)

# Events aus Annotationen holen (769–772)
event_id_gdf = {'769': 769, '770': 770, '771': 771, '772': 772}
events_gdf, _ = mne.events_from_annotations(
    raw_gdf, event_id=event_id_gdf, regexp='^(769|770|771|772)$'
)

# komplettes Epochs (288 Trials à 4s)
epochs_gdf = mne.Epochs(
    raw_gdf, events_gdf, event_id=list(event_id_gdf.values()),
    tmin=0.0, tmax=4.0, baseline=None, preload=True
)
data_gdf_all = epochs_gdf.get_data()
labels_gdf_all = epochs_gdf.events[:, 2]
sfreq = raw_gdf.info['sfreq']

# --------------- 2) MOABB-Daten laden und epochieren pro Run -------------

ds = BNCI2014001()
moabb_data = ds.get_data(subjects=[1])[1]['0train']
event_id_moabb = {'feet': 771, 'left_hand': 769, 'right_hand': 770, 'tongue': 772}

data_moabb_runs = {}
labels_moabb_runs = {}
for run in sorted(moabb_data.keys(), key=int):
    raw_run = moabb_data[run].copy().pick_channels(EEG_CH)
    events_run, _ = mne.events_from_annotations(
        raw_run, event_id=event_id_moabb
    )
    epochs_run = mne.Epochs(
        raw_run, events_run, event_id=list(event_id_moabb.values()),
        tmin=0.0, tmax=4.0, baseline=None, preload=True
    )
    data_moabb_runs[int(run)] = epochs_run.get_data()
    labels_moabb_runs[int(run)] = epochs_run.events[:, 2]

# --------------- 3) Vergleich pro Run ---------------------------------------

for run in range(6):
    start = run * 48
    gdf = data_gdf_all[start:start+48]
    gdf_lbls = labels_gdf_all[start:start+48]
    moabb = data_moabb_runs[run]
    moabb_lbls = labels_moabb_runs[run]
    assert gdf.shape == moabb.shape
    assert np.all(gdf_lbls == moabb_lbls)
    diff = np.abs(gdf - moabb)
    print(f"Run{run}: max diff = {diff.max():.1e}, mean diff = {diff.mean():.1e}")

# --------------- 4) Visualisierung eines kurzen Ausschnitts ----------------

# Beispiel: Run 0, Trial 0, Kanal 0, Samples 100–200
run = 0; trial = 0; chan = 0
s0, s1 = 100, 200
t = np.arange(s0, s1) / sfreq  # Zeitachse in Sekunden

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
# Links: GDF
axes[0].plot(t, data_gdf_all[trial, chan, s0:s1])
axes[0].set_title('GDF Run0, Trial0, Ch0')
axes[0].set_xlabel('Zeit [s]')
axes[0].set_ylabel('Amplitude [V]')

# Rechts: MOABB
axes[1].plot(t, data_moabb_runs[0][trial, chan, s0:s1])
axes[1].set_title('MOABB Run0, Trial0, Ch0')
axes[1].set_xlabel('Zeit [s]')

plt.tight_layout()
plt.show()
