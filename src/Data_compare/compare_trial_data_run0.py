import os
import numpy as np
import mne
from moabb.datasets import BNCI2014001

# -------------- 1) GDF-Datensatz laden und vorverarbeiten ----------------

# Pfad zum GDF-File
gdf_path = r"E:\schirri_test_braindecode\data\subject1_gdf\A01T.gdf"

# 1.1 Raw aus GDF
raw_gdf = mne.io.read_raw_gdf(
    gdf_path,
    preload=True,
    verbose='error'  # nur Fehler anzeigen
)

# 1.2 Kanalnamen umbenennen
RENAMING = {
    'EEG-Fz': 'Fz', 'EEG-0': 'FC3', 'EEG-1': 'FC1', 'EEG-2': 'FCz',
    'EEG-3': 'FC2', 'EEG-4': 'FC4', 'EEG-5': 'C5',   'EEG-C3': 'C3',
    'EEG-6': 'C1',  'EEG-Cz': 'Cz',  'EEG-7': 'C2',  'EEG-C4': 'C4',
    'EEG-8': 'C6',  'EEG-9': 'CP3','EEG-10':'CP1','EEG-11':'CPz',
    'EEG-12':'CP2','EEG-13':'CP4','EEG-14':'P1',  'EEG-Pz':'Pz',
    'EEG-15':'P2',  'EEG-16':'POz',
    # die EOG benennen wir gleich, aber picken sie später nicht:
    'EOG-left':  'EOG-left',
    'EOG-central':'EOG-central',
    'EOG-right': 'EOG-right'
}
raw_gdf.rename_channels(RENAMING)

# 1.3 Nur die 22 EEG-Kanäle + 3 EOG behalten (wird später für GDF-Run0 auf 22 reduziert)
KEEP_GDF = list(RENAMING.values())
raw_gdf.pick_channels(KEEP_GDF)

# 1.4 Events aus den Annotations holen (769–772 sind die vier Klassen)
# Wir mappen hier zuerst auf 1–4, mappen später aber wieder zurück
event_id_gdf = {'769': 1, '770': 2, '771': 3, '772': 4}
events_gdf, _ = mne.events_from_annotations(
    raw_gdf,
    event_id=event_id_gdf,
    regexp='^(769|770|771|772)$'
)

# 1.5 Aus Epochs ein Trials-Array bauen (0–4 s, kein Baseline, preload)
epochs_gdf_all = mne.Epochs(
    raw_gdf, events_gdf, event_id=list(event_id_gdf.values()),
    tmin=0.0, tmax=4.0, baseline=None, preload=True
)
data_gdf_all = epochs_gdf_all.get_data()           # Shape = (288, 25, 1001)
labels_gdf_all = epochs_gdf_all.events[:, 2]       # Shape = (288,)

print("GDF Gesamt:", data_gdf_all.shape,
      "Labels (1–4):", np.unique(labels_gdf_all))

# 1.6 Labels 1–4 --> zurück auf Originalcodes 769–772 mappen
rev_map = {v: int(k) for k, v in event_id_gdf.items()}  # {1:769, 2:770, 3:771, 4:772}
labels_gdf_all = np.array([rev_map[lbl] for lbl in labels_gdf_all])

# 1.7 Run 0 herausschneiden (die ersten 48 Trials gehören zu Run 0)
data_gdf = data_gdf_all[:48]
labels_gdf = labels_gdf_all[:48]
# Für fairen Vergleich nur die 22 EEG-Kanäle nutzen (EOG weglassen)
# im Original haben wir 25 Kanäle (22 EEG + 3 EOG)
data_gdf = data_gdf[:, :22, :]
print("GDF Run0:", data_gdf.shape,
      "Labels (769–772):", np.unique(labels_gdf))


# -------------- 2) MOABB-Daten laden und im gleichen Format ---------------

# 2.1 Dataset laden (Subject 1)
ds = BNCI2014001()
all_data = ds.get_data(subjects=[1])

# Gibt zwei Keys: '0train' und '1test'
runs = all_data[1]['0train']
print("Verfügbare Runs:", list(runs.keys()))

# 2.2 Den Run '0' als Raw verwenden
raw_moabb = runs['0']

# 2.3 nur EEG-Kanäle aus den moabb-Daten behalten, EOG weglassen
KEEP_EEG = RENAMING.copy()
for k in ['EOG-left','EOG-central','EOG-right']:
    KEEP_EEG.pop(k, None)
KEEP_EEG = list(KEEP_EEG.values())[:22]  # sicherheitshalber nur die ersten 22
raw_moabb = raw_moabb.pick_channels(KEEP_EEG)

# 2.4 Events aus den Annotations holen (MOABB benutzt 'feet', 'left_hand', etc.)
event_id_moabb = {
    'feet':       771,
    'left_hand':  769,
    'right_hand': 770,
    'tongue':     772
}
events_moabb, _ = mne.events_from_annotations(
    raw_moabb,
    event_id=event_id_moabb
)
print("MOABB Events (erste 5):", events_moabb[:5])

# 2.5 Epochs bauen exakt wie oben (0–4 s, kein Baseline, preload)
epochs_moabb = mne.Epochs(
    raw_moabb, events_moabb, event_id=list(event_id_moabb.values()),
    tmin=0.0, tmax=4.0, baseline=None, preload=True
)
data_moabb = epochs_moabb.get_data()        # Shape = (48, 22, 1001)
labels_moabb = epochs_moabb.events[:, 2]    # Shape = (48,)
print("MOABB Run0:", data_moabb.shape,
      "Labels (769–772):", np.unique(labels_moabb))


# -------------- 3) Direktvergleich GDF vs MOABB ---------------------------

# 3.1 Formen & Labels prüfen
assert data_gdf.shape     == data_moabb.shape, \
    f"Shapes unterscheiden sich: GDF {data_gdf.shape} vs MOABB {data_moabb.shape}"
assert np.all(labels_gdf == labels_moabb),        \
    f"Labels unterscheiden sich: GDF {np.unique(labels_gdf)} vs MOABB {np.unique(labels_moabb)}"

# 3.2 Differenz berechnen
diff = np.abs(data_gdf - data_moabb)
print("Max-Abweichung:", diff.max())
print("Mittlere Abweichung  über Channel/Time:", diff.mean(axis=(0,2)).shape)

# 3.3 Beispielausgabe eines einzelnen Samples
i_trial, i_chan, i_time = 0, 0, 100
print("Beispiel:",
      "GDF[0,0,100] =", data_gdf[i_trial,i_chan,i_time],
      "MOABB[0,0,100] =", data_moabb[i_trial,i_chan,i_time])
