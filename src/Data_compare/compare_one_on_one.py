import mne
import numpy as np
from moabb.datasets import BNCI2014_001

# 1) MOABB laden
ds = BNCI2014_001()
all_data = ds.get_data()

# 2) Subject / Session / Run auswählen
subj = list(all_data.keys())[0]
sess = '0train'
run = '0'
raw_moabb = all_data[subj][sess][run]
moabb_chs = raw_moabb.info['ch_names']
print(f"MOABB-Kanäle ({sess}, Run {run}):\n{moabb_chs}\n")

# 3) GDF laden
gdf_path = 'data/subject1_gdf/A01T.gdf'
raw_gdf = mne.io.read_raw_gdf(gdf_path, preload=True)

# 4) Dummy-Stim-Kanal (misc) hinzufügen, falls fehlt
if 'stim' not in raw_gdf.ch_names:
    sf = raw_gdf.info['sfreq']
    stim_data = np.zeros((1, raw_gdf.n_times))
    info = mne.create_info(['stim'], sf, ['misc'])
    raw_stim = mne.io.RawArray(stim_data, info)
    raw_gdf.add_channels([raw_stim], force_update_info=True)

# 5) Kanal-Mapping MOABB→GDF
mapping = {
    'Fz':   'EEG-Fz','FC3':'EEG-0','FC1':'EEG-1','FCz':'EEG-2','FC2':'EEG-3',
    'FC4':'EEG-4','C5':'EEG-5','C3':'EEG-C3','C1':'EEG-6','Cz':'EEG-Cz',
    'C2':'EEG-7','C4':'EEG-C4','C6':'EEG-8','CP3':'EEG-9','CP1':'EEG-10',
    'CPz':'EEG-11','CP2':'EEG-12','CP4':'EEG-13','P1':'EEG-14','Pz':'EEG-Pz',
    'P2':'EEG-15','POz':'EEG-16','EOG1':'EOG-left','EOG2':'EOG-central',
    'EOG3':'EOG-right','stim':'stim',
}

# 6) GDF in MOABB-Reihenfolge picken
gdf_order = [mapping[ch] for ch in moabb_chs]
raw_gdf_sel = raw_gdf.copy().pick_channels(gdf_order)

print("GDF→MOABB pick order:\n", gdf_order)
print("Shapes nach pick:")
print(" MOABB:", raw_moabb.get_data().shape)
print(" GDF  :", raw_gdf_sel.get_data().shape)
