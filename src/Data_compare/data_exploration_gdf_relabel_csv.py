import os
import mne
import numpy as np
import pandas as pd

# ----------- Setup paths -----------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
gdf_path = os.path.join(project_root, 'data', 'subject1_gdf', 'A01T.gdf')

# ----------- Load GDF file -----------
raw = mne.io.read_raw_gdf(gdf_path, preload=True)

# ----------- Rename channels -----------
gdf_channel_names = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1',
                     'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
                     'P1', 'Pz', 'P2', 'POz', 'EOG-left', 'EOG-central', 'EOG-right']
raw.rename_channels({old: new for old, new in zip(raw.ch_names, gdf_channel_names)})

# ----------- Define full annotation mapping -----------
event_desc_map = {
    '276': 'eyes_open',
    '277': 'eyes_closed',
    '768': 'start_trial',
    '769': 'left_hand',
    '770': 'right_hand',
    '771': 'feet',
    '772': 'tongue',
    '783': 'cue_unknown',
    '1023': 'rejected_trial',
    '1072': 'eye_movement',
    '32766': 'new_run'
}

# ----------- Extract annotations -----------
onsets = raw.annotations.onset
durations = raw.annotations.duration
descriptions = raw.annotations.description

# ----------- Build labeled DataFrame -----------
sfreq = raw.info['sfreq']
samples = (onsets * sfreq).astype(int)

labels = [event_desc_map.get(desc, f"unknown_{desc}") for desc in descriptions]

df = pd.DataFrame({
    'sample': samples,
    'time_sec': onsets,
    'duration_sec': durations,
    'event_code': descriptions,
    'event_label': labels
})

print(df.head(10))
print(f"✅ Extracted {len(df)} total annotations.")

# ----------- Save to CSV -----------
log_dir = os.path.join(project_root, 'log')
os.makedirs(log_dir, exist_ok=True)
csv_path = os.path.join(log_dir, 'all_gdf_annotations.csv')
df.to_csv(csv_path, index=False)
print(f"✅ Saved to: {csv_path}")
