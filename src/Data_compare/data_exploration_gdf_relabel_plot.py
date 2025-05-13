import os
import mne
import numpy as np
import matplotlib.pyplot as plt

# ----------- Setup paths -----------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
gdf_path = os.path.join(project_root, 'data', 'subject1_gdf', 'A01T.gdf')

print(f"Loading GDF file from: {gdf_path}")
raw = mne.io.read_raw_gdf(gdf_path, preload=True)

# ----------- Rename GDF channels -----------
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

event_label_colors = {
    'eyes_open': 'blue',
    'eyes_closed': 'green',
    'start_trial': 'gray',
    'left_hand': 'red',
    'right_hand': 'orange',
    'feet': 'purple',
    'tongue': 'yellow',
    'cue_unknown': 'pink',
    'rejected_trial': 'black',
    'eye_movement': 'brown',
    'new_run': 'teal'
}

# ----------- Extract annotations and map to labels -----------
onsets = raw.annotations.onset
descriptions = raw.annotations.description
sfreq = raw.info['sfreq']
samples = (onsets * sfreq).astype(int)
labels = [event_desc_map.get(desc, f"unknown_{desc}") for desc in descriptions]

print(f"\nUnique annotations in GDF: {sorted(set(descriptions))}")
print("\nMapped annotation preview:")
for i in range(min(10, len(samples))):
    print(f"  Sample {samples[i]:6d} | Code: {descriptions[i]} -> Label: {labels[i]}")

# ----------- Build unique event_id and event_color mappings -----------
unique_labels = sorted(set(labels))
event_id = {label: idx + 1 for idx, label in enumerate(unique_labels)}
event_color = {event_id[label]: event_label_colors.get(label, 'gray') for label in unique_labels}

# ----------- Build final MNE-compatible events array -----------
events = np.array([[sample, 0, event_id[label]] for sample, label in zip(samples, labels)])

print(f"\nConstructed event_id (label → ID):\n{event_id}")
print(f"\nConstructed event_color (ID → color):\n{event_color}")

# ----------- Plot setup -----------
picks = mne.pick_types(raw.info, eeg=True, eog=True, stim=False)
start_time_sec = 0
duration_sec = 30

fig = raw.plot(
    duration=duration_sec,
    start=start_time_sec,
    n_channels=len(picks),
    picks=picks,
    title='GDF: All Annotations (First 30s)',
    show=False,
    events=events,
    event_id=event_id,
    event_color=event_color
)

# ----------- Show interactive plot -----------
plt.show(block=True)
