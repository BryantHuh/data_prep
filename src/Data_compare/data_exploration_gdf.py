import os
import mne
import matplotlib.pyplot as plt

# ----------- Setup paths -----------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
gdf_path = os.path.join(project_root, 'data', 'subject1_gdf', 'A01T.gdf')

# ----------- Load GDF file -----------
raw = mne.io.read_raw_gdf(gdf_path, preload=True)

# ----------- Rename GDF channels (BCI IV 2a standard layout) -----------
gdf_channel_names = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1',
                     'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
                     'P1', 'Pz', 'P2', 'POz', 'EOG-left', 'EOG-central', 'EOG-right']
raw.rename_channels({old: new for old, new in zip(raw.ch_names, gdf_channel_names)})

# ----------- Extract all annotations as events -----------
events_all, event_id = mne.events_from_annotations(raw, event_id='auto')
print("Used Annotations descriptions:", list(event_id.keys()))

# ----------- Pick EEG + EOG channels -----------
picks = mne.pick_types(raw.info, eeg=True, eog=True, stim=False)

# ----------- Plot window configuration -----------
start_time_sec = 700  # ⏱ Plot starting at 700 seconds
duration_sec = 30     # Show 30 seconds of data

# ----------- Plot raw data with numeric cue markers -----------
fig = raw.plot(
    duration=duration_sec,
    start=start_time_sec,
    n_channels=len(picks),
    picks=picks,
    title='GDF segment with cue markers (700s)',
    show=False,
    events=events_all,
    event_color={v: 'red' for v in event_id.values()}
)

# ----------- Save figure -----------
save_path = os.path.join(project_root, 'log', 'gdf_snapshot_at_700s_cues_only.png')
fig.savefig(save_path, dpi=300)
print(f"✅ Saved figure to: {save_path}")

# ----------- Keep plot open -----------
plt.show(block=True)
