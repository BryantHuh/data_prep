import os
import mne
import matplotlib.pyplot as plt

# ----------- Paths -----------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
fif_path = os.path.join(project_root, 'data', 'subject1_moabb', '0', '0-raw.fif')

# ----------- Load MOABB raw file -----------
raw = mne.io.read_raw_fif(fif_path, preload=True)

# ----------- Extract events and mapping -----------
# MOABB already maps events to human-readable annotations like 'left_hand', etc.
events, event_id = mne.events_from_annotations(raw)
print("Event ID mapping:", event_id)

# ----------- Plot all EEG and EOG channels -----------
# Pick only EEG + EOG (exclude stim)
picks = mne.pick_types(raw.info, eeg=True, eog=True, stim=False)

# Set plotting range (you can crop for faster display if needed)
start = 0  # in seconds
duration = 30  # seconds to show

# Plot raw data with event markers
fig = raw.plot(
    duration=duration,
    start=start,
    n_channels=len(picks),
    picks=picks,
    title='MOABB Run 0 with Cue Events',
    show=True,
    events=events,
    event_color={v: 'red' for v in event_id.values()}
)
import matplotlib.pyplot as plt
plt.show(block=True)
# Create a static image of a raw signal segment
raw.plot(duration=10, start=0, picks=picks, show=False)
plt.savefig('moabb_run0_with_cues.png', dpi=300)
plt.close()
