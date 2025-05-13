import os
import mne
import numpy as np
import matplotlib.pyplot as plt

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

# Load MOABB .fif file (e.g., first run of training session)
raw_moabb = mne.io.read_raw_fif(
    f'{project_root}/data/subject1_moabb/0/0-raw.fif', preload=True)

# Load local GDF file (full training session)
raw_local = mne.io.read_raw_gdf(
    f'{project_root}/data/subject1_gdf/A01T.gdf', preload=True)

# Rename GDF channels to match MOABB
gdf_channel_names = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1',
                     'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
                     'P1', 'Pz', 'P2', 'POz', 'EOG-left', 'EOG-central', 'EOG-right']
raw_local.rename_channels({old: new for old, new in zip(raw_local.ch_names, gdf_channel_names)})

# Optional: print info and events
print("MOABB info:")
print(raw_moabb.info)
print("\nLocal GDF info:")
print(raw_local.info)

events_moabb = mne.events_from_annotations(raw_moabb)[0]
events_local = mne.events_from_annotations(raw_local)[0]
print("MOABB events:", events_moabb[:5])
print("Local events:", events_local[:5])

# Crop GDF to MOABB duration
raw_local_cropped = raw_local.copy().crop(tmax=raw_moabb.times[-1])

# Compare a single channel (e.g., 'Cz')
ch_name = 'Cz'
idx_moabb = raw_moabb.ch_names.index(ch_name)
idx_local = raw_local_cropped.ch_names.index(ch_name)

# Extract and scale data (from volts to microvolts)
signal_moabb = raw_moabb.get_data(picks=idx_moabb)[0] * 1e6
signal_local = raw_local_cropped.get_data(picks=idx_local)[0] * 1e6

# Optional: remove mean to center signals
signal_moabb -= np.mean(signal_moabb)
signal_local -= np.mean(signal_local)

# Create common time vector
sfreq = raw_moabb.info['sfreq']
n_samples = max(len(signal_moabb), len(signal_local))
times = np.arange(n_samples) / sfreq

# Pad shorter signal with NaNs
signal_local_padded = np.full(n_samples, np.nan)
signal_moabb_padded = np.full(n_samples, np.nan)
signal_local_padded[:len(signal_local)] = signal_local
signal_moabb_padded[:len(signal_moabb)] = signal_moabb

# Only compare valid (non-NaN) overlapping region
valid_mask = ~np.isnan(signal_local_padded) & ~np.isnan(signal_moabb_padded)

aligned_gdf = signal_local_padded[valid_mask]
aligned_moabb = signal_moabb_padded[valid_mask]

# Compare using numpy
max_diff = np.max(np.abs(aligned_gdf - aligned_moabb))
mae = np.mean(np.abs(aligned_gdf - aligned_moabb))
rmse = np.sqrt(np.mean((aligned_gdf - aligned_moabb) ** 2))
corr = np.corrcoef(aligned_gdf, aligned_moabb)[0, 1]

print(f"\n📊 Comparison Metrics for channel {ch_name}:")
print(f"   Max absolute difference: {max_diff:.2f} µV")
print(f"   Mean absolute error (MAE): {mae:.2f} µV")
print(f"   Root Mean Square Error (RMSE): {rmse:.2f} µV")
print(f"   Correlation coefficient (r): {corr:.4f}")


# Plot GDF and MOABB vertically aligned
plt.figure(figsize=(15, 6), constrained_layout=True)

plt.subplot(2, 1, 1)
plt.plot(times, signal_local_padded, color='tab:blue')
plt.title(f'{ch_name} - Local GDF')
plt.ylabel('Amplitude (µV)')
plt.xlim([0, times[-1]])
plt.ylim([-100, 100])


plt.subplot(2, 1, 2)
plt.plot(times, signal_moabb_padded, color='tab:orange')
plt.title(f'{ch_name} - MOABB')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (µV)')
plt.xlim([0, times[-1]])
plt.ylim([-100, 100])


plt.suptitle(f'Channel {ch_name} comparison - GDF vs. MOABB', fontsize=14)
plt.show()

# Print comparison metrics
max_diff = np.max(np.abs(signal_moabb - signal_local))
corr = np.corrcoef(signal_moabb, signal_local)[0, 1]
print(f"\nMax abs diff on channel {ch_name}: {max_diff:.4f} µV")
print(f"Correlation between MOABB and GDF on {ch_name}: {corr:.4f}")

plt.figure(figsize=(15, 3))
plt.plot(times[valid_mask], aligned_gdf - aligned_moabb, color='purple')
plt.title(f'Difference Signal (GDF - MOABB) for {ch_name}')
plt.ylabel('Δ Amplitude (µV)')
plt.xlabel('Time (s)')
plt.axhline(0, color='black', linestyle='--')
plt.show()
