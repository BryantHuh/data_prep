import os
import pyxdf

# Pfad zur Datei
xdf_path = os.path.join(os.path.dirname(__file__), '../../data/stream/test3.xdf')
streams, header = pyxdf.load_xdf(xdf_path)

# EEG-Stream suchen
eeg_stream = next((s for s in streams if s['info']['type'][0] == 'EEG'), None)
if eeg_stream is None:
    raise RuntimeError("Kein EEG-Stream gefunden!")

# Marker-Stream suchen (typisch "Markers" oder "Stimulus")
marker_stream = next((s for s in streams if s['info']['type'][0] in ['Markers', 'Marker', 'Stimulus']), None)

# EEG-Infos extrahieren
n_channels = int(eeg_stream['info']['channel_count'][0])
# Samplingrate robust auslesen
if 'nominal_srate' in eeg_stream['info']:
    sfreq = float(eeg_stream['info']['nominal_srate'][0])
elif 'effective_srate' in eeg_stream['info']:
    sfreq = float(eeg_stream['info']['effective_srate'][0])
else:
    raise RuntimeError("Keine Samplingrate gefunden!")

duration = eeg_stream['time_stamps'][-1] - eeg_stream['time_stamps'][0]
n_samples = len(eeg_stream['time_stamps'])

print(f"📈 EEG-Daten:")
print(f"- Samplingrate: {sfreq} Hz")
print(f"- Anzahl Kanäle: {n_channels}")
print(f"- Samples: {n_samples}")
print(f"- Dauer: {duration:.2f} Sekunden")

# Marker ausgeben, falls vorhanden
if marker_stream is not None:
    print(f"\n🧷 Marker (Anzahl: {len(marker_stream['time_stamps'])}):")
    for timestamp, value in zip(marker_stream['time_stamps'], marker_stream['time_series']):
        print(f"- {timestamp:.3f}s: {value[0]}")
else:
    print("\n⚠️ Keine Marker im Stream gefunden.")
