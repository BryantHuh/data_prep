import pyxdf
import os

xdf_path = os.path.join(os.path.dirname(__file__), '../../data/stream/test3.xdf')
streams, header = pyxdf.load_xdf(xdf_path)
eeg_stream = [s for s in streams if s['info']['type'][0] == 'EEG'][0]

print(eeg_stream['info'].keys())
print(eeg_stream['info'].get('samplingrate'))


data = eeg_stream['time_series']
sfreq = eeg_stream['info'].get('nominal_srate', [None])[0]
if sfreq is None:
    print("Samplingrate nicht gefunden, setze auf 125 Hz")
    sfreq = 125.0
else:
    sfreq = float(sfreq)

print(f"Geladene EEG-Daten mit {data.shape[1]} Kanälen und {sfreq} Hz Samplingrate")

from pylsl import StreamInfo, StreamOutlet
import time

# Meta-Info für den Stream
n_channels = data.shape[1]
info = StreamInfo(name='EEG', type='EEG', channel_count=n_channels,
                  nominal_srate=sfreq, channel_format='float32', source_id='sim_eeg')

outlet = StreamOutlet(info)

# Simulierter Stream
print("Starte LSL-Stream...")
for sample in data:
    outlet.push_sample(sample.tolist())
    time.sleep(1.0 / sfreq)

