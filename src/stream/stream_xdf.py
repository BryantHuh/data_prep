# -*- coding: utf-8 -*-
"""
Replay eines XDF-Datensatzes über LSL
- Liest EEG- und Marker-Streams aus einer XDF-Datei
- Erstellt zwei LSL-Outlet-Streams ("EEG" und "Markers")
- Push Sample für Sample gemäß Original-Timestamps

Verwendung:
    python stream_xdf_to_lsl.py --xdf /pfad/zu/test3.xdf
"""
import time
import argparse
from pylsl import StreamInfo, StreamOutlet
try:
    import pyxdf
except ImportError:
    raise ImportError("pyxdf ist nicht installiert. Bitte: pip install pyxdf")

# Argumentparser
def main():
    parser = argparse.ArgumentParser(description='XDF Replay via LSL')
    parser.add_argument('--xdf', type=str, required=True,
                        help='Pfad zur XDF-Datei mit EEG- und Marker-Streams')
    args = parser.parse_args()

    # XDF laden
    print(f"Lade XDF-Datei: {args.xdf}")
    streams, _ = pyxdf.load_xdf(args.xdf)

    # EEG-Stream finden
    eeg_stream = next((s for s in streams if s['info']['type'][0] == 'EEG'), None)
    if eeg_stream is None:
        raise RuntimeError("Kein EEG-Stream in der XDF-Datei gefunden.")
    eeg_data = eeg_stream['time_series']
    eeg_ts   = eeg_stream['time_stamps']
    n_channels = len(eeg_data[0])
    nominal_srate = eeg_stream['info']['nominal_srate'][0]

    # Marker-Stream finden (optional)
    marker_stream = next((s for s in streams if s['info']['type'][0] in ('Markers','Marker')), None)
    if marker_stream:
        markers = [(marker_ts, marker[0])
                   for marker_ts, marker in zip(marker_stream['time_stamps'], marker_stream['time_series'])]
    else:
        markers = []

    print(f"EEG-Kanäle: {n_channels}, Sampling-Rate: {nominal_srate} Hz")
    print(f"Marker gefunden: {len(markers)} Events")

    # LSL-Outlets erstellen
    eeg_info = StreamInfo('EEG', 'EEG', channel_count=n_channels,
                          nominal_srate=nominal_srate, channel_format='float32', source_id='xdf_eeg')
    # ggf. Kanallabels in info.desc
    eeg_outlet = StreamOutlet(eeg_info)

    marker_info = StreamInfo('Markers', 'Markers', channel_count=1,
                             nominal_srate=nominal_srate, channel_format='string', source_id='xdf_markers')
    marker_outlet = StreamOutlet(marker_info)

    # Replay Loop: iteriere über EEG-Samples und pushe Marker, wenn fällig
    marker_idx = 0
    start_time = time.time()
    t0 = eeg_ts[0]

    for sample, ts in zip(eeg_data, eeg_ts):
        # Warte, bis Zeitpunkt erreicht ist (real-time relative)
        elapsed = time.time() - start_time
        target = ts - t0
        to_wait = target - elapsed
        if to_wait > 0:
            time.sleep(to_wait)

        # Push EEG-Sample
        eeg_outlet.push_sample(list(sample))

        # Push Marker, falls Timestamp überschritten
        while marker_idx < len(markers) and markers[marker_idx][0] <= ts:
            marker_value = str(markers[marker_idx][1])
            marker_outlet.push_sample([marker_value])
            print(f"Marker gesendet: {marker_value} @ {markers[marker_idx][0]:.3f}")
            marker_idx += 1

    print("Replay beendet.")

if __name__ == '__main__':
    main()
