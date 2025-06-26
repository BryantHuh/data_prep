# compare_trial_data.py
import mne
from moabb.datasets import BNCI2014001
import numpy as np

# 1) Daten holen
ds = BNCI2014001()
data = ds.get_data(subjects=[1])  # nur Subject 1

# 2) Inspect & Vergleich
for subj, run_groups in data.items():
    print(f"=== Subject {subj} ===")
    for group_name, runs in run_groups.items():
        print(f"Run group: {group_name}, contains runs:", runs.keys())
        for run_idx, session in runs.items():
            print(f"  -> Einzel-Run {run_idx}, keys:", session.keys())
            # Das sind die Objekte:
            raw_moabb   = session['runs']    # MNE-Raw-Objekt
            events_moabb= session['events']  # Dict: label → ndarray (n_trials,n_ch,n_times)
            labels_moabb= session['y']       # Labels 0..3, shape (n_trials,)
            sfreq       = session['sfreq']   # Samplingrate

            print("     sfreq:", sfreq)
            print("     Labels shape:", labels_moabb.shape, "→", np.unique(labels_moabb))
            print("     events keys:", list(events_moabb.keys()))
            # Wähle mal den left_hand Trial-Block:
            arr = events_moabb.get('left_hand', None)
            print("      left_hand array shape:", None if arr is None else arr.shape)

            # Hier könntet ihr synchron in eurer GDF-Version laden und
            # mit arr vergleichen - das zeige ich weiter unten.

            break  # nur ersten Run
        break      # nur erste Run-Gruppe
    break         # nur erster Subject

# 3) Laden der Original-GDF mit MNE
gdf_path = r"E:\schirri_test_braindecode\data\subject1_gdf\A01T.gdf"
raw_gdf = mne.io.read_raw_gdf(gdf_path, preload=True)

# 4) MOABB-Sample für “left_hand” extrahieren
moabb_left = arr           # shape = (n_trials, n_ch, n_times)
# n_ch und n_times müssen mit raw_gdf übereinstimmen:
print("MOABB left  :", moabb_left.shape)
print("GDF raw info:", (len(raw_gdf.ch_names), int(raw_gdf.times.size)))

# 5) Beispiel-Vergleich für erste Trial und ersten Kanal
print("erste MOABB-Werte:", moabb_left[0,0,:10])
gdf_data = raw_gdf.get_data(picks=0)  # Kanal 0, alle Samples
print("erste GDF-Werte :", gdf_data[:10])

# Hier könnt ihr natürlich noch stufenweise debuggen,
# z.B. MOABB-crop start sample mit raw_gdf.times abgleichen,
# usw.

