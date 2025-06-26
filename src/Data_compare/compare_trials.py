#!/usr/bin/env python3
# compare_trials.py
# Vergleicht Trial-Start-Samples zwischen MOABB-Raw und reinem GDF-Raw

import os
import mne
from moabb.datasets import BNCI2014001

# 1) Lade MOABB-Raw (Run 0 des 1. Sessions)
ds = BNCI2014001()
subject = list(ds.get_data().keys())[0]
raw_moabb = ds.get_data()[subject]['0train']['0']    # erstes Run
print(f"MOABB: {len(raw_moabb.annotations)} Annotations, Kanäle = {len(raw_moabb.ch_names)}")
print("  MOABB-Chans:", raw_moabb.ch_names)

# 2) Lade das Original-GDF
gdf_path = os.path.join("data", f"subject{subject}_gdf", "A01T.gdf")
raw_gdf = mne.io.read_raw_gdf(gdf_path, stim_channel='auto', preload=True, verbose=False)
print(f"GDF geladen, Kanäle = {len(raw_gdf.ch_names)}")
print("  GDF-MNE-Chans:", raw_gdf.ch_names)

# 3) Kanal-Reihenfolge von MOABB übernehmen, "stim" entfernen, wenn im GDF nicht existiert
moabb_chs = raw_moabb.ch_names.copy()
if 'stim' in moabb_chs and 'stim' not in raw_gdf.ch_names:
    moabb_chs.remove('stim')
    print(">> 'stim' aus MOABB-Liste entfernt")

# 4) 1:1-Mapping von raw_gdf.ch_names → moabb_chs
if len(raw_gdf.ch_names) != len(moabb_chs):
    raise RuntimeError(f"Channel-Anzahl mismatch: GDF({len(raw_gdf.ch_names)}) vs MOABB({len(moabb_chs)})")
mapping = {old: new for old,new in zip(raw_gdf.ch_names, moabb_chs)}
raw_gdf.rename_channels(mapping)
print("GDF umbenannte Chans:", raw_gdf.ch_names)

# 5) exakt diese Reihenfolge picken
raw_gdf = raw_gdf.copy().pick(moabb_chs)
print("Nach pick():", raw_gdf.ch_names)

# 6) Events / Trial-Starts extrahieren
# --- MOABB ---
events_moabb, _ = mne.events_from_annotations(raw_moabb)
starts_moabb = sorted(evt[0] for evt in events_moabb if evt[2] in (1,2,3,4))
print("MOABB Trial-Starts:", starts_moabb)

# --- GDF (Codes 769..772 entsprechen 1..4) ---
events_gdf, _ = mne.events_from_annotations(raw_gdf)
gdf_codes = {769:1, 770:2, 771:3, 772:4}
starts_gdf = sorted(evt[0] for evt in events_gdf if evt[2] in gdf_codes)
print("GDF   Trial-Starts:", starts_gdf)

# 7) Vergleich
if len(starts_moabb) != len(starts_gdf):
    print(f"Anzahl Trials mismatch: MOABB({len(starts_moabb)}) vs GDF({len(starts_gdf)})")
else:
    diffs = [(i, m, g) for i,(m,g) in enumerate(zip(starts_moabb,starts_gdf)) if m!=g]
    if diffs:
        for i,m,g in diffs:
            print(f"✗ Trial {i}: MOABB {m} != GDF {g}")
    else:
        print("✅ Alle Trial-Starts identisch!")
