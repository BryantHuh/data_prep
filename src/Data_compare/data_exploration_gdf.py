import os
import mne

# --- Konfiguration ---
SUBJECT = "1"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_PATH = os.path.join(PROJECT_ROOT, f"data/subject{SUBJECT}_gdf/A01T.gdf")

# --- GDF-Datei laden ---
raw = mne.io.read_raw_gdf(DATA_PATH, preload=False)

# --- Kanalnamen anzeigen ---
print("\nAlle Kanalnamen im GDF:")
print(raw.ch_names)

# --- Kanaltypen zählen ---
ch_types = raw.get_channel_types()
print("\nVerteilung der Kanaltypen:")
for ch_type in set(ch_types):
    print(f"{ch_type}: {ch_types.count(ch_type)}")

# --- Annotations anzeigen ---
print("\nAnnotationen:")
print(raw.annotations)

# Optional: Info anzeigen
print("\nInfo:")
print(raw.info)
