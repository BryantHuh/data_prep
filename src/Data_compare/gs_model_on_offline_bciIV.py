import os
import numpy as np
import torch
import mne
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from braindecode.models import ShallowFBCSPNet
from braindecode.preprocessing import exponential_moving_standardize
from braindecode.visualization import plot_confusion_matrix
from braindecode.datasets import BaseConcatDataset
from braindecode.preprocessing.windowers import create_windows_from_events

# --- Konfiguration ---
SUBJECT = "1"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_PATH = os.path.join(PROJECT_ROOT, f"data/subject{SUBJECT}_gdf/A01T.gdf")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models/moabb_downsampled_good_subjects_model_full.pth")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "evaluation")
LABELS_TEXT = ['left_hand', 'right_hand', 'feet', 'tongue']
EVENT_ID = {
    '769': 0,
    '770': 1,
    '771': 2,
    '772': 3
}
label_to_int = EVENT_ID
LOWCUT, HIGHCUT = 4, 38
FS = 125
WINDOW_SIZE_SEC = 4.0

# --- Gerät wählen ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Modell laden ---
torch.serialization.add_safe_globals([ShallowFBCSPNet])
model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.eval()

# --- Sliding-Window-Parameter ---
input_window_samples = 500
n_preds_per_input = model.get_output_shape()[2]  # automatische Ableitung

# --- GDF-Datei laden ---
raw = mne.io.read_raw_gdf(DATA_PATH, preload=True)
print("Annotations descriptions:", np.unique(raw.annotations.description))

# --- Passende Kanäle anhand tatsächlicher GDF-Namen auswählen ---
INCLUDED_CHANNELS = [
    'EEG-C3', 'EEG-C4', 'EEG-Cz', 'EEG-Pz',
    'EEG-Fz',
    'EEG-0', 'EEG-1', 'EEG-2', 'EEG-3',
    'EEG-9', 'EEG-10', 'EEG-11', 'EEG-12', 'EEG-13', 'EEG-14', 'EEG-15'
]
raw.pick(INCLUDED_CHANNELS)

# --- Preprocessing (Braindecode-Standard) ---
raw.filter(l_freq=LOWCUT, h_freq=HIGHCUT)
data = raw.get_data()
standardized_data = exponential_moving_standardize(data, factor_new=1e-3, init_block_size=100)
raw._data = standardized_data

# --- Events aus Annotationen ---
events, _ = mne.events_from_annotations(raw, event_id=EVENT_ID)
# epochs = mne.Epochs(raw, events, event_id=EVENT_ID, tmin=-0.5, tmax=4.0, baseline=None, preload=True)
# epochs = mne.Epochs(raw, events, event_id=EVENT_ID, tmin=0.0, tmax=4.0, baseline=None, preload=True)

# --- Debug: Länge der Trials überprüfen ---
print("\n--- Trial-Debug ---")
for i, (onset, duration, desc) in enumerate(zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description)):
    if desc in EVENT_ID:
        start_sample = int(onset * raw.info['sfreq'])
        trial_samples = int(duration * raw.info['sfreq'])
        print(f"Trial {i}: Start={start_sample}, Dauer={trial_samples} Samples, Event={desc}")
print("--- Ende Trial-Debug ---\n")

from braindecode.datasets import BaseDataset
dataset = BaseConcatDataset([BaseDataset(raw=raw, description=None)])
trial_start_offset_samples = int(-0.5 * FS)
trial_stop_offset_samples = trial_start_offset_samples + input_window_samples
n_preds_per_input = model.get_output_shape()[2]
print("Mapping übergeben an create_windows_from_events:", label_to_int)
base_ds = create_windows_from_events(
    dataset,
    trial_start_offset_samples=trial_start_offset_samples,
    trial_stop_offset_samples=trial_stop_offset_samples,
    window_size_samples=input_window_samples,
    window_stride_samples=n_preds_per_input,
    drop_last_window=False,
    mapping=label_to_int,
    preload=True
)

labels = [sample[1] for sample in base_ds]
print("Verteilung der Labels im Datensatz:", pd.Series(labels).value_counts())

# --- Sliding Window Inferenz ---
y_true, y_pred = [], []
for i in range(len(base_ds)):
    x = base_ds[i][0][np.newaxis]  # shape: [1, C, T]
    label = base_ds[i][1]
    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
        pred = model(x_tensor)
        #print("Shape der Modellvorhersage:", pred.shape)
        probs = torch.softmax(pred, dim=1).mean(dim=2).cpu().numpy().ravel()
        #print("Probs:", probs)
        pred_class = int(np.argmax(probs))
    y_true.append(label)
    y_pred.append(pred_class)

# --- Evaluation speichern ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
report = classification_report(y_true, y_pred, target_names=LABELS_TEXT, output_dict=True)
pd.DataFrame(report).transpose().to_csv(os.path.join(OUTPUT_DIR, "classification_report_bciiv2a_a01t.csv"))

# --- Konfusionsmatrix plotten ---
conf_mat = confusion_matrix(y_true, y_pred)
fig = plot_confusion_matrix(conf_mat, class_names=LABELS_TEXT)
fig.suptitle("Confusion Matrix – BCI IV 2a A01T")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "confmat_bciiv2a_a01t.png"))
plt.show()
