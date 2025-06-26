# gs_model_on_offline_bciIV.py
import os
import numpy as np
import mne
import torch
from skorch.helper import predefined_split
from sklearn.metrics import confusion_matrix

from braindecode.models import ShallowFBCSPNet
from braindecode.util import set_random_seeds
from braindecode import EEGClassifier
from braindecode.training import CroppedLoss

from braindecode.preprocessing import exponential_moving_standardize
from braindecode.preprocessing import Preprocessor, preprocess, create_windows_from_events

# 1) ---- RAW einlesen und Kanäle umbenennen / picken ----

gdf_path = r"E:\schirri_test_braindecode\data\subject1_gdf\A01T.gdf"
raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose='ERROR')

# Original‐Annotation‐Labels: '769','770','771','772'
EVENT_ID = {'769': 769, '770': 770, '771': 771, '772': 772}

# Rename alle EEG-Kanäle aus der GDF auf 16, wie beim Training
RENAMING = {
    'EEG-Fz':'Fz',  'EEG-0':'FC3',  'EEG-1':'FC1', 'EEG-2':'FCz',
    'EEG-3':'FC2','EEG-4':'FC4','EEG-5':'C5','EEG-C3':'C3',
    'EEG-6':'C1','EEG-Cz':'Cz','EEG-7':'C2','EEG-C4':'C4',
    'EEG-8':'C6','EEG-9':'CP3','EEG-10':'CP1','EEG-11':'CPz',
    'EEG-12':'CP2','EEG-13':'CP4','EEG-14':'P1','EEG-Pz':'Pz',
    'EEG-15':'P2','EEG-16':'POz'
}
raw.rename_channels(RENAMING)

# nur genau diese 16 Kanäle
included_ch = [
    'C3','C4','Cz',
    'FC1','FC2','FCz',
    'CP1','CP2','CPz',
    'P1','P2','Pz',
    'C1','C2','CP3','CP4'
]
raw.pick_channels(included_ch)

# 2) ---- Events aus den Annotations ziehen ----
#    wir mappen direkt auf 769–772, keine 7/8/9/10‐Remapperei
events, _ = mne.events_from_annotations(
    raw,
    event_id=EVENT_ID,
    regexp='^(769|770|771|772)$'
)

# 3) ---- Full‐Epochs über 0.0–4.0 s bauen ----
epochs = mne.Epochs(
    raw, events,
    event_id=EVENT_ID,
    tmin=0.0, tmax=4.0,
    baseline=None,
    preload=True,
    verbose='ERROR'
)
X = epochs.get_data()                    # shape = (288, 16, 500)
y = epochs.events[:, 2].astype(int)     # 769,770,771,772

# 4) ---- Preprocessing exakt wie im Training ----
# (V → µV, resample auf 125 Hz, bandpass 4–38 Hz, ExponentialMovingStd)
dataset = [(X, y)]  # wrap in list, damit preprocess() funktioniert
preprocessors = [
    Preprocessor('pick_channels', ch_names=included_ch, ordered=True),
    Preprocessor(lambda arr: arr * 1e6),           # V→µV
    Preprocessor('resample', sfreq=125),
    Preprocessor('filter', l_freq=4, h_freq=38),
    Preprocessor(
        exponential_moving_standardize,
        apply_on_array=True,
        factor_new=1e-3,
        init_block_size=1000
    )
]
preprocess(dataset, preprocessors, n_jobs=1)
X_pre, y_pre = dataset[0]  # nach Preprocessing

# 5) ---- Fenster (Crops) generieren wie im Training ----
sfreq = 125
input_window_samples = 500                   # 4 s @125 Hz
trial_start_offset = int(-0.5 * sfreq)       # −0.5 s Offset
# n_preds_per_input = wie im Train‐Model
# also:
model = ShallowFBCSPNet(
    n_chans=len(included_ch),
    n_classes=4,
    input_window_samples=input_window_samples,
    final_conv_length='auto'
)
model.to_dense_prediction_model()
n_preds_per_input = model.get_output_shape()[2]

windows = create_windows_from_events(
    dataset=[(X_pre, y_pre)],
    trial_start_offset_samples=trial_start_offset,
    trial_stop_offset_samples=0,
    window_size_samples=input_window_samples,
    window_stride_samples=n_preds_per_input,
    drop_last_window=False,
    preload=True
)
# split nur künstlich, wir faken hier "session" – alles in einen Set
# weil wir nur offline evaluieren
windows.datasets[0].metadata['session'] = 0
spl = windows.split('session')
windows_train = spl['0train']
windows_test  = spl['0test']

# 6) ---- Model laden und auf Test‐Windows anwenden ----
# path zu deinem pth
model_path = r"E:\schirri_test_braindecode\models\moabb_…_model.pth"
# gleiche Architektur wie beim Training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)

clf = EEGClassifier(
    model,
    cropped=True,
    criterion=CroppedLoss,
    criterion__loss_function=torch.nn.functional.nll_loss,
    optimizer=torch.optim.Adam,  # wird im Predict nicht gebraucht
    train_split=predefined_split(windows_train),  # dummy
    iterator_train__shuffle=False,
    batch_size=64,
    device=device,
    classes=[769,770,771,772]
)
# Predict auf Test‐Windows
y_pred = clf.predict(windows_test)

# 7) ---- Auswertung ----
# wir müssen y_true in dieselbe Form bringen – windows_test.targets liefert
# die passenden 769/770/…
y_true = windows_test.get_metadata().target
# accuracy
acc = np.mean(y_pred == y_true) * 100
print(f"Offline‐Accuracy = {acc:.1f}%")
# Confusion‐Matrix
cm = confusion_matrix(y_true, y_pred, labels=[769,770,771,772])
print("Confusion‐Matrix:\n", cm)
