# leave_one_subject_out.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from braindecode.datasets import MOABBDataset, BaseConcatDataset
from braindecode.preprocessing import (Preprocessor, preprocess, create_windows_from_events,
                                       exponential_moving_standardize)
from braindecode.models import ShallowFBCSPNet
from braindecode import EEGClassifier
from braindecode.training import CroppedLoss
from braindecode.visualization import plot_confusion_matrix
from braindecode.util import set_random_seeds
from sklearn.metrics import confusion_matrix
from skorch.helper import predefined_split
from skorch.callbacks import LRScheduler

# ------------------------------
# Parameter
# ------------------------------
all_subjects = [1, 3, 8, 9]
train_subjects = [1, 3, 9]
test_subjects = [8]

sfreq_target = 125
input_window_samples = 500
included_channels = [
    'C3', 'C4', 'Cz', 'FC1', 'FC2', 'FCz',
    'CP1', 'CP2', 'CPz', 'P1', 'P2', 'Pz',
    'C1', 'C2', 'CP3', 'CP4']

# ------------------------------
# Daten laden & Preprocessing
# ------------------------------
print("Lade Daten und wende Preprocessing an…")
datasets = [MOABBDataset("BNCI2014_001", subject_ids=[sid]) for sid in all_subjects]
dataset = BaseConcatDataset(datasets)

preprocessors = [
    Preprocessor("pick_channels", ch_names=included_channels, ordered=True),
    Preprocessor(lambda x: x * 1e6),
    Preprocessor("resample", sfreq=sfreq_target),
    Preprocessor("filter", l_freq=4, h_freq=38),
    Preprocessor(exponential_moving_standardize, apply_on_array=True,
                 factor_new=1e-3, init_block_size=1000)
]
preprocess(dataset, preprocessors, n_jobs=1)

# ------------------------------
# Fensterung
# ------------------------------
sfreq = dataset.datasets[0].raw.info["sfreq"]
trial_start_offset_samples = int(-0.5 * sfreq)

windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=trial_start_offset_samples,
    trial_stop_offset_samples=0,
    window_size_samples=input_window_samples,
    window_stride_samples=12,
    drop_last_window=False,
    preload=True,
)

# ------------------------------
# Splitten in Train/Test (nur wenn vorhanden)
# ------------------------------
splitted = windows_dataset.split("subject")
splitted_keys = list(splitted.keys())
print(f"Verfügbare Subjects im Split: {splitted_keys}")

train_subsets = [ds for subj, ds in splitted.items() if int(subj) in train_subjects]
test_subsets = [ds for subj, ds in splitted.items() if int(subj) in test_subjects]

assert len(train_subsets) > 0, f"Keine Trainingsdaten gefunden. Vorhandene Subjects: {splitted_keys}"
assert len(test_subsets) > 0, f"Keine Testdaten gefunden. Vorhandene Subjects: {splitted_keys}"

train_set = BaseConcatDataset(train_subsets)
test_set = BaseConcatDataset(test_subsets)

# ------------------------------
# Modelltraining
# ------------------------------
n_classes = 4
n_chans = train_set[0][0].shape[0]
model = ShallowFBCSPNet(n_chans, n_classes, input_window_samples, final_conv_length='auto')
model.to_dense_prediction_model()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    model.cuda()
    torch.backends.cudnn.benchmark = True

set_random_seeds(seed=42, cuda=torch.cuda.is_available())

clf = EEGClassifier(
    model,
    cropped=True,
    criterion=CroppedLoss,
    criterion__loss_function=torch.nn.functional.nll_loss,
    optimizer=torch.optim.AdamW,
    train_split=predefined_split(test_set),
    optimizer__lr=0.000625,
    batch_size=64,
    callbacks=["accuracy", ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=24))],
    device=device,
    classes=list(range(n_classes))
)

print("Starte Training…")
clf.fit(train_set, y=None, epochs=25)

# ------------------------------
# Evaluation
# ------------------------------
y_true = test_set.get_metadata().target
y_pred = clf.predict(test_set)

acc = 100 * np.mean(y_true == y_pred)
print(f"\n🎯 Leave-One-Subject-Out Accuracy: {acc:.2f}%")

conf_mat = confusion_matrix(y_true, y_pred)

# Sicherstellen, dass Mapping korrekt extrahiert wird
label_dict = test_set.datasets[0].description.get('mapping', None)
if label_dict is None:
    label_dict = test_set.datasets[0].window_kwargs[1]['mapping']  # fallback

labels = [k for k, v in sorted(label_dict.items(), key=lambda kv: kv[1])]

fig_cm = plot_confusion_matrix(conf_mat, class_names=labels)
plt.tight_layout()
plt.show()