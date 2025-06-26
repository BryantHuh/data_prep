# src/gs_model_verification/check_train_and_test_same_data.py
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from sklearn.metrics import confusion_matrix

from braindecode.datasets import MOABBDataset, BaseConcatDataset
from braindecode.preprocessing import (Preprocessor, preprocess, create_windows_from_events, exponential_moving_standardize)
from braindecode.models import ShallowFBCSPNet
from braindecode import EEGClassifier
from braindecode.training import CroppedLoss
from braindecode.util import set_random_seeds
from braindecode.visualization import plot_confusion_matrix
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split

# ------------------ Parameter ------------------
subject_ids = [1, 3, 8, 9]  # Train & validate on same data
included_channels = [
    'C3', 'C4', 'Cz',
    'FC1', 'FC2', 'FCz',
    'CP1', 'CP2', 'CPz',
    'P1', 'P2', 'Pz',
    'C1', 'C2',
    'CP3', 'CP4'
]
sfreq = 125
win_len = 500  # samples
n_classes = 4
n_epochs = 20
batch_size = 64

# ------------------ Daten laden ------------------
datasets = [MOABBDataset("BNCI2014_001", subject_ids=[sid]) for sid in subject_ids]
dataset = BaseConcatDataset(datasets)

preprocessors = [
    Preprocessor('pick_channels', ch_names=included_channels, ordered=True),
    Preprocessor(lambda x: x * 1e6),
    Preprocessor('resample', sfreq=sfreq),
    Preprocessor('filter', l_freq=4, h_freq=38),
    Preprocessor(exponential_moving_standardize, apply_on_array=True, factor_new=1e-3, init_block_size=1000)
]
preprocess(dataset, preprocessors, n_jobs=1)

# ------------------ Modell ------------------
n_chans = dataset[0][0].shape[0]
model = ShallowFBCSPNet(n_chans, n_classes, input_window_samples=win_len, final_conv_length='auto')
model.to_dense_prediction_model()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

set_random_seeds(seed=42, cuda=device=='cuda')
n_preds_per_input = model.get_output_shape()[2]

# ------------------ Fensterung ------------------
windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=int(-0.5 * sfreq),
    window_size_samples=win_len,
    window_stride_samples=n_preds_per_input,
    drop_last_window=False,
    preload=True,
)

train_set = windows_dataset
valid_set = windows_dataset

# ------------------ Training ------------------
clf = EEGClassifier(
    model,
    cropped=True,
    criterion=CroppedLoss,
    criterion__loss_function=torch.nn.functional.nll_loss,
    optimizer=torch.optim.AdamW,
    train_split=predefined_split(valid_set),
    optimizer__lr=0.000625,
    batch_size=batch_size,
    callbacks=["accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs-1))],
    device=device,
    classes=list(range(n_classes))
)

print("\nStarte Training auf Trainingsdaten (gleiches Set für Validierung)…")
t0 = time()
clf.fit(train_set, y=None, epochs=n_epochs)
print(f"Training abgeschlossen in {time() - t0:.1f}s\n")

# ------------------ Vorhersage & Auswertung ------------------
y_true = valid_set.get_metadata().target
y_pred = clf.predict(valid_set)
conf_mat = confusion_matrix(y_true, y_pred)

labels = ['left_hand', 'right_hand', 'feet', 'tongue']
fig_cm = plot_confusion_matrix(conf_mat, class_names=labels)
fig_cm.suptitle("Confusion Matrix (Train==Test)")
plt.show()

acc = np.mean(y_true == y_pred)
print(f"\nGesamtgenauigkeit (Train==Test): {acc*100:.2f}%")
