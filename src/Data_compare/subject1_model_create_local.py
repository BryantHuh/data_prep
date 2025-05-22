import os
import mne
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from braindecode.datautil import create_windows_from_events
from braindecode.datasets import BaseDataset, BaseConcatDataset, WindowsDataset
from braindecode.preprocessing import (Preprocessor, preprocess)
from braindecode.preprocessing.windowers import create_windows_from_events
from braindecode.models import ShallowFBCSPNet
from braindecode.util import set_random_seeds
from braindecode import EEGClassifier
from braindecode.training import CroppedLoss
from braindecode.visualization import plot_confusion_matrix

from torch import nn
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split

# ------------------ Load raw GDF ------------------
file_path = "e:/schirri_test_braindecode/data/subject1_gdf/A01T.gdf"
raw = mne.io.read_raw_gdf(file_path, preload=True)
sfreq = raw.info['sfreq']

# ------------------ Preprocessing ------------------
factor = 1e6
low_cut_hz = 4.
high_cut_hz = 38.
factor_new = 1e-3
init_block_size = 1000

preprocessors = [
    Preprocessor('pick_types', eeg=True, meg=False, stim=False),
    Preprocessor(lambda data: data * factor),
    Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),
    Preprocessor(lambda data: (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)),
]

# Wrap in BaseDataset and BaseConcatDataset
base_dataset = BaseDataset(raw, description={"subject": 1})
concat_ds = BaseConcatDataset([base_dataset])
preprocess(concat_ds, preprocessors)

# ------------------ Create windows using cropped training ------------------
input_window_samples = 1000  # 4 seconds @ 250 Hz

# Define label mapping
mapping = {
    "769": 0,  # left hand
    "770": 1,  # right hand
    "771": 2,  # feet
    "772": 3,  # tongue
}

windows_dataset = create_windows_from_events(
    concat_ds,
    trial_start_offset_samples=0,
    trial_stop_offset_samples=0,
    window_size_samples=input_window_samples,
    window_stride_samples=1,  # stride 1 for dense prediction
    drop_last_window=False,
    mapping=mapping,
    preload=True
)

# ------------------ Split dataset ------------------
n_total = len(windows_dataset)
n_valid = int(0.2 * n_total)
train_set = windows_dataset[:n_total - n_valid]
valid_set = windows_dataset[n_total - n_valid:]

# ------------------ Build Model ------------------
set_random_seeds(seed=20200220, cuda=torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'

n_classes = 4
n_chans = windows_dataset[0][0].shape[0]
model = ShallowFBCSPNet(n_chans, n_classes, input_window_samples=input_window_samples, final_conv_length="auto")
model.to(device)
model.to_dense_prediction_model()

# ------------------ Classifier ------------------
clf = EEGClassifier(
    model,
    cropped=True,
    criterion=CroppedLoss,
    criterion__loss_function=nn.NLLLoss(),
    optimizer=torch.optim.AdamW,
    train_split=predefined_split(valid_set),
    optimizer__lr=0.000625,
    optimizer__weight_decay=0.0,
    batch_size=64,
    callbacks=[
        "accuracy",
        ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=249)),
    ],
    device=device,
    classes=list(mapping.values()),
)

clf.fit(train_set, y=None, epochs=250)

# ------------------ Save model ------------------
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/subject1_cropped_model.pth")

# ------------------ Plot Confusion Matrix ------------------
y_true = valid_set.get_metadata().target
y_pred = clf.predict(valid_set)
conf_mat = confusion_matrix(y_true, y_pred)
labels = ["left", "right", "feet", "tongue"]
fig = plot_confusion_matrix(conf_mat, class_names=labels)
fig.savefig("models/subject1_confusion_matrix_cropped.png")
plt.close(fig)
