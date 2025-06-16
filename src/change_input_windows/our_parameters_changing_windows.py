# -*- coding: utf-8 -*-
"""
Train ShallowFBCSPNet on MOABB (BNCI2014_001) using only our 16 OpenBCI channels,
resampled to 125 Hz, using Cropped Decoding and only the "good" subjects (1, 3, 8 & 9),
für verschiedene Fenstergrößen.
"""

import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix

from braindecode.datasets import MOABBDataset
from braindecode.datasets.base import BaseConcatDataset
from braindecode.preprocessing import Preprocessor, preprocess, create_windows_from_events, exponential_moving_standardize
from braindecode.models import ShallowFBCSPNet
from braindecode.util import set_random_seeds
from braindecode import EEGClassifier
from braindecode.training import CroppedLoss
from braindecode.visualization import plot_confusion_matrix

from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split

# -------------------------------------------
# Load MOABB dataset (only subjects 1, 3, 8, 9)
# -------------------------------------------
subject_ids = [1, 3, 8, 9]
datasets = [MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[sid]) for sid in subject_ids]
dataset = BaseConcatDataset(datasets)

# -------------------------------------------
# Kanal-Auswahl & Preprocessing
# -------------------------------------------
included_channels = [
    'C3', 'C4', 'Cz',
    'FC1', 'FC2', 'FCz',
    'CP1', 'CP2', 'CPz',
    'P1', 'P2', 'Pz',
    'C1', 'C2',
    'CP3', 'CP4'
]
preprocessors = [
    Preprocessor('pick_channels', ch_names=included_channels, ordered=True),
    Preprocessor(lambda data: data * 1e6),  # V -> µV
    Preprocessor('resample', sfreq=125),
    Preprocessor('filter', l_freq=4, h_freq=38),
    Preprocessor(
        exponential_moving_standardize,
        apply_on_array=True,
        factor_new=1e-3,
        init_block_size=1000
    )
]
preprocess(dataset, preprocessors, n_jobs=-1)

# -------------------------------------------
# Output-Verzeichnisse anlegen
# -------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
log_dir = os.path.join(project_root, 'log')
model_dir = os.path.join(project_root, 'models')
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# -------------------------------------------
# Gemeinsame Parameter
# -------------------------------------------
sfreq = dataset.datasets[0].raw.info['sfreq']
trial_start_offset_samples = int(-0.5 * sfreq)
n_classes = 4
n_chans = dataset[0][0].shape[0]
lr = 0.0625 * 0.01
batch_size = 64
n_epochs = 250

# Verschiedene Fenstergrößen (in Samples)
input_window_samples_list = [500, 250, 125, 100, 75, 50, 25]

# -------------------------------------------
# Training in Schleife für jede Fenstergröße
# -------------------------------------------
for input_window_samples in input_window_samples_list:
    print(f"=== Training mit {input_window_samples} Samples ({input_window_samples/sfreq:.2f} s) ===")

    # Modell-Definition
    model = ShallowFBCSPNet(
        n_chans,
        n_classes,
        input_window_samples=input_window_samples,
        final_conv_length='auto',
    )
    model.to_dense_prediction_model()

    cuda = torch.cuda.is_available()
    device = 'cuda' if cuda else 'cpu'
    if cuda:
        model.cuda()
        torch.backends.cudnn.benchmark = True
    set_random_seeds(seed=20200220, cuda=cuda)

    # Fenster-Datensatz erstellen
    n_preds_per_input = model.get_output_shape()[2]
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=0,
        window_size_samples=input_window_samples,
        window_stride_samples=input_window_samples // 2,
        drop_last_window=False,
        preload=True
    )
    splitted = windows_dataset.split('session')
    train_set = splitted['0train']
    valid_set = splitted['1test']

    # EEGClassifier konfigurieren
    clf = EEGClassifier(
        model,
        cropped=True,
        criterion=CroppedLoss,
        criterion__loss_function=torch.nn.functional.nll_loss,
        optimizer=torch.optim.AdamW,
        train_split=predefined_split(valid_set),
        optimizer__lr=lr,
        optimizer__weight_decay=0,
        iterator_train__shuffle=True,
        batch_size=batch_size,
        callbacks=[
            "accuracy",
            ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1))
        ],
        device=device,
        classes=list(range(n_classes))
    )

    # Training
    clf.fit(train_set, y=None, epochs=n_epochs)

    # Ergebnisse visualisieren und speichern
    suffix = f"{input_window_samples}samples"
    # Loss & Misclassification Plot
    results_columns = ['train_loss', 'valid_loss', 'train_accuracy', 'valid_accuracy']
    df = pd.DataFrame(clf.history[:, results_columns], columns=results_columns, index=clf.history[:, 'epoch'])
    df = df.assign(train_misclass=100 - 100 * df.train_accuracy, valid_misclass=100 - 100 * df.valid_accuracy)
    fig, ax1 = plt.subplots(figsize=(8, 3))
    df[['train_loss', 'valid_loss']].plot(ax=ax1, style=['-', ':'], marker='o', legend=False)
    ax1.set_ylabel("Loss")
    ax2 = ax1.twinx()
    df[['train_misclass', 'valid_misclass']].plot(ax=ax2, style=['-', ':'], marker='o', legend=False)
    ax2.set_ylabel("Misclassification [%]")
    ax1.set_xlabel("Epoch")
    handles = [
        Line2D([0], [0], color='black', linestyle='-', label='Train'),
        Line2D([0], [0], color='black', linestyle=':', label='Valid')
    ]
    plt.legend(handles=handles)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"training_{suffix}.png"))
    plt.close(fig)

    # Confusion Matrix
    y_true = valid_set.get_metadata().target
    y_pred = clf.predict(valid_set)
    conf_mat = confusion_matrix(y_true, y_pred)
    label_dict = valid_set.datasets[0].window_kwargs[0][1]['mapping']
    labels = [k for k, v in sorted(label_dict.items(), key=lambda kv: kv[1])]
    fig_cm = plot_confusion_matrix(conf_mat, class_names=labels)
    fig_cm.savefig(os.path.join(log_dir, f"confmat_{suffix}.png"))
    plt.close(fig_cm)

    # Modell speichern
    torch.save(model.state_dict(), os.path.join(model_dir, f"model_{suffix}.pth"))
    torch.save(model, os.path.join(model_dir, f"model_full_{suffix}.pth"))

print("Alle Modelle trainiert und gespeichert.")
