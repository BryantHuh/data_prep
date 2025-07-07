# -*- coding: utf-8 -*-
"""
Train EEGNetv4 on MOABB (BNCI2014_001) using only subject 3,
16 OpenBCI channels, resampled to 125 Hz, using minimal preprocessing.
EEGNetv4 is designed for real-time BCI with minimal preprocessing requirements.
"""

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import Preprocessor, preprocess, create_windows_from_events
from braindecode.util import set_random_seeds
from braindecode import EEGClassifier
from braindecode.visualization import plot_confusion_matrix
from braindecode.models import EEGNetv4

from skorch.callbacks import LRScheduler, EarlyStopping
from skorch.helper import predefined_split
from sklearn.metrics import confusion_matrix

import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

# -------------------------------------------
# Load only subject 3
# -------------------------------------------
subject_ids = [3]
dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=subject_ids)

included_channels = [
    'C3', 'C4', 'Cz',
    'FC1', 'FC2', 'FCz',
    'CP1', 'CP2', 'CPz',
    'P1', 'P2', 'Pz',
    'C1', 'C2',
    'CP3', 'CP4'
]

# Minimal preprocessing for real-time compatibility
preprocessors = [
    Preprocessor('pick_channels', ch_names=included_channels, ordered=True),
    Preprocessor(lambda data: data * 1e6),  # Scale to microvolts
    Preprocessor('resample', sfreq=125),
    # No filtering or standardization - EEGNet handles this internally
]

preprocess(dataset, preprocessors, n_jobs=-1)

# -------------------------------------------
# Model and window parameters
# -------------------------------------------
input_window_samples = 250  # 2 seconds * 125 Hz
n_classes = 4
n_chans = dataset[0][0].shape[0]

# EEGNetv4 parameters optimized for motor imagery
model = EEGNetv4(
    n_chans=n_chans,
    n_outputs=n_classes,
    n_times=input_window_samples,
    drop_prob=0.25,  # Dropout rate
    kernel_length=64,  # Temporal kernel size
)

cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'
if cuda:
    model.cuda()
    torch.backends.cudnn.benchmark = True

set_random_seeds(seed=20200220, cuda=cuda)

# For EEGNet, we want one window per trial, not cropped decoding
sfreq = dataset.datasets[0].raw.info['sfreq']
trial_start_offset_samples = int(-0.5 * sfreq)

windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=trial_start_offset_samples,
    trial_stop_offset_samples=0,
    window_size_samples=input_window_samples,
    window_stride_samples=input_window_samples,  # No overlap - one window per trial
    drop_last_window=False,
    preload=True
)

print(f"Number of original trials: {len(dataset.datasets[0].raw.annotations)}")
print(f"Number of generated windows: {len(windows_dataset)}")
print("Window metadata:")
print(windows_dataset.get_metadata().head())

splitted = windows_dataset.split('session')
train_set = splitted['0train']
valid_set = splitted['1test']

# -------------------------------------------
# Training
# -------------------------------------------
lr = 0.001  # Standard learning rate for EEGNet
batch_size = 32
n_epochs = 100

clf = EEGClassifier(
    model,
    cropped=False,  # EEGNet doesn't use cropped decoding
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.AdamW,
    train_split=predefined_split(valid_set),
    optimizer__lr=lr,
    optimizer__weight_decay=1e-4,
    iterator_train__shuffle=True,
    batch_size=batch_size,
    callbacks=[
        "accuracy",
        ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
        EarlyStopping(patience=15, threshold=0.001)
    ],
    device=device,
    classes=list(range(n_classes))
)

print("Starting EEGNetv4 training...")
_ = clf.fit(train_set, y=None, epochs=n_epochs)
print("✅ Training completed. Starting evaluation and model saving...")

# -------------------------------------------
# Plot Results and Save Model
# -------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
log_dir = os.path.join(project_root, 'log')
model_dir = os.path.join(project_root, 'models')
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

plot_path = os.path.join(log_dir, 'eegnetv4_subj3_training.png')
conf_mat_path = os.path.join(log_dir, 'eegnetv4_subj3_confmat.png')

results_columns = ['train_loss', 'valid_loss', 'train_accuracy', 'valid_accuracy']
df = pd.DataFrame(clf.history[:, results_columns], columns=results_columns, index=clf.history[:, 'epoch'])
df = df.assign(train_misclass=100 - 100 * df.train_accuracy, valid_misclass=100 - 100 * df.valid_accuracy)

fig, ax1 = plt.subplots(figsize=(8, 3))
df[['train_loss', 'valid_loss']].plot(ax=ax1, style=['-', ':'], marker='o', color='tab:blue', legend=False)
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_ylabel("Loss", color='tab:blue')

ax2 = ax1.twinx()
df[['train_misclass', 'valid_misclass']].plot(ax=ax2, style=['-', ':'], marker='o', color='tab:red', legend=False)
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.set_ylabel("Misclassification [%]", color='tab:red')
ax1.set_xlabel("Epoch")

handles = [Line2D([0], [0], color='black', linestyle='-', label='Train'), Line2D([0], [0], color='black', linestyle=':', label='Valid')]
plt.legend(handles=handles)
plt.tight_layout()
plt.savefig(plot_path)
plt.close()

y_true = valid_set.get_metadata().target
y_pred = clf.predict(valid_set)
conf_mat = confusion_matrix(y_true, y_pred)

# Get label mapping
if hasattr(valid_set.datasets[0], 'window_kwargs') and valid_set.datasets[0].window_kwargs:
    label_dict = valid_set.datasets[0].window_kwargs[0][1]['mapping']
    labels = [k for k, v in sorted(label_dict.items(), key=lambda kv: kv[1])]
else:
    labels = ['feet', 'left_hand', 'right_hand', 'tongue']

fig_cm = plot_confusion_matrix(conf_mat, class_names=labels)
fig_cm.savefig(conf_mat_path)
plt.close(fig_cm)

# Calculate and print accuracy
accuracy = np.mean(y_true == y_pred)
print(f"Final validation accuracy: {accuracy*100:.2f}%")

# Save model
torch.save(model.state_dict(), os.path.join(model_dir, 'eegnetv4_subj3_model_250.pth'))
torch.save(model, os.path.join(model_dir, 'eegnetv4_subj3_model_250_full.pth'))

print(f"✅ Model saved to {model_dir}")
print(f"✅ Training plots saved to {log_dir}")
print(f"✅ Final accuracy: {accuracy*100:.2f}%")