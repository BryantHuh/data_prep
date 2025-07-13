#!/usr/bin/env python3
"""
ShallowFBCSPNet Training Script - Good Subjects

Trains ShallowFBCSPNet model on MOABB BNCI2014_001 dataset with all 4 good subjects (1, 3, 8, 9).
This model is based on Schirrmeister et al. (2017) and works well for offline classification tasks.

This script trains a multi-subject model that can potentially improve generalization
compared to single-subject models. However, note that ShallowFBCSPNet has shown
complications with streaming accuracy in real-time applications.
"""

import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

# Braindecode imports
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import Preprocessor, preprocess, create_windows_from_events
from braindecode.models import ShallowFBCSPNet
from braindecode.util import set_random_seeds
from braindecode import EEGClassifier
from braindecode.visualization import plot_confusion_matrix
from skorch.callbacks import LRScheduler, EarlyStopping
from skorch.helper import predefined_split

# Logging utility
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
from utils.logging_utils import setup_logging

# Setup logger
logger = setup_logging('shallow_fbcsp_good_subjects_training', log_dir='logs', level='INFO')

# Configuration
subject_ids = [1, 3, 8, 9]  # Good subjects
included_channels = [
    'C3', 'C4', 'Cz',
    'FC1', 'FC2', 'FCz',
    'CP1', 'CP2', 'CPz',
    'P1', 'P2', 'Pz',
    'C1', 'C2',
    'CP3', 'CP4'
]
input_window_samples = 250  # 2 seconds * 125 Hz
n_classes = 4
batch_size = 32
n_epochs = 100
learning_rate = 0.001

# Ensure output directories exist
log_dir = project_root / 'logs'
model_dir = project_root / 'models'
log_dir.mkdir(exist_ok=True)
model_dir.mkdir(exist_ok=True)

# Load dataset
logger.info("Loading MOABB dataset with good subjects...")
dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=subject_ids)

# Preprocessing
preprocessors = [
    Preprocessor('pick_channels', ch_names=included_channels, ordered=True),
    Preprocessor(lambda data: data * 1e6),  # Scale to microvolts
    Preprocessor('resample', sfreq=125),
]
logger.info("Applying preprocessing...")
preprocess(dataset, preprocessors, n_jobs=-1)
logger.info("Preprocessing completed.")

# Model and window parameters
n_chans = dataset[0][0].shape[0]
model = ShallowFBCSPNet(
    n_chans=n_chans,
    n_outputs=n_classes,
    n_times=input_window_samples,
    final_conv_length='auto',
)

cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'
if cuda:
    model.cuda()
    torch.backends.cudnn.benchmark = True
set_random_seeds(seed=20200220, cuda=cuda)

# Windows dataset
sfreq = dataset.datasets[0].raw.info['sfreq']
trial_start_offset_samples = int(-0.5 * sfreq)
windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=trial_start_offset_samples,
    trial_stop_offset_samples=0,
    window_size_samples=input_window_samples,
    window_stride_samples=input_window_samples,  # No overlap
    drop_last_window=False,
    preload=True
)
logger.info(f"Number of original trials: {len(dataset.datasets[0].raw.annotations)}")
logger.info(f"Number of generated windows: {len(windows_dataset)}")

splitted = windows_dataset.split('session')
train_set = splitted['0train']
valid_set = splitted['1test']
logger.info(f"Train set: {len(train_set)} samples")
logger.info(f"Validation set: {len(valid_set)} samples")

# Training
clf = EEGClassifier(
    model,
    cropped=False,  # ShallowFBCSPNet doesn't use cropped decoding
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.AdamW,
    train_split=predefined_split(valid_set),
    optimizer__lr=learning_rate,
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

logger.info("Starting ShallowFBCSPNet training with good subjects...")
_ = clf.fit(train_set, y=None, epochs=n_epochs)
logger.info("Training completed. Starting evaluation and model saving...")

# Plot Results and Save Model
plot_path = log_dir / 'shallow_fbcsp_good_subjects_training.png'
conf_mat_path = log_dir / 'shallow_fbcsp_good_subjects_confmat.png'

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

plt.tight_layout()
plt.savefig(plot_path)
plt.close()
logger.info(f"Training plot saved to {plot_path}")

y_true = valid_set.get_metadata().target
y_pred = clf.predict(valid_set)
conf_mat = confusion_matrix(y_true, y_pred)
labels = ['feet', 'left_hand', 'right_hand', 'tongue']
fig_cm = plot_confusion_matrix(conf_mat, class_names=labels)
fig_cm.savefig(conf_mat_path)
plt.close(fig_cm)
logger.info(f"Confusion matrix saved to {conf_mat_path}")

accuracy = np.mean(y_true == y_pred)
logger.info(f"Final validation accuracy: {accuracy*100:.2f}%")

torch.save(model.state_dict(), model_dir / 'shallow_fbcsp_good_subjects_model_250.pth')
torch.save(model, model_dir / 'shallow_fbcsp_good_subjects_model_250_full.pth')
logger.info(f"Model saved to {model_dir}")
logger.info(f"Training completed successfully!")