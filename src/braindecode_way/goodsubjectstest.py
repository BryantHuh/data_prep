from braindecode.datasets import MOABBDataset, BaseConcatDataset
from braindecode.preprocessing import Preprocessor, exponential_moving_standardize, preprocess
from braindecode.models import ShallowFBCSPNet
from braindecode.util import set_random_seeds
from braindecode.preprocessing import create_windows_from_events
from braindecode import EEGClassifier
from braindecode.training import CroppedLoss
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
from braindecode.visualization import plot_confusion_matrix

import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from numpy import multiply
from sklearn.metrics import confusion_matrix

# Define the good subjects
subject_ids_all = [1, 3, 8, 9]
subject_id_eval = 9
subject_ids_train = [s for s in subject_ids_all if s != subject_id_eval]

# Load and preprocess data
all_subjects = [MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[sid]) for sid in subject_ids_train]
dataset = BaseConcatDataset(all_subjects)

low_cut_hz = 4.
high_cut_hz = 38.
factor_new = 1e-3
init_block_size = 1000
factor = 1e6

preprocessors = [
    Preprocessor('pick_types', eeg=True, meg=False, stim=False),
    Preprocessor(lambda data: multiply(data, factor)),
    Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),
    Preprocessor(exponential_moving_standardize, factor_new=factor_new, init_block_size=init_block_size)
]

preprocess(dataset, preprocessors, n_jobs=-1)

input_window_samples = 1000

cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True

seed = 20200220
set_random_seeds(seed=seed, cuda=cuda)

n_classes = 4
classes = list(range(n_classes))
n_chans = dataset[0][0].shape[0]

model = ShallowFBCSPNet(
    n_chans,
    n_classes,
    input_window_samples=input_window_samples,
    final_conv_length=30,
)

print(model)
if cuda:
    model.cuda()

model.to_dense_prediction_model()
n_preds_per_input = model.get_output_shape()[2]

trial_start_offset_seconds = -0.5
sfreq = dataset.datasets[0].raw.info['sfreq']
trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=trial_start_offset_samples,
    trial_stop_offset_samples=0,
    window_size_samples=input_window_samples,
    window_stride_samples=n_preds_per_input,
    drop_last_window=False,
    preload=True
)

splitted = windows_dataset.split('session')
train_set = splitted['0train']
valid_set = splitted['1test']

lr = 0.0625 * 0.01
weight_decay = 0
batch_size = 64
n_epochs = 250

clf = EEGClassifier(
    model,
    cropped=True,
    criterion=CroppedLoss,
    criterion__loss_function=torch.nn.functional.nll_loss,
    optimizer=torch.optim.AdamW,
    train_split=predefined_split(valid_set),
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    iterator_train__shuffle=True,
    batch_size=batch_size,
    callbacks=[
        "accuracy",
        ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
    ],
    device=device,
    classes=classes,
)

_ = clf.fit(train_set, y=None, epochs=n_epochs)

whichplot = 'training_performance'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
log_dir = os.path.join(project_root, 'log/leave_one_out')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
model_dir = os.path.join(project_root, 'models')
os.makedirs(log_dir, exist_ok=True)

plot_path = os.path.join(log_dir, f'leave_one_out_subject{subject_id_eval}_{whichplot}.png')
conf_mat_path = os.path.join(log_dir, f'leave_one_out_subject{subject_id_eval}_confusion_matrix.png')

results_columns = ['train_loss', 'valid_loss', 'train_accuracy', 'valid_accuracy']
df = pd.DataFrame(clf.history[:, results_columns], columns=results_columns,
                  index=clf.history[:, 'epoch'])

df = df.assign(train_misclass=100 - 100 * df.train_accuracy,
               valid_misclass=100 - 100 * df.valid_accuracy)

fig, ax1 = plt.subplots(figsize=(8, 3))
df.loc[:, ['train_loss', 'valid_loss']].plot(ax=ax1, style=['-', ':'], marker='o',
                                             color='tab:blue', legend=False, fontsize=14)
ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=14)
ax1.set_ylabel("Loss", color='tab:blue', fontsize=14)

ax2 = ax1.twinx()
df.loc[:, ['train_misclass', 'valid_misclass']].plot(ax=ax2, style=['-', ':'], marker='o',
                                                     color='tab:red', legend=False)
ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=14)
ax2.set_ylabel("Misclassification Rate [%]", color='tab:red', fontsize=14)
ax2.set_ylim(ax2.get_ylim()[0], 85)
ax1.set_xlabel("Epoch", fontsize=14)

handles = [
    Line2D([0], [0], color='black', linewidth=1, linestyle='-', label='Train'),
    Line2D([0], [0], color='black', linewidth=1, linestyle=':', label='Valid')
]
plt.legend(handles=handles, fontsize=14)
plt.tight_layout()
plt.savefig(plot_path)
plt.close()

y_true = valid_set.get_metadata().target
y_pred = clf.predict(valid_set)
conf_mat = confusion_matrix(y_true, y_pred)

label_dict = valid_set.datasets[0].window_kwargs[0][1]['mapping']
labels = [k for k, v in sorted(label_dict.items(), key=lambda kv: kv[1])]

fig_cm = plot_confusion_matrix(conf_mat, class_names=labels)
fig_cm.savefig(conf_mat_path)
plt.close(fig_cm)

torch.save(model.state_dict(), os.path.join(model_dir, f'leave_one_out_subject{subject_id_eval}_model.pth'))
