import os
import numpy as np
import pandas as pd
import torch
from braindecode.datasets import MOABBDataset
from braindecode.datasets.base import BaseConcatDataset
from braindecode.preprocessing import Preprocessor, preprocess, create_windows_from_events
from braindecode.models import ShallowFBCSPNet
from braindecode.util import set_random_seeds
from braindecode import EEGClassifier
from braindecode.training import CroppedLoss
from braindecode.preprocessing import exponential_moving_standardize
from skorch.callbacks import LRScheduler, EarlyStopping
from skorch.helper import predefined_split


def run_training(subject_id, run_id):
    included_channels = [
        'C3', 'C4', 'Cz',
        'FC1', 'FC2', 'FCz',
        'CP1', 'CP2', 'CPz',
        'P1', 'P2', 'Pz',
        'C1', 'C2',
        'CP3', 'CP4'
    ]

    dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[subject_id])
    preprocessors = [
        Preprocessor('pick_channels', ch_names=included_channels, ordered=True),
        Preprocessor(lambda data: data * 1e6),
        Preprocessor('resample', sfreq=125),
        Preprocessor('filter', l_freq=4, h_freq=38),
        Preprocessor(exponential_moving_standardize, apply_on_array=True, factor_new=1e-3, init_block_size=1000)
    ]
    preprocess(dataset, preprocessors, n_jobs=1)

    input_window_samples = 500
    n_classes = 4
    n_chans = dataset[0][0].shape[0]
    model = ShallowFBCSPNet(n_chans, n_classes, input_window_samples=input_window_samples, final_conv_length='auto')
    model.to_dense_prediction_model()

    cuda = torch.cuda.is_available()
    device = 'cuda' if cuda else 'cpu'
    if cuda:
        model.cuda()
        torch.backends.cudnn.benchmark = True

    set_random_seeds(seed=2020 + run_id, cuda=cuda)

    n_preds_per_input = model.get_output_shape()[2]
    sfreq = dataset.datasets[0].raw.info['sfreq']
    trial_start_offset_samples = int(-0.5 * sfreq)

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

    clf = EEGClassifier(
        model,
        cropped=True,
        criterion=CroppedLoss,
        criterion__loss_function=torch.nn.functional.nll_loss,
        optimizer=torch.optim.AdamW,
        train_split=predefined_split(valid_set),
        optimizer__lr=0.0625 * 0.01,
        optimizer__weight_decay=0,
        iterator_train__shuffle=True,
        batch_size=64,
        callbacks=[
            "accuracy",
            ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=249)),
            ("early_stopping", EarlyStopping(patience=20, monitor='valid_loss', lower_is_better=True))
        ],
        device=device,
        classes=list(range(n_classes))
    )

    clf.fit(train_set, y=None, epochs=250)
    return clf.history[-1]['valid_accuracy'] * 100


def main():
    subject_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    n_runs = 5
    all_results = []

    for subject_id in subject_ids:
        print(f"\n==== Subjekt {subject_id} ====")
        accuracies = []

        for run in range(n_runs):
            print(f"--- Run {run+1} ---")
            acc = run_training(subject_id, run)
            accuracies.append(acc)

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        all_results.append({
            "subject": subject_id,
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "run1": accuracies[0],
            "run2": accuracies[1],
            "run3": accuracies[2],
            "run4": accuracies[3],
            "run5": accuracies[4],
        })

    df_results = pd.DataFrame(all_results)
    print("\n===== Zusammenfassung =====")
    print(df_results)

    os.makedirs("results", exist_ok=True)
    df_results.to_csv("results/shallow_fbcsp_crossval_results.csv", index=False)


if __name__ == "__main__":
    main()
