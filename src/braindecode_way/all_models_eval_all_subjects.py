from braindecode.datasets import MOABBDataset
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
log_dir = os.path.join(project_root, 'log')
model_dir = os.path.join(project_root, 'models')
os.makedirs(log_dir, exist_ok=True)

for i in range(1, 10):
    subject_id = i
    dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[subject_id])

    from numpy import multiply

    from braindecode.preprocessing import (
        Preprocessor,
        exponential_moving_standardize,
        preprocess,
    )

    low_cut_hz = 4.  # low cut frequency for filtering
    high_cut_hz = 38.  # high cut frequency for filtering
    # Parameters for exponential moving standardization
    factor_new = 1e-3
    init_block_size = 1000
    # Factor to convert from V to uV
    factor = 1e6

    preprocessors = [
        Preprocessor('pick_types', eeg=True, meg=False, stim=False),
        # Keep EEG sensors
        Preprocessor(lambda data: multiply(data, factor)),  # Convert from V to uV
        Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),
        # Bandpass filter
        Preprocessor(exponential_moving_standardize,
                    # Exponential moving standardization
                    factor_new=factor_new, init_block_size=init_block_size)
    ]

    # Transform the data
    preprocess(dataset, preprocessors, n_jobs=-1)

    input_window_samples = 1000
    import torch

    from braindecode.models import ShallowFBCSPNet
    from braindecode.util import set_random_seeds

    cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
    device = 'cuda' if cuda else 'cpu'
    if cuda:
        torch.backends.cudnn.benchmark = True

    print(f"Using device: {device}")
    # Set random seed to be able to roughly reproduce results
    # Note that with cudnn benchmark set to True, GPU indeterminism
    # may still make results substantially different between runs.
    # To obtain more consistent results at the cost of increased computation time,
    # you can set `cudnn_benchmark=False` in `set_random_seeds`
    # or remove `torch.backends.cudnn.benchmark = True`
    seed = 20200220
    set_random_seeds(seed=seed, cuda=cuda)

    n_classes = 4
    classes = list(range(n_classes))
    # Extract number of chans from dataset
    n_chans = dataset[0][0].shape[0]

    model = ShallowFBCSPNet(
        n_chans,
        n_classes,
        input_window_samples=input_window_samples,
        final_conv_length=30,
    )

    # Display torchinfo table describing the model
    print(model)

    # Send model to GPU
    if cuda:
        _ = model.cuda()

    n_preds_per_input = model.get_output_shape()[2]

    ######################################################################
    # Cut the data into windows
    # -------------------------
    # In contrast to trialwise decoding, we have to supply an explicit
    # window size and window stride to the ``create_windows_from_events``
    # function.
    #

    from braindecode.preprocessing import create_windows_from_events

    trial_start_offset_seconds = -0.5
    # Extract sampling frequency, check that they are same in all datasets
    sfreq = dataset.datasets[0].raw.info['sfreq']
    assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])

    # Calculate the trial start offset in samples.
    trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

    # Create windows using braindecode function for this. It needs parameters to define how
    # trials should be used.
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=0,
        window_size_samples=input_window_samples,
        window_stride_samples=n_preds_per_input,
        drop_last_window=False,
        preload=True
    )

    ######################################################################
    # Split the dataset
    # -----------------
    #
    # This code is the same as in trialwise decoding.
    #

    splitted = windows_dataset.split('session')
    train_set = splitted['0train']  # Session train
    valid_set = splitted['1test']  # Session evaluation

    from skorch.callbacks import LRScheduler
    from skorch.helper import predefined_split

    from braindecode import EEGClassifier
    from braindecode.training import CroppedLoss

    # These values we found good for shallow network:
    lr = 0.0625 * 0.01
    weight_decay = 0

    # For deep4 they should be:
    # lr = 1 * 0.01
    # weight_decay = 0.5 * 0.001

    batch_size = 64
    n_epochs = 100


    # Model training for a specified number of epochs. `y` is None as it is already supplied
    # in the dataset.
    #_ = clf.fit(train_set, y=None, epochs=n_epochs)
    for x in range(1, 10):
        model.load_state_dict(torch.load(os.path.join(model_dir, f"subject{x}_model.pth")))

        model.eval()

        clf = EEGClassifier(
        model,
        cropped=True,
        criterion=CroppedLoss,
        criterion__loss_function=torch.nn.functional.nll_loss,
        train_split=predefined_split(valid_set),
        optimizer=torch.optim.AdamW,
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        iterator_train__shuffle=True,
        batch_size=batch_size,
        callbacks=["accuracy"],
        device=device,
        classes=classes,
        )

        _ = clf.initialize()

        with torch.no_grad():
            y_pred = clf.predict(valid_set)
            y_true = valid_set.get_metadata().target
            acc = (y_pred == y_true).mean()


        import matplotlib.pyplot as plt
        import pandas as pd
        from matplotlib.lines import Line2D
        from sklearn.metrics import confusion_matrix
        from braindecode.visualization import plot_confusion_matrix
        conf_mat_path = os.path.join(log_dir, f'subject{subject_id}_confusion_matrix.png')
            # Generate and save confusion matrix
        conf_mat = confusion_matrix(y_true, y_pred)
        label_dict = valid_set.datasets[0].window_kwargs[0][1]['mapping']
        labels = [k for k, v in sorted(label_dict.items(), key=lambda kv: kv[1])]
        fig_cm = plot_confusion_matrix(conf_mat, class_names=labels)
        conf_mat_path = os.path.join(log_dir, f'subject{subject_id}_model{x}_confusion_matrix.png')
        fig_cm.savefig(conf_mat_path)
        plt.close(fig_cm)