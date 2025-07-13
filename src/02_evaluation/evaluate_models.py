#!/usr/bin/env python3
"""
Model Evaluation Script

Evaluates trained EEGNet and ShallowFBCSP models with basic metrics including
accuracy, confusion matrices, and classification reports.

This script provides comprehensive evaluation of model performance on the test set
and can be used to compare different models or training approaches.
"""

import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# Braindecode imports
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import Preprocessor, preprocess, create_windows_from_events
from braindecode.models import EEGNetv4, ShallowFBCSPNet
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
logger = setup_logging('model_evaluation', log_dir='logs', level='INFO')

# Configuration
subject_ids = [3]  # Evaluate on subject 3
included_channels = [
    'C3', 'C4', 'Cz',
    'FC1', 'FC2', 'FCz',
    'CP1', 'CP2', 'CPz',
    'P1', 'P2', 'Pz',
    'C1', 'C2',
    'CP3', 'CP4'
]
input_window_samples = 250
n_classes = 4
batch_size = 32

# Ensure output directories exist
log_dir = project_root / 'logs'
model_dir = project_root / 'models'
log_dir.mkdir(exist_ok=True)
model_dir.mkdir(exist_ok=True)

def evaluate_model(model_path: str, model_type: str = 'eegnet'):
    """Evaluate a trained model"""
    logger.info(f"Evaluating {model_type} model: {model_path}")

    # Load dataset
    logger.info("Loading MOABB dataset...")
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

    # Create model
    n_chans = dataset[0][0].shape[0]
    if model_type == 'eegnet':
        model = EEGNetv4(
            n_chans=n_chans,
            n_outputs=n_classes,
            n_times=input_window_samples,
            drop_prob=0.25,
            kernel_length=64,
        )
    elif model_type == 'shallow_fbcsp':
        model = ShallowFBCSPNet(
            n_chans=n_chans,
            n_outputs=n_classes,
            n_times=input_window_samples,
            final_conv_length='auto',
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load trained weights
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    logger.info(f"Model loaded successfully")

    # Setup device
    cuda = torch.cuda.is_available()
    device = 'cuda' if cuda else 'cpu'
    if cuda:
        model.cuda()
        torch.backends.cudnn.benchmark = True
    set_random_seeds(seed=20200220, cuda=cuda)

    # Create windows dataset
    sfreq = dataset.datasets[0].raw.info['sfreq']
    trial_start_offset_samples = int(-0.5 * sfreq)
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=0,
        window_size_samples=input_window_samples,
        window_stride_samples=input_window_samples,
        drop_last_window=False,
        preload=True
    )
    logger.info(f"Number of windows: {len(windows_dataset)}")

    # Split data
    splitted = windows_dataset.split('session')
    train_set = splitted['0train']
    valid_set = splitted['1test']
    logger.info(f"Validation set: {len(valid_set)} samples")

    # Evaluate model directly without creating a new classifier
    logger.info("Starting evaluation...")

    # Prepare data for evaluation
    y_true = valid_set.get_metadata().target

    # Create predictions using the loaded model directly
    model.eval()
    predictions = []

    with torch.no_grad():
        for i in range(len(valid_set)):
            # Get sample
            sample = valid_set[i][0]  # Get the EEG data

            # Reshape for model input (batch_size=1, channels, time_points)
            if len(sample.shape) == 2:  # (channels, time_points)
                sample = np.expand_dims(sample, axis=0)  # Add batch dimension

            # Convert to torch tensor and move to device
            sample = torch.from_numpy(sample).float().to(device)

            # Get prediction
            output = model(sample)
            probabilities = torch.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1).cpu().numpy()[0]
            predictions.append(prediction)

    y_pred = np.array(predictions)

    # Calculate metrics
    accuracy = np.mean(y_true == y_pred)
    conf_mat = confusion_matrix(y_true, y_pred)

    # Print results
    logger.info(f"Evaluation completed!")
    logger.info(f"Accuracy: {accuracy*100:.2f}%")
    logger.info(f"Confusion Matrix:")
    logger.info(conf_mat)

    # Print classification report
    labels = ['feet', 'left_hand', 'right_hand', 'tongue']
    report = classification_report(y_true, y_pred, target_names=labels)
    logger.info(f"Classification Report:")
    logger.info(report)

    # Save confusion matrix plot
    fig_cm = plot_confusion_matrix(conf_mat, class_names=labels)
    conf_mat_path = log_dir / f'{model_type}_evaluation_confmat.png'
    fig_cm.savefig(conf_mat_path)
    plt.close(fig_cm)
    logger.info(f"Confusion matrix saved to {conf_mat_path}")

    return accuracy, conf_mat, report

def main():
    """Main evaluation function"""
    logger.info("=" * 60)
    logger.info("Model Evaluation Script")
    logger.info("=" * 60)

    # Evaluate EEGNet model
    eegnet_path = model_dir / 'eegnetv4_subj3_model_250.pth'
    if eegnet_path.exists():
        logger.info("Evaluating EEGNet model...")
        eegnet_accuracy, eegnet_conf_mat, eegnet_report = evaluate_model(
            str(eegnet_path), 'eegnet'
        )
    else:
        logger.warning(f"EEGNet model not found: {eegnet_path}")

    # Evaluate ShallowFBCSP model
    shallow_path = model_dir / 'shallow_fbcsp_subj3_model_250.pth'
    if shallow_path.exists():
        logger.info("Evaluating ShallowFBCSP model...")
        shallow_accuracy, shallow_conf_mat, shallow_report = evaluate_model(
            str(shallow_path), 'shallow_fbcsp'
        )
    else:
        logger.warning(f"ShallowFBCSP model not found: {shallow_path}")

    # Compare results if both models exist
    if eegnet_path.exists() and shallow_path.exists():
        logger.info("=" * 60)
        logger.info("Model Comparison")
        logger.info("=" * 60)
        logger.info(f"EEGNet Accuracy: {eegnet_accuracy*100:.2f}%")
        logger.info(f"ShallowFBCSP Accuracy: {shallow_accuracy*100:.2f}%")

        if eegnet_accuracy > shallow_accuracy:
            logger.info("EEGNet performed better")
        elif shallow_accuracy > eegnet_accuracy:
            logger.info("ShallowFBCSP performed better")
        else:
            logger.info("Both models performed equally")

    logger.info("Evaluation completed successfully!")

if __name__ == "__main__":
    main()