#!/usr/bin/env python3
"""
EEGNetv4 Training Script for XDF Files

Trains EEGNetv4 model on XDF files recorded with OpenBCI setup.
This script allows training on custom EEG recordings from OpenBCI hardware
and can be used to create personalized models for real-time BCI applications.

The script supports both marker-based training (with labeled trials) and
continuous training (without markers). EEGNetv4 is the preferred model for
real-time classification due to its robust performance.
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from collections import defaultdict

# XDF imports
try:
    import pyxdf
except ImportError:
    print("pyxdf not found. Install with: pip install pyxdf")
    sys.exit(1)

# Braindecode imports
from braindecode.models import EEGNetv4
from braindecode.util import set_random_seeds

# Logging utility
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
from utils.logging_utils import setup_logging

# Setup logger
logger = setup_logging('eegnet_xdf_training', log_dir='logs', level='INFO')

class XDFTrainer:
    """Trainer for XDF files"""

    def __init__(self, xdf_path, config=None):
        self.xdf_path = xdf_path
        self.config = config or self._get_default_config()

        # Data
        self.eeg_data = None
        self.marker_data = None
        self.eeg_timestamps = None
        self.marker_timestamps = None
        self.eeg_stream_info = None
        self.marker_stream_info = None

        # Training data
        self.train_data = None
        self.train_labels = None
        self.val_data = None
        self.val_labels = None

        # Model
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f"Initialized XDF trainer for {xdf_path}")
        logger.info(f"Using device: {self.device}")

    def _get_default_config(self):
        """Get default configuration"""
        return {
            'window_size_samples': 250,  # 2 seconds at 125 Hz
            'window_stride_samples': 125,  # 50% overlap
            'n_classes': 4,
            'batch_size': 32,
            'learning_rate': 0.001,
            'n_epochs': 100,
            'validation_split': 0.2,
            'random_seed': 20200220,
            'included_channels': [
                'C3', 'C4', 'Cz',
                'FC1', 'FC2', 'FCz',
                'CP1', 'CP2', 'CPz',
                'P1', 'P2', 'Pz',
                'C1', 'C2',
                'CP3', 'CP4'
            ],
            'class_mapping': {
                'left': 0,
                'right': 1,
                'up': 2,
                'down': 3
            }
        }

    def load_xdf_data(self):
        """Load XDF file data"""
        try:
            logger.info(f"Loading XDF file: {self.xdf_path}")

            # Load XDF file
            streams, header = pyxdf.load_xdf(self.xdf_path)
            logger.info(f"Loaded {len(streams)} streams")

            # Find EEG and marker streams
            eeg_stream = None
            marker_stream = None

            for i, stream in enumerate(streams):
                stream_info = stream['info']
                stream_name = stream_info.get('name', ['Unknown'])[0]
                stream_type = stream_info.get('type', ['Unknown'])[0]

                logger.info(f"Stream {i}: {stream_name} ({stream_type})")

                if 'eeg' in stream_name.lower() or 'eeg' in stream_type.lower():
                    eeg_stream = stream
                    logger.info(f"Found EEG stream: {stream_name}")
                elif 'marker' in stream_name.lower() or 'marker' in stream_type.lower():
                    marker_stream = stream
                    logger.info(f"Found marker stream: {stream_name}")

            if eeg_stream is None:
                raise ValueError("No EEG stream found in XDF file")

            # Extract EEG data
            self.eeg_data = np.array(eeg_stream['time_series'])
            self.eeg_timestamps = np.array(eeg_stream['time_stamps'])
            self.eeg_stream_info = eeg_stream['info']

            logger.info(f"EEG data shape: {self.eeg_data.shape}")
            logger.info(f"EEG sampling rate: {self.eeg_stream_info.get('nominal_srate', ['Unknown'])[0]}")

            # Extract marker data if available
            if marker_stream is not None:
                self.marker_data = np.array(marker_stream['time_series'])
                self.marker_timestamps = np.array(marker_stream['time_stamps'])
                self.marker_stream_info = marker_stream['info']

                logger.info(f"Marker data shape: {self.marker_data.shape}")
                logger.info(f"Number of markers: {len(self.marker_data)}")
            else:
                logger.warning("No marker stream found - will use continuous classification")

        except Exception as e:
            logger.error(f"Failed to load XDF data: {e}")
            raise

    def preprocess_eeg_data(self):
        """Preprocess EEG data"""
        try:
            logger.info("Preprocessing EEG data...")

            # Select channels if specified
            if 'included_channels' in self.config:
                # Get channel names from stream info
                ch_names = self.eeg_stream_info.get('desc', {}).get('channels', {}).get('channel', [])
                if ch_names:
                    ch_names = [ch.get('label', [f'CH{i}'])[0] for ch in ch_names]
                    logger.info(f"Available channels: {ch_names}")

                    # Find indices of included channels
                    included_indices = []
                    for ch_name in self.config['included_channels']:
                        if ch_name in ch_names:
                            included_indices.append(ch_names.index(ch_name))
                        else:
                            logger.warning(f"Channel {ch_name} not found in data")

                    if included_indices:
                        self.eeg_data = self.eeg_data[:, included_indices]
                        logger.info(f"Selected {len(included_indices)} channels")
                    else:
                        logger.warning("No included channels found, using all channels")

            # Normalize data
            self.eeg_data = (self.eeg_data - np.mean(self.eeg_data, axis=0)) / \
                           (np.std(self.eeg_data, axis=0) + 1e-8)

            logger.info(f"Preprocessed EEG data shape: {self.eeg_data.shape}")

        except Exception as e:
            logger.error(f"Failed to preprocess EEG data: {e}")
            raise

    def create_windows_from_markers(self):
        """Create training windows from marker data"""
        try:
            if self.marker_data is None:
                logger.warning("No marker data available - cannot create labeled windows")
                return False

            logger.info("Creating windows from markers...")

            window_size = self.config['window_size_samples']
            window_stride = self.config['window_stride_samples']
            class_mapping = self.config['class_mapping']

            windows = []
            labels = []

            # Process each marker
            for i, marker in enumerate(self.marker_data):
                marker_time = self.marker_timestamps[i]
                marker_value = marker[0] if isinstance(marker, (list, np.ndarray)) else marker

                # Skip start markers
                if marker_value == 'start':
                    continue

                # Map marker to class
                if marker_value in class_mapping:
                    class_id = class_mapping[marker_value]
                else:
                    logger.warning(f"Unknown marker: {marker_value}")
                    continue

                # Find corresponding EEG data
                eeg_idx = np.argmin(np.abs(self.eeg_timestamps - marker_time))

                # Create window around marker
                start_idx = max(0, eeg_idx - window_size // 2)
                end_idx = min(len(self.eeg_data), start_idx + window_size)

                if end_idx - start_idx == window_size:
                    window = self.eeg_data[start_idx:end_idx]
                    windows.append(window)
                    labels.append(class_id)

                    logger.debug(f"Created window for marker {marker_value} -> class {class_id}")

            if windows:
                self.train_data = np.array(windows)
                self.train_labels = np.array(labels)

                logger.info(f"Created {len(windows)} windows")
                logger.info(f"Class distribution: {np.bincount(self.train_labels)}")

                return True
            else:
                logger.error("No valid windows created")
                return False

        except Exception as e:
            logger.error(f"Failed to create windows: {e}")
            raise

    def create_windows_continuous(self):
        """Create windows from continuous EEG data (no markers)"""
        try:
            logger.info("Creating continuous windows...")

            window_size = self.config['window_size_samples']
            window_stride = self.config['window_stride_samples']

            windows = []

            # Create overlapping windows
            for start_idx in range(0, len(self.eeg_data) - window_size + 1, window_stride):
                window = self.eeg_data[start_idx:start_idx + window_size]
                windows.append(window)

            if windows:
                self.train_data = np.array(windows)
                # For continuous data, we don't have labels
                self.train_labels = np.zeros(len(windows))  # Dummy labels

                logger.info(f"Created {len(windows)} continuous windows")
                return True
            else:
                logger.error("No windows created")
                return False

        except Exception as e:
            logger.error(f"Failed to create continuous windows: {e}")
            raise

    def split_data(self):
        """Split data into train and validation sets"""
        try:
            if self.train_data is None:
                raise ValueError("No training data available")

            n_samples = len(self.train_data)
            val_size = int(n_samples * self.config['validation_split'])

            # Random shuffle
            indices = np.random.permutation(n_samples)

            val_indices = indices[:val_size]
            train_indices = indices[val_size:]

            self.val_data = self.train_data[val_indices]
            self.val_labels = self.train_labels[val_indices]
            self.train_data = self.train_data[train_indices]
            self.train_labels = self.train_labels[train_indices]

            logger.info(f"Train set: {len(self.train_data)} samples")
            logger.info(f"Validation set: {len(self.val_data)} samples")

        except Exception as e:
            logger.error(f"Failed to split data: {e}")
            raise

    def create_model(self):
        """Create EEGNetv4 model"""
        try:
            n_channels = self.train_data.shape[1]
            n_times = self.train_data.shape[2]
            n_classes = self.config['n_classes']

            logger.info(f"Creating EEGNetv4 model: {n_channels} channels, {n_times} time points, {n_classes} classes")

            self.model = EEGNetv4(
                n_chans=n_channels,
                n_outputs=n_classes,
                n_times=n_times,
                drop_prob=0.25,
                kernel_length=64,
            )

            self.model = self.model.to(self.device)

            # Set random seeds
            set_random_seeds(seed=self.config['random_seed'], cuda=self.device.type == 'cuda')

            logger.info("Model created successfully")

        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            raise

    def train_model(self):
        """Train the model"""
        try:
            logger.info("Starting model training...")

            # Convert data to tensors
            train_data = torch.FloatTensor(self.train_data).to(self.device)
            train_labels = torch.LongTensor(self.train_labels).to(self.device)
            val_data = torch.FloatTensor(self.val_data).to(self.device)
            val_labels = torch.LongTensor(self.val_labels).to(self.device)

            # Setup optimizer and loss
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=1e-4
            )
            criterion = torch.nn.CrossEntropyLoss()

            # Training loop
            train_losses = []
            val_losses = []
            train_accuracies = []
            val_accuracies = []

            best_val_acc = 0.0
            patience_counter = 0
            patience = 15

            for epoch in range(self.config['n_epochs']):
                # Training
                self.model.train()
                optimizer.zero_grad()

                outputs = self.model(train_data)
                loss = criterion(outputs, train_labels)
                loss.backward()
                optimizer.step()

                # Calculate training accuracy
                _, predicted = torch.max(outputs.data, 1)
                train_acc = (predicted == train_labels).float().mean().item()
                train_loss = loss.item()

                # Validation
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(val_data)
                    val_loss = criterion(val_outputs, val_labels)
                    _, val_predicted = torch.max(val_outputs.data, 1)
                    val_acc = (val_predicted == val_labels).float().mean().item()

                # Record metrics
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                train_accuracies.append(train_acc)
                val_accuracies.append(val_acc)

                # Log progress
                logger.info(f"Epoch {epoch+1}/{self.config['n_epochs']}: "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(),
                             Path(self.config.get('model_save_dir', 'models')) / 'eegnet_xdf_best.pth')
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

            logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")

            # Save training history
            history = {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies
            }

            return history

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def evaluate_model(self):
        """Evaluate the trained model"""
        try:
            logger.info("Evaluating model...")

            self.model.eval()
            with torch.no_grad():
                val_data = torch.FloatTensor(self.val_data).to(self.device)
                outputs = self.model(val_data)
                _, predicted = torch.max(outputs.data, 1)

                # Calculate accuracy
                accuracy = (predicted == torch.LongTensor(self.val_labels).to(self.device)).float().mean().item()

                # Confusion matrix
                conf_mat = confusion_matrix(self.val_labels, predicted.cpu().numpy())

                logger.info(f"Validation accuracy: {accuracy:.4f}")
                logger.info(f"Confusion matrix:\n{conf_mat}")

                return accuracy, conf_mat

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise

    def save_results(self, history, accuracy, conf_mat):
        """Save training results"""
        try:
            # Ensure output directories exist
            log_dir = Path(self.config.get('log_dir', 'logs'))
            model_dir = Path(self.config.get('model_save_dir', 'models'))
            log_dir.mkdir(exist_ok=True)
            model_dir.mkdir(exist_ok=True)

            # Save final model
            torch.save(self.model.state_dict(), model_dir / 'eegnet_xdf_final.pth')
            torch.save(self.model, model_dir / 'eegnet_xdf_full.pth')

            # Plot training history
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            # Loss plot
            ax1.plot(history['train_losses'], label='Train Loss')
            ax1.plot(history['val_losses'], label='Val Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Loss')
            ax1.legend()
            ax1.grid(True)

            # Accuracy plot
            ax2.plot(history['train_accuracies'], label='Train Acc')
            ax2.plot(history['val_accuracies'], label='Val Acc')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Training Accuracy')
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()
            plt.savefig(log_dir / 'eegnet_xdf_training.png')
            plt.close()

            # Save confusion matrix
            class_names = ['left', 'right', 'up', 'down']
            fig_cm = plt.figure(figsize=(8, 6))
            plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=45)
            plt.yticks(tick_marks, class_names)

            # Add text annotations
            thresh = conf_mat.max() / 2.
            for i in range(conf_mat.shape[0]):
                for j in range(conf_mat.shape[1]):
                    plt.text(j, i, format(conf_mat[i, j], 'd'),
                            horizontalalignment="center",
                            color="white" if conf_mat[i, j] > thresh else "black")

            plt.tight_layout()
            plt.savefig(log_dir / 'eegnet_xdf_confmat.png')
            plt.close()

            logger.info(f"Results saved to {log_dir} and {model_dir}")

        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise

    def run_training(self):
        """Run complete training pipeline"""
        try:
            logger.info("=" * 60)
            logger.info("EEGNetv4 XDF Training Pipeline")
            logger.info("=" * 60)

            # Load XDF data
            self.load_xdf_data()

            # Preprocess EEG data
            self.preprocess_eeg_data()

            # Create windows
            if self.marker_data is not None:
                success = self.create_windows_from_markers()
                if not success:
                    logger.warning("Falling back to continuous windows")
                    self.create_windows_continuous()
            else:
                self.create_windows_continuous()

            # Split data
            self.split_data()

            # Create model
            self.create_model()

            # Train model
            history = self.train_model()

            # Evaluate model
            accuracy, conf_mat = self.evaluate_model()

            # Save results
            self.save_results(history, accuracy, conf_mat)

            logger.info("Training pipeline completed successfully!")

        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train EEGNetv4 on XDF files')
    parser.add_argument('xdf_path', type=str, help='Path to XDF file')
    parser.add_argument('--window-size', type=int, default=250, help='Window size in samples')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--validation-split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--model-save-dir', type=str, default='models', help='Model save directory')
    parser.add_argument('--log-dir', type=str, default='logs', help='Log directory')

    args = parser.parse_args()

    # Update config with command line arguments
    config = {
        'window_size_samples': args.window_size,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'n_epochs': args.epochs,
        'validation_split': args.validation_split,
        'model_save_dir': args.model_save_dir,
        'log_dir': args.log_dir
    }

    try:
        # Create trainer
        trainer = XDFTrainer(args.xdf_path, config)

        # Run training
        trainer.run_training()

    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()