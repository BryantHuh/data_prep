# -*- coding: utf-8 -*-
"""
FIXED Real-time EEGNetv4 classifier for BCI applications with trial-aligned windows.
Designed for proper timing between predictions and ground truth labels.
"""

import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
import time
from braindecode.models import EEGNetv4

class EEGNetv4RealtimeClassifierFixed:
    """
    Fixed real-time EEGNetv4 classifier with trial-aligned windows.

    Args:
        model_path: Path to the trained EEGNetv4 model
        window_size: Number of samples in the input window (default: 250)
        sample_rate: Sampling rate in Hz (default: 125)
        channels: List of channel names (default: 16 OpenBCI channels)
        device: Device to run inference on ('cpu' or 'cuda')
    """

    def __init__(self, model_path, window_size=250, sample_rate=125,
                 channels=None, device='cpu'):

        self.window_size = window_size
        self.sample_rate = sample_rate
        self.device = device

        # Default channels (16 OpenBCI channels)
        if channels is None:
            self.channels = [
                'C3', 'C4', 'Cz',
                'FC1', 'FC2', 'FCz',
                'CP1', 'CP2', 'CPz',
                'P1', 'P2', 'Pz',
                'C1', 'C2',
                'CP3', 'CP4'
            ]
        else:
            self.channels = channels

        self.n_channels = len(self.channels)
        self.n_classes = 4

        # Load the model
        self.model = self._load_model(model_path)
        self.model.eval()

        # Initialize data buffer (growing buffer for trial alignment)
        self.buffer = []
        self.sample_idx = 0

        # Trial tracking
        self.trial_starts = []  # List of (sample_idx, label) tuples
        self.processed_trials = set()  # Track which trials we've already processed

        # Class labels
        self.class_labels = ['feet', 'left_hand', 'right_hand', 'tongue']

        # Statistics for monitoring
        self.prediction_history = deque(maxlen=100)
        self.confidence_history = deque(maxlen=100)

    def _load_model(self, model_path):
        """Load the trained EEGNetv4 model."""
        try:
            # Try to load the full model first with weights_only=False for PyTorch 2.6+
            model = torch.load(model_path, map_location=self.device, weights_only=False)
            if isinstance(model, EEGNetv4):
                return model
            else:
                # Load state dict and create model
                model = EEGNetv4(
                    n_chans=self.n_channels,
                    n_outputs=self.n_classes,
                    n_times=self.window_size,
                    drop_prob=0.25,
                    kernel_length=64
                )
                model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
                return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def preprocess_sample(self, sample):
        """
        Preprocess a single sample for EEGNetv4.

        Args:
            sample: Raw EEG sample of shape (n_channels,)

        Returns:
            Preprocessed sample
        """
        # Convert to microvolts (if not already)
        sample = sample * 1e6

        # No additional preprocessing needed for EEGNetv4
        # The model handles normalization internally through batch normalization
        return sample

    def add_sample(self, sample):
        """
        Add a new sample to the buffer.

        Args:
            sample: Raw EEG sample of shape (n_channels,)
        """
        if sample.shape[0] != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {sample.shape[0]}")

        # Preprocess the sample
        processed_sample = self.preprocess_sample(sample)

        # Add to buffer
        self.buffer.append(processed_sample)
        self.sample_idx += 1

    def add_trial_marker(self, label):
        """
        Add a trial marker indicating the start of a new trial.

        Args:
            label: Trial label (e.g., 'feet', 'left_hand', etc.)
        """
        self.trial_starts.append((self.sample_idx, label))

    def get_trial_window(self, trial_start_idx):
        """
        Get window data aligned with trial start.

        Args:
            trial_start_idx: Sample index where trial started

        Returns:
            window_data: Window data if available, None otherwise
        """
        window_end_idx = trial_start_idx + self.window_size

        # Check if we have enough data
        if window_end_idx > len(self.buffer):
            return None

        # Extract window from trial start
        window_data = np.array(self.buffer[trial_start_idx:window_end_idx])  # shape: (window_size, n_channels)
        window_data = window_data.T  # shape: (n_channels, window_size)
        return window_data

    def predict_trials(self):
        """
        Make predictions for all completed trials.

        Returns:
            list: List of prediction results for completed trials, or empty list if none
        """
        results = []

        if len(self.trial_starts) == 0:
            return results

        for trial_start_idx, trial_label in self.trial_starts:
            # Skip if we've already processed this trial
            if trial_start_idx in self.processed_trials:
                continue

            # Check if we have enough data for this trial
            window_data = self.get_trial_window(trial_start_idx)
            if window_data is None:
                continue  # Not enough data yet

            # Mark this trial as processed
            self.processed_trials.add(trial_start_idx)

            # Convert to tensor
            x = torch.FloatTensor(window_data).unsqueeze(0)  # (1, n_chans, n_times)
            x = x.to(self.device)

            # Make prediction
            with torch.no_grad():
                logits = self.model(x)
                probabilities = F.softmax(logits, dim=1)

                # Get predicted class and confidence
                confidence, predicted_class = torch.max(probabilities, 1)

                # Convert to numpy
                predicted_class = predicted_class.cpu().numpy()[0]
                confidence = confidence.cpu().numpy()[0]
                probabilities = probabilities.cpu().numpy()[0]

            # Store statistics
            self.prediction_history.append(predicted_class)
            self.confidence_history.append(confidence)

            # Calculate window end for display
            window_end_idx = trial_start_idx + self.window_size

            result = {
                'trial_start': trial_start_idx,
                'trial_end': window_end_idx,
                'true_label': trial_label,
                'class': predicted_class,
                'class_label': self.class_labels[predicted_class],
                'confidence': confidence,
                'probabilities': probabilities
            }
            results.append(result)

        return results

    def predict_sliding(self):
        """
        Make a prediction using sliding window (for backward compatibility).
        This is NOT recommended for accuracy evaluation due to timing issues.

        Returns:
            dict: Prediction results with 'class', 'confidence', 'probabilities'
                 or None if not enough data
        """
        if len(self.buffer) < self.window_size:
            return None

        # Get last window_size samples (sliding window)
        window = np.array(self.buffer[-self.window_size:])  # shape: (window_size, n_chans)
        window = window.T  # shape: (n_chans, window_size)

        # Convert to tensor
        x = torch.FloatTensor(window).unsqueeze(0)  # (1, n_chans, n_times)
        x = x.to(self.device)

        # Make prediction
        with torch.no_grad():
            logits = self.model(x)
            probabilities = F.softmax(logits, dim=1)

            # Get predicted class and confidence
            confidence, predicted_class = torch.max(probabilities, 1)

            # Convert to numpy
            predicted_class = predicted_class.cpu().numpy()[0]
            confidence = confidence.cpu().numpy()[0]
            probabilities = probabilities.cpu().numpy()[0]

        # Store statistics
        self.prediction_history.append(predicted_class)
        self.confidence_history.append(confidence)

        return {
            'class': predicted_class,
            'class_label': self.class_labels[predicted_class],
            'confidence': confidence,
            'probabilities': probabilities
        }

    def get_statistics(self):
        """Get prediction statistics."""
        if not self.prediction_history:
            return {}

        return {
            'recent_accuracy': np.mean(self.prediction_history),
            'avg_confidence': np.mean(self.confidence_history),
            'prediction_count': len(self.prediction_history),
            'total_trials': len(self.trial_starts),
            'processed_trials': len(self.processed_trials)
        }

    def reset(self):
        """Reset the classifier buffer and statistics."""
        self.buffer.clear()
        self.trial_starts.clear()
        self.processed_trials.clear()
        self.sample_idx = 0
        self.prediction_history.clear()
        self.confidence_history.clear()

# Example usage and testing
if __name__ == "__main__":
    import os

    # Path to the trained model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    model_path = os.path.join(project_root, 'models', 'eegnetv4_subj3_model_250_full.pth')

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train the EEGNetv4 model first using train_eegnet.py")
    else:
        # Initialize classifier
        classifier = EEGNetv4RealtimeClassifierFixed(model_path)

        # Simulate some test data with trial markers
        print("Testing EEGNetv4 classifier with trial-aligned windows...")

        # Generate random test data
        test_data = np.random.randn(16, 1000) * 10  # 16 channels, 1000 samples

        # Add samples and simulate trial markers
        for i in range(1000):
            sample = test_data[:, i]
            classifier.add_sample(sample)

            # Simulate trial markers every 300 samples
            if i % 300 == 0 and i > 0:
                trial_labels = ['feet', 'left_hand', 'right_hand', 'tongue']
                trial_label = trial_labels[(i // 300) % 4]
                classifier.add_trial_marker(trial_label)
                print(f"Trial marker '{trial_label}' at sample {i}")

            # Try to predict trials
            results = classifier.predict_trials()
            if results:
                for result in results:
                    print(f"Trial {result['trial_start']}-{result['trial_end']} | True: {result['true_label']} | Pred: {result['class_label']} | Conf: {result['confidence']:.3f}")

        print("✅ EEGNetv4 classifier test completed!")