# -*- coding: utf-8 -*-
"""
Real-time EEGNetv4 classifier for BCI applications.
Designed for minimal preprocessing and fast inference using Braindecode's EEGNetv4.
"""

import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
import time
from braindecode.models import EEGNetv4

class EEGNetv4RealtimeClassifier:
    """
    Real-time EEGNetv4 classifier for BCI applications.

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

        # Initialize data buffer
        self.buffer = deque(maxlen=window_size)

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

    def predict(self):
        """
        Make a prediction if enough data is available.

        Returns:
            dict: Prediction results with 'class', 'confidence', 'probabilities'
                 or None if not enough data
        """
        if len(self.buffer) < self.window_size:
            return None

        # Convert buffer to numpy array
        window = np.array(list(self.buffer))  # shape: (window_size, n_chans) or (n_chans, window_size)
        if window.shape == (self.window_size, self.n_channels):
            window = window.T  # shape: (n_chans, window_size)
        elif window.shape == (self.n_channels, self.window_size):
            pass  # already correct
        else:
            raise ValueError(f"Unexpected window shape: {window.shape}, expected ({self.window_size}, {self.n_channels}) or ({self.n_channels}, {self.window_size})")

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
            'prediction_count': len(self.prediction_history)
        }

    def reset(self):
        """Reset the classifier buffer and statistics."""
        self.buffer.clear()
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
        classifier = EEGNetv4RealtimeClassifier(model_path)

        # Simulate some test data
        print("Testing EEGNetv4 classifier with simulated data...")

        # Generate random test data
        test_data = np.random.randn(16, 250) * 10  # 16 channels, 250 samples

        # Add samples one by one
        for i in range(250):
            sample = test_data[:, i]
            classifier.add_sample(sample)

            # Try to predict
            result = classifier.predict()
            if result is not None:
                print(f"Prediction: {result['class_label']} (confidence: {result['confidence']:.3f})")
                break

        print("✅ EEGNetv4 classifier test completed!")