# -*- coding: utf-8 -*-
"""
Interactive EEGNetv4 real-time classifier with continuous sliding window predictions.
This version provides fluid, interactive predictions without waiting for trial markers.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
import time
from braindecode.models import EEGNetv4
from online_standardizer_fixed import FixedOnlinePreprocessor

class EEGNetv4InteractiveClassifier:
    """
    Interactive EEGNetv4 classifier for continuous real-time predictions.

    This version provides:
    - Continuous sliding window predictions
    - No dependency on trial markers
    - Smooth, interactive experience
    - Real-time confidence updates
    """

    def __init__(self, model_path, window_size=250, sample_rate=125,
                 channels=None, device='cpu', prediction_interval=50):
        """
        Initialize the interactive classifier.

        Args:
            model_path: Path to the trained EEGNetv4 model
            window_size: Size of the prediction window in samples
            sample_rate: EEG sampling rate in Hz
            channels: List of channel names (optional)
            device: Device to run inference on ('cpu' or 'cuda')
            prediction_interval: How often to make predictions (in samples)
        """
        self.window_size = window_size
        self.sample_rate = sample_rate
        self.channels = channels or [
            'C3', 'C4', 'Cz', 'FC1', 'FC2', 'FCz', 'CP1', 'CP2', 'CPz',
            'P1', 'P2', 'Pz', 'C1', 'C2', 'CP3', 'CP4'
        ]
        self.device = device
        self.prediction_interval = prediction_interval

        # Data buffer
        self.buffer = deque(maxlen=window_size * 2)  # Keep extra samples for smooth operation

        # Preprocessor
        self.preprocessor = FixedOnlinePreprocessor(
            n_channels=len(self.channels),
            sample_rate=sample_rate,
            filter_low=4,
            filter_high=38,
            factor_new=1e-3,
            init_block_size=1000
        )

        # Model
        self.model = self._load_model(model_path)

        # Statistics
        self.sample_count = 0
        self.prediction_count = 0
        self.last_prediction_time = time.time()
        self.prediction_history = deque(maxlen=100)
        self.confidence_history = deque(maxlen=100)

        # Class names
        self.class_names = ['feet', 'left_hand', 'right_hand', 'tongue']

        print(f"✅ Interactive EEGNetv4 classifier initialized")
        print(f"   - Window size: {window_size} samples ({window_size/sample_rate:.1f}s)")
        print(f"   - Prediction interval: {prediction_interval} samples")
        print(f"   - Device: {device}")
        print(f"   - Channels: {len(self.channels)}")

    def _load_model(self, model_path):
        """Load the EEGNetv4 model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        device = torch.device(self.device)
        torch.serialization.add_safe_globals([EEGNetv4])
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.to(device).eval()

        print(f"✅ Model loaded from {model_path}")
        return model

    def add_sample(self, sample):
        """
        Add a new EEG sample to the buffer.

        Args:
            sample: EEG sample as numpy array of shape (n_channels,)
        """
        # Ensure sample has correct shape
        if len(sample) != len(self.channels):
            raise ValueError(f"Expected {len(self.channels)} channels, got {len(sample)}")

        # Scale to microvolts (same as training)
        sample_scaled = sample * 1e6

        # Add to buffer
        self.buffer.append(sample_scaled)
        self.sample_count += 1

        # Feed to preprocessor for calibration
        self.preprocessor.feed_sample(sample_scaled)

    def predict(self):
        """
        Make a prediction if enough data is available and prediction interval is met.

        Returns:
            Prediction result dict or None if not ready
        """
        # Check if we have enough data
        if len(self.buffer) < self.window_size:
            return None

        # Check if it's time for a new prediction
        if self.sample_count % self.prediction_interval != 0:
            return None

        # Check if preprocessor is ready
        if not self.preprocessor.is_ready():
            return None

        try:
            # Get the latest window
            window_data = np.array(list(self.buffer)[-self.window_size:]).T  # (n_channels, window_size)

            # Preprocess the window
            preprocessed_window = self.preprocessor.preprocess_window(window_data)

            # Convert to tensor
            x_tensor = torch.tensor(preprocessed_window, dtype=torch.float32, device=self.device).unsqueeze(0)

            # Run inference
            with torch.no_grad():
                logits = self.model(x_tensor)

                # Handle cropped decoding if needed
                if logits.ndim == 3:
                    logits = logits.mean(dim=2)

                # Get probabilities
                probabilities = F.softmax(logits, dim=1)

                # Get prediction and confidence
                confidence, predicted_class = torch.max(probabilities, 1)
                predicted_class = predicted_class.cpu().numpy()[0]
                confidence = confidence.cpu().numpy()[0]
                probabilities = probabilities.cpu().numpy()[0]

            # Create result
            result = {
                'sample_idx': self.sample_count,
                'timestamp': time.time(),
                'class': predicted_class,
                'class_label': self.class_names[predicted_class],
                'confidence': confidence,
                'probabilities': probabilities,
                'window_start': self.sample_count - self.window_size + 1,
                'window_end': self.sample_count
            }

            # Update statistics
            self.prediction_count += 1
            self.last_prediction_time = time.time()
            self.prediction_history.append(result)
            self.confidence_history.append(confidence)

            return result

        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None

    def get_latest_prediction(self):
        """Get the most recent prediction result."""
        if self.prediction_history:
            return self.prediction_history[-1]
        return None

    def get_prediction_trend(self, window_size=10):
        """
        Get the trend of recent predictions.

        Args:
            window_size: Number of recent predictions to analyze

        Returns:
            Dict with trend information
        """
        if len(self.prediction_history) < window_size:
            return None

        recent_predictions = list(self.prediction_history)[-window_size:]

        # Get most common prediction
        class_counts = {}
        for pred in recent_predictions:
            class_label = pred['class_label']
            class_counts[class_label] = class_counts.get(class_label, 0) + 1

        most_common_class = max(class_counts.items(), key=lambda x: x[1])

        # Calculate average confidence
        avg_confidence = np.mean([pred['confidence'] for pred in recent_predictions])

        # Calculate stability (how consistent predictions are)
        stability = most_common_class[1] / window_size

        return {
            'trend_class': most_common_class[0],
            'trend_confidence': most_common_class[1] / window_size,
            'avg_confidence': avg_confidence,
            'stability': stability,
            'window_size': window_size
        }

    def get_statistics(self):
        """Get classifier statistics."""
        current_time = time.time()

        stats = {
            'sample_count': self.sample_count,
            'prediction_count': self.prediction_count,
            'buffer_size': len(self.buffer),
            'preprocessor_ready': self.preprocessor.is_ready(),
            'calibration_progress': self.preprocessor.get_calibration_progress(),
            'last_prediction_time': self.last_prediction_time,
            'time_since_last_prediction': current_time - self.last_prediction_time if self.last_prediction_time else None,
            'prediction_rate': self.prediction_count / (current_time - self.last_prediction_time + 1e-6) if self.last_prediction_time else 0,
            'avg_confidence': np.mean(list(self.confidence_history)) if self.confidence_history else 0,
            'device': self.device
        }

        # Add trend information
        trend = self.get_prediction_trend()
        if trend:
            stats.update(trend)

        return stats

    def reset(self):
        """Reset the classifier."""
        self.buffer.clear()
        self.preprocessor.reset()
        self.sample_count = 0
        self.prediction_count = 0
        self.last_prediction_time = time.time()
        self.prediction_history.clear()
        self.confidence_history.clear()
        print("✅ Classifier reset")

    def is_ready(self):
        """Check if the classifier is ready to make predictions."""
        return (len(self.buffer) >= self.window_size and
                self.preprocessor.is_ready())

    def get_calibration_progress(self):
        """Get calibration progress as percentage."""
        return self.preprocessor.get_calibration_progress()

    def get_prediction_frequency(self):
        """Get the current prediction frequency in Hz."""
        if self.prediction_count < 2:
            return 0

        time_span = time.time() - self.last_prediction_time
        if time_span <= 0:
            return 0

        return self.prediction_count / time_span