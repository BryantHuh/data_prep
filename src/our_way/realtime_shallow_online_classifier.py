# -*- coding: utf-8 -*-
"""
Real-time BCI classifier with online-compatible standardization.
This implementation ensures that training and inference use identical preprocessing
by using the same ExponentialMovingStandardize algorithm with apply_on_array=False.
"""

import numpy as np
import torch
from collections import deque
import time
from braindecode.models import ShallowFBCSPNet
from online_standardizer import OnlinePreprocessor
import mne

class RealtimeBCIClassifierOnline:
    """
    Real-time BCI classifier with online-compatible standardization.

    This classifier uses the same preprocessing pipeline as training:
    1. Scale to microvolts (V -> μV)
    2. Bandpass filtering (4-38 Hz)
    3. Online exponential moving standardization

    Args:
        model_path: Path to the trained model
        window_size: Number of samples in the input window (default: 250)
        sample_rate: Sampling rate in Hz (default: 125)
        channels: List of channel names (default: 16 OpenBCI channels)
        device: Device to run inference on ('cpu' or 'cuda')
        filter_low: Low frequency cutoff (default: 4 Hz)
        filter_high: High frequency cutoff (default: 38 Hz)
        factor_new: Standardization factor (default: 1e-3)
        init_block_size: Calibration samples (default: 1000)
    """

    def __init__(self, model_path, window_size=250, sample_rate=125,
                 channels=None, device='cpu', filter_low=4, filter_high=38,
                 factor_new=1e-3, init_block_size=250):

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

        # Initialize online preprocessor
        self.preprocessor = OnlinePreprocessor(
            n_channels=self.n_channels,
            sample_rate=self.sample_rate,
            filter_low=filter_low,
            filter_high=filter_high,
            factor_new=factor_new,
            init_block_size=init_block_size
        )

        # Initialize data buffer
        self.buffer = deque(maxlen=window_size)

        # Class labels
        self.class_labels = ['feet', 'left_hand', 'right_hand', 'tongue']

        # Statistics for monitoring
        self.prediction_history = deque(maxlen=100)
        self.confidence_history = deque(maxlen=100)
        self.sample_count = 0
        self.calibration_complete = False

    def _load_model(self, model_path):
        """Load the trained ShallowFBCSPNet model."""
        try:
            # Try to load the full model first
            model = torch.load(model_path, map_location=self.device, weights_only=False)
            if isinstance(model, ShallowFBCSPNet):
                return model
            else:
                # Load state dict and create model
                model = ShallowFBCSPNet(
                    n_chans=self.n_channels,
                    n_outputs=self.n_classes,
                    input_window_samples=self.window_size,
                    final_conv_length=30  # Fixed value instead of 'auto'
                )
                model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
                return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def add_sample(self, sample):
        """
        Add a new sample to the buffer and preprocess it.

        Args:
            sample: Raw EEG sample of shape (n_channels,)
        """
        if sample.shape[0] != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {sample.shape[0]}")

        # Preprocess the sample using online preprocessing
        processed_sample = self.preprocessor.preprocess_sample(sample)

        # Add to buffer
        self.buffer.append(processed_sample)
        self.sample_count += 1

        # Check if calibration is complete
        if not self.calibration_complete and self.preprocessor.is_ready():
            self.calibration_complete = True
            print(f"✅ Calibration complete after {self.sample_count} samples")

    def predict(self):
        """
        Make a prediction if enough data is available and calibration is complete.

        Returns:
            dict: Prediction results with 'class', 'confidence', 'probabilities'
                 or None if not enough data or calibration incomplete
        """
        if len(self.buffer) < self.window_size:
            return None

        if not self.calibration_complete:
            return None

        # Convert buffer to numpy array
        window = np.array(list(self.buffer))  # shape: (window_size, n_chans)
        window = window.T  # shape: (n_chans, window_size)

        # Convert to tensor
        x = torch.FloatTensor(window).unsqueeze(0)  # (1, n_chans, n_times)
        x = x.to(self.device)

        # Make prediction
        with torch.no_grad():
            output = self.model(x)
            if output.ndim == 3:
                output = output.mean(dim=2)  # Average over time dimension for cropped decoding

            # Apply softmax to get probabilities (model might return logits)
            probabilities = torch.softmax(output, dim=1)

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

    def get_calibration_progress(self):
        """Get calibration progress as a percentage."""
        return self.preprocessor.get_calibration_progress()

    def is_ready(self):
        """Check if the classifier is ready for inference."""
        return self.calibration_complete and len(self.buffer) >= self.window_size

    def get_statistics(self):
        """Get prediction and calibration statistics."""
        stats = {
            'sample_count': self.sample_count,
            'calibration_complete': self.calibration_complete,
            'calibration_progress': self.get_calibration_progress(),
            'buffer_size': len(self.buffer),
            'ready_for_inference': self.is_ready()
        }

        if self.prediction_history:
            stats.update({
                'recent_accuracy': np.mean(self.prediction_history),
                'avg_confidence': np.mean(self.confidence_history),
                'prediction_count': len(self.prediction_history)
            })

        return stats

    def reset(self):
        """Reset the classifier buffer, statistics, and preprocessor."""
        self.buffer.clear()
        self.prediction_history.clear()
        self.confidence_history.clear()
        self.preprocessor.reset()
        self.sample_count = 0
        self.calibration_complete = False

    def get_preprocessor_statistics(self):
        """Get detailed preprocessor statistics."""
        return self.preprocessor.standardizer.get_statistics()


class RealtimeBCIClassifierOnlineWithFiltering(RealtimeBCIClassifierOnline):
    """
    Enhanced real-time BCI classifier with proper real-time filtering.

    This version includes a more sophisticated filtering implementation
    using scipy's lfilter for real-time bandpass filtering.
    """

    def __init__(self, model_path, window_size=250, sample_rate=125,
                 channels=None, device='cpu', filter_low=4, filter_high=38,
                 factor_new=1e-3, init_block_size=250):

        super().__init__(model_path, window_size, sample_rate, channels, device,
                        filter_low, filter_high, factor_new, init_block_size)

        # Initialize filter coefficients
        self._setup_filter()

    def _setup_filter(self):
        """Set up bandpass filter coefficients."""
        try:
            from scipy import signal

            # Design bandpass filter
            nyquist = self.sample_rate / 2
            low_norm = self.preprocessor.filter_low / nyquist
            high_norm = self.preprocessor.filter_high / nyquist

            # Butterworth bandpass filter
            self.filter_b, self.filter_a = signal.butter(
                N=4,  # Filter order
                Wn=[low_norm, high_norm],
                btype='band',
                analog=False
            )

            # Initialize filter state
            self.filter_state = None

            print(f"✅ Bandpass filter initialized: {self.preprocessor.filter_low}-{self.preprocessor.filter_high} Hz")

        except ImportError:
            print("⚠️  scipy not available, using simplified filtering")
            self.filter_b = None
            self.filter_a = None
            self.filter_state = None

    def _apply_real_time_filter(self, sample):
        """Apply real-time bandpass filter to a single sample."""
        if self.filter_b is None or self.filter_a is None:
            return sample

        try:
            from scipy import signal

            # Apply filter with state preservation
            if self.filter_state is None:
                filtered_sample, self.filter_state = signal.lfilter(
                    self.filter_b, self.filter_a,
                    sample.reshape(1, -1)
                )
            else:
                filtered_sample, self.filter_state = signal.lfilter(
                    self.filter_b, self.filter_a,
                    sample.reshape(1, -1),
                    zi=self.filter_state
                )

            return filtered_sample.squeeze()

        except ImportError:
            return sample

    def add_sample(self, sample):
        """
        Add a new sample with real-time filtering.

        Args:
            sample: Raw EEG sample of shape (n_channels,)
        """
        if sample.shape[0] != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {sample.shape[0]}")

        # Step 1: Scale to microvolts
        sample_uv = sample * 1e6

        # Step 2: Apply real-time filtering
        filtered_sample = self._apply_real_time_filter(sample_uv)

        # Step 3: Feed to standardizer for calibration
        self.preprocessor.standardizer.feed_sample(filtered_sample)

        # Step 4: Apply standardization (if initialized)
        if self.preprocessor.standardizer.is_initialized():
            standardized_sample = self.preprocessor.standardizer.standardize_sample(filtered_sample)
        else:
            standardized_sample = filtered_sample

        # Add to buffer
        self.buffer.append(standardized_sample)
        self.sample_count += 1

        # Check if calibration is complete
        if not self.calibration_complete and self.preprocessor.is_ready():
            self.calibration_complete = True
            print(f"✅ Calibration complete after {self.sample_count} samples")


# Example usage and testing
if __name__ == "__main__":
    import os

    # Path to the trained model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    model_path = os.path.join(project_root, 'models', 'subj3_model_250_full.pth')

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train the model first using the training scripts")
    else:
        # Initialize classifier
        classifier = RealtimeBCIClassifierOnline(model_path)

        # Simulate some test data
        print("Testing online classifier with simulated data...")
        print("Calibrating standardizer...")

        # Generate random test data
        test_data = np.random.randn(16, 1500) * 10  # 16 channels, 1500 samples

        # Add samples one by one
        for i in range(1500):
            sample = test_data[:, i]
            classifier.add_sample(sample)

            # Check calibration progress
            if i % 100 == 0:
                progress = classifier.get_calibration_progress()
                print(f"Calibration progress: {progress:.1f}%")

            # Try to predict once calibration is complete
            if classifier.is_ready():
                result = classifier.predict()
                if result is not None:
                    print(f"✅ First prediction: {result['class_label']} (confidence: {result['confidence']:.3f})")
                    break

        # Show final statistics
        stats = classifier.get_statistics()
        print(f"\nFinal statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        print("✅ Online classifier test completed!")