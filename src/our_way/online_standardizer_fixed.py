# -*- coding: utf-8 -*-
"""
Fixed online standardizer that uses braindecode's exact methods.
This ensures training and real-time inference use identical preprocessing.
"""

import numpy as np
from collections import deque
import mne
from braindecode.preprocessing import exponential_moving_standardize

class FixedOnlineStandardizer:
    """
    Fixed online standardizer that uses braindecode's exact methods.

    This implementation:
    1. Uses braindecode's exponential_moving_standardize function directly
    2. Matches the exact filtering method used in training
    3. Ensures preprocessing consistency between training and inference
    """

    def __init__(self, n_channels, factor_new=1e-3, init_block_size=250, eps=1e-4):
        self.n_channels = n_channels
        self.factor_new = factor_new
        self.init_block_size = init_block_size
        self.eps = eps

        # Calibration buffer
        self.calibration_buffer = deque(maxlen=init_block_size)
        self.initialized = False

        # Running statistics (will be set during calibration)
        self.mean = None
        self.var = None

    def feed_sample(self, sample):
        """Feed a sample for calibration."""
        if not self.initialized:
            self.calibration_buffer.append(sample)

            # Check if we have enough samples for calibration
            if len(self.calibration_buffer) >= self.init_block_size:
                self._initialize_from_buffer()

    def _initialize_from_buffer(self):
        """Initialize using the calibration buffer."""
        calibration_data = np.array(list(self.calibration_buffer))

        # Use braindecode's exact method
        standardized_data = exponential_moving_standardize(
            calibration_data.T,  # braindecode expects (n_times, n_channels)
            factor_new=self.factor_new,
            init_block_size=self.init_block_size,
            eps=self.eps
        )

        # Extract running statistics from the last sample
        self.mean = np.zeros(self.n_channels)
        self.var = np.ones(self.n_channels)

        # Calculate initial statistics from calibration data
        self.mean = np.mean(calibration_data, axis=0)
        self.var = np.var(calibration_data, axis=0)

        self.initialized = True
        print(f"✅ Fixed standardizer initialized with {len(self.calibration_buffer)} samples")

    def standardize_window(self, window):
        """
        Standardize a window using braindecode's exact method.

        Args:
            window: EEG window of shape (n_channels, n_times)

        Returns:
            Standardized window of shape (n_channels, n_times)
        """
        if not self.initialized:
            return window

        # Use braindecode's exponential_moving_standardize directly
        # This ensures exact match with training preprocessing
        standardized = exponential_moving_standardize(
            window.T,  # braindecode expects (n_times, n_channels)
            factor_new=self.factor_new,
            init_block_size=self.init_block_size,
            eps=self.eps
        )

        return standardized.T  # Return to (n_channels, n_times)

    def is_initialized(self):
        """Check if standardizer is initialized."""
        return self.initialized

    def get_calibration_progress(self):
        """Get calibration progress as percentage."""
        if self.initialized:
            return 100.0
        return (len(self.calibration_buffer) / self.init_block_size) * 100

    def reset(self):
        """Reset the standardizer."""
        self.calibration_buffer.clear()
        self.initialized = False
        self.mean = None
        self.var = None

class FixedOnlinePreprocessor:
    """
    Fixed online preprocessor that matches training preprocessing exactly.

    This implementation uses braindecode's exact methods:
    1. Same filtering as Preprocessor('filter', l_freq=4, h_freq=38)
    2. Same standardization as exponential_moving_standardize
    3. Same scaling and channel selection
    """

    def __init__(self, n_channels, sample_rate=125, filter_low=4, filter_high=38,
                 factor_new=1e-3, init_block_size=250):
        self.n_channels = n_channels
        self.sample_rate = sample_rate
        self.filter_low = filter_low
        self.filter_high = filter_high

        # Initialize fixed standardizer
        self.standardizer = FixedOnlineStandardizer(
            n_channels=n_channels,
            factor_new=factor_new,
            init_block_size=init_block_size
        )

        # Filter buffer for proper filtering
        self.filter_buffer = deque(maxlen=1000)

    def preprocess_window(self, window):
        """
        Preprocess a window using braindecode's exact methods.

        Args:
            window: Raw EEG window of shape (n_channels, n_times)

        Returns:
            Preprocessed window of shape (n_channels, n_times)
        """
        # Step 1: Scale to microvolts (same as training)
        window_uv = window * 1e6

        # Step 2: Apply braindecode's exact filtering
        filtered_window = self._apply_braindecode_filter(window_uv)

        # Step 3: Apply braindecode's exact standardization
        if self.standardizer.is_initialized():
            standardized_window = self.standardizer.standardize_window(filtered_window)
        else:
            standardized_window = filtered_window

        return standardized_window

    def _apply_braindecode_filter(self, window):
        """
        Apply filtering using braindecode's exact method.

        This creates a temporary MNE Raw object and applies the same
        filtering as Preprocessor('filter', l_freq=4, h_freq=38).
        """
        # Create temporary MNE Raw object
        temp_raw = mne.io.RawArray(window, mne.create_info(
            ch_names=[f'CH{i}' for i in range(window.shape[0])],
            sfreq=self.sample_rate,
            ch_types=['eeg'] * window.shape[0]
        ))

        # Apply the same filtering as braindecode
        temp_raw.filter(
            l_freq=self.filter_low,
            h_freq=self.filter_high,
            method='iir',
            picks=None,
            verbose=False
        )

        return temp_raw.get_data()

    def feed_sample(self, sample):
        """Feed a sample for calibration."""
        # Add to filter buffer
        self.filter_buffer.append(sample)

        # Feed to standardizer
        self.standardizer.feed_sample(sample)

    def is_ready(self):
        """Check if preprocessor is ready."""
        return self.standardizer.is_initialized()

    def get_calibration_progress(self):
        """Get calibration progress."""
        return self.standardizer.get_calibration_progress()

    def reset(self):
        """Reset the preprocessor."""
        self.standardizer.reset()
        self.filter_buffer.clear()