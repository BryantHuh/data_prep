# -*- coding: utf-8 -*-
"""
Real-time BCI classifier class for LSL streaming, preprocessing, and model inference.
Can be used for both console output and GUI visualization.
"""

import os
import time
import numpy as np
import torch
from torch.nn.functional import softmax
from pylsl import StreamInlet, resolve_byprop
from braindecode.models import ShallowFBCSPNet
import pandas as pd
import mne
from online_standardizer import OnlineExponentialStandardizer

class RealtimeBCIClassifier:
    def __init__(self, subject_id=3, model_path=None, output_dir=None):
        """
        Initialize the real-time BCI classifier.

        Args:
            subject_id: Subject ID for the model
            model_path: Path to the trained model
            output_dir: Directory to save results
        """
        self.subject_id = subject_id
        self.sfreq = 125
        self.input_window_samples = 250
        self.included_channels = [
            'C3', 'C4', 'Cz',
            'FC1', 'FC2', 'FCz',
            'CP1', 'CP2', 'CPz',
            'P1', 'P2', 'Pz',
            'C1', 'C2',
            'CP3', 'CP4'
        ]

        # Set up paths
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
            model_path = os.path.join(project_root, 'models', 'subj3_model_250_full.pth')

        if output_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = current_dir

        self.model_path = model_path
        self.output_dir = output_dir

        # Preprocessing parameters
        self.filter_low = 4
        self.filter_high = 38

        # Label mapping
        self.label_dict = {'feet': 0, 'left_hand': 1, 'right_hand': 2, 'tongue': 3}
        self.inv_label_dict = {v: k for k, v in self.label_dict.items()}

        # Initialize components
        self._load_model()
        self._setup_lsl()
        self._setup_preprocessing()

        # State variables
        self.running = False
        self.results = []
        self.callback = None  # For GUI callbacks

    def _load_model(self):
        """Load the trained model."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        torch.serialization.add_safe_globals([ShallowFBCSPNet])
        self.model = torch.load(self.model_path, map_location=device, weights_only=False)
        self.model.to(device).eval()
        self.device = device

        print(f"Model loaded from {self.model_path}")

    def _setup_lsl(self):
        """Set up LSL streams."""
        print("Looking for EEG and marker streams...")
        eeg_streams = resolve_byprop('type', 'EEG', timeout=60)
        marker_streams = resolve_byprop('type', 'Markers', timeout=60)

        self.eeg_inlet = StreamInlet(eeg_streams[0])
        self.marker_inlet = StreamInlet(marker_streams[0])

        print("Connected to LSL streams.")

    def _setup_preprocessing(self):
        """Set up preprocessing components."""
        # Use custom OnlineExponentialStandardizer for real-time streaming
        self.standardizer = OnlineExponentialStandardizer(
            n_channels=len(self.included_channels),
            factor_new=1e-3,
            init_block_size=250
        )
        self.buffer = []
        self.marker_buffer = []
        self.marker_times = []
        self.current_label = None
        self.sample_idx = 0
        self.waiting_for_first_marker = True
        self.first_marker_sample_idx = None

    def preprocess_window(self, window_data):
        """Apply preprocessing to a window of data."""
        # window_data is already scaled to microvolts from sample processing

        # Bandpass filter (4-38 Hz) - use MNE filter to match offline
        window_data = mne.filter.filter_data(
            window_data,
            sfreq=125,
            l_freq=self.filter_low,
            h_freq=self.filter_high,
            method='iir',
            picks=None,
            verbose=False
        )

        return window_data

    def set_callback(self, callback_func):
        """Set a callback function for real-time updates (for GUI)."""
        self.callback = callback_func

    def process_sample(self):
        """Process one sample from the LSL stream."""
        # Pull EEG sample
        sample, ts = self.eeg_inlet.pull_sample(timeout=0.1)
        if sample is not None:
            # Convert sample to numpy array and apply scaling
            sample_array = np.array(sample)
            scaled_sample = sample_array * 1e6  # Scale to microvolts
            self.buffer.append(scaled_sample)
            self.sample_idx += 1

            # Feed sample to standardizer for initialization
            if not self.waiting_for_first_marker:
                self.standardizer.feed_sample(scaled_sample)

        # Pull marker
        marker, mts = self.marker_inlet.pull_sample(timeout=0.0)
        if marker is not None and marker[0] and marker[0] != 'start':
            self.marker_buffer.append(marker[0])
            self.marker_times.append(self.sample_idx)
            self.current_label = marker[0]
            if self.waiting_for_first_marker:
                self.waiting_for_first_marker = False
                self.first_marker_sample_idx = self.sample_idx
                print(f"First real marker '{self.current_label}' received at sample {self.sample_idx}. Starting standardizer initialization.")

        # Run inference if ready
        if not self.waiting_for_first_marker and self.standardizer.initialized and len(self.buffer) >= self.input_window_samples:
            return self._run_inference()

        return None

    def _run_inference(self):
        """Run model inference on the current window."""
        # Get window from buffer
        window_data = np.array(self.buffer[-self.input_window_samples:]).T  # shape: (n_channels, window)

        # Apply preprocessing
        window_data = self.preprocess_window(window_data)

        # Apply running standardization
        window_data = self.standardizer.standardize_window(window_data)

        # Model inference
        x_tensor = torch.tensor(window_data, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x_tensor)
            if logits.ndim == 3:
                logits = logits.mean(dim=2)
            # Apply softmax to get probabilities (model might return logits)
            probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
            pred = int(np.argmax(probs))
            conf = float(np.max(probs))

        # Get true label
        true_label = self.current_label if self.current_label is not None else 'unknown'

        # Create result
        result = {
            'sample_idx': self.sample_idx,
            'true_label': true_label,
            'pred_label': self.inv_label_dict.get(pred, pred),
            'pred_class': pred,
            'confidence': conf,
            'conf_0': probs[0],
            'conf_1': probs[1],
            'conf_2': probs[2],
            'conf_3': probs[3],
        }

        # Store result
        self.results.append(result)

        # Call callback if set (for GUI)
        if self.callback:
            self.callback(result)

        return result

    def run_console(self):
        """Run the classifier with console output."""
        print("Starting real-time classification. Press Ctrl+C to stop...")
        self.running = True

        try:
            while self.running:
                result = self.process_sample()
                if result:
                    print(f"Window ending at sample {result['sample_idx']:5d} | "
                          f"True: {result['true_label']:10s} | "
                          f"Pred: {result['pred_label']:10s} | "
                          f"Conf: {result['confidence']:.2f}")
                time.sleep(0.001)

        except KeyboardInterrupt:
            print("Stopped by user.")
            self.running = False

        finally:
            self._save_results()

    def _save_results(self):
        """Save results to CSV."""
        if self.results:
            df = pd.DataFrame(self.results)
            out_path = os.path.join(self.output_dir, f'lsl_model_results_{self.subject_id}.csv')
            df.to_csv(out_path, index=False)
            print(f"Saved results to {out_path}")

    def get_accuracy(self):
        """Calculate current accuracy (excluding 'unknown' labels)."""
        if not self.results:
            return 0.0

        df = pd.DataFrame(self.results)
        df_valid = df[df['true_label'] != 'unknown']

        if len(df_valid) == 0:
            return 0.0

        correct = (df_valid['true_label'] == df_valid['pred_label']).sum()
        return correct / len(df_valid)

    def get_confusion_matrix(self):
        """Get confusion matrix for current results."""
        if not self.results:
            return None

        df = pd.DataFrame(self.results)
        df_valid = df[df['true_label'] != 'unknown']

        if len(df_valid) == 0:
            return None

        from sklearn.metrics import confusion_matrix
        labels = ['feet', 'left_hand', 'right_hand', 'tongue']
        cm = confusion_matrix(df_valid['true_label'], df_valid['pred_label'], labels=labels)
        return cm, labels

    def stop(self):
        """Stop the classifier."""
        self.running = False