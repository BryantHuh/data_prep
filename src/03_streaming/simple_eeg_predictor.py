#!/usr/bin/env python3
"""
Simple EEG Predictor

Real-time EEG classification without GUI - just predictions.
This script provides a lightweight alternative to the GUI classifier for
applications that only need classification results without visualization.

Features:
- Real-time EEG classification with EEGNetv4 or ShallowFBCSPNet
- LSL stream input and output
- Performance monitoring
- No GUI overhead for faster processing
"""

import os
import sys
import time
import argparse
from pathlib import Path
from collections import deque
import numpy as np
import torch
from pylsl import StreamInlet, resolve_byprop, StreamInfo, StreamOutlet

# Braindecode imports
from braindecode.models import EEGNetv4, ShallowFBCSPNet

# Logging utility
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
from utils.logging_utils import setup_logging

# Setup logger
logger = setup_logging('simple_predictor', log_dir='logs', level='INFO')

class SimpleEEGPredictor:
    """Simple EEG predictor without markers"""

    def __init__(self, model_path, model_type='eegnet', eeg_stream_name='MOABB_EEG_RAW'):
        self.model_path = model_path
        self.model_type = model_type
        self.eeg_stream_name = eeg_stream_name

        # Model parameters
        self.window_size_samples = 250
        self.n_classes = 4
        self.class_names = ['feet', 'left_hand', 'right_hand', 'tongue']

        # LSL components
        self.eeg_inlet = None
        self.prediction_outlet = None

        # Data buffers
        self.eeg_buffer = deque(maxlen=self.window_size_samples)

        # Model
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Performance monitoring
        self.sample_count = 0
        self.prediction_count = 0
        self.start_time = None

        logger.info(f"Initialized simple EEG predictor on {self.device}")

    def load_model(self):
        """Load trained model"""
        try:
            logger.info(f"Loading {self.model_type} model from {self.model_path}")

            # Create model
            if self.model_type == 'eegnet':
                self.model = EEGNetv4(
                    n_chans=16,
                    n_outputs=4,
                    n_times=250,
                    drop_prob=0.25,
                    kernel_length=64,
                )
            elif self.model_type == 'shallow_fbcsp':
                self.model = ShallowFBCSPNet(
                    n_chans=16,
                    n_outputs=4,
                    n_times=250,
                    final_conv_length='auto',
                )
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

            # Load trained weights
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)

            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info(f"Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def setup_lsl_streams(self):
        """Setup LSL streams"""
        try:
            # Connect to EEG stream
            logger.info(f"Resolving EEG stream: {self.eeg_stream_name}")
            eeg_streams = resolve_byprop('name', self.eeg_stream_name, timeout=10)

            if not eeg_streams:
                raise RuntimeError(f"EEG stream '{self.eeg_stream_name}' not found")

            self.eeg_inlet = StreamInlet(eeg_streams[0])
            logger.info(f"Connected to EEG stream: {eeg_streams[0].name()}")

            # Create prediction output stream
            info = StreamInfo(
                'BCI_Predictions',
                'Predictions',
                1,
                0,
                'string',
                'bci_predictions_uid'
            )
            self.prediction_outlet = StreamOutlet(info)
            logger.info("Created prediction output stream")

        except Exception as e:
            logger.error(f"Failed to setup LSL streams: {e}")
            raise

    def preprocess_eeg_window(self, eeg_window):
        """Preprocess EEG window for classification"""
        # Simple preprocessing: normalize
        eeg_window = (eeg_window - np.mean(eeg_window, axis=-1, keepdims=True)) / \
                    (np.std(eeg_window, axis=-1, keepdims=True) + 1e-8)

        # Reshape for model input (batch_size=1, channels, time_points)
        eeg_window = eeg_window.reshape(1, eeg_window.shape[0], eeg_window.shape[1])

        return eeg_window

    def classify_window(self, eeg_window):
        """Classify EEG window"""
        try:
            # Preprocess window
            processed_window = self.preprocess_eeg_window(eeg_window)

            # Convert to tensor
            input_tensor = torch.FloatTensor(processed_window).to(self.device)

            # Predict
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)

            # Get prediction and confidence
            probabilities_np = probabilities.cpu().numpy()[0]
            predicted_class = np.argmax(probabilities_np)
            confidence = np.max(probabilities_np)

            return predicted_class, confidence, probabilities_np

        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return None, None, None

    def process_eeg_stream(self):
        """Main processing loop"""
        logger.info("Starting EEG stream processing")
        self.start_time = time.time()

        while True:
            try:
                # Get EEG sample
                sample, timestamp = self.eeg_inlet.pull_sample(timeout=0.1)

                if sample is not None:
                    # Add to buffer
                    self.eeg_buffer.append(sample)
                    self.sample_count += 1

                    # Check if we have enough samples for classification
                    if len(self.eeg_buffer) >= self.window_size_samples:
                        # Extract window
                        eeg_window = np.array(list(self.eeg_buffer)[-self.window_size_samples:])
                        eeg_window = eeg_window.T  # Transpose to (channels, time_points)

                        # Classify window
                        prediction, confidence, probabilities = self.classify_window(eeg_window)

                        if prediction is not None:
                            # Send prediction result
                            result = self.class_names[prediction]
                            self.prediction_outlet.push_sample([result])
                            self.prediction_count += 1

                            # Log result
                            logger.info(f"Prediction: {result} (confidence: {confidence:.3f})")

                            # Update buffer stride (simple approach)
                            for _ in range(self.window_size_samples // 2):  # 50% overlap
                                if self.eeg_buffer:
                                    self.eeg_buffer.popleft()

                # Small delay
                time.sleep(0.001)

            except KeyboardInterrupt:
                logger.info("Received interrupt signal, stopping...")
                break
            except Exception as e:
                logger.error(f"Error in EEG processing loop: {e}")
                time.sleep(0.1)

        logger.info("EEG stream processing stopped")

    def print_statistics(self):
        """Print processing statistics"""
        if self.start_time:
            elapsed_time = time.time() - self.start_time
            logger.info("=" * 60)
            logger.info("Processing Statistics")
            logger.info("=" * 60)
            logger.info(f"Total samples processed: {self.sample_count}")
            logger.info(f"Total predictions made: {self.prediction_count}")
            logger.info(f"Processing time: {elapsed_time:.2f} seconds")
            logger.info(f"Average samples per second: {self.sample_count/elapsed_time:.1f}")
            logger.info(f"Average predictions per second: {self.prediction_count/elapsed_time:.1f}")

    def start_prediction(self):
        """Start real-time prediction"""
        try:
            # Load model
            self.load_model()

            # Setup LSL streams
            self.setup_lsl_streams()

            # Start processing
            self.process_eeg_stream()

        except Exception as e:
            logger.error(f"Failed to start prediction: {e}")
            raise
        finally:
            self.print_statistics()

    def stop_prediction(self):
        """Stop real-time prediction"""
        # Close LSL streams
        if self.eeg_inlet:
            self.eeg_inlet.close_stream()

        if self.prediction_outlet:
            self.prediction_outlet.close_stream()

        logger.info("Real-time prediction stopped")

def main():
    """Main prediction function"""
    parser = argparse.ArgumentParser(description='Simple EEG Predictor')
    parser.add_argument('--model-path', type=str, default='models/eegnetv4_subj3_model_250.pth',
                       help='Path to trained model')
    parser.add_argument('--model-type', type=str, default='eegnet', choices=['eegnet', 'shallow_fbcsp'],
                       help='Model type')
    parser.add_argument('--eeg-stream', type=str, default='MOABB_EEG_RAW',
                       help='LSL EEG stream name')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Simple EEG Predictor")
    logger.info("=" * 60)

    try:
        # Initialize predictor
        predictor = SimpleEEGPredictor(
            model_path=args.model_path,
            model_type=args.model_type,
            eeg_stream_name=args.eeg_stream
        )

        # Start prediction
        predictor.start_prediction()

    except KeyboardInterrupt:
        logger.info("Received interrupt signal, stopping...")
        predictor.stop_prediction()
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()