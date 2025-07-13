#!/usr/bin/env python3
"""
EEGNet GUI Classifier

Real-time EEG classification GUI using EEGNetv4 with LSL streams.
This is the main real-time classification interface for BCI experiments.

Features:
- Real-time EEG classification with EEGNetv4
- Optional marker stream support for validation
- Dynamic marker visualization in confidence plots
- CSV export functionality for detailed analysis
- Performance monitoring and statistics
- Robust error handling and logging
"""

import os
import sys
import time
import threading
import argparse
from pathlib import Path
from collections import deque
import numpy as np
import torch
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pylsl import StreamInlet, resolve_byprop
import pandas as pd
from datetime import datetime

# Braindecode imports
from braindecode.models import EEGNetv4

# Logging utility
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
from utils.logging_utils import setup_logging

# Setup logger
logger = setup_logging('eegnet_gui', log_dir='logs', level='INFO')

class EEGNetRealtimeClassifier:
    """Real-time EEGNet classifier"""

    def __init__(self, model_path, device='cpu'):
        self.model_path = model_path
        self.device = torch.device(device)
        self.model = None
        self.window_size = 250  # 2 seconds at 125 Hz
        self.eeg_buffer = deque(maxlen=self.window_size)
        self.sample_count = 0

        self.load_model()
        logger.info(f"Initialized EEGNet classifier on {device}")

    def load_model(self):
        """Load trained EEGNet model"""
        try:
            # Create model
            self.model = EEGNetv4(
                n_chans=16,
                n_outputs=4,
                n_times=250,
                drop_prob=0.25,
                kernel_length=64,
            )

            # Load trained weights
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)

            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info(f"Model loaded from {self.model_path}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def add_sample(self, sample):
        """Add EEG sample to buffer"""
        self.eeg_buffer.append(sample)
        self.sample_count += 1

    def predict(self):
        """Make prediction if enough samples"""
        if len(self.eeg_buffer) < self.window_size or self.model is None:
            return None

        try:
            # Get window
            window = np.array(list(self.eeg_buffer)[-self.window_size:])
            window = window.T  # (channels, time_points)

            # Preprocess
            window = self.preprocess_window(window)

            # Convert to tensor
            input_tensor = torch.FloatTensor(window).unsqueeze(0).to(self.device)

            # Predict
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)

            # Get results
            probabilities_np = probabilities.cpu().numpy()[0]
            predicted_class = np.argmax(probabilities_np)
            confidence = np.max(probabilities_np)

            class_names = ['feet', 'left_hand', 'right_hand', 'tongue']

            return {
                'class_label': class_names[predicted_class],
                'confidence': confidence,
                'probabilities': probabilities_np,
                'sample_count': self.sample_count,
                'window_end_sample': self.sample_count  # End of the 2-second window
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None

    def preprocess_window(self, window):
        """Preprocess EEG window"""
        # Simple normalization
        window = (window - np.mean(window, axis=-1, keepdims=True)) / \
                (np.std(window, axis=-1, keepdims=True) + 1e-8)
        return window

    def reset(self):
        """Reset classifier state"""
        self.eeg_buffer.clear()
        self.sample_count = 0

class EEGNetGUI:
    """EEGNet classification GUI"""

    def __init__(self, model_path, eeg_stream_name='MOABB_EEG_RAW', marker_stream_name='MOABB_Markers'):
        self.model_path = model_path
        self.eeg_stream_name = eeg_stream_name
        self.marker_stream_name = marker_stream_name

        # Initialize components
        self.classifier = None
        self.eeg_inlet = None
        self.marker_inlet = None
        self.running = False
        self.update_thread = None

        # Class names
        self.class_names = ['feet', 'left_hand', 'right_hand', 'tongue']

        # Statistics
        self.results = []
        self.true_labels = []
        self.predicted_labels = []

        # Performance monitoring
        self.latencies = deque(maxlen=100)
        self.start_time = None

        # Marker tracking
        self.markers = []  # List of (sample_idx, marker) tuples
        self.current_marker = None
        self.marker_window_size = 250  # 2 seconds for alignment

        # CSV export
        self.session_start_time = None
        self.csv_data = []

        logger.info("Initialized EEGNet GUI")

    def create_gui(self):
        """Create GUI window"""
        self.root = tk.Tk()
        self.root.title("Real-time BCI Classifier - EEGNetv4")
        self.root.geometry("900x750")

        # Create widgets
        self._create_widgets()

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=1)

        logger.info("GUI created")

    def _create_widgets(self):
        """Create GUI widgets"""
        # Control frame
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.grid(row=0, column=0, sticky="nsew")

        # Start/Stop button
        self.start_button = ttk.Button(control_frame, text="Start Classification", command=self.start_classification)
        self.start_button.grid(row=0, column=0, padx=5, pady=5)

        self.stop_button = ttk.Button(control_frame, text="Stop Classification", command=self.stop_classification, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=5, pady=5)

        # CSV Export buttons
        self.save_csv_button = ttk.Button(control_frame, text="Save Results CSV", command=self.save_results_csv, state="disabled")
        self.save_csv_button.grid(row=0, column=2, padx=5, pady=5)

        self.save_session_button = ttk.Button(control_frame, text="Save Session CSV", command=self.save_session_csv, state="disabled")
        self.save_session_button.grid(row=0, column=3, padx=5, pady=5)

        # Status label
        self.status_label = ttk.Label(control_frame, text="Status: Ready")
        self.status_label.grid(row=0, column=4, padx=20, pady=5)

        # Performance label
        self.performance_label = ttk.Label(control_frame, text="Latency: -- ms")
        self.performance_label.grid(row=0, column=5, padx=20, pady=5)

        # Marker status
        self.marker_status_label = ttk.Label(control_frame, text="Marker: None")
        self.marker_status_label.grid(row=0, column=6, padx=20, pady=5)

        # Results frame
        results_frame = ttk.LabelFrame(self.root, text="Classification Results", padding="10")
        results_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

        # Prediction display
        prediction_frame = ttk.Frame(results_frame)
        prediction_frame.grid(row=0, column=0, columnspan=3, pady=10, sticky="ew")

        # Prediction label
        self.prediction_label = ttk.Label(prediction_frame, text="Prediction: Waiting...",
                                        font=("Arial", 14), width=25, anchor="w")
        self.prediction_label.grid(row=0, column=0, sticky="w", padx=(0, 20))

        # True label
        self.true_label = ttk.Label(prediction_frame, text="True: None",
                                  font=("Arial", 12), width=15, anchor="w", foreground="blue")
        self.true_label.grid(row=0, column=1, sticky="w")

        # Class confidence bars
        self.confidence_bars = {}
        self.confidence_labels = {}

        for i, class_name in enumerate(self.class_names):
            # Class label
            label = ttk.Label(results_frame, text=f"{class_name}:")
            label.grid(row=i+1, column=0, sticky="w", pady=2)

            # Confidence bar
            bar = ttk.Progressbar(results_frame, length=300, mode='determinate')
            bar.grid(row=i+1, column=1, sticky="w", padx=5, pady=2)

            # Confidence value label
            conf_label = ttk.Label(results_frame, text="0.00")
            conf_label.grid(row=i+1, column=2, sticky="w", padx=5, pady=2)

            self.confidence_bars[class_name] = bar
            self.confidence_labels[class_name] = conf_label

        # Statistics
        self.accuracy_label = ttk.Label(results_frame, text="Accuracy: 0.00%")
        self.accuracy_label.grid(row=len(self.class_names)+1, column=0, columnspan=3, pady=10)

        self.sample_label = ttk.Label(results_frame, text="Samples processed: 0")
        self.sample_label.grid(row=len(self.class_names)+2, column=0, columnspan=3, pady=5)

        # Confidence plot
        plot_frame = ttk.LabelFrame(self.root, text="Confidence History", padding="10")
        plot_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)

        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def connect_lsl_streams(self):
        """Connect to LSL streams"""
        try:
            logger.info("Connecting to LSL streams...")

            # Connect to EEG stream
            eeg_streams = resolve_byprop('name', self.eeg_stream_name, timeout=10)
            if not eeg_streams:
                raise Exception(f"EEG stream '{self.eeg_stream_name}' not found")

            self.eeg_inlet = StreamInlet(eeg_streams[0])
            logger.info(f"Connected to EEG stream: {eeg_streams[0].name()}")

            # Try to connect to marker stream (optional)
            try:
                marker_streams = resolve_byprop('name', self.marker_stream_name, timeout=5)
                if marker_streams:
                    self.marker_inlet = StreamInlet(marker_streams[0])
                    logger.info(f"Connected to marker stream: {marker_streams[0].name()}")
                else:
                    logger.warning(f"Marker stream '{self.marker_stream_name}' not found - running without markers")
                    self.marker_inlet = None
            except Exception as e:
                logger.warning(f"Could not connect to marker stream: {e} - running without markers")
                self.marker_inlet = None

            return True

        except Exception as e:
            logger.error(f"LSL connection failed: {e}")
            return False

    def find_marker_for_prediction(self, prediction_sample_idx):
        """Find the marker that corresponds to this prediction"""
        # Look for markers that occurred 2 seconds (250 samples) before this prediction
        target_marker_time = prediction_sample_idx - self.marker_window_size

        for marker_sample_idx, marker in self.markers:
            if marker_sample_idx <= target_marker_time <= marker_sample_idx + 50:  # Allow some tolerance
                return marker

        return None

    def start_classification(self):
        """Start real-time classification"""
        try:
            # Connect to LSL
            if not self.connect_lsl_streams():
                messagebox.showerror("Error", "Failed to connect to LSL streams")
                return

            # Initialize classifier
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {device}")

            self.classifier = EEGNetRealtimeClassifier(self.model_path, device=device)

            # Start classification
            self.running = True
            self.start_time = time.time()
            self.session_start_time = datetime.now()
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            self.status_label.config(text="Status: Running")

            # Start update thread
            self.update_thread = threading.Thread(target=self._run_classifier)
            self.update_thread.daemon = True
            self.update_thread.start()

            logger.info("Classification started")

        except Exception as e:
            logger.error(f"Failed to start classification: {e}")
            messagebox.showerror("Error", f"Failed to start classifier: {str(e)}")

    def stop_classification(self):
        """Stop real-time classification"""
        self.running = False

        if self.classifier:
            self.classifier.reset()

        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_label.config(text="Status: Stopped")

        # Enable CSV export buttons
        self.save_csv_button.config(state="normal")
        self.save_session_button.config(state="normal")

        # Show final results
        if self.true_labels and self.predicted_labels:
            correct = sum(1 for true, pred in zip(self.true_labels, self.predicted_labels) if true == pred)
            accuracy = correct / len(self.true_labels) * 100
            messagebox.showinfo("Results", f"Classification stopped.\nFinal accuracy: {accuracy:.2f}% ({correct}/{len(self.true_labels)})")

        logger.info("Classification stopped")

    def save_results_csv(self):
        """Save classification results to CSV"""
        if not self.results:
            messagebox.showwarning("Warning", "No results to save!")
            return

        try:
            # Create results DataFrame
            df_data = []
            for result in self.results:
                df_data.append({
                    'timestamp': result.get('timestamp', ''),
                    'sample_idx': result['sample_idx'],
                    'window_end_sample': result['window_end_sample'],
                    'true_label': result['true_label'],
                    'predicted_label': result['pred_label'],
                    'confidence': result['confidence'],
                    'feet_prob': result['probabilities'][0],
                    'left_hand_prob': result['probabilities'][1],
                    'right_hand_prob': result['probabilities'][2],
                    'tongue_prob': result['probabilities'][3]
                })

            df = pd.DataFrame(df_data)

            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"classification_results_{timestamp}.csv"

            # Create logs directory if it doesn't exist
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)

            filepath = log_dir / filename
            df.to_csv(filepath, index=False)

            logger.info(f"Results saved to {filepath}")
            messagebox.showinfo("Success", f"Results saved to:\n{filepath}")

        except Exception as e:
            logger.error(f"Failed to save results CSV: {e}")
            messagebox.showerror("Error", f"Failed to save CSV: {str(e)}")

    def save_session_csv(self):
        """Save session data to CSV"""
        if not self.csv_data:
            messagebox.showwarning("Warning", "No session data to save!")
            return

        try:
            # Create session DataFrame
            df = pd.DataFrame(self.csv_data)

            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"session_data_{timestamp}.csv"

            # Create logs directory if it doesn't exist
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)

            filepath = log_dir / filename
            df.to_csv(filepath, index=False)

            logger.info(f"Session data saved to {filepath}")
            messagebox.showinfo("Success", f"Session data saved to:\n{filepath}")

        except Exception as e:
            logger.error(f"Failed to save session CSV: {e}")
            messagebox.showerror("Error", f"Failed to save session CSV: {str(e)}")

    def _run_classifier(self):
        """Run classifier in separate thread"""
        try:
            while self.running and self.eeg_inlet is not None:
                # Get EEG sample
                sample, ts = self.eeg_inlet.pull_sample(timeout=0.01)
                if sample is not None and self.classifier is not None:
                    # Convert to numpy array
                    sample_array = np.array(sample, dtype=np.float32)

                    # Add to classifier
                    self.classifier.add_sample(sample_array)

                # Get marker (if available)
                if self.marker_inlet and self.classifier is not None:
                    try:
                        marker, mts = self.marker_inlet.pull_sample(timeout=0.0)
                        if marker is not None and marker[0] and marker[0] != 'start':
                            # Store marker with current sample count
                            self.markers.append((self.classifier.sample_count, marker[0]))
                            self.current_marker = marker[0]
                            logger.info(f"Received marker: {marker[0]} at sample {self.classifier.sample_count}")
                    except:
                        pass

                # Make prediction
                if self.classifier is not None:
                    result = self.classifier.predict()
                if result is not None:
                    # Find corresponding marker for this prediction
                    corresponding_marker = self.find_marker_for_prediction(result['window_end_sample'])

                    # Record true label if marker found
                    if corresponding_marker:
                        self.true_labels.append(corresponding_marker)
                        self.predicted_labels.append(result['class_label'])

                    # Create result dict
                    result_dict = {
                        'timestamp': datetime.now().isoformat(),
                        'sample_idx': result['sample_count'],
                        'window_end_sample': result['window_end_sample'],
                        'true_label': corresponding_marker if corresponding_marker else 'unknown',
                        'pred_label': result['class_label'],
                        'confidence': result['confidence'],
                        'probabilities': result['probabilities']
                    }
                    self.results.append(result_dict)

                    # Add to CSV data
                    csv_row = {
                        'timestamp': result_dict['timestamp'],
                        'sample_idx': result_dict['sample_idx'],
                        'window_end_sample': result_dict['window_end_sample'],
                        'true_label': result_dict['true_label'],
                        'predicted_label': result_dict['pred_label'],
                        'confidence': result_dict['confidence'],
                        'feet_prob': result_dict['probabilities'][0],
                        'left_hand_prob': result_dict['probabilities'][1],
                        'right_hand_prob': result_dict['probabilities'][2],
                        'tongue_prob': result_dict['probabilities'][3]
                    }
                    self.csv_data.append(csv_row)

                    # Update GUI
                    self.root.after(0, self._update_gui, result_dict)

                time.sleep(0.001)

        except Exception as e:
            logger.error(f"Classification error: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Classification error: {str(e)}"))
            self.root.after(0, self.stop_classification)

    def _update_gui(self, result):
        """Update GUI with new result"""
        if not self.running:
            return

        # Update prediction
        self.prediction_label.config(text=f"Prediction: {result['pred_label']}")
        self.true_label.config(text=f"True: {result['true_label']}")

        # Update marker status
        if self.current_marker:
            self.marker_status_label.config(text=f"Marker: {self.current_marker}")
        else:
            self.marker_status_label.config(text="Marker: None")

        # Update confidence bars
        for i, class_name in enumerate(self.class_names):
            confidence = result['probabilities'][i]
            self.confidence_bars[class_name]['value'] = confidence * 100
            self.confidence_labels[class_name].config(text=f"{confidence:.3f}")

        # Update accuracy
        if self.true_labels and self.predicted_labels:
            correct = sum(1 for true, pred in zip(self.true_labels, self.predicted_labels) if true == pred)
            accuracy = correct / len(self.true_labels) * 100
            self.accuracy_label.config(text=f"Accuracy: {accuracy:.2f}%")

        # Update sample count
        self.sample_label.config(text=f"Samples processed: {len(self.results)}")

        # Update performance
        if self.start_time:
            latency = (time.time() - self.start_time) * 1000  # ms
            self.latencies.append(latency)
            avg_latency = sum(self.latencies) / len(self.latencies)
            self.performance_label.config(text=f"Latency: {avg_latency:.1f} ms")

        # Update plot
        self._update_confidence_plot()

    def _update_confidence_plot(self):
        """Update confidence history plot with markers"""
        if not self.results:
            return

        # Get last 50 results
        recent_results = self.results[-50:]
        sample_indices = [r['window_end_sample'] for r in recent_results]

        self.ax.clear()

        # Plot confidence for each class
        for i, class_name in enumerate(self.class_names):
            confidences = [r['probabilities'][i] for r in recent_results]
            self.ax.plot(sample_indices, confidences, label=class_name, alpha=0.8, linewidth=2)

        # Plot markers if available - only show markers within the current x-range
        if self.markers and sample_indices:
            marker_times = []
            marker_labels = []
            marker_confidences = []  # Use actual confidence values for y-position

            x_min = sample_indices[0]
            x_max = sample_indices[-1]

            for marker_sample, marker in self.markers:
                # Only show markers within the current x-range
                if x_min <= marker_sample <= x_max:
                    marker_times.append(marker_sample)
                    marker_labels.append(marker)

                    # Find the confidence value at this time point for y-position
                    # Use the maximum confidence across all classes
                    marker_idx = None
                    for i, sample_idx in enumerate(sample_indices):
                        if sample_idx >= marker_sample:
                            marker_idx = i
                            break

                    if marker_idx is not None and marker_idx < len(recent_results):
                        max_conf = np.max(recent_results[marker_idx]['probabilities'])
                        marker_confidences.append(max_conf + 0.05)  # Slightly above the curve
                    else:
                        marker_confidences.append(0.8)  # Default position

            if marker_times:
                # Plot markers at dynamic y-positions based on confidence
                self.ax.scatter(marker_times, marker_confidences,
                              c='red', s=60, alpha=0.9, label='Markers', zorder=5, marker='^')

                # Add marker labels
                for i, (marker_time, marker_label) in enumerate(zip(marker_times, marker_labels)):
                    y_pos = marker_confidences[i]
                    self.ax.annotate(marker_label, (marker_time, y_pos),
                                   xytext=(0, 8), textcoords='offset points',
                                   ha='center', va='bottom', fontsize=7,
                                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9))

        # Set proper y-limits
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel('Sample Index (Window End)')
        self.ax.set_ylabel('Confidence')
        self.ax.set_title('Confidence History by Class (with Markers)')
        self.ax.legend(loc='upper right')
        self.ax.grid(True, alpha=0.3)

        # Ensure tight layout to prevent cutoff
        self.fig.tight_layout()
        self.canvas.draw()

    def run(self):
        """Run the GUI"""
        try:
            self.create_gui()
            self.root.mainloop()
        except Exception as e:
            logger.error(f"GUI failed: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='EEGNet GUI Classifier')
    parser.add_argument('--model-path', type=str, default='models/eegnetv4_subj3_model_250.pth',
                       help='Path to trained EEGNet model')
    parser.add_argument('--eeg-stream', type=str, default='MOABB_EEG_RAW',
                       help='LSL EEG stream name')
    parser.add_argument('--marker-stream', type=str, default='MOABB_Markers',
                       help='LSL marker stream name (optional)')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("EEGNet GUI Classifier")
    logger.info("=" * 60)

    try:
        # Create GUI
        gui = EEGNetGUI(
            model_path=args.model_path,
            eeg_stream_name=args.eeg_stream,
            marker_stream_name=args.marker_stream
        )

        # Run GUI
        gui.run()

    except Exception as e:
        logger.error(f"GUI failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()