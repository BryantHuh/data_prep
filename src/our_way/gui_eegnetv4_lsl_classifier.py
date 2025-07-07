# -*- coding: utf-8 -*-
"""
GUI example for the real-time BCI classifier using EEGNetv4 with LSL streams.
Optimized for GPU usage and better performance.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from realtime_eegnetv4_classifier import EEGNetv4RealtimeClassifier
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import os
from pylsl import StreamInlet, resolve_byprop
from collections import deque
import torch

class EEGNetv4LSLClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-time BCI Classifier - EEGNetv4 with LSL (GPU Optimized)")
        self.root.geometry("1000x900")

        # Initialize classifier and LSL
        self.classifier = None
        self.eeg_inlet = None
        self.marker_inlet = None
        self.running = False
        self.update_thread = None

        # Class names for display
        self.class_names = ['feet', 'left_hand', 'right_hand', 'tongue']

        # Statistics with better tracking
        self.prediction_history = deque(maxlen=100)
        self.confidence_history = deque(maxlen=100)
        self.true_labels = []
        self.predicted_labels = []
        self.sample_count = 0
        self.last_update_time = time.time()
        self.update_interval = 0.1  # Update GUI every 100ms for better performance

        # Create GUI elements
        self._create_widgets()

    def _create_widgets(self):
        """Create the GUI widgets."""
        # Control frame
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.grid(row=0, column=0, sticky="nsew")

        # Start/Stop button
        self.start_button = ttk.Button(control_frame, text="Start LSL Classification", command=self.start_classification)
        self.start_button.grid(row=0, column=0, padx=5, pady=5)

        self.stop_button = ttk.Button(control_frame, text="Stop Classification", command=self.stop_classification, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=5, pady=5)

        # Status label
        self.status_label = ttk.Label(control_frame, text="Status: Ready")
        self.status_label.grid(row=0, column=2, padx=20, pady=5)

        # LSL Status
        self.lsl_status_label = ttk.Label(control_frame, text="LSL: Not connected")
        self.lsl_status_label.grid(row=0, column=3, padx=20, pady=5)

        # Device Status
        self.device_label = ttk.Label(control_frame, text="Device: CPU")
        self.device_label.grid(row=0, column=4, padx=20, pady=5)

        # Results frame
        results_frame = ttk.LabelFrame(self.root, text="Results", padding="10")
        results_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

        # Prediction display with separate, fixed-width elements
        prediction_frame = ttk.Frame(results_frame)
        prediction_frame.grid(row=0, column=0, columnspan=3, pady=10, sticky="ew")

        # Prediction label (fixed width)
        self.prediction_label = ttk.Label(prediction_frame, text="Prediction: Waiting...",
                                        font=("Arial", 16, "bold"), width=25, anchor="w")
        self.prediction_label.grid(row=0, column=0, sticky="w", padx=(0, 20))

        # True label (fixed width)
        self.true_label = ttk.Label(prediction_frame, text="True: None",
                                  font=("Arial", 14), width=15, anchor="w", foreground="blue")
        self.true_label.grid(row=0, column=1, sticky="w")

        # Class confidence bars
        self.confidence_bars = {}
        self.confidence_labels = {}

        for i, class_name in enumerate(self.class_names):
            # Class label
            label = ttk.Label(results_frame, text=f"{class_name}:", font=("Arial", 10))
            label.grid(row=i+1, column=0, sticky="w", pady=2)

            # Confidence bar
            bar = ttk.Progressbar(results_frame, length=400, mode='determinate')
            bar.grid(row=i+1, column=1, sticky="w", padx=5, pady=2)

            # Confidence value label
            conf_label = ttk.Label(results_frame, text="0.00", font=("Arial", 10))
            conf_label.grid(row=i+1, column=2, sticky="w", padx=5, pady=2)

            self.confidence_bars[class_name] = bar
            self.confidence_labels[class_name] = conf_label

        # Confidence display
        self.confidence_label = ttk.Label(results_frame, text="Confidence: 0.00", font=("Arial", 12))
        self.confidence_label.grid(row=len(self.class_names)+1, column=0, columnspan=3, pady=10)

        # Accuracy display
        self.accuracy_label = ttk.Label(results_frame, text="Accuracy: 0.00%", font=("Arial", 12))
        self.accuracy_label.grid(row=len(self.class_names)+2, column=0, columnspan=3, pady=5)

        # Sample count and FPS
        self.sample_label = ttk.Label(results_frame, text="Samples: 0 | FPS: 0", font=("Arial", 10))
        self.sample_label.grid(row=len(self.class_names)+3, column=0, columnspan=3, pady=5)

        # Confidence plot
        plot_frame = ttk.LabelFrame(self.root, text="Confidence History", padding="10")
        plot_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)

        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=1)

    def _connect_lsl(self):
        """Connect to LSL streams."""
        try:
            print("Looking for EEG and marker streams...")
            eeg_streams = resolve_byprop('type', 'EEG', timeout=10)
            marker_streams = resolve_byprop('type', 'Markers', timeout=10)

            if not eeg_streams:
                raise Exception("No EEG stream found")
            if not marker_streams:
                raise Exception("No marker stream found")

            self.eeg_inlet = StreamInlet(eeg_streams[0])
            self.marker_inlet = StreamInlet(marker_streams[0])

            self.lsl_status_label.config(text="LSL: Connected")
            print("Connected to LSL streams")
            return True

        except Exception as e:
            self.lsl_status_label.config(text="LSL: Connection failed")
            print(f"LSL connection failed: {e}")
            return False

    def start_classification(self):
        """Start the real-time classification."""
        try:
            # Connect to LSL
            if not self._connect_lsl():
                messagebox.showerror("Error", "Failed to connect to LSL streams")
                return

            # Initialize EEGNetv4 classifier with GPU
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
            model_path = os.path.join(project_root, 'models', 'eegnetv4_subj3_model_250_full.pth')

            if not os.path.exists(model_path):
                messagebox.showerror("Error", f"Model not found at {model_path}")
                return

            # Use GPU if available
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.device_label.config(text=f"Device: {device.upper()}")

            self.classifier = EEGNetv4RealtimeClassifier(model_path, device=device)

            self.running = True
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            self.status_label.config(text="Status: Running")

            # Start update thread
            self.update_thread = threading.Thread(target=self._run_classifier)
            self.update_thread.daemon = True
            self.update_thread.start()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start classifier: {str(e)}")

    def stop_classification(self):
        """Stop the real-time classification."""
        self.running = False
        if self.classifier:
            self.classifier.reset()

        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_label.config(text="Status: Stopped")

        # Calculate and show final accuracy
        if self.true_labels and self.predicted_labels:
            correct = sum(1 for true, pred in zip(self.true_labels, self.predicted_labels) if true == pred)
            accuracy = correct / len(self.true_labels) * 100
            messagebox.showinfo("Results", f"Classification stopped.\nFinal accuracy: {accuracy:.2f}% ({correct}/{len(self.true_labels)})\nTotal samples: {self.sample_count}")

    def _run_classifier(self):
        """Run the classifier in a separate thread."""
        try:
            current_label = None
            sample_idx = 0
            start_time = time.time()

            while self.running:
                # Pull EEG sample
                sample, ts = self.eeg_inlet.pull_sample(timeout=0.01)  # Reduced timeout for better performance
                if sample is not None:
                    # Convert sample to numpy array
                    sample_array = np.array(sample, dtype=np.float32)

                    # Add sample to classifier
                    self.classifier.add_sample(sample_array)
                    sample_idx += 1
                    self.sample_count += 1

                # Pull marker
                marker, mts = self.marker_inlet.pull_sample(timeout=0.0)
                if marker is not None and marker[0] and marker[0] != 'start':
                    current_label = marker[0]

                # Try to predict
                result = self.classifier.predict()
                if result is not None:
                    # Add true label if available
                    if current_label:
                        self.true_labels.append(current_label)
                        self.predicted_labels.append(result['class_label'])

                    # Store history for plotting
                    self.prediction_history.append(result['class'])
                    self.confidence_history.append(result['confidence'])

                    # Update GUI periodically for better performance
                    current_time = time.time()
                    if current_time - self.last_update_time >= self.update_interval:
                        self.root.after(0, self._update_gui, result, current_label, sample_idx, start_time)
                        self.last_update_time = current_time

                # Small delay to prevent busy waiting
                time.sleep(0.001)

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Classification error: {str(e)}"))
            self.root.after(0, self.stop_classification)

    def _update_gui(self, result, true_label=None, sample_idx=0, start_time=0):
        """Update GUI with new classification result."""
        if not self.running or not self.classifier:
            return

        # Calculate FPS
        elapsed_time = time.time() - start_time
        fps = sample_idx / elapsed_time if elapsed_time > 0 else 0

        # Update prediction (separate, fixed-width elements)
        self.prediction_label.config(text=f"Prediction: {result['class_label']}")

        # Update true label (separate, fixed-width element)
        if true_label:
            self.true_label.config(text=f"True: {true_label}", foreground="blue")
        else:
            self.true_label.config(text="True: None", foreground="gray")

        # Update confidence bars for each class
        for i, class_name in enumerate(self.class_names):
            confidence = result['probabilities'][i]
            self.confidence_bars[class_name]['value'] = confidence * 100
            self.confidence_labels[class_name].config(text=f"{confidence:.3f}")

        # Update overall confidence
        self.confidence_label.config(text=f"Confidence: {result['confidence']:.3f}")

        # Update accuracy
        if self.true_labels and self.predicted_labels:
            correct = sum(1 for true, pred in zip(self.true_labels, self.predicted_labels) if true == pred)
            accuracy = correct / len(self.true_labels) * 100
            self.accuracy_label.config(text=f"Accuracy: {accuracy:.2f}% ({correct}/{len(self.true_labels)})")

        # Update sample count and FPS
        self.sample_label.config(text=f"Samples: {self.sample_count} | FPS: {fps:.1f}")

        # Update confidence plot
        self._update_confidence_plot()

    def _update_confidence_plot(self):
        """Update the confidence history plot."""
        if not self.classifier or not self.confidence_history:
            return

        # Clear and redraw plot
        self.ax.clear()

        # Plot confidence history
        confidence_values = list(self.confidence_history)
        sample_indices = list(range(len(confidence_values)))

        self.ax.plot(sample_indices, confidence_values, 'b-', linewidth=2, alpha=0.8)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel('Prediction Index')
        self.ax.set_ylabel('Confidence')
        self.ax.set_title(f'EEGNetv4 Confidence History (Last {len(confidence_values)} predictions)')
        self.ax.grid(True, alpha=0.3)

        # Add horizontal line at 0.25 (random chance for 4 classes)
        self.ax.axhline(y=0.25, color='r', linestyle='--', alpha=0.5, label='Random chance')
        self.ax.legend()

        self.canvas.draw()

def main():
    """Main function to run the GUI."""
    root = tk.Tk()
    app = EEGNetv4LSLClassifierGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()