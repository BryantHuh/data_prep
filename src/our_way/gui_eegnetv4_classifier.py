# -*- coding: utf-8 -*-
"""
GUI example for the real-time BCI classifier using EEGNetv4 with simulated data.
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
from collections import deque
import torch

class EEGNetv4ClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-time BCI Classifier - EEGNetv4 (GPU Optimized)")
        self.root.geometry("1000x900")

        # Initialize classifier
        self.classifier = None
        self.running = False
        self.update_thread = None

        # Class names for display
        self.class_names = ['feet', 'left_hand', 'right_hand', 'tongue']

        # Statistics with better tracking
        self.prediction_history = deque(maxlen=100)
        self.confidence_history = deque(maxlen=100)
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
        self.start_button = ttk.Button(control_frame, text="Start Classification", command=self.start_classification)
        self.start_button.grid(row=0, column=0, padx=5, pady=5)

        self.stop_button = ttk.Button(control_frame, text="Stop Classification", command=self.stop_classification, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=5, pady=5)

        # Status label
        self.status_label = ttk.Label(control_frame, text="Status: Ready")
        self.status_label.grid(row=0, column=2, padx=20, pady=5)

        # Device Status
        self.device_label = ttk.Label(control_frame, text="Device: CPU")
        self.device_label.grid(row=0, column=3, padx=20, pady=5)

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

        # True label (fixed width) - for simulated data, this will show "Simulated"
        self.true_label = ttk.Label(prediction_frame, text="True: Simulated",
                                  font=("Arial", 14), width=15, anchor="w", foreground="gray")
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

        # Sample count and FPS
        self.sample_label = ttk.Label(results_frame, text="Samples: 0 | FPS: 0", font=("Arial", 10))
        self.sample_label.grid(row=len(self.class_names)+2, column=0, columnspan=3, pady=5)

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

    def start_classification(self):
        """Start the real-time classification."""
        try:
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

        # Show statistics
        if self.classifier:
            stats = self.classifier.get_statistics()
            if stats:
                messagebox.showinfo("Results", f"Classification stopped.\nAverage confidence: {stats.get('avg_confidence', 0):.3f}\nPredictions made: {stats.get('prediction_count', 0)}\nTotal samples: {self.sample_count}")

    def _run_classifier(self):
        """Run the classifier in a separate thread."""
        try:
            sample_count = 0
            start_time = time.time()

            # Pre-generate some simulated data for better performance
            simulated_data = np.random.randn(16, 1000) * 10  # 16 channels, 1000 samples
            data_idx = 0

            while self.running:
                # Use pre-generated data for better performance
                if data_idx >= simulated_data.shape[1]:
                    # Generate new batch when we run out
                    simulated_data = np.random.randn(16, 1000) * 10
                    data_idx = 0

                simulated_sample = simulated_data[:, data_idx]
                data_idx += 1

                # Add sample to classifier
                self.classifier.add_sample(simulated_sample)
                sample_count += 1
                self.sample_count += 1

                # Try to predict
                result = self.classifier.predict()
                if result is not None:
                    # Store history for plotting
                    self.prediction_history.append(result['class'])
                    self.confidence_history.append(result['confidence'])

                    # Update GUI periodically for better performance
                    current_time = time.time()
                    if current_time - self.last_update_time >= self.update_interval:
                        self.root.after(0, self._update_gui, result, sample_count, start_time)
                        self.last_update_time = current_time

                # Simulate 125 Hz sampling rate
                time.sleep(0.008)

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Classification error: {str(e)}"))
            self.root.after(0, self.stop_classification)

    def _update_gui(self, result, sample_count=0, start_time=0):
        """Update GUI with new classification result."""
        if not self.running or not self.classifier:
            return

        # Calculate FPS
        elapsed_time = time.time() - start_time
        fps = sample_count / elapsed_time if elapsed_time > 0 else 0

        # Update prediction
        self.prediction_label.config(text=f"Prediction: {result['class_label']}")
        self.true_label.config(text=f"True: {result['true_label']}")

        # Update confidence bars for each class
        for i, class_name in enumerate(self.class_names):
            confidence = result['probabilities'][i]
            self.confidence_bars[class_name]['value'] = confidence * 100
            self.confidence_labels[class_name].config(text=f"{confidence:.3f}")

        # Update overall confidence
        self.confidence_label.config(text=f"Confidence: {result['confidence']:.3f}")

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
    app = EEGNetv4ClassifierGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()