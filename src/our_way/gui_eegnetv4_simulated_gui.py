# -*- coding: utf-8 -*-
"""
GUI example for the real-time BCI classifier using EEGNetv4 with simulated data.
Uses realtime_eegnetv4_classifier.py as backend.
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
import torch

class EEGNetv4SimulatedGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-time BCI Classifier - EEGNetv4 (Simulated)")
        self.root.geometry("900x800")

        # Initialize classifier
        self.classifier = None
        self.running = False
        self.update_thread = None

        # Class names for display (correct order based on MOABB mapping)
        self.class_names = ['feet', 'left_hand', 'right_hand', 'tongue']

        # Statistics
        self.results = []
        self.true_labels = []
        self.predicted_labels = []
        self.sample_count = 0

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

        # Device label
        device = 'CUDA' if torch.cuda.is_available() else 'CPU'
        self.device_label = ttk.Label(control_frame, text=f"Device: {device}")
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

        # True label (fixed width) - for simulated data
        self.true_label = ttk.Label(prediction_frame, text="True: Simulated",
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

        # Sample count
        self.sample_label = ttk.Label(results_frame, text="Samples processed: 0")
        self.sample_label.grid(row=len(self.class_names)+1, column=0, columnspan=3, pady=5)

        # Confidence plot
        plot_frame = ttk.LabelFrame(self.root, text="Confidence History", padding="10")
        plot_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)

        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=1)

    def start_classification(self):
        """Start the real-time classification with simulated data."""
        try:
            # Initialize EEGNetv4 classifier
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
            model_path = os.path.join(project_root, 'models', 'eegnetv4_subj3_model_250_full.pth')

            if not os.path.exists(model_path):
                messagebox.showerror("Error", f"Model not found at {model_path}")
                return

            # Use GPU if available
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Using device: {device}")

            self.classifier = EEGNetv4RealtimeClassifier(
                model_path=model_path,
                window_size=250,
                sample_rate=125,
                device=device
            )

            self.running = True
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            self.status_label.config(text="Status: Running (Simulated)")

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

        messagebox.showinfo("Results", f"Simulation stopped.\nSamples processed: {self.sample_count}")

    def _run_classifier(self):
        """Run the classifier with simulated data."""
        try:
            sample_idx = 0

            # Generate simulated EEG data (random noise with some structure)
            np.random.seed(42)  # For reproducible results

            while self.running:
                # Generate simulated EEG sample (16 channels)
                simulated_sample = np.random.normal(0, 1, 16) * 1e-6  # Scale to microvolts

                # Add some periodic structure to make it more realistic
                t = sample_idx / 125.0  # Time in seconds
                for ch in range(16):
                    simulated_sample[ch] += 0.1 * np.sin(2 * np.pi * 10 * t + ch) * 1e-6

                # Add sample to classifier
                self.classifier.add_sample(simulated_sample)
                sample_idx += 1

                # Get prediction if ready
                if len(self.classifier.buffer) >= self.classifier.window_size:
                    result = self.classifier.predict()
                    if result:
                        # Update GUI in main thread
                        self.root.after(0, self._update_gui, result, sample_idx)

                time.sleep(0.008)  # ~125 Hz sampling rate

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Classification error: {str(e)}"))
            self.root.after(0, self.stop_classification)

    def _update_gui(self, result, sample_count):
        """Update GUI with new classification result."""
        if not self.running:
            return

        # Store results
        self.results.append(result)
        self.predicted_labels.append(result['class_label'])
        self.sample_count = sample_count

        # Update prediction
        self.prediction_label.config(text=f"Prediction: {result['class_label']}")

        # Update confidence bars for each class
        for i, class_name in enumerate(self.class_names):
            confidence = result[f'conf_{i}']
            self.confidence_bars[class_name]['value'] = confidence * 100
            self.confidence_labels[class_name].config(text=f"{confidence:.3f}")

        # Update sample count
        self.sample_label.config(text=f"Samples processed: {sample_count}")

        # Update confidence plot
        self._update_confidence_plot()

    def _update_confidence_plot(self):
        """Update the confidence history plot."""
        if not self.results:
            return

        # Get last 50 results for plotting
        recent_results = self.results[-50:]
        sample_indices = list(range(len(recent_results)))

        self.ax.clear()

        # Plot confidence for each class
        for i, class_name in enumerate(self.class_names):
            confidences = [r[f'conf_{i}'] for r in recent_results]
            self.ax.plot(sample_indices, confidences, label=class_name, alpha=0.8, linewidth=2)

        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel('Sample Index')
        self.ax.set_ylabel('Confidence')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)

        self.canvas.draw()

def main():
    root = tk.Tk()
    app = EEGNetv4SimulatedGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()