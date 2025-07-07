# -*- coding: utf-8 -*-
"""
GUI example for the real-time BCI classifier using ShallowFBCSPNet.
Uses realtime_shallow_classifier.py as backend.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from realtime_shallow_classifier import RealtimeBCIClassifier
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class BCIClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-time BCI Classifier")
        self.root.geometry("800x700")

        # Initialize classifier
        self.classifier = None
        self.running = False
        self.update_thread = None

        # Class names for display
        self.class_names = ['feet', 'left_hand', 'right_hand', 'tongue']

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

        # Results frame
        results_frame = ttk.LabelFrame(self.root, text="Results", padding="10")
        results_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

        # Prediction display with separate, fixed-width elements
        prediction_frame = ttk.Frame(results_frame)
        prediction_frame.grid(row=0, column=0, columnspan=3, pady=10, sticky="ew")

        # Prediction label (fixed width)
        self.prediction_label = ttk.Label(prediction_frame, text="Prediction: Waiting...",
                                        font=("Arial", 14), width=25, anchor="w")
        self.prediction_label.grid(row=0, column=0, sticky="w", padx=(0, 20))

        # True label (fixed width)
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

        # Accuracy display
        self.accuracy_label = ttk.Label(results_frame, text="Accuracy: 0.00%")
        self.accuracy_label.grid(row=len(self.class_names)+1, column=0, columnspan=3, pady=10)

        # Sample count
        self.sample_label = ttk.Label(results_frame, text="Samples processed: 0")
        self.sample_label.grid(row=len(self.class_names)+2, column=0, columnspan=3, pady=5)

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
        """Start the real-time classification."""
        try:
            self.classifier = RealtimeBCIClassifier(subject_id=3)
            self.classifier.set_callback(self._update_gui)

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
            self.classifier.stop()

        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_label.config(text="Status: Stopped")

        # Save results
        if self.classifier and self.classifier.results:
            self.classifier._save_results()
            accuracy = self.classifier.get_accuracy()
            messagebox.showinfo("Results", f"Classification stopped.\nFinal accuracy: {accuracy*100:.2f}%")

    def _run_classifier(self):
        """Run the classifier in a separate thread."""
        try:
            while self.running:
                result = self.classifier.process_sample()
                if result:
                    # Update GUI in main thread
                    self.root.after(0, self._update_gui, result)
                time.sleep(0.001)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Classification error: {str(e)}"))
            self.root.after(0, self.stop_classification)

    def _update_gui(self, result):
        """Update GUI with new classification result."""
        if not self.running or not self.classifier:
            return

        # Update prediction
        self.prediction_label.config(text=f"Prediction: {result['pred_label']}")
        self.true_label.config(text=f"True: {result['true_label']}")

        # Update confidence bars for each class
        for i, class_name in enumerate(self.class_names):
            confidence = result[f'conf_{i}']
            self.confidence_bars[class_name]['value'] = confidence * 100
            self.confidence_labels[class_name].config(text=f"{confidence:.3f}")

        # Update accuracy
        accuracy = self.classifier.get_accuracy()
        self.accuracy_label.config(text=f"Accuracy: {accuracy*100:.2f}%")

        # Update sample count
        self.sample_label.config(text=f"Samples processed: {len(self.classifier.results)}")

        # Update confidence plot
        self._update_confidence_plot()

    def _update_confidence_plot(self):
        """Update the confidence history plot."""
        if not self.classifier or not self.classifier.results:
            return

        # Get last 50 results for plotting
        recent_results = self.classifier.results[-50:]
        sample_indices = [r['sample_idx'] for r in recent_results]

        self.ax.clear()

        # Plot confidence for each class
        for i, class_name in enumerate(self.class_names):
            confidences = [r[f'conf_{i}'] for r in recent_results]
            self.ax.plot(sample_indices, confidences, label=class_name, alpha=0.8, linewidth=2)

        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel('Sample Index')
        self.ax.set_ylabel('Confidence')
        self.ax.set_title('Confidence History by Class')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)

        self.canvas.draw()

def main():
    """Main function to run the GUI."""
    root = tk.Tk()
    app = BCIClassifierGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()