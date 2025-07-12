# -*- coding: utf-8 -*-
"""
Interactive console demo for continuous EEGNetv4 classification.
This script demonstrates the interactive classifier with either simulated data or LSL streams.
"""

import os
import time
import numpy as np
import argparse
from realtime_eegnetv4_classifier_interactive import EEGNetv4InteractiveClassifier

def simulate_eeg_data(duration_seconds=60, sample_rate=125):
    """
    Generate simulated EEG data for testing.

    Args:
        duration_seconds: How long to simulate
        sample_rate: Sampling rate in Hz

    Yields:
        EEG samples as numpy arrays
    """
    n_channels = 16
    n_samples = duration_seconds * sample_rate

    print(f"🎮 Generating {duration_seconds}s of simulated EEG data at {sample_rate}Hz...")

    for i in range(n_samples):
        # Generate realistic EEG-like data
        t = i / sample_rate  # Time in seconds

        # Base noise
        sample = np.random.normal(0, 1, n_channels) * 1e-6

        # Add some periodic components to make it more realistic
        for ch in range(n_channels):
            # Alpha rhythm (8-13 Hz)
            sample[ch] += 0.2 * np.sin(2 * np.pi * 10 * t + ch) * 1e-6
            # Beta rhythm (13-30 Hz)
            sample[ch] += 0.1 * np.sin(2 * np.pi * 20 * t + ch * 0.5) * 1e-6
            # Slow drift
            sample[ch] += 0.05 * np.sin(2 * np.pi * 0.1 * t + ch) * 1e-6

        yield sample

def run_interactive_demo(model_path, use_lsl=False, duration=60, prediction_interval=25):
    """
    Run the interactive classification demo.

    Args:
        model_path: Path to the EEGNetv4 model
        use_lsl: Whether to use LSL streams or simulated data
        duration: Duration in seconds (for simulated data)
        prediction_interval: How often to make predictions
    """
    print("🚀 Starting Interactive EEGNetv4 Classification Demo")
    print("=" * 60)

    # Initialize classifier
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📱 Using device: {device}")

    classifier = EEGNetv4InteractiveClassifier(
        model_path=model_path,
        window_size=250,
        sample_rate=125,
        device=device,
        prediction_interval=prediction_interval
    )

    print(f"⏱️  Prediction interval: {prediction_interval} samples ({prediction_interval/125:.2f}s)")
    print(f"🔄 Window size: 250 samples (2.0s)")
    print()

    # Data source
    if use_lsl:
        try:
            from pylsl import StreamInlet, resolve_byprop
            print("🔌 Connecting to LSL EEG stream...")
            eeg_streams = resolve_byprop('type', 'EEG', timeout=10)
            if not eeg_streams:
                raise Exception("No EEG stream found")
            eeg_inlet = StreamInlet(eeg_streams[0])
            print("✅ Connected to LSL stream")
            data_source = "LSL"
        except Exception as e:
            print(f"❌ LSL connection failed: {e}")
            print("🔄 Falling back to simulated data...")
            data_source = "Simulated"
            eeg_inlet = None
    else:
        data_source = "Simulated"
        eeg_inlet = None

    print(f"📊 Data source: {data_source}")
    print()

    # Main loop
    start_time = time.time()
    sample_count = 0
    prediction_count = 0

    try:
        if data_source == "LSL":
            # LSL data loop
            print("🎯 Starting LSL classification (press Ctrl+C to stop)...")
            print()

            while True:
                sample, ts = eeg_inlet.pull_sample(timeout=0.01)
                if sample is not None:
                    sample_array = np.array(sample, dtype=np.float32)
                    classifier.add_sample(sample_array)
                    sample_count += 1

                    # Try to predict
                    result = classifier.predict()
                    if result is not None:
                        prediction_count += 1
                        _print_prediction(result, sample_count, prediction_count)

                time.sleep(0.001)
        else:
            # Simulated data loop
            print("🎮 Starting simulated data classification...")
            print()

            for sample in simulate_eeg_data(duration=duration):
                classifier.add_sample(sample)
                sample_count += 1

                # Try to predict
                result = classifier.predict()
                if result is not None:
                    prediction_count += 1
                    _print_prediction(result, sample_count, prediction_count)

                # Check if we should stop
                if time.time() - start_time > duration:
                    break

                time.sleep(0.008)  # ~125 Hz

    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")

    # Final statistics
    elapsed_time = time.time() - start_time
    stats = classifier.get_statistics()

    print("\n" + "=" * 60)
    print("📈 FINAL STATISTICS")
    print("=" * 60)
    print(f"⏱️  Total time: {elapsed_time:.1f}s")
    print(f"📊 Total samples: {sample_count}")
    print(f"🎯 Total predictions: {prediction_count}")
    print(f"⚡ Sample rate: {sample_count/elapsed_time:.1f} Hz")
    print(f"🎯 Prediction rate: {stats.get('prediction_rate', 0):.1f} Hz")
    print(f"📊 Average confidence: {stats.get('avg_confidence', 0):.3f}")

    # Show trend analysis
    trend = classifier.get_prediction_trend(window_size=20)
    if trend:
        print(f"📈 Final trend: {trend['trend_class']}")
        print(f"🔒 Stability: {trend['stability']:.2f}")

    print("✅ Demo completed!")

def _print_prediction(result, sample_count, prediction_count):
    """Print a formatted prediction result."""
    class_names = ['feet', 'left_hand', 'right_hand', 'tongue']
    colors = ['🔴', '🟢', '🔵', '🟡']

    class_idx = result['class']
    class_name = class_names[class_idx]
    color = colors[class_idx]
    confidence = result['confidence']

    # Create confidence bar
    bar_length = 20
    filled_length = int(confidence * bar_length)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)

    # Format the output
    print(f"🎯 [{prediction_count:3d}] {color} {class_name:10s} | "
          f"Confidence: {confidence:.3f} | {bar} | "
          f"Sample: {sample_count:5d}")

def main():
    """Main function with command line arguments."""
    parser = argparse.ArgumentParser(description="Interactive EEGNetv4 Classification Demo")
    parser.add_argument("--model", type=str,
                       default="models/eegnetv4_subj3_model_250_full.pth",
                       help="Path to the EEGNetv4 model")
    parser.add_argument("--lsl", action="store_true",
                       help="Use LSL streams instead of simulated data")
    parser.add_argument("--duration", type=int, default=60,
                       help="Duration in seconds (for simulated data)")
    parser.add_argument("--interval", type=int, default=25,
                       help="Prediction interval in samples")

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Model not found at {args.model}")
        print("💡 Make sure you have trained the EEGNetv4 model first")
        return

    # Run the demo
    run_interactive_demo(
        model_path=args.model,
        use_lsl=args.lsl,
        duration=args.duration,
        prediction_interval=args.interval
    )

if __name__ == "__main__":
    import torch
    main()