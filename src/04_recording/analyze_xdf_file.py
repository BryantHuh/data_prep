#!/usr/bin/env python3
"""
XDF File Analyzer

Analyzes XDF files to extract EEG data and markers for BCI experiments.
This script provides comprehensive analysis of XDF recordings including data quality
assessment, visualization, and detailed reporting.

Features:
- Loads and analyzes XDF files from OpenBCI recordings
- Extracts EEG data and marker streams
- Data quality assessment and statistics
- Visualization of EEG signals and markers
- Detailed analysis reports
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

# XDF imports
try:
    import pyxdf
except ImportError:
    print("pyxdf not found. Install with: pip install pyxdf")
    sys.exit(1)

# Logging utility
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
from utils.logging_utils import setup_logging

# Setup logger
logger = setup_logging('xdf_analyzer', log_dir='logs', level='INFO')

class XDFAnalyzer:
    """Analyzes XDF files for BCI experiments"""

    def __init__(self, xdf_path):
        self.xdf_path = xdf_path
        self.streams = None
        self.header = None
        self.eeg_stream = None
        self.marker_stream = None
        self.eeg_data = None
        self.marker_data = None
        self.eeg_timestamps = None
        self.marker_timestamps = None

    def load_xdf_file(self):
        """Load XDF file"""
        try:
            logger.info(f"Loading XDF file: {self.xdf_path}")

            if not os.path.exists(self.xdf_path):
                raise FileNotFoundError(f"XDF file not found: {self.xdf_path}")

            self.streams, self.header = pyxdf.load_xdf(self.xdf_path)

            logger.info(f"Loaded XDF file with {len(self.streams)} streams")
            logger.info(f"Header info: {self.header}")

            return True

        except Exception as e:
            logger.error(f"Failed to load XDF file: {e}")
            return False

    def analyze_streams(self):
        """Analyze and categorize streams"""
        try:
            logger.info("Analyzing streams...")

            for i, stream in enumerate(self.streams):
                stream_info = stream['info']
                stream_name = stream_info.get('name', ['Unknown'])[0]
                stream_type = stream_info.get('type', ['Unknown'])[0]

                logger.info(f"Stream {i}: {stream_name} ({stream_type})")

                # Categorize streams
                if 'eeg' in stream_name.lower() or 'eeg' in stream_type.lower():
                    self.eeg_stream = stream
                    logger.info(f"  -> EEG stream identified")
                elif 'marker' in stream_name.lower() or 'marker' in stream_type.lower():
                    self.marker_stream = stream
                    logger.info(f"  -> Marker stream identified")
                else:
                    logger.info(f"  -> Unknown stream type")

            return True

        except Exception as e:
            logger.error(f"Failed to analyze streams: {e}")
            return False

    def extract_eeg_data(self):
        """Extract EEG data from stream"""
        if self.eeg_stream is None:
            logger.warning("No EEG stream found")
            return False

        try:
            logger.info("Extracting EEG data...")

            self.eeg_data = np.array(self.eeg_stream['time_series'])
            self.eeg_timestamps = np.array(self.eeg_stream['time_stamps'])

            logger.info(f"EEG data shape: {self.eeg_data.shape}")
            logger.info(f"EEG duration: {self.eeg_timestamps[-1] - self.eeg_timestamps[0]:.2f} seconds")
            logger.info(f"EEG sampling rate: {len(self.eeg_timestamps) / (self.eeg_timestamps[-1] - self.eeg_timestamps[0]):.1f} Hz")

            return True

        except Exception as e:
            logger.error(f"Failed to extract EEG data: {e}")
            return False

    def extract_marker_data(self):
        """Extract marker data from stream"""
        if self.marker_stream is None:
            logger.warning("No marker stream found")
            return False

        try:
            logger.info("Extracting marker data...")

            self.marker_data = np.array(self.marker_stream['time_series'])
            self.marker_timestamps = np.array(self.marker_stream['time_stamps'])

            logger.info(f"Number of markers: {len(self.marker_data)}")
            logger.info(f"Marker duration: {self.marker_timestamps[-1] - self.marker_timestamps[0]:.2f} seconds")

            # Analyze marker types
            unique_markers = set()
            for marker in self.marker_data:
                if isinstance(marker, (list, np.ndarray)):
                    unique_markers.add(marker[0])
                else:
                    unique_markers.add(marker)

            logger.info(f"Unique markers: {sorted(unique_markers)}")

            return True

        except Exception as e:
            logger.error(f"Failed to extract marker data: {e}")
            return False

    def analyze_data_quality(self):
        """Analyze data quality"""
        try:
            logger.info("Analyzing data quality...")

            if self.eeg_data is not None:
                # EEG quality analysis
                eeg_stats = {
                    'mean': np.mean(self.eeg_data, axis=0),
                    'std': np.std(self.eeg_data, axis=0),
                    'min': np.min(self.eeg_data, axis=0),
                    'max': np.max(self.eeg_data, axis=0),
                    'range': np.max(self.eeg_data, axis=0) - np.min(self.eeg_data, axis=0)
                }

                logger.info("EEG Quality Statistics:")
                logger.info(f"  Mean range: {np.min(eeg_stats['mean']):.2f} to {np.max(eeg_stats['mean']):.2f}")
                logger.info(f"  Std range: {np.min(eeg_stats['std']):.2f} to {np.max(eeg_stats['std']):.2f}")
                logger.info(f"  Data range: {np.min(eeg_stats['range']):.2f} to {np.max(eeg_stats['range']):.2f}")

                # Check for artifacts (high amplitude)
                artifact_threshold = np.mean(eeg_stats['std']) * 5
                artifacts = np.sum(np.abs(self.eeg_data) > artifact_threshold, axis=0)
                logger.info(f"  Potential artifacts per channel: {artifacts}")

            if self.marker_data is not None:
                # Marker analysis
                marker_counts = defaultdict(int)
                for marker in self.marker_data:
                    if isinstance(marker, (list, np.ndarray)):
                        marker_counts[marker[0]] += 1
                    else:
                        marker_counts[marker] += 1

                logger.info("Marker Analysis:")
                for marker, count in sorted(marker_counts.items()):
                    logger.info(f"  {marker}: {count} occurrences")

            return True

        except Exception as e:
            logger.error(f"Failed to analyze data quality: {e}")
            return False

    def create_visualizations(self, output_dir='logs'):
        """Create visualizations of the data"""
        try:
            logger.info("Creating visualizations...")

            # Ensure output directory exists
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)

            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'XDF Analysis: {Path(self.xdf_path).name}', fontsize=16)

            # Plot 1: EEG data overview
            if self.eeg_data is not None:
                ax1 = axes[0, 0]
                # Plot first few channels
                n_channels_to_plot = min(4, self.eeg_data.shape[1])
                for i in range(n_channels_to_plot):
                    ax1.plot(self.eeg_timestamps, self.eeg_data[:, i],
                            label=f'Channel {i+1}', alpha=0.7)
                ax1.set_xlabel('Time (s)')
                ax1.set_ylabel('Amplitude')
                ax1.set_title('EEG Data Overview')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

            # Plot 2: EEG power spectrum
            if self.eeg_data is not None:
                ax2 = axes[0, 1]
                # Calculate power spectrum for first channel
                from scipy import signal
                f, psd = signal.welch(self.eeg_data[:, 0], fs=125)  # Assuming 125 Hz
                ax2.semilogy(f, psd)
                ax2.set_xlabel('Frequency (Hz)')
                ax2.set_ylabel('Power Spectral Density')
                ax2.set_title('EEG Power Spectrum (Channel 1)')
                ax2.grid(True, alpha=0.3)

            # Plot 3: Marker timeline
            if self.marker_data is not None:
                ax3 = axes[1, 0]
                marker_types = []
                marker_times = []

                for i, marker in enumerate(self.marker_data):
                    if isinstance(marker, (list, np.ndarray)):
                        marker_types.append(marker[0])
                    else:
                        marker_types.append(marker)
                    marker_times.append(self.marker_timestamps[i])

                # Create color mapping for different markers
                unique_markers = list(set(marker_types))
                colors = plt.cm.Set3(np.linspace(0, 1, len(unique_markers)))

                for i, marker in enumerate(unique_markers):
                    marker_indices = [j for j, m in enumerate(marker_types) if m == marker]
                    marker_times_subset = [marker_times[j] for j in marker_indices]
                    ax3.scatter(marker_times_subset, [i] * len(marker_times_subset),
                              label=marker, color=colors[i], s=50)

                ax3.set_xlabel('Time (s)')
                ax3.set_ylabel('Marker Type')
                ax3.set_title('Marker Timeline')
                ax3.legend()
                ax3.grid(True, alpha=0.3)

            # Plot 4: Data quality metrics
            if self.eeg_data is not None:
                ax4 = axes[1, 1]
                channel_std = np.std(self.eeg_data, axis=0)
                channel_range = np.max(self.eeg_data, axis=0) - np.min(self.eeg_data, axis=0)

                x = range(len(channel_std))
                ax4.bar([i-0.2 for i in x], channel_std, width=0.4, label='Std Dev', alpha=0.7)
                ax4.bar([i+0.2 for i in x], channel_range, width=0.4, label='Range', alpha=0.7)
                ax4.set_xlabel('Channel')
                ax4.set_ylabel('Amplitude')
                ax4.set_title('Channel Quality Metrics')
                ax4.legend()
                ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_path / 'xdf_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Visualizations saved to {output_path / 'xdf_analysis.png'}")
            return True

        except Exception as e:
            logger.error(f"Failed to create visualizations: {e}")
            return False

    def save_analysis_report(self, output_dir='logs'):
        """Save analysis report"""
        try:
            logger.info("Saving analysis report...")

            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)

            report = {
                'file_path': self.xdf_path,
                'num_streams': len(self.streams) if self.streams else 0,
                'eeg_data_shape': self.eeg_data.shape if self.eeg_data is not None else None,
                'num_markers': len(self.marker_data) if self.marker_data is not None else 0,
                'eeg_duration': self.eeg_timestamps[-1] - self.eeg_timestamps[0] if self.eeg_timestamps is not None else None,
                'eeg_sampling_rate': len(self.eeg_timestamps) / (self.eeg_timestamps[-1] - self.eeg_timestamps[0]) if self.eeg_timestamps is not None else None
            }

            # Save as JSON
            import json
            with open(output_path / 'xdf_analysis_report.json', 'w') as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"Analysis report saved to {output_path / 'xdf_analysis_report.json'}")
            return True

        except Exception as e:
            logger.error(f"Failed to save analysis report: {e}")
            return False

    def analyze_xdf(self, output_dir='logs'):
        """Complete XDF analysis pipeline"""
        try:
            logger.info("=" * 60)
            logger.info("XDF File Analysis Pipeline")
            logger.info("=" * 60)

            # Load XDF file
            if not self.load_xdf_file():
                return False

            # Analyze streams
            if not self.analyze_streams():
                return False

            # Extract EEG data
            self.extract_eeg_data()

            # Extract marker data
            self.extract_marker_data()

            # Analyze data quality
            self.analyze_data_quality()

            # Create visualizations
            self.create_visualizations(output_dir)

            # Save analysis report
            self.save_analysis_report(output_dir)

            logger.info("XDF analysis completed successfully!")
            return True

        except Exception as e:
            logger.error(f"XDF analysis failed: {e}")
            return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Analyze XDF file')
    parser.add_argument('xdf_path', type=str, help='Path to XDF file')
    parser.add_argument('--output-dir', type=str, default='logs', help='Output directory for results')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Set log level
    if args.verbose:
        logger.setLevel('DEBUG')

    try:
        # Create analyzer
        analyzer = XDFAnalyzer(args.xdf_path)

        # Analyze XDF file
        success = analyzer.analyze_xdf(args.output_dir)

        if success:
            logger.info("Analysis completed successfully!")
        else:
            logger.error("Analysis failed!")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()