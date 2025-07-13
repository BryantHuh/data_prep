#!/usr/bin/env python3
"""
MOABB Data Streamer

Streams MOABB BNCI2014_001 dataset over LSL for real-time BCI experiments.
This script creates a simulated real-time EEG stream from pre-recorded data,
useful for testing and development of real-time BCI applications.

Features:
- Streams pre-recorded EEG data at configurable sampling rates
- Includes marker streams for validation
- Configurable subject selection and duration
- Simulates real-time data flow for testing
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
from pylsl import StreamInfo, StreamOutlet, cf_float32, cf_string
from braindecode.datasets import MOABBDataset

# Logging utility
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
from utils.logging_utils import setup_logging

# Setup logger
logger = setup_logging('moabb_streamer', log_dir='logs', level='INFO')

class MOABBStreamer:
    """Streams MOABB dataset over LSL"""

    def __init__(self, subject_id=3, sfreq=125, stream_duration=None):
        self.subject_id = subject_id
        self.sfreq = sfreq
        self.stream_duration = stream_duration

        # Channel configuration
        self.included_channels = [
            'C3', 'C4', 'Cz',
            'FC1', 'FC2', 'FCz',
            'CP1', 'CP2', 'CPz',
            'P1', 'P2', 'Pz',
            'C1', 'C2',
            'CP3', 'CP4'
        ]

        # LSL outlets
        self.eeg_outlet = None
        self.marker_outlet = None

        # Data
        self.raw_data = None
        self.annotations = None
        self.marker_events = []

        logger.info(f"Initialized MOABB streamer for subject {subject_id}")

    def load_moabb_data(self):
        """Load MOABB dataset"""
        try:
            logger.info(f"Loading MOABB dataset for subject {self.subject_id}...")
            dataset = MOABBDataset("BNCI2014_001", subject_ids=[self.subject_id])

            # Get raw data
            raw = getattr(dataset.datasets[0], '_raw', None)
            if raw is None:
                raw = getattr(dataset.datasets[0], 'raw', None)
            if raw is None:
                raise AttributeError('Could not find raw or _raw attribute in dataset.datasets[0]')

            # Select channels
            raw.pick_channels(self.included_channels)

            # Resample if needed
            if self.sfreq is not None and raw.info['sfreq'] != self.sfreq:
                logger.info(f"Resampling from {raw.info['sfreq']} Hz to {self.sfreq} Hz")
                raw.resample(self.sfreq)

            self.raw_data = raw
            self.annotations = raw.annotations

            logger.info(f"Loaded data: {raw.get_data().shape[1]} samples at {raw.info['sfreq']} Hz")
            logger.info(f"Number of annotations: {len(self.annotations)}")

        except Exception as e:
            logger.error(f"Failed to load MOABB data: {e}")
            raise

    def setup_lsl_streams(self):
        """Setup LSL streams for EEG and markers"""
        try:
            # EEG stream
            eeg_info = StreamInfo(
                name='MOABB_EEG_RAW',
                type='EEG',
                channel_count=len(self.included_channels),
                nominal_srate=self.raw_data.info['sfreq'],
                channel_format=cf_float32,
                source_id=f'moabb_subj{self.subject_id}_eeg_raw'
            )

            # Add channel labels
            chns = eeg_info.desc().append_child("channels")
            for ch in self.included_channels:
                chns.append_child("channel").append_child_value("label", ch)

            self.eeg_outlet = StreamOutlet(eeg_info, chunk_size=1, max_buffered=360)
            logger.info("Created EEG LSL stream")

            # Marker stream
            marker_info = StreamInfo(
                name='MOABB_Markers',
                type='Markers',
                channel_count=1,
                nominal_srate=0,
                channel_format=cf_string,
                source_id=f'moabb_subj{self.subject_id}_markers'
            )
            self.marker_outlet = StreamOutlet(marker_info)
            logger.info("Created marker LSL stream")

        except Exception as e:
            logger.error(f"Failed to setup LSL streams: {e}")
            raise

    def prepare_marker_events(self):
        """Prepare marker events from annotations"""
        try:
            # Start marker
            self.marker_events = [(0, 'start')]

            # Add annotation markers
            for onset, desc in zip(self.annotations.onset, self.annotations.description):
                sample_idx = int(onset * self.raw_data.info['sfreq'])
                self.marker_events.append((sample_idx, desc))

            # Sort by sample index
            self.marker_events.sort()

            logger.info(f"Prepared {len(self.marker_events)} marker events")

        except Exception as e:
            logger.error(f"Failed to prepare marker events: {e}")
            raise

    def stream_data(self):
        """Stream EEG data and markers"""
        try:
            # Get data
            data = self.raw_data.get_data(picks=self.included_channels)
            total_samples = data.shape[1]

            # Determine stop sample
            if self.stream_duration:
                stop_sample = min(total_samples, int(self.stream_duration * self.raw_data.info['sfreq']))
                logger.info(f"Streaming for {self.stream_duration} seconds ({stop_sample} samples)")
            else:
                stop_sample = total_samples
                logger.info(f"Streaming full session ({stop_sample} samples)")

            # Stream data
            marker_idx = 0
            start_time = time.time()

            logger.info("Starting data stream...")

            for i in range(stop_sample):
                # Send EEG sample
                sample = data[:, i].astype(np.float32)
                self.eeg_outlet.push_sample(sample.tolist())

                # Check for markers
                while marker_idx < len(self.marker_events) and self.marker_events[marker_idx][0] == i:
                    desc = self.marker_events[marker_idx][1]
                    self.marker_outlet.push_sample([desc])
                    logger.info(f"Sent marker: {desc} at sample {i}")
                    marker_idx += 1

                # Simulate real-time
                time.sleep(1.0 / self.raw_data.info['sfreq'])

                # Progress update every 5 seconds
                if i % (int(self.raw_data.info['sfreq']) * 5) == 0:
                    elapsed = time.time() - start_time
                    logger.info(f"Streamed {i/self.raw_data.info['sfreq']:.1f} seconds...")

            logger.info("Data streaming completed")

        except Exception as e:
            logger.error(f"Failed to stream data: {e}")
            raise

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Stream MOABB dataset over LSL')
    parser.add_argument('--subject-id', type=int, default=3, help='Subject ID to stream')
    parser.add_argument('--sfreq', type=int, default=125, help='Sampling frequency')
    parser.add_argument('--duration', type=float, default=None, help='Stream duration in seconds (None for full session)')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("MOABB Data Streamer")
    logger.info("=" * 60)

    try:
        # Create streamer
        streamer = MOABBStreamer(
            subject_id=args.subject_id,
            sfreq=args.sfreq,
            stream_duration=args.duration
        )

        # Load data
        streamer.load_moabb_data()

        # Setup LSL streams
        streamer.setup_lsl_streams()

        # Prepare markers
        streamer.prepare_marker_events()

        # Start streaming
        streamer.stream_data()

    except KeyboardInterrupt:
        logger.info("Streaming interrupted by user")
    except Exception as e:
        logger.error(f"Streaming failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()