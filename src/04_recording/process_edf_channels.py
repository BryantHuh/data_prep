#!/usr/bin/env python3
"""
EDF Channel Processor

Processes EDF files to standardize channel names and ordering for BCI experiments.
This script helps prepare EDF files from different sources for consistent use
in BCI applications by mapping channel names to standard formats.

Features:
- Loads and processes EDF files
- Maps channel names to standard BCI format
- Reorders channels for consistent processing
- Supports multiple EDF format variations
- Detailed channel information reporting
"""

import os
import sys
import argparse
from pathlib import Path
import mne
import numpy as np

# Logging utility
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
from utils.logging_utils import setup_logging

# Setup logger
logger = setup_logging('edf_processor', log_dir='logs', level='INFO')

class EDFChannelProcessor:
    """Processes EDF files to standardize channel configuration"""

    def __init__(self, edf_path):
        self.edf_path = edf_path
        self.raw = None

        # Standard channel configuration for BCI experiments
        self.standard_channels = [
            'Fp1', 'Fp2', 'C3', 'C4', 'T5', 'T6', 'O1', 'O2',
            'F7', 'F8', 'F3', 'F4', 'T3', 'T4', 'P3', 'P4'
        ]

        # Channel mapping from various EDF formats to standard names
        self.channel_mappings = {
            # Common EDF format with dots
            'dot_format': {
                'Fp1.': 'Fp1', 'Fp2.': 'Fp2', 'F3..': 'F3', 'F4..': 'F4',
                'F7..': 'F7', 'F8..': 'F8', 'C3..': 'C3', 'C4..': 'C4',
                'T7..': 'T3', 'T8..': 'T4', 'P7..': 'T5', 'P8..': 'T6',
                'P3..': 'P3', 'P4..': 'P4', 'O1..': 'O1', 'O2..': 'O2'
            },
            # Alternative format without dots
            'clean_format': {
                'Fp1': 'Fp1', 'Fp2': 'Fp2', 'F3': 'F3', 'F4': 'F4',
                'F7': 'F7', 'F8': 'F8', 'C3': 'C3', 'C4': 'C4',
                'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6',
                'P3': 'P3', 'P4': 'P4', 'O1': 'O1', 'O2': 'O2'
            }
        }

    def load_edf_file(self):
        """Load EDF file"""
        try:
            logger.info(f"Loading EDF file: {self.edf_path}")

            if not os.path.exists(self.edf_path):
                raise FileNotFoundError(f"EDF file not found: {self.edf_path}")

            self.raw = mne.io.read_raw_edf(self.edf_path, preload=True)

            logger.info(f"Loaded EDF file with {len(self.raw.ch_names)} channels")
            logger.info(f"Sampling frequency: {self.raw.info['sfreq']} Hz")
            logger.info(f"Duration: {self.raw.times[-1]:.2f} seconds")

            return True

        except Exception as e:
            logger.error(f"Failed to load EDF file: {e}")
            return False

    def print_channel_info(self):
        """Print information about available channels"""
        if self.raw is None:
            logger.error("No EDF file loaded")
            return

        logger.info("=" * 60)
        logger.info("Channel Information")
        logger.info("=" * 60)
        logger.info(f"Total channels: {len(self.raw.ch_names)}")
        logger.info("Available channels:")
        for i, ch_name in enumerate(self.raw.ch_names):
            logger.info(f"  {i+1:2d}. {ch_name}")
        logger.info("=" * 60)

    def find_matching_channels(self):
        """Find channels that match our standard configuration"""
        if self.raw is None:
            logger.error("No EDF file loaded")
            return []

        available_channels = self.raw.ch_names
        matching_channels = []

        # Try different channel mappings
        for mapping_name, mapping in self.channel_mappings.items():
            logger.info(f"Trying {mapping_name} mapping...")

            for edf_ch, std_ch in mapping.items():
                if edf_ch in available_channels:
                    matching_channels.append(edf_ch)
                    logger.info(f"  Found: {edf_ch} -> {std_ch}")

        logger.info(f"Found {len(matching_channels)} matching channels")
        return matching_channels

    def select_and_rename_channels(self):
        """Select and rename channels to standard format"""
        if self.raw is None:
            logger.error("No EDF file loaded")
            return False

        try:
            # Find matching channels
            matching_channels = self.find_matching_channels()

            if not matching_channels:
                logger.warning("No matching channels found")
                return False

            # Pick matching channels
            logger.info("Selecting matching channels...")
            self.raw.pick_channels(matching_channels)

            # Rename channels using the first working mapping
            for mapping_name, mapping in self.channel_mappings.items():
                try:
                    # Check if this mapping works with our selected channels
                    rename_mapping = {}
                    for edf_ch, std_ch in mapping.items():
                        if edf_ch in self.raw.ch_names:
                            rename_mapping[edf_ch] = std_ch

                    if rename_mapping:
                        logger.info(f"Applying {mapping_name} mapping...")
                        self.raw.rename_channels(rename_mapping)
                        break

                except Exception as e:
                    logger.warning(f"Mapping {mapping_name} failed: {e}")
                    continue

            logger.info("Channel selection and renaming completed")
            return True

        except Exception as e:
            logger.error(f"Failed to select and rename channels: {e}")
            return False

    def reorder_channels(self):
        """Reorder channels to standard order"""
        if self.raw is None:
            logger.error("No EDF file loaded")
            return False

        try:
            logger.info("Reordering channels to standard order...")

            # Create reorder list based on available channels
            reorder_list = []
            for std_ch in self.standard_channels:
                if std_ch in self.raw.ch_names:
                    reorder_list.append(std_ch)

            if reorder_list:
                self.raw.reorder_channels(reorder_list)
                logger.info(f"Reordered {len(reorder_list)} channels")
                return True
            else:
                logger.warning("No channels to reorder")
                return False

        except Exception as e:
            logger.error(f"Failed to reorder channels: {e}")
            return False

    def save_processed_data(self, output_path):
        """Save processed data"""
        if self.raw is None:
            logger.error("No processed data to save")
            return False

        try:
            logger.info(f"Saving processed data to: {output_path}")
            self.raw.save(output_path, overwrite=True)
            logger.info("Data saved successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to save data: {e}")
            return False

    def print_final_channel_info(self):
        """Print final channel information"""
        if self.raw is None:
            logger.error("No EDF file loaded")
            return

        logger.info("=" * 60)
        logger.info("Final Channel Configuration")
        logger.info("=" * 60)
        logger.info(f"Total channels: {len(self.raw.ch_names)}")
        logger.info("Channel order:")
        for i, ch_name in enumerate(self.raw.ch_names):
            logger.info(f"  {i+1:2d}. {ch_name}")
        logger.info("=" * 60)

    def process_edf(self, output_path=None):
        """Complete EDF processing pipeline"""
        try:
            logger.info("=" * 60)
            logger.info("EDF Channel Processing Pipeline")
            logger.info("=" * 60)

            # Load EDF file
            if not self.load_edf_file():
                return False

            # Print initial channel info
            self.print_channel_info()

            # Select and rename channels
            if not self.select_and_rename_channels():
                return False

            # Reorder channels
            if not self.reorder_channels():
                return False

            # Print final channel info
            self.print_final_channel_info()

            # Save processed data if output path provided
            if output_path:
                if not self.save_processed_data(output_path):
                    return False

            logger.info("EDF processing completed successfully!")
            return True

        except Exception as e:
            logger.error(f"EDF processing failed: {e}")
            return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Process EDF file channels')
    parser.add_argument('edf_path', type=str, help='Path to EDF file')
    parser.add_argument('--output', type=str, default=None, help='Output path for processed data')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Set log level
    if args.verbose:
        logger.setLevel('DEBUG')

    try:
        # Create processor
        processor = EDFChannelProcessor(args.edf_path)

        # Process EDF file
        success = processor.process_edf(args.output)

        if success:
            logger.info("Processing completed successfully!")
        else:
            logger.error("Processing failed!")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()