#!/usr/bin/env python3
"""
LSL Stream Tester

Simple script to test and list available LSL streams.
Useful for debugging OpenBCI GUI connection issues and verifying
LSL stream availability before running real-time BCI applications.

Features:
- Lists all available LSL streams with details
- Tests connection to specific streams
- Common stream name and type testing
- Connection validation for EEG and marker streams
"""

import time
from pylsl import resolve_streams, StreamInlet

def list_available_streams():
    """List all available LSL streams"""
    print("=" * 60)
    print("Available LSL Streams")
    print("=" * 60)

    streams = resolve_streams()

    if not streams:
        print("No LSL streams found!")
        return

    for i, stream in enumerate(streams):
        print(f"\nStream {i+1}:")
        print(f"  Name: {stream.name()}")
        print(f"  Type: {stream.type()}")
        print(f"  Channels: {stream.channel_count()}")
        print(f"  Sampling Rate: {stream.nominal_srate()}")
        print(f"  Source ID: {stream.source_id()}")

def test_stream_connection(stream_name=None, stream_type=None):
    """Test connection to a specific stream"""
    print("=" * 60)
    print("Testing Stream Connection")
    print("=" * 60)

    try:
        if stream_name:
            streams = resolve_streams(name=stream_name, timeout=5)
            print(f"Looking for stream with name: {stream_name}")
        elif stream_type:
            streams = resolve_streams(type=stream_type, timeout=5)
            print(f"Looking for streams with type: {stream_type}")
        else:
            streams = resolve_streams(type='EEG', timeout=5)
            print("Looking for EEG streams")

        if not streams:
            print("No streams found!")
            return False

        # Test first stream
        stream = streams[0]
        print(f"Testing connection to: {stream.name()}")

        inlet = StreamInlet(stream)

        # Try to get a sample
        print("Attempting to receive sample...")
        sample, timestamp = inlet.pull_sample(timeout=2.0)

        if sample is not None:
            print(f"✅ Success! Received sample with {len(sample)} channels")
            print(f"Sample data: {sample[:5]}...")  # Show first 5 values
            print(f"Timestamp: {timestamp}")
            return True
        else:
            print("❌ No sample received within timeout")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main function"""
    print("LSL Stream Tester")
    print("Use this to debug OpenBCI GUI connections")
    print()

    # List all available streams
    list_available_streams()

    print("\n" + "=" * 60)
    print("Testing Common Stream Types")
    print("=" * 60)

    # Test common stream types
    test_stream_connection(stream_type='EEG')
    test_stream_connection(stream_type='Markers')

    print("\n" + "=" * 60)
    print("Testing Common Stream Names")
    print("=" * 60)

    # Test common stream names
    common_names = [
        'OpenBCI_EEG',
        'EEG_RAW',
        'OpenBCI',
        'EEG',
        'MOABB_EEG_RAW'
    ]

    for name in common_names:
        test_stream_connection(stream_name=name)

if __name__ == "__main__":
    main()