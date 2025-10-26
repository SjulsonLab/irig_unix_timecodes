import numpy as np
import sys
import os
from datetime import datetime
import gc
import argparse
from typing import List, Tuple, Optional
from irig_h_gpio import identify_pulse_length, decode_irig_bits, irig_h_to_posix, to_irig_bits


class IRIGExtractor:
    """
    Extracts IRIG timecode pulses from DAT files and decodes them into unix timestamps.
    """

    def __init__(self, input_file: str, irig_threshold: int = 2500,
                 irig_channel: int = 32, total_channels: int = 40,
                 chunk_size: int = 2500000):
        """
        Initialize the IRIG extractor.

        Args:
            input_file: Path to the .dat file
            irig_threshold: Threshold for IRIG signal detection
            irig_channel: Channel index for IRIG signal
            total_channels: Total number of channels in DAT file
            chunk_size: Number of samples per chunk for processing
        """
        self.input_file = input_file
        self.irig_threshold = irig_threshold
        self.irig_channel = irig_channel
        self.total_channels = total_channels
        self.chunk_size = chunk_size

        # Data storage
        self.pulses = None  # Will be structured array
        self.estimated_sample_rate = None
        self.discontinuities = []

    def extract_pulses_from_dat(self) -> np.ndarray:
        """
        Extract pulses from the DAT file.
        Returns a list of complete pulses with (on_sample, off_sample) as uint64.
        """
        print(f"Processing {self.input_file}")

        rising_edges_list = []
        falling_edges_list = []

        rising_edges = np.empty(0, dtype=np.uint64)
        falling_edges = np.empty(0, dtype=np.uint64)

        with open(self.input_file, 'rb') as f:
            chunk_num = 0
            last_bit = False

            while True:
                try:
                    chunk_starting_index = chunk_num * self.chunk_size
                    bytes_per_sample = 2 * self.total_channels
                    chunk_bytes = self.chunk_size * bytes_per_sample

                    # Read chunk
                    raw_chunk = f.read(chunk_bytes)
                    if len(raw_chunk) == 0:
                        print("Reached end of file")
                        break

                    # Convert to int16 array
                    int16_data = np.frombuffer(raw_chunk, dtype=np.int16)
                    samples_in_chunk = len(int16_data) // self.total_channels

                    if samples_in_chunk == 0:
                        print("No complete samples in chunk")
                        break

                    # Reshape and extract IRIG channel
                    int16_data = int16_data[:samples_in_chunk * self.total_channels]
                    chunk_data = int16_data.reshape(samples_in_chunk, self.total_channels)
                    irig_raw = chunk_data[:, self.irig_channel]

                    # Convert to binary using threshold
                    irig_binary = (irig_raw > self.irig_threshold).astype(np.bool_)

                    # Create array with previous state prepended for diff calculation
                    irig_with_prev = np.concatenate(([last_bit], irig_binary))

                    # Find where changes occur using diff
                    irig_diffs = np.diff(irig_with_prev.astype(np.int8))

                    # Find rising edges (0->1, diff = 1) and falling edges (1->0, diff = -1)
                    rising_index = np.where(irig_diffs == 1)[0] + chunk_starting_index
                    falling_index = np.where(irig_diffs == -1)[0] + chunk_starting_index

                    # Extend the edge lists
                    rising_edges_list.extend(rising_index)
                    falling_edges_list.extend(falling_index)

                    # Update last state
                    last_bit = irig_binary[-1] if len(irig_binary) > 0 else last_bit

                    # Progress reporting
                    if chunk_num % 20 == 0:
                        print(f"Chunk {chunk_num}: {chunk_starting_index + self.chunk_size:,} samples processed")
                        sys.stdout.flush()
                    elif chunk_num % 5 == 0:
                        print(f"Chunk {chunk_num}... ", end="", flush=True)

                    # Periodically flush lists to numpy arrays
                    if chunk_num % 40 == 0:
                        print("Flushing python lists to numpy arrays.")
                        rising_edges = np.concatenate((rising_edges, np.fromiter(rising_edges_list, dtype=np.uint64)))
                        falling_edges = np.concatenate((falling_edges, np.fromiter(falling_edges_list, dtype=np.uint64)))
                        rising_edges_list.clear()
                        falling_edges_list.clear()

                    if chunk_num % 10 == 0:
                        gc.collect()

                except Exception as e:
                    print(f"Error processing chunk {chunk_num}: {e}")
                    print(f"Chunk size was: {len(raw_chunk)} bytes")
                    break
                finally:
                    chunk_num += 1

        # Final flush
        if rising_edges_list or falling_edges_list:
            rising_edges = np.concatenate((rising_edges, np.fromiter(rising_edges_list, dtype=np.uint64)))
            falling_edges = np.concatenate((falling_edges, np.fromiter(falling_edges_list, dtype=np.uint64)))

        print(f"Found {len(rising_edges)} rising edges and {len(falling_edges)} falling edges")

        # Build complete pulse list
        pulses = self._build_complete_pulses(rising_edges, falling_edges)

        return pulses

    def _build_complete_pulses(self, rising_edges: np.ndarray, falling_edges: np.ndarray) -> np.ndarray:
        """
        Build a list of complete pulses where both onset and offset were recorded.
        Returns array of (on_sample, off_sample) tuples.
        """
        print("Building complete pulse list...")

        complete_pulses = []
        rising_idx = 0
        falling_idx = 0

        # Match each rising edge with the next falling edge
        while rising_idx < len(rising_edges) and falling_idx < len(falling_edges):
            rise = rising_edges[rising_idx]

            # Find the next falling edge after this rising edge
            while falling_idx < len(falling_edges) and falling_edges[falling_idx] <= rise:
                falling_idx += 1

            if falling_idx < len(falling_edges):
                fall = falling_edges[falling_idx]
                complete_pulses.append((rise, fall))
                rising_idx += 1
                falling_idx += 1
            else:
                break

        print(f"Found {len(complete_pulses)} complete pulses")

        # Convert to structured array
        pulse_array = np.array(complete_pulses, dtype=[('on_sample', np.uint64), ('off_sample', np.uint64)])

        return pulse_array

    def estimate_sampling_rate(self) -> float:
        """
        Estimate the sampling rate by calculating the median duration between pulse onsets.
        Returns estimated samples per second.
        """
        if self.pulses is None or len(self.pulses) < 2:
            print("Not enough pulses to estimate sampling rate")
            return None

        print("Estimating sampling rate...")

        # Get onset samples
        onsets = self.pulses['on_sample']

        # Calculate intervals between consecutive onsets
        intervals = np.diff(onsets)

        # Remove outliers (e.g., > 2 seconds worth of samples)
        valid_intervals = intervals[intervals < 100000]

        if len(valid_intervals) == 0:
            print("No valid intervals found")
            return None

        # Median interval should be approximately 1 second
        median_interval = np.median(valid_intervals)

        self.estimated_sample_rate = float(median_interval)

        print(f"Estimated sampling rate: {self.estimated_sample_rate:.2f} samples/sec")
        print(f"Mean interval: {np.mean(valid_intervals):.2f}, Std: {np.std(valid_intervals):.2f}")

        return self.estimated_sample_rate

    def classify_and_decode(self):
        """
        Classify each pulse type and decode IRIG frames into unix timestamps.
        Updates self.pulses with pulse_type and unix_time fields.
        """
        if self.pulses is None:
            print("No pulses to classify. Run extract_pulses_from_dat() first.")
            return

        print("Classifying pulses and decoding IRIG frames...")

        # Create new structured array with additional fields
        n_pulses = len(self.pulses)
        classified_pulses = np.zeros(n_pulses, dtype=[
            ('on_sample', np.uint64),
            ('off_sample', np.uint64),
            ('pulse_type', np.int8),
            ('unix_time', np.float64),
            ('frame_id', np.int32)
        ])

        # Copy existing data
        classified_pulses['on_sample'] = self.pulses['on_sample']
        classified_pulses['off_sample'] = self.pulses['off_sample']

        # Initialize unix_time as NaN and frame_id as -1
        classified_pulses['unix_time'] = np.nan
        classified_pulses['frame_id'] = -1

        # Calculate pulse durations
        durations = self.pulses['off_sample'] - self.pulses['on_sample']

        # Classify each pulse using identify_pulse_length from irig_h_gpio.py
        for i, duration in enumerate(durations):
            pulse_type_raw = identify_pulse_length(int(duration))

            # Map to int8: 'P'->2, True->1, False->0, None->-1
            if pulse_type_raw == 'P':
                classified_pulses['pulse_type'][i] = 2
            elif pulse_type_raw == True:
                classified_pulses['pulse_type'][i] = 1
            elif pulse_type_raw == False:
                classified_pulses['pulse_type'][i] = 0
            else:
                classified_pulses['pulse_type'][i] = -1

        print(f"Pulse type distribution:")
        print(f"  Short (0): {np.sum(classified_pulses['pulse_type'] == 0)}")
        print(f"  Long (1): {np.sum(classified_pulses['pulse_type'] == 1)}")
        print(f"  Marker (2): {np.sum(classified_pulses['pulse_type'] == 2)}")
        print(f"  Error (-1): {np.sum(classified_pulses['pulse_type'] == -1)}")

        # Find frames and decode
        self._decode_frames(classified_pulses)

        self.pulses = classified_pulses

    def _decode_frames(self, pulses: np.ndarray):
        """
        Find 60-bit IRIG frames and decode them into unix timestamps.
        """
        print("Finding and decoding IRIG frames...")

        # Convert pulse types back to IRIG bit format for decode_irig_bits
        irig_bits = []
        for i, pulse in enumerate(pulses):
            ptype = pulse['pulse_type']
            onset_time = float(pulse['on_sample'])

            if ptype == 2:
                irig_bits.append(('P', onset_time))
            elif ptype == 1:
                irig_bits.append((True, onset_time))
            elif ptype == 0:
                irig_bits.append((False, onset_time))
            # Skip error pulses (type -1)

        if len(irig_bits) < 60:
            print("Not enough valid pulses to decode frames")
            return

        # Use decode_irig_bits from irig_h_gpio.py to find frames
        try:
            decoded_frames = decode_irig_bits(irig_bits)
            print(f"Decoded {len(decoded_frames)} frames")
        except ValueError as e:
            print(f"Error decoding frames: {e}")
            return

        # Now map the decoded frames back to pulse indices
        # This is tricky because decode_irig_bits works with the filtered irig_bits list
        # We need to track which pulse index corresponds to each irig_bit

        # Create a mapping from irig_bits index to pulse index
        irig_to_pulse_idx = []
        irig_idx = 0
        for pulse_idx, pulse in enumerate(pulses):
            if pulse['pulse_type'] != -1:  # Valid pulse
                irig_to_pulse_idx.append(pulse_idx)

        # Find frame boundaries in the original irig_bits
        frame_id = 0
        i = 0

        # Find first frame start (two consecutive 'P' markers)
        while i < len(irig_bits) - 1:
            if irig_bits[i][0] == 'P' and irig_bits[i+1][0] == 'P':
                frame_start = i + 1
                break
            i += 1
        else:
            print("No frame start found")
            return

        # Process frames
        current_pos = frame_start
        for frame_time, measurement_time in decoded_frames:
            if current_pos + 60 > len(irig_bits):
                break

            # Mark pulses in this frame
            for offset in range(60):
                irig_idx = current_pos + offset
                if irig_idx < len(irig_to_pulse_idx):
                    pulse_idx = irig_to_pulse_idx[irig_idx]
                    pulses['frame_id'][pulse_idx] = frame_id
                    pulses['unix_time'][pulse_idx] = frame_time + offset

            frame_id += 1

            # Find next frame (next pair of 'P' markers)
            search_start = current_pos + 60
            found_next = False
            for i in range(search_start, min(search_start + 120, len(irig_bits) - 1)):
                if irig_bits[i][0] == 'P' and irig_bits[i+1][0] == 'P':
                    current_pos = i + 1
                    found_next = True
                    break

            if not found_next:
                break

        print(f"Assigned timestamps to {frame_id} frames")

    def detect_discontinuities(self):
        """
        Detect potential issues like concatenated files, dropped frames, or time jumps.
        """
        if self.pulses is None:
            print("No pulses to analyze")
            return

        print("Detecting discontinuities...")

        self.discontinuities = []

        # Check for large gaps in pulse onsets
        onsets = self.pulses['on_sample']
        intervals = np.diff(onsets)

        # Expected interval is approximately the sampling rate
        if self.estimated_sample_rate:
            expected = self.estimated_sample_rate
            threshold = expected * 2  # Flag gaps > 2 seconds

            large_gaps = np.where(intervals > threshold)[0]

            for idx in large_gaps:
                gap_samples = intervals[idx]
                gap_seconds = gap_samples / self.estimated_sample_rate
                self.discontinuities.append({
                    'type': 'large_gap',
                    'pulse_index': idx,
                    'sample': onsets[idx],
                    'gap_samples': int(gap_samples),
                    'gap_seconds': gap_seconds
                })
                print(f"  Large gap at pulse {idx}: {gap_seconds:.2f} seconds")

        # Check for timestamp discontinuities
        unix_times = self.pulses['unix_time']
        valid_times = unix_times[~np.isnan(unix_times)]

        if len(valid_times) > 1:
            time_diffs = np.diff(valid_times)

            # Flag jumps > 2 seconds (should be 1 second between pulses)
            large_time_jumps = np.where(np.abs(time_diffs - 1.0) > 1.0)[0]

            for idx in large_time_jumps:
                jump = time_diffs[idx]
                self.discontinuities.append({
                    'type': 'time_jump',
                    'pulse_index': idx,
                    'expected': 1.0,
                    'actual': jump,
                    'difference': jump - 1.0
                })
                print(f"  Time jump at pulse {idx}: {jump:.2f} seconds (expected 1.0)")

        # Check for dropped frames
        frame_ids = self.pulses['frame_id']
        valid_frames = frame_ids[frame_ids >= 0]

        if len(valid_frames) > 1:
            frame_diffs = np.diff(np.unique(valid_frames))
            dropped = frame_diffs[frame_diffs > 1]

            if len(dropped) > 0:
                print(f"  Warning: {len(dropped)} potential dropped frames detected")
                self.discontinuities.append({
                    'type': 'dropped_frames',
                    'count': len(dropped)
                })

        if len(self.discontinuities) == 0:
            print("No discontinuities detected")
        else:
            print(f"Found {len(self.discontinuities)} discontinuities")

    def save_to_npz(self, output_file: Optional[str] = None):
        """
        Save the pulse data to an NPZ file with structured array.
        """
        if self.pulses is None:
            print("No pulses to save")
            return

        if output_file is None:
            # Auto-generate filename from input file's modification date
            if os.path.exists(self.input_file):
                mod_time = os.path.getmtime(self.input_file)
                date_str = datetime.fromtimestamp(mod_time).strftime('%Y%m%d_%H%M%S')
                output_file = f'irig_pulses_{date_str}.npz'
            else:
                output_file = 'irig_pulses.npz'

        print(f"Saving results to {output_file}")

        # Save pulses and metadata
        np.savez_compressed(
            output_file,
            pulses=self.pulses,
            estimated_sample_rate=self.estimated_sample_rate,
            discontinuities=np.array(self.discontinuities) if self.discontinuities else np.array([])
        )

        print(f"Saved {len(self.pulses)} pulses to {output_file}")

    def process(self, output_file: Optional[str] = None):
        """
        Run the complete extraction and decoding pipeline.
        """
        print("="*60)
        print("IRIG Extractor - Processing Pipeline")
        print("="*60)

        # Step 1: Extract pulses
        self.pulses = self.extract_pulses_from_dat()

        # Step 2: Estimate sampling rate
        self.estimate_sampling_rate()

        # Step 3: Classify and decode
        self.classify_and_decode()

        # Step 4: Detect discontinuities
        self.detect_discontinuities()

        # Step 5: Save results
        self.save_to_npz(output_file)

        print("="*60)
        print("Processing complete!")
        print("="*60)


def main():
    """
    Command-line interface for IRIG extraction.
    """
    parser = argparse.ArgumentParser(
        description='Extract and decode IRIG timecodes from DAT files'
    )
    parser.add_argument(
        'input_file',
        help='Path to input .dat file'
    )
    parser.add_argument(
        '-o', '--output',
        help='Path to output .npz file (auto-generated if not specified)'
    )
    parser.add_argument(
        '-t', '--threshold',
        type=int,
        default=2500,
        help='IRIG signal threshold (default: 2500)'
    )
    parser.add_argument(
        '-c', '--channel',
        type=int,
        default=32,
        help='IRIG channel index (default: 32)'
    )
    parser.add_argument(
        '--total-channels',
        type=int,
        default=40,
        help='Total channels in DAT file (default: 40)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=2500000,
        help='Samples per chunk (default: 2500000)'
    )

    args = parser.parse_args()

    # Verify input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist")
        sys.exit(1)

    # Create extractor and run
    extractor = IRIGExtractor(
        input_file=args.input_file,
        irig_threshold=args.threshold,
        irig_channel=args.channel,
        total_channels=args.total_channels,
        chunk_size=args.chunk_size
    )

    extractor.process(output_file=args.output)


if __name__ == '__main__':
    main()
