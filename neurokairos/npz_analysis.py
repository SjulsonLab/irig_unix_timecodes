"""
Post-hoc analysis of IRIG and PPS pulse data stored in NPZ files.

This module operates on NPZ files produced by extract_from_camera_events.py, which
contain 'starts' (rising edges) and 'ends' (falling edges) arrays indexed by camera
frame number. It provides tools for estimating sampling rate, measuring IRIG-vs-PPS
timing error, and decoding IRIG frames from the NPZ data.

Functions:
    to_pulse_lengths()   -- Converts parallel rising/falling edge arrays into
                            (duration, start_index) tuples for IRIG decoding.
    find_sample_rate()   -- Estimates the sampling rate (frames per IRIG bit) by
                            computing the median interval between consecutive pulse onsets.
    error_analysis()     -- Compares IRIG and PPS pulse onset times to measure systematic
                            timing offsets. Handles "jumps" where the IRIG-PPS difference
                            wraps around a full bit period (indicating a missed/extra pulse).
                            Reports average offset with and without outliers.
    decode_analysis()    -- Converts NPZ pulse data to IRIG bits, finds 60-bit frames,
                            decodes POSIX timestamps, and writes results to CSV.

Script-level code at the bottom auto-discovers recent NPZ files and runs decode_analysis().
"""

from typing import List, Tuple
import numpy as np
import glob
import os
import csv
from . import irig_h_gpio as irig
import threading
import matplotlib as plt

irig_files = glob.glob("data/irig_data_*.npz")
recent_file = max(irig_files, key=os.path.getmtime)
input_files = []
for irig_file in irig_files:
    if abs(os.path.getmtime(recent_file) - os.path.getmtime(irig_file)) < 10:
        input_files.append(irig_file)
# pps_filename = max(glob.glob("data/pps_data_*.npz"), key=os.path.getmtime)
print(input_files)

def to_pulse_lengths(rising_edges: np.ndarray, falling_edges: np.ndarray) -> List[Tuple[int, int]]:
    """
    Converts parallel arrays of rising and falling edge indices into
    (pulse_duration, start_index) tuples for downstream IRIG pulse classification.
    """
    return np.column_stack((falling_edges - rising_edges, rising_edges)).tolist()

def find_sample_rate(irig_filename:str):
    """
    Estimates the sampling rate (samples per IRIG bit period) from an NPZ file.

    Loads the rising edge array ('array1'), computes the median interval between
    consecutive onsets, and rounds to the nearest integer. Since IRIG-H sends one
    bit per second, this median interval equals the sampling rate in samples/sec.
    """
    irig_starts = np.load(irig_filename)['array1']
    diff = np.diff(irig_starts)

    return round(np.median(diff), 0)

def error_analysis(irig_filename:str, pps_filename:str):
    """
    Measures the sample-level timing offset between IRIG and PPS pulse onsets.

    For each pulse pair, computes (irig_onset - pps_onset). When this difference
    wraps below -0.95 * sample_rate, a "JUMP" marker is inserted and subsequent
    values are corrected by one full period. This handles cases where a PPS or
    IRIG pulse was missed, causing the two streams to slip by one period.

    Reports average offset both with and without outliers (values >= 15 samples),
    and writes the full error series to a CSV file.
    """
    sample_rate = find_sample_rate(irig_filename)
    print(f'Sample rate: {sample_rate}')

    jumps = 0

    irig_starts = np.load(irig_filename)['array1'].astype(np.int64)
    pps_starts = np.load(pps_filename)['array2'].astype(np.int64)

    min_len = min(len(irig_starts), len(pps_starts))
    irig_starts = irig_starts[:min_len]
    pps_starts = pps_starts[:min_len]

    # Calculate and correct for systematic timing offset
    
    irig_starts = irig_starts.tolist()
    pps_starts = pps_starts.tolist()

    result = []

    for irig_start, pps_start in zip(irig_starts, pps_starts):
        error = (irig_start - pps_start)
        while error < -sample_rate * (0.95 + jumps):
            jumps +=1
            result.append('JUMP')
        result.append(error + (sample_rate * jumps))

    cumulative = 0
    no_outliers = [r for r in result if type(r) != str and r < 15]
    for d in no_outliers:
        cumulative += d
    cumulative /= len(no_outliers)
    print(f'Average index delay without outliers: {cumulative}\nSeconds: {cumulative/sample_rate}')
    cumulative = 0
    with_outliers = [r for r in result if type(r) != str]
    for d in with_outliers:
        cumulative += d
    cumulative /= len(with_outliers)
    print(f'Average index delay with outliers: {cumulative}\nSeconds: {cumulative/sample_rate}')

    output_filename = f'error_indexes_{irig_filename[15:-4]}.csv'

    with open(output_filename, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(result)

def decode_analysis(irig_filename:str):
    """
    Decodes IRIG-H frames from an NPZ file and writes POSIX timestamps to CSV.

    Loads rising and falling edge arrays, converts to pulse lengths, classifies
    each pulse as an IRIG bit type, finds 60-bit frame boundaries, and decodes
    each frame to a POSIX timestamp. One timestamp per row in the output CSV.
    """
    output_filename = f'decoded_timecodes_{irig_filename[15:-4]}.csv'

    print(irig_filename)

    irig_file = np.load(irig_filename)
    for item in irig_file['array2']:
        print(item)
        
    with open(output_filename, 'w', newline='') as file:
        pulse_lengths = to_pulse_lengths(irig_file['array1'], irig_file['array2'])
        irig_bits = irig.to_irig_bits(pulse_lengths)
        decoded = irig.decode_irig_bits(irig_bits)

        csv_writer = csv.writer(file)
        csv_writer.writerows([[timestamp] for timestamp in decoded])

if __name__ == "__main__":
    # error_analysis(irig_filename='data/irig_data_000000.npz', pps_filename='data/pps_data_000000.npz')
    decode_analysis(irig_filename='data/irig_data_000000.npz')
