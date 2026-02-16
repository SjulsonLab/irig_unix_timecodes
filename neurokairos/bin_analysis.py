"""
Analysis tools for bit-packed binary IRIG and PPS recordings.

This module works with bit-packed binary files where each byte contains 8 samples
(one bit per sample). These files are typically produced by earlier extraction steps
that threshold raw DAT recordings and pack the boolean results.

Two main analysis modes are supported:

    Error analysis (PPS vs IRIG timing offset):
        Compares the IRIG signal against a PPS (pulse-per-second) reference signal
        to measure how many samples the IRIG pulse onset lags behind the PPS rising
        edge. This quantifies the systematic timing offset of the IRIG sender.

    Decode analysis (IRIG frame decoding):
        Unpacks the IRIG binary, classifies pulse widths, finds 60-bit frames, and
        decodes each frame to a POSIX timestamp. Results are written to CSV.

Functions:
    byte_unpack_generator() -- Memory-efficient generator that reads a bit-packed binary
                               file in chunks and yields individual boolean samples.
    find_errors()           -- Iterates IRIG and PPS bit streams in lockstep, measuring
                               the sample delay between each PPS rising edge and the
                               corresponding IRIG pulse onset.
    error_analysis()        -- High-level wrapper: unpacks both files, runs find_errors(),
                               writes results to CSV.
    decode_analysis()       -- High-level wrapper: unpacks IRIG file, decodes frames via
                               irig_h_gpio, writes decoded timestamps to CSV.
    get_pulse_lengths()     -- Extracts raw pulse lengths from the IRIG binary and writes
                               them to CSV for manual inspection.

Script-level code at the bottom runs error_analysis() on the default file paths.
"""

from typing import List, Literal, Generator
import csv
import os
from . import irig_h_gpio as irig
import numpy as np

irig_file = "data/binary/irig_binary2.bin"
pps_file = "data/binary/pps_binary2.bin"

error_filename = 'data/indexes_of_error.csv'
decoded_filename = 'data/decoded_timecodes.csv'
pulselength_filename = 'data/pulse_lengths.csv'

def byte_unpack_generator(file_path: str, chunk_size: int = 1024*1024, total_samples: int = None) -> Generator[bool, None, None]:
    """
    Generator that unpacks bit-packed binary files created by extract_from_dat.py
    
    Args:
        file_path: Path to the bit-packed binary file
        chunk_size: Number of bytes to read per chunk (default 1MB)
        total_samples: Total number of valid samples to yield (trims padding bits)
    
    Yields:
        bool: Individual bits as boolean values (True for 1, False for 0)
    """
    samples_yielded = 0
    
    with open(file_path, 'rb') as f:
        while True:
            # Stop if we've reached the total samples limit
            if total_samples is not None and samples_yielded >= total_samples:
                break
                
            # Read chunk of packed bytes
            packed_chunk = f.read(chunk_size)
            if len(packed_chunk) == 0:
                break
            
            # Convert bytes to numpy array
            packed_data = np.frombuffer(packed_chunk, dtype=np.uint8)
            
            # Unpack bits from bytes (8 bits per byte)
            unpacked_bits = np.unpackbits(packed_data)
            
            # Yield each bit as a boolean, respecting total_samples limit
            for bit in unpacked_bits:
                if total_samples is not None and samples_yielded >= total_samples:
                    return
                yield bool(bit)
                samples_yielded += 1

def find_errors(irig: Generator[bool, None, None], pps: Generator[bool, None, None]) -> List[int]:
    """
    Measures the sample-level timing offset between PPS and IRIG pulse onsets.

    Iterates both bit streams in lockstep. While PPS is HIGH and IRIG is LOW,
    counts samples (the IRIG pulse hasn't started yet). When IRIG goes HIGH,
    records the accumulated count as the delay for that pulse. The resulting
    list contains one delay value per PPS pulse.
    """
    errors_index_length = []
    tracking_length = 0
    processed_bits = 0
    for irig_bit, pps_bit in zip(irig, pps):
        if pps_bit == 1:
            if irig_bit == 0:
                tracking_length += 1
            else:
                errors_index_length.append(tracking_length)
                tracking_length = 0
    
        processed_bits += 1
        if processed_bits % 24000000 == 0:  # Progress every ~8MB of bits
            print(f'Processed {processed_bits} bits, found {len(errors_index_length)} errors')

    print(f'Processing complete. Total bits: {processed_bits}, errors found: {len(errors_index_length)}')
    return errors_index_length

def error_analysis(irig_gen=None, pps_gen=None):
    """
    Runs PPS-vs-IRIG timing error analysis and writes results to CSV.

    If generators are not provided, unpacks the default binary files.
    Output is a single-row CSV where each value is the sample delay for one PPS pulse.
    """
    if irig_gen is None:
        print('Unpacking irig bytes into bits.')
        irig_gen = byte_unpack_generator(irig_file)

    if pps_gen is None:
        print('Unpacking pps bytes into bits.')
        pps_gen = byte_unpack_generator(pps_file)

    print('Unpacking done. Calculating errors...')
    # This means, the amount of samples taken where the PPS pulse has started and the IRIG timecode has not.
    errors = find_errors(irig_gen, pps_gen)

    # errors_seconds = [error * 1/3000 for error in errors]

    print('Error calculations done. Writing to file...')
    with open(error_filename, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(errors)

    print('File writing done. Enjoy!')

def decode_analysis(irig_gen=None):
    """
    Decodes IRIG-H frames from a bit-packed binary file and writes timestamps to CSV.

    Unpacks the binary, classifies pulses into IRIG bits, finds 60-bit frame boundaries,
    decodes each frame to a POSIX timestamp, and writes one timestamp per row to CSV.
    """
    if irig_gen is None:
        print('Unpacking irig bytes into bits.')
        irig_gen = byte_unpack_generator(irig_file)
    
    print('Unpacking done. Decoding IRIG-H timecodes.')
    decoded = irig.decode_irig_bits(irig.to_irig_bits(irig_gen))

    print('Decoding done. Writing to file...')
    with open(decoded_filename, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        for tuple in decoded:
            csv_writer.writerow(tuple)

    print('File writing done. Enjoy!')

def get_pulse_lengths():
    """
    Extracts raw pulse lengths from the IRIG binary file and writes to CSV.

    Each row in the output CSV contains (pulse_length_in_samples, start_sample_index).
    Useful for inspecting pulse width distributions and diagnosing threshold issues.
    """
    print('Unpacking irig bytes into bits.')
    irig_gen = byte_unpack_generator(irig_file)

    print('Unpacking done. Decoding IRIG-H timecodes.')
    pulse_lengths = irig.find_pulse_length(irig_gen)

    with open(pulselength_filename, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        for pulse_length in pulse_lengths:
            csv_writer.writerow(pulse_length)

irig_size = os.path.getsize(irig_file)
pps_size = os.path.getsize(pps_file)
min_size = min(irig_size, pps_size)

print(f'File sizes - IRIG: {irig_size}, PPS: {pps_size}, processing: {min_size} bytes')

# get_pulse_lengths()
# for bit in matlab_style_unpack_generator(irig_file):
#     print(bit)

irig_gen = byte_unpack_generator(irig_file)
error_analysis(irig_gen=irig_gen)

