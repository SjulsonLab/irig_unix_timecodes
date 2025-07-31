
from typing import List, Literal, Generator
import csv
import os
import irig_h_gpio as irig
import numpy as np

irig_file = "data/binary/irig_binary.bin"
pps_file = "data/binary/pps_binary.bin"

error_filename = 'data/indexes_of_error.csv'
decoded_filename = 'data/decoded_timecodes.csv'

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

def error_analysis():
    print('Unpacking irig bytes into bits.')
    irig_gen = byte_unpack_generator(irig_file)
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

def decode_analysis():
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


irig_size = os.path.getsize(irig_file)
pps_size = os.path.getsize(pps_file)
min_size = min(irig_size, pps_size)

print(f'File sizes - IRIG: {irig_size}, PPS: {pps_size}, processing: {min_size} bytes')

# decode_analysis()
for bit in byte_unpack_generator(irig_file):
    print(bit)

