import numpy as np
import sys
import os
from datetime import datetime
import gc

total_channels = 40
adc1_index = 32  # IRIG
adc2_index = 33  # PPS
sample_rate = 30000

# Thresholds based on your debug output
irig_threshold = 2500  # Adjust if needed
pps_threshold = 2500   # Adjust if needed

# File paths
input_file = 'continuous.dat'  # Change to your .dat file path

# Generate output filenames with input file's modification date
if os.path.exists(input_file):
    # Get file modification time
    mod_time = os.path.getmtime(input_file)
    date_str = datetime.fromtimestamp(mod_time).strftime('%Y%m%d_%H%M%S')
    
    # Create output filenames with date
    pps_output = f'pps_data_{date_str}.npz'
    irig_output = f'irig_data_{date_str}.npz'
else:
    # Fallback if file doesn't exist
    raise FileExistsError(f'The input file \'{input_file}\' does not exist. Are you in the right directory?')

chunk_size = 2500000  # Samples per chunk (reduced from 1M)

print(f"Processing {input_file}")
print(f"Output files will be: {pps_output}, {irig_output}")

with open(input_file, 'rb') as f:
    chunk_num = 0

    last_irig_bit = False
    last_pps_bit = False

    irig_rising_edges_list = []
    irig_falling_edges_list = []
    
    pps_rising_edges_list = []
    pps_falling_edges_list = []

    irig_rising_edges = np.empty(0, dtype=np.int64)
    irig_falling_edges = np.empty(0, dtype=np.int64)
    pps_rising_edges = np.empty(0, dtype=np.int64)
    pps_falling_edges = np.empty(0, dtype=np.int64)

    
    while True:
        try:
            chunk_starting_index = chunk_num * chunk_size
            # Calculate bytes to read
            bytes_per_sample = 2 * total_channels
            chunk_bytes = chunk_size * bytes_per_sample
            
            # Read chunk
            raw_chunk = f.read(chunk_bytes)
            if len(raw_chunk) == 0:
                print("Reached end of file")
                break
                
            # Convert to int16 array
            int16_data = np.frombuffer(raw_chunk, dtype=np.int16)
            samples_in_chunk = len(int16_data) // total_channels
            
            if samples_in_chunk == 0:
                print("No complete samples in chunk")
                break
                
            # Reshape and extract channels
            int16_data = int16_data[:samples_in_chunk * total_channels]
            chunk_data = int16_data.reshape(samples_in_chunk, total_channels)
            
            # Extract ADC channels
            irig_raw = chunk_data[:, adc1_index]  # IRIG
            pps_raw = chunk_data[:, adc2_index]   # PPS
            
            # Convert to binary using thresholds
            irig_binary = (irig_raw > irig_threshold).astype(np.bool_)
            pps_binary = (pps_raw > pps_threshold).astype(np.bool_)

            # Create arrays with previous state prepended for diff calculation
            irig_with_prev = np.concatenate(([last_irig_bit], irig_binary))
            pps_with_prev = np.concatenate(([last_pps_bit], pps_binary))

            # Find where changes occur using diff
            irig_diffs = np.diff(irig_with_prev.astype(np.int8))
            pps_diffs = np.diff(pps_with_prev.astype(np.int8))

            # Find rising edges (0->1, diff = 1) and falling edges (1->0, diff = -1)
            irig_rising_index = np.where(irig_diffs == 1)[0] + chunk_starting_index
            irig_falling_index = np.where(irig_diffs == -1)[0] + chunk_starting_index

            pps_rising_index = np.where(pps_diffs == 1)[0] + chunk_starting_index
            pps_falling_index = np.where(pps_diffs == -1)[0] + chunk_starting_index

            # Extend the edge lists
            irig_rising_edges_list.extend(irig_rising_index)
            irig_falling_edges_list.extend(irig_falling_index)
            pps_rising_edges_list.extend(pps_rising_index)
            pps_falling_edges_list.extend(pps_falling_index)

            # Update last states
            last_irig_bit = irig_binary[-1] if len(irig_binary) > 0 else last_irig_bit
            last_pps_bit = pps_binary[-1] if len(pps_binary) > 0 else last_pps_bit

            if chunk_num % 20 == 0:  # Every 20 chunks instead of 50
                print(f"Chunk {chunk_num}: {chunk_starting_index + chunk_size:,} samples processed, ({round(((chunk_starting_index + chunk_size) / sample_rate) / 3600, 2)} hours)")
                sys.stdout.flush()

            # Heartbeat every 5 chunks (more frequent due to smaller chunks)
            elif chunk_num % 5 == 0:
                print(f"Chunk {chunk_num}... ", end="", flush=True)
            
            if chunk_num % 40 == 0:
                print("Flushing python lists to numpy arrays.")
                irig_rising_edges = np.concatenate((irig_rising_edges, np.fromiter(irig_rising_edges_list,dtype=np.int64)))
                irig_falling_edges = np.concatenate((irig_falling_edges, np.fromiter(irig_falling_edges_list,dtype=np.int64)))
                pps_rising_edges = np.concatenate((pps_rising_edges, np.fromiter(pps_rising_edges_list,dtype=np.int64)))
                pps_falling_edges = np.concatenate((pps_falling_edges, np.fromiter(pps_falling_edges_list,dtype=np.int64)))

                irig_rising_edges_list.clear()
                irig_falling_edges_list.clear()
    
                pps_rising_edges_list.clear()
                pps_falling_edges_list.clear()

            if chunk_num % 10 == 0:
                gc.collect()
                
        except Exception as e:
            print(f"Error processing chunk {chunk_num}: {e}")
            print(f"Chunk size was: {len(raw_chunk)} bytes")
            break
        finally:
            chunk_num += 1

    np.savez(file=irig_output, starts=irig_rising_edges, ends=irig_falling_edges)
    np.savez(file=pps_output, starts=pps_rising_edges, ends=pps_falling_edges)
