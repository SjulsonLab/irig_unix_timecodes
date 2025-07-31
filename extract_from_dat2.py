import numpy as np
import sys

total_channels = 40
adc1_index = 32  # IRIG
adc2_index = 33  # PPS
sample_rate = 30000

# Thresholds based on your debug output
irig_threshold = 2500  # Adjust if needed
pps_threshold = 2500   # Adjust if needed

# File paths
input_file = 'continuous.dat'  # Change to your .dat file path
pps_output = 'pps_binary.bin'
irig_output = 'irig_binary.bin'

chunk_size = 250000  # Samples per chunk (reduced from 1M)

with open(input_file, 'rb') as f:
    chunk_num = 0

    last_irig_bit = False
    last_pps_bit = False

    irig_rising_edges = []
    irig_falling_edges = []
    
    pps_rising_edges = []
    pps_falling_edges = []
    
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
            irig_binary = (irig_raw > irig_threshold).astype(np.bool)
            pps_binary = (pps_raw > pps_threshold).astype(np.bool)

            for i in range(chunk_size):
                if irig_binary[i] != last_irig_bit:
                    (irig_rising_edges if irig_binary[i] else irig_falling_edges).append(i + chunk_starting_index)
                if pps_binary[i] != last_pps_bit:
                    (pps_rising_edges if irig_binary[i] else pps_falling_edges).append(i + chunk_starting_index)
                last_irig_bit = irig_binary[i]
                last_pps_bit = pps_binary[i]

            if chunk_num % 20 == 0:  # Every 20 chunks instead of 50
                print(f"Chunk {chunk_num}: {chunk_starting_index + chunk_size:,} samples processed, ({round((chunk_starting_index + chunk_size / sample_rate) / 3600, 2)} hours)")
                sys.stdout.flush()
                
            # Heartbeat every 5 chunks (more frequent due to smaller chunks)
            elif chunk_num % 5 == 0:
                print(f"Chunk {chunk_num}... ", end="", flush=True)
                
        except Exception as e:
            print(f"Error processing chunk {chunk_num}: {e}")
            print(f"Chunk size was: {len(raw_chunk)} bytes")
            break
        finally:
            chunk_num += 1

    irig_starts = np.array(irig_rising_edges, dtype=np.uint64)
    irig_ends = np.array(irig_falling_edges, dtype=np.uint64)

    pps_starts = np.array(pps_rising_edges, dtype=np.uint64)
    pps_ends = np.array(pps_falling_edges, dtype=np.uint64)

    np.savez(file=irig_output, array1=irig_starts, array2=irig_ends)
    np.savez(file=pps_output, array1=pps_starts, array2=pps_ends)
