import datetime
import os
from typing import List, Tuple
import numpy as np
import glob

input_files = glob.glob("data/timestamps_events_absolute*.csv")


irig_outputs=[] # a list of tuples (output file, input file)

for input_file, i  in zip(input_files, range(len(input_files))):
    if os.path.exists(input_file):

        formatted_time = datetime.datetime.fromtimestamp(os.path.getmtime(input_file)).strftime("%Y%m%d_%H%M%S")        
        # Create output filenames with date
        irig_outputs.append((f'irig_data_{formatted_time}_{i}.npz', input_file))
    else:
        # Fallback if file doesn't exist
        raise FileExistsError(f'The input file \'{input_file}\' does not exist. Are you in the right directory?')
    
for t in irig_outputs:
    print(f"Processing {t[1]}")
    print(f"Output file will be: {t[0]}")

    stamps = np.genfromtxt(t[1], delimiter=',', usecols=3, dtype=str)

    starts = []
    ends = []
    i = 0

    for stamp in stamps:
        # print(stamp)
        if stamp[:5] == 'frame':
            i += 1
        elif stamp == 'TimeP (pin 17)':
            # print(f'found start: {i}')
            starts.append(i)
        elif stamp == 'TimeN (pin 27)':
            # print(f'found end: {i}')
            if len(starts) != 0:
                ends.append(i)

    min_len = min(len(starts),len(ends))
    starts = starts[:min_len]
    ends = ends[:min_len]

    np.savez(f'data/{t[0]}', starts=starts, ends=ends)

    def to_pulse_lengths(rising_edges: np.ndarray, falling_edges: np.ndarray) -> List[Tuple[int, int]]:
        return np.column_stack((falling_edges - rising_edges, rising_edges)).tolist()
    
    pulses = to_pulse_lengths(np.array(starts, dtype=np.int64), np.array(ends, dtype=np.int64))
    print(pulses)