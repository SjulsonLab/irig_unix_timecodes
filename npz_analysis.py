from typing import List, Tuple
import numpy as np
import glob
import os
import csv
import irig_h_gpio as irig
import threading

irig_filename = max(glob.glob("data/irig_data_*.npz"), key=os.path.getmtime)
pps_filename = max(glob.glob("data/pps_data_*.npz"), key=os.path.getmtime)

error_filename = 'data/indexes_of_error.csv'
decoded_filename = 'data/decoded_timecodes.csv'
pulselength_filename = 'data/pulse_lengths.csv'

def to_pulse_lengths(rising_edges: np.ndarray, falling_edges: np.ndarray) -> List[Tuple[int, int]]:
    return np.column_stack((falling_edges - rising_edges, rising_edges)).tolist()

def find_sample_rate():
    irig_starts = np.load(irig_filename)['starts']
    return round(np.diff(irig_starts).mean(), 0)
    
sample_rate = find_sample_rate()
print(f'Sample rate: {sample_rate}')

def error_analysis():
    jumps = 0

    irig_starts = np.load(irig_filename)['starts'] #TODO change thsi!!!!!
    pps_starts = np.load(pps_filename)['ends']

    min_len = min(len(irig_starts), len(pps_starts))
    irig_starts = irig_starts[:min_len].tolist()
    pps_starts = pps_starts[:min_len].tolist()

    result = []

    for irig_start, pps_start in zip(irig_starts, pps_starts):
        error = irig_start - pps_start
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


    with open(error_filename, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(result)

def decode_analysis():
    irig_file = np.load(irig_filename)

    with open(decoded_filename, 'w', newline='') as file:
        pulse_lengths = to_pulse_lengths(irig_file['starts'], irig_file['ends'])
        irig_bits = irig.to_irig_bits(pulse_lengths)
        decoded = irig.decode_irig_bits(irig_bits)

        csv_writer = csv.writer(file)
        csv_writer.writerows(decoded)

error_thread = threading.Thread(target=error_analysis)
decode_thread = threading.Thread(target=decode_analysis)
error_thread.start()
decode_thread.start()