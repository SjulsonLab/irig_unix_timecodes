import numpy as np
import glob
import os
import csv
import irig_h_gpio as irig

SAMPLE_RATE = 30_000

irig_filename = max(glob.glob("data/irig_data_*.npz"), key=os.path.getmtime)
pps_filename = max(glob.glob("data/pps_data_*.npz"), key=os.path.getmtime)

error_filename = 'data/indexes_of_error.csv'
decoded_filename = 'data/decoded_timecodes.csv'
pulselength_filename = 'data/pulse_lengths.csv'

def error_analysis():
    jumps = 0

    irig_starts = np.load(irig_filename)['array1'] #TODO change thsi!!!!!
    pps_starts = np.load(pps_filename)['array1']

    min_len = min(len(irig_starts), len(pps_starts))
    irig_starts = irig_starts[:min_len].tolist()
    pps_starts = pps_starts[:min_len].tolist()

    result = []

    for irig_start, pps_start in zip(irig_starts, pps_starts):
        error = irig_start - pps_start
        while error < -SAMPLE_RATE * (0.95 + jumps):
            jumps +=1
            result.append('JUMP')
        result.append(error + (SAMPLE_RATE * jumps))

    cumulative = 0
    no_outliers = [r for r in result if type(r) != str and r < 15]
    for d in no_outliers:
        cumulative += d
    cumulative /= len(no_outliers)
    print(f'Average index delay without outliers: {cumulative}\nSeconds: {cumulative/SAMPLE_RATE}')
    cumulative = 0
    with_outliers = [r for r in result if type(r) != str]
    for d in with_outliers:
        cumulative += d
    cumulative /= len(with_outliers)
    print(f'Average index delay with outliers: {cumulative}\nSeconds: {cumulative/SAMPLE_RATE}')


    with open(error_filename, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(result)

def decode_analysis():
    irig_file = np.load(irig_filename)

    with open(decoded_filename, 'w', newline='') as file:
        pulse_lengths = irig.to_pulse_lengths(irig_file['array1'], irig_file['array2'])
        irig_bits = irig.to_irig_bits(pulse_lengths)
        decoded = irig.decode_irig_bits(irig_bits)

        csv_writer = csv.writer(file)
        csv_writer.writerows(decoded)

error_analysis()
decode_analysis()