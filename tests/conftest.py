"""Test data generator for neurokairos.

Imports ``bcd_encode`` and weight constants from NeuroKairos to build
authentic IRIG-H bit patterns.  The generated ``.dat`` file starts and
ends mid-frame, matching real-world recordings.
"""

import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from neurokairos.irig import (
    bcd_encode,
    SECONDS_WEIGHTS,
    MINUTES_WEIGHTS,
    HOURS_WEIGHTS,
    DAY_OF_YEAR_WEIGHTS,
    DECISECONDS_WEIGHTS,
    YEARS_WEIGHTS,
)


# -- Frame generator -----------------------------------------------------------

def generate_irig_h_frame(t):
    """Generate a 60-element IRIG-H frame for datetime *t*.

    Encodes the time literally — same as the C sender.
    """
    seconds_bcd = bcd_encode(t.second, SECONDS_WEIGHTS)
    minutes_bcd = bcd_encode(t.minute, MINUTES_WEIGHTS)
    hours_bcd = bcd_encode(t.hour, HOURS_WEIGHTS)
    day_of_year_bcd = bcd_encode(t.timetuple().tm_yday, DAY_OF_YEAR_WEIGHTS)
    deciseconds_bcd = bcd_encode(0, DECISECONDS_WEIGHTS)
    year_bcd = bcd_encode(t.year % 100, YEARS_WEIGHTS)

    frame = []

    # Bit 00: Pr
    frame.append("P")
    # Bits 01-04: seconds units (1, 2, 4, 8)
    frame.extend(seconds_bcd[0:4])
    # Bit 05: unused
    frame.append(False)
    # Bits 06-08: seconds tens (10, 20, 40)
    frame.extend(seconds_bcd[4:7])
    # Bit 09: P1
    frame.append("P")
    # Bits 10-13: minutes units
    frame.extend(minutes_bcd[0:4])
    # Bit 14: unused
    frame.append(False)
    # Bits 15-17: minutes tens
    frame.extend(minutes_bcd[4:7])
    # Bit 18: unused
    frame.append(False)
    # Bit 19: P2
    frame.append("P")
    # Bits 20-23: hours units
    frame.extend(hours_bcd[0:4])
    # Bit 24: unused
    frame.append(False)
    # Bits 25-26: hours tens
    frame.extend(hours_bcd[4:6])
    # Bits 27-28: unused
    frame.extend([False, False])
    # Bit 29: P3
    frame.append("P")
    # Bits 30-33: day-of-year units
    frame.extend(day_of_year_bcd[0:4])
    # Bit 34: unused
    frame.append(False)
    # Bits 35-38: day-of-year tens
    frame.extend(day_of_year_bcd[4:8])
    # Bit 39: P4
    frame.append("P")
    # Bits 40-41: day-of-year hundreds
    frame.extend(day_of_year_bcd[8:10])
    # Bits 42-44: unused
    frame.extend([False, False, False])
    # Bits 45-48: deciseconds
    frame.extend(deciseconds_bcd[0:4])
    # Bit 49: P5
    frame.append("P")
    # Bits 50-53: year units
    frame.extend(year_bcd[0:4])
    # Bit 54: unused
    frame.append(False)
    # Bits 55-58: year tens
    frame.extend(year_bcd[4:8])
    # Bit 59: P6
    frame.append("P")

    assert len(frame) == 60
    return frame


def _bit_to_pulse_frac(bit):
    """Convert a frame element to a pulse-width fraction of 1 second."""
    if bit == "P":
        return 0.8  # reference marker
    elif bit:
        return 0.5  # binary 1
    else:
        return 0.2  # binary 0


# -- Fixture -------------------------------------------------------------------

@pytest.fixture(scope="session")
def generate_test_dat():
    """Generate ``temp/test_irig.dat`` — a 240-second, 3-channel, 30 kHz
    interleaved int16 file with an IRIG-H signal on channel 2.

    The recording starts at 14:30:37 UTC on 2026-01-15, which is mid-frame
    (bit 37 of the 14:30 frame).  Returns a metadata dict.
    """
    sample_rate = 30_000
    n_channels = 3
    duration_s = 240
    start_dt = datetime(2026, 1, 15, 14, 30, 37, tzinfo=timezone.utc)

    # -- Build the IRIG bit sequence -------------------------------------------
    # Frames are aligned to minute boundaries.  We generate complete frames
    # for each relevant minute, then slice out the recording window.
    minute_start = start_dt.replace(second=0, microsecond=0)
    all_bits = []
    frame_time = minute_start
    end_dt = start_dt + timedelta(seconds=duration_s)
    while frame_time < end_dt:
        frame = generate_irig_h_frame(frame_time)
        all_bits.extend(frame)
        frame_time += timedelta(seconds=60)

    # Slice to our recording window (starting at bit = start second)
    start_bit = start_dt.second  # 37
    recording_bits = all_bits[start_bit : start_bit + duration_s]
    assert len(recording_bits) == duration_s

    # -- Generate the analog signals -------------------------------------------
    n_samples = sample_rate * duration_s
    rng = np.random.default_rng(42)

    # IRIG channel: square-wave pulses + noise
    irig = np.zeros(n_samples, dtype=np.float64)
    for i, bit in enumerate(recording_bits):
        frac = _bit_to_pulse_frac(bit)
        pw = int(frac * sample_rate)
        s0 = i * sample_rate
        irig[s0 : s0 + pw] = 10_000.0
    irig += rng.normal(0, 500, n_samples)

    # Other channels: simple sinusoids
    t = np.arange(n_samples, dtype=np.float64) / sample_rate
    ch0 = 10_000.0 * np.sin(2 * np.pi * 10 * t)
    ch1 = 10_000.0 * np.sin(2 * np.pi * 50 * t)

    # -- Interleave and write --------------------------------------------------
    data = np.empty((n_samples, n_channels), dtype=np.int16)
    data[:, 0] = np.clip(ch0, -32768, 32767).astype(np.int16)
    data[:, 1] = np.clip(ch1, -32768, 32767).astype(np.int16)
    data[:, 2] = np.clip(irig, -32768, 32767).astype(np.int16)

    temp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    path = os.path.join(temp_dir, "test_irig.dat")
    data.tofile(path)

    # -- Expected (source, reference) pairs ------------------------------------
    # Pulse 0 is not detected: the signal starts HIGH (mid-pulse), so there
    # is no rising edge at sample 0.  The orphaned falling edge is discarded.
    start_ts = start_dt.timestamp()
    expected_pairs = [
        (float(i * sample_rate), start_ts + float(i))
        for i in range(1, duration_s)
    ]

    return {
        "path": path,
        "start_time": start_dt,
        "start_timestamp": start_ts,
        "sample_rate": sample_rate,
        "n_channels": n_channels,
        "irig_channel": 2,
        "duration_s": duration_s,
        "expected_pairs": expected_pairs,
    }


# -- Helper to build IRIG signal from bit sequence -----------------------------

def _build_irig_signal(recording_bits, sample_rate, n_samples, rng):
    """Build an IRIG int16 signal array from a sequence of frame elements."""
    irig = np.zeros(n_samples, dtype=np.float64)
    for i, bit in enumerate(recording_bits):
        frac = _bit_to_pulse_frac(bit)
        pw = int(frac * sample_rate)
        s0 = i * sample_rate
        if s0 + pw <= n_samples:
            irig[s0 : s0 + pw] = 10_000.0
    irig += rng.normal(0, 500, n_samples)
    return irig


def _write_dat(irig, sample_rate, n_samples, n_channels, path):
    """Interleave channels and write a .dat file."""
    t = np.arange(n_samples, dtype=np.float64) / sample_rate
    ch0 = 10_000.0 * np.sin(2 * np.pi * 10 * t)
    ch1 = 10_000.0 * np.sin(2 * np.pi * 50 * t)

    data = np.empty((n_samples, n_channels), dtype=np.int16)
    data[:, 0] = np.clip(ch0, -32768, 32767).astype(np.int16)
    data[:, 1] = np.clip(ch1, -32768, 32767).astype(np.int16)
    data[:, 2] = np.clip(irig, -32768, 32767).astype(np.int16)
    data.tofile(path)


# -- Missing-pulses fixture ----------------------------------------------------

@pytest.fixture(scope="session")
def generate_test_dat_missing_pulses():
    """Generate a .dat file with missing pulses (zeroed-out HIGH regions).

    Deletes pulses at indices 5, 10 (in initial partial frame) and 220
    (in final partial frame) — positions outside any complete frame's decode
    range, so frame decoding is unaffected.
    """
    sample_rate = 30_000
    n_channels = 3
    duration_s = 240
    start_dt = datetime(2026, 1, 15, 14, 30, 37, tzinfo=timezone.utc)

    # Build bit sequence (same as generate_test_dat)
    minute_start = start_dt.replace(second=0, microsecond=0)
    all_bits = []
    frame_time = minute_start
    end_dt = start_dt + timedelta(seconds=duration_s)
    while frame_time < end_dt:
        frame = generate_irig_h_frame(frame_time)
        all_bits.extend(frame)
        frame_time += timedelta(seconds=60)

    start_bit = start_dt.second
    recording_bits = all_bits[start_bit : start_bit + duration_s]

    n_samples = sample_rate * duration_s
    rng = np.random.default_rng(42)
    irig = _build_irig_signal(recording_bits, sample_rate, n_samples, rng)

    # Zero out the HIGH region for deleted pulses (safe positions:
    # 5, 10 are in the initial partial frame before boundary at pulse 23;
    # 220 is in the final partial frame after the last complete frame at 143+60=203)
    deleted_indices = [5, 10, 220]
    for idx in deleted_indices:
        s0 = idx * sample_rate
        frac = _bit_to_pulse_frac(recording_bits[idx])
        pw = int(frac * sample_rate)
        irig[s0 : s0 + pw] = irig[s0 + pw]  # set to baseline noise level

    temp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    path = os.path.join(temp_dir, "test_irig_missing.dat")
    _write_dat(irig, sample_rate, n_samples, n_channels, path)

    # Expected: all pulses except the deleted ones.
    # Pulse 0 is also lost (signal starts HIGH — no rising edge).
    start_ts = start_dt.timestamp()
    surviving_indices = [
        i for i in range(1, duration_s) if i not in deleted_indices
    ]
    expected_pairs = [
        (float(i * sample_rate), start_ts + float(i))
        for i in surviving_indices
    ]

    return {
        "path": path,
        "start_time": start_dt,
        "start_timestamp": start_ts,
        "sample_rate": sample_rate,
        "n_channels": n_channels,
        "irig_channel": 2,
        "duration_s": duration_s,
        "deleted_indices": deleted_indices,
        "n_surviving": len(surviving_indices),
        "expected_pairs": expected_pairs,
    }


# -- Extra-pulses fixture -----------------------------------------------------

@pytest.fixture(scope="session")
def generate_test_dat_extra_pulses():
    """Generate a .dat file with short noise spikes between real pulses.

    Inserts 1 ms spikes halfway between pulses 75-76 and 150-151.
    """
    sample_rate = 30_000
    n_channels = 3
    duration_s = 240
    start_dt = datetime(2026, 1, 15, 14, 30, 37, tzinfo=timezone.utc)

    minute_start = start_dt.replace(second=0, microsecond=0)
    all_bits = []
    frame_time = minute_start
    end_dt = start_dt + timedelta(seconds=duration_s)
    while frame_time < end_dt:
        frame = generate_irig_h_frame(frame_time)
        all_bits.extend(frame)
        frame_time += timedelta(seconds=60)

    start_bit = start_dt.second
    recording_bits = all_bits[start_bit : start_bit + duration_s]

    n_samples = sample_rate * duration_s
    rng = np.random.default_rng(42)
    irig = _build_irig_signal(recording_bits, sample_rate, n_samples, rng)

    # Insert noise spikes (1 ms = 30 samples) halfway between pulses
    spike_width = int(0.001 * sample_rate)  # 30 samples
    spike_after_pulses = [75, 150]
    for pulse_idx in spike_after_pulses:
        midpoint = int((pulse_idx + 0.5) * sample_rate)
        irig[midpoint : midpoint + spike_width] = 10_000.0

    temp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    path = os.path.join(temp_dir, "test_irig_extra.dat")
    _write_dat(irig, sample_rate, n_samples, n_channels, path)

    # Expected: same as clean signal (extras should be removed).
    # Pulse 0 is lost (signal starts HIGH — no rising edge).
    start_ts = start_dt.timestamp()
    expected_pairs = [
        (float(i * sample_rate), start_ts + float(i))
        for i in range(1, duration_s)
    ]

    return {
        "path": path,
        "start_time": start_dt,
        "start_timestamp": start_ts,
        "sample_rate": sample_rate,
        "n_channels": n_channels,
        "irig_channel": 2,
        "duration_s": duration_s,
        "spike_after_pulses": spike_after_pulses,
        "expected_pairs": expected_pairs,
    }


# -- Concatenated-files fixture ------------------------------------------------

@pytest.fixture(scope="session")
def generate_test_dat_concatenated():
    """Generate a .dat file from two concatenated recordings with a time gap.

    First segment: 120 seconds starting at 14:30:37 UTC.
    Dead zone: 2 seconds of silence (creates a detectable inter-onset gap).
    Second segment: 120 seconds starting at 18:00:00 UTC.
    """
    sample_rate = 30_000
    n_channels = 3
    seg1_duration = 120
    seg2_duration = 120
    dead_zone_s = 2
    seg1_start = datetime(2026, 1, 15, 14, 30, 37, tzinfo=timezone.utc)
    seg2_start = datetime(2026, 1, 15, 18, 0, 0, tzinfo=timezone.utc)

    rng = np.random.default_rng(42)

    def _make_segment(start_dt, dur):
        minute_start = start_dt.replace(second=0, microsecond=0)
        bits = []
        ft = minute_start
        end = start_dt + timedelta(seconds=dur)
        while ft < end:
            bits.extend(generate_irig_h_frame(ft))
            ft += timedelta(seconds=60)
        sb = start_dt.second
        rec_bits = bits[sb : sb + dur]
        ns = sample_rate * dur
        sig = _build_irig_signal(rec_bits, sample_rate, ns, rng)
        return sig, ns

    sig1, ns1 = _make_segment(seg1_start, seg1_duration)
    sig2, ns2 = _make_segment(seg2_start, seg2_duration)

    # Insert a dead zone (silence + noise) between segments
    dead_samples = dead_zone_s * sample_rate
    dead_zone = rng.normal(0, 500, dead_samples)

    total_samples = ns1 + dead_samples + ns2
    irig = np.concatenate([sig1, dead_zone, sig2])

    temp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    path = os.path.join(temp_dir, "test_irig_concat.dat")

    t = np.arange(total_samples, dtype=np.float64) / sample_rate
    ch0 = 10_000.0 * np.sin(2 * np.pi * 10 * t)
    ch1 = 10_000.0 * np.sin(2 * np.pi * 50 * t)
    data = np.empty((total_samples, n_channels), dtype=np.int16)
    data[:, 0] = np.clip(ch0, -32768, 32767).astype(np.int16)
    data[:, 1] = np.clip(ch1, -32768, 32767).astype(np.int16)
    data[:, 2] = np.clip(irig, -32768, 32767).astype(np.int16)
    data.tofile(path)

    # Expected pairs: seg1 pulses then seg2 pulses.
    # Seg1 pulse 0 is lost (signal starts HIGH — no rising edge).
    # Seg2 pulse 0 IS detected (preceded by LOW dead zone → real rising edge).
    seg1_ts = seg1_start.timestamp()
    seg2_ts = seg2_start.timestamp()
    seg2_sample_offset = (seg1_duration + dead_zone_s) * sample_rate
    expected_pairs = []
    for i in range(1, seg1_duration):
        expected_pairs.append((float(i * sample_rate), seg1_ts + float(i)))
    for i in range(seg2_duration):
        expected_pairs.append(
            (float(seg2_sample_offset + i * sample_rate), seg2_ts + float(i)))

    return {
        "path": path,
        "start_time_seg1": seg1_start,
        "start_time_seg2": seg2_start,
        "sample_rate": sample_rate,
        "n_channels": n_channels,
        "irig_channel": 2,
        "seg1_duration": seg1_duration,
        "seg2_duration": seg2_duration,
        "total_pulses": (seg1_duration - 1) + seg2_duration,
        "expected_pairs": expected_pairs,
    }
