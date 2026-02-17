"""Tests for decode_intervals_irig — IRIG decoding from pre-extracted pulse intervals."""

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

from neurokairos import ClockTable, decode_intervals_irig


# -- Local helpers (same logic as conftest.py) ---------------------------------

def _generate_irig_h_frame(t):
    """Generate a 60-element IRIG-H frame for datetime *t*."""
    seconds_bcd = bcd_encode(t.second, SECONDS_WEIGHTS)
    minutes_bcd = bcd_encode(t.minute, MINUTES_WEIGHTS)
    hours_bcd = bcd_encode(t.hour, HOURS_WEIGHTS)
    day_of_year_bcd = bcd_encode(t.timetuple().tm_yday, DAY_OF_YEAR_WEIGHTS)
    deciseconds_bcd = bcd_encode(0, DECISECONDS_WEIGHTS)
    year_bcd = bcd_encode(t.year % 100, YEARS_WEIGHTS)

    frame = []
    frame.append("P")
    frame.extend(seconds_bcd[0:4])
    frame.append(False)
    frame.extend(seconds_bcd[4:7])
    frame.append("P")
    frame.extend(minutes_bcd[0:4])
    frame.append(False)
    frame.extend(minutes_bcd[4:7])
    frame.append(False)
    frame.append("P")
    frame.extend(hours_bcd[0:4])
    frame.append(False)
    frame.extend(hours_bcd[4:6])
    frame.extend([False, False])
    frame.append("P")
    frame.extend(day_of_year_bcd[0:4])
    frame.append(False)
    frame.extend(day_of_year_bcd[4:8])
    frame.append("P")
    frame.extend(day_of_year_bcd[8:10])
    frame.extend([False, False, False])
    frame.extend(deciseconds_bcd[0:4])
    frame.append("P")
    frame.extend(year_bcd[0:4])
    frame.append(False)
    frame.extend(year_bcd[4:8])
    frame.append("P")
    assert len(frame) == 60
    return frame


def _bit_to_pulse_frac(bit):
    if bit == "P":
        return 0.8
    elif bit:
        return 0.5
    else:
        return 0.2


# -- Fixture: generate IRIG pulse timing analytically --------------------------

DURATION_S = 240
START_DT = datetime(2026, 1, 15, 14, 30, 37, tzinfo=timezone.utc)
START_TS = START_DT.timestamp()


@pytest.fixture(scope="module")
def irig_intervals():
    """Generate IRIG pulse onset/offset arrays in seconds (no raw signal)."""
    minute_start = START_DT.replace(second=0, microsecond=0)
    all_bits = []
    frame_time = minute_start
    end_dt = START_DT + timedelta(seconds=DURATION_S)
    while frame_time < end_dt:
        frame = _generate_irig_h_frame(frame_time)
        all_bits.extend(frame)
        frame_time += timedelta(seconds=60)

    start_bit = START_DT.second  # 37
    recording_bits = all_bits[start_bit : start_bit + DURATION_S]

    onsets = np.arange(DURATION_S, dtype=np.float64)
    widths = np.array([_bit_to_pulse_frac(b) for b in recording_bits])
    offsets = onsets + widths

    return {
        "onsets": onsets,
        "offsets": offsets,
        "widths": widths,
        "recording_bits": recording_bits,
    }


# -- TestDecodeIntervalsArrays -------------------------------------------------

class TestDecodeIntervalsArrays:
    """Test decode_intervals_irig with raw onset/offset arrays."""

    def test_returns_clock_table(self, irig_intervals):
        ct = decode_intervals_irig(
            irig_intervals["onsets"], irig_intervals["offsets"])
        assert isinstance(ct, ClockTable)

    def test_entry_count(self, irig_intervals):
        ct = decode_intervals_irig(
            irig_intervals["onsets"], irig_intervals["offsets"])
        assert len(ct.source) == DURATION_S

    def test_monotonically_increasing(self, irig_intervals):
        ct = decode_intervals_irig(
            irig_intervals["onsets"], irig_intervals["offsets"])
        assert np.all(np.diff(ct.source) > 0)
        assert np.all(np.diff(ct.reference) > 0)

    def test_one_second_spacing(self, irig_intervals):
        ct = decode_intervals_irig(
            irig_intervals["onsets"], irig_intervals["offsets"])
        ref_diffs = np.diff(ct.reference)
        np.testing.assert_allclose(ref_diffs, 1.0, atol=0.01)

    def test_timestamps_match_expected(self, irig_intervals):
        ct = decode_intervals_irig(
            irig_intervals["onsets"], irig_intervals["offsets"])
        expected_ref = np.array([START_TS + i for i in range(DURATION_S)])
        np.testing.assert_allclose(ct.reference, expected_ref, atol=0.01)

    def test_source_units_default(self, irig_intervals):
        ct = decode_intervals_irig(
            irig_intervals["onsets"], irig_intervals["offsets"])
        assert ct.source_units == "seconds"

    def test_source_units_custom(self, irig_intervals):
        ct = decode_intervals_irig(
            irig_intervals["onsets"], irig_intervals["offsets"],
            source_units="milliseconds")
        assert ct.source_units == "milliseconds"

    def test_nominal_rate_approx_one(self, irig_intervals):
        ct = decode_intervals_irig(
            irig_intervals["onsets"], irig_intervals["offsets"])
        assert abs(ct.nominal_rate - 1.0) < 0.1

    def test_metadata_decoding_stats(self, irig_intervals):
        ct = decode_intervals_irig(
            irig_intervals["onsets"], irig_intervals["offsets"])
        assert isinstance(ct.metadata, dict)
        assert ct.metadata["n_raw_pulses"] == DURATION_S
        assert ct.metadata["n_extra_removed"] == 0
        assert ct.metadata["n_frames_decoded"] >= 1
        assert "source_file" not in ct.metadata

    def test_metadata_survives_source_units_copy(self, irig_intervals):
        ct = decode_intervals_irig(
            irig_intervals["onsets"], irig_intervals["offsets"],
            source_units="milliseconds")
        assert ct.metadata["n_raw_pulses"] == DURATION_S
        assert ct.source_units == "milliseconds"


# -- TestDecodeIntervalsIntervalSet --------------------------------------------

class TestDecodeIntervalsIntervalSet:
    """Test decode_intervals_irig with pynapple IntervalSet input."""

    @pytest.fixture
    def interval_set(self, irig_intervals):
        pynapple = pytest.importorskip("pynapple")
        return pynapple.IntervalSet(
            start=irig_intervals["onsets"],
            end=irig_intervals["offsets"],
        )

    def test_returns_clock_table(self, interval_set):
        ct = decode_intervals_irig(interval_set)
        assert isinstance(ct, ClockTable)

    def test_entry_count(self, interval_set):
        ct = decode_intervals_irig(interval_set)
        assert len(ct.source) == DURATION_S

    def test_timestamps_match_expected(self, interval_set):
        ct = decode_intervals_irig(interval_set)
        expected_ref = np.array([START_TS + i for i in range(DURATION_S)])
        np.testing.assert_allclose(ct.reference, expected_ref, atol=0.01)

    def test_offsets_ignored(self, interval_set, irig_intervals):
        """offsets param should be ignored when IntervalSet-like is passed."""
        bogus = np.zeros(10)
        ct = decode_intervals_irig(interval_set, offsets=bogus)
        assert len(ct.source) == DURATION_S


# -- TestDecodeIntervalsDuckTyped ----------------------------------------------

class TestDecodeIntervalsDuckTyped:
    """Test duck-typing: any object with .start/.end works."""

    def test_duck_typed_object(self, irig_intervals):
        class FakeIntervals:
            def __init__(self, start, end):
                self.start = start
                self.end = end

        obj = FakeIntervals(irig_intervals["onsets"], irig_intervals["offsets"])
        ct = decode_intervals_irig(obj)
        assert isinstance(ct, ClockTable)
        assert len(ct.source) == DURATION_S

    def test_duck_typed_timestamps(self, irig_intervals):
        class FakeIntervals:
            def __init__(self, start, end):
                self.start = start
                self.end = end

        obj = FakeIntervals(irig_intervals["onsets"], irig_intervals["offsets"])
        ct = decode_intervals_irig(obj)
        expected_ref = np.array([START_TS + i for i in range(DURATION_S)])
        np.testing.assert_allclose(ct.reference, expected_ref, atol=0.01)


# -- TestDecodeIntervalsJitter -------------------------------------------------

class TestDecodeIntervalsJitter:
    """Test with realistic conditions: jitter on onsets, nonzero start time."""

    def test_jittered_onsets(self, irig_intervals):
        rng = np.random.default_rng(123)
        jitter = rng.normal(0, 0.001, DURATION_S)  # 1ms std
        jittered_onsets = irig_intervals["onsets"] + jitter
        jittered_offsets = jittered_onsets + irig_intervals["widths"]

        ct = decode_intervals_irig(jittered_onsets, jittered_offsets)
        assert len(ct.source) == DURATION_S
        expected_ref = np.array([START_TS + i for i in range(DURATION_S)])
        np.testing.assert_allclose(ct.reference, expected_ref, atol=0.01)

    def test_nonzero_start(self, irig_intervals):
        offset = 100.0
        shifted_onsets = irig_intervals["onsets"] + offset
        shifted_offsets = irig_intervals["offsets"] + offset

        ct = decode_intervals_irig(shifted_onsets, shifted_offsets)
        assert len(ct.source) == DURATION_S
        # Source values should be shifted
        np.testing.assert_allclose(ct.source, shifted_onsets, atol=0.001)
        # Reference timestamps should still be correct UTC
        expected_ref = np.array([START_TS + i for i in range(DURATION_S)])
        np.testing.assert_allclose(ct.reference, expected_ref, atol=0.01)


# -- TestDecodeIntervalsSave ---------------------------------------------------

class TestDecodeIntervalsSave:
    """Test persistence via the save parameter."""

    def test_save_and_load(self, irig_intervals, tmp_path):
        save_path = tmp_path / "intervals.clocktable.npz"
        ct = decode_intervals_irig(
            irig_intervals["onsets"], irig_intervals["offsets"],
            save=str(save_path))

        assert save_path.exists()
        ct2 = ClockTable.load(save_path)
        np.testing.assert_array_equal(ct.source, ct2.source)
        np.testing.assert_array_equal(ct.reference, ct2.reference)

    def test_no_save_by_default(self, irig_intervals, tmp_path):
        ct = decode_intervals_irig(
            irig_intervals["onsets"], irig_intervals["offsets"])
        # No file should be created anywhere (we can't easily check this
        # fully, but at least confirm save=None is the default behavior)
        assert ct is not None


# -- TestDecodeIntervalsErrors -------------------------------------------------

class TestDecodeIntervalsErrors:
    """Test validation and error handling."""

    def test_missing_offsets_raises(self, irig_intervals):
        with pytest.raises(ValueError, match="offsets is required"):
            decode_intervals_irig(irig_intervals["onsets"])

    def test_mismatched_lengths_raises(self, irig_intervals):
        with pytest.raises(ValueError, match="same length"):
            decode_intervals_irig(
                irig_intervals["onsets"], irig_intervals["offsets"][:100])
