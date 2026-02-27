import numpy as np
import pytest
from datetime import datetime, timezone, timedelta

from neurokairos.irig import (
    PULSE_ZERO,
    PULSE_ONE,
    PULSE_MARKER,
    PULSE_INVALID,
    estimate_samples_per_second,
    classify_pulses,
    find_frame_boundaries,
    bcd_decode,
    bcd_encode,
    decode_frame,
    decode_partial_frame,
    compute_elapsed_seconds,
    remove_extra_pulses,
    assign_timestamps_from_anchors,
    SECONDS_WEIGHTS,
    MINUTES_WEIGHTS,
    HOURS_WEIGHTS,
    DAY_OF_YEAR_WEIGHTS,
    DECISECONDS_WEIGHTS,
    YEARS_WEIGHTS,
)


class TestEstimateSamplesPerSecond:
    def test_uniform_spacing(self):
        onsets = np.arange(0, 300_000 * 10, 30_000, dtype=np.float64)
        sps = estimate_samples_per_second(onsets)
        assert abs(sps - 30_000) < 100

    def test_with_jitter(self):
        rng = np.random.default_rng(7)
        onsets = np.arange(0, 30_000 * 100, 30_000, dtype=np.float64)
        onsets += rng.normal(0, 50, len(onsets))
        onsets.sort()
        sps = estimate_samples_per_second(onsets)
        assert abs(sps - 30_000) < 200


class TestClassifyPulses:
    def test_known_widths(self):
        sps = 30_000.0
        widths = np.array([
            0.2 * sps,   # binary 0
            0.5 * sps,   # binary 1
            0.8 * sps,   # marker
            0.02 * sps,  # too short -> invalid
        ], dtype=np.float64)
        types = classify_pulses(widths, sps)
        assert types[0] == PULSE_ZERO
        assert types[1] == PULSE_ONE
        assert types[2] == PULSE_MARKER
        assert types[3] == PULSE_INVALID


class TestFindFrameBoundaries:
    def test_consecutive_markers(self):
        # Simulate: ...data... P P ...data...
        types = np.array([0, 1, 0, 2, 2, 0, 1, 0, 2, 2, 0], dtype=np.int8)
        starts = find_frame_boundaries(types)
        np.testing.assert_array_equal(starts, [4, 9])

    def test_no_boundaries(self):
        types = np.array([0, 1, 2, 0, 1, 2, 0], dtype=np.int8)
        starts = find_frame_boundaries(types)
        assert len(starts) == 0


class TestBcdDecode:
    def test_zero(self):
        assert bcd_decode(np.array([0, 0, 0, 0]), np.array([1, 2, 4, 8])) == 0

    def test_seven(self):
        # 7 = 1 + 2 + 4
        assert bcd_decode(np.array([1, 1, 1, 0]), np.array([1, 2, 4, 8])) == 7

    def test_thirty_seven(self):
        # 37 = 1+2+4 + 10+20 = 7 + 30
        bits = np.array([1, 1, 1, 0, 1, 1, 0], dtype=np.float64)
        result = (
            bcd_decode(bits[0:4], SECONDS_WEIGHTS[0:4])
            + bcd_decode(bits[4:7], SECONDS_WEIGHTS[4:7])
        )
        assert result == 37


class TestDecodeFrame:
    def _make_frame(self, second=0, minute=30, hour=14, day=15, year=26):
        """Build a valid 60-element pulse_types array for the given time."""
        s_bcd = bcd_encode(second, SECONDS_WEIGHTS.tolist())
        m_bcd = bcd_encode(minute, MINUTES_WEIGHTS.tolist())
        h_bcd = bcd_encode(hour, HOURS_WEIGHTS.tolist())
        d_bcd = bcd_encode(day, DAY_OF_YEAR_WEIGHTS.tolist())
        ds_bcd = bcd_encode(0, DECISECONDS_WEIGHTS.tolist())
        y_bcd = bcd_encode(year, YEARS_WEIGHTS.tolist())

        frame = np.zeros(60, dtype=np.int8)

        # Markers
        for pos in [0, 9, 19, 29, 39, 49, 59]:
            frame[pos] = PULSE_MARKER

        # Seconds
        for i, v in enumerate(s_bcd[0:4]):
            frame[1 + i] = PULSE_ONE if v else PULSE_ZERO
        for i, v in enumerate(s_bcd[4:7]):
            frame[6 + i] = PULSE_ONE if v else PULSE_ZERO

        # Minutes
        for i, v in enumerate(m_bcd[0:4]):
            frame[10 + i] = PULSE_ONE if v else PULSE_ZERO
        for i, v in enumerate(m_bcd[4:7]):
            frame[15 + i] = PULSE_ONE if v else PULSE_ZERO

        # Hours
        for i, v in enumerate(h_bcd[0:4]):
            frame[20 + i] = PULSE_ONE if v else PULSE_ZERO
        for i, v in enumerate(h_bcd[4:6]):
            frame[25 + i] = PULSE_ONE if v else PULSE_ZERO

        # Day of year
        for i, v in enumerate(d_bcd[0:4]):
            frame[30 + i] = PULSE_ONE if v else PULSE_ZERO
        for i, v in enumerate(d_bcd[4:8]):
            frame[35 + i] = PULSE_ONE if v else PULSE_ZERO
        for i, v in enumerate(d_bcd[8:10]):
            frame[40 + i] = PULSE_ONE if v else PULSE_ZERO

        # Deciseconds (always 0)
        for i, v in enumerate(ds_bcd[0:4]):
            frame[45 + i] = PULSE_ONE if v else PULSE_ZERO

        # Year
        for i, v in enumerate(y_bcd[0:4]):
            frame[50 + i] = PULSE_ONE if v else PULSE_ZERO
        for i, v in enumerate(y_bcd[4:8]):
            frame[55 + i] = PULSE_ONE if v else PULSE_ZERO

        # Unused positions stay 0 (PULSE_ZERO) — correct
        return frame

    def test_decode_known_time(self):
        frame = self._make_frame(second=0, minute=31, hour=14, day=15, year=26)
        ts = decode_frame(frame)
        assert ts is not None
        expected = datetime(2026, 1, 15, 14, 31, 0, tzinfo=timezone.utc)
        assert abs(ts - expected.timestamp()) < 0.01

    def test_decode_nonzero_seconds(self):
        frame = self._make_frame(second=37, minute=30, hour=14, day=15, year=26)
        ts = decode_frame(frame)
        assert ts is not None
        expected = datetime(2026, 1, 15, 14, 30, 37, tzinfo=timezone.utc)
        assert abs(ts - expected.timestamp()) < 0.01

    def test_corrupt_marker_returns_none(self):
        frame = self._make_frame()
        frame[0] = PULSE_ZERO  # corrupt the Pr marker
        assert decode_frame(frame) is None

    def test_invalid_pulse_returns_none(self):
        frame = self._make_frame()
        frame[3] = PULSE_INVALID
        assert decode_frame(frame) is None

    def test_wrong_length_returns_none(self):
        assert decode_frame(np.zeros(59, dtype=np.int8)) is None


class TestDecodePartialFrame:
    """Test partial-frame decoding at a boundary between two incomplete frames."""

    def _make_frame(self, second=0, minute=30, hour=14, day=15, year=26):
        """Build a valid 60-element pulse_types array (same as TestDecodeFrame)."""
        s = bcd_encode(second, SECONDS_WEIGHTS.tolist())
        m = bcd_encode(minute, MINUTES_WEIGHTS.tolist())
        h = bcd_encode(hour, HOURS_WEIGHTS.tolist())
        d = bcd_encode(day, DAY_OF_YEAR_WEIGHTS.tolist())
        ds = bcd_encode(0, DECISECONDS_WEIGHTS.tolist())
        y = bcd_encode(year, YEARS_WEIGHTS.tolist())

        frame = np.zeros(60, dtype=np.int8)
        for pos in [0, 9, 19, 29, 39, 49, 59]:
            frame[pos] = PULSE_MARKER
        for i, v in enumerate(s[0:4]): frame[1+i] = PULSE_ONE if v else PULSE_ZERO
        for i, v in enumerate(s[4:7]): frame[6+i] = PULSE_ONE if v else PULSE_ZERO
        for i, v in enumerate(m[0:4]): frame[10+i] = PULSE_ONE if v else PULSE_ZERO
        for i, v in enumerate(m[4:7]): frame[15+i] = PULSE_ONE if v else PULSE_ZERO
        for i, v in enumerate(h[0:4]): frame[20+i] = PULSE_ONE if v else PULSE_ZERO
        for i, v in enumerate(h[4:6]): frame[25+i] = PULSE_ONE if v else PULSE_ZERO
        for i, v in enumerate(d[0:4]): frame[30+i] = PULSE_ONE if v else PULSE_ZERO
        for i, v in enumerate(d[4:8]): frame[35+i] = PULSE_ONE if v else PULSE_ZERO
        for i, v in enumerate(d[8:10]): frame[40+i] = PULSE_ONE if v else PULSE_ZERO
        for i, v in enumerate(ds[0:4]): frame[45+i] = PULSE_ONE if v else PULSE_ZERO
        for i, v in enumerate(y[0:4]): frame[50+i] = PULSE_ONE if v else PULSE_ZERO
        for i, v in enumerate(y[4:8]): frame[55+i] = PULSE_ONE if v else PULSE_ZERO
        return frame

    def test_year_from_previous_frame(self):
        """Short recording: time fields from new frame, year from previous."""
        prev = self._make_frame(second=0, minute=30, hour=14, day=15, year=26)
        curr = self._make_frame(second=0, minute=31, hour=14, day=15, year=26)

        # Simulate: last 27 pulses of prev frame + first 48 of curr frame = 75 pulses
        # (mirrors the real IRIG_Only.dat scenario)
        pulse_types = np.concatenate([prev[33:60], curr[0:48]])
        boundary_idx = 27  # curr frame starts at pulse 27

        ts = decode_partial_frame(pulse_types, boundary_idx)
        assert ts is not None
        expected = datetime(2026, 1, 15, 14, 31, 0, tzinfo=timezone.utc)
        assert abs(ts - expected.timestamp()) < 0.01

    def test_all_from_new_frame(self):
        """New frame has all 60 bits — partial decoder should still work."""
        prev = self._make_frame(second=0, minute=30, hour=14, day=15, year=26)
        curr = self._make_frame(second=0, minute=31, hour=14, day=15, year=26)

        pulse_types = np.concatenate([prev[33:60], curr])
        boundary_idx = 27
        ts = decode_partial_frame(pulse_types, boundary_idx)
        assert ts is not None
        expected = datetime(2026, 1, 15, 14, 31, 0, tzinfo=timezone.utc)
        assert abs(ts - expected.timestamp()) < 0.01

    def test_not_enough_data_returns_none(self):
        """Only 10 pulses after boundary — can't even get hours."""
        curr = self._make_frame(second=0, minute=31, hour=14, day=15, year=26)
        pulse_types = curr[0:10]
        ts = decode_partial_frame(pulse_types, 0)
        assert ts is None

    def test_day_from_previous_when_new_too_short(self):
        """Day bits beyond new frame range, recovered from previous frame."""
        prev = self._make_frame(second=0, minute=59, hour=23, day=15, year=26)
        curr = self._make_frame(second=0, minute=0, hour=0, day=15, year=26)

        # Only 30 bits of new frame (enough for seconds/minutes/hours but not day)
        # Plus 30 bits of previous frame tail (bits 30-59, has day and year)
        pulse_types = np.concatenate([prev[30:60], curr[0:30]])
        boundary_idx = 30

        ts = decode_partial_frame(pulse_types, boundary_idx)
        assert ts is not None
        expected = datetime(2026, 1, 15, 0, 0, 0, tzinfo=timezone.utc)
        assert abs(ts - expected.timestamp()) < 0.01


class TestComputeElapsedSeconds:
    def test_uniform_spacing(self):
        """All intervals = sps -> all elapsed = 1."""
        sps = 30_000.0
        onsets = np.arange(0, 10 * sps, sps, dtype=np.float64)
        elapsed = compute_elapsed_seconds(onsets, sps)
        np.testing.assert_array_equal(elapsed, np.ones(9, dtype=np.int64))

    def test_double_gap(self):
        """A 2x gap rounds to 2 (missing pulse)."""
        sps = 30_000.0
        onsets = np.array([0, sps, 3 * sps, 4 * sps], dtype=np.float64)
        elapsed = compute_elapsed_seconds(onsets, sps)
        np.testing.assert_array_equal(elapsed, [1, 2, 1])

    def test_half_gap(self):
        """A ~0.3x gap rounds to 0 (extra pulse)."""
        sps = 30_000.0
        onsets = np.array([0, 0.3 * sps, sps, 2 * sps], dtype=np.float64)
        elapsed = compute_elapsed_seconds(onsets, sps)
        np.testing.assert_array_equal(elapsed, [0, 1, 1])

    def test_jitter_still_rounds_to_one(self):
        """Jitter within +/-0.3 sps still rounds to 1."""
        sps = 30_000.0
        onsets = np.array([0, sps + 50, 2 * sps - 30, 3 * sps + 100],
                          dtype=np.float64)
        elapsed = compute_elapsed_seconds(onsets, sps)
        np.testing.assert_array_equal(elapsed, [1, 1, 1])

    def test_triple_gap(self):
        """A 3x gap rounds to 3 (two missing pulses)."""
        sps = 30_000.0
        onsets = np.array([0, 3 * sps], dtype=np.float64)
        elapsed = compute_elapsed_seconds(onsets, sps)
        np.testing.assert_array_equal(elapsed, [3])


class TestRemoveExtraPulses:
    def test_single_extra_removed(self):
        """One extra pulse (elapsed=0) is removed, keeping the valid one."""
        sps = 30_000.0
        # Pulse 0: real, pulse 1: noise spike (invalid) at 0.3s, pulse 2: real
        onsets = np.array([0, 0.3 * sps, sps], dtype=np.float64)
        widths = np.array([0.2 * sps, 0.01 * sps, 0.5 * sps], dtype=np.float64)
        types = np.array([PULSE_ZERO, PULSE_INVALID, PULSE_ONE], dtype=np.int8)
        elapsed = compute_elapsed_seconds(onsets, sps)

        o, w, t, e = remove_extra_pulses(onsets, widths, types, elapsed, sps)
        assert len(o) == 2
        np.testing.assert_array_equal(t, [PULSE_ZERO, PULSE_ONE])

    def test_invalid_preferentially_removed(self):
        """When both are valid types, keep the one closer to expected onset."""
        sps = 30_000.0
        # Two pulses very close: one valid at expected position, one invalid
        onsets = np.array([0, sps, sps + 100], dtype=np.float64)
        widths = np.array([0.2 * sps, 0.5 * sps, 0.01 * sps], dtype=np.float64)
        types = np.array([PULSE_ZERO, PULSE_ONE, PULSE_INVALID], dtype=np.int8)
        elapsed = compute_elapsed_seconds(onsets, sps)

        o, w, t, e = remove_extra_pulses(onsets, widths, types, elapsed, sps)
        assert len(o) == 2
        assert t[1] == PULSE_ONE  # kept the valid one

    def test_no_extras_unchanged(self):
        """No extra pulses -> arrays unchanged."""
        sps = 30_000.0
        onsets = np.arange(0, 5 * sps, sps, dtype=np.float64)
        widths = np.full(5, 0.2 * sps, dtype=np.float64)
        types = np.full(5, PULSE_ZERO, dtype=np.int8)
        elapsed = compute_elapsed_seconds(onsets, sps)

        o, w, t, e = remove_extra_pulses(onsets, widths, types, elapsed, sps)
        assert len(o) == 5
        np.testing.assert_array_equal(o, onsets)

    def test_consecutive_extras(self):
        """Multiple extras in a cluster -> only one survives."""
        sps = 30_000.0
        # Pulse 0: real, pulses 1-2: noise spikes, pulse 3: real at 1.0s
        onsets = np.array([0, 0.2 * sps, 0.4 * sps, sps], dtype=np.float64)
        widths = np.array([0.2 * sps, 0.01 * sps, 0.01 * sps, 0.5 * sps],
                          dtype=np.float64)
        types = np.array([PULSE_ZERO, PULSE_INVALID, PULSE_INVALID, PULSE_ONE],
                         dtype=np.int8)
        elapsed = compute_elapsed_seconds(onsets, sps)

        o, w, t, e = remove_extra_pulses(onsets, widths, types, elapsed, sps)
        assert len(o) == 2
        np.testing.assert_array_equal(t, [PULSE_ZERO, PULSE_ONE])


class TestAssignTimestampsFromAnchors:
    def test_single_anchor_clean(self):
        """Single anchor with uniform spacing -> simple propagation."""
        sps = 30_000.0
        n = 10
        onsets = np.arange(0, n * sps, sps, dtype=np.float64)
        elapsed = np.ones(n - 1, dtype=np.int64)
        decoded = [(5, 1000.0)]  # anchor at pulse 5 = time 1000

        ref = assign_timestamps_from_anchors(onsets, elapsed, decoded, sps)
        expected = 1000.0 + np.arange(n) - 5
        np.testing.assert_allclose(ref, expected)

    def test_two_anchors_clean(self):
        """Two consistent anchors -> same result as single anchor."""
        sps = 30_000.0
        n = 20
        onsets = np.arange(0, n * sps, sps, dtype=np.float64)
        elapsed = np.ones(n - 1, dtype=np.int64)
        decoded = [(5, 1000.0), (15, 1010.0)]

        ref = assign_timestamps_from_anchors(onsets, elapsed, decoded, sps)
        expected = 1000.0 + np.arange(n) - 5
        np.testing.assert_allclose(ref, expected)

    def test_missing_pulse_gap(self):
        """Missing pulse (elapsed=2) skips correctly in timestamp sequence."""
        sps = 30_000.0
        # 5 pulses, but pulse 2->3 has a 2-second gap (missing pulse)
        onsets = np.array([0, sps, 2 * sps, 4 * sps, 5 * sps],
                          dtype=np.float64)
        elapsed = np.array([1, 1, 2, 1], dtype=np.int64)
        decoded = [(0, 1000.0)]

        ref = assign_timestamps_from_anchors(onsets, elapsed, decoded, sps)
        np.testing.assert_allclose(ref, [1000, 1001, 1002, 1004, 1005])

    def test_concatenation_gap(self):
        """Time jump between anchors -> detected as concatenation boundary."""
        sps = 30_000.0
        n = 20
        onsets = np.arange(0, n * sps, sps, dtype=np.float64)
        elapsed = np.ones(n - 1, dtype=np.int64)
        # Anchors with a 3600-second time gap
        decoded = [(5, 1000.0), (15, 4610.0)]

        ref = assign_timestamps_from_anchors(onsets, elapsed, decoded, sps)
        # Before boundary (midpoint ~10): propagated from anchor A
        np.testing.assert_allclose(ref[0:6], [995, 996, 997, 998, 999, 1000])
        # After boundary: propagated from anchor B
        np.testing.assert_allclose(ref[15:20], [4610, 4611, 4612, 4613, 4614])
        # There should be a jump in ref at the boundary
        diffs = np.diff(ref)
        assert np.max(diffs) > 100
