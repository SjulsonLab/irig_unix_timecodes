import datetime as _dt
from pathlib import Path

import numpy as np
import pytest

from neurokairos import ClockTable, decode_dat_irig
from neurokairos.decoders.irig import (
    bcd_encode, build_clock_table,
    SECONDS_WEIGHTS, MINUTES_WEIGHTS, HOURS_WEIGHTS,
    DAY_OF_YEAR_WEIGHTS, YEARS_WEIGHTS,
)


class TestDecodeDatIrig:
    def test_returns_clock_table(self, generate_test_dat):
        meta = generate_test_dat
        ct = decode_dat_irig(meta["path"], meta["n_channels"], meta["irig_channel"])
        assert isinstance(ct, ClockTable)

    def test_entry_count(self, generate_test_dat):
        meta = generate_test_dat
        ct = decode_dat_irig(meta["path"], meta["n_channels"], meta["irig_channel"])
        assert len(ct.source) == len(meta["expected_pairs"])

    def test_monotonically_increasing(self, generate_test_dat):
        meta = generate_test_dat
        ct = decode_dat_irig(meta["path"], meta["n_channels"], meta["irig_channel"])
        assert np.all(np.diff(ct.source) > 0)
        assert np.all(np.diff(ct.reference) > 0)

    def test_one_second_spacing(self, generate_test_dat):
        meta = generate_test_dat
        ct = decode_dat_irig(meta["path"], meta["n_channels"], meta["irig_channel"])
        ref_diffs = np.diff(ct.reference)
        np.testing.assert_allclose(ref_diffs, 1.0, atol=0.01)

    def test_nominal_rate(self, generate_test_dat):
        meta = generate_test_dat
        ct = decode_dat_irig(meta["path"], meta["n_channels"], meta["irig_channel"])
        # Linear-fit rate on perfect 30 kHz synthetic data should be very
        # close to the true rate — within 1 Hz, not the 100 Hz tolerance
        # acceptable for the old median estimate.
        assert abs(ct.nominal_rate - meta["sample_rate"]) < 1

    def test_decoded_timestamps_match_expected(self, generate_test_dat):
        meta = generate_test_dat
        ct = decode_dat_irig(meta["path"], meta["n_channels"], meta["irig_channel"])

        expected = meta["expected_pairs"]
        for i, (exp_src, exp_ref) in enumerate(expected):
            # Source (rising-edge sample index) should be very close
            assert abs(ct.source[i] - exp_src) < 5, (
                f"pulse {i}: source {ct.source[i]} != expected {exp_src}"
            )
            # Reference (UTC timestamp) should match exactly
            assert abs(ct.reference[i] - exp_ref) < 0.01, (
                f"pulse {i}: reference {ct.reference[i]} != expected {exp_ref}"
            )

    def test_source_to_reference_at_rising_edges(self, generate_test_dat):
        meta = generate_test_dat
        ct = decode_dat_irig(meta["path"], meta["n_channels"], meta["irig_channel"])

        # Query at known rising-edge sample positions
        sample_rate = meta["sample_rate"]
        query_samples = np.array([0, 60, 120]) * sample_rate
        expected_times = np.array([
            meta["start_timestamp"],
            meta["start_timestamp"] + 60,
            meta["start_timestamp"] + 120,
        ])
        result = ct.source_to_reference(query_samples.astype(np.float64))
        np.testing.assert_allclose(result, expected_times, atol=0.1)

    def test_source_units(self, generate_test_dat):
        meta = generate_test_dat
        ct = decode_dat_irig(meta["path"], meta["n_channels"], meta["irig_channel"])
        assert ct.source_units == "samples"

    def test_metadata_provenance(self, generate_test_dat):
        meta = generate_test_dat
        ct = decode_dat_irig(meta["path"], meta["n_channels"], meta["irig_channel"])
        assert isinstance(ct.metadata, dict)
        assert ct.metadata["source_file"] == Path(meta["path"]).name
        assert ct.metadata["n_channels"] == meta["n_channels"]
        assert ct.metadata["irig_channel"] == meta["irig_channel"]

    def test_metadata_decoding_stats(self, generate_test_dat):
        meta = generate_test_dat
        ct = decode_dat_irig(meta["path"], meta["n_channels"], meta["irig_channel"])
        # One fewer raw pulse: the signal starts HIGH so the first pulse
        # has no detectable rising edge.
        assert ct.metadata["n_raw_pulses"] == len(meta["expected_pairs"])
        assert ct.metadata["n_extra_removed"] == 0
        assert ct.metadata["n_missing_gaps"] == 0
        assert ct.metadata["n_frames_decoded"] >= 1
        assert ct.metadata["n_concat_boundaries"] == 0


class TestDecodeDatIrigMissingPulses:
    def test_returns_clock_table(self, generate_test_dat_missing_pulses):
        meta = generate_test_dat_missing_pulses
        ct = decode_dat_irig(meta["path"], meta["n_channels"], meta["irig_channel"])
        assert isinstance(ct, ClockTable)

    def test_fewer_entries_than_clean(self, generate_test_dat_missing_pulses):
        meta = generate_test_dat_missing_pulses
        ct = decode_dat_irig(meta["path"], meta["n_channels"], meta["irig_channel"])
        assert len(ct.source) == meta["n_surviving"]

    def test_monotonically_increasing(self, generate_test_dat_missing_pulses):
        meta = generate_test_dat_missing_pulses
        ct = decode_dat_irig(meta["path"], meta["n_channels"], meta["irig_channel"])
        assert np.all(np.diff(ct.source) > 0)
        assert np.all(np.diff(ct.reference) > 0)

    def test_surviving_timestamps_correct(self, generate_test_dat_missing_pulses):
        meta = generate_test_dat_missing_pulses
        ct = decode_dat_irig(meta["path"], meta["n_channels"], meta["irig_channel"])

        for i, (exp_src, exp_ref) in enumerate(meta["expected_pairs"]):
            assert abs(ct.source[i] - exp_src) < 5, (
                f"pulse {i}: source {ct.source[i]} != expected {exp_src}")
            assert abs(ct.reference[i] - exp_ref) < 0.01, (
                f"pulse {i}: reference {ct.reference[i]} != expected {exp_ref}")

    def test_interpolation_spans_gaps(self, generate_test_dat_missing_pulses):
        meta = generate_test_dat_missing_pulses
        ct = decode_dat_irig(meta["path"], meta["n_channels"], meta["irig_channel"])
        # Query at the sample position where a deleted pulse would have been
        deleted = meta["deleted_indices"][0]
        query = np.array([float(deleted * meta["sample_rate"])])
        result = ct.source_to_reference(query)
        expected = meta["start_timestamp"] + float(deleted)
        np.testing.assert_allclose(result, [expected], atol=0.1)

    def test_metadata_missing_gaps(self, generate_test_dat_missing_pulses):
        meta = generate_test_dat_missing_pulses
        ct = decode_dat_irig(meta["path"], meta["n_channels"], meta["irig_channel"])
        assert ct.metadata["n_missing_gaps"] > 0


class TestDecodeDatIrigExtraPulses:
    def test_returns_clock_table(self, generate_test_dat_extra_pulses):
        meta = generate_test_dat_extra_pulses
        ct = decode_dat_irig(meta["path"], meta["n_channels"], meta["irig_channel"])
        assert isinstance(ct, ClockTable)

    def test_same_count_as_clean(self, generate_test_dat_extra_pulses):
        meta = generate_test_dat_extra_pulses
        ct = decode_dat_irig(meta["path"], meta["n_channels"], meta["irig_channel"])
        assert len(ct.source) == len(meta["expected_pairs"])

    def test_timestamps_match_clean(self, generate_test_dat_extra_pulses):
        meta = generate_test_dat_extra_pulses
        ct = decode_dat_irig(meta["path"], meta["n_channels"], meta["irig_channel"])

        for i, (exp_src, exp_ref) in enumerate(meta["expected_pairs"]):
            assert abs(ct.source[i] - exp_src) < 5, (
                f"pulse {i}: source {ct.source[i]} != expected {exp_src}")
            assert abs(ct.reference[i] - exp_ref) < 0.01, (
                f"pulse {i}: reference {ct.reference[i]} != expected {exp_ref}")

    def test_one_second_spacing(self, generate_test_dat_extra_pulses):
        meta = generate_test_dat_extra_pulses
        ct = decode_dat_irig(meta["path"], meta["n_channels"], meta["irig_channel"])
        ref_diffs = np.diff(ct.reference)
        np.testing.assert_allclose(ref_diffs, 1.0, atol=0.01)

    def test_metadata_extra_removed(self, generate_test_dat_extra_pulses):
        meta = generate_test_dat_extra_pulses
        ct = decode_dat_irig(meta["path"], meta["n_channels"], meta["irig_channel"])
        assert ct.metadata["n_extra_removed"] > 0


class TestDecodeDatIrigConcatenated:
    def test_returns_clock_table(self, generate_test_dat_concatenated):
        meta = generate_test_dat_concatenated
        ct = decode_dat_irig(meta["path"], meta["n_channels"], meta["irig_channel"])
        assert isinstance(ct, ClockTable)

    def test_entry_count(self, generate_test_dat_concatenated):
        meta = generate_test_dat_concatenated
        ct = decode_dat_irig(meta["path"], meta["n_channels"], meta["irig_channel"])
        assert len(ct.source) == meta["total_pulses"]

    def test_source_monotonic(self, generate_test_dat_concatenated):
        meta = generate_test_dat_concatenated
        ct = decode_dat_irig(meta["path"], meta["n_channels"], meta["irig_channel"])
        assert np.all(np.diff(ct.source) > 0)

    def test_reference_monotonic(self, generate_test_dat_concatenated):
        meta = generate_test_dat_concatenated
        ct = decode_dat_irig(meta["path"], meta["n_channels"], meta["irig_channel"])
        assert np.all(np.diff(ct.reference) > 0)

    def test_time_jump_at_boundary(self, generate_test_dat_concatenated):
        meta = generate_test_dat_concatenated
        ct = decode_dat_irig(meta["path"], meta["n_channels"], meta["irig_channel"])
        # Reference diffs should be 1.0 everywhere EXCEPT at the boundary.
        # Seg1 has seg1_duration - 1 entries (pulse 0 lost — starts HIGH),
        # so the boundary diff is at index seg1_duration - 2.
        ref_diffs = np.diff(ct.reference)
        boundary = meta["seg1_duration"] - 2  # diff index at the junction
        # All diffs except boundary should be ~1.0
        normal = np.concatenate([ref_diffs[:boundary], ref_diffs[boundary + 1:]])
        np.testing.assert_allclose(normal, 1.0, atol=0.01)
        # The boundary diff should be the time gap
        seg1_end_time = meta["start_time_seg1"].timestamp() + meta["seg1_duration"] - 1
        seg2_start_time = meta["start_time_seg2"].timestamp()
        expected_gap = seg2_start_time - seg1_end_time
        assert abs(ref_diffs[boundary] - expected_gap) < 1.0

    def test_timestamps_correct_both_sides(self, generate_test_dat_concatenated):
        meta = generate_test_dat_concatenated
        ct = decode_dat_irig(meta["path"], meta["n_channels"], meta["irig_channel"])

        for i, (exp_src, exp_ref) in enumerate(meta["expected_pairs"]):
            assert abs(ct.source[i] - exp_src) < 5, (
                f"pulse {i}: source {ct.source[i]} != expected {exp_src}")
            assert abs(ct.reference[i] - exp_ref) < 0.01, (
                f"pulse {i}: reference {ct.reference[i]} != expected {exp_ref}")

    def test_metadata_concat_boundaries(self, generate_test_dat_concatenated):
        meta = generate_test_dat_concatenated
        ct = decode_dat_irig(meta["path"], meta["n_channels"], meta["irig_channel"])
        assert ct.metadata["n_concat_boundaries"] >= 1


# -- Linear-fit rate tests (standalone functions) -----------------------------

def _generate_irig_frame(t):
    """Generate a 60-element IRIG-H frame for datetime *t*.

    Returns a list of pulse-width fractions (0.2, 0.5, or 0.8).
    """
    sec_bcd = bcd_encode(t.second, SECONDS_WEIGHTS)
    min_bcd = bcd_encode(t.minute, MINUTES_WEIGHTS)
    hr_bcd = bcd_encode(t.hour, HOURS_WEIGHTS)
    doy_bcd = bcd_encode(t.timetuple().tm_yday, DAY_OF_YEAR_WEIGHTS)
    yr_bcd = bcd_encode(t.year % 100, YEARS_WEIGHTS)

    # Build bit list: True/False for data, "P" for markers
    bits = []
    bits.append("P")                       # 0: Pr
    bits.extend(sec_bcd[0:4])              # 1-4
    bits.append(False)                     # 5: unused
    bits.extend(sec_bcd[4:7])              # 6-8
    bits.append("P")                       # 9: P1
    bits.extend(min_bcd[0:4])              # 10-13
    bits.append(False)                     # 14: unused
    bits.extend(min_bcd[4:7])              # 15-17
    bits.append(False)                     # 18: unused
    bits.append("P")                       # 19: P2
    bits.extend(hr_bcd[0:4])              # 20-23
    bits.append(False)                     # 24: unused
    bits.extend(hr_bcd[4:6])              # 25-26
    bits.extend([False, False])            # 27-28: unused
    bits.append("P")                       # 29: P3
    bits.extend(doy_bcd[0:4])             # 30-33
    bits.append(False)                     # 34: unused
    bits.extend(doy_bcd[4:8])             # 35-38
    bits.append("P")                       # 39: P4
    bits.extend(doy_bcd[8:10])            # 40-41
    bits.extend([False, False, False])     # 42-44: unused
    bits.extend([False, False, False, False])  # 45-48: deciseconds (0)
    bits.append("P")                       # 49: P5
    bits.extend(yr_bcd[0:4])              # 50-53
    bits.append(False)                     # 54: unused
    bits.extend(yr_bcd[4:8])              # 55-58
    bits.append("P")                       # 59: P6

    # Convert bits to pulse-width fractions
    fracs = []
    for b in bits:
        if b == "P":
            fracs.append(0.8)
        elif b:
            fracs.append(0.5)
        else:
            fracs.append(0.2)
    return fracs


def test_nominal_rate_mean_accuracy():
    """Mean-based rate should recover fractional sample rates that median cannot.

    With true_rate=30000.7 sps, inter-onset intervals alternate between
    30000 and 30001 samples (integer rounding). The median of these
    intervals is 30001 (since >50% round up), but the mean-based rate
    (total span / elapsed seconds) recovers ~30000.7.
    """
    true_rate = 30000.7
    start_dt = _dt.datetime(2026, 1, 15, 14, 0, 0, tzinfo=_dt.timezone.utc)
    n_seconds = 180  # 3 full frames

    # Generate IRIG frame bits for 3 minutes
    all_fracs = []
    for frame_i in range(3):
        frame_dt = start_dt + _dt.timedelta(seconds=frame_i * 60)
        all_fracs.extend(_generate_irig_frame(frame_dt))

    # Build pulse onsets and widths at the fractional rate
    pulse_onsets = np.array(
        [round(i * true_rate) for i in range(n_seconds)], dtype=np.float64
    )
    pulse_widths = np.array(
        [round(all_fracs[i] * true_rate) for i in range(n_seconds)],
        dtype=np.float64,
    )

    ct = build_clock_table(pulse_onsets, pulse_widths)

    # Mean-based rate should recover the true rate to within 0.1 Hz
    assert abs(ct.nominal_rate - true_rate) < 0.1, (
        f"nominal_rate {ct.nominal_rate:.4f} should be within 0.1 Hz "
        f"of true rate {true_rate} (median would give ~30001)"
    )


def test_drift_residuals_zero_mean_with_fractional_rate():
    """Drift residuals should be near-zero-mean with a mean-based rate.

    With a median-based rate on fractional-rate data, residuals accumulate
    a systematic bias. The mean-based rate (source span / reference span)
    eliminates this bias.
    """
    true_rate = 30000.7
    start_dt = _dt.datetime(2026, 1, 15, 14, 0, 0, tzinfo=_dt.timezone.utc)
    n_seconds = 180

    # Generate IRIG frame bits
    all_fracs = []
    for frame_i in range(3):
        frame_dt = start_dt + _dt.timedelta(seconds=frame_i * 60)
        all_fracs.extend(_generate_irig_frame(frame_dt))

    pulse_onsets = np.array(
        [round(i * true_rate) for i in range(n_seconds)], dtype=np.float64
    )
    pulse_widths = np.array(
        [round(all_fracs[i] * true_rate) for i in range(n_seconds)],
        dtype=np.float64,
    )

    ct = build_clock_table(pulse_onsets, pulse_widths)

    # Compute drift residuals (same as report._draw_clock_drift)
    expected = ct.reference[0] + (ct.source - ct.source[0]) / ct.nominal_rate
    residual_ms = (ct.reference - expected) * 1000.0

    # With a mean-based rate, mean residual should be ~0.
    # With median rate (30001), mean residual would be ~0.9 ms.
    assert abs(np.mean(residual_ms)) < 0.01, (
        f"Mean drift residual {np.mean(residual_ms):.4f} ms should be ~0 "
        f"with mean-based rate"
    )
