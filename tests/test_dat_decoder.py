from pathlib import Path

import numpy as np
import pytest

from neurokairos import ClockTable, decode_dat_irig


class TestDecodeDatIrig:
    def test_returns_clock_table(self, generate_test_dat):
        meta = generate_test_dat
        ct = decode_dat_irig(meta["path"], meta["n_channels"], meta["irig_channel"])
        assert isinstance(ct, ClockTable)

    def test_entry_count(self, generate_test_dat):
        meta = generate_test_dat
        ct = decode_dat_irig(meta["path"], meta["n_channels"], meta["irig_channel"])
        assert len(ct.source) == meta["duration_s"]

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
        assert abs(ct.nominal_rate - meta["sample_rate"]) < 100

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
        assert ct.metadata["n_raw_pulses"] == meta["duration_s"]
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
        assert len(ct.source) == meta["duration_s"]

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
        # Reference diffs should be 1.0 everywhere EXCEPT at the boundary
        ref_diffs = np.diff(ct.reference)
        boundary = meta["seg1_duration"] - 1  # diff index at the junction
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
