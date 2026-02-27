import os
import warnings

import numpy as np
import pytest

from neurokairos.clock_table import ClockTable, _EXTRAP_LIMIT_S


class TestConstruction:
    def test_basic(self):
        ct = ClockTable(
            source=np.array([0.0, 30000.0, 60000.0]),
            reference=np.array([1000.0, 1001.0, 1002.0]),
            nominal_rate=30000.0,
        )
        assert len(ct.source) == 3
        assert ct.nominal_rate == 30000.0
        assert ct.source_units is None

    def test_with_source_units(self):
        ct = ClockTable(
            source=np.array([0.0, 1.0]),
            reference=np.array([100.0, 101.0]),
            nominal_rate=1.0,
            source_units="frames",
        )
        assert ct.source_units == "frames"

    def test_repr(self):
        ct = ClockTable(
            source=np.array([0.0, 30000.0]),
            reference=np.array([1000.0, 1001.0]),
            nominal_rate=30000.0,
            source_units="samples",
        )
        r = repr(ct)
        assert "2 entries" in r
        assert "samples" in r
        assert "30000.0" in r
        assert "source=" in r
        assert "reference=" in r
        # Should show both start and stop
        assert "recording:" in r
        assert "\u2192" in r


class TestValidation:
    def test_too_few_entries(self):
        with pytest.raises(ValueError, match="at least 2"):
            ClockTable(
                source=np.array([0.0]),
                reference=np.array([1000.0]),
                nominal_rate=1.0,
            )

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            ClockTable(
                source=np.array([0.0, 1.0, 2.0]),
                reference=np.array([100.0, 101.0]),
                nominal_rate=1.0,
            )

    def test_source_not_monotonic(self):
        with pytest.raises(ValueError, match="source.*monotonically"):
            ClockTable(
                source=np.array([0.0, 2.0, 1.0]),
                reference=np.array([100.0, 101.0, 102.0]),
                nominal_rate=1.0,
            )

    def test_reference_not_monotonic(self):
        with pytest.raises(ValueError, match="reference.*monotonically"):
            ClockTable(
                source=np.array([0.0, 1.0, 2.0]),
                reference=np.array([100.0, 102.0, 101.0]),
                nominal_rate=1.0,
            )


class TestInterpolation:
    @pytest.fixture
    def ct(self):
        return ClockTable(
            source=np.array([0.0, 30000.0, 60000.0]),
            reference=np.array([1000.0, 1001.0, 1002.0]),
            nominal_rate=30000.0,
        )

    def test_source_to_reference_exact(self, ct):
        result = ct.source_to_reference(np.array([0.0, 30000.0, 60000.0]))
        np.testing.assert_allclose(result, [1000.0, 1001.0, 1002.0])

    def test_source_to_reference_interpolated(self, ct):
        result = ct.source_to_reference(np.array([15000.0]))
        np.testing.assert_allclose(result, [1000.5])

    def test_reference_to_source_exact(self, ct):
        result = ct.reference_to_source(np.array([1000.0, 1001.0, 1002.0]))
        np.testing.assert_allclose(result, [0.0, 30000.0, 60000.0])

    def test_reference_to_source_interpolated(self, ct):
        result = ct.reference_to_source(np.array([1000.5]))
        np.testing.assert_allclose(result, [15000.0])

    def test_round_trip(self, ct):
        sources = np.array([0.0, 10000.0, 25000.0, 45000.0, 60000.0])
        refs = ct.source_to_reference(sources)
        recovered = ct.reference_to_source(refs)
        np.testing.assert_allclose(recovered, sources, atol=1e-6)

    def test_scalar_input(self, ct):
        result = ct.source_to_reference(15000.0)
        assert isinstance(result, (float, np.floating))
        np.testing.assert_allclose(result, 1000.5)

    def test_scalar_input_reference_to_source(self, ct):
        result = ct.reference_to_source(1000.5)
        assert isinstance(result, (float, np.floating))
        np.testing.assert_allclose(result, 15000.0)


class TestExtrapolation:
    """Extrapolation behavior when values fall outside the ClockTable range."""

    @pytest.fixture
    def ct_offset(self):
        """ClockTable where source starts at 30000 (1 s into a 30 kHz recording).

        This simulates a recording where the first IRIG pulse is detected
        1 second after the recording starts.
        """
        return ClockTable(
            source=np.array([30000.0, 60000.0, 90000.0]),
            reference=np.array([1000.0, 1001.0, 1002.0]),
            nominal_rate=30000.0,
        )

    def test_source_to_reference_extrapolates_below(self, ct_offset):
        """source_to_reference(0.0) should extrapolate back to sample 0."""
        result = ct_offset.source_to_reference(0.0)
        # slope = 1/30000 s/sample, so 30000 samples back = 1 second
        np.testing.assert_allclose(result, 999.0)

    def test_source_to_reference_extrapolates_above(self, ct_offset):
        result = ct_offset.source_to_reference(105000.0)
        # 15000 samples beyond source[-1], slope = 1/30000
        np.testing.assert_allclose(result, 1002.5)

    def test_reference_to_source_extrapolates_below(self, ct_offset):
        result = ct_offset.reference_to_source(999.0)
        np.testing.assert_allclose(result, 0.0)

    def test_reference_to_source_extrapolates_above(self, ct_offset):
        result = ct_offset.reference_to_source(1002.5)
        np.testing.assert_allclose(result, 105000.0)

    def test_extrapolation_round_trip(self, ct_offset):
        """Extrapolated values should round-trip correctly."""
        ref = ct_offset.source_to_reference(0.0)
        recovered = ct_offset.reference_to_source(ref)
        np.testing.assert_allclose(recovered, 0.0, atol=1e-6)

    def test_no_warning_within_limit(self, ct_offset):
        """Extrapolation within 1.5 s should not warn."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ct_offset.source_to_reference(0.0)  # 1.0 s extrapolation

    def test_warning_beyond_limit(self):
        """Extrapolation beyond 1.5 s should warn and clamp."""
        ct = ClockTable(
            source=np.array([90000.0, 120000.0, 150000.0]),
            reference=np.array([1003.0, 1004.0, 1005.0]),
            nominal_rate=30000.0,
        )
        # source=0.0 is 90000 samples = 3.0 s before first entry
        with pytest.warns(UserWarning, match="exceeds the"):
            result = ct.source_to_reference(0.0)
        # Should clamp to boundary value
        np.testing.assert_allclose(result, 1003.0)

    def test_warning_beyond_limit_above(self):
        ct = ClockTable(
            source=np.array([0.0, 30000.0, 60000.0]),
            reference=np.array([1000.0, 1001.0, 1002.0]),
            nominal_rate=30000.0,
        )
        # 120000 samples = 2.0 s beyond last entry
        with pytest.warns(UserWarning, match="exceeds the"):
            result = ct.source_to_reference(120000.0)
        np.testing.assert_allclose(result, 1002.0)

    def test_reference_to_source_warning_beyond_limit(self):
        ct = ClockTable(
            source=np.array([0.0, 30000.0, 60000.0]),
            reference=np.array([1000.0, 1001.0, 1002.0]),
            nominal_rate=30000.0,
        )
        with pytest.warns(UserWarning, match="exceeds the"):
            result = ct.reference_to_source(998.0)  # 2.0 s before
        np.testing.assert_allclose(result, 0.0)

    def test_array_extrapolation(self, ct_offset):
        """Extrapolation works correctly for arrays with mixed in/out values."""
        values = np.array([0.0, 30000.0, 60000.0, 90000.0])
        result = ct_offset.source_to_reference(values)
        np.testing.assert_allclose(result, [999.0, 1000.0, 1001.0, 1002.0])


class TestMetadata:
    def test_default_none(self):
        ct = ClockTable(
            source=np.array([0.0, 1.0]),
            reference=np.array([100.0, 101.0]),
            nominal_rate=1.0,
        )
        assert ct.metadata is None

    def test_with_dict(self):
        meta = {"source_file": "test.dat", "n_channels": 3}
        ct = ClockTable(
            source=np.array([0.0, 1.0]),
            reference=np.array([100.0, 101.0]),
            nominal_rate=1.0,
            metadata=meta,
        )
        assert ct.metadata == meta

    def test_rejects_non_dict(self):
        with pytest.raises(TypeError, match="metadata must be a dict"):
            ClockTable(
                source=np.array([0.0, 1.0]),
                reference=np.array([100.0, 101.0]),
                nominal_rate=1.0,
                metadata="not a dict",
            )

    def test_save_load_with_metadata(self, tmp_path):
        meta = {
            "source_file": "recording.dat",
            "n_channels": 3,
            "nested": {"key": [1, 2, 3]},
            "flag": True,
            "nothing": None,
        }
        ct = ClockTable(
            source=np.array([0.0, 30000.0]),
            reference=np.array([1737000000.0, 1737000001.0]),
            nominal_rate=30000.0,
            metadata=meta,
        )
        path = tmp_path / "test.clocktable.npz"
        ct.save(path)
        loaded = ClockTable.load(path)
        assert loaded.metadata == meta

    def test_save_load_without_metadata(self, tmp_path):
        ct = ClockTable(
            source=np.array([0.0, 1.0]),
            reference=np.array([100.0, 101.0]),
            nominal_rate=1.0,
        )
        path = tmp_path / "test.clocktable.npz"
        ct.save(path)
        loaded = ClockTable.load(path)
        assert loaded.metadata is None

    def test_non_serializable_metadata_raises(self, tmp_path):
        ct = ClockTable(
            source=np.array([0.0, 1.0]),
            reference=np.array([100.0, 101.0]),
            nominal_rate=1.0,
            metadata={"bad": object()},
        )
        with pytest.raises(TypeError, match="JSON-serializable"):
            ct.save(tmp_path / "test.npz")

    def test_repr_includes_utc_datetime(self):
        # 1737000000.0 ~ 2025-01-16 02:40:00 UTC
        ct = ClockTable(
            source=np.array([0.0, 30000.0]),
            reference=np.array([1737000000.0, 1737000001.0]),
            nominal_rate=30000.0,
        )
        r = repr(ct)
        assert "2025-01-16" in r
        assert "\u2192" in r  # shows both start and stop

    def test_repr_includes_source_file_and_path(self):
        ct = ClockTable(
            source=np.array([0.0, 30000.0]),
            reference=np.array([1737000000.0, 1737000001.0]),
            nominal_rate=30000.0,
            metadata={
                "source_file": "recording.dat",
                "source_path": "/data/recordings/recording.dat",
            },
        )
        r = repr(ct)
        assert "recording.dat" in r
        assert "/data/recordings/recording.dat" in r

    def test_repr_source_file_without_path(self):
        ct = ClockTable(
            source=np.array([0.0, 30000.0]),
            reference=np.array([1737000000.0, 1737000001.0]),
            nominal_rate=30000.0,
            metadata={"source_file": "recording.dat"},
        )
        r = repr(ct)
        assert "file: recording.dat" in r
        assert "(" not in r.split("file:")[1].split("\n")[0]

    def test_repr_without_metadata(self):
        ct = ClockTable(
            source=np.array([0.0, 30000.0]),
            reference=np.array([1000.0, 1001.0]),
            nominal_rate=30000.0,
        )
        r = repr(ct)
        assert "2 entries" in r
        assert "file:" not in r

    def test_repr_with_recording_start_stop_from_metadata(self):
        ct = ClockTable(
            source=np.array([0.0, 30000.0]),
            reference=np.array([1737000000.0, 1737000240.0]),
            nominal_rate=30000.0,
            metadata={
                "recording_start": "2025-01-16T02:40:00Z",
                "recording_stop": "2025-01-16T02:44:00Z",
            },
        )
        r = repr(ct)
        assert "2025-01-16T02:40:00Z" in r
        assert "2025-01-16T02:44:00Z" in r


class TestSaveLoad:
    def test_round_trip_basic(self, tmp_path):
        ct = ClockTable(
            source=np.array([0.0, 30000.0, 60000.0]),
            reference=np.array([1000.0, 1001.0, 1002.0]),
            nominal_rate=30000.0,
        )
        path = tmp_path / "test.clocktable.npz"
        ct.save(path)
        loaded = ClockTable.load(path)
        np.testing.assert_array_equal(loaded.source, ct.source)
        np.testing.assert_array_equal(loaded.reference, ct.reference)
        assert loaded.nominal_rate == ct.nominal_rate
        assert loaded.source_units is None

    def test_round_trip_with_source_units(self, tmp_path):
        ct = ClockTable(
            source=np.array([0.0, 1.0]),
            reference=np.array([100.0, 101.0]),
            nominal_rate=1.0,
            source_units="samples",
        )
        path = tmp_path / "test.clocktable.npz"
        ct.save(path)
        loaded = ClockTable.load(path)
        assert loaded.source_units == "samples"

    def test_round_trip_all_fields(self, tmp_path):
        ct = ClockTable(
            source=np.array([0.0, 30000.0]),
            reference=np.array([1737000000.0, 1737000001.0]),
            nominal_rate=30000.0,
            source_units="samples",
        )
        path = tmp_path / "test.clocktable.npz"
        ct.save(path)
        loaded = ClockTable.load(path)
        np.testing.assert_array_equal(loaded.source, ct.source)
        np.testing.assert_array_equal(loaded.reference, ct.reference)
        assert loaded.nominal_rate == ct.nominal_rate
        assert loaded.source_units == "samples"

    def test_interpolation_after_load(self, tmp_path):
        ct = ClockTable(
            source=np.array([0.0, 30000.0, 60000.0]),
            reference=np.array([1000.0, 1001.0, 1002.0]),
            nominal_rate=30000.0,
        )
        path = tmp_path / "test.clocktable.npz"
        ct.save(path)
        loaded = ClockTable.load(path)
        result = loaded.source_to_reference(np.array([15000.0]))
        np.testing.assert_allclose(result, [1000.5])
