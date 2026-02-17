import os

import numpy as np
import pytest

from neurokairos.clock_table import ClockTable


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
