"""Tests for SpikeGLX metadata reading and IRIG decoding."""

import numpy as np
import pytest

from neurokairos.decoders.sglx import (
    read_meta,
    get_n_channels,
    get_sample_rate,
    get_irig_channel,
    decode_sglx_irig,
)
from neurokairos.clock_table import ClockTable

# sglx_nidq fixture is provided by conftest.py (session-scoped)


# -- Tests ---------------------------------------------------------------------

class TestReadMeta:
    def test_returns_correct_fields(self, sglx_nidq):
        meta = read_meta(sglx_nidq["bin_path"])
        assert meta["nSavedChans"] == "3"
        assert meta["typeThis"] == "nidq"
        assert meta["niSampRate"] == "30000"
        assert meta["snsMnMaXaDw"] == "2,0,0,1"

    def test_missing_meta_raises(self, tmp_path):
        fake_bin = tmp_path / "no_meta.bin"
        fake_bin.write_bytes(b"\x00" * 100)
        with pytest.raises(FileNotFoundError, match=".meta"):
            read_meta(fake_bin)


class TestMetaHelpers:
    def test_get_n_channels(self, sglx_nidq):
        meta = read_meta(sglx_nidq["bin_path"])
        assert get_n_channels(meta) == 3

    def test_get_sample_rate_nidq(self, sglx_nidq):
        meta = read_meta(sglx_nidq["bin_path"])
        assert get_sample_rate(meta) == 30_000.0

    def test_get_sample_rate_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown SpikeGLX stream type"):
            get_sample_rate({"typeThis": "bogus"})

    def test_get_irig_channel_int(self, sglx_nidq):
        meta = read_meta(sglx_nidq["bin_path"])
        assert get_irig_channel(meta, 2) == 2

    def test_get_irig_channel_sync(self, sglx_nidq):
        meta = read_meta(sglx_nidq["bin_path"])
        assert get_irig_channel(meta, "sync") == 2  # last channel

    def test_get_irig_channel_invalid(self, sglx_nidq):
        meta = read_meta(sglx_nidq["bin_path"])
        with pytest.raises(ValueError, match="irig_channel"):
            get_irig_channel(meta, "bad")


class TestDecodeSglxIrig:
    def test_returns_valid_clock_table(self, sglx_nidq):
        ct = decode_sglx_irig(
            sglx_nidq["bin_path"], sglx_nidq["irig_channel"], save=False
        )
        assert len(ct.source) == len(ct.reference)
        assert len(ct.source) >= 200  # 240s recording, expect ~240 pulses

    def test_timestamps_are_correct(self, sglx_nidq):
        ct = decode_sglx_irig(
            sglx_nidq["bin_path"], sglx_nidq["irig_channel"], save=False
        )
        # Check a few expected pairs
        expected = sglx_nidq["expected_pairs"]
        for src, ref in expected[::30]:  # every 30th pulse
            idx = np.argmin(np.abs(ct.source - src))
            assert abs(ct.source[idx] - src) < 100  # within a few ms
            assert abs(ct.reference[idx] - ref) < 1.5

    def test_auto_saves_clock_table(self, sglx_nidq):
        ct = decode_sglx_irig(
            sglx_nidq["bin_path"], sglx_nidq["irig_channel"], save=True
        )
        ct_path = sglx_nidq["bin_path"].parent / (
            sglx_nidq["bin_path"].name + ".clocktable.npz"
        )
        assert ct_path.exists()

        # Verify the saved file loads correctly
        ct_loaded = ClockTable.load(ct_path)
        np.testing.assert_array_equal(ct.source, ct_loaded.source)
        np.testing.assert_array_equal(ct.reference, ct_loaded.reference)

    def test_sync_channel_shortcut(self, sglx_nidq):
        ct = decode_sglx_irig(
            sglx_nidq["bin_path"], "sync", save=False
        )
        assert len(ct.source) >= 200

    def test_metadata_provenance(self, sglx_nidq):
        ct = decode_sglx_irig(
            sglx_nidq["bin_path"], sglx_nidq["irig_channel"], save=False
        )
        assert isinstance(ct.metadata, dict)
        assert ct.metadata["source_file"] == sglx_nidq["bin_path"].name
        assert ct.metadata["n_channels"] == sglx_nidq["n_channels"]
        assert ct.metadata["irig_channel"] == sglx_nidq["irig_channel"]
        assert ct.metadata["stream_type"] == "nidq"

    def test_metadata_decoding_stats(self, sglx_nidq):
        ct = decode_sglx_irig(
            sglx_nidq["bin_path"], sglx_nidq["irig_channel"], save=False
        )
        assert ct.metadata["n_raw_pulses"] > 0
        assert ct.metadata["n_frames_decoded"] >= 1
