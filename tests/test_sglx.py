"""Tests for SpikeGLX metadata reading and IRIG decoding."""

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from neurokairos.sglx import (
    read_meta,
    get_n_channels,
    get_sample_rate,
    get_irig_channel,
    decode_sglx_irig,
)
from neurokairos.clock_table import ClockTable

# conftest is auto-loaded by pytest but not directly importable;
# add the tests directory so we can import the shared helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import generate_irig_h_frame, _bit_to_pulse_frac, _build_irig_signal


# -- Fixture: synthetic SpikeGLX nidq recording --------------------------------

@pytest.fixture(scope="session")
def sglx_nidq(tmp_path_factory):
    """Generate a synthetic SpikeGLX nidq .bin + .meta file pair.

    3 channels (2 MN analog + 1 DW digital word), 30 kHz, 240 seconds.
    IRIG signal on the last channel (index 2, the DW channel).
    """
    sample_rate = 30_000
    n_channels = 3
    duration_s = 240
    start_dt = datetime(2026, 1, 15, 14, 30, 37, tzinfo=timezone.utc)

    # Build IRIG bit sequence
    minute_start = start_dt.replace(second=0, microsecond=0)
    all_bits = []
    frame_time = minute_start
    end_dt = start_dt + timedelta(seconds=duration_s)
    while frame_time < end_dt:
        frame = generate_irig_h_frame(frame_time)
        all_bits.extend(frame)
        frame_time += timedelta(seconds=60)

    start_bit = start_dt.second  # 37
    recording_bits = all_bits[start_bit : start_bit + duration_s]

    n_samples = sample_rate * duration_s
    rng = np.random.default_rng(42)

    # Build IRIG signal
    irig = _build_irig_signal(recording_bits, sample_rate, n_samples, rng)

    # Other channels: sinusoids
    t = np.arange(n_samples, dtype=np.float64) / sample_rate
    ch0 = 10_000.0 * np.sin(2 * np.pi * 10 * t)
    ch1 = 10_000.0 * np.sin(2 * np.pi * 50 * t)

    # Interleave and write .bin
    data = np.empty((n_samples, n_channels), dtype=np.int16)
    data[:, 0] = np.clip(ch0, -32768, 32767).astype(np.int16)
    data[:, 1] = np.clip(ch1, -32768, 32767).astype(np.int16)
    data[:, 2] = np.clip(irig, -32768, 32767).astype(np.int16)

    tmp_dir = tmp_path_factory.mktemp("sglx")
    bin_path = tmp_dir / "test_nidq.bin"
    data.tofile(bin_path)

    file_size_bytes = data.nbytes

    # Write minimal .meta file
    meta_path = bin_path.with_suffix(".meta")
    meta_lines = [
        f"nSavedChans={n_channels}",
        "typeThis=nidq",
        f"niSampRate={sample_rate}",
        f"fileSizeBytes={file_size_bytes}",
        "snsMnMaXaDw=2,0,0,1",
    ]
    meta_path.write_text("\n".join(meta_lines) + "\n")

    start_ts = start_dt.timestamp()
    expected_pairs = [
        (float(i * sample_rate), start_ts + float(i))
        for i in range(duration_s)
    ]

    return {
        "bin_path": bin_path,
        "meta_path": meta_path,
        "start_time": start_dt,
        "start_timestamp": start_ts,
        "sample_rate": sample_rate,
        "n_channels": n_channels,
        "irig_channel": 2,
        "duration_s": duration_s,
        "expected_pairs": expected_pairs,
    }


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
