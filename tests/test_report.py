"""Tests for sync report generation (neurokairos/decoders/report.py).

Tests cover:
- Report generation triggered by decode entry points (dat, sglx)
- Valid PNG output
- Graceful degradation when raw_signal is None (intervals path)
- Graceful skip when matplotlib is not installed
"""

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from neurokairos import ClockTable, decode_dat_irig
from neurokairos.decoders.sglx import decode_sglx_irig
from neurokairos.decoders.report import generate_sync_report

# PNG magic bytes: first 8 bytes of any valid PNG file
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# -- Helpers -------------------------------------------------------------------

def _make_clock_table(n_pulses=120, sps=30_000.0):
    """Build a minimal ClockTable for testing report generation."""
    source = np.arange(n_pulses, dtype=np.float64) * sps
    # Reference: 1 second apart starting at an arbitrary UTC epoch
    reference = np.arange(n_pulses, dtype=np.float64) + 1737000000.0
    return ClockTable(
        source=source,
        reference=reference,
        nominal_rate=sps,
        source_units="samples",
        metadata={"source_file": "test.dat", "source_path": "/tmp/test.dat"},
    )


def _make_raw_signal(n_pulses=120, sps=30_000):
    """Build a synthetic raw signal matching the ClockTable."""
    n_samples = n_pulses * sps
    signal = np.zeros(n_samples, dtype=np.int16)
    rng = np.random.default_rng(42)
    # Simple square-wave pulses (0.2 s width = "binary 0")
    for i in range(n_pulses):
        s0 = i * sps
        pw = int(0.2 * sps)
        signal[s0 : s0 + pw] = 10_000
    # Add noise
    signal = signal.astype(np.float64) + rng.normal(0, 200, n_samples)
    return signal


# -- Integration: decode_dat_irig produces a .sync_report.png ------------------

def test_report_generated_on_decode_dat(generate_test_dat):
    """decode_dat_irig should produce a .sync_report.png alongside the NPZ."""
    meta = generate_test_dat
    ct = decode_dat_irig(
        meta["path"], meta["n_channels"], meta["irig_channel"], save=True
    )

    png_path = Path(meta["path"]).parent / (
        Path(meta["path"]).name + ".sync_report.png"
    )
    assert png_path.exists(), f"Expected sync report at {png_path}"
    assert png_path.stat().st_size > 1000, "PNG file suspiciously small"


def test_report_is_valid_png(generate_test_dat):
    """The sync report PNG should have valid PNG magic bytes."""
    meta = generate_test_dat
    decode_dat_irig(
        meta["path"], meta["n_channels"], meta["irig_channel"], save=True
    )

    png_path = Path(meta["path"]).parent / (
        Path(meta["path"]).name + ".sync_report.png"
    )
    with open(png_path, "rb") as f:
        header = f.read(8)
    assert header == PNG_MAGIC, f"Invalid PNG header: {header!r}"


# -- Integration: decode_sglx_irig produces a .sync_report.png ----------------

def test_report_generated_on_decode_sglx(sglx_nidq):
    """decode_sglx_irig should produce a .sync_report.png alongside the NPZ."""
    ct = decode_sglx_irig(
        sglx_nidq["bin_path"], sglx_nidq["irig_channel"], save=True
    )

    png_path = sglx_nidq["bin_path"].parent / (
        sglx_nidq["bin_path"].name + ".sync_report.png"
    )
    assert png_path.exists(), f"Expected sync report at {png_path}"


# -- Unit: generate_sync_report with raw signal data --------------------------

def test_report_with_raw_signal(tmp_path):
    """generate_sync_report should produce a PNG when given raw signal data."""
    ct = _make_clock_table()
    signal = _make_raw_signal()
    threshold = 5000.0
    # Pulse widths: n_pulses - 1 entries (from measure_pulse_widths)
    pulse_widths = np.full(119, 0.2 * 30_000)
    out = tmp_path / "report.png"

    generate_sync_report(
        ct, out,
        raw_signal=signal,
        threshold=threshold,
        pulse_widths=pulse_widths,
        sps=30_000.0,
    )

    assert out.exists()
    with open(out, "rb") as f:
        assert f.read(8) == PNG_MAGIC


# -- Unit: generate_sync_report without raw signal (intervals path) ------------

def test_report_without_raw_signal(tmp_path):
    """generate_sync_report should still produce a PNG with only ClockTable.

    When raw_signal is None (intervals path), panels 1-3 are skipped but
    panels 4 (drift) and 5 (sync status) should still appear.
    """
    ct = _make_clock_table()
    out = tmp_path / "report_no_signal.png"

    generate_sync_report(ct, out, raw_signal=None)

    assert out.exists()
    with open(out, "rb") as f:
        assert f.read(8) == PNG_MAGIC


# -- Graceful degradation: no matplotlib available -----------------------------

def test_report_skip_without_matplotlib(generate_test_dat):
    """When matplotlib is not installed, decoding should succeed with no PNG."""
    meta = generate_test_dat

    # Remove any pre-existing report PNG
    png_path = Path(meta["path"]).parent / (
        Path(meta["path"]).name + ".sync_report.png"
    )
    if png_path.exists():
        png_path.unlink()

    # Mock matplotlib as unavailable inside report module
    with patch("neurokairos.decoders.report.HAS_MATPLOTLIB", False):
        ct = decode_dat_irig(
            meta["path"], meta["n_channels"], meta["irig_channel"], save=True
        )

    # Decode should succeed
    assert isinstance(ct, ClockTable)
    # But no PNG should be created
    assert not png_path.exists(), "PNG should not be created without matplotlib"
