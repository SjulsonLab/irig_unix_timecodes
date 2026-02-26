"""Tests for video IRIG decoding."""

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from neurokairos.clock_table import ClockTable
from neurokairos.video import (
    extract_led_signal,
    decode_video_irig,
    dropped_frame_report,
)

# Import shared helpers from conftest
sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import generate_irig_h_frame, _bit_to_pulse_frac


# -- Fixture: synthetic AVI (clean) -------------------------------------------

@pytest.fixture(scope="session")
def generate_test_avi(tmp_path_factory):
    """Generate a synthetic AVI with an IRIG LED in a 3x3 ROI.

    240 seconds, 30 fps = 7200 frames.  64x64 grayscale frames.
    Upper-left 3x3 ROI: bright (200) during IRIG HIGH, dark (20) during LOW.
    Rest of frame: uniform gray (128) + slight noise.
    Same IRIG timing as DAT test data: starts at 14:30:37 UTC on 2026-01-15.
    """
    try:
        import cv2
    except ImportError:
        pytest.skip("OpenCV (cv2) not installed")

    fps = 30
    duration_s = 240
    n_frames = fps * duration_s
    frame_h, frame_w = 64, 64
    roi = (0, 3, 0, 3)
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

    # Build per-frame brightness: for each frame, determine if the IRIG
    # signal is HIGH or LOW based on which second we're in and the pulse width
    led_bright = 200
    led_dark = 20
    bg_gray = 128

    rng = np.random.default_rng(42)

    tmp_dir = tmp_path_factory.mktemp("video")
    avi_path = str(tmp_dir / "test_irig.avi")

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(avi_path, fourcc, fps, (frame_w, frame_h), False)
    if not writer.isOpened():
        pytest.skip("cv2.VideoWriter could not open with MJPG codec")

    expected_pairs = []  # (frame_index, unix_timestamp) for each pulse onset

    for frame_idx in range(n_frames):
        # Which second of the recording and fractional position within it
        second_idx = frame_idx // fps
        frac_in_second = (frame_idx % fps) / fps

        # Determine pulse width for this second
        if second_idx < len(recording_bits):
            pulse_frac = _bit_to_pulse_frac(recording_bits[second_idx])
        else:
            pulse_frac = 0.0

        is_high = frac_in_second < pulse_frac

        # Build frame
        bg_noise = rng.integers(125, 132, size=(frame_h, frame_w), dtype=np.uint8)
        frame = bg_noise.copy()
        r0, r1, c0, c1 = roi
        if is_high:
            frame[r0:r1, c0:c1] = led_bright
        else:
            frame[r0:r1, c0:c1] = led_dark

        writer.write(frame)

        # Record onset frames (first frame of each second = pulse onset)
        if frame_idx % fps == 0 and second_idx < len(recording_bits):
            unix_ts = start_dt.timestamp() + second_idx
            expected_pairs.append((float(frame_idx), unix_ts))

    writer.release()

    # Pulse 0 is not detected: the video starts with the LED HIGH
    # (mid-pulse), so there is no rising edge at frame 0.
    expected_pairs = expected_pairs[1:]

    return {
        "path": avi_path,
        "roi": roi,
        "fps": fps,
        "start_time": start_dt,
        "start_timestamp": start_dt.timestamp(),
        "duration_s": duration_s,
        "n_frames": n_frames,
        "expected_pairs": expected_pairs,
    }


# -- Fixture: synthetic AVI with dropped frames --------------------------------

@pytest.fixture(scope="session")
def generate_test_avi_dropped_frames(tmp_path_factory):
    """Generate a synthetic AVI with ~3% of frames dropped.

    Drops are concentrated in specific 1-second windows to create
    detectable bursty loss. Frames are simply omitted (the video has
    fewer total frames), matching real camera behavior.
    """
    try:
        import cv2
    except ImportError:
        pytest.skip("OpenCV (cv2) not installed")

    fps = 30
    duration_s = 240
    n_frames_full = fps * duration_s
    frame_h, frame_w = 64, 64
    roi = (0, 3, 0, 3)
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

    start_bit = start_dt.second
    recording_bits = all_bits[start_bit : start_bit + duration_s]

    led_bright = 200
    led_dark = 20

    rng = np.random.default_rng(123)

    # Choose seconds in which to drop frames — spread across the recording,
    # avoiding the first and last few seconds.  Drop 2-4 frames per affected
    # second (out of 30), ensuring we never drop the first frame of a second
    # (the pulse onset frame) to keep IRIG pulses visible.
    drop_seconds = [10, 30, 55, 80, 110, 140, 170, 200, 225]
    drops_per_second_expected = np.zeros(duration_s, dtype=np.int64)
    dropped_frames = set()
    for sec in drop_seconds:
        n_drop = rng.integers(2, 5)  # 2-4 frames
        # Pick frames within this second, excluding frame 0 (onset) and
        # frames in the pulse-HIGH region (first ~pulse_frac * fps frames)
        if sec < len(recording_bits):
            pulse_frac = _bit_to_pulse_frac(recording_bits[sec])
        else:
            pulse_frac = 0.2
        safe_start = int(pulse_frac * fps) + 1  # after pulse ends
        candidates = list(range(sec * fps + safe_start, (sec + 1) * fps))
        if len(candidates) < n_drop:
            n_drop = len(candidates)
        chosen = rng.choice(candidates, size=n_drop, replace=False)
        for f in chosen:
            dropped_frames.add(int(f))
        drops_per_second_expected[sec] = n_drop

    tmp_dir = tmp_path_factory.mktemp("video_dropped")
    avi_path = str(tmp_dir / "test_irig_dropped.avi")

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(avi_path, fourcc, fps, (frame_w, frame_h), False)
    if not writer.isOpened():
        pytest.skip("cv2.VideoWriter could not open with MJPG codec")

    n_written = 0
    for frame_idx in range(n_frames_full):
        if frame_idx in dropped_frames:
            continue

        second_idx = frame_idx // fps
        frac_in_second = (frame_idx % fps) / fps

        if second_idx < len(recording_bits):
            pulse_frac = _bit_to_pulse_frac(recording_bits[second_idx])
        else:
            pulse_frac = 0.0

        is_high = frac_in_second < pulse_frac

        bg_noise = rng.integers(125, 132, size=(frame_h, frame_w), dtype=np.uint8)
        frame = bg_noise.copy()
        r0, r1, c0, c1 = roi
        if is_high:
            frame[r0:r1, c0:c1] = led_bright
        else:
            frame[r0:r1, c0:c1] = led_dark

        writer.write(frame)
        n_written += 1

    writer.release()

    total_dropped = len(dropped_frames)

    return {
        "path": avi_path,
        "roi": roi,
        "fps": fps,
        "start_time": start_dt,
        "start_timestamp": start_dt.timestamp(),
        "duration_s": duration_s,
        "n_frames_full": n_frames_full,
        "n_frames_written": n_written,
        "total_dropped": total_dropped,
        "drops_per_second": drops_per_second_expected,
        "drop_seconds": drop_seconds,
    }


# -- Tests ---------------------------------------------------------------------

class TestExtractLedSignal:
    def test_signal_shape(self, generate_test_avi):
        meta = generate_test_avi
        signal = extract_led_signal(meta["path"], meta["roi"])
        assert signal.ndim == 1
        assert len(signal) == meta["n_frames"]

    def test_signal_dtype(self, generate_test_avi):
        meta = generate_test_avi
        signal = extract_led_signal(meta["path"], meta["roi"])
        assert signal.dtype == np.float64

    def test_bimodal_distribution(self, generate_test_avi):
        meta = generate_test_avi
        signal = extract_led_signal(meta["path"], meta["roi"])
        # The signal should have two modes: ~20 (dark) and ~200 (bright)
        # Check that values cluster around these two levels
        # (MJPEG compression may shift values slightly)
        low_mask = signal < 110
        high_mask = signal >= 110
        assert np.sum(low_mask) > 100, "Expected many LOW frames"
        assert np.sum(high_mask) > 100, "Expected many HIGH frames"
        assert np.mean(signal[low_mask]) < 80
        assert np.mean(signal[high_mask]) > 150

    def test_invalid_path_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="Cannot open"):
            extract_led_signal(tmp_path / "nonexistent.avi", (0, 3, 0, 3))


class TestDecodeVideoIrig:
    def test_returns_clock_table(self, generate_test_avi):
        meta = generate_test_avi
        ct = decode_video_irig(meta["path"], meta["roi"], save=False)
        assert isinstance(ct, ClockTable)
        assert ct.source_units == "frames"

    def test_pulse_count(self, generate_test_avi):
        meta = generate_test_avi
        ct = decode_video_irig(meta["path"], meta["roi"], save=False)
        # 240-second recording -> ~240 pulses
        assert len(ct.source) >= 200

    def test_timestamps_are_correct(self, generate_test_avi):
        meta = generate_test_avi
        ct = decode_video_irig(meta["path"], meta["roi"], save=False)
        expected = meta["expected_pairs"]
        # Check every 30th expected pair
        for src, ref in expected[::30]:
            idx = np.argmin(np.abs(ct.source - src))
            # Source (frame index) should be within a couple frames
            assert abs(ct.source[idx] - src) < 3, (
                f"Source mismatch: expected ~{src}, got {ct.source[idx]}"
            )
            # Reference (unix timestamp) should be within 1.5 seconds
            assert abs(ct.reference[idx] - ref) < 1.5, (
                f"Reference mismatch: expected ~{ref}, got {ct.reference[idx]}"
            )

    def test_fps_sanity_check(self, generate_test_avi):
        meta = generate_test_avi
        # Should not warn when fps matches
        ct = decode_video_irig(
            meta["path"], meta["roi"], fps=meta["fps"], save=False
        )
        assert ct is not None

    def test_monotonically_increasing(self, generate_test_avi):
        meta = generate_test_avi
        ct = decode_video_irig(meta["path"], meta["roi"], save=False)
        assert np.all(np.diff(ct.source) > 0)
        assert np.all(np.diff(ct.reference) > 0)

    def test_metadata_provenance(self, generate_test_avi):
        meta = generate_test_avi
        ct = decode_video_irig(meta["path"], meta["roi"], save=False)
        assert isinstance(ct.metadata, dict)
        assert ct.metadata["source_file"] == Path(meta["path"]).name
        assert ct.metadata["roi"] == list(meta["roi"])

    def test_metadata_decoding_stats(self, generate_test_avi):
        meta = generate_test_avi
        ct = decode_video_irig(meta["path"], meta["roi"], save=False)
        assert ct.metadata["n_raw_pulses"] > 0
        assert ct.metadata["n_frames_decoded"] >= 1

    def test_metadata_survives_save_load(self, generate_test_avi):
        meta = generate_test_avi
        ct = decode_video_irig(meta["path"], meta["roi"], save=True)
        ct_path = Path(meta["path"]).parent / (
            Path(meta["path"]).name + ".clocktable.npz"
        )
        ct_loaded = ClockTable.load(ct_path)
        assert ct_loaded.metadata == ct.metadata


class TestDecodeVideoIrigDroppedFrames:
    def test_decodes_despite_drops(self, generate_test_avi_dropped_frames):
        meta = generate_test_avi_dropped_frames
        ct = decode_video_irig(meta["path"], meta["roi"], save=False)
        assert isinstance(ct, ClockTable)
        # Should still decode most pulses
        assert len(ct.source) >= 200

    def test_timestamps_still_correct(self, generate_test_avi_dropped_frames):
        meta = generate_test_avi_dropped_frames
        ct = decode_video_irig(meta["path"], meta["roi"], save=False)
        # Reference timestamps should still be valid UTC times
        # The span should cover ~240 seconds
        time_span = ct.reference[-1] - ct.reference[0]
        assert time_span > 200
        assert time_span < 260

    def test_dropped_frame_report_detects_drops(
        self, generate_test_avi_dropped_frames
    ):
        meta = generate_test_avi_dropped_frames
        ct = decode_video_irig(meta["path"], meta["roi"], save=False)
        report = dropped_frame_report(ct, meta["fps"])

        assert report["total_dropped"] > 0
        assert len(report["seconds_with_drops"]) > 0
        assert report["total_actual"] < report["total_expected"]

    def test_dropped_frame_report_correct_seconds(
        self, generate_test_avi_dropped_frames
    ):
        meta = generate_test_avi_dropped_frames
        ct = decode_video_irig(meta["path"], meta["roi"], save=False)
        report = dropped_frame_report(ct, meta["fps"])

        # The seconds that had drops should include the ones we injected.
        # Report indices are relative to ref[0], which may be offset from
        # second 0 of the recording (e.g. if the first pulse is lost because
        # the signal starts HIGH).  Allow ±1 second tolerance.
        reported_drop_secs = set(report["seconds_with_drops"].tolist())
        expected_drop_secs = set(meta["drop_seconds"])
        overlap = set()
        for exp in expected_drop_secs:
            for off in (-1, 0, 1):
                if (exp + off) in reported_drop_secs:
                    overlap.add(exp)
                    break
        assert len(overlap) >= len(expected_drop_secs) // 2, (
            f"Expected drops in {expected_drop_secs}, "
            f"reported drops in {reported_drop_secs}"
        )


class TestSaveLoad:
    def test_auto_saves_clock_table(self, generate_test_avi):
        meta = generate_test_avi
        ct = decode_video_irig(meta["path"], meta["roi"], save=True)
        ct_path = Path(meta["path"]).parent / (
            Path(meta["path"]).name + ".clocktable.npz"
        )
        assert ct_path.exists()

    def test_saved_clock_table_loads_correctly(self, generate_test_avi):
        meta = generate_test_avi
        ct = decode_video_irig(meta["path"], meta["roi"], save=True)
        ct_path = Path(meta["path"]).parent / (
            Path(meta["path"]).name + ".clocktable.npz"
        )

        ct_loaded = ClockTable.load(ct_path)
        np.testing.assert_array_equal(ct.source, ct_loaded.source)
        np.testing.assert_array_equal(ct.reference, ct_loaded.reference)
        assert ct_loaded.source_units == "frames"

    def test_save_false_does_not_create_file(self, generate_test_avi, tmp_path):
        """Verify save=False doesn't write a file (use a copy to avoid side effects)."""
        import shutil
        meta = generate_test_avi
        # Copy the AVI to tmp_path so we can check cleanly
        src = Path(meta["path"])
        dst = tmp_path / src.name
        shutil.copy2(src, dst)

        decode_video_irig(str(dst), meta["roi"], save=False)
        ct_path = tmp_path / (dst.name + ".clocktable.npz")
        assert not ct_path.exists()
