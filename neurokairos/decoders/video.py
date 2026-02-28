"""Video IRIG decoding — extract LED brightness and build ClockTable."""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from ..clock_table import ClockTable
from .irig import build_clock_table
from .ttl import auto_threshold, detect_edges, measure_pulse_widths

logger = logging.getLogger(__name__)


def extract_led_signal(
    video_path: Union[str, Path],
    roi: Tuple[int, int, int, int],
) -> np.ndarray:
    """Extract mean LED brightness per frame from a video file.

    Parameters
    ----------
    video_path : str or Path
        Path to the video file (AVI, MP4, etc.).
    roi : tuple of int
        ``(row_start, row_end, col_start, col_end)`` defining the LED region.

    Returns
    -------
    ndarray
        1-D float64 array of mean brightness values, one per frame.

    Raises
    ------
    ImportError
        If OpenCV (cv2) is not installed.
    RuntimeError
        If the video file cannot be opened or contains no frames.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "OpenCV (cv2) is required for video decoding. "
            "Install it with: pip install opencv-python"
        )

    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    r0, r1, c0, c1 = roi
    values = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert to grayscale if color
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            patch = frame[r0:r1, c0:c1]
            values.append(float(patch.mean()))
    finally:
        cap.release()

    if not values:
        raise RuntimeError(f"No frames read from {video_path}")

    return np.array(values, dtype=np.float64)


def decode_video_irig(
    video_path: Union[str, Path],
    roi: Tuple[int, int, int, int],
    fps: Optional[float] = None,
    save: bool = True,
) -> ClockTable:
    """Decode IRIG timecodes from a video file with a visible IRIG LED.

    Parameters
    ----------
    video_path : str or Path
        Path to the video file.
    roi : tuple of int
        ``(row_start, row_end, col_start, col_end)`` defining the LED region.
    fps : float, optional
        Expected frame rate. If provided, used as a sanity check against the
        auto-estimated rate (warns if they differ by >5%).
    save : bool
        If True (default), save the ClockTable to
        ``<video_path>.clocktable.npz`` alongside the video file.

    Returns
    -------
    ClockTable
        Sparse frame-index-to-UTC-time mapping with one entry per IRIG
        pulse (~1 Hz).
    """
    video_path = Path(video_path)

    signal = extract_led_signal(video_path, roi)
    threshold = auto_threshold(signal)
    rising, falling = detect_edges(signal, threshold)
    onsets, widths = measure_pulse_widths(rising, falling)

    ct = build_clock_table(onsets, widths)

    # Sanity-check against provided fps
    if fps is not None:
        estimated = ct.nominal_rate
        if abs(estimated - fps) / fps > 0.05:
            logger.warning(
                "Estimated frame rate (%.1f) differs from provided fps (%.1f) "
                "by more than 5%%",
                estimated, fps,
            )

    # Override source_units to "frames", carrying over metadata
    metadata = dict(ct.metadata) if ct.metadata else {}
    metadata["source_file"] = video_path.name
    metadata["source_path"] = str(video_path.resolve())
    metadata["roi"] = list(roi)

    ct = ClockTable(
        source=ct.source,
        reference=ct.reference,
        nominal_rate=ct.nominal_rate,
        source_units="frames",
        metadata=metadata,
        sync_stratum=ct.sync_stratum,
        sync_dispersion_upperbound_ms=ct.sync_dispersion_upperbound_ms,
    )

    if save:
        ct_path = video_path.parent / (video_path.name + ".clocktable.npz")
        ct.save(ct_path)
        logger.info("Saved clock table to %s", ct_path)

        # Generate visual sync report alongside the NPZ
        png_path = video_path.parent / (
            video_path.name + ".sync_report.png"
        )
        from .report import _try_generate_report
        _try_generate_report(
            ct, png_path,
            raw_signal=signal,
            threshold=threshold,
            pulse_widths=widths,
        )

    return ct


def dropped_frame_report(
    clock_table: ClockTable,
    expected_fps: float,
) -> dict:
    """Analyse dropped frames by comparing actual vs expected frame intervals.

    Parameters
    ----------
    clock_table : ClockTable
        A video ClockTable (source in frame indices, reference in UTC).
    expected_fps : float
        The camera's nominal frame rate (e.g. 30.0).

    Returns
    -------
    dict
        ``total_expected`` : int — expected frames over the time span.
        ``total_actual`` : int — actual frames (from source range).
        ``total_dropped`` : int — difference.
        ``drops_per_second`` : ndarray — drops in each 1-second window.
        ``seconds_with_drops`` : ndarray — indices of seconds that had drops.
    """
    src = clock_table.source
    ref = clock_table.reference

    total_time = ref[-1] - ref[0]
    n_seconds = int(np.floor(total_time))

    total_expected = int(np.round(total_time * expected_fps))
    total_actual = int(src[-1] - src[0])
    total_dropped = total_expected - total_actual

    # Per-second analysis: for each 1-second window of reference time,
    # count how many source frames span that window vs expected
    drops_per_second = np.zeros(n_seconds, dtype=np.int64)
    for i in range(n_seconds):
        t0 = ref[0] + i
        t1 = ref[0] + i + 1
        # Find source frames in this reference-time window
        s0 = clock_table.reference_to_source(t0)
        s1 = clock_table.reference_to_source(t1)
        actual_in_window = int(np.round(s1 - s0))
        expected_in_window = int(np.round(expected_fps))
        drop = expected_in_window - actual_in_window
        drops_per_second[i] = max(0, drop)

    seconds_with_drops = np.where(drops_per_second > 0)[0]

    return {
        "total_expected": total_expected,
        "total_actual": total_actual,
        "total_dropped": total_dropped,
        "drops_per_second": drops_per_second,
        "seconds_with_drops": seconds_with_drops,
    }
