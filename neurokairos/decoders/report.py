"""Sync report PNG generation for IRIG-H decoding results.

Generates a multi-panel diagnostic figure alongside the ClockTable NPZ
so users can visually verify that synchronization was correct.

matplotlib is imported lazily — if not installed, report generation is
skipped with a warning. Core decoding never depends on this module.
"""

import datetime as _dt
import logging

import numpy as np

logger = logging.getLogger(__name__)

# Guard matplotlib import — remains an optional dependency
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for PNG output
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# -- Panel drawing helpers -----------------------------------------------------

def _draw_text_header(fig, ct, raw_signal):
    """Draw summary statistics as text at the top of the figure."""
    meta = ct.metadata or {}
    lines = []

    # Source file
    if meta.get("source_file"):
        lines.append(f"File: {meta['source_file']}")

    # Recording start/stop
    start_str = meta.get("recording_start", "")
    stop_str = meta.get("recording_stop", "")
    if start_str and stop_str:
        lines.append(f"Recording: {start_str} → {stop_str}")

    # Duration and measured rate
    duration_s = ct.reference[-1] - ct.reference[0]
    duration_str = str(_dt.timedelta(seconds=int(duration_s)))
    units = ct.source_units
    rate = ct.nominal_rate
    if units == "samples":
        rate_str = f"{rate:.2f} Hz"
    elif units == "frames":
        rate_str = f"{rate:.2f} fps"
    else:
        rate_str = f"{rate:.6f} s/s"
    lines.append(f"Duration: {duration_str}  |  Measured rate: {rate_str}")

    # Pulse/frame stats
    lines.append(
        f"Pulses: {len(ct.source)} detected"
        f"  |  Frames decoded: {meta.get('n_frames_decoded', '?')}"
    )

    # Extra/missing/concat
    extras = meta.get("n_extra_removed", 0)
    missing = meta.get("n_missing_gaps", 0)
    concat = meta.get("n_concat_boundaries", 0)
    if extras or missing or concat:
        warn_parts = []
        if extras:
            warn_parts.append(f"Extra removed: {extras}")
        if missing:
            warn_parts.append(f"Missing gaps: {missing}")
        if concat:
            warn_parts.append(f"Concat boundaries: {concat}")
        lines.append("  |  ".join(warn_parts))

    # Sync quality
    stratum = meta.get("stratum")
    precision = meta.get("UTC_sync_precision")
    if stratum is not None:
        lines.append(
            f"Sync: worst stratum={stratum}, "
            f"root dispersion {precision or 'unknown'}"
        )

    text = "\n".join(lines)
    fig.text(
        0.05, 0.97, text,
        fontsize=8, fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                  edgecolor="gray", alpha=0.9),
    )


def _draw_zoomed_signal(ax, raw_signal, threshold, pulse_widths, sps):
    """Panel 1: zoomed ~10 s snippet of the raw signal with edges."""
    from .irig import PULSE_FRAC_ZERO, PULSE_FRAC_ONE, PULSE_FRAC_MARKER
    from .ttl import detect_edges

    # Show first ~10 seconds
    n_show = min(int(10 * sps), len(raw_signal))
    snippet = np.asarray(raw_signal[:n_show], dtype=np.float64)
    t = np.arange(n_show) / sps

    ax.plot(t, snippet, linewidth=0.3, color="steelblue", alpha=0.8)
    ax.axhline(threshold, color="red", linewidth=0.8, linestyle="--",
               label="threshold")

    # Detect edges in the snippet to mark pulse onsets
    rising, falling = detect_edges(
        np.asarray(raw_signal[:n_show]), threshold
    )

    # Classify and label pulses
    for i, r in enumerate(rising):
        ax.axvline(r / sps, color="green", linewidth=0.5, alpha=0.6)
        # Find matching falling edge for classification
        falls_after = falling[falling > r]
        if len(falls_after) > 0:
            width_frac = (falls_after[0] - r) / sps
            if width_frac < 0.35:
                label = "0"
            elif width_frac < 0.65:
                label = "1"
            else:
                label = "P"
            # Place label above signal
            ax.text(
                r / sps, threshold * 1.15, label,
                fontsize=6, ha="center", color="darkgreen",
            )

    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Signal", fontsize=8)
    ax.set_title("Signal snippet (first ~10 s)", fontsize=9)
    ax.tick_params(labelsize=7)


def _draw_pulse_histogram(ax, pulse_widths, sps):
    """Panel 2: histogram of pulse widths with classification boundaries."""
    from .irig import (
        PULSE_FRAC_ZERO, PULSE_FRAC_ONE, PULSE_FRAC_MARKER,
        BOUNDARY_ZERO_ONE, BOUNDARY_ONE_MARKER,
    )

    # Convert widths to fractions of 1 second
    fracs = pulse_widths / sps

    # Color each pulse by its classification
    zeros = fracs[fracs < BOUNDARY_ZERO_ONE]
    ones = fracs[(fracs >= BOUNDARY_ZERO_ONE) & (fracs < BOUNDARY_ONE_MARKER)]
    markers = fracs[fracs >= BOUNDARY_ONE_MARKER]

    bins = np.linspace(0.0, 1.0, 80)
    ax.hist(zeros, bins=bins, color="skyblue", alpha=0.8, label="0 (zero)")
    ax.hist(ones, bins=bins, color="orange", alpha=0.8, label="1 (one)")
    ax.hist(markers, bins=bins, color="salmon", alpha=0.8, label="P (marker)")

    # Classification boundaries
    ax.axvline(BOUNDARY_ZERO_ONE, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(BOUNDARY_ONE_MARKER, color="gray", linestyle="--", linewidth=0.8)

    # Nominal positions
    for nom, lbl in [(PULSE_FRAC_ZERO, "0.2"),
                     (PULSE_FRAC_ONE, "0.5"),
                     (PULSE_FRAC_MARKER, "0.8")]:
        ax.axvline(nom, color="black", linestyle=":", linewidth=0.5, alpha=0.5)

    ax.set_xlabel("Pulse width (fraction of 1 s)", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.set_title("Pulse width distribution", fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)


def _draw_full_overview(ax, raw_signal, ct):
    """Panel 3: downsampled signal envelope across the entire recording."""
    n = len(raw_signal)
    sps = ct.nominal_rate

    # Downsample to ~2000 points for display
    chunk_size = max(1, n // 2000)
    n_chunks = n // chunk_size
    trimmed = np.asarray(
        raw_signal[: n_chunks * chunk_size], dtype=np.float64
    )
    reshaped = trimmed.reshape(n_chunks, chunk_size)
    envelope_max = reshaped.max(axis=1)
    envelope_min = reshaped.min(axis=1)

    # Build UTC x-axis from chunk midpoints (in sample space → reference)
    chunk_mids = (np.arange(n_chunks) + 0.5) * chunk_size
    ref_times = ct.source_to_reference(chunk_mids)

    # Convert unix timestamps to datetime for x-axis labeling
    dates = [_dt.datetime.fromtimestamp(t, tz=_dt.timezone.utc)
             for t in ref_times]

    ax.fill_between(dates, envelope_min, envelope_max,
                    color="steelblue", alpha=0.5)
    ax.set_ylabel("Signal", fontsize=8)
    ax.set_title("Full recording overview", fontsize=9)
    ax.tick_params(labelsize=7)

    # Format x-axis as UTC times
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.set_xlabel("UTC time", fontsize=8)
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")


def _draw_clock_drift(ax, ct):
    """Panel 4: residual from nominal clock rate (drift in ms)."""
    src = ct.source
    ref = ct.reference

    # Residual: actual reference minus expected from nominal rate
    expected = ref[0] + (src - src[0]) / ct.nominal_rate
    residual_ms = (ref - expected) * 1000.0

    # Build UTC x-axis
    dates = [_dt.datetime.fromtimestamp(t, tz=_dt.timezone.utc)
             for t in ref]

    ax.plot(dates, residual_ms, linewidth=0.5, color="navy")
    # Format rate with appropriate units for the source domain
    units = ct.source_units
    rate = ct.nominal_rate
    if units == "samples":
        rate_label = f"{rate:.2f} Hz"
    elif units == "frames":
        rate_label = f"{rate:.2f} fps"
    else:
        rate_label = f"{rate:.6f} s/s"

    ax.set_ylabel("Drift (ms)", fontsize=8)
    ax.set_title(f"Clock jitter (measured rate: {rate_label})", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.set_xlabel("UTC time", fontsize=8)
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")


def _draw_sync_status(ax, ct):
    """Panel 5: per-pulse sync status timeline (stratum + dispersion)."""
    ref = ct.reference
    dates = [_dt.datetime.fromtimestamp(t, tz=_dt.timezone.utc)
             for t in ref]

    has_stratum = ct.sync_stratum is not None
    has_disp = ct.sync_dispersion_upperbound_ms is not None

    if not has_stratum and not has_disp:
        ax.text(0.5, 0.5, "No sync status data available",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="gray")
        ax.set_title("Sync status", fontsize=9)
        return

    # Stratum on left y-axis
    if has_stratum:
        ax.step(dates, ct.sync_stratum, where="post",
                color="darkorange", linewidth=0.8, label="Stratum")
        ax.set_ylabel("Stratum", fontsize=8, color="darkorange")
        ax.set_ylim(0.5, 4.5)
        ax.set_yticks([1, 2, 3, 4])
        ax.tick_params(axis="y", labelcolor="darkorange", labelsize=7)

    # Dispersion on right y-axis
    if has_disp:
        ax2 = ax.twinx()
        ax2.step(dates, ct.sync_dispersion_upperbound_ms, where="post",
                 color="teal", linewidth=0.8, label="Dispersion")
        ax2.set_ylabel("Dispersion upper bound (ms)", fontsize=8, color="teal")
        ax2.set_yscale("log")
        ax2.tick_params(axis="y", labelcolor="teal", labelsize=7)

    ax.set_title("Per-pulse sync status", fontsize=9)
    ax.tick_params(axis="x", labelsize=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.set_xlabel("UTC time", fontsize=8)
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")


# -- Main entry point ----------------------------------------------------------

def generate_sync_report(
    clock_table,
    output_path,
    raw_signal=None,
    threshold=None,
    pulse_widths=None,
    sps=None,
):
    """Generate a multi-panel sync report PNG.

    Parameters
    ----------
    clock_table : ClockTable
        Fully built ClockTable with metadata.
    output_path : str or path-like
        Path for the output PNG file.
    raw_signal : ndarray, optional
        1-D raw signal array. If None, panels 1-3 are skipped.
    threshold : float, optional
        Threshold used for edge detection (for panel 1).
    pulse_widths : ndarray, optional
        Raw pulse widths in samples (for panel 2 histogram).
    sps : float, optional
        Samples per second. Defaults to clock_table.nominal_rate.
    """
    if not HAS_MATPLOTLIB:
        logger.warning(
            "matplotlib not installed — skipping sync report generation. "
            "Install with: pip install matplotlib"
        )
        return

    if sps is None:
        sps = clock_table.nominal_rate

    has_signal = raw_signal is not None

    # Determine panel layout: 6 panels with signal, 3 without
    if has_signal:
        # Text header + 5 panels
        fig = plt.figure(figsize=(12, 16))
        gs = fig.add_gridspec(
            5, 2,
            top=0.88, bottom=0.04,
            left=0.08, right=0.95,
            hspace=0.45, wspace=0.3,
            height_ratios=[1, 1, 1, 1, 1],
        )
        # Panel 1: zoomed signal (top-left)
        ax_zoom = fig.add_subplot(gs[0, 0])
        _draw_zoomed_signal(ax_zoom, raw_signal, threshold, pulse_widths, sps)

        # Panel 2: histogram (top-right)
        ax_hist = fig.add_subplot(gs[0, 1])
        if pulse_widths is not None:
            _draw_pulse_histogram(ax_hist, pulse_widths, sps)
        else:
            ax_hist.text(0.5, 0.5, "No pulse width data",
                         ha="center", va="center", transform=ax_hist.transAxes)

        # Panel 3: full overview (row 2, full width)
        ax_overview = fig.add_subplot(gs[1, :])
        _draw_full_overview(ax_overview, raw_signal, clock_table)

        # Panel 4: clock drift (row 3, full width)
        ax_drift = fig.add_subplot(gs[2, :])
        _draw_clock_drift(ax_drift, clock_table)

        # Panel 5: sync status (row 4, full width)
        ax_sync = fig.add_subplot(gs[3, :])
        _draw_sync_status(ax_sync, clock_table)

    else:
        # No raw signal — only drift + sync panels
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(
            2, 1,
            top=0.82, bottom=0.08,
            left=0.08, right=0.95,
            hspace=0.45,
        )
        # Panel 4: clock drift
        ax_drift = fig.add_subplot(gs[0])
        _draw_clock_drift(ax_drift, clock_table)

        # Panel 5: sync status
        ax_sync = fig.add_subplot(gs[1])
        _draw_sync_status(ax_sync, clock_table)

    # Text header (always drawn)
    _draw_text_header(fig, clock_table, raw_signal)

    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved sync report to %s", output_path)


def _try_generate_report(clock_table, output_path, raw_signal=None,
                         threshold=None, pulse_widths=None, sps=None):
    """Wrapper that silently skips report generation on any error.

    Called from decode entry points. Catches ImportError (no matplotlib)
    and any unexpected exceptions to ensure decoding is never blocked by
    report generation failures.
    """
    try:
        generate_sync_report(
            clock_table, output_path,
            raw_signal=raw_signal,
            threshold=threshold,
            pulse_widths=pulse_widths,
            sps=sps,
        )
    except Exception:
        logger.warning(
            "Failed to generate sync report at %s", output_path,
            exc_info=True,
        )
