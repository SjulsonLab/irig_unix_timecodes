"""
Analyze mock IRIG sender timing results.

Reads the mock GPIO log CSV (rising_edge_ns, falling_edge_ns) and computes
per-pulse timing errors relative to the ideal IRIG-H schedule. Produces
histograms showing the distribution of rising-edge and falling-edge latencies,
plus a time-series scatter plot of jitter over the full run.

Usage:
    python mock_analysis.py mock_results.csv
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_mock_csv(path):
    """Load mock GPIO log CSV and return rising/falling edge arrays in nanoseconds."""
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.uint64)
    rising_ns = data[:, 0]
    falling_ns = data[:, 1]
    return rising_ns, falling_ns


def compute_ideal_schedule(rising_ns, falling_ns):
    """
    Reconstruct the ideal pulse schedule from the data.

    The first pulse's rising edge defines the frame start. Subsequent pulses
    should arrive at exactly 1-second intervals. Pulse widths should be exactly
    0.2s, 0.5s, or 0.8s depending on bit type (zero, one, position marker).

    The OFFSET_NS (20 µs) is subtracted from ideal bit start times in the C
    sender to compensate for GPIO write latency. We account for this when
    computing ideal rising edge times.

    Returns:
        ideal_rising_ns: expected rising edge times
        ideal_falling_ns: expected falling edge times (rising + ideal pulse width)
        pulse_durations_ms: actual pulse durations in ms
        ideal_durations_ms: nearest ideal pulse duration in ms
    """
    OFFSET_NS = 20000  # matches C sender
    NS_PER_SEC = 1_000_000_000

    n_pulses = len(rising_ns)

    # Determine which frame (minute) each pulse belongs to by looking for
    # gaps > 1.5s between consecutive rising edges (frame boundaries).
    # Within a frame, pulses are spaced exactly 1s apart.

    # Identify frame boundaries: first pulse, then any pulse where the gap
    # from the previous pulse is > 1.5 seconds (shouldn't happen in contiguous
    # frames, but be safe).
    frame_starts = [0]
    for i in range(1, n_pulses):
        gap_ns = rising_ns[i] - rising_ns[i - 1]
        if gap_ns > 1_500_000_000:
            frame_starts.append(i)

    # Build ideal rising edge times: each frame's first pulse defines the
    # reference, subsequent pulses at +1s intervals.
    ideal_rising_ns = np.empty(n_pulses, dtype=np.int64)
    for fi, start_idx in enumerate(frame_starts):
        end_idx = frame_starts[fi + 1] if fi + 1 < len(frame_starts) else n_pulses
        ref_ns = int(rising_ns[start_idx])
        for j in range(start_idx, end_idx):
            bit_index = j - start_idx
            ideal_rising_ns[j] = ref_ns + bit_index * NS_PER_SEC

    # Actual pulse durations
    pulse_durations_ns = falling_ns.astype(np.int64) - rising_ns.astype(np.int64)
    pulse_durations_ms = pulse_durations_ns / 1e6

    # Classify each pulse to nearest ideal duration (200ms, 500ms, 800ms)
    ideal_duration_options_ms = np.array([200.0, 500.0, 800.0])
    ideal_durations_ms = np.empty(n_pulses)
    for i in range(n_pulses):
        diffs = np.abs(ideal_duration_options_ms - pulse_durations_ms[i])
        ideal_durations_ms[i] = ideal_duration_options_ms[np.argmin(diffs)]

    # Ideal falling edge = actual rising edge + ideal pulse duration
    # (This isolates falling-edge jitter from rising-edge jitter)
    ideal_falling_ns = rising_ns.astype(np.int64) + (ideal_durations_ms * 1e6).astype(np.int64)

    return ideal_rising_ns, ideal_falling_ns, pulse_durations_ms, ideal_durations_ms


def analyze_and_plot(csv_path, output_dir):
    """Run full analysis and save plots."""
    rising_ns, falling_ns = load_mock_csv(csv_path)
    n_pulses = len(rising_ns)
    print(f"Loaded {n_pulses} pulses from {csv_path}")

    ideal_rising_ns, ideal_falling_ns, pulse_durations_ms, ideal_durations_ms = \
        compute_ideal_schedule(rising_ns, falling_ns)

    # Rising edge error: how late (positive) or early (negative) each rising edge is
    rising_error_us = (rising_ns.astype(np.int64) - ideal_rising_ns) / 1e3

    # Falling edge error: actual falling - ideal falling (based on actual rising + ideal width)
    falling_error_us = (falling_ns.astype(np.int64) - ideal_falling_ns) / 1e3

    # Pulse width error: actual duration - ideal duration
    width_error_us = (pulse_durations_ms - ideal_durations_ms) * 1e3

    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"TIMING ANALYSIS SUMMARY ({n_pulses} pulses, {n_pulses/60:.0f} frames)")
    print(f"{'='*60}")

    for name, errors in [("Rising edge", rising_error_us),
                          ("Falling edge", falling_error_us),
                          ("Pulse width", width_error_us)]:
        print(f"\n{name} error (µs):")
        print(f"  Mean:   {np.mean(errors):+.2f}")
        print(f"  Median: {np.median(errors):+.2f}")
        print(f"  Std:    {np.std(errors):.2f}")
        print(f"  Min:    {np.min(errors):+.2f}")
        print(f"  Max:    {np.max(errors):+.2f}")
        print(f"  P1:     {np.percentile(errors, 1):+.2f}")
        print(f"  P99:    {np.percentile(errors, 99):+.2f}")

    # Pulse type breakdown
    print(f"\nPulse type classification:")
    for dur_ms, label in [(200.0, "Zero (200ms)"), (500.0, "One (500ms)"), (800.0, "P marker (800ms)")]:
        mask = ideal_durations_ms == dur_ms
        count = np.sum(mask)
        if count > 0:
            mean_actual = np.mean(pulse_durations_ms[mask])
            print(f"  {label}: n={count}, mean actual={mean_actual:.3f} ms")

    # ── Plots ──

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # ── Figure 1: Overview (full range) ──

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Mock IRIG Sender Timing Analysis — Full Range\n"
                 f"{n_pulses} pulses ({n_pulses/60:.0f} frames), macOS (no SCHED_FIFO)", fontsize=14)

    # 1. Rising edge error histogram (log y-scale to show tail)
    ax = axes[0, 0]
    ax.hist(rising_error_us, bins=100, edgecolor="black", linewidth=0.5, alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_xlabel("Rising edge error (µs)")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Rising Edge Latency — Full Range")
    ax.text(0.98, 0.95, f"median={np.median(rising_error_us):+.1f} µs\n"
                         f"P99={np.percentile(rising_error_us, 99):+.0f} µs\n"
                         f"max={np.max(rising_error_us):+.0f} µs",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    # 2. Pulse width error histogram (log y-scale)
    ax = axes[0, 1]
    ax.hist(width_error_us, bins=100, edgecolor="black", linewidth=0.5, alpha=0.8, color="orange")
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_xlabel("Pulse width error (µs)")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Pulse Width Error — Full Range")
    ax.text(0.98, 0.95, f"median={np.median(width_error_us):+.1f} µs\n"
                         f"P99={np.percentile(width_error_us, 99):+.0f} µs\n"
                         f"max={np.max(width_error_us):+.0f} µs",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    # 3. Rising edge error over time (scatter)
    ax = axes[1, 0]
    elapsed_s = (rising_ns - rising_ns[0]) / 1e9
    ax.scatter(elapsed_s, rising_error_us, s=8, alpha=0.6)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Time since first pulse (s)")
    ax.set_ylabel("Rising edge error (µs)")
    ax.set_title("Rising Edge Jitter Over Time")

    # 4. Pulse width error by pulse type (log y-scale)
    ax = axes[1, 1]
    colors = {"200ms (zero)": "tab:blue", "500ms (one)": "tab:green", "800ms (P)": "tab:red"}
    for dur_ms, label in [(200.0, "200ms (zero)"), (500.0, "500ms (one)"), (800.0, "800ms (P)")]:
        mask = ideal_durations_ms == dur_ms
        if np.sum(mask) > 0:
            ax.hist(width_error_us[mask], bins=50, alpha=0.6, label=label,
                    edgecolor="black", linewidth=0.3, color=colors[label])
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_xlabel("Pulse width error (µs)")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Pulse Width Error by Bit Type")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plot_path = output_dir / "mock_timing_analysis.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")
    plt.close()

    # ── Figure 2: Zoomed to P99 (main distribution body) ──

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    rising_p99 = np.percentile(np.abs(rising_error_us), 99)
    width_p99 = np.percentile(np.abs(width_error_us), 99)
    # Zoom limit: 1.5× the P99, minimum 50 µs
    rising_zoom = max(rising_p99 * 1.5, 50)
    width_zoom = max(width_p99 * 1.5, 50)

    n_rising_in_range = np.sum(np.abs(rising_error_us) <= rising_zoom)
    n_width_in_range = np.sum(np.abs(width_error_us) <= width_zoom)

    fig.suptitle(f"Mock IRIG Sender Timing Analysis — Zoomed to 99th Percentile\n"
                 f"{n_pulses} pulses ({n_pulses/60:.0f} frames), macOS (no SCHED_FIFO)", fontsize=14)

    # 1. Rising edge error histogram (zoomed)
    ax = axes[0, 0]
    mask_zoom = np.abs(rising_error_us) <= rising_zoom
    ax.hist(rising_error_us[mask_zoom], bins=80, edgecolor="black", linewidth=0.5, alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlim(-rising_zoom, rising_zoom)
    ax.set_xlabel("Rising edge error (µs)")
    ax.set_ylabel("Count")
    ax.set_title(f"Rising Edge Latency — Zoomed ({n_rising_in_range}/{n_pulses} shown)")
    ax.text(0.98, 0.95, f"median={np.median(rising_error_us):+.1f} µs\n"
                         f"std={np.std(rising_error_us):.1f} µs\n"
                         f"{n_pulses - n_rising_in_range} outliers clipped",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    # 2. Pulse width error histogram (zoomed)
    ax = axes[0, 1]
    mask_zoom_w = np.abs(width_error_us) <= width_zoom
    ax.hist(width_error_us[mask_zoom_w], bins=80, edgecolor="black", linewidth=0.5, alpha=0.8, color="orange")
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlim(-width_zoom, width_zoom)
    ax.set_xlabel("Pulse width error (µs)")
    ax.set_ylabel("Count")
    ax.set_title(f"Pulse Width Error — Zoomed ({n_width_in_range}/{n_pulses} shown)")
    ax.text(0.98, 0.95, f"median={np.median(width_error_us):+.1f} µs\n"
                         f"std={np.std(width_error_us):.1f} µs\n"
                         f"{n_pulses - n_width_in_range} outliers clipped",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    # 3. Rising edge jitter over time (zoomed y-axis)
    ax = axes[1, 0]
    ax.scatter(elapsed_s, rising_error_us, s=8, alpha=0.6)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_ylim(-rising_zoom, rising_zoom)
    ax.set_xlabel("Time since first pulse (s)")
    ax.set_ylabel("Rising edge error (µs)")
    ax.set_title("Rising Edge Jitter Over Time — Zoomed")

    # 4. CDF of absolute rising edge error
    ax = axes[1, 1]
    sorted_abs = np.sort(np.abs(rising_error_us))
    cdf = np.arange(1, len(sorted_abs) + 1) / len(sorted_abs)
    ax.plot(sorted_abs, cdf * 100, linewidth=2)
    ax.axhline(99, color="gray", linestyle=":", linewidth=1, label="99th percentile")
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Median")
    ax.set_xlabel("|Rising edge error| (µs)")
    ax.set_ylabel("Cumulative % of pulses")
    ax.set_title("CDF of Absolute Rising Edge Error")
    ax.set_xscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path_zoom = output_dir / "mock_timing_analysis_zoomed.png"
    plt.savefig(plot_path_zoom, dpi=150)
    print(f"Zoomed plot saved to {plot_path_zoom}")
    plt.close()

    # Save raw error data as NPZ for further analysis
    npz_path = output_dir / "mock_timing_errors.npz"
    np.savez(npz_path,
             rising_ns=rising_ns,
             falling_ns=falling_ns,
             rising_error_us=rising_error_us,
             falling_error_us=falling_error_us,
             width_error_us=width_error_us,
             pulse_durations_ms=pulse_durations_ms,
             ideal_durations_ms=ideal_durations_ms)
    print(f"Raw error data saved to {npz_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <mock_results.csv> [output_dir]")
        sys.exit(1)

    csv_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    analyze_and_plot(csv_path, output_dir)
