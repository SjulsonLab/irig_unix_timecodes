import numpy as np


def auto_threshold(signal):
    """Find the optimal threshold separating a bimodal signal using Otsu's method.

    Works on the raw signal values — no scipy dependency.  Returns a single
    float threshold.
    """
    n_bins = 256
    hist, bin_edges = np.histogram(signal, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    total = hist.sum()
    if total == 0:
        return float(bin_centers[len(bin_centers) // 2])

    cum_weight = np.cumsum(hist)
    cum_mean = np.cumsum(hist * bin_centers)
    total_mean = cum_mean[-1]

    weight_fg = total - cum_weight
    valid = (cum_weight > 0) & (weight_fg > 0)

    # Safe division — only divide where both classes are non-empty
    safe_wbg = np.where(valid, cum_weight, 1)
    safe_wfg = np.where(valid, weight_fg, 1)

    mean_bg = np.where(valid, cum_mean / safe_wbg, 0.0)
    mean_fg = np.where(valid, (total_mean - cum_mean) / safe_wfg, 0.0)

    between_var = np.where(
        valid,
        cum_weight * weight_fg * (mean_bg - mean_fg) ** 2,
        0.0,
    )

    # When two well-separated populations produce a flat plateau of maximum
    # variance, pick the midpoint of the plateau rather than the first bin.
    max_var = between_var.max()
    if max_var == 0:
        return float(bin_centers[len(bin_centers) // 2])
    at_max = between_var >= max_var * (1 - 1e-9)
    candidates = np.where(at_max)[0]
    best_idx = int(candidates[len(candidates) // 2])
    return float(bin_centers[best_idx])


def detect_edges(signal, threshold):
    """Threshold *signal* and return (rising_edges, falling_edges) as sample
    index arrays.

    A rising edge index is the first sample >= *threshold* after a run of
    samples below.  A falling edge index is the first sample < *threshold*
    after a run of samples at-or-above.
    """
    binary = (signal >= threshold).astype(np.int8)
    diff = np.diff(binary)
    rising_edges = np.where(diff == 1)[0] + 1
    falling_edges = np.where(diff == -1)[0] + 1

    # If the signal starts HIGH we are mid-pulse — there is no real rising
    # edge.  The orphaned falling edge is silently discarded downstream by
    # measure_pulse_widths, which only pairs rising edges with subsequent
    # falling edges.

    return rising_edges, falling_edges


def measure_pulse_widths(rising_edges, falling_edges):
    """Pair each rising edge with the next falling edge.

    Returns ``(pulse_onsets, pulse_widths)`` — both int64 arrays.  Unmatched
    edges (rising edge with no subsequent falling edge, or falling edges before
    the first rising edge) are silently discarded.
    """
    if len(rising_edges) == 0 or len(falling_edges) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    # For each rising edge, find the index of the next falling edge
    idx = np.searchsorted(falling_edges, rising_edges, side="left")

    valid = idx < len(falling_edges)
    rising_valid = rising_edges[valid]
    falling_valid = falling_edges[idx[valid]]

    return rising_valid, falling_valid - rising_valid
