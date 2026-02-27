#!/usr/bin/env python3
"""Generate IRIG-H frame structure figure for NeuroKairos documentation.

Creates a technical diagram showing all 60 bits of the NeuroKairos-modified
IRIG-H timecode frame, with pulse-width encoding, BCD weights, field
groupings, and sync status extensions.

Usage:
    uv run --with matplotlib python docs/generate_irig_h_figure.py
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ---------------------------------------------------------------------------
# IRIG-H frame definition: 60 bits
# Each entry: (bit_index, bcd_weight_label, field_name)
# ---------------------------------------------------------------------------
BITS = [
    # Bit 0: Reference marker
    (0, "P", "marker"),
    # Bits 1-8: Seconds (always 0 in IRIG-H)
    (1, "1", "seconds"), (2, "2", "seconds"), (3, "4", "seconds"),
    (4, "8", "seconds"),
    (5, "0", "unused"),
    (6, "10", "seconds"), (7, "20", "seconds"), (8, "40", "seconds"),
    # Bit 9: P1
    (9, "P", "marker"),
    # Bits 10-18: Minutes
    (10, "1", "minutes"), (11, "2", "minutes"), (12, "4", "minutes"),
    (13, "8", "minutes"),
    (14, "0", "unused"),
    (15, "10", "minutes"), (16, "20", "minutes"), (17, "40", "minutes"),
    (18, "0", "unused"),
    # Bit 19: P2
    (19, "P", "marker"),
    # Bits 20-28: Hours
    (20, "1", "hours"), (21, "2", "hours"), (22, "4", "hours"),
    (23, "8", "hours"),
    (24, "0", "unused"),
    (25, "10", "hours"), (26, "20", "hours"),
    (27, "0", "unused"), (28, "0", "unused"),
    # Bit 29: P3
    (29, "P", "marker"),
    # Bits 30-38: Day of year (ones + tens)
    (30, "1", "day"), (31, "2", "day"), (32, "4", "day"), (33, "8", "day"),
    (34, "0", "unused"),
    (35, "10", "day"), (36, "20", "day"), (37, "40", "day"), (38, "80", "day"),
    # Bit 39: P4
    (39, "P", "marker"),
    # Bits 40-41: Day of year (hundreds)
    (40, "100", "day"), (41, "200", "day"),
    # Bit 42: Reserved (NeuroKairos: reserved, always 0)
    (42, "0", "nk_reserved"),
    # Bits 43-44: NeuroKairos stratum
    (43, "S0", "nk_stratum"), (44, "S1", "nk_stratum"),
    # Bit 45: Reserved (NeuroKairos: reserved, always 0)
    (45, "0", "nk_reserved"),
    # Bits 46-48: NeuroKairos root dispersion
    (46, "D0", "nk_dispersion"), (47, "D1", "nk_dispersion"),
    (48, "D2", "nk_dispersion"),
    # Bit 49: P5
    (49, "P", "marker"),
    # Bits 50-58: Year
    (50, "1", "year"), (51, "2", "year"), (52, "4", "year"),
    (53, "8", "year"),
    (54, "0", "unused"),
    (55, "10", "year"), (56, "20", "year"), (57, "40", "year"),
    (58, "80", "year"),
    # Bit 59: P6 / next frame P_R
    (59, "P", "marker"),
]

# ---------------------------------------------------------------------------
# Color scheme
# ---------------------------------------------------------------------------
FIELD_COLORS = {
    "marker":        "#2c3e50",   # dark blue-gray
    "seconds":       "#bdc3c7",   # light gray (always 0)
    "minutes":       "#3498db",   # blue
    "hours":         "#2ecc71",   # green
    "day":           "#e67e22",   # orange
    "year":          "#9b59b6",   # purple
    "unused":        "#ecf0f1",   # very light gray
    "nk_reserved":   "#f5b7b1",   # light red
    "nk_stratum":    "#e74c3c",   # red
    "nk_dispersion": "#e74c3c",   # red
}

FIELD_TEXT_COLORS = {
    "marker": "white", "seconds": "black", "minutes": "white",
    "hours": "white", "day": "white", "year": "white",
    "unused": "#95a5a6", "nk_reserved": "#7f1d1d",
    "nk_stratum": "white", "nk_dispersion": "white",
}

# Marker labels (LaTeX subscripts)
MARKER_LABELS = {
    0: "$P_R$", 9: "$P_1$", 19: "$P_2$", 29: "$P_3$",
    39: "$P_4$", 49: "$P_5$", 59: "$P_6$",
}

# Field groupings: (start_bit, end_bit, label)
FIELD_GROUPS = [
    (1, 8, "SECONDS\n(always 00)"),
    (10, 17, "MINUTES"),
    (20, 26, "HOURS"),
    (30, 41, "DAY OF YEAR"),
    (42, 48, "SYNC STATUS\n(NeuroKairos)"),
    (50, 58, "YEAR"),
]


def draw_pulse_example(ax, x_center, y_base, width_frac, label, sublabel):
    """Draw a single pulse-width example with callout annotations."""
    period = 2.5   # visual width of one full period
    h = 1.2        # pulse height
    x_left = x_center - period / 2

    # LOW-HIGH-LOW waveform
    pulse_w = width_frac * period
    ax.plot([x_left, x_left], [y_base, y_base + h],
            color="black", linewidth=1.5)
    ax.plot([x_left, x_left + pulse_w], [y_base + h, y_base + h],
            color="black", linewidth=1.5)
    ax.plot([x_left + pulse_w, x_left + pulse_w], [y_base + h, y_base],
            color="black", linewidth=1.5)
    ax.plot([x_left + pulse_w, x_left + period], [y_base, y_base],
            color="black", linewidth=1.0)

    # Width arrow above pulse
    arrow_y = y_base + h + 0.3
    ax.annotate(
        "", xy=(x_left + pulse_w, arrow_y), xytext=(x_left, arrow_y),
        arrowprops=dict(arrowstyle="<->", color="black", lw=1.0),
    )
    ax.text(
        x_left + pulse_w / 2, arrow_y + 0.15, sublabel,
        ha="center", va="bottom", fontsize=9, fontstyle="italic",
    )

    # Label below
    ax.text(
        x_center, y_base - 0.5, label,
        ha="center", va="top", fontsize=10, fontweight="bold",
    )


def draw_field_bracket(ax, x_start, x_end, y, label):
    """Draw a bracket with label above a group of bit boxes."""
    mid = (x_start + x_end) / 2
    tick = 0.3
    ax.plot([x_start, x_start], [y, y + tick], color="black", lw=0.8)
    ax.plot([x_end, x_end], [y, y + tick], color="black", lw=0.8)
    ax.plot([x_start, x_end], [y + tick, y + tick], color="black", lw=0.8)
    ax.text(
        mid, y + tick + 0.15, label,
        ha="center", va="bottom", fontsize=8.5, fontweight="bold",
        linespacing=1.05,
    )


def main():
    # -----------------------------------------------------------------------
    # Layout: x = column (0..29), y = vertical position
    # No equal aspect — let matplotlib scale naturally
    # -----------------------------------------------------------------------
    cols = 30
    box_w = 0.85    # fraction of one column
    box_h = 1.5     # data units
    col_step = 1.0  # one column per data unit

    # Row y-centers
    row0_y = 10.0   # bits 0-29
    row1_y = 2.5    # bits 30-59
    row_gap = row0_y - row1_y  # vertical separation

    fig, ax = plt.subplots(1, 1, figsize=(22, 13))
    ax.set_xlim(-1.5, cols + 0.5)
    ax.set_ylim(-7.5, 16.5)
    ax.axis("off")

    # Title
    ax.text(
        cols / 2, 15.8,
        "NeuroKairos IRIG-H Frame Structure  (60 bits, 1 bit/second)",
        ha="center", va="center", fontsize=16, fontweight="bold",
    )
    ax.text(
        cols / 2, 15.0,
        "1 frame = 60 seconds = 1 minute.  Each pulse rising edge occurs at the start of a second.",
        ha="center", va="center", fontsize=10, color="#555555",
    )

    # -----------------------------------------------------------------------
    # Draw bit boxes for each row
    # -----------------------------------------------------------------------
    rows = [
        ([b for b in BITS if b[0] < 30], row0_y, 0),
        ([b for b in BITS if b[0] >= 30], row1_y, 30),
    ]

    for bits_in_row, y_ctr, bit_offset in rows:
        for bit_idx, weight_label, field in bits_in_row:
            col = bit_idx - bit_offset
            x = col * col_step + (1 - box_w) / 2  # center in column
            color = FIELD_COLORS[field]
            text_color = FIELD_TEXT_COLORS[field]

            # Box
            rect = FancyBboxPatch(
                (x, y_ctr - box_h / 2), box_w, box_h,
                boxstyle="round,pad=0.03",
                facecolor=color, edgecolor="black", linewidth=0.8,
            )
            ax.add_patch(rect)

            # Bit index above box
            ax.text(
                col * col_step + 0.5, y_ctr + box_h / 2 + 0.15,
                str(bit_idx), ha="center", va="bottom",
                fontsize=7, color="#666666",
            )

            # Content inside box
            cx = col * col_step + 0.5
            if field == "marker":
                ax.text(
                    cx, y_ctr, MARKER_LABELS.get(bit_idx, "P"),
                    ha="center", va="center", fontsize=11,
                    color=text_color, fontweight="bold",
                )
            elif field in ("nk_stratum", "nk_dispersion"):
                ax.text(
                    cx, y_ctr, weight_label,
                    ha="center", va="center", fontsize=9,
                    color=text_color, fontweight="bold",
                )
            elif weight_label == "0":
                ax.text(
                    cx, y_ctr, "0",
                    ha="center", va="center", fontsize=9,
                    color=text_color,
                )
            else:
                ax.text(
                    cx, y_ctr, weight_label,
                    ha="center", va="center", fontsize=9,
                    color=text_color, fontweight="bold",
                )

    # -----------------------------------------------------------------------
    # Field group brackets
    # -----------------------------------------------------------------------
    bracket_offset = box_h / 2 + 0.55

    for start, end, label in FIELD_GROUPS:
        if end < 30:
            # Row 0
            row_off = 0
            y = row0_y + bracket_offset
        elif start >= 30:
            row_off = 30
            y = row1_y + bracket_offset
        else:
            # Spans both rows — show on row 1
            row_off = 30
            y = row1_y + bracket_offset

        x_start = (start - row_off) * col_step + (1 - box_w) / 2
        x_end = (end - row_off) * col_step + (1 - box_w) / 2 + box_w
        draw_field_bracket(ax, x_start, x_end, y, label)

    # -----------------------------------------------------------------------
    # Row labels (left side)
    # -----------------------------------------------------------------------
    ax.text(
        -1.2, row0_y, "Bits\n0–29", ha="center", va="center",
        fontsize=9, fontweight="bold", color="#555555",
    )
    ax.text(
        -1.2, row1_y, "Bits\n30–59", ha="center", va="center",
        fontsize=9, fontweight="bold", color="#555555",
    )

    # -----------------------------------------------------------------------
    # Timeline ruler below each row
    # -----------------------------------------------------------------------
    box_left_offset = (1 - box_w) / 2  # left edge of box within its column
    for row_y, sec_offset in [(row0_y, 0), (row1_y, 30)]:
        ruler_y = row_y - box_h / 2 - 0.5
        for s in range(30):
            # Align ticks with the left edge of each box (= rising edge)
            x = s * col_step + box_left_offset
            tick_h = 0.2 if (s + sec_offset) % 10 != 0 else 0.35
            ax.plot([x, x], [ruler_y, ruler_y - tick_h],
                    color="black", lw=0.5)
            if s % 5 == 0:
                ax.text(
                    x, ruler_y - tick_h - 0.1, f"{s + sec_offset}s",
                    ha="center", va="top", fontsize=7, color="#666666",
                )

    # -----------------------------------------------------------------------
    # Pulse-width examples (bottom section)
    # -----------------------------------------------------------------------
    pulse_y = -3.8
    draw_pulse_example(ax, 4.0, pulse_y, 0.2, "Binary 0", "0.2 s")
    draw_pulse_example(ax, 10.0, pulse_y, 0.5, "Binary 1", "0.5 s")
    draw_pulse_example(ax, 16.5, pulse_y, 0.8, "Position Marker (P)", "0.8 s")

    # 1-second period annotation under first pulse
    p_left = 4.0 - 2.5 / 2
    p_right = p_left + 2.5
    ann_y = pulse_y - 0.65
    ax.annotate(
        "", xy=(p_right, ann_y), xytext=(p_left, ann_y),
        arrowprops=dict(arrowstyle="<->", color="#666666", lw=0.8),
    )
    ax.text(
        4.0, ann_y - 0.25, "1 second period",
        ha="center", va="top", fontsize=8, color="#666666",
    )

    # -----------------------------------------------------------------------
    # Legend (bottom right)
    # -----------------------------------------------------------------------
    legend_x = 21.5
    legend_y = -2.8
    legend_items = [
        (FIELD_COLORS["marker"], "Position Marker (P)"),
        (FIELD_COLORS["seconds"], "Seconds (always 00)"),
        (FIELD_COLORS["minutes"], "Minutes (BCD)"),
        (FIELD_COLORS["hours"], "Hours (BCD)"),
        (FIELD_COLORS["day"], "Day of Year (BCD)"),
        (FIELD_COLORS["year"], "Year (BCD)"),
        (FIELD_COLORS["nk_stratum"], "Sync Status (NeuroKairos)"),
        (FIELD_COLORS["nk_reserved"], "NK Reserved (always 0)"),
        (FIELD_COLORS["unused"], "Unused / Reserved (0)"),
    ]

    ax.text(
        legend_x, legend_y, "Legend",
        fontsize=10, fontweight="bold", va="bottom",
    )

    for i, (color, label) in enumerate(legend_items):
        ly = legend_y - 0.7 - i * 0.6
        rect = FancyBboxPatch(
            (legend_x, ly - 0.2), 0.5, 0.4,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor="black", linewidth=0.5,
        )
        ax.add_patch(rect)
        ax.text(
            legend_x + 0.75, ly, label,
            va="center", fontsize=8.5,
        )

    # -----------------------------------------------------------------------
    # NeuroKairos sync status detail box (below pulse examples)
    # -----------------------------------------------------------------------
    info_x = 1.0
    info_y = -6.5
    info_text = (
        "NeuroKairos Extension (bits 42–48)\n"
        "S0,S1 = NTP Stratum  (00→1, 01→2, 10→3, 11→4+)\n"
        "D0-D2 = Root Dispersion  (000→<0.25ms … 111→≥16ms)\n"
        "Bits 42, 45 = Reserved (always 0)"
    )
    ax.text(
        info_x, info_y, info_text,
        fontsize=8, fontfamily="monospace", va="top",
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor="#fef9e7",
            edgecolor="#e74c3c", linewidth=1.0,
        ),
    )

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    fig.tight_layout(pad=1.0)
    out_path = "docs/neurokairos_irig_h_frame.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
