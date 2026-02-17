#!/usr/bin/env python3
"""Mock IRIG sender test: compile, run, and verify pulse timing.

Compiles irig_sender.c with -DMOCK_GPIO, runs it for 1 frame (~60s + up
to 60s wait for minute alignment), then verifies the logged pulse CSV:

1. Minute alignment — first rising edge within 20ms of a :00 boundary
2. Low latency     — all rising edges within 20ms of ideal second boundaries
3. Pulse widths    — classify as 0/1/P, match expected IRIG-H frame pattern
4. BCD seconds     — decoded seconds field is 0
5. No off-by-one   — decoded timestamp matches system time at first pulse

Usage:
    python sender/test_mock_sender.py [--frames N] [--keep]

Options:
    --frames N  Number of frames to send (default: 1)
    --keep      Keep the compiled binary and CSV after the test
"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

# IRIG-H BCD weights (same as in the C sender)
SECONDS_WEIGHTS = [1, 2, 4, 8, 10, 20, 40]
MINUTES_WEIGHTS = [1, 2, 4, 8, 10, 20, 40]
HOURS_WEIGHTS = [1, 2, 4, 8, 10, 20]
DAY_OF_YEAR_WEIGHTS = [1, 2, 4, 8, 10, 20, 40, 80, 100, 200]
YEARS_WEIGHTS = [1, 2, 4, 8, 10, 20, 40, 80]

# Marker positions in a 60-bit IRIG-H frame
MARKER_POSITIONS = {0, 9, 19, 29, 39, 49, 59}

NS_PER_SEC = 1_000_000_000


def bcd_decode(bits, weights):
    """Decode BCD bits using weight array."""
    return sum(b * w for b, w in zip(bits, weights))


def classify_pulse(width_s):
    """Classify pulse width (seconds) as 0, 1, or P (marker)."""
    if 0.10 <= width_s < 0.35:
        return 0
    elif 0.35 <= width_s < 0.60:
        return 1
    elif 0.60 <= width_s < 0.95:
        return 2  # P (marker)
    else:
        return -1  # invalid


def decode_irig_frame(pulse_types):
    """Decode a 60-pulse IRIG-H frame. Returns dict or None on error."""
    if len(pulse_types) != 60:
        return None

    # Verify markers at expected positions
    for pos in MARKER_POSITIONS:
        if pulse_types[pos] != 2:
            return None

    # Extract BCD fields
    seconds_bits = list(pulse_types[1:5]) + list(pulse_types[6:9])
    minutes_bits = list(pulse_types[10:14]) + list(pulse_types[15:18])
    hours_bits = list(pulse_types[20:24]) + list(pulse_types[25:27])
    doy_bits = (list(pulse_types[30:34]) + list(pulse_types[35:39])
                + list(pulse_types[40:42]))
    year_bits = list(pulse_types[50:54]) + list(pulse_types[55:59])

    seconds = bcd_decode(seconds_bits, SECONDS_WEIGHTS)
    minutes = bcd_decode(minutes_bits, MINUTES_WEIGHTS)
    hours = bcd_decode(hours_bits, HOURS_WEIGHTS)
    day_of_year = bcd_decode(doy_bits, DAY_OF_YEAR_WEIGHTS)
    year = 2000 + bcd_decode(year_bits, YEARS_WEIGHTS)

    return {
        "seconds": seconds,
        "minutes": minutes,
        "hours": hours,
        "day_of_year": day_of_year,
        "year": year,
    }


def run_test(n_frames=1, keep=False):
    """Compile, run, and verify the mock IRIG sender."""
    sender_dir = Path(__file__).parent
    src_path = sender_dir / "irig_sender.c"

    if not src_path.exists():
        print(f"FAIL: Source file not found: {src_path}")
        return False

    # Use a temp directory for build artifacts
    tmpdir = tempfile.mkdtemp(prefix="irig_mock_")
    binary_path = os.path.join(tmpdir, "irig_test")
    csv_path = os.path.join(tmpdir, "irig_mock.csv")

    results = {}

    try:
        # ── Compile ──────────────────────────────────────────────────────
        print(f"Compiling {src_path} with -DMOCK_GPIO ...")
        compile_cmd = [
            "gcc", "-DMOCK_GPIO", "-O2", "-Wall",
            "-o", binary_path,
            str(src_path),
            "-lpthread",
        ]
        result = subprocess.run(compile_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FAIL: Compilation failed:\n{result.stderr}")
            return False
        print("  Compiled OK")

        # ── Run ──────────────────────────────────────────────────────────
        wait_secs = 60 - time.time() % 60
        run_time = int(wait_secs + n_frames * 62 + 10)
        print(f"Running mock sender for {n_frames} frame(s) "
              f"(~{wait_secs:.0f}s wait + {n_frames * 60}s sending, "
              f"timeout {run_time}s) ...")

        run_cmd = [
            binary_path,
            "--frames", str(n_frames),
            "--mock-log", csv_path,
        ]
        result = subprocess.run(
            run_cmd, capture_output=True, text=True, timeout=run_time
        )
        if result.returncode != 0:
            print(f"FAIL: Sender exited with code {result.returncode}")
            print(result.stdout[-500:] if result.stdout else "")
            print(result.stderr[-500:] if result.stderr else "")
            return False
        print("  Ran OK")

        # ── Parse CSV ────────────────────────────────────────────────────
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            print("FAIL: No pulses logged")
            return False

        rising_ns = [int(r["rising_edge_ns"]) for r in rows]
        falling_ns = [int(r["falling_edge_ns"]) for r in rows]
        n_pulses = len(rows)
        print(f"  Parsed {n_pulses} pulses")

        expected_pulses = n_frames * 60
        if n_pulses != expected_pulses:
            print(f"FAIL: Expected {expected_pulses} pulses, got {n_pulses}")
            return False

        # ── Check 1: Minute alignment ────────────────────────────────────
        first_rising_s = rising_ns[0] / NS_PER_SEC
        fractional_minute = first_rising_s % 60
        # Should be close to 0 (or 60)
        minute_error_ms = min(fractional_minute, 60 - fractional_minute) * 1000
        threshold_ms = 20.0
        ok = minute_error_ms < threshold_ms
        results["minute_alignment"] = ok
        dt = datetime.fromtimestamp(first_rising_s, tz=timezone.utc)
        print(f"\n1. Minute alignment: first pulse at {dt.strftime('%H:%M:%S.%f')} UTC")
        print(f"   Error from :00 boundary: {minute_error_ms:.1f} ms "
              f"{'PASS' if ok else 'FAIL'} (threshold {threshold_ms:.0f} ms)")

        # ── Check 2: Low latency (rising edges vs ideal 1s grid) ─────────
        max_error_ms = 0.0
        errors_ms = []
        for i in range(n_pulses):
            expected_ns = rising_ns[0] + i * NS_PER_SEC
            error_ms = abs(rising_ns[i] - expected_ns) / 1e6
            errors_ms.append(error_ms)
            if error_ms > max_error_ms:
                max_error_ms = error_ms

        ok = max_error_ms < threshold_ms
        results["low_latency"] = ok
        print(f"\n2. Low latency: max jitter = {max_error_ms:.2f} ms "
              f"{'PASS' if ok else 'FAIL'} (threshold {threshold_ms:.0f} ms)")
        if not ok:
            # Show worst offenders
            worst = sorted(enumerate(errors_ms), key=lambda x: -x[1])[:5]
            for idx, err in worst:
                print(f"   Pulse {idx}: {err:.2f} ms")

        # ── Check 3: Correct pulse widths ────────────────────────────────
        pulse_types = []
        bad_pulses = []
        for i in range(n_pulses):
            width_s = (falling_ns[i] - rising_ns[i]) / NS_PER_SEC
            pt = classify_pulse(width_s)
            pulse_types.append(pt)
            if pt == -1:
                bad_pulses.append((i, width_s))

        ok = len(bad_pulses) == 0
        results["pulse_widths"] = ok
        print(f"\n3. Pulse width classification: {sum(1 for p in pulse_types if p == 0)} zeros, "
              f"{sum(1 for p in pulse_types if p == 1)} ones, "
              f"{sum(1 for p in pulse_types if p == 2)} markers, "
              f"{len(bad_pulses)} invalid  "
              f"{'PASS' if ok else 'FAIL'}")
        if bad_pulses:
            for idx, w in bad_pulses[:5]:
                print(f"   Pulse {idx}: width = {w*1000:.1f} ms")

        # Verify markers at expected positions
        marker_ok = True
        for pos in MARKER_POSITIONS:
            for frame_idx in range(n_frames):
                p = frame_idx * 60 + pos
                if p < n_pulses and pulse_types[p] != 2:
                    print(f"   Expected marker at position {pos} (frame {frame_idx}), "
                          f"got type {pulse_types[p]}")
                    marker_ok = False
        results["marker_positions"] = marker_ok
        print(f"   Marker positions: {'PASS' if marker_ok else 'FAIL'}")

        # ── Check 4: BCD seconds = 0 ────────────────────────────────────
        all_seconds_zero = True
        for frame_idx in range(n_frames):
            frame_pulses = pulse_types[frame_idx * 60 : (frame_idx + 1) * 60]
            decoded = decode_irig_frame(frame_pulses)
            if decoded is None:
                print(f"\n4. BCD decode: FAIL (frame {frame_idx} decode error)")
                all_seconds_zero = False
                break
            if decoded["seconds"] != 0:
                print(f"\n4. BCD seconds: FAIL (frame {frame_idx}: "
                      f"seconds={decoded['seconds']}, expected 0)")
                all_seconds_zero = False

        results["bcd_seconds_zero"] = all_seconds_zero
        if all_seconds_zero:
            print(f"\n4. BCD seconds = 0: PASS")

        # ── Check 5: No off-by-one ──────────────────────────────────────
        # Decode the first frame and compare to system time
        first_frame = pulse_types[0:60]
        decoded = decode_irig_frame(first_frame)
        if decoded:
            # Build expected datetime from the first pulse timestamp
            first_pulse_dt = datetime.fromtimestamp(
                rising_ns[0] / NS_PER_SEC, tz=timezone.utc
            )
            expected_minute = first_pulse_dt.minute
            expected_hour = first_pulse_dt.hour
            expected_doy = first_pulse_dt.timetuple().tm_yday
            expected_year = first_pulse_dt.year

            match = (
                decoded["seconds"] == 0
                and decoded["minutes"] == expected_minute
                and decoded["hours"] == expected_hour
                and decoded["day_of_year"] == expected_doy
                and decoded["year"] == expected_year
            )
            results["no_off_by_one"] = match
            print(f"\n5. No off-by-one: decoded={decoded['year']}-{decoded['day_of_year']:03d} "
                  f"{decoded['hours']:02d}:{decoded['minutes']:02d}:{decoded['seconds']:02d}, "
                  f"expected={expected_year}-{expected_doy:03d} "
                  f"{expected_hour:02d}:{expected_minute:02d}:00  "
                  f"{'PASS' if match else 'FAIL'}")
        else:
            results["no_off_by_one"] = False
            print(f"\n5. No off-by-one: FAIL (could not decode first frame)")

        # ── Summary ──────────────────────────────────────────────────────
        all_pass = all(results.values())
        print(f"\n{'='*60}")
        print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
        for name, ok in results.items():
            print(f"  {name}: {'PASS' if ok else 'FAIL'}")
        print(f"{'='*60}")
        return all_pass

    except subprocess.TimeoutExpired:
        print("FAIL: Sender timed out")
        return False
    finally:
        if not keep:
            for f in [binary_path, csv_path]:
                if os.path.exists(f):
                    os.remove(f)
            if os.path.exists(tmpdir):
                os.rmdir(tmpdir)
        else:
            print(f"\nArtifacts kept in: {tmpdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test mock IRIG sender")
    parser.add_argument("--frames", type=int, default=1,
                        help="Number of frames to send (default: 1)")
    parser.add_argument("--keep", action="store_true",
                        help="Keep compiled binary and CSV after test")
    args = parser.parse_args()

    ok = run_test(n_frames=args.frames, keep=args.keep)
    sys.exit(0 if ok else 1)
