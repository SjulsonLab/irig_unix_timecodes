"""Tests for neurokairos.events — MedPC / CSV event parsing and IRIG pulse
extraction from behavioral event logs.

Uses generate_irig_h_frame from conftest.py to build authentic IRIG-H bit
sequences interleaved with behavioral events.
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import generate_irig_h_frame, _bit_to_pulse_frac

from neurokairos.events import (
    MedPCHeader,
    MedPCData,
    parse_medpc_file,
    extract_medpc_events,
    parse_csv_events,
    extract_irig_pulses,
    filter_non_pulse_events,
    convert_events_to_utc,
    write_events_csv,
)
from neurokairos.irig import decode_intervals_irig


# -- Constants for synthetic data ----------------------------------------------

PULSE_HIGH_CODE = 11
PULSE_LOW_CODE = 12
TIME_RESOLUTION = 0.01  # centiseconds

# Behavioral event codes
LICK_CODE = 1
NOSEPOKE_CODE = 2
REWARD_CODE = 3
LEVER_PRESS_CODE = 4

# Known behavioral event times (local seconds) and their codes
BEHAVIORAL_EVENTS = [
    (5.0, LICK_CODE),
    (12.5, NOSEPOKE_CODE),
    (30.0, REWARD_CODE),
    (45.75, LICK_CODE),
    (60.0, LEVER_PRESS_CODE),
    (90.25, NOSEPOKE_CODE),
    (120.0, REWARD_CODE),
    (150.5, LICK_CODE),
    (180.0, LEVER_PRESS_CODE),
    (200.0, NOSEPOKE_CODE),
]

START_DT = datetime(2026, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
DURATION_S = 240


# -- Helper: build event list from IRIG bits + behavioral events ---------------

def _build_medpc_event_list():
    """Build a list of (centisecond_time, event_code) tuples.

    Interleaves IRIG pulse events (HIGH/LOW) with behavioral events.
    IRIG pulses start at the second boundary: pulse HIGH at t, pulse LOW
    at t + pulse_width_fraction.
    """
    events = []

    # Generate IRIG pulse events for DURATION_S seconds
    minute_start = START_DT
    all_bits = []
    frame_time = minute_start
    end_dt = START_DT + timedelta(seconds=DURATION_S)
    while frame_time < end_dt:
        frame = generate_irig_h_frame(frame_time)
        all_bits.extend(frame)
        frame_time += timedelta(seconds=60)

    recording_bits = all_bits[:DURATION_S]

    for i, bit in enumerate(recording_bits):
        frac = _bit_to_pulse_frac(bit)
        onset_cs = int(i / TIME_RESOLUTION)
        offset_cs = int((i + frac) / TIME_RESOLUTION)
        events.append((onset_cs, PULSE_HIGH_CODE))
        events.append((offset_cs, PULSE_LOW_CODE))

    # Add behavioral events
    for t_sec, code in BEHAVIORAL_EVENTS:
        cs = int(t_sec / TIME_RESOLUTION)
        events.append((cs, code))

    # Sort by time, then by code (to ensure deterministic ordering)
    events.sort(key=lambda x: (x[0], x[1]))
    return events, recording_bits


# -- Fixtures ------------------------------------------------------------------

@pytest.fixture(scope="module")
def medpc_file(tmp_path_factory):
    """Generate a synthetic MedPC file with IRIG pulses and behavioral events.

    The MedPC format encodes TIME.CODE values in arrays, with groups of
    5 values per line, preceded by an index prefix.  Array C holds
    timestamped events.
    """
    events, recording_bits = _build_medpc_event_list()

    # Build MedPC file content
    lines = []
    lines.append("Start Date: 01/15/26")
    lines.append("End Date:   01/15/26")
    lines.append("Subject: TestMouse001")
    lines.append("Experiment: IRIG_Test")
    lines.append("Group: Control")
    lines.append("Box: 1")
    lines.append("Start Time: 14:30:00")
    lines.append("End Time: 14:34:00")
    lines.append("MSN: IRIG_PULSES_V1")
    lines.append("")

    # Array A: simple counter (10 values for testing)
    lines.append("A:")
    a_values = list(range(10)) + [-987.987]
    _write_medpc_array(lines, a_values)
    lines.append("")

    # Array B: reward timestamps (just a few)
    lines.append("B:")
    b_values = [30.0, 120.0, -987.987]
    _write_medpc_array(lines, b_values)
    lines.append("")

    # Array C: timestamped events (TIME.CODE format in centiseconds)
    lines.append("C:")
    c_values = []
    for cs_time, code in events:
        # Encode as TIME.CODE: integer part is centisecond time,
        # fractional part (3 digits) is the event code
        c_values.append(f"{cs_time}.{code:03d}")
    c_values.append("-987.987")
    _write_medpc_array_strings(lines, c_values)
    lines.append("")

    tmp_dir = tmp_path_factory.mktemp("medpc")
    path = tmp_dir / "test_medpc.txt"
    path.write_text("\n".join(lines) + "\n")

    return {
        "path": path,
        "n_events": len(events),
        "n_pulse_events": DURATION_S * 2,  # HIGH + LOW for each pulse
        "n_behavioral_events": len(BEHAVIORAL_EVENTS),
        "recording_bits": recording_bits,
        "start_dt": START_DT,
    }


@pytest.fixture(scope="module")
def csv_events_file(tmp_path_factory):
    """Generate a CSV file with the same events as medpc_file."""
    events, recording_bits = _build_medpc_event_list()

    lines = ["timestamp,event"]
    for cs_time, code in events:
        t_sec = cs_time * TIME_RESOLUTION
        lines.append(f"{t_sec},{code}")

    tmp_dir = tmp_path_factory.mktemp("csv_events")
    path = tmp_dir / "events.csv"
    path.write_text("\n".join(lines) + "\n")

    return {
        "path": path,
        "n_events": len(events),
        "recording_bits": recording_bits,
        "start_dt": START_DT,
    }


@pytest.fixture(scope="module")
def tsv_events_file(tmp_path_factory):
    """Generate a TSV file with the same events."""
    events, _ = _build_medpc_event_list()

    lines = ["timestamp\tevent"]
    for cs_time, code in events:
        t_sec = cs_time * TIME_RESOLUTION
        lines.append(f"{t_sec}\t{code}")

    tmp_dir = tmp_path_factory.mktemp("tsv_events")
    path = tmp_dir / "events.tsv"
    path.write_text("\n".join(lines) + "\n")

    return {"path": path, "n_events": len(events)}


# -- Helpers for writing MedPC arrays ------------------------------------------

def _write_medpc_array(lines, values):
    """Write a MedPC-style array (groups of 5 per line with index prefix)."""
    for i in range(0, len(values), 5):
        chunk = values[i : i + 5]
        prefix = f"     {i}:"
        formatted = "".join(f"{v:13.3f}" for v in chunk)
        lines.append(f"{prefix}{formatted}")


def _write_medpc_array_strings(lines, values):
    """Write a MedPC-style array from string representations."""
    for i in range(0, len(values), 5):
        chunk = values[i : i + 5]
        prefix = f"     {i}:"
        formatted = "".join(f"{v:>13s}" for v in chunk)
        lines.append(f"{prefix}{formatted}")


# -- Test classes --------------------------------------------------------------

class TestParseMedPCFile:
    """Test MedPC file parsing: header fields and array data."""

    def test_header_fields(self, medpc_file):
        data = parse_medpc_file(medpc_file["path"])
        assert data.header.subject == "TestMouse001"
        assert data.header.experiment == "IRIG_Test"
        assert data.header.group == "Control"
        assert data.header.box == "1"
        assert data.header.msn == "IRIG_PULSES_V1"
        assert data.header.start_time == "14:30:00"
        assert data.header.end_time == "14:34:00"

    def test_header_dates(self, medpc_file):
        data = parse_medpc_file(medpc_file["path"])
        assert data.header.start_date == "01/15/26"
        assert data.header.end_date == "01/15/26"

    def test_array_names(self, medpc_file):
        data = parse_medpc_file(medpc_file["path"])
        assert "A" in data.arrays
        assert "B" in data.arrays
        assert "C" in data.arrays

    def test_array_a_values(self, medpc_file):
        data = parse_medpc_file(medpc_file["path"])
        # Array A has 10 values (0-9), sentinel excluded
        assert len(data.arrays["A"]) == 10
        np.testing.assert_array_almost_equal(
            data.arrays["A"], np.arange(10, dtype=np.float64)
        )

    def test_array_b_values(self, medpc_file):
        data = parse_medpc_file(medpc_file["path"])
        # Array B has 2 values, sentinel excluded
        assert len(data.arrays["B"]) == 2
        np.testing.assert_array_almost_equal(
            data.arrays["B"], [30.0, 120.0]
        )

    def test_sentinel_excluded(self, medpc_file):
        data = parse_medpc_file(medpc_file["path"])
        # No array should contain the sentinel value
        for name, arr in data.arrays.items():
            assert not np.any(np.isclose(arr, -987.987)), (
                f"Array {name} contains sentinel value"
            )

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_medpc_file(tmp_path / "nonexistent.txt")


class TestExtractMedPCEvents:
    """Test extraction of timestamped events from MedPC data."""

    def test_event_count(self, medpc_file):
        data = parse_medpc_file(medpc_file["path"])
        timestamps, codes = extract_medpc_events(data, array_name="C")
        assert len(timestamps) == medpc_file["n_events"]
        assert len(codes) == medpc_file["n_events"]

    def test_timestamps_in_seconds(self, medpc_file):
        data = parse_medpc_file(medpc_file["path"])
        timestamps, codes = extract_medpc_events(data, array_name="C")
        # First IRIG pulse at t=0
        assert timestamps[0] == pytest.approx(0.0, abs=0.01)
        # Timestamps should be non-negative
        assert np.all(timestamps >= 0)

    def test_codes_are_integers(self, medpc_file):
        data = parse_medpc_file(medpc_file["path"])
        timestamps, codes = extract_medpc_events(data, array_name="C")
        # All codes should be small positive integers (1-12)
        assert codes.dtype == np.int64 or codes.dtype == np.int32
        assert np.all(codes > 0)
        assert np.all(codes <= 999)

    def test_three_digit_codes(self, medpc_file):
        data = parse_medpc_file(medpc_file["path"])
        timestamps, codes = extract_medpc_events(data, array_name="C")
        # IRIG codes: 11 (HIGH) and 12 (LOW)
        unique_codes = set(codes.tolist())
        assert PULSE_HIGH_CODE in unique_codes
        assert PULSE_LOW_CODE in unique_codes
        # Behavioral codes
        assert LICK_CODE in unique_codes
        assert NOSEPOKE_CODE in unique_codes

    def test_custom_resolution(self, medpc_file):
        data = parse_medpc_file(medpc_file["path"])
        # With millisecond resolution (0.001), timestamps would be 10x larger
        timestamps_cs, _ = extract_medpc_events(
            data, array_name="C", time_resolution=0.01
        )
        timestamps_ms, _ = extract_medpc_events(
            data, array_name="C", time_resolution=0.001
        )
        # Millisecond resolution makes integer parts 10x larger,
        # so timestamps 10x smaller
        # Actually: time = integer_part * time_resolution
        # At 0.01: time = 500 * 0.01 = 5.0 s
        # At 0.001: time = 500 * 0.001 = 0.5 s
        ratio = timestamps_cs[5] / timestamps_ms[5]
        assert ratio == pytest.approx(10.0, rel=0.01)


class TestParseCSVEvents:
    """Test CSV/TSV event file parsing."""

    def test_csv_with_header(self, csv_events_file):
        timestamps, codes = parse_csv_events(csv_events_file["path"])
        assert len(timestamps) == csv_events_file["n_events"]
        assert len(codes) == csv_events_file["n_events"]

    def test_tsv_auto_detect(self, tsv_events_file):
        timestamps, codes = parse_csv_events(tsv_events_file["path"])
        assert len(timestamps) == tsv_events_file["n_events"]

    def test_column_by_name(self, csv_events_file):
        timestamps, codes = parse_csv_events(
            csv_events_file["path"],
            time_column="timestamp",
            event_column="event",
        )
        assert len(timestamps) == csv_events_file["n_events"]

    def test_column_by_index(self, csv_events_file):
        timestamps, codes = parse_csv_events(
            csv_events_file["path"],
            time_column=0,
            event_column=1,
        )
        assert len(timestamps) == csv_events_file["n_events"]

    def test_time_scale(self, csv_events_file):
        # With time_scale=1000, timestamps are in ms instead of s
        timestamps_s, _ = parse_csv_events(csv_events_file["path"])
        timestamps_ms, _ = parse_csv_events(
            csv_events_file["path"], time_scale=1000.0
        )
        # timestamps_ms should be 1000x larger than timestamps_s
        np.testing.assert_allclose(timestamps_ms, timestamps_s * 1000.0)


class TestExtractIRIGPulses:
    """Test extraction of IRIG pulse onsets/offsets from event data."""

    def test_pulse_count(self, csv_events_file):
        timestamps, codes = parse_csv_events(csv_events_file["path"])
        onsets, offsets = extract_irig_pulses(
            timestamps, codes, PULSE_HIGH_CODE, PULSE_LOW_CODE
        )
        assert len(onsets) == DURATION_S
        assert len(offsets) == DURATION_S

    def test_widths_match_irig_fractions(self, medpc_file):
        data = parse_medpc_file(medpc_file["path"])
        timestamps, codes = extract_medpc_events(data, array_name="C")
        onsets, offsets = extract_irig_pulses(
            timestamps, codes, PULSE_HIGH_CODE, PULSE_LOW_CODE
        )
        widths = offsets - onsets
        # All widths should be close to 0.2, 0.5, or 0.8 seconds
        for w in widths:
            assert (
                abs(w - 0.2) < 0.02
                or abs(w - 0.5) < 0.02
                or abs(w - 0.8) < 0.02
            ), f"Unexpected pulse width: {w}"

    def test_onsets_monotonic(self, csv_events_file):
        timestamps, codes = parse_csv_events(csv_events_file["path"])
        onsets, offsets = extract_irig_pulses(
            timestamps, codes, PULSE_HIGH_CODE, PULSE_LOW_CODE
        )
        assert np.all(np.diff(onsets) > 0)

    def test_approximately_one_second_spacing(self, csv_events_file):
        timestamps, codes = parse_csv_events(csv_events_file["path"])
        onsets, _ = extract_irig_pulses(
            timestamps, codes, PULSE_HIGH_CODE, PULSE_LOW_CODE
        )
        intervals = np.diff(onsets)
        # All intervals should be ~1 second
        np.testing.assert_allclose(intervals, 1.0, atol=0.02)


class TestFilterNonPulseEvents:
    """Test filtering out IRIG pulse events to get behavioral events only."""

    def test_excludes_pulse_events(self, csv_events_file):
        timestamps, codes = parse_csv_events(csv_events_file["path"])
        beh_ts, beh_codes = filter_non_pulse_events(
            timestamps, codes, PULSE_HIGH_CODE, PULSE_LOW_CODE
        )
        assert PULSE_HIGH_CODE not in set(beh_codes.tolist())
        assert PULSE_LOW_CODE not in set(beh_codes.tolist())

    def test_retains_behavioral_events(self, csv_events_file):
        timestamps, codes = parse_csv_events(csv_events_file["path"])
        beh_ts, beh_codes = filter_non_pulse_events(
            timestamps, codes, PULSE_HIGH_CODE, PULSE_LOW_CODE
        )
        assert len(beh_ts) == len(BEHAVIORAL_EVENTS)
        unique_codes = set(beh_codes.tolist())
        assert LICK_CODE in unique_codes
        assert NOSEPOKE_CODE in unique_codes
        assert REWARD_CODE in unique_codes
        assert LEVER_PRESS_CODE in unique_codes

    def test_correct_event_codes(self, csv_events_file):
        timestamps, codes = parse_csv_events(csv_events_file["path"])
        beh_ts, beh_codes = filter_non_pulse_events(
            timestamps, codes, PULSE_HIGH_CODE, PULSE_LOW_CODE
        )
        # Check codes match expected behavioral events
        expected_codes = [code for _, code in BEHAVIORAL_EVENTS]
        np.testing.assert_array_equal(beh_codes, expected_codes)


class TestEndToEndMedPC:
    """Full pipeline: parse MedPC → extract IRIG pulses → decode → ClockTable."""

    def test_full_pipeline(self, medpc_file):
        # Parse
        data = parse_medpc_file(medpc_file["path"])
        timestamps, codes = extract_medpc_events(data, array_name="C")

        # Extract IRIG pulses
        onsets, offsets = extract_irig_pulses(
            timestamps, codes, PULSE_HIGH_CODE, PULSE_LOW_CODE
        )

        # Decode IRIG
        ct = decode_intervals_irig(onsets, offsets, source_units="seconds")

        # Verify: should have ~240 entries (one per pulse, minus a few lost)
        assert len(ct.source) >= 200
        assert ct.source_units == "seconds"

    def test_correct_timestamps(self, medpc_file):
        data = parse_medpc_file(medpc_file["path"])
        timestamps, codes = extract_medpc_events(data, array_name="C")
        onsets, offsets = extract_irig_pulses(
            timestamps, codes, PULSE_HIGH_CODE, PULSE_LOW_CODE
        )
        ct = decode_intervals_irig(onsets, offsets, source_units="seconds")

        # The first pulse onset (t=0) should map to START_DT
        start_ts = START_DT.timestamp()
        # Check a pulse near t=60s → should be START_DT + 60s
        ref_at_60 = ct.source_to_reference(60.0)
        assert abs(ref_at_60 - (start_ts + 60.0)) < 1.5


class TestEndToEndCSV:
    """Full pipeline via CSV input."""

    def test_full_pipeline(self, csv_events_file):
        timestamps, codes = parse_csv_events(csv_events_file["path"])
        onsets, offsets = extract_irig_pulses(
            timestamps, codes, PULSE_HIGH_CODE, PULSE_LOW_CODE
        )
        ct = decode_intervals_irig(onsets, offsets, source_units="seconds")

        assert len(ct.source) >= 200
        assert ct.source_units == "seconds"


class TestConvertEventsToUTC:
    """Test conversion of behavioral event timestamps to UTC."""

    def test_correct_utc_timestamps(self, medpc_file):
        data = parse_medpc_file(medpc_file["path"])
        timestamps, codes = extract_medpc_events(data, array_name="C")
        onsets, offsets = extract_irig_pulses(
            timestamps, codes, PULSE_HIGH_CODE, PULSE_LOW_CODE
        )
        ct = decode_intervals_irig(onsets, offsets, source_units="seconds")

        # Get behavioral events
        beh_ts, beh_codes = filter_non_pulse_events(
            timestamps, codes, PULSE_HIGH_CODE, PULSE_LOW_CODE
        )

        events = convert_events_to_utc(ct, beh_ts, beh_codes)

        start_ts = START_DT.timestamp()

        # First behavioral event: lick at 5.0s → 14:30:05 UTC
        assert abs(events[0]["utc_timestamp"] - (start_ts + 5.0)) < 1.5
        assert events[0]["event_code"] == LICK_CODE

        # Second: nosepoke at 12.5s → 14:30:12.5 UTC
        assert abs(events[1]["utc_timestamp"] - (start_ts + 12.5)) < 1.5
        assert events[1]["event_code"] == NOSEPOKE_CODE

    def test_event_names_dict(self, medpc_file):
        data = parse_medpc_file(medpc_file["path"])
        timestamps, codes = extract_medpc_events(data, array_name="C")
        onsets, offsets = extract_irig_pulses(
            timestamps, codes, PULSE_HIGH_CODE, PULSE_LOW_CODE
        )
        ct = decode_intervals_irig(onsets, offsets, source_units="seconds")

        beh_ts, beh_codes = filter_non_pulse_events(
            timestamps, codes, PULSE_HIGH_CODE, PULSE_LOW_CODE
        )

        event_names = {
            LICK_CODE: "lick",
            NOSEPOKE_CODE: "nosepoke",
            REWARD_CODE: "reward",
            LEVER_PRESS_CODE: "lever_press",
        }

        events = convert_events_to_utc(
            ct, beh_ts, beh_codes, event_names=event_names
        )

        assert events[0]["event_name"] == "lick"
        assert events[1]["event_name"] == "nosepoke"
        assert events[2]["event_name"] == "reward"

    def test_event_dict_structure(self, medpc_file):
        data = parse_medpc_file(medpc_file["path"])
        timestamps, codes = extract_medpc_events(data, array_name="C")
        onsets, offsets = extract_irig_pulses(
            timestamps, codes, PULSE_HIGH_CODE, PULSE_LOW_CODE
        )
        ct = decode_intervals_irig(onsets, offsets, source_units="seconds")
        beh_ts, beh_codes = filter_non_pulse_events(
            timestamps, codes, PULSE_HIGH_CODE, PULSE_LOW_CODE
        )

        events = convert_events_to_utc(ct, beh_ts, beh_codes)

        # Check structure of each event dict
        for ev in events:
            assert "utc_timestamp" in ev
            assert "local_timestamp" in ev
            assert "event_code" in ev
            assert "event_name" in ev
            assert isinstance(ev["utc_timestamp"], float)
            assert isinstance(ev["local_timestamp"], float)
            assert isinstance(ev["event_code"], (int, np.integer))


class TestWriteEventsCSV:
    """Test writing converted events to CSV."""

    def test_file_created(self, medpc_file, tmp_path):
        data = parse_medpc_file(medpc_file["path"])
        timestamps, codes = extract_medpc_events(data, array_name="C")
        onsets, offsets = extract_irig_pulses(
            timestamps, codes, PULSE_HIGH_CODE, PULSE_LOW_CODE
        )
        ct = decode_intervals_irig(onsets, offsets, source_units="seconds")
        beh_ts, beh_codes = filter_non_pulse_events(
            timestamps, codes, PULSE_HIGH_CODE, PULSE_LOW_CODE
        )
        events = convert_events_to_utc(ct, beh_ts, beh_codes)

        out_path = tmp_path / "events_out.csv"
        write_events_csv(events, out_path)
        assert out_path.exists()

    def test_correct_columns(self, medpc_file, tmp_path):
        data = parse_medpc_file(medpc_file["path"])
        timestamps, codes = extract_medpc_events(data, array_name="C")
        onsets, offsets = extract_irig_pulses(
            timestamps, codes, PULSE_HIGH_CODE, PULSE_LOW_CODE
        )
        ct = decode_intervals_irig(onsets, offsets, source_units="seconds")
        beh_ts, beh_codes = filter_non_pulse_events(
            timestamps, codes, PULSE_HIGH_CODE, PULSE_LOW_CODE
        )
        events = convert_events_to_utc(ct, beh_ts, beh_codes)

        out_path = tmp_path / "events_cols.csv"
        write_events_csv(events, out_path)

        with open(out_path) as f:
            header = f.readline().strip()
        expected = "utc_timestamp,utc_datetime,local_timestamp,event_code,event_name"
        assert header == expected

    def test_iso8601_datetime(self, medpc_file, tmp_path):
        data = parse_medpc_file(medpc_file["path"])
        timestamps, codes = extract_medpc_events(data, array_name="C")
        onsets, offsets = extract_irig_pulses(
            timestamps, codes, PULSE_HIGH_CODE, PULSE_LOW_CODE
        )
        ct = decode_intervals_irig(onsets, offsets, source_units="seconds")
        beh_ts, beh_codes = filter_non_pulse_events(
            timestamps, codes, PULSE_HIGH_CODE, PULSE_LOW_CODE
        )

        event_names = {LICK_CODE: "lick"}
        events = convert_events_to_utc(
            ct, beh_ts, beh_codes, event_names=event_names
        )

        out_path = tmp_path / "events_iso.csv"
        write_events_csv(events, out_path)

        with open(out_path) as f:
            lines = f.readlines()
        # Second line (first data row) should have ISO 8601 datetime
        parts = lines[1].strip().split(",")
        utc_datetime_str = parts[1]
        # Should parse as ISO format (e.g. "2026-01-15T14:30:05.000000+00:00")
        dt = datetime.fromisoformat(utc_datetime_str)
        assert dt.tzinfo is not None  # Must be timezone-aware

    def test_row_count(self, medpc_file, tmp_path):
        data = parse_medpc_file(medpc_file["path"])
        timestamps, codes = extract_medpc_events(data, array_name="C")
        onsets, offsets = extract_irig_pulses(
            timestamps, codes, PULSE_HIGH_CODE, PULSE_LOW_CODE
        )
        ct = decode_intervals_irig(onsets, offsets, source_units="seconds")
        beh_ts, beh_codes = filter_non_pulse_events(
            timestamps, codes, PULSE_HIGH_CODE, PULSE_LOW_CODE
        )
        events = convert_events_to_utc(ct, beh_ts, beh_codes)

        out_path = tmp_path / "events_rows.csv"
        write_events_csv(events, out_path)

        with open(out_path) as f:
            lines = f.readlines()
        # Header + one row per event
        assert len(lines) == len(BEHAVIORAL_EVENTS) + 1
