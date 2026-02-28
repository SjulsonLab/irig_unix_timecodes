"""Tests for neurokairos.decoder — unified IRIGDecoder class.

Reuses fixtures from conftest.py (generate_test_dat, sglx_nidq) and
from test_events.py (medpc_file, csv_events_file).
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import generate_irig_h_frame, _bit_to_pulse_frac

from neurokairos.decoders.decoder import IRIGDecoder
from neurokairos.clock_table import ClockTable
from neurokairos.decoders.irig import decode_dat_irig, decode_intervals_irig
from neurokairos.decoders.sglx import decode_sglx_irig
from neurokairos.decoders.events import (
    parse_medpc_file,
    extract_medpc_events,
    extract_irig_pulses,
    parse_csv_events,
    filter_non_pulse_events,
)

# Event codes (same as test_events.py)
PULSE_HIGH_CODE = 11
PULSE_LOW_CODE = 12
LICK_CODE = 1
NOSEPOKE_CODE = 2
REWARD_CODE = 3
LEVER_PRESS_CODE = 4

START_DT = datetime(2026, 1, 15, 14, 30, 0, tzinfo=timezone.utc)


# -- Import fixtures from test_events -----------------------------------------
# pytest discovers fixtures from conftest.py automatically; for test_events
# fixtures we re-import them here so they're available.

from test_events import medpc_file, csv_events_file  # noqa: F401


# -- Tests: from_dat ----------------------------------------------------------

class TestIRIGDecoderFromDat:
    """Test IRIGDecoder.from_dat produces correct ClockTable."""

    def test_returns_clock_table(self, generate_test_dat):
        meta = generate_test_dat
        decoder = IRIGDecoder.from_dat(
            meta["path"], meta["n_channels"], meta["irig_channel"], save=False
        )
        ct = decoder.decode()
        assert isinstance(ct, ClockTable)
        assert len(ct.source) >= 200

    def test_matches_standalone(self, generate_test_dat):
        meta = generate_test_dat
        # Standalone function
        ct_standalone = decode_dat_irig(
            meta["path"], meta["n_channels"], meta["irig_channel"], save=False
        )
        # Decoder
        decoder = IRIGDecoder.from_dat(
            meta["path"], meta["n_channels"], meta["irig_channel"], save=False
        )
        ct_decoder = decoder.decode()

        np.testing.assert_array_equal(ct_standalone.source, ct_decoder.source)
        np.testing.assert_array_equal(
            ct_standalone.reference, ct_decoder.reference
        )

    def test_source_type(self, generate_test_dat):
        meta = generate_test_dat
        decoder = IRIGDecoder.from_dat(
            meta["path"], meta["n_channels"], meta["irig_channel"], save=False
        )
        assert decoder.source_type == "dat"


# -- Tests: from_sglx ---------------------------------------------------------

class TestIRIGDecoderFromSglx:
    """Test IRIGDecoder.from_sglx produces correct ClockTable."""

    def test_returns_clock_table(self, sglx_nidq):
        decoder = IRIGDecoder.from_sglx(
            sglx_nidq["bin_path"], sglx_nidq["irig_channel"], save=False
        )
        ct = decoder.decode()
        assert isinstance(ct, ClockTable)
        assert len(ct.source) >= 200

    def test_matches_standalone(self, sglx_nidq):
        ct_standalone = decode_sglx_irig(
            sglx_nidq["bin_path"], sglx_nidq["irig_channel"], save=False
        )
        decoder = IRIGDecoder.from_sglx(
            sglx_nidq["bin_path"], sglx_nidq["irig_channel"], save=False
        )
        ct_decoder = decoder.decode()

        np.testing.assert_array_equal(ct_standalone.source, ct_decoder.source)
        np.testing.assert_array_equal(
            ct_standalone.reference, ct_decoder.reference
        )


# -- Tests: from_intervals ----------------------------------------------------

class TestIRIGDecoderFromIntervals:
    """Test IRIGDecoder.from_intervals with pre-extracted pulse data."""

    def test_returns_clock_table(self, medpc_file):  # noqa: F811
        data = parse_medpc_file(medpc_file["path"])
        timestamps, codes = extract_medpc_events(data, array_name="C")
        onsets, offsets = extract_irig_pulses(
            timestamps, codes, PULSE_HIGH_CODE, PULSE_LOW_CODE
        )

        decoder = IRIGDecoder.from_intervals(onsets, offsets)
        ct = decoder.decode()
        assert isinstance(ct, ClockTable)
        assert len(ct.source) >= 200

    def test_matches_standalone(self, medpc_file):  # noqa: F811
        data = parse_medpc_file(medpc_file["path"])
        timestamps, codes = extract_medpc_events(data, array_name="C")
        onsets, offsets = extract_irig_pulses(
            timestamps, codes, PULSE_HIGH_CODE, PULSE_LOW_CODE
        )

        ct_standalone = decode_intervals_irig(
            onsets, offsets, source_units="seconds"
        )
        decoder = IRIGDecoder.from_intervals(onsets, offsets)
        ct_decoder = decoder.decode()

        np.testing.assert_array_equal(ct_standalone.source, ct_decoder.source)
        np.testing.assert_array_equal(
            ct_standalone.reference, ct_decoder.reference
        )


# -- Tests: from_events (MedPC) -----------------------------------------------

class TestIRIGDecoderFromEvents:
    """Test IRIGDecoder.from_events with MedPC format."""

    def test_requires_pulse_codes(self, medpc_file):  # noqa: F811
        with pytest.raises(ValueError, match="pulse_high_code"):
            IRIGDecoder.from_events(medpc_file["path"], format="medpc")

    def test_full_pipeline(self, medpc_file):  # noqa: F811
        decoder = IRIGDecoder.from_events(
            medpc_file["path"],
            format="medpc",
            pulse_high_code=PULSE_HIGH_CODE,
            pulse_low_code=PULSE_LOW_CODE,
        )
        ct = decoder.decode()
        assert isinstance(ct, ClockTable)
        assert len(ct.source) >= 200

    def test_source_type(self, medpc_file):  # noqa: F811
        decoder = IRIGDecoder.from_events(
            medpc_file["path"],
            format="medpc",
            pulse_high_code=PULSE_HIGH_CODE,
            pulse_low_code=PULSE_LOW_CODE,
        )
        assert decoder.source_type == "events"


# -- Tests: from_events (CSV) -------------------------------------------------

class TestIRIGDecoderFromEventsCSV:
    """Test IRIGDecoder.from_events with CSV format."""

    def test_full_pipeline(self, csv_events_file):  # noqa: F811
        decoder = IRIGDecoder.from_events(
            csv_events_file["path"],
            format="csv",
            pulse_high_code=PULSE_HIGH_CODE,
            pulse_low_code=PULSE_LOW_CODE,
        )
        ct = decoder.decode()
        assert isinstance(ct, ClockTable)
        assert len(ct.source) >= 200


# -- Tests: behavioral events -------------------------------------------------

class TestBehavioralEventsUTC:
    """Test post-decode behavioral event extraction."""

    def test_get_behavioral_events_utc(self, medpc_file):  # noqa: F811
        decoder = IRIGDecoder.from_events(
            medpc_file["path"],
            format="medpc",
            pulse_high_code=PULSE_HIGH_CODE,
            pulse_low_code=PULSE_LOW_CODE,
        )
        decoder.decode()

        events = decoder.get_behavioral_events_utc()
        assert len(events) == 10  # 10 behavioral events defined in test_events

        start_ts = START_DT.timestamp()
        # First event: lick at 5.0s
        assert abs(events[0]["utc_timestamp"] - (start_ts + 5.0)) < 1.5
        assert events[0]["event_code"] == LICK_CODE

    def test_event_names(self, medpc_file):  # noqa: F811
        decoder = IRIGDecoder.from_events(
            medpc_file["path"],
            format="medpc",
            pulse_high_code=PULSE_HIGH_CODE,
            pulse_low_code=PULSE_LOW_CODE,
        )
        decoder.decode()

        event_names = {
            LICK_CODE: "lick",
            NOSEPOKE_CODE: "nosepoke",
            REWARD_CODE: "reward",
            LEVER_PRESS_CODE: "lever_press",
        }
        events = decoder.get_behavioral_events_utc(event_names=event_names)
        assert events[0]["event_name"] == "lick"
        assert events[1]["event_name"] == "nosepoke"

    def test_save_behavioral_events_csv(self, medpc_file, tmp_path):  # noqa: F811
        decoder = IRIGDecoder.from_events(
            medpc_file["path"],
            format="medpc",
            pulse_high_code=PULSE_HIGH_CODE,
            pulse_low_code=PULSE_LOW_CODE,
        )
        decoder.decode()

        out_path = tmp_path / "behavioral_events.csv"
        decoder.save_behavioral_events_csv(out_path)
        assert out_path.exists()

        with open(out_path) as f:
            lines = f.readlines()
        # Header + 10 behavioral events
        assert len(lines) == 11


# -- Tests: error handling -----------------------------------------------------

class TestDecoderErrors:
    """Test error conditions."""

    def test_events_without_pulse_codes_raises(self, medpc_file):  # noqa: F811
        with pytest.raises(ValueError, match="pulse_high_code"):
            IRIGDecoder.from_events(medpc_file["path"], format="medpc")

    def test_behavioral_events_before_decode_raises(self, medpc_file):  # noqa: F811
        decoder = IRIGDecoder.from_events(
            medpc_file["path"],
            format="medpc",
            pulse_high_code=PULSE_HIGH_CODE,
            pulse_low_code=PULSE_LOW_CODE,
        )
        with pytest.raises(RuntimeError, match="decode"):
            decoder.get_behavioral_events_utc()

    def test_behavioral_events_on_non_events_source_raises(
        self, generate_test_dat
    ):
        meta = generate_test_dat
        decoder = IRIGDecoder.from_dat(
            meta["path"], meta["n_channels"], meta["irig_channel"], save=False
        )
        decoder.decode()
        with pytest.raises(TypeError, match="events"):
            decoder.get_behavioral_events_utc()

    def test_clock_table_none_before_decode(self, generate_test_dat):
        meta = generate_test_dat
        decoder = IRIGDecoder.from_dat(
            meta["path"], meta["n_channels"], meta["irig_channel"], save=False
        )
        assert decoder.clock_table is None
