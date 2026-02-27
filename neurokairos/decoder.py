"""Unified IRIGDecoder — facade over the standalone decode_* functions.

Provides a single class with factory classmethods for each supported input
format.  ``decode()`` dispatches to the appropriate existing function.
Event-based inputs additionally expose behavioral event extraction after
decoding.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .clock_table import ClockTable

logger = logging.getLogger(__name__)


class IRIGDecoder:
    """Unified interface for IRIG-H timecode decoding.

    Use one of the ``from_*`` classmethods to create an instance, then
    call ``decode()`` to produce a :class:`ClockTable`.

    For event-based inputs (``from_events``), post-decode methods are
    available to extract and convert behavioral events to UTC.
    """

    def __init__(self, source_type: str, config: dict):
        """Private constructor — use classmethods instead."""
        self._source_type = source_type
        self._config = config
        self._clock_table: Optional[ClockTable] = None
        # Stored by _decode_events for later behavioral event extraction
        self._behavioral_timestamps: Optional[np.ndarray] = None
        self._behavioral_codes: Optional[np.ndarray] = None

    # -- Factory classmethods --------------------------------------------------

    @classmethod
    def from_dat(cls, dat_path, n_channels, irig_channel, save=True):
        """Create a decoder for an interleaved int16 .dat file."""
        config = {
            "dat_path": dat_path,
            "n_channels": n_channels,
            "irig_channel": irig_channel,
            "save": save,
        }
        return cls("dat", config)

    @classmethod
    def from_sglx(cls, bin_path, irig_channel, save=True):
        """Create a decoder for a SpikeGLX .bin file."""
        config = {
            "bin_path": bin_path,
            "irig_channel": irig_channel,
            "save": save,
        }
        return cls("sglx", config)

    @classmethod
    def from_video(cls, video_path, roi, fps=None, save=True):
        """Create a decoder for a video file with a visible IRIG LED."""
        config = {
            "video_path": video_path,
            "roi": roi,
            "fps": fps,
            "save": save,
        }
        return cls("video", config)

    @classmethod
    def from_intervals(cls, intervals, offsets=None, source_units="seconds",
                       save=None):
        """Create a decoder from pre-extracted pulse intervals."""
        config = {
            "intervals": intervals,
            "offsets": offsets,
            "source_units": source_units,
            "save": save,
        }
        return cls("intervals", config)

    @classmethod
    def from_events(cls, path, format="medpc", pulse_high_code=None,
                    pulse_low_code=None, array_name="C",
                    time_resolution=0.01, time_column=0, event_column=1,
                    delimiter=None, has_header=True, time_scale=1.0,
                    save=None):
        """Create a decoder from an event log file (MedPC or CSV/TSV).

        Requires ``pulse_high_code`` and ``pulse_low_code`` to identify
        which event codes represent IRIG pulse transitions.
        """
        if pulse_high_code is None or pulse_low_code is None:
            raise ValueError(
                "pulse_high_code and pulse_low_code are required for "
                "event-based decoding"
            )

        config = {
            "path": path,
            "format": format,
            "pulse_high_code": pulse_high_code,
            "pulse_low_code": pulse_low_code,
            "array_name": array_name,
            "time_resolution": time_resolution,
            "time_column": time_column,
            "event_column": event_column,
            "delimiter": delimiter,
            "has_header": has_header,
            "time_scale": time_scale,
            "save": save,
        }
        return cls("events", config)

    # -- Core ------------------------------------------------------------------

    def decode(self) -> ClockTable:
        """Run the decoding pipeline and return a ClockTable."""
        dispatch = {
            "dat": self._decode_dat,
            "sglx": self._decode_sglx,
            "video": self._decode_video,
            "intervals": self._decode_intervals,
            "events": self._decode_events,
        }
        self._clock_table = dispatch[self._source_type]()
        return self._clock_table

    # -- Properties ------------------------------------------------------------

    @property
    def source_type(self) -> str:
        """The input format: 'dat', 'sglx', 'video', 'intervals', 'events'."""
        return self._source_type

    @property
    def clock_table(self) -> Optional[ClockTable]:
        """The decoded ClockTable, or None if decode() hasn't been called."""
        return self._clock_table

    # -- Event-specific post-decode methods ------------------------------------

    def get_behavioral_events_utc(self, event_names=None) -> List[dict]:
        """Convert stored behavioral events to UTC timestamps.

        Only available after ``decode()`` on an ``events`` source.

        Parameters
        ----------
        event_names : dict, optional
            Mapping from event code → human-readable name.

        Returns
        -------
        list of dict
            Each dict has keys: utc_timestamp, local_timestamp,
            event_code, event_name.
        """
        if self._clock_table is None:
            raise RuntimeError(
                "Must call decode() before get_behavioral_events_utc()"
            )
        if self._source_type != "events":
            raise TypeError(
                "get_behavioral_events_utc() is only available for events "
                "sources, not '{}'".format(self._source_type)
            )

        from .events import convert_events_to_utc

        return convert_events_to_utc(
            self._clock_table,
            self._behavioral_timestamps,
            self._behavioral_codes,
            event_names=event_names,
        )

    def save_behavioral_events_csv(self, path, event_names=None):
        """Write behavioral events to a CSV file.

        Parameters
        ----------
        path : str or Path
            Output CSV path.
        event_names : dict, optional
            Mapping from event code → human-readable name.
        """
        from .events import write_events_csv

        events = self.get_behavioral_events_utc(event_names=event_names)
        write_events_csv(events, path)

    # -- Internal dispatch methods ---------------------------------------------

    def _decode_dat(self) -> ClockTable:
        """Decode from interleaved int16 .dat file."""
        from .irig import decode_dat_irig

        return decode_dat_irig(
            self._config["dat_path"],
            self._config["n_channels"],
            self._config["irig_channel"],
            save=self._config["save"],
        )

    def _decode_sglx(self) -> ClockTable:
        """Decode from SpikeGLX .bin file."""
        from .sglx import decode_sglx_irig

        return decode_sglx_irig(
            self._config["bin_path"],
            self._config["irig_channel"],
            save=self._config["save"],
        )

    def _decode_video(self) -> ClockTable:
        """Decode from video file with IRIG LED."""
        from .video import decode_video_irig

        return decode_video_irig(
            self._config["video_path"],
            self._config["roi"],
            fps=self._config["fps"],
            save=self._config["save"],
        )

    def _decode_intervals(self) -> ClockTable:
        """Decode from pre-extracted pulse intervals."""
        from .irig import decode_intervals_irig

        return decode_intervals_irig(
            self._config["intervals"],
            offsets=self._config["offsets"],
            source_units=self._config["source_units"],
            save=self._config["save"],
        )

    def _decode_events(self) -> ClockTable:
        """Decode from event log file (MedPC or CSV/TSV).

        Parses the file, extracts IRIG pulses, decodes them, and stores
        the non-pulse behavioral events for later UTC conversion.
        """
        from .events import (
            parse_medpc_file,
            extract_medpc_events,
            parse_csv_events,
            extract_irig_pulses,
            filter_non_pulse_events,
        )
        from .irig import decode_intervals_irig

        cfg = self._config
        fmt = cfg["format"]

        # Parse events based on format
        if fmt == "medpc":
            medpc_data = parse_medpc_file(cfg["path"])
            timestamps, codes = extract_medpc_events(
                medpc_data,
                array_name=cfg["array_name"],
                time_resolution=cfg["time_resolution"],
            )
        elif fmt in ("csv", "tsv"):
            timestamps, codes = parse_csv_events(
                cfg["path"],
                time_column=cfg["time_column"],
                event_column=cfg["event_column"],
                delimiter=cfg["delimiter"],
                has_header=cfg["has_header"],
                time_scale=cfg["time_scale"],
            )
        else:
            raise ValueError(f"Unknown event format: {fmt!r}")

        # Extract IRIG pulses
        onsets, offsets = extract_irig_pulses(
            timestamps, codes,
            cfg["pulse_high_code"],
            cfg["pulse_low_code"],
        )

        # Store behavioral (non-pulse) events for later UTC conversion
        self._behavioral_timestamps, self._behavioral_codes = (
            filter_non_pulse_events(
                timestamps, codes,
                cfg["pulse_high_code"],
                cfg["pulse_low_code"],
            )
        )

        # Decode IRIG from extracted pulses
        return decode_intervals_irig(
            onsets, offsets,
            source_units="seconds",
            save=cfg["save"],
        )
