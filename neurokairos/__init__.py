"""NeuroKairos: GPS-disciplined IRIG-H timecode system for neuroscience."""

__version__ = "0.2.0"

from .clock_table import ClockTable
from .decoder import IRIGDecoder
from .events import (
    parse_medpc_file,
    parse_csv_events,
    extract_irig_pulses,
    convert_events_to_utc,
)
from .irig import (
    bcd_encode, bcd_decode,
    decode_dat_irig, decode_intervals_irig,
    SECONDS_WEIGHTS, MINUTES_WEIGHTS, HOURS_WEIGHTS,
    DAY_OF_YEAR_WEIGHTS, DECISECONDS_WEIGHTS, YEARS_WEIGHTS,
)
from .sglx import decode_sglx_irig
from .video import decode_video_irig
