"""NeuroKairos: GPS-disciplined IRIG-H timecode system for neuroscience."""

__version__ = "0.2.0"

from .clock_table import ClockTable
from .decoders.decoder import IRIGDecoder
from .decoders.events import (
    parse_medpc_file,
    parse_csv_events,
    extract_irig_pulses,
    convert_events_to_utc,
)
from .decoders.irig import (
    bcd_encode, bcd_decode,
    decode_dat_irig, decode_intervals_irig,
    ROOT_DISPERSION_UPPER_MS,
    SECONDS_WEIGHTS, MINUTES_WEIGHTS, HOURS_WEIGHTS,
    DAY_OF_YEAR_WEIGHTS, DECISECONDS_WEIGHTS, YEARS_WEIGHTS,
)
from .decoders.sglx import decode_sglx_irig
from .decoders.video import decode_video_irig
