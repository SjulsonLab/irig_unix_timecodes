"""Event log parsing for MedPC files and CSV/TSV, with IRIG pulse extraction.

Supports two input formats:
  - MedPC files with TIME.CODE encoded arrays
  - CSV/TSV files with timestamp and event code columns

Provides utilities to separate IRIG pulse events from behavioral events,
convert behavioral event timestamps to UTC via a ClockTable, and write
the results to CSV.
"""

import csv
import datetime as _dt
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .clock_table import ClockTable

# Sentinel value that terminates MedPC data arrays
_MEDPC_SENTINEL = -987.987
_SENTINEL_TOLERANCE = 0.001


# -- Dataclasses ---------------------------------------------------------------

@dataclass
class MedPCHeader:
    """Header fields from a MedPC data file."""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    subject: Optional[str] = None
    experiment: Optional[str] = None
    group: Optional[str] = None
    box: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    msn: Optional[str] = None


@dataclass
class MedPCData:
    """Parsed MedPC file: header metadata + named arrays."""
    header: MedPCHeader
    arrays: Dict[str, np.ndarray] = field(default_factory=dict)


# -- MedPC parsing -------------------------------------------------------------

# Maps header line prefixes to MedPCHeader field names
_HEADER_MAP = {
    "Start Date": "start_date",
    "End Date": "end_date",
    "Subject": "subject",
    "Experiment": "experiment",
    "Group": "group",
    "Box": "box",
    "Start Time": "start_time",
    "End Time": "end_time",
    "MSN": "msn",
}

# Matches an array label line like "C:" (single uppercase letter followed by colon)
_ARRAY_LABEL_RE = re.compile(r"^([A-Z]):$")

# Matches a data line like "     0:    5.001   12.002  ..."
_DATA_LINE_RE = re.compile(r"^\s+\d+:")


def parse_medpc_file(path: Union[str, Path]) -> MedPCData:
    """Parse a MedPC data file into header metadata and named arrays.

    Parameters
    ----------
    path : str or Path
        Path to the MedPC text file.

    Returns
    -------
    MedPCData
        Parsed header and arrays. Sentinel values (-987.987) are excluded.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MedPC file not found: {path}")

    header = MedPCHeader()
    arrays: Dict[str, List[str]] = {}
    current_array: Optional[str] = None

    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")

            # Try to match a header field
            matched_header = False
            for prefix, attr in _HEADER_MAP.items():
                if line.startswith(prefix + ":"):
                    value = line[len(prefix) + 1:].strip()
                    setattr(header, attr, value)
                    matched_header = True
                    current_array = None
                    break

            if matched_header:
                continue

            # Try to match an array label (e.g. "C:")
            label_match = _ARRAY_LABEL_RE.match(line.strip())
            if label_match:
                current_array = label_match.group(1)
                arrays[current_array] = []
                continue

            # Try to match a data line within the current array
            if current_array is not None and _DATA_LINE_RE.match(line):
                # Extract the values after the "N:" prefix
                _, values_str = line.split(":", 1)
                # Keep as strings to preserve precision for TIME.CODE parsing
                tokens = values_str.split()
                arrays[current_array].extend(tokens)
                continue

    # Convert string arrays to numpy, excluding sentinel values
    result_arrays: Dict[str, np.ndarray] = {}
    for name, str_values in arrays.items():
        filtered = []
        for s in str_values:
            val = float(s)
            if abs(val - _MEDPC_SENTINEL) < _SENTINEL_TOLERANCE:
                break  # Sentinel terminates the array
            filtered.append(val)
        result_arrays[name] = np.array(filtered, dtype=np.float64)

    return MedPCData(header=header, arrays=result_arrays)


def extract_medpc_events(
    medpc_data: MedPCData,
    array_name: str = "C",
    time_resolution: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract timestamped events from a MedPC TIME.CODE array.

    Each value in the array encodes ``INTEGER_PART.CODE`` where:
    - ``integer_part * time_resolution`` → time in seconds
    - ``round(fractional_part * 1000)`` → event code (3-digit integer)

    Parsing is done from the string representation to avoid float precision
    issues.

    Parameters
    ----------
    medpc_data : MedPCData
        Parsed MedPC file.
    array_name : str
        Name of the array containing TIME.CODE values (default ``"C"``).
    time_resolution : float
        Seconds per integer time unit (default 0.01 = centiseconds).

    Returns
    -------
    timestamps : ndarray (float64)
        Event times in seconds.
    event_codes : ndarray (int64)
        Event codes (3-digit integers).
    """
    if array_name not in medpc_data.arrays:
        raise KeyError(f"Array '{array_name}' not found in MedPC data")

    # We need the raw string values for precision — re-read from the file
    # But we've already parsed to float. Instead, use the float values
    # and parse carefully via string formatting to recover integer/fractional parts.
    values = medpc_data.arrays[array_name]

    timestamps = []
    codes = []

    for val in values:
        # Convert float back to string with enough precision to recover
        # the original TIME.CODE format
        val_str = f"{val:.3f}"
        parts = val_str.split(".")
        integer_part = int(parts[0])
        frac_part = int(parts[1])

        t_seconds = integer_part * time_resolution
        timestamps.append(t_seconds)
        codes.append(frac_part)

    return (
        np.array(timestamps, dtype=np.float64),
        np.array(codes, dtype=np.int64),
    )


# -- CSV/TSV parsing -----------------------------------------------------------

def parse_csv_events(
    path: Union[str, Path],
    time_column: Union[int, str] = 0,
    event_column: Union[int, str] = 1,
    delimiter: Optional[str] = None,
    has_header: bool = True,
    time_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Parse event timestamps and codes from a CSV or TSV file.

    Parameters
    ----------
    path : str or Path
        Path to the CSV/TSV file.
    time_column : int or str
        Column for timestamps — by index (int) or header name (str).
    event_column : int or str
        Column for event codes — by index (int) or header name (str).
    delimiter : str, optional
        Column delimiter. Auto-detects comma vs tab if None.
    has_header : bool
        Whether the file has a header row (default True).
    time_scale : float
        Multiply timestamps by this factor (default 1.0). Useful for
        unit conversion (e.g. 1000.0 to convert seconds → milliseconds).

    Returns
    -------
    timestamps : ndarray (float64)
        Event times (scaled by time_scale).
    event_codes : ndarray (int64)
        Integer event codes.
    """
    path = Path(path)

    # Auto-detect delimiter from first line
    if delimiter is None:
        with open(path) as f:
            first_line = f.readline()
        if "\t" in first_line:
            delimiter = "\t"
        else:
            delimiter = ","

    timestamps = []
    codes = []

    with open(path, newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)

        if has_header:
            header_row = next(reader)
            # Resolve column names to indices
            time_idx = _resolve_column(time_column, header_row)
            event_idx = _resolve_column(event_column, header_row)
        else:
            time_idx = time_column if isinstance(time_column, int) else 0
            event_idx = event_column if isinstance(event_column, int) else 1

        for row in reader:
            if not row:
                continue
            t = float(row[time_idx]) * time_scale
            code = int(float(row[event_idx]))
            timestamps.append(t)
            codes.append(code)

    return (
        np.array(timestamps, dtype=np.float64),
        np.array(codes, dtype=np.int64),
    )


def _resolve_column(col, header_row):
    """Resolve a column specifier (int index or str name) to an int index."""
    if isinstance(col, int):
        return col
    try:
        return header_row.index(col)
    except ValueError:
        raise ValueError(
            f"Column '{col}' not found in header: {header_row}"
        )


# -- Pulse extraction ----------------------------------------------------------

def extract_irig_pulses(
    timestamps: np.ndarray,
    event_codes: np.ndarray,
    pulse_high_code: int,
    pulse_low_code: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract IRIG pulse onsets and offsets from event data.

    Pairs each HIGH event with the next LOW event to form pulses.

    Parameters
    ----------
    timestamps : ndarray
        Event timestamps (seconds).
    event_codes : ndarray
        Event codes.
    pulse_high_code : int
        Code indicating pulse onset (HIGH).
    pulse_low_code : int
        Code indicating pulse offset (LOW).

    Returns
    -------
    onsets : ndarray (float64)
        Pulse onset times.
    offsets : ndarray (float64)
        Pulse offset times.
    """
    onsets = []
    offsets = []

    # Find all HIGH and LOW events
    high_mask = event_codes == pulse_high_code
    low_mask = event_codes == pulse_low_code

    high_times = timestamps[high_mask]
    low_times = timestamps[low_mask]

    # Pair each HIGH with the next LOW
    low_idx = 0
    for h_time in high_times:
        # Advance low_idx to the first LOW after this HIGH
        while low_idx < len(low_times) and low_times[low_idx] <= h_time:
            low_idx += 1
        if low_idx < len(low_times):
            onsets.append(h_time)
            offsets.append(low_times[low_idx])
            low_idx += 1

    return (
        np.array(onsets, dtype=np.float64),
        np.array(offsets, dtype=np.float64),
    )


def filter_non_pulse_events(
    timestamps: np.ndarray,
    event_codes: np.ndarray,
    pulse_high_code: int,
    pulse_low_code: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return only behavioral events, excluding IRIG pulse HIGH/LOW.

    Parameters
    ----------
    timestamps : ndarray
        All event timestamps.
    event_codes : ndarray
        All event codes.
    pulse_high_code : int
        Code for pulse HIGH events to exclude.
    pulse_low_code : int
        Code for pulse LOW events to exclude.

    Returns
    -------
    timestamps : ndarray
        Behavioral event timestamps only.
    event_codes : ndarray
        Behavioral event codes only.
    """
    mask = (event_codes != pulse_high_code) & (event_codes != pulse_low_code)
    return timestamps[mask], event_codes[mask]


# -- UTC conversion ------------------------------------------------------------

def convert_events_to_utc(
    clock_table: ClockTable,
    event_timestamps: np.ndarray,
    event_codes: np.ndarray,
    event_names: Optional[Dict[int, str]] = None,
) -> List[dict]:
    """Convert local event timestamps to UTC using a ClockTable.

    Parameters
    ----------
    clock_table : ClockTable
        Mapping from local time to UTC (from IRIG decoding).
    event_timestamps : ndarray
        Local event timestamps (same domain as clock_table.source).
    event_codes : ndarray
        Integer event codes.
    event_names : dict, optional
        Mapping from event code → human-readable name.

    Returns
    -------
    list of dict
        Each dict has keys: ``utc_timestamp``, ``local_timestamp``,
        ``event_code``, ``event_name``.
    """
    utc_timestamps = clock_table.source_to_reference(event_timestamps)

    events = []
    for i in range(len(event_timestamps)):
        code = int(event_codes[i])
        name = event_names.get(code, "") if event_names else ""
        events.append({
            "utc_timestamp": float(utc_timestamps[i]) if np.ndim(utc_timestamps) > 0 else float(utc_timestamps),
            "local_timestamp": float(event_timestamps[i]),
            "event_code": code,
            "event_name": name,
        })

    return events


def write_events_csv(
    events: List[dict],
    path: Union[str, Path],
) -> None:
    """Write converted events to a CSV file.

    Columns: utc_timestamp, utc_datetime, local_timestamp, event_code, event_name.

    Parameters
    ----------
    events : list of dict
        Output from ``convert_events_to_utc``.
    path : str or Path
        Output CSV file path.
    """
    path = Path(path)
    fieldnames = [
        "utc_timestamp",
        "utc_datetime",
        "local_timestamp",
        "event_code",
        "event_name",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ev in events:
            utc_dt = _dt.datetime.fromtimestamp(
                ev["utc_timestamp"], tz=_dt.timezone.utc
            )
            row = {
                "utc_timestamp": ev["utc_timestamp"],
                "utc_datetime": utc_dt.isoformat(),
                "local_timestamp": ev["local_timestamp"],
                "event_code": ev["event_code"],
                "event_name": ev.get("event_name", ""),
            }
            writer.writerow(row)
