"""SpikeGLX metadata reader and IRIG decoding entry point."""

import logging
from pathlib import Path
from typing import Union

import numpy as np

from .clock_table import ClockTable
from .ttl import auto_threshold, detect_edges, measure_pulse_widths
from .irig import build_clock_table

logger = logging.getLogger(__name__)


def read_meta(bin_path: Union[str, Path]) -> dict:
    """Read the SpikeGLX ``.meta`` file paired with a ``.bin`` file.

    Parameters
    ----------
    bin_path : str or Path
        Path to the ``.bin`` file.  The ``.meta`` file is expected to be
        in the same directory with the same stem.

    Returns
    -------
    dict
        Metadata key-value pairs (values are strings).

    Raises
    ------
    FileNotFoundError
        If the ``.meta`` file does not exist.
    """
    bin_path = Path(bin_path)
    meta_path = bin_path.with_suffix(".meta")
    if not meta_path.exists():
        raise FileNotFoundError(f"No .meta file found at {meta_path}")

    meta = {}
    with meta_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            # Strip leading '~' (SpikeGLX convention for array-valued keys)
            if key.startswith("~"):
                key = key[1:]
            meta[key] = value
    return meta


def get_n_channels(meta: dict) -> int:
    """Return total number of saved channels from a SpikeGLX meta dict."""
    return int(meta["nSavedChans"])


def get_sample_rate(meta: dict) -> float:
    """Return sample rate from a SpikeGLX meta dict.

    Handles imec, nidq, and obx stream types.
    """
    stream_type = meta["typeThis"]
    if stream_type == "imec":
        return float(meta["imSampRate"])
    elif stream_type == "nidq":
        return float(meta["niSampRate"])
    elif stream_type == "obx":
        return float(meta["obSampRate"])
    else:
        raise ValueError(f"Unknown SpikeGLX stream type: {stream_type!r}")


def get_irig_channel(meta: dict, irig_channel: Union[int, str]) -> int:
    """Resolve an IRIG channel specification to a zero-based channel index.

    Parameters
    ----------
    meta : dict
        SpikeGLX metadata.
    irig_channel : int or str
        If ``int``, used directly as a zero-based channel index.
        If ``"sync"``, resolves to the last channel (SY word for imec,
        last DW word for nidq).

    Returns
    -------
    int
        Zero-based channel index.
    """
    if isinstance(irig_channel, int):
        return irig_channel

    if irig_channel == "sync":
        # Last channel in the saved file
        return int(meta["nSavedChans"]) - 1

    raise ValueError(
        f"irig_channel must be an int or 'sync', got {irig_channel!r}"
    )


def decode_sglx_irig(
    bin_path: Union[str, Path],
    irig_channel: Union[int, str],
    save: bool = True,
) -> ClockTable:
    """Decode IRIG timecodes from a SpikeGLX ``.bin`` file.

    Reads the paired ``.meta`` file to determine channel count, then runs
    the standard TTL detection and IRIG decoding pipeline.

    Parameters
    ----------
    bin_path : str or Path
        Path to the SpikeGLX ``.bin`` file.
    irig_channel : int or str
        Zero-based channel index, or ``"sync"`` for the last channel.
    save : bool
        If True (default), save the ClockTable to
        ``<bin_path>.clocktable.npz`` alongside the data file.

    Returns
    -------
    ClockTable
        Sparse sample-index-to-UTC-time mapping with one entry per IRIG
        pulse (~1 Hz).
    """
    bin_path = Path(bin_path)
    meta = read_meta(bin_path)
    n_channels = get_n_channels(meta)
    ch_idx = get_irig_channel(meta, irig_channel)

    raw = np.memmap(bin_path, dtype=np.int16, mode="r")
    data = raw.reshape(-1, n_channels)
    irig_signal = data[:, ch_idx]

    threshold = auto_threshold(irig_signal)
    rising, falling = detect_edges(irig_signal, threshold)
    onsets, widths = measure_pulse_widths(rising, falling)

    ct = build_clock_table(onsets, widths)

    ct.metadata["source_file"] = bin_path.name
    ct.metadata["source_path"] = str(bin_path.resolve())
    ct.metadata["n_channels"] = n_channels
    ct.metadata["irig_channel"] = ch_idx
    ct.metadata["stream_type"] = meta.get("typeThis", "unknown")

    if save:
        ct_path = bin_path.parent / (bin_path.name + ".clocktable.npz")
        ct.save(ct_path)
        logger.info("Saved clock table to %s", ct_path)

    return ct
