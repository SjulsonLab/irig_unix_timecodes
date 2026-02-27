import datetime as _dt
import json as _json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np

_EXTRAP_LIMIT_S = 1.5


@dataclass
class ClockTable:
    """Sparse mapping between source domain (e.g. sample indices) and reference
    domain (e.g. UTC unix timestamps).

    Both arrays must be float64, monotonically increasing, and the same length
    (minimum 2 entries).  Interpolation is done via ``np.interp``.
    """

    source: np.ndarray
    reference: np.ndarray
    nominal_rate: float
    source_units: Optional[str] = None
    metadata: Optional[dict] = field(default=None, repr=False)

    def __post_init__(self):
        self.source = np.asarray(self.source, dtype=np.float64)
        self.reference = np.asarray(self.reference, dtype=np.float64)
        if len(self.source) < 2:
            raise ValueError("ClockTable requires at least 2 entries")
        if len(self.source) != len(self.reference):
            raise ValueError("source and reference must have the same length")
        if not np.all(np.diff(self.source) > 0):
            raise ValueError("source must be monotonically increasing")
        if not np.all(np.diff(self.reference) > 0):
            raise ValueError("reference must be monotonically increasing")
        if self.metadata is not None and not isinstance(self.metadata, dict):
            raise TypeError(
                f"metadata must be a dict or None, got {type(self.metadata).__name__}"
            )

    def source_to_reference(self, values) -> np.ndarray:
        """Convert source-domain values to reference-domain via interpolation.

        Linearly extrapolates up to 1.5 s beyond the ClockTable boundaries
        (one IRIG-H pulse interval).  Beyond that, clamps to the boundary
        value and issues a warning.
        """
        values_arr = np.asarray(values, dtype=np.float64)
        scalar = values_arr.ndim == 0
        v = np.atleast_1d(values_arr)
        result = np.interp(v, self.source, self.reference)

        below = v < self.source[0]
        if np.any(below):
            slope = (self.reference[1] - self.reference[0]) / (
                self.source[1] - self.source[0]
            )
            extrap = self.reference[0] + (v[below] - self.source[0]) * slope
            max_dist = float(np.max(self.reference[0] - extrap))
            if max_dist <= _EXTRAP_LIMIT_S:
                result[below] = extrap
            else:
                warnings.warn(
                    f"source_to_reference: extrapolating {max_dist:.1f} s before "
                    f"the first ClockTable entry exceeds the {_EXTRAP_LIMIT_S} s "
                    f"limit. Clamping to boundary value.",
                    stacklevel=2,
                )

        above = v > self.source[-1]
        if np.any(above):
            slope = (self.reference[-1] - self.reference[-2]) / (
                self.source[-1] - self.source[-2]
            )
            extrap = self.reference[-1] + (v[above] - self.source[-1]) * slope
            max_dist = float(np.max(extrap - self.reference[-1]))
            if max_dist <= _EXTRAP_LIMIT_S:
                result[above] = extrap
            else:
                warnings.warn(
                    f"source_to_reference: extrapolating {max_dist:.1f} s beyond "
                    f"the last ClockTable entry exceeds the {_EXTRAP_LIMIT_S} s "
                    f"limit. Clamping to boundary value.",
                    stacklevel=2,
                )

        return result[0] if scalar else result

    def reference_to_source(self, values) -> np.ndarray:
        """Convert reference-domain values to source-domain via interpolation.

        Linearly extrapolates up to 1.5 s beyond the ClockTable boundaries
        (one IRIG-H pulse interval).  Beyond that, clamps to the boundary
        value and issues a warning.
        """
        values_arr = np.asarray(values, dtype=np.float64)
        scalar = values_arr.ndim == 0
        v = np.atleast_1d(values_arr)
        result = np.interp(v, self.reference, self.source)

        below = v < self.reference[0]
        if np.any(below):
            slope = (self.source[1] - self.source[0]) / (
                self.reference[1] - self.reference[0]
            )
            extrap = self.source[0] + (v[below] - self.reference[0]) * slope
            max_dist = float(np.max(self.reference[0] - v[below]))
            if max_dist <= _EXTRAP_LIMIT_S:
                result[below] = extrap
            else:
                warnings.warn(
                    f"reference_to_source: extrapolating {max_dist:.1f} s before "
                    f"the first ClockTable entry exceeds the {_EXTRAP_LIMIT_S} s "
                    f"limit. Clamping to boundary value.",
                    stacklevel=2,
                )

        above = v > self.reference[-1]
        if np.any(above):
            slope = (self.source[-1] - self.source[-2]) / (
                self.reference[-1] - self.reference[-2]
            )
            extrap = self.source[-1] + (v[above] - self.reference[-1]) * slope
            max_dist = float(np.max(v[above] - self.reference[-1]))
            if max_dist <= _EXTRAP_LIMIT_S:
                result[above] = extrap
            else:
                warnings.warn(
                    f"reference_to_source: extrapolating {max_dist:.1f} s beyond "
                    f"the last ClockTable entry exceeds the {_EXTRAP_LIMIT_S} s "
                    f"limit. Clamping to boundary value.",
                    stacklevel=2,
                )

        return result[0] if scalar else result

    def save(self, path: Union[str, Path]) -> None:
        """Save to an NPZ file."""
        path = Path(path)
        arrays = dict(
            source=self.source,
            reference=self.reference,
            nominal_rate=np.array(self.nominal_rate),
        )
        if self.source_units is not None:
            arrays["source_units"] = np.array(self.source_units)
        if self.metadata is not None:
            try:
                arrays["_metadata"] = np.array(_json.dumps(self.metadata))
            except TypeError as e:
                raise TypeError(
                    f"metadata values must be JSON-serializable: {e}"
                ) from e
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ClockTable":
        """Load from an NPZ file."""
        path = Path(path)
        data = np.load(path)
        source_units = str(data["source_units"]) if "source_units" in data else None
        metadata = None
        if "_metadata" in data:
            metadata = _json.loads(str(data["_metadata"]))
        return cls(
            source=data["source"],
            reference=data["reference"],
            nominal_rate=float(data["nominal_rate"]),
            source_units=source_units,
            metadata=metadata,
        )

    def __repr__(self):
        units = f" ({self.source_units})" if self.source_units else ""
        lines = [
            f"ClockTable: {len(self.source)} entries{units}, "
            f"rate={self.nominal_rate:.1f}",
        ]

        # Recording start/stop from metadata, falling back to reference array
        meta = self.metadata or {}
        start_str = meta.get("recording_start")
        stop_str = meta.get("recording_stop")

        if start_str is None:
            try:
                dt = _dt.datetime.fromtimestamp(
                    self.reference[0], tz=_dt.timezone.utc
                )
                start_str = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            except (OSError, OverflowError, ValueError):
                pass
        if stop_str is None:
            try:
                dt = _dt.datetime.fromtimestamp(
                    self.reference[-1], tz=_dt.timezone.utc
                )
                stop_str = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            except (OSError, OverflowError, ValueError):
                pass

        if start_str and stop_str:
            lines.append(f"  recording: {start_str} → {stop_str}")
        elif start_str:
            lines.append(f"  recording start: {start_str}")

        # File info from metadata
        source_file = meta.get("source_file")
        source_path = meta.get("source_path")
        if source_file and source_path:
            lines.append(f"  file: {source_file} ({source_path})")
        elif source_file:
            lines.append(f"  file: {source_file}")

        lines.append(
            f"  source=[{self.source[0]:.1f}..{self.source[-1]:.1f}], "
            f"reference=[{self.reference[0]:.1f}..{self.reference[-1]:.1f}]"
        )

        return "\n".join(lines)
