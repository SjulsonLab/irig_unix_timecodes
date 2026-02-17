# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NeuroKairos: A GPS-disciplined IRIG-H timecode system for synchronizing neuroscience data streams (electrophysiology, cameras, behavioral data) to UTC time. Runs on Raspberry Pi 4B with a GPS timing receiver.

IRIG-H encodes UTC time as pulse-width modulated TTL signals: 60 bits/frame at 1 Hz, using 0.2s (binary 0), 0.5s (binary 1), and 0.8s (position marker P) pulse widths in BCD format.

## Build Commands

```bash
# Compile the C sender (runs on Raspberry Pi, requires root for /dev/mem)
cd sender && make

# Install as systemd service (default pins)
./scripts/install.sh

# Install with custom pins
./scripts/install.sh -p 17 -n 27

# Install with custom LED warning threshold
./scripts/install.sh -p 17 -w 2.0

# Uninstall systemd service
./scripts/uninstall.sh

# Run with default pins (BCM GPIO 11, inverted disabled)
./sender/irig_sender

# Run with custom pins
./sender/irig_sender -p 17 -n 27

# Run with custom LED warning threshold (blink when root dispersion > 2ms)
./sender/irig_sender -w 2.0

# Install Python package (editable/development mode)
pip install -e ".[test]"

# Run tests
pytest tests/ -v
```

## Architecture

Two-phase system: **generation** (on Raspberry Pi) and **decoding** (post-hoc on any machine).

### Generation (C sender)
- `sender/irig_sender.c` — Production sender. Uses direct `/dev/mem` GPIO register access with hybrid sleep/busy-wait for nanosecond-level timing precision. Default output on BCM GPIO 11 (normal), inverted disabled. Both pins configurable via CLI flags (`-p`/`-n`). Polls chrony every ~60 seconds and encodes sync status (stratum, root dispersion) in unused IRIG-H frame bits 43-44 and 46-48. Controls the RPi ACT LED to indicate sync quality. Runs as systemd service at Nice -20.

### Chrony Integration
- `scripts/install_chrony_server.sh` — Installs chrony + gpsd on the RPi with GPS. Configures PPS-disciplined stratum 1 NTP server.
- `scripts/install_chrony_client.sh` — Installs chrony as NTP client (no GPS). Supports custom server (`--server`).
- `scripts/test_chrony.sh` — Diagnostic script for checking chrony/gpsd status.

### Core Python Library (`neurokairos/`)
- `ttl.py` — Signal processing: `auto_threshold` (Otsu's method), `detect_edges`, `measure_pulse_widths`. NumPy only, no dependencies on other modules.
- `clock_table.py` — `ClockTable` dataclass: sparse time mapping (source <-> reference) with bidirectional `np.interp`, save/load to NPZ, JSON-serializable metadata.
- `irig.py` — Complete IRIG-H decoder pipeline: pulse classification, BCD encode/decode, frame decoding (complete + partial), robust handling of missing/extra pulses and concatenated files, `build_clock_table` orchestrator, plus top-level entry points `decode_dat_irig` and `decode_intervals_irig`.
- `sglx.py` — SpikeGLX `.meta` reader + `decode_sglx_irig` entry point.
- `video.py` — Video LED extraction + `decode_video_irig` entry point. OpenCV (`cv2`) is an optional dependency.

### Public API (`neurokairos/__init__.py`)
- `ClockTable` — sparse time mapping
- `bcd_encode`, `bcd_decode` — BCD encoding/decoding
- `decode_dat_irig` — decode from interleaved int16 `.dat` files
- `decode_sglx_irig` — decode from SpikeGLX `.bin` + `.meta`
- `decode_video_irig` — decode from video files with IRIG LED
- `decode_intervals_irig` — decode from pre-extracted pulse intervals
- BCD weight constants: `SECONDS_WEIGHTS`, `MINUTES_WEIGHTS`, `HOURS_WEIGHTS`, `DAY_OF_YEAR_WEIGHTS`, `DECISECONDS_WEIGHTS`, `YEARS_WEIGHTS`

## Key Constants (neurokairos/irig.py)

Pulse-width fractions of 1 second: `PULSE_FRAC_ZERO` (0.2), `PULSE_FRAC_ONE` (0.5), `PULSE_FRAC_MARKER` (0.8). Classification boundaries are midpoints: 0.35 (0/1), 0.65 (1/marker). Min valid: 0.1, max: 0.95.

## Sync Status Encoding (NeuroKairos Extension)

The C sender polls chrony every ~60 seconds and encodes sync quality in previously unused IRIG-H frame bits. Bits 43-44 carry a 2-bit stratum code (1->0, 2->1, 3->2, >=4->3). Bits 46-48 carry a 3-bit root dispersion bucket on a doubling scale from <0.25ms (0) to >=16ms (7). Bits 42 and 45 remain zero (reserved). Old recordings with all-zero status bits are ambiguous with stratum 1 / best dispersion. See `docs/irig-h-standard.md` for the full encoding table.

## Known Bug History

- **Day-of-year off-by-one (C sender, fixed in `6def02b`):** C's `tm_yday` is 0-indexed (0-365) but IRIG-H expects 1-indexed (1-366). The original C sender omitted the `+1`, causing every transmitted day to be one too low. Python was never affected (`timetuple().tm_yday` is already 1-indexed). Old branches `less-cpu` and `charlie-irig` still have the unfixed code.
- **Frames must start on minute boundaries:** The C sender waits for the next :00 second before transmitting its first frame. Each frame is 60 bits (60 seconds), so subsequent frames naturally align to minute boundaries. The BCD seconds field is always 0.
