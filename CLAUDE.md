# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NeuroKairos: A GPS-disciplined IRIG-H timecode system for synchronizing neuroscience data streams (electrophysiology, cameras, behavioral data) to UTC time. Runs on Raspberry Pi 4B with a GPS timing receiver.

IRIG-H encodes UTC time as pulse-width modulated TTL signals: 60 bits/frame at 1 Hz, using 0.2s (binary 0), 0.5s (binary 1), and 0.8s (position marker P) pulse widths in BCD format.

## Build Commands

```bash
# Compile the C sender (runs on Raspberry Pi, requires root for /dev/mem)
cd sender && make

# Install as systemd service
./scripts/install.sh

# Uninstall systemd service
./scripts/uninstall.sh

# Run with default pins (BCM GPIO 11, inverted disabled)
./sender/irig_sender

# Run with custom pins
./sender/irig_sender -p 17 -n 27

# Run with custom LED warning threshold (blink when root dispersion > 2ms)
./sender/irig_sender -w 2.0

# Install Python package (editable/development mode)
pip install -e .

# Or install normally
pip install .
```

## Key Usage

```bash
# Extract IRIG timecodes from a DAT recording file (after pip install)
neurokairos-extract recording.dat -o output.npz

# Or run directly
python -m neurokairos.extract_from_dat recording.dat -o output.npz

# extract_from_dat.py options: -t/--threshold (default 2500), -c/--channel (default 32),
# --total-channels (default 40), --chunk-size (default 2500000)
```

## Architecture

Two-phase system: **generation** (on Raspberry Pi) and **decoding** (post-hoc on any machine).

### Generation
- `sender/irig_sender.c` — Production sender. Uses direct `/dev/mem` GPIO register access with hybrid sleep/busy-wait for nanosecond-level timing precision. Default output on BCM GPIO 11 (normal), inverted disabled. Both pins configurable via CLI flags (`-p`/`-n`). Polls chrony every ~60 seconds and encodes sync status (stratum, root dispersion) in unused IRIG-H frame bits 43-44 and 46-48. Controls the RPi ACT LED to indicate sync quality. Runs as systemd service at Nice -20.
- `neurokairos/irig_h_gpio.py:IrigHSender` — Python sender for testing only (uses pigpio daemon).

### Chrony Integration
- `scripts/install_chrony_server.sh` — Installs chrony + gpsd on the RPi with GPS. Configures PPS-disciplined stratum 1 NTP server.
- `scripts/install_chrony_client.sh` — Installs chrony as NTP client (no GPS). Supports custom server (`--server`).
- `scripts/test_chrony.sh` — Diagnostic script for checking chrony/gpsd status.

### Core Library
- `neurokairos/irig_h_gpio.py` — Shared encoding/decoding library. Contains BCD encode/decode, pulse classification (`identify_pulse_length`), frame detection (`decode_irig_bits`), and POSIX timestamp conversion (`irig_h_to_posix`). Both the C sender and Python extraction tools implement the same IRIG-H frame structure.

### Decoding/Extraction
- `neurokairos/extract_from_dat.py` — Main CLI tool. `IRIGExtractor` class reads binary DAT files in chunks, detects edges, classifies pulses, decodes frames, detects discontinuities, and outputs to compressed NPZ.
- `neurokairos/extract_from_camera_events.py` — Processes camera event CSVs looking for TimeP/TimeN pin events.
- `neurokairos/vendor/readSGLX.py` — Vendored SpikeGLX binary/metadata file reader (supports IMEC, NIDQ, OBX data types). Used upstream to extract IRIG channels from SpikeGLX recordings.

### Analysis
- `neurokairos/bin_analysis.py` — Analyzes bit-packed binary IRIG data with generator-based unpacking.
- `neurokairos/npz_analysis.py` — Analyzes NPZ output for sampling rate, IRIG-vs-PPS error/latency, and systematic offset detection.

## Key Constants (neurokairos/irig_h_gpio.py)

Pulse classification thresholds are defined relative to `SENDING_BIT_LENGTH` (1 second) and `DECODE_BIT_PERIOD` (1/30000s). When modifying thresholds, both `P_THRESHOLD`, `ONE_THRESHOLD`, and `ZERO_THRESHOLD` must stay consistent with the 0.2/0.5/0.8 pulse width ratios.

## Data Formats

NPZ output contains structured arrays with fields: `on_sample` (uint64), `off_sample` (uint64), `pulse_type` (int8: 0/1/2 for zero/one/P, -1 for error), `unix_time` (float64), `frame_id` (int32), `stratum` (int8: 1-4, -1 for unknown), `root_dispersion_bucket` (int8: 0-7, -1 for unknown).

## Sync Status Encoding (NeuroKairos Extension)

The C sender polls chrony every ~60 seconds and encodes sync quality in previously unused IRIG-H frame bits. Bits 43-44 carry a 2-bit stratum code (1→0, 2→1, 3→2, >=4→3). Bits 46-48 carry a 3-bit root dispersion bucket on a doubling scale from <0.25ms (0) to >=16ms (7). Bits 42 and 45 remain zero (reserved). Old recordings with all-zero status bits are ambiguous with stratum 1 / best dispersion. See `docs/irig-h-standard.md` for the full encoding table.

## Known Bug History

- **Day-of-year off-by-one (C sender, fixed in `6def02b`):** C's `tm_yday` is 0-indexed (0-365) but IRIG-H expects 1-indexed (1-366). The original C sender omitted the `+1`, causing every transmitted day to be one too low. Python was never affected (`timetuple().tm_yday` is already 1-indexed). Old branches `less-cpu` and `charlie-irig` still have the unfixed code.
- **`seconds + 1` in encoders is intentional:** Both the C and Python senders encode `seconds + 1` because frames are generated before the second boundary but start transmitting at the next second.
