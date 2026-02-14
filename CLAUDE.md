# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NeuroKairos: A GPS-disciplined IRIG-H timecode system for synchronizing neuroscience data streams (electrophysiology, cameras, behavioral data) to UTC time. Runs on Raspberry Pi 4B with a GPS timing receiver.

IRIG-H encodes UTC time as pulse-width modulated TTL signals: 60 bits/frame at 1 Hz, using 0.2s (binary 0), 0.5s (binary 1), and 0.8s (position marker P) pulse widths in BCD format.

## Build Commands

```bash
# Compile the C sender (runs on Raspberry Pi, requires root for /dev/mem)
gcc -o irig_sender irig_sender.c -lpthread -lm

# Install as systemd service
./setup.sh

# Uninstall systemd service
./desetup.sh

# Run with default pins (BCM GPIO 11, inverted disabled)
./irig_sender

# Run with custom pins
./irig_sender -p 17 -n 27

# Python dependencies
pip install numpy pandas
```

## Key Usage

```bash
# Extract IRIG timecodes from a DAT recording file
python extract_from_dat.py recording.dat -o output.npz

# extract_from_dat.py options: -t/--threshold (default 2500), -c/--channel (default 32),
# --total-channels (default 40), --chunk-size (default 2500000)
```

## Architecture

Two-phase system: **generation** (on Raspberry Pi) and **decoding** (post-hoc on any machine).

### Generation
- `irig_sender.c` — Production sender. Uses direct `/dev/mem` GPIO register access with hybrid sleep/busy-wait for nanosecond-level timing precision. Default output on BCM GPIO 11 (normal), inverted disabled. Both pins configurable via CLI flags (`-p`/`-n`). Runs as systemd service at Nice -20.
- `irig_h_gpio.py:IrigHSender` — Python sender for testing only (uses pigpio daemon).

### Core Library
- `irig_h_gpio.py` — Shared encoding/decoding library. Contains BCD encode/decode, pulse classification (`identify_pulse_length`), frame detection (`decode_irig_bits`), and POSIX timestamp conversion (`irig_h_to_posix`). Both the C sender and Python extraction tools implement the same IRIG-H frame structure.

### Decoding/Extraction
- `extract_from_dat.py` — Main CLI tool. `IRIGExtractor` class reads binary DAT files in chunks, detects edges, classifies pulses, decodes frames, detects discontinuities, and outputs to compressed NPZ.
- `extract_from_camera_events.py` — Processes camera event CSVs looking for TimeP/TimeN pin events.
- `readSGLX.py` — SpikeGLX binary/metadata file reader (supports IMEC, NIDQ, OBX data types). Used upstream to extract IRIG channels from SpikeGLX recordings.

### Analysis
- `bin_analysis.py` — Analyzes bit-packed binary IRIG data with generator-based unpacking.
- `npz_analysis.py` — Analyzes NPZ output for sampling rate, IRIG-vs-PPS error/latency, and systematic offset detection.

## Key Constants (irig_h_gpio.py)

Pulse classification thresholds are defined relative to `SENDING_BIT_LENGTH` (1 second) and `DECODE_BIT_PERIOD` (1/30000s). When modifying thresholds, both `P_THRESHOLD`, `ONE_THRESHOLD`, and `ZERO_THRESHOLD` must stay consistent with the 0.2/0.5/0.8 pulse width ratios.

## Data Formats

NPZ output contains structured arrays with fields: `on_sample` (uint64), `off_sample` (uint64), `pulse_type` (int8: 0/1/2 for zero/one/P, -1 for error), `unix_time` (float64), `frame_id` (int32).
