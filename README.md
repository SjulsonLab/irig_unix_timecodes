# NeuroKairos: GPS-based synchronization for neuroscience experiments

A universal, open-source timing synchronization solution for neuroscience experiments. NeuroKairos continuously obtains the earth's Coordinated Universal Time (UTC) from the atomic clocks inside GPS satellites and encodes it a sequence of TTL pulses known as an IRIG-H timecode. Any instrument that can record this timecode through TTL pulses or a blinking LED can therefore continuously timestamp its simultaneously-recorded data with objective UTC time. This enables virtually any device to synchronize to UTC time, which provides a common reference for aligning different data streams with each other.

![NeuroKairos System Architecture](docs/system_architecture.jpg)

*NeuroKairos system architecture: a GPS-disciplined Raspberry Pi serves as both a stratum-1 NTP server for network-connected devices and an IRIG-H timecode generator for direct hardware timing signals.*

## The Problem

Modern neuroscience experiments require precise synchronization of multiple data streams — electrophysiology, cameras, behavioral apparatus — but each device runs on its own internal clock. These clocks drift apart during experiments, and timing errors as small as one millisecond can reduce neural decoding accuracy from perfect performance to random chance. No universal solution exists: laboratories are forced to build unreliable custom systems or purchase expensive commercial timing hardware with restrictive compatibility requirements.

## The Solution

NeuroKairos combines two mature technologies — GPS atomic clock disciplining and IRIG timecodes — to create a universal synchronization system on consumer hardware. The system works with **any recording device** capable of sampling voltage pulses or imaging LEDs, requiring no modifications to existing equipment.

**Dual-mode timing distribution:**
1. **Network Time Protocol (NTP)**: The GPS-disciplined Raspberry Pi serves as a stratum-1 NTP server for all network-connected devices
2. **IRIG-H hardware signals**: GPIO pins output pulse-width modulated timecodes that can be recorded directly alongside experimental data

Validation against 30,000 Hz electrophysiology demonstrated an average timing accuracy of **33 microseconds** with 99.44% of events at sub-sample precision and zero decoding errors over 25 hours of continuous recording.

## What is IRIG-H?

IRIG-H is one of the simplest formats in the IRIG timecode family (IRIG Standard 200), transmitting one pulse per second with 60 pulses per frame. Each pulse encodes a binary value through its width: 0.2s for binary 0, 0.5s for binary 1, and 0.8s for position markers. Frames encode minutes, hours, day of year, and year in Binary Coded Decimal (BCD) format.

For detailed specifications including the complete bit map and encoding tables, see the [IRIG-H Standard Reference](docs/irig-h-standard.md).

## Hardware Requirements

- **Raspberry Pi 4 Model B**
- **Waveshare NEO-M8T GNSS Timing Hat** or compatible GPS timing receiver with PPS output
- **GPS antenna** with direct sky visibility
- **GPIO connections**: Default BCM GPIO 11 for IRIG output (configurable via `-p`/`-n` flags)

## Installation

### 1. Install Python Package

```bash
pip install .

# Or for development (editable install)
pip install -e .
```

This installs the `neurokairos` package and the `neurokairos-extract` command-line tool.

### 2. Configure GPS Clock Disciplining

Install and configure chrony to use the GPS receiver as timing source:

```bash
sudo apt install chrony
```

Edit `/etc/chrony/chrony.conf` to add GPS as stratum-0 source with appropriate refid and trust parameters.

### 3. Compile C Sender

```bash
cd sender
make
```

### 4. Install as System Service

```bash
# Install with default pins (BCM GPIO 11, inverted disabled)
./scripts/install.sh

# Install with custom pins
./scripts/install.sh -p 17 -n 27

# Install with custom LED warning threshold (ms)
./scripts/install.sh -p 17 -w 2.0

# See all install options
./scripts/install.sh -h
```

The install script compiles the sender, copies it to `/usr/local/bin/`, generates the systemd service file with your pin configuration baked in, and starts the service. To change pins later, just re-run `install.sh` with the new flags — it will restart the service automatically.

The service runs with Nice -20 priority and SCHED_FIFO real-time scheduling for low-latency timing.

### 5. Pin Configuration (Optional)

The sender defaults to BCM GPIO 11 (normal output) with inverted output disabled. When running manually (not as a service):

```bash
# Run with specific pins
./sender/irig_sender -p 17 -n 27

# Run with only inverted output
./sender/irig_sender -p -1 -n 22

# Show all options
./sender/irig_sender -h
```

**Note**: BCM GPIO pins 0, 1 (I2C EEPROM) and 14, 15 (UART/GPS serial) are blocked. BCM GPIO 4 (GPS PPS) triggers a warning but is allowed.

## Usage

### Generation Workflow

1. **Clock synchronization**: System clock is synchronized to GPS via chrony
2. **Continuous generation**: `irig_sender` runs as a systemd service, generating IRIG-H frames
3. **Signal output**: GPIO pin(s) output pulse-width modulated signals encoding UTC time
4. **Data recording**: Recording equipment samples GPIO signals alongside experimental data

### Decoding Workflow

1. **Extract IRIG channel**: Use `neurokairos-extract` for electrophysiology or `extract_from_camera_events.py` for camera data
2. **Pulse detection**: Pulses are detected, classified (0/1/P), and grouped into 60-bit frames
3. **Frame decoding**: BCD fields are decoded to extract time components
4. **Timestamp conversion**: Decoded frames are converted to POSIX timestamps
5. **Analysis**: Use `npz_analysis.py` or `bin_analysis.py` to verify accuracy

### Example: Extract from DAT File

```bash
neurokairos-extract recording.dat output.npz
```

Optional parameters:
- `--threshold`: Signal threshold for pulse detection
- `--channel`: DAT file channel containing IRIG signal
- `--sample-rate`: Override sampling rate (otherwise estimated)

## Key Components

### IRIG Sender (C)

**File**: `sender/irig_sender.c`

Low-latency C program generating IRIG-H timecodes via direct GPIO register access (`/dev/mem`). Uses a hybrid sleep/busy-wait approach: sleeps until ~1ms before the second boundary, then enters a busy-wait loop polling the system clock to detect the exact transition and immediately set GPIO HIGH. Runs as a systemd service with SCHED_FIFO real-time scheduling.

### IRIG Decoder Library (Python)

**File**: `neurokairos/irig_h_gpio.py`

BCD encoding/decoding, pulse length classification, frame detection and synchronization, and POSIX timestamp conversion. Also includes a Python-based sender class (`IrigHSender`) using pigpio for testing.

### Data Extraction Tools

**`neurokairos/extract_from_dat.py`** — Extracts IRIG pulses from binary DAT files (electrophysiology recordings). Processes data in chunks, detects edges, classifies pulses, decodes frames, and outputs to compressed NPZ format.

**`neurokairos/extract_from_camera_events.py`** — Extracts IRIG timecodes from camera event CSV files.

### Analysis Scripts

**`neurokairos/bin_analysis.py`** — Analyzes bit-packed binary IRIG data with generator-based unpacking and PPS error detection.

**`neurokairos/npz_analysis.py`** — Analyzes NPZ output for sampling rate, IRIG-vs-PPS error/latency, and systematic offset detection.

### SpikeGLX Integration

**File**: `neurokairos/vendor/readSGLX.py`

Vendored SpikeGLX binary/metadata reader supporting IMEC, NIDQ, and OBX data types with memory-mapped file access.

## Performance

Validated against 30,000 Hz electrophysiology recording over 25 hours:

| Metric | Value |
|--------|-------|
| Average timing delay (IRIG vs PPS) | 33 microseconds |
| Events at primary peak (33 us) | 99.44% |
| Events at secondary peak (67 us) | 0.56% |
| Rare excursions (0.7-4 ms) | 0.005% |
| Pulse duration standard deviation | < 1 ms |
| Decoding errors | 0 |

## Data Formats

### NPZ Output

Structured NumPy arrays with fields: `on_sample` (uint64), `off_sample` (uint64), `pulse_type` (int8: 0/1/2 for zero/one/P, -1 for error), `unix_time` (float64), `frame_id` (int32).

### CSV Output

Decoded timestamps with columns: `frame_number`, `unix_timestamp`, `datetime`, `samples_since_last`.

## System Service Management

```bash
# Check status
sudo systemctl status irig-sender.service

# View logs
sudo journalctl -u irig-sender.service -f

# Stop/start
sudo systemctl stop irig-sender.service
sudo systemctl start irig-sender.service

# Restart after config changes
sudo systemctl daemon-reload
sudo systemctl restart irig-sender.service
```

## File Structure

```
irig_unix_timecodes/
├── sender/                              # C sender (Raspberry Pi hardware)
│   ├── irig_sender.c
│   └── Makefile
├── neurokairos/                         # Python package
│   ├── __init__.py
│   ├── irig_h_gpio.py                  # Core IRIG-H encoding/decoding library
│   ├── extract_from_dat.py             # Main CLI tool
│   ├── extract_from_camera_events.py   # Camera event extraction
│   ├── bin_analysis.py                 # Binary data analysis
│   ├── npz_analysis.py                 # NPZ data analysis
│   └── vendor/
│       ├── __init__.py
│       └── readSGLX.py                 # Vendored SpikeGLX reader
├── tests/
│   └── test_irig_h.py
├── systemd/
│   └── irig-sender.service
├── scripts/
│   ├── install.sh
│   └── uninstall.sh
├── docs/
│   ├── system_architecture.jpg
│   ├── irig_h_frame_structure.jpg
│   └── irig-h-standard.md
├── pyproject.toml
├── LICENSE
├── README.md
└── CLAUDE.md
```

## Citation

If you use NeuroKairos in your research, please cite:

> Kerr, C. (2025). NeuroKairos: A Universal GPS Satellite-Based Solution to Synchronization in Neuroscience Experiments.

## References

- Range Commanders Council, Telecommunications and Timing Group. (2016). *IRIG Serial Time Code Formats*. RCC Standard 200-16. White Sands Missile Range, New Mexico.
- [IRIG timecode (Wikipedia)](https://en.wikipedia.org/wiki/IRIG_timecode)
- [Chrony documentation](https://chrony.tuxfamily.org/)
- [NIST Time and Frequency Division](https://www.nist.gov/pml/time-and-frequency-division/)

## License

MIT License. See [LICENSE](LICENSE) for details.
