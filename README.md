# NeuroKairos: GPS-Disciplined IRIG-H Timecode System

A high-precision timing synchronization system for neuroscience experiments using GPS satellite atomic clock references and IRIG-H timecode standards. The system enables synchronization of multiple data streams (electrophysiology recordings, camera frames, behavioral data) to UTC time by encoding timestamps as pulse-width modulated TTL signals.

## Overview

NeuroKairos implements a GPS-disciplined IRIG-H timecode generation and decoding system on Raspberry Pi 4B. It provides dual-mode timing distribution through both Network Time Protocol (NTP) for network-connected devices and hardware IRIG-H signals for direct instrumentation. The system continuously generates timecodes that encode current UTC time as pulse-width modulated signals on GPIO pins, which can be recorded alongside experimental data for post-hoc synchronization.

## What is IRIG-H?

IRIG-H is a timecode format from the Inter-Range Instrumentation Group (IRIG) standards, originally developed in the 1950s for military missile testing and aerospace telemetry. The format transmits timing information as pulse-width modulated signals:

- **Frame structure**: 60 bits per frame, transmitted at 1 Hz (1 second per bit)
- **Pulse encoding**: Three pulse widths encode different values:
  - 0.2 seconds = binary 0
  - 0.5 seconds = binary 1
  - 0.8 seconds = position marker (P)
- **Time encoding**: Binary Coded Decimal (BCD) format encoding seconds, minutes, hours, day of year, and year
- **Position markers**: Located at bits 0, 9, 19, 29, 39, 49, 59 for frame synchronization

Each frame encodes:
- Seconds (bits 1-8)
- Minutes (bits 10-17)
- Hours (bits 20-26)
- Day of year (bits 30-41)
- Year, 2-digit (bits 50-58)
- Deciseconds (bits 45-48, always 0 in this implementation)

## System Architecture

### GPS-Disciplined Clock

The system uses a GPS receiver with Pulse Per Second (PPS) output to discipline the Raspberry Pi's system clock via chrony (Network Time Protocol daemon). The GPS receiver provides a stratum-0 timing reference directly phase-locked to GPS satellite atomic clocks, enabling the Raspberry Pi to operate as a stratum-1 NTP server.

### Dual-Mode Timing Distribution

1. **Network Time Protocol (NTP)**: The GPS-disciplined Raspberry Pi serves as an NTP server for network-connected devices (computers, data acquisition systems)
2. **IRIG-H Hardware Signals**: GPIO pins output IRIG-H timecodes that can be recorded by any system capable of sampling voltage pulses or imaging LEDs

### Signal Outputs

- **GPIO 17**: IRIG-H timecode output (normal polarity)
- **GPIO 27**: IRIG-H timecode output (inverted polarity)

## Key Components

### 1. IRIG Sender (C Implementation)

**File**: `irig_sender.c`

Low-latency C program that generates IRIG-H timecodes via direct GPIO register access:

- Uses `/dev/mem` for direct hardware register access (bypasses kernel overhead)
- Nanosecond-level timing control using `clock_gettime(CLOCK_REALTIME)`
- Hybrid sleep/busy-wait approach for precision timing
- 20 microsecond offset compensation for GPIO pin toggle latency
- Pre-calculates frame timing 200ms before transmission to minimize jitter
- Runs as systemd service with Nice -20 priority for scheduler preference
- Outputs on GPIO 17 (normal) and GPIO 27 (inverted)

**Timing precision features**:
- Sleeps until ~1ms before second boundary
- Enters busy-wait loop polling system clock at maximum frequency
- Detects exact moment of second transition
- Immediately sets GPIO HIGH to generate pulse rising edge

### 2. IRIG Decoder Library (Python)

**File**: `irig_h_gpio.py`

Complete IRIG-H encoding and decoding library:

- BCD encoding/decoding utilities
- Pulse length classification (0.2s, 0.5s, 0.8s)
- Frame detection and synchronization
- Converts IRIG frames to POSIX timestamps (Unix time)
- Python-based sender class (`IrigHSender`) using pigpio for testing

### 3. Data Extraction Tools

#### extract_from_dat.py

Extracts IRIG pulses from binary DAT files (electrophysiology recordings):

- Command-line interface: `extract_from_dat.py [input_file] [output_file]`
- Configurable parameters: signal threshold, channel selection, sampling rate
- Processes data in chunks for memory efficiency
- Edge detection (rising/falling) to identify pulse boundaries
- Pulse classification and frame decoding
- Sampling rate estimation
- Discontinuity detection (gaps, time jumps, dropped frames)
- Outputs to compressed NPZ format with structured arrays

#### extract_from_camera_events.py

Processes camera event CSV files:

- Extracts IRIG timecode pulse information from event timestamps
- Identifies TimeP/TimeN pin events (GPIO 17/27)
- Outputs to NPZ format for analysis

### 4. Analysis Scripts

#### bin_analysis.py

Analyzes bit-packed binary IRIG data:

- Memory-efficient generator-based unpacking
- Error detection by comparing IRIG to PPS (pulse-per-second) signals
- Pulse length analysis
- CSV export of decoded timecodes

#### npz_analysis.py

Analyzes NPZ-formatted IRIG data:

- Sampling rate calculation
- Error/latency analysis between IRIG and PPS
- Systematic offset detection and correction
- Decode validation

### 5. SpikeGLX Integration

**File**: `readSGLX.py`

Reads SpikeGLX binary and metadata files:

- Supports IMEC, NIDQ, and OBX data types
- Channel extraction and gain correction
- Digital line extraction for IRIG signals
- Memory-mapped file access for large datasets

## Hardware Requirements

- **Raspberry Pi 4 Model B** (required for pigpio compatibility with Python tools)
- **Waveshare NEO-M8T GNSS Timing Hat** or compatible GPS timing receiver
  - u-blox NEO-M8T GNSS receiver chip with dedicated timing mode
  - Precision PPS output phase-locked to GPS atomic clocks
- **GPS antenna** with direct sky visibility
- **GPIO connections**: Pins 17 and 27 for IRIG output

## Software Dependencies

### C Sender
- pthread library
- math library (`-lm`)
- `/dev/mem` access (requires root privileges)

### Python Tools
- numpy
- pandas
- pigpio (for Python sender only, not required for C sender)

### System Services
- chrony (NTP daemon for GPS disciplining)
- systemd (service management)

## Installation

### 1. Configure GPS Clock Disciplining

Install and configure chrony to use the GPS receiver as timing source:

```bash
sudo apt install chrony
```

Edit `/etc/chrony/chrony.conf` to add GPS as stratum-0 source with appropriate refid and trust parameters.

### 2. Compile C Sender

```bash
gcc -o irig_sender irig_sender.c -lpthread -lm
```

### 3. Install as System Service

```bash
# Copy binary to system location
sudo cp irig_sender /usr/local/bin/

# Copy systemd service file
sudo cp systemctl/irig-sender.service /etc/systemd/system/

# Enable and start service
sudo systemctl enable irig-sender.service
sudo systemctl start irig-sender.service
```

The service runs with Nice -20 priority for low-latency scheduling.

### 4. Install Python Dependencies

```bash
pip install numpy pandas
```

For Python-based sender (testing only):
```bash
sudo apt install pigpio
sudo systemctl enable pigpiod
sudo systemctl start pigpiod
```

## Usage

### Generation Workflow

1. **Clock Synchronization**: System clock is synchronized to GPS via chrony
2. **Continuous Generation**: `irig_sender.c` runs as systemd service, continuously generating IRIG-H frames
3. **Signal Output**: GPIO 17/27 output pulse-width modulated signals encoding UTC time
4. **Data Recording**: Recording equipment samples GPIO signals alongside experimental data

### Decoding Workflow

1. **Extract IRIG Channel**: Use `extract_from_dat.py` for electrophysiology or `extract_from_camera_events.py` for camera data
2. **Pulse Detection**: Pulses are detected, classified (0/1/P), and grouped into 60-bit frames
3. **Frame Decoding**: BCD fields are decoded to extract time components (year, day, hour, minute, second)
4. **Timestamp Conversion**: Decoded frames are converted to POSIX timestamps (seconds since Unix epoch)
5. **Analysis**: Use `npz_analysis.py` or `bin_analysis.py` to verify accuracy and detect errors

### Example: Extract from DAT File

```bash
python extract_from_dat.py recording.dat output.npz
```

Optional parameters:
- `--threshold`: Signal threshold for pulse detection
- `--channel`: DAT file channel containing IRIG signal
- `--sample-rate`: Override sampling rate (otherwise estimated)

## IRIG-H Frame Structure

Complete 60-bit frame specification:

| Bit Position | BCD Weight | Time Information | Bit Position | BCD Weight | Time Information |
|--------------|-----------|------------------|--------------|-----------|------------------|
| 00 | P | Reference Marker | 30 | 1 | Day of Year (1-366) |
| 01 | 1 | Seconds (00-59) | 31 | 2 | Day of Year |
| 02 | 2 | Seconds | 32 | 4 | Day of Year |
| 03 | 4 | Seconds | 33 | 8 | Day of Year |
| 04 | 8 | Seconds | 34 | 0 | Reserved |
| 05 | 0 | Unused | 35 | 10 | Day of Year |
| 06 | 10 | Seconds | 36 | 20 | Day of Year |
| 07 | 20 | Seconds | 37 | 40 | Day of Year |
| 08 | 40 | Seconds | 38 | 80 | Day of Year |
| 09 | P | Position Identifier | 39 | P | Position Identifier |
| 10 | 1 | Minutes (00-59) | 40 | 100 | Day of Year |
| 11 | 2 | Minutes | 41 | 200 | Day of Year |
| 12 | 4 | Minutes | 42 | 0 | Reserved |
| 13 | 8 | Minutes | 43 | 0 | Reserved |
| 14 | 0 | Reserved | 44 | 0 | Reserved |
| 15 | 10 | Minutes | 45 | 0.1 | Tenths Seconds (0.0-0.9) |
| 16 | 20 | Minutes | 46 | 0.2 | Tenths Seconds |
| 17 | 40 | Minutes | 47 | 0.4 | Tenths Seconds |
| 18 | 0 | Reserved | 48 | 0.8 | Tenths Seconds |
| 19 | P | Position Identifier | 49 | P | Position Identifier |
| 20 | 1 | Hours (0-23) | 50 | 1 | Year (00-99) |
| 21 | 2 | Hours | 51 | 2 | Year |
| 22 | 4 | Hours | 52 | 4 | Year |
| 23 | 8 | Hours | 53 | 8 | Year |
| 24 | 0 | Reserved | 54 | 0 | Reserved |
| 25 | 10 | Hours | 55 | 10 | Year |
| 26 | 20 | Hours | 56 | 20 | Year |
| 27 | 0 | Unused | 57 | 40 | Year |
| 28 | 0 | Unused | 58 | 80 | Year |
| 29 | P | Position Identifier | 59 | P | Position Identifier |

**Note**: Deciseconds (bits 45-48) are always 0 in this implementation. Position markers (P) provide frame synchronization.

## Data Formats

### NPZ Output Format

Structured NumPy arrays with fields:
- `on_sample`: Sample index of pulse rising edge
- `off_sample`: Sample index of pulse falling edge
- `pulse_type`: Classified pulse type (0, 1, or P)
- `unix_time`: Decoded POSIX timestamp
- `frame_id`: Frame number

### CSV Output Format

Decoded timestamps with columns:
- `frame_number`: Sequential frame identifier
- `unix_timestamp`: Seconds since Unix epoch (Jan 1, 1970)
- `datetime`: Human-readable date/time string
- `samples_since_last`: Sample count between frames

### Sender Log Files

Format: `irig_output_timestamps_[datetime].csv`

Columns:
- `encoded_time`: The time value encoded in the frame
- `sending_start`: System timestamp when transmission began

## Technical Implementation Details

### Timing Precision Mechanisms

1. **Direct Hardware Access**: `/dev/mem` access bypasses kernel GPIO drivers for faster pin toggling
2. **Hybrid Sleep/Busy-Wait**: Sleeps until ~1ms before target time, then busy-waits for precision
3. **Offset Compensation**: 20 microsecond `OFFSET_NS` parameter compensates for GPIO toggle latency
4. **Frame Pre-calculation**: Timing calculations performed 200ms before frame transmission
5. **Configurable Busy-Wait**: `BUSY_WAIT_SLEEP_NS` parameter (default: 0) for maximum precision

### BCD Encoding Scheme

Binary Coded Decimal uses weighted bit positions to encode decimal digits:

**Example**: 45 seconds
- Ones place: 5 = 1 + 4 → bits with weights 1 and 4 are HIGH
- Tens place: 4 = 4 → bit with weight 40 is HIGH
- Result: Bits 1, 3, and 7 are HIGH in the seconds field

### Pulse Classification

During decoding, pulse widths are classified by thresholds:
- **Short pulse** (< 0.35 × bit_length): Binary 0
- **Medium pulse** (0.35-0.65 × bit_length): Binary 1
- **Long pulse** (> 0.65 × bit_length): Position marker P

## Use Cases

1. **Electrophysiology**: Synchronize neural recordings (SpikeGLX, Intan, etc.) with behavioral events
2. **Multi-camera systems**: Align multiple camera streams via GPIO-triggered events
3. **Cross-system synchronization**: UTC timestamps enable alignment across separate recording systems
4. **Long-duration recordings**: Continuous timecode stream maintains sync over hours/days
5. **Mobile recordings**: Post-hoc timing recovery through embedded timecodes

## System Service Management

### Check Service Status
```bash
sudo systemctl status irig-sender.service
```

### View Logs
```bash
sudo journalctl -u irig-sender.service -f
```

### Stop/Start Service
```bash
sudo systemctl stop irig-sender.service
sudo systemctl start irig-sender.service
```

### Restart After Configuration Changes
```bash
sudo systemctl daemon-reload
sudo systemctl restart irig-sender.service
```

## File Structure

```
irig_unix_timecodes/
├── irig_sender.c              # C implementation of IRIG-H generator
├── irig_h_gpio.py             # Python IRIG-H encoder/decoder library
├── extract_from_dat.py        # DAT file IRIG extraction
├── extract_from_camera_events.py  # Camera event IRIG extraction
├── bin_analysis.py            # Binary IRIG data analysis
├── npz_analysis.py            # NPZ format IRIG data analysis
├── readSGLX.py                # SpikeGLX file reader
├── systemctl/
│   └── irig-sender.service    # Systemd service file
└── README.md
```

## References

- IRIG Standard 200-16, Range Commanders Council (2016)
- IRIG timecode specifications: https://en.wikipedia.org/wiki/IRIG_timecode
- Chrony documentation: https://chrony.tuxfamily.org/
- GPS disciplined oscillators: https://www.nist.gov/pml/time-and-frequency-division/

## License

Open source implementation for neuroscience research applications.
