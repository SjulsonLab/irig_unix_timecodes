# NeuroKairos Pipeline Summary

## The Problem

In a multi-modal neuroscience experiment you might record electrophysiology at 30 kHz, a behavior camera at 30 fps, and a photometry signal at 1 kHz. Each device has its own internal clock. These clocks drift relative to each other — typically 10-100 parts per million — so after a 1-hour recording, timestamps from different devices can disagree by hundreds of milliseconds. You cannot align streams by simply assuming they started at the same instant.

NeuroKairos solves this by embedding an objective UTC time reference into every recording. Each device independently records the same timing signal, so you can anchor all streams to a common clock after the fact.

## System Overview

The pipeline has two phases: **generation** (during the experiment) and **decoding** (post-hoc analysis).

```
Phase 1: Generation (during experiment)
                                          ┌──────────────────┐
  GPS satellites ──► Raspberry Pi ────────┤  GPIO TTL output  │
                     (UTC clock +         └────────┬─────────┘
                      IRIG encoder)                │
                                        ┌──────────┼──────────┐
                                        │          │          │
                                        ▼          ▼          ▼
                                     Ephys       Camera     Other
                                    (spare     (LED in      device
                                    channel)    view)

Phase 2: Decoding (post-hoc)

  Recording file ──► neurokairos ──► ClockTable ──► pynapple
  (with IRIG           Python        (.npz)        (synchronized
   signal inside)      decoder                      time series)
```

## Phase 1: Generating the Timing Signal

### Hardware

- **Raspberry Pi 4B** with a GPS timing receiver (e.g., Waveshare NEO-M8T GNSS hat)
- GPS antenna with sky visibility
- A BNC or wire from the Pi's GPIO pin to each recording device

### What happens

1. The GPS receiver locks to satellites and produces a pulse-per-second (PPS) signal accurate to ~100 ns of UTC.
2. Chrony (an NTP daemon) disciplines the Pi's system clock to the GPS PPS signal, achieving stratum-1 accuracy.
3. A C program (`irig_sender`) reads the system clock and continuously generates IRIG-H pulses on a GPIO pin.

### The IRIG-H signal

IRIG-H is a pulse-width-modulated timecode: one pulse per second, where the pulse width encodes a binary digit.

| Pulse width | Meaning |
|-------------|---------|
| 0.2 seconds | Binary 0 |
| 0.5 seconds | Binary 1 |
| 0.8 seconds | Position marker |

60 pulses form one frame (= 1 minute), encoding the current UTC time (minutes, hours, day of year, year) in Binary Coded Decimal (BCD).

### How it reaches your equipment

The GPIO output is a TTL voltage signal (0 V / 3.3 V). You split this signal and feed it to:
- **Electrophysiology**: one spare analog-in channel (e.g., the last channel of a Neuropixels NI-DAQ, or a spare Open Ephys ADC channel). The signal appears as a square wave recorded alongside your neural data.
- **Cameras**: point the camera at an LED driven by the same GPIO pin. The LED blinks in the IRIG-H pattern, visible in the video.
- **Any other device with analog input or TTL input**: same principle.

No modifications to your existing equipment are required — you just need one spare input channel or an LED visible to the camera.

## Phase 2: Decoding IRIG from Recordings

After the experiment, you run the `neurokairos` Python decoder on each recording file. The decoder:

1. **Reads the IRIG channel** from the recording (one channel of the binary file, or LED brightness extracted from video frames).
2. **Detects edges** in the signal (rising/falling threshold crossings) using Otsu's method for automatic thresholding.
3. **Measures pulse widths** and classifies each pulse as binary 0, 1, or position marker based on duration.
4. **Finds frame boundaries** (two consecutive markers) and decodes the BCD-encoded UTC time from each complete 60-pulse frame.
5. **Assigns a UTC timestamp to every pulse** by combining decoded frame times with inter-pulse timing.
6. **Produces a ClockTable**: a sparse mapping between the recording's native time base (sample indices or frame numbers) and UTC.

### Concrete example: decoding a SpikeGLX recording

```python
import neurokairos

# Input:  recording.bin (raw SpikeGLX binary) + recording.meta
# The IRIG signal is on the last ("sync") channel
clock_table = neurokairos.decode_sglx_irig("recording.bin", irig_channel="sync")

# Output: a ClockTable saved to recording.bin.clocktable.npz
print(clock_table)
# ClockTable: 238 entries (samples), rate=30000.0
#   recording: 2025-01-15T14:30:37Z -> 2025-01-15T14:34:35Z
#   source=[6000.0..7134000.0], reference=[1736950237.0..1736950475.0]
```

### What is a ClockTable?

A ClockTable is the core output of neurokairos. It contains:

- **source**: an array of values in the recording's native domain (e.g., sample indices for ephys, frame indices for video)
- **reference**: an array of corresponding UTC timestamps (Unix epoch seconds)
- **nominal_rate**: the sampling rate (e.g., 30000 Hz)
- **metadata**: decoding quality, NTP sync status, file provenance

There is roughly one entry per IRIG pulse (~1 per second). Between entries, linear interpolation handles the conversion. This is what corrects for clock drift — the mapping between sample index and UTC is not perfectly linear, and the ClockTable captures that.

```python
import numpy as np

# Convert sample indices to UTC timestamps
utc_times = clock_table.source_to_reference(np.array([30000, 60000, 90000]))

# Convert UTC timestamps back to sample indices
samples = clock_table.reference_to_source(np.array([1736950240.0, 1736950250.0]))
```

### Available decoders

| Decoder | Input | Source units |
|---------|-------|--------------|
| `decode_sglx_irig` | SpikeGLX `.bin` + `.meta` | sample indices |
| `decode_dat_irig` | Interleaved int16 `.dat` (Open Ephys, Intan, etc.) | sample indices |
| `decode_video_irig` | Video file (AVI, MP4) with visible IRIG LED | frame indices |
| `decode_intervals_irig` | Pre-extracted pulse onset/offset times | seconds (or custom) |

## From ClockTable to Pynapple

Pynapple expects time series (`Tsd`, `TsdFrame`) with timestamps in seconds relative to some origin. The ClockTable bridges the gap:

```python
import numpy as np
import pynapple as nap
import neurokairos

# --- Step 1: Decode IRIG from each recording ---
ephys_ct = neurokairos.decode_sglx_irig("ephys.bin", irig_channel="sync")
video_ct = neurokairos.decode_video_irig("behavior.avi", roi=(10, 20, 10, 20))

# --- Step 2: Convert to pynapple time series ---
# For ephys: convert sample indices to UTC, then to relative seconds
sample_indices = np.arange(0, 120000, dtype=np.float64)  # 4 seconds at 30 kHz
utc_times = ephys_ct.source_to_reference(sample_indices)
time_origin = ephys_ct.reference[0]                       # first UTC timestamp
ephys_tsd = nap.Tsd(t=utc_times - time_origin, d=my_neural_data)

# For video: convert frame indices to UTC, then to relative seconds
frame_indices = np.arange(len(x_positions), dtype=np.float64)
utc_times = video_ct.source_to_reference(frame_indices)
tracking_tsd = nap.Tsd(t=utc_times - time_origin, d=x_positions)
#                                       ^^^^^^^^^^^
#                        use the SAME time_origin for both streams

# --- Step 3: Analyze together in pynapple ---
# Both Tsd objects now share a common time axis (seconds since ephys start)
# You can compute cross-correlations, restrict to intervals, etc.
```

The key insight: because both streams were decoded against the same UTC reference, converting them to the same `time_origin` puts them on a common time axis — even though the original devices had independent, drifting clocks.

## How Precise Is This?

### End-to-end timing accuracy

Validated over 25 continuous hours against a 30 kHz electrophysiology recording:

| Metric | Value |
|--------|-------|
| Average timing accuracy (IRIG pulse vs GPS PPS) | **33 microseconds** |
| Events at primary peak (33 us) | 99.44% |
| Events at secondary peak (67 us) | 0.56% |
| Rare excursions (0.7-4 ms) | < 0.005% |
| Decoding errors over 25 hours | **0** |

### Where does the precision come from?

The precision budget breaks down roughly as follows:

1. **GPS to Pi system clock** (~100 ns): The GPS PPS signal disciplines chrony to sub-microsecond accuracy.
2. **Pi system clock to GPIO pulse** (~33 us): The C sender uses real-time scheduling (SCHED_FIFO) and direct memory-mapped GPIO writes with a hybrid sleep/busy-wait loop. This is the dominant source of timing error.
3. **GPIO pulse to ADC sample** (< 1 sample period): The recording device digitizes the pulse at its sampling rate. At 30 kHz, one sample = 33 us.
4. **Decoding interpolation** (< 1 sample period): The decoder finds threshold crossings at sample resolution, then interpolates between ~1 Hz anchor points.

**For a 30 kHz electrophysiology setup, the total synchronization error is typically ~33 us (1 sample).** This is sub-millisecond and sufficient for virtually all neuroscience applications, including spike-timing analyses.

### What about video?

For a 30 fps camera, precision is limited by the frame rate: ~33 ms (1 frame). The IRIG signal itself is far more precise than this — the camera's frame rate is the bottleneck. A 120 fps camera would give ~8 ms precision.

### NTP sync quality monitoring

NeuroKairos encodes the GPS/NTP synchronization quality directly into the IRIG signal (using previously unused bits in the IRIG-H frame). The decoder extracts this and stores it in the ClockTable metadata:

```python
ct.metadata["stratum"]             # NTP stratum (1 = GPS-locked, best)
ct.metadata["UTC_sync_precision"]  # e.g., "< 0.25 ms"
```

This lets you verify after the fact that the timing signal was properly GPS-locked during your recording.

## Summary of Inputs and Outputs

| Stage | Input | Output |
|-------|-------|--------|
| **GPS receiver** | Satellite signals | PPS pulse + NMEA time |
| **Chrony** | PPS + NMEA | Disciplined system clock (stratum 1) |
| **IRIG sender** | System clock | TTL pulses on GPIO pin |
| **Recording** | TTL pulses (or LED) | Binary data file with IRIG on one channel |
| **neurokairos decoder** | Recording file | **ClockTable** (.npz): sample index <-> UTC mapping |
| **Your analysis code** | ClockTable + raw data | Pynapple Tsd/TsdFrame with UTC-referenced timestamps |

The ClockTable is the handoff point. NeuroKairos produces it; your analysis code (with pynapple or anything else) consumes it.
