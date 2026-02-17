# C Sender Timing Loop

This document explains the main timing loop in `sender/irig_sender.c` — how the IRIG-H sender achieves microsecond-level pulse timing on a Raspberry Pi while keeping CPU usage reasonable.

## Overview

The sender transmits one IRIG-H frame per minute: 60 pulses at 1 Hz, each pulse-width modulated (0.2s, 0.5s, or 0.8s) to encode a binary 0, 1, or position marker. The challenge is hitting each pulse edge within a few microseconds of its target time on a non-realtime OS.

The timing system has three layers:

1. **Frame loop** (`continuous_irig_sending`) — orchestrates the 60-bit frame sequence
2. **Wait function** (`ultra_wait_until_ns`) — sleeps until a target nanosecond timestamp
3. **Pulse function** (`ultra_fast_pulse`) — holds a GPIO pin high for a precise duration

## Frame Loop: `continuous_irig_sending`

```
Thread start
  ↓
Set SCHED_FIFO priority 80
  ↓
Poll chrony for initial sync status
  ↓
Compute next minute boundary → next_frame_time
Pre-calculate frame data for next_frame_time
  ↓
┌─────────────────────────────────────────────┐
│ OUTER LOOP (one iteration per frame/minute) │
│                                             │
│  Copy next_frame → current_frame            │
│                                             │
│  ┌────────────────────────────────────────┐ │
│  │ INNER LOOP: for i = 0..59             │ │
│  │                                       │ │
│  │  ultra_wait_until_ns(bit_start_time)  │ │
│  │  ultra_fast_pulse(pulse_duration)     │ │
│  │                                       │ │
│  └────────────────────────────────────────┘ │
│                                             │
│  Poll chrony status                         │
│  Update LED indicator                       │
│  next_frame_time += 60                      │
│  Pre-calculate next frame                   │
│                                             │
└─────────────────────────────────────────────┘
```

### Key design choices

**Deterministic for-loop instead of polling.** The inner loop iterates over all 60 bits sequentially, calling `ultra_wait_until_ns()` for each bit's pre-calculated absolute start time. There is no polling or `usleep()` between bits — the wait function itself handles all sleeping. This eliminates accumulated drift: each bit targets an absolute timestamp, so any jitter on one bit doesn't affect the next.

**Pre-calculated absolute timestamps.** Before each frame, `precalculate_next_frame()` computes all 60 bit start times as absolute nanosecond values:

```c
bit_start_times[i] = frame_start_ns + (i * NS_PER_SEC) - OFFSET_NS;
```

The `OFFSET_NS` (20 µs) compensates for GPIO register write latency, measured via oscilloscope. Because these are absolute times (not deltas), any jitter in one pulse cannot propagate to subsequent pulses.

**Frame preparation during dead time.** `precalculate_next_frame()` (BCD encoding, chrony status injection, timestamp computation) runs in the ~200ms gap after the last pulse of a frame ends and before the first pulse of the next frame begins. This keeps the hot path (wait → pulse → wait → pulse) free of any computation.

**SCHED_FIFO at priority 80.** The sending thread requests real-time FIFO scheduling. This prevents the kernel from preempting the thread during the critical busy-wait windows, reducing worst-case latency from ~10ms (normal CFS scheduling) to <0.5ms. This is the single most impactful change for timing precision. CPU usage impact is negligible — SCHED_FIFO only changes scheduling policy, not how much CPU the thread consumes.

**Minute-boundary alignment.** The first frame always starts at a :00 second boundary. Since each frame is exactly 60 bits at 1 Hz, subsequent frames naturally align to minutes. This means the BCD seconds field in every frame is always 0, simplifying decoder logic.

## Wait Function: `ultra_wait_until_ns`

This is the core scheduling primitive. It must bridge gaps ranging from ~200ms (between the end of one pulse and the start of the next) up to ~60s (waiting for the first minute boundary), while waking up within microseconds of the target.

```
                         target_ns
                            │
 ──────┬──────────┬─────────┤
       │  sleep   │busy-wait│
       │ (capped) │ (spin)  │
       ▼          ▼         ▼
    nanosleep   enter      done
    in chunks   spin loop
```

### Phase 1: Coarse sleep (nanosleep)

When `remaining_ns > BUSY_WAIT_BUFFER_NS` (10 ms), the function sleeps for `remaining - 10ms`, yielding the CPU to other processes. Two safeguards limit jitter:

- **100ms sleep cap (`MAX_SLEEP_NS`).** Individual `nanosleep()` calls are capped at 100ms. Long sleeps (seconds) can overshoot by 10ms+ on Linux due to timer coalescing and CFS scheduling; short sleeps (<100ms) typically overshoot by <5ms. The function loops, re-checking the clock after each capped sleep, converging on the target in steps.

- **10ms early wakeup (`BUSY_WAIT_BUFFER_NS`).** Sleep stops 10ms before the target to leave room for the busy-wait phase. This must exceed worst-case `nanosleep()` overshoot: 1–10ms without RT scheduling, <0.5ms with SCHED_FIFO. The 10ms value is conservative and works with or without RT.

**CPU usage:** During this phase the thread is fully sleeping — zero CPU. For a typical inter-pulse gap of ~200ms, the thread sleeps for ~190ms and only burns CPU for the final ~10ms. For the initial minute-boundary wait (up to 60s), the thread sleeps in 100ms chunks, waking briefly to recheck the clock — negligible CPU.

**Latency:** The sleep cap ensures worst-case overshoot per sleep call stays bounded. Even if one `nanosleep(100ms)` overshoots by 5ms, the function simply re-enters the sleep loop with the corrected remaining time. The 10ms buffer is the key parameter: it determines the maximum error that the busy-wait phase must absorb.

### Phase 2: Busy-wait (spin loop)

Once within 10ms of the target, the function enters a tight spin loop:

```c
do {
    clock_gettime(CLOCK_REALTIME, &current_time);
    current_ns = timespec_to_ns(&current_time);
} while (current_ns < target_ns && running);
```

This is a pure busy-wait (`BUSY_WAIT_SLEEP_NS = 0`). The loop polls `clock_gettime()` continuously until the target time is reached.

**CPU usage:** Burns one full CPU core during the spin. At 10ms per pulse with 60 pulses/minute, that's ~600ms of busy-wait per minute for the inter-pulse waits alone. Combined with the pulse-hold busy-waits (see below), total busy-wait CPU is roughly 1–1.5 seconds per minute, or ~2% of one core. Acceptable on a quad-core RPi 4.

**Latency:** `clock_gettime(CLOCK_REALTIME)` on the RPi 4 takes ~50–100ns per call via the vDSO (no syscall overhead). The spin loop therefore overshoots the target by at most one `clock_gettime` call duration — well under 1 µs. With SCHED_FIFO, the thread won't be preempted during the spin, so this precision is reliable.

### Compile-time option: `BUSY_WAIT_SLEEP_NS`

If set to a nonzero value, the busy-wait phase calls `nanosleep()` between clock checks instead of pure spinning. This would reduce CPU during the final approach but increase jitter. Currently set to 0 (pure busy-wait) for maximum precision.

## Pulse Function: `ultra_fast_pulse`

Holds the GPIO pin high for a precise duration (200ms, 500ms, or 800ms). Uses the same two-phase sleep/busy-wait strategy as `ultra_wait_until_ns`, but also manages the GPIO transitions.

```
 ┌─────────────────────────────────┐
 │         pulse_duration_ns       │
 │                                 │
 │  GPIO SET ──────────── GPIO CLR │
 │  (rising)              (falling)│
 │                                 │
 │  [nanosleep...]  [busy-wait]    │
 └─────────────────────────────────┘
```

### GPIO register access

The function uses memory-mapped GPIO registers for minimal latency:

```c
*(sender->gpio_set_reg) = sender->gpio_mask;      // rising edge
// ... wait for pulse duration ...
*(sender->gpio_clr_reg) = sender->gpio_mask;      // falling edge
```

Direct register writes via `/dev/mem` take ~10ns, vs. ~10µs for sysfs or ~100µs for pigpio daemon calls. The register pointers and bitmasks are cached in the sender struct at init time to avoid any per-pulse lookup.

When a pin is disabled (`-1`), its mask is 0, and writing 0 to GPSET0/GPCLR0 is a hardware no-op — no branch needed in the hot path.

### Timing structure

The pulse function computes its own `target_ns = now + pulse_duration_ns` at entry, then uses the same capped-nanosleep → busy-wait pattern. The falling edge (GPIO clear) fires immediately when the spin loop exits.

**CPU usage:** Same as the wait function — sleeps for most of the pulse, busy-waits for the last 10ms. A 200ms pulse busy-waits for ~10ms; a 800ms pulse also busy-waits for ~10ms. The sleep phase is fully idle.

**Latency:** The falling edge precision is the same as the wait function: <1µs with SCHED_FIFO. The rising edge has no timing concern — it fires immediately when `ultra_wait_until_ns` returns, with only a register write (~10ns) in between.

## Timing Constants Summary

| Constant | Value | Purpose |
|---|---|---|
| `OFFSET_NS` | 20 µs | Compensates for GPIO write latency (tuned via oscilloscope) |
| `BUSY_WAIT_BUFFER_NS` | 10 ms | How early to stop sleeping and start spinning |
| `BUSY_WAIT_SLEEP_NS` | 0 | Spin loop sleep interval (0 = pure busy-wait) |
| `MAX_SLEEP_NS` | 100 ms | Cap on individual `nanosleep()` calls |
| `bit_length_ns` | 1,000,000,000 | Bit period (1 second) |

## CPU and Latency Budget

Per 60-second frame:

| Phase | Duration | CPU | Latency |
|---|---|---|---|
| Coarse sleep (inter-pulse) | ~50s total | ~0% (sleeping) | N/A (not timing-critical) |
| Busy-wait (inter-pulse) | ~600ms total (60 × 10ms) | 100% of one core | <1 µs overshoot per wakeup |
| Coarse sleep (during pulse) | ~27s total | ~0% (sleeping) | N/A |
| Busy-wait (during pulse) | ~600ms total (60 × 10ms) | 100% of one core | <1 µs overshoot at falling edge |
| Frame prep + chrony poll | ~1ms | Negligible | N/A |

**Total busy-wait CPU: ~1.2s per 60s = ~2% of one core.** The remaining ~98% of the time, the thread is sleeping and consuming no CPU. On a quad-core RPi 4, this is well within budget.

**Worst-case pulse edge jitter with SCHED_FIFO: <1 µs.** Without SCHED_FIFO, kernel preemption during the busy-wait phase can cause occasional spikes of 1–10ms. The `BUSY_WAIT_BUFFER_NS` of 10ms is sized to absorb worst-case `nanosleep()` overshoot even without RT scheduling, but SCHED_FIFO eliminates preemption during the spin and brings jitter down to the `clock_gettime()` resolution.
