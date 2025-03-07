#!/usr/bin/env python3

import time
import math
import pigpio
from typing import List
import timecode_generators as TC


"""
This script implements IRIG and Unix timecode generators that play timecodes
on different GPIO pins. This was tested on the Raspberry Pi 4B.

--------------
IRIG timecodes
--------------

GPIO pin 4: IRIG B, 600 ms, 60 bits at 100 Hz (0.01 seconds per bit)
GPIO pin 5: IRIG E, 6 seconds, 60 bits at 10 Hz (0.1 seconds per bit)
GPIO pin 6: IRIG H, 60 seconds, 60 bits at 1 Hz (1 second per bit)

The specifications of the IRIG timecode are here: 
https://en.wikipedia.org/wiki/IRIG_timecode (or see included IRIG_timecode.png)

The full IRIG timecode is usually 100 bits, but in this case the first 60 are
sufficient. A "zero" is a pulse that is 20% of the bit duration, a "one" is 50%
of the bit duration, and a position marker bit is 80% of the bit duration.

--------------
UNIX timecodes
--------------

GPIO pin 7: Unix, 600 ms, 60 bits at 100 Hz (0.01 seconds per bit)
GPIO pin 8: Unix, 6 seconds, 60 bits at 10 Hz (0.1 seconds per bit)
GPIO pin 9: Unix, 60 seconds, 60 bits at 1 Hz (1 second per bit)

This implements a Unix timecode, which is not standardized. Briefly, it is like
an IRIG timecode, except it outputs the UTC time in Unix format (the number of
seconds since Jan. 1, 1970). This is a sequence of 60 bits, where the first and
last bits are position markers. The Unix time is therefore encoded as a signed
integer with the 58 remaining bits. The first bit in the integer is the LSB,
and the last bit is the sign. The reason for using a 58-bit signed int is 
1) to avoid the "2038 problem," when a 32-bit signed int rolls over in the year
2038, and 2) to make the sequence 60 bits long like the IRIG codes.

by Luke Sjulson, 2025-03-08

"""



pi = pigpio.pi()
if not pi.connected:
    raise RuntimeError("Could not connect to pigpio daemon.")

irig_b = TC.IRIG60BitGenerator(pi, gpio_pin=4,  bit_duration=0.01, name="IRIG-B")
irig_e = TC.IRIG60BitGenerator(pi, gpio_pin=5,  bit_duration=0.1,  name="IRIG-E")
irig_h = TC.IRIG60BitGenerator(pi, gpio_pin=6,  bit_duration=1.0,  name="IRIG-H")

# Unix B/E/H => TC.Unix60BitGenerator
unix_b = TC.Unix60BitGenerator(pi, gpio_pin=7, bit_duration=0.01, name="Unix-B")
unix_e = TC.Unix60BitGenerator(pi, gpio_pin=8, bit_duration=0.1,  name="Unix-E")
unix_h = TC.Unix60BitGenerator(pi, gpio_pin=9, bit_duration=1.0,  name="Unix-H")

# Precompute wave IDs for next boundary
now = time.time()
current_boundary = math.floor(now) + 1

wave_ids_for_boundary = TC.build_wave_ids_for_boundary(current_boundary,
                                                    irig_b, irig_e, irig_h,
                                                    unix_b, unix_e, unix_h)
wave_ids_old = []

print("Starting combined IRIG/Unix 60-bit generator (debug prints every 10s).")
try:
    while True:
        # 1) Sleep until boundary-0.2
        wake_time = current_boundary - 0.2
        now = time.time()

        # # for debugging only
        # print(f"current_boundary = {current_boundary:.2f}") # wotan
        # print(f"wake time = {wake_time:.2f}") # wotan
        # print(f"now = {now:.2f}") # wotan

        if wake_time > now:
            time.sleep(wake_time - now)

        # 2) wave_send_once
        for wid in wave_ids_for_boundary:
            pi.wave_send_once(wid)

        # 3) wave_delete old wave IDs
        for wid in wave_ids_old:
            pi.wave_delete(wid)
        wave_ids_old = wave_ids_for_boundary

        # # for debugging only
        # # 4) If boundary multiple of 10 => debug print
        # if (current_boundary % 10) == 0:
        #     TC.debug_print_bits(current_boundary, [irig_b, irig_e, irig_h, unix_b, unix_e, unix_h])

        # 5) Build wave IDs for next boundary
        next_boundary = current_boundary + 1
        wave_ids_for_boundary = TC.build_wave_ids_for_boundary(next_boundary,
                                                            irig_b, irig_e, irig_h,
                                                            unix_b, unix_e, unix_h)

        current_boundary = next_boundary

except KeyboardInterrupt:
    print("Stopping on Ctrl+C.")
finally:
    for wid in wave_ids_for_boundary:
        pi.wave_delete(wid)
    for wid in wave_ids_old:
        pi.wave_delete(wid)
    pi.stop()


