# This contains the classes for the IRIG and Unix timecode generators. See 
# README file for more details
#
# Luke Sjulson, 2025-03-07

import time
import math
import pigpio
from typing import List


##############################################################################
#                            FULL IRIG 60-BIT
##############################################################################

class IRIG60BitGenerator:
    """
    Generates a 60-bit IRIG code in the layout we discussed:
      - bit0 = 'P'
      - bits1..4 = seconds (ones), bit5=0 (index), bits6..8= seconds (tens)
      - bit9='P'
      - bits10..13= minutes(ones), bit14=0, bits15..17= minutes(tens), bit18=0
      - bit19='P'
      - bits20..23= hour(ones), bit24=0, bits25..26= hour(tens), bits27..28=0
      - bit29='P'
      - bits30..33= day-of-year(ones), bit34=0, bits35..38= day-of-year(tens),
        bit39='P', bits40..41= day-of-year hundreds, bits42..44=0, bits45..48=0 (tenths),
        bit49='P'
      - bits50..53= year(ones), bit54=0, bits55..58= year(tens), bit59='P'
    Each bit is 20% high for '0', 50% for '1', 80% for 'P'. We only store 2-digit year.
    """

    def __init__(self, pi: pigpio.pi, gpio_pin: int, bit_duration: float, name="IRIG"):
        self.pi = pi
        self.gpio_pin = gpio_pin
        self.bit_duration = bit_duration
        self.name = name

        self.bits_per_frame = 60
        self.frame_time = self.bits_per_frame * self.bit_duration

        # Pulse width mapping
        self.pulse_map = {
            '0': 0.2,
            '1': 0.5,
            'P': 0.8
        }

    def create_wave(self, boundary_time: float) -> int:
        """
        Build pigpio pulses for the 60-bit IRIG wave, with leading silence
        so bit0 starts exactly at boundary_time.
        """
        now = time.time()
        lead_silence = boundary_time - now
        if lead_silence < 0:
            lead_silence = 0.0

        bits = self._generate_60_bits(boundary_time)

        pulses = []
        if lead_silence > 0:
            pulses.append(pigpio.pulse(0, 1 << self.gpio_pin, int(1_000_000 * lead_silence)))

        for bit in bits:
            frac = self.pulse_map.get(bit, 0.2)
            t_high = self.bit_duration * frac
            t_low = self.bit_duration - t_high

            pulses.append(pigpio.pulse(1 << self.gpio_pin, 0, int(t_high * 1e6)))
            pulses.append(pigpio.pulse(0, 1 << self.gpio_pin, int(t_low * 1e6)))

        self.pi.wave_add_generic(pulses)
        wave_id = self.pi.wave_create()
        return wave_id

    def _generate_60_bits(self, frame_start: float) -> List[str]:
        """
        Return the official 60-bit IRIG pattern for the local time at frame_start,
        using the layout described in detail above.
        """
        bits = ['0'] * 60

        # Position markers
        for p in [0,9,19,29,39,49,59]:
            bits[p] = 'P'

        # Index bits always '0'
        for i_bit in [5,14,24,34,54]:
            bits[i_bit] = '0'

        # local time
        tm = time.localtime(frame_start)
        sec = tm.tm_sec
        minute = tm.tm_min
        hour = tm.tm_hour
        yday = tm.tm_yday    # 1..366
        yr2 = tm.tm_year % 100

        # Seconds => bits1..4 => ones, bits6..8 => tens
        self._write_bcd_digit(bits, 1, (sec % 10), 4)
        self._write_bcd_digit(bits, 6, (sec // 10), 3)

        # Minutes => bits10..13 => ones, bits15..17 => tens
        self._write_bcd_digit(bits, 10, (minute % 10), 4)
        self._write_bcd_digit(bits, 15, (minute // 10), 3)

        # Hours => bits20..23 => ones, bits25..26 => tens
        self._write_bcd_digit(bits, 20, (hour % 10), 4)
        self._write_bcd_digit(bits, 25, (hour // 10), 2)

        # Day-of-year => bits30..33 => ones, bits35..38 => tens, bits40..41 => hundreds
        day_ones = yday % 10
        day_tens = (yday // 10) % 10
        day_hund = (yday // 100) % 10
        self._write_bcd_digit(bits, 30, day_ones, 4)
        self._write_bcd_digit(bits, 35, day_tens, 4)
        self._write_bcd_digit(bits, 40, day_hund, 2)

        # Year => bits50..53 => ones, bits55..58 => tens
        yr_ones = yr2 % 10
        yr_tens = (yr2 // 10) % 10
        self._write_bcd_digit(bits, 50, yr_ones, 4)
        self._write_bcd_digit(bits, 55, yr_tens, 4)

        return bits

    def _write_bcd_digit(self, bits: List[str], start_bit: int, digit: int, bit_count: int):
        """
        Write 'digit' (0..9, etc.) in BCD form (LSB at bits[start_bit]).
        Only up to 'bit_count' bits are used, typically 2..4.
        """
        for i in range(bit_count):
            if digit & (1 << i):
                bits[start_bit + i] = '1'
            else:
                bits[start_bit + i] = '0'


##############################################################################
#                           UNIX 60-BIT CLASS
##############################################################################

class Unix60BitGenerator:
    """
    60-bit "Unix" code:
      - bit0='P'
      - bits1..58 => 58-bit two's complement of boundary_time
      - bit59='P'
    Each bit uses the same 20%/50%/80% high ratio for '0','1','P'.
    bit_duration => e.g. 0.01 for B, 0.1 for E, 1.0 for H.
    """

    def __init__(self, pi: pigpio.pi, gpio_pin: int, bit_duration: float, name="Unix"):
        self.pi = pi
        self.gpio_pin = gpio_pin
        self.bit_duration = bit_duration
        self.name = name

        self.bits_per_frame = 60
        self.frame_time = self.bits_per_frame * self.bit_duration

        self.pulse_map = {
            '0': 0.2,
            '1': 0.5,
            'P': 0.8
        }

    def create_wave(self, boundary_time: float) -> int:
        now = time.time()
        lead_silence = boundary_time - now
        if lead_silence < 0:
            lead_silence = 0.0

        bits = self._generate_60_bits(boundary_time)

        pulses = []
        if lead_silence > 0:
            pulses.append(pigpio.pulse(0, 1 << self.gpio_pin, int(1_000_000 * lead_silence)))

        for bit in bits:
            frac = self.pulse_map.get(bit, 0.2)
            t_high = self.bit_duration * frac
            t_low = self.bit_duration - t_high

            pulses.append(pigpio.pulse(1 << self.gpio_pin, 0, int(t_high * 1e6)))
            pulses.append(pigpio.pulse(0, 1 << self.gpio_pin, int(t_low * 1e6)))

        self.pi.wave_add_generic(pulses)
        return self.pi.wave_create()

    def _generate_60_bits(self, boundary_time: float) -> List[str]:
        bits = ['0'] * 60
        bits[0] = 'P'
        bits[59] = 'P'

        unix_ts = int(boundary_time)
        # 58-bit two's complement
        mask_58 = (1 << 58) - 1
        twos_val = unix_ts & mask_58

        for i in range(58):
            bit_i = (twos_val >> i) & 1
            bits[i+1] = '1' if bit_i else '0'

        return bits


##############################################################################
#                           misc helper functions
##############################################################################


def build_wave_ids_for_boundary(boundary: float,
                                irig_b, irig_e, irig_h,
                                unix_b, unix_e, unix_h):
    """
    Build waveforms for B every second, E if boundary%10==0, H if boundary%60==0
    """
    wave_ids = []
    # IRIG B + Unix B => every second
    wave_ids.append(irig_b.create_wave(boundary))
    wave_ids.append(unix_b.create_wave(boundary))

    # E => if multiple of 10
    if (boundary % 10) == 0:
        wave_ids.append(irig_e.create_wave(boundary))
        wave_ids.append(unix_e.create_wave(boundary))

    # H => if multiple of 60
    if (boundary % 60) == 0:
        wave_ids.append(irig_h.create_wave(boundary))
        wave_ids.append(unix_h.create_wave(boundary))

    return wave_ids


def debug_print_bits(boundary_time: float, code_gens: List):
    """
    Print the 60-bit pattern for each generator in code_gens, grouped in tens,
    for the given boundary_time. Called once every 10 seconds in the main loop.
    """
    local_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(boundary_time))
    print(f"\n--- Debug at boundary={boundary_time} ({local_str}) ---")

    for gen in code_gens:
        # Each generator has an internal _generate_60_bits() method
        # We'll call that to see the pattern for boundary_time
        bits = gen._generate_60_bits(boundary_time)
        # group bits in tens
        group_size = 10
        grouped = []
        for i in range(0, len(bits), group_size):
            chunk = bits[i:i+group_size]
            grouped.append(''.join(chunk))

        print(f"{gen.name} bits:")
        for i, grp in enumerate(grouped):
            start_i = i * group_size
            end_i = start_i + len(grp) - 1
            print(f"  Bits {start_i:02d}-{end_i:02d}: {grp}")
        print()