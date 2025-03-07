# irig_unix_timecodes
Plays IRIG and Unix timecodes on the GPIO pins of a Raspberry Pi 4B

<h1>NOTE: THIS CODE HAS NOT BEEN TESTED YET AND SHOULD NOT BE USED FOR EXPERIMENTS</h1>

The overarching goal of this project is to enable synchronization of multiple data streams to UTC time, a universal time standard independent of timezone. To this end, we implement IRIG timecodes, a method enabling UTC timestamps to be encoded in sequences of short and long TTL pulses from the Raspberry Pi's GPIO pins that can be sampled by an electrophysiology recording setup or used to blink an LED recorded by an imaging setup. This enables your recordings to be timestamped with the year, date, and time with millisecond precision.

The UTC time comes from the Raspberry Pi's internal clock, which should ideally be synchronized via chrony (a high-precision implementation of NTP, the Network Time Protocol) to a local stratum 1 NTP server with a direct link to GPS satellites. If this is not available to you, chrony will synchronize to an NTP server over the internet, which will lead to timestamps that are less objectively accurate with respect to true UTC time. However, they will still be equally precise, i.e. the frame timestamps for the RPi camera and the IRIG/Unix timecodes will still be internally consistent, which is likely adequate for most experiments.

This python code uses the pigpio library for low-latency GPIO pin toggling. As of March 2025, this does not run on the Raspberry Pi 5, so this code is intended for the RPi 4B. 

--------------
IRIG timecodes
--------------

GPIO pin 4: IRIG B, 600 ms, 60 bits at 100 Hz (0.01 seconds per bit)\
GPIO pin 5: IRIG E, 6 seconds, 60 bits at 10 Hz (0.1 seconds per bit)\
GPIO pin 6: IRIG H, 60 seconds, 60 bits at 1 Hz (1 second per bit)\

The specifications of the IRIG timecode are here:\
https://en.wikipedia.org/wiki/IRIG_timecode (also see included IRIG_timecode.png)

This implementation is only 60 bits long (most IRIG implementations are 100 bits, but the extra bits do not carry useful information).

--------------
UNIX timecodes
--------------

GPIO pin 7: Unix, 600 ms, 60 bits at 100 Hz (0.01 seconds per bit)\
GPIO pin 8: Unix, 6 seconds, 60 bits at 10 Hz (0.1 seconds per bit)\
GPIO pin 9: Unix, 60 seconds, 60 bits at 1 Hz (1 second per bit)

This implements a Unix timecode, which is not standardized. Briefly, it is like an IRIG timecode, except it outputs the UTC time in Unix format (the number of seconds since Jan. 1, 1970). This is a sequence of 60 bits, where the first and last bits are position markers. The Unix time is therefore encoded as a signed integer with the 58 remaining bits. The first bit in the integer is the LSB, and the last bit is the sign. The reason for using a 58-bit signed int is 1) to avoid the "2038 problem," when a 32-bit signed int rolls over in the year 2038, and 2) to make the sequence 60 bits long, consistent with IRIG-H.


-------------------
PIGPIO INSTALLATION
-------------------

This timecode generator runs continuously in the background as a daemon. If a real-time kernel is used, the precision will be higher.

You will also need to install pigpio:

sudo apt install pigpio\
sudo systemctl enable pigpiod

then:\
sudo nano /lib/systemd/system/pigpiod.service

and edit the ExecStart line to say:\
ExecStart=/usr/bin/pigpiod -l -m -s 10

next:\
sudo systemctl daemon-reload\
sudo systemctl start pigpiod


-------------------------------
TIMECODE GENERATOR INSTALLATION
-------------------------------

Next, copy this repository to /home/pi/IRIG and test whether the timecode generators run. Type:

~/IRIG/make_timecodes.py

you should see:

Starting combined IRIG/Unix 60-bit generator (debug prints every 10s).

Uncomment the necessary lines to see debugging output.

If you need to install any dependencies, do that now so that make_timecodes.py will run.

The next step is to install the timecode generator as a system daemon:

sudo cp ~/IRIG/systemctl/*.service /etc/systemd/system\
sudo systemctl enable irig_unix_timecodes.service\
sudo systemctl start irig_unix_timecodes.service

The timecodes should now run continuously, even after rebooting.