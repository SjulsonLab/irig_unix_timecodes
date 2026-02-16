"""
Manual integration test for the Python IRIG-H sender.

Starts the IrigHSender on GPIO pin 6 and lets it run until interrupted with Ctrl-C.
This is not an automated test suite -- it requires a Raspberry Pi with pigpiod running
and is meant for manually verifying that IRIG-H frames are being transmitted correctly
(e.g., by observing the GPIO output on an oscilloscope or logic analyzer).
"""

from neurokairos.irig_h_gpio import IrigHSender
import time

sender = IrigHSender(sending_gpio_pin=6)

try:
    sender.start()
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print('keyboard interrupt recieved. stopping...')
finally:
    sender.finish()