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