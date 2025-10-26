#!/bin/bash

# THIS ONLY DE-SETS UP THE IRIG SENDING SERVICE! Chrony config needs to be done seperately

# Stop the service
sudo systemctl stop irig-sender.service

# Disable service from starting on boot
sudo systemctl disable irig-sender.service

# Remove the service file
sudo rm -f /etc/systemd/system/irig-sender.service

# Reload systemd
sudo systemctl daemon-reload

# Remove log directory
sudo rm -rf /var/log/irig-sender

# Remove binary from system location
sudo rm -f /usr/local/bin/irig_sender

# Remove compiled binary from local directory
rm -f irig_sender