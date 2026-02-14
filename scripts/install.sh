#!/bin/bash

# Install the IRIG sender service. Chrony config needs to be done separately.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Compile the program
make -C "$REPO_DIR/sender"

# Copy to system location
sudo cp "$REPO_DIR/sender/irig_sender" /usr/local/bin/
sudo chmod +x /usr/local/bin/irig_sender

sudo mkdir -p /var/log/irig-sender

# Save the service file
sudo cp "$REPO_DIR/systemd/irig-sender.service" /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable irig-sender.service

# Start the service now
sudo systemctl start irig-sender.service
