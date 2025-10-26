# Compile the program
gcc -o irig_sender irig_sender.c -lpthread -lm

# Copy to system location
sudo cp irig_sender /usr/local/bin/
sudo chmod +x /usr/local/bin/irig_sender

sudo mkdir -p /var/log/irig-sender

# Save the service file
sudo cp systemctl/irig-sender.service /etc/systemd/system/
# (paste the content from above)

# Reload systemd
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable irig-sender.service

# Start the service now
sudo systemctl start irig-sender.service