#!/bin/bash
# On the ZED Box (ZED SDK already installed): installs the camera server as a
# systemd service that starts on boot.
#   ./install.sh wrist=<serial> tripod=<serial>     (serials: python3 camera_server.py --list)
set -euo pipefail
[ $# -ge 1 ] || { python3 "$(dirname "$0")/camera_server.py" --list; echo "usage: $0 wrist=<serial> tripod=<serial>"; exit 1; }
python3 -c "import pyzed" 2>/dev/null || python3 /usr/local/zed/get_python_api.py
# apt's python3-opencv is built against numpy 1 and fails to import beside the numpy 2 the ZED SDK brings
python3 -c "import cv2" 2>/dev/null || pip3 install --user "opencv-python-headless<4.12"
sudo install -D -m 644 "$(dirname "$0")/camera_server.py" /opt/inspect-robots-nero/camera_server.py
sudo tee /etc/systemd/system/zed-cameras.service >/dev/null <<EOF
[Unit]
Description=ZED X Nano frames over HTTP
# The SDK reads each camera's motion sensors through IMU_Daemon; without it the cameras fail to open.
Wants=IMU_Daemon.service
After=network-online.target zed_x_daemon.service IMU_Daemon.service

[Service]
# camera_server.py exits 3 when a camera stops giving frames (a GMSL link drop). Argus keeps the dropped camera's dead
# session until its daemons restart, so that exit restarts them (as root: "+") before systemd starts the server again.
ExecStopPost=+/bin/sh -c '[ "\$EXIT_STATUS" = 3 ] || exit 0; systemctl restart zed_x_daemon.service; sleep 3; systemctl restart nvargus-daemon.service; sleep 10'
ExecStart=/usr/bin/python3 /opt/inspect-robots-nero/camera_server.py $*
Restart=always
RestartSec=3
User=$USER

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now zed-cameras
echo "serving: curl http://$(hostname -I | cut -d' ' -f1):8080/wrist.jpg"
