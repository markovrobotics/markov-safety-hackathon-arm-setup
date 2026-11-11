#!/bin/bash
# On the ZED Box: serves the NERO's CAN bus to the laptop over Ethernet, so the
# laptop needs no CAN adapter or driver (the candleLight stick plugs into the box).
#   - kernel gs_usb names the stick nerocan and nero-can@ brings it up at 1 Mbit
#     whenever it appears (the Jetson's own CAN controller keeps can0)
#   - socketcand (linux-can) serves nerocan on <box>:29536; the laptop opens it with
#     python-can's socketcand interface: -E can_interface=socketcand -E can=nerocan@192.168.123.50
#   ./install-can.sh [ethernet interface facing the laptop, default: the one holding 192.168.123.50]
set -euo pipefail
LISTEN=${1:-$(ip -o -4 addr show | awk '$4 ~ /^192\.168\.123\.50\// {print $2; exit}')}
[ -n "$LISTEN" ] || { echo "no interface holds 192.168.123.50; pass the one facing the laptop"; exit 1; }

# JetPack's kernel ships without gs_usb, the candleLight stick's driver: build it from the same stable kernel release
# (5.15.148 for JetPack 6.1's 5.15.148-tegra) against the installed headers.
if ! modinfo gs_usb >/dev/null 2>&1; then
  kver=$(uname -r)
  src=$(mktemp -d)
  curl -fsSL "https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git/plain/drivers/net/can/usb/gs_usb.c?h=v${kver%%-*}" -o "$src/gs_usb.c"
  echo "obj-m += gs_usb.o" > "$src/Makefile"
  make -C "/lib/modules/$kver/build" M="$src" modules
  sudo install -D -m 644 "$src/gs_usb.ko" "/lib/modules/$kver/updates/gs_usb.ko"
  sudo depmod -a
fi
sudo modprobe gs_usb

sudo tee /etc/udev/rules.d/99-nero-can.rules >/dev/null <<'EOF'
ACTION!="remove", SUBSYSTEM=="net", ATTRS{idVendor}=="1d50", ATTRS{idProduct}=="606f", NAME="nerocan", TAG+="systemd", ENV{SYSTEMD_WANTS}+="nero-can@nerocan.service"
EOF
sudo tee /etc/systemd/system/nero-can@.service >/dev/null <<'EOF'
[Unit]
Description=Configure SocketCAN interface %I for NERO
BindsTo=sys-subsystem-net-devices-%i.device
After=sys-subsystem-net-devices-%i.device

[Service]
Type=oneshot
ExecStart=/usr/sbin/ip link set dev %i down
ExecStart=/usr/sbin/ip link set dev %i type can bitrate 1000000
ExecStart=/usr/sbin/ip link set dev %i up
RemainAfterExit=yes
ExecStop=/usr/sbin/ip link set dev %i down
EOF

if ! command -v socketcand >/dev/null; then
  sudo apt-get install -y meson ninja-build libconfig-dev git
  src=$(mktemp -d)
  git clone --depth 1 https://github.com/linux-can/socketcand "$src"
  (cd "$src" && meson setup build && ninja -C build && sudo install -m 755 build/src/socketcand /usr/local/bin/socketcand)
fi
sudo tee /etc/systemd/system/socketcand.service >/dev/null <<EOF
[Unit]
Description=NERO CAN (nerocan) for the laptop over Ethernet, port 29536
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=/usr/local/bin/socketcand -i nerocan -l $LISTEN -q
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
EOF
sudo udevadm control --reload
sudo systemctl daemon-reload
sudo systemctl enable --now socketcand
echo "laptop: uv run inspect-robots run ... -E can_interface=socketcand -E can=nerocan@192.168.123.50"
