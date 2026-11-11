#!/bin/bash
# Linux laptop: bridge the Ethernet ports the ZED Box, the hand and the arm controller plug into
# (br-rig), give the laptop 192.168.123.10 (box subnet), 192.168.11.10 (the hand's factory subnet)
# and an IPv6 link-local address (the arm controller), and NAT the box out through the laptop's
# internet link so it can apt/pip.
#   sudo laptop/net-linux.sh eth1 eth2 eth3     (the rig ports; `ip -br link` lists them)
set -euo pipefail
[ $# -ge 1 ] || { ip -br link; echo "usage: sudo $0 <rig port> [<rig port>]"; exit 1; }
ip link show br-rig >/dev/null 2>&1 || ip link add br-rig type bridge
for dev in "$@"; do ip link set "$dev" master br-rig; ip link set "$dev" up; done
ip addr replace 192.168.123.10/24 dev br-rig
ip addr replace 192.168.11.10/24 dev br-rig
ip link set br-rig up
sysctl -qw net.ipv4.ip_forward=1
wan=$(ip route show default | awk '{print $5; exit}')
iptables -t nat -C POSTROUTING -s 192.168.123.0/24 -o "$wan" -j MASQUERADE 2>/dev/null \
  || iptables -t nat -A POSTROUTING -s 192.168.123.0/24 -o "$wan" -j MASQUERADE
echo "rig network up on br-rig ($*), NAT out $wan"
