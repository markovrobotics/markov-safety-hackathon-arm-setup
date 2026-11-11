#!/bin/bash
# macOS laptop: bridges every USB Ethernet adapter into bridge1, so the ZED Box, the hand and the arm controller
# work in any adapter. Gives the laptop 192.168.123.10 (box subnet), 192.168.11.10 (the hand's factory subnet,
# hand at 192.168.11.210) and an IPv6 link-local address (the arm controller), and NATs the rig network out
# Wi-Fi (en0) so the box reaches the internet. Run as root; rerunnable (run it again after plugging in an adapter):
#   sudo laptop/net-macos.sh
BR=bridge1; IP=192.168.123.10; LAN=192.168.123.0/24; WAN=en0
ifconfig $BR >/dev/null 2>&1 || ifconfig $BR create
for dev in $(networksetup -listallhardwareports | awk '/^Hardware Port: USB 10\/100\/1000 LAN/{f=1;next} f&&/^Device:/{print $2;f=0}'); do
  ifconfig $BR | grep -q "member: $dev " || ifconfig $BR addm $dev
  ifconfig $dev up
done
ifconfig $BR | grep -q "inet $IP " || ifconfig $BR inet $IP netmask 255.255.255.0
ifconfig $BR | grep -q "inet 192.168.11.10 " || ifconfig $BR alias 192.168.11.10 netmask 255.255.255.0  # Inspire hand factory subnet, hand at 192.168.11.210
ifconfig $BR up
ifconfig $BR | grep -q 'inet6 fe80' || ipconfig set $BR AUTOMATIC-V6
[ "$(sysctl -n net.inet.ip.forwarding)" = 1 ] || sysctl -w net.inet.ip.forwarding=1
pfctl -s nat 2>/dev/null | grep -q 'nat-anchor "com.apple/\*"' || pfctl -q -f /etc/pf.conf
pfctl -a com.apple/kitnat -s nat 2>/dev/null | grep -q "from $LAN" || echo "nat on $WAN inet from $LAN to any -> ($WAN)" | pfctl -q -a com.apple/kitnat -f -
pfctl -s info 2>/dev/null | grep -q 'Status: Enabled' || pfctl -q -e
