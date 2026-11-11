"""The NERO arm over CAN, through AgileX's pyAgxArm: a USB-CAN adapter on this machine, or the one on the ZED box over Ethernet.

Targets go through move_j, the firmware's position-velocity mode: the controller
smooths each target and keeps its own joint limits.
"""

import fcntl
import math
import os
import subprocess
import threading
import time

import can

from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config
from pyAgxArm.api.agx_arm_factory import ROBOT_JOINT_LIMIT_PRESET

LIMITS = list(ROBOT_JOINT_LIMIT_PRESET[ArmModel.NERO].values())  # rad, per joint (lo, hi); pyAgxArm clamps every move_j to these


def _expect_reply(bus, msg: str) -> None:
    """python-can's socketcand handshake, reading one reply rather than one recv: it compared a whole recv() to the
    expected "< ok >", but once the bus is in raw mode the server streams the arm's frames right behind that ok, and
    one recv often held both ("< ok >< frame 255 ... >"), so the open failed about half the time. Frames read past
    the reply stay in the receive buffer for the bus to parse."""
    buf = bus._SocketCanDaemonBus__receive_buffer
    while ">" not in buf:
        chunk = bus._SocketCanDaemonBus__socket.recv(256).decode("ascii")
        if not chunk:
            raise can.CanError(f"socketcand closed the connection, expected '{msg}'")
        buf += chunk
    end = buf.index(">") + 1
    bus._SocketCanDaemonBus__receive_buffer = buf[end:]
    if buf[:end].strip() != msg:
        raise can.CanError(f"Expected '{msg}' got: '{buf[:end]}'")


# Arm statuses that leave it unable to move until it is reset: emergency stop, joint communication, brake not released
# (J7's latched driver error on 11-11 showed as this), collision, joint status, other error, controller over-temperature.
DEAD_STATUS = {0x01, 0x05, 0x06, 0x07, 0x09, 0x0A, 0x0E, 0x0F}

STICK_LOCK = "/tmp/candlelight.lock"  # held by whichever process has the candleLight stick open

# The CAN frames each reading is built from. pyAgxArm starts a reading at zeros, fills each part as its frame lands and
# returns it once any one part has come, so a read right after connecting gave [0, 0, 0, 0, j5, j6, 0]: a reading
# counts only once all of its parts have arrived.
PARTS = {
    "get_joint_angles": ("joint_12", "joint_34", "joint_56", "joint_7"),
    "get_flange_pose": ("end_pose_xy", "end_pose_zrx", "end_pose_ryrz"),
}


class Arm:
    def __init__(self, channel: str = "can0", speed: int = 50, interface: str = "socketcan", enable: bool = True):
        if interface == "gs_usb":
            # One process at a time on the stick: opening it resets it over USB, so a second opener (a restart while the
            # old jog.py was still exiting, a probe while the bus logger ran) reset it under the first and hung it at
            # 14:37 on 11-10 until a replug. The lock is the kernel's, freed when its holder dies however it dies.
            self._lock = open(STICK_LOCK, "w")
            deadline = time.monotonic() + 10
            while True:
                try:
                    fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() > deadline:
                        raise RuntimeError(f"another process holds the CAN stick ({STICK_LOCK}); not opening it")
                    time.sleep(0.1)
            # macOS: no kernel CAN driver, so python-can opens the candleLight adapter itself through libusb.
            # Loading libusb-package's bundled libusb here makes pyusb reuse it, so no brew libusb is needed.
            import libusb_package
            import usb.core

            libusb_package.get_libusb1_backend()
            # gs_usb detaches a kernel driver before it starts the adapter. libusb on macOS reports one on every
            # device (Apple's composite driver sits on the device, not on the CAN interface), and its detach needs
            # root, so the start failed with "Access denied". Interface 0 has no driver on macOS: skip the detach.
            usb.core.Device.is_kernel_driver_active = lambda self, interface: False
            channel = int(channel)  # gs_usb's channel is the adapter's index among plugged-in adapters
        if interface == "socketcand":
            # The CAN stick is on the ZED box, which serves it over Ethernet with socketcand (box/install-can.sh):
            # channel is <box interface>@<box host>[:port], e.g. nerocan@192.168.123.50. pyAgxArm opens the bus with
            # channel, interface and bitrate only; python-can takes the server's host and port from can.rc.
            import sys

            from can.interfaces.socketcand import socketcand

            channel, _, server = channel.partition("@")
            host, _, port = server.partition(":")
            # tcp_tune (no Nagle, quick ACKs) uses Linux-only socket options; python-can raises on a Mac
            can.rc.update(host=host, port=int(port or 29536), tcp_tune=sys.platform == "linux")
            socketcand.SocketCanDaemonBus._expect_msg = _expect_reply
        cfg = create_agx_arm_config(
            robot=ArmModel.NERO, firmeware_version=NeroFW.DEFAULT, interface=interface, channel=channel
        )
        self.robot = AgxArmFactory.create_arm(cfg)
        self.robot.connect()
        # Firmware before 1.12 (this arm runs 1.11) pushes no feedback after power-up until it is put in normal mode
        # with CAN push on; enable() would send that, but the arm is read before it is enabled. Joints stay unpowered.
        self.robot.set_normal_mode()
        # Nothing more is sent until the arm answers: on a bus where no node acknowledges, the adapter retries a frame
        # forever, and more frames behind it hung the candleLight stick. Closing resets the channel, dropping that frame.
        self._seen: dict[str, tuple[float, float]] = {}  # per reading: its last CAN timestamp, and when it last changed
        try:
            self._read(self.robot.get_joint_angles, within=1.5)
        except RuntimeError:
            self.robot.disconnect()
            raise RuntimeError(f"NERO on {channel} sent no full joint feedback within 1.5 s of normal mode: is it powered?")
        self.channel, self.speed, self.enabled = channel, speed, False
        self.target: list[float] | None = None  # the last joint target sent; None until the first move
        self._send = threading.Lock()  # the embodiment's watcher thread holds the arm from beside the control loop
        if enable:
            self.enable()

    def enable(self) -> None:
        """Power the joints (they hold where they are) and set the speed; until then the arm is only read."""
        deadline = time.monotonic() + 5
        while not self.robot.enable():
            if time.monotonic() > deadline:
                raise RuntimeError(f"NERO on {self.channel} did not enable in 5 s: is it powered and is {self.channel} up?")
            time.sleep(0.01)
        self.robot.set_speed_percent(self.speed)
        self.enabled = True

    def joints(self) -> list[float]:
        """7 joint angles, rad."""
        return self._read(self.robot.get_joint_angles)

    def flange_pose(self) -> list[float]:
        """x, y, z (m), roll, pitch, yaw (rad) of the flange in the base frame."""
        return self._read(self.robot.get_flange_pose)

    def health(self) -> dict:
        """What the controller last said about itself: arm status, lowest driver voltage, J1-7 bus current (A), and any
        driver fault flag set. On 11-10 the arm went quiet mid-move and came back clean after normal mode, with nothing
        read before the cut to say why; jog.py logs this with every reading so the next cut shows its last state."""
        s = self.robot.get_arm_status()
        drivers = [self.robot.get_driver_states(j) for j in range(1, 8)]
        flags = {f"J{j}": _flags(d) for j, d in enumerate(drivers, 1) if d is not None}
        motors = [self.robot.get_motor_states(j) for j in range(1, 8)]
        return {
            "arm_status": str(s.msg.arm_status) if s else None,
            "volts": min((round(d.msg.vol, 1) for d in drivers if d), default=None),
            "amps": [round(d.msg.bus_current, 1) if d else None for d in drivers],  # each driver's supply-side current
            "motor_amps": [round(m.msg.current, 2) if m else None for m in motors],  # the motor's own current
            "torque": [round(m.msg.torque, 2) if m else None for m in motors],  # N·m
            "motor_c": [d.msg.motor_temp if d else None for d in drivers],
            "driver_c": [d.msg.foc_temp if d else None for d in drivers],
            "faults": {j: f for j, f in flags.items() if f},
        }

    def limits(self) -> dict:
        """The limits the controller holds the arm to, read off it: per joint its angle range (deg), top speed (rad/s)
        and acceleration (rad/s²), and its collision protection level (0 off, 1 least sensitive .. 8 most). Each is one
        query frame and its reply; None where the controller did not answer within 1 s. The drive current ceiling is
        not here: it sits on the controller's internal CAN, reached only from a shell on the controller."""
        angle_vel = [self.robot.get_joint_angle_vel_limits(j) for j in range(1, 8)]
        acc = [self.robot.get_joint_acc_limits(j) for j in range(1, 8)]
        crash = self.robot.get_crash_protection_rating()

        out = {"deg": [], "max_speed": [], "max_acc": [], "collision": list(crash.msg) if crash else None}
        for j in range(7):
            a, c = angle_vel[j] and angle_vel[j].msg, acc[j] and acc[j].msg  # each the one joint's message
            ok = a is not None and a.min_angle_limit is not None
            out["deg"].append([round(math.degrees(a.min_angle_limit)), round(math.degrees(a.max_angle_limit))] if ok else None)
            out["max_speed"].append(round(a.max_joint_spd, 2) if ok else None)
            out["max_acc"].append(round(c.max_joint_acc, 2) if c is not None and c.max_joint_acc is not None else None)
        return out

    def move(self, joints: list[float]) -> None:
        with self._send:
            self.target = [float(q) for q in joints]
            self.robot.move_j(self.target)

    def hold(self) -> None:
        """Command the arm where it is now, so nothing keeps pushing it."""
        self.move(self.joints())

    def trouble(self) -> tuple[list[str], bool]:
        """What the controller says is wrong right now, and whether the arm is dead (a joint unpowered, or a status
        that leaves it unable to move until reset). Fresh feedback is read first, so a silent bus raises."""
        self.joints()
        found, dead = [], False
        s = self.robot.get_arm_status()
        if s is not None and int(s.msg.arm_status) != 0:
            name = getattr(s.msg.arm_status, "name", str(s.msg.arm_status))
            found.append(f"arm status {name}")
            dead = dead or int(s.msg.arm_status) in DEAD_STATUS
        for j in range(1, 8):
            d = self.robot.get_driver_states(j)
            if d is None:
                continue
            flags = _flags(d)
            if flags:
                found.append(f"j{j} driver {'+'.join(flags)} (motor {d.msg.motor_temp} C)")
            if self.enabled and not d.msg.foc_status.driver_enable_status:
                found.append(f"j{j} unpowered")
                dead = True
            dead = dead or d.msg.foc_status.driver_error_status
        return found, dead

    def recover(self, controller: str | None) -> list[str]:
        """Bring a dead arm back where it stands: normal mode and enable over CAN first (what brought it back on 11-10),
        then the controller's own reset (cleanError, enable all, CAN control back; controller_reset.py,
        run on the controller over ssh), then enable again. Returns what it did; raises when the arm is still dead."""
        did = []
        for rung in ("can", "controller"):
            if rung == "controller":
                if not controller:
                    break
                r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", controller, "python3 -"],
                                   input=open(os.path.join(os.path.dirname(__file__), "controller_reset.py")).read(),
                                   capture_output=True, text=True, timeout=40)
                out = (r.stdout.strip().splitlines() or [r.stderr.strip()[-200:]])[-1]
                did.append(f"controller reset: {out}")
            with self._send:
                self.robot.set_normal_mode()
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                with self._send:
                    ok = self.robot.enable()
                if ok:
                    break
                time.sleep(0.05)
            with self._send:
                self.robot.set_speed_percent(self.speed)
            time.sleep(0.3)  # a fresh driver state after the enable
            try:
                found, dead = self.trouble()
            except RuntimeError as e:
                found, dead = [str(e)], True
            did.append(f"{rung} re-enable: {'ok' if not dead else '; '.join(found)}")
            if not dead:
                self.hold()
                return did
        raise RuntimeError("the arm is still dead after a reset (" + "; ".join(did) + "): it needs its power cycled")

    def close(self) -> None:
        self.robot.disconnect()

    def _read(self, get, within: float = 1) -> list[float]:
        # pyAgxArm keeps the last message of each kind and updates it in place, so a reading stays there after the arm
        # goes quiet: a reading counts only while its CAN timestamp keeps moving, and only once all its PARTS have come
        parts = PARTS[get.__name__]
        deadline = time.monotonic() + within
        while True:
            whole = all(getattr(self.robot._parser, p, None) is not None for p in parts)
            m = get() if whole else None
            if m is not None:
                last = self._seen.get(get.__name__)
                if last is None or last[0] != m.timestamp:
                    self._seen[get.__name__] = (m.timestamp, time.monotonic())
                    return list(m.msg)
                if time.monotonic() - last[1] < 1:
                    return list(m.msg)
            if time.monotonic() > deadline:
                raise RuntimeError(f"no full NERO feedback on CAN for {within:g} s")
            time.sleep(0.005)


def _flags(driver) -> list[str]:
    """The fault flags a joint driver has set (its enable bit is state, not a fault)."""
    return [k for k, v in vars(driver.msg.foc_status).items() if v and k != "driver_enable_status"]
