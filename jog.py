"""Jog NERO joints one at a time from stdin; print what the arm reads as JSON lines on stdout.

    uv run jog.py [can=nerocan@192.168.123.50] [can_interface=socketcand] [speed=20]

Each input line is "<joint 1-7> <degrees>", e.g. "3 -5": that joint's target moves by that much and the arm goes there
through move_j (the firmware's own smoothing and limits). Targets start at where the arm is when a jog comes after
a pause of over a second (taps faster than that add up), clamped to the limits pyAgxArm holds every move to
(a joint past one goes back to it on the first jog);
the arm is only read until then, and enabled (holding where it is) by that first jog.
"hold <joint> <+1|-1>", repeated while a button is held, moves the joint on until "stop <joint>";
"hand open|fist|point|closed" moves the Inspire hand toward that pose and "hand stop" holds it where it is
(hand_ip=192.168.11.210). Output, about 5 times a second
while anything changes: {"deg": [7 joint angles], "target": [7 or null], "enabled": bool, "error": str or null,
"health": Arm.health(), "limits": Arm.limits()}.
"""

import atexit
import json
import math
import select
import signal
import sys
import time

# A kill must still close the adapter: its stop resets the CAN channel, which drops a frame the bus never acknowledged.
# Killed without it, the candleLight stick kept retrying that frame and stopped answering USB until it lost power.
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
signal.signal(signal.SIGHUP, lambda *_: sys.exit(0))

from inspect_robots_nero.arm import LIMITS, Arm
from inspect_robots_nero.hand import Hand


opts = dict(a.split("=", 1) for a in sys.argv[1:])
arm = Arm(opts.get("can", "nerocan@192.168.123.50"), int(opts.get("speed", 20)), opts.get("can_interface", "socketcand"), enable=False)
atexit.register(arm.close)  # every exit closes the adapter: the end of stdin, an error, SIGTERM and SIGHUP alike
RESEED_AFTER = 1.0  # s without a tap after which the next one starts from the arm's measured pose
HOLD_LEAD = math.radians(8)  # how far ahead of a held joint its target runs
HAND_SPEED = 400  # the Inspire hand's own motor speed, 0..1000: fist to open in about a second
# The hand's 6 DOFs (pinky, ring, middle, index, thumb bend, thumb rotation), 1 open, 0 closed. Closed fingers stop at
# 0.1 and the thumb bend at 0.4, so the thumb folds over the fingers rather than into them.
HAND_POSES = {
    "open": [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
    "fist": [0.1, 0.1, 0.1, 0.1, 0.4, 1.0],
    "point": [0.1, 0.1, 0.1, 1.0, 0.4, 1.0],
    # Fist with only the thumb's rotation changed: it turns in under
    # the fingers instead of standing up beside them; the index closes onto it and stops there (about 0.46).
    "closed": [0.1, 0.1, 0.1, 0.1, 0.4, 0.0],
}
hand: Hand | None = None  # opened on the first hand command, so a hand that is off never stops the arm jog
target: list[float] | None = None
last_tap = 0.0
error: str | None = None
last = None
limits: dict | str | None = None  # what the controller holds the arm to (Arm.limits), read at start and on each enable


def read_limits() -> None:
    global limits
    try:
        limits = arm.limits()
    except Exception as e:  # a limit read never stops the jog
        limits = f"unreadable: {e}"


def emit() -> None:
    global last
    try:
        deg = [round(math.degrees(q), 1) for q in arm.joints()]
    except RuntimeError as e:
        deg = None
        out = {"deg": None, "target": None, "enabled": arm.enabled, "error": "; ".join(x for x in (str(e), error) if x)}
    else:
        out = {
            "deg": deg,
            "target": [round(math.degrees(q), 1) for q in target] if target else None,
            "enabled": arm.enabled,
            "error": error,
        }
    try:
        out["health"] = arm.health()
    except Exception as e:  # a status read never stops the jog
        out["health"] = f"unreadable: {e}"
    out["limits"] = limits
    if out != last:
        print(json.dumps(out), flush=True)
        last = out


def base() -> list[float]:
    """The target this command starts from: the last one while taps keep coming, else where the arm is (see RESEED_AFTER)."""
    global last_tap
    t = target if target is not None and time.monotonic() - last_tap <= RESEED_AFTER else arm.joints()
    last_tap = time.monotonic()
    return [min(max(q, lo), hi) for q, (lo, hi) in zip(t, LIMITS)]  # held at a limit, not wound past it


def joint_index(joint: str) -> int:
    i = int(joint) - 1
    if not 0 <= i < 7:
        raise ValueError(f"joint {joint} is not 1-7")
    return i


read_limits()
while True:
    ready, _, _ = select.select([sys.stdin], [], [], 0.2)
    if ready:
        line = sys.stdin.readline()
        if not line:
            break
        try:
            words = line.split()
            if words[0] == "hand":
                # Fist, open and pointer at the hand's own speed (HAND_SPEED), so it moves smoothly between them; the
                # panel sends the pose when its button is pressed and "hand stop" when it is let go, which holds the
                # hand where it has got to.
                if hand is None:
                    hand = Hand(opts.get("hand_ip", "192.168.11.210"), speed=HAND_SPEED)
                hand.move(hand.angles() if words[1] == "stop" else HAND_POSES[words[1]])
            else:
                if not arm.enabled:
                    arm.enable()
                    read_limits()  # enabling can follow a controller reset, which may bring other limits
                if words[0] == "hold":
                    # Sent about 7 times a second while a button is held: the joint's target stays HOLD_LEAD ahead of
                    # where it is, so it moves at the firmware's own smoothed speed, and stops at most HOLD_LEAD past
                    # the last tick if the release never comes.
                    i, direction = joint_index(words[1]), float(words[2])
                    target = base()
                    target[i] = arm.joints()[i] + math.copysign(HOLD_LEAD, direction)
                elif words[0] == "stop":
                    i = joint_index(words[1])  # released: the joint stops where it is
                    target = base()
                    target[i] = arm.joints()[i]
                else:
                    i, d = joint_index(words[0]), math.radians(float(words[1]))
                    target = base()
                    target[i] += d
                target[i] = min(max(target[i], LIMITS[i][0]), LIMITS[i][1])
                arm.move(target)
            error = None
        except Exception as e:  # a bad line or an arm that stopped answering is shown, and the next jog tries again
            error = f"{type(e).__name__}: {e}"
    emit()
