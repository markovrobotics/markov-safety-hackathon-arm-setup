"""Inspect Robots embodiment: NERO arm + Inspire hand + two ZED X Nano cameras.

Action and joint_pos state are the same 13 numbers: 7 arm joint targets (rad)
then 6 hand DOFs (0 closed .. 1 open). Configure with -E, e.g.
-E can=nerocan@192.168.123.50 -E can_interface=socketcand (the CAN stick on the ZED box, served over Ethernet;
-E can=can0 -E can_interface=socketcan with it on a Linux laptop, -E can=0 -E can_interface=gs_usb on a Mac) -E hand_ip=192.168.11.210 -E box_url=http://192.168.123.50:8080
"""

import sys
import threading
import time

import numpy as np
from inspect_robots.embodiment import SELF_PACED, EmbodimentInfo
from inspect_robots.scene import Scene
from inspect_robots.spaces import ActionSemantics, Box, CameraSpec, ObservationSpace, StateField, StateSpec
from inspect_robots.types import Action, Observation, StepResult

from inspect_robots_nero import cameras
from inspect_robots_nero.arm import LIMITS, Arm
from inspect_robots_nero.hand import DOFS, Hand

LABELS = tuple(f"j{i}" for i in range(1, 8)) + tuple(f"hand_{d}" for d in DOFS)
CAMERAS = ("wrist", "tripod")
CAM_H, CAM_W = 360, 640  # what box/camera_server.py serves
# The stall stop: a joint that cannot reach its target is stopped here and the model is told which way to go. On 11-11 the agent
# pressed the open hand into the table and kept asking for more; the motors sat stalled at the 7 A cap until J7's driver
# latched an error and the arm died. A joint more than STALL_RAD from its target that has not moved STALL_STILL_RAD in
# STALL_S is stalled: the arm is held where it is, the rest of the move is dropped, and the model is told which joint and
# which way. Normal tracking lag at the agent's speeds is under 0.02 rad and a joint at 0.18 rad/s (its slowest move)
# covers STALL_STILL_RAD in 60 ms, so a moving joint never reads as stalled.
STALL_RAD, STALL_STILL_RAD, STALL_S = 0.05, 0.01, 0.5
# A joint re-enabled by a reset first moved 2.0 s and 0.67 s after its next command (11-11, j7, two resets); judged
# at STALL_S it read as a stall it wasn't. No stall is called this long after a reset.
RESET_GRACE_S = 3.0
BACKOFF_RAD = 0.05  # how far a stalled joint is backed off its stall: 2.9 deg, past the ~1 deg the j7 stall sprang back
WATCH_S = 0.02  # the watcher's period: it runs beside the control loop, so it also catches a push while the model thinks
# The NERO controller for the arm reset, by its IPv6 link-local address on the rig bridge that laptop/net-macos.sh
# (bridge1) or laptop/net-linux.sh (br-rig) makes: the controller has no IPv4 address on the rig network.
CONTROLLER = f"agx@fe80::6ce8:14ff:fe96:c66%{'bridge1' if sys.platform == 'darwin' else 'br-rig'}"

DOCS = """\
Right-arm rig: AgileX NERO 7-DOF arm with an Inspire 6-DOF right hand on its flange.
j1..j7 are the arm joints in radians, base to wrist. hand_* are the hand's DOFs:
1.0 is open/extended, 0.0 closed/curled; hand_thumb_rotation 0 is the thumb
tucked beside the palm, 1 swung across to oppose the fingers.
Cameras: "wrist" rides on the flange looking along the fingers; "tripod" is a
fixed view of the table. eef_pose is the flange [x, y, z] in metres and
[roll, pitch, yaw] in radians, in the arm base frame.
"""


class NeroEmbodiment:
    def __init__(
        self,
        can: str = "nerocan@192.168.123.50",
        can_interface: str = "socketcand",
        hand_ip: str = "192.168.11.210",
        box_url: str = "http://192.168.123.50:8080",
        control_hz: float = 20.0,
        speed: int = 50,
        controller: str = CONTROLLER,
    ):
        self.can, self.can_interface, self.hand_ip, self.box_url, self.speed = can, can_interface, hand_ip, box_url, speed
        self.controller = controller or None
        self._watcher: threading.Thread | None = None
        self._stop: str | None = None  # why the watcher stopped the arm, until the model has been told
        self._alive_at: list[float] | None = None
        self._grace_until = 0.0  # no stall verdicts before this: a re-enabled joint is slow to start
        self._dead: str | None = None  # why the arm died, until step() has reset it
        self._walls: dict[int, tuple[float, float]] = {}  # joint -> (where it stalled, the direction it was pushing)
        self._still: dict[int, tuple[float, float]] = {}  # joint -> (position, since when) while it is off its target
        self._seen_trouble: tuple[str, ...] = ()
        self._skipping = False
        self._lock = threading.Lock()  # a stop and the control loop's next command never interleave
        self.arm: Arm | None = None
        self.hand: Hand | None = None
        self._instruction: str | None = None
        self._t = 0.0
        # The action box is the range each joint can reach: the limits pyAgxArm clamps every move to, which match
        # what the controller reports (Arm.limits(), read 11-11). AgileX's spec sheet ranges are wider (j2 -15..190 deg).
        low = [lo for lo, _ in LIMITS] + [0.0] * 6
        high = [hi for _, hi in LIMITS] + [1.0] * 6
        self.info = EmbodimentInfo(
            name="nero",
            action_space=Box(
                shape=(13,),
                low=np.array(low),
                high=np.array(high),
                semantics=ActionSemantics(control_mode="joint_pos", dim_labels=LABELS),
            ),
            observation_space=ObservationSpace(
                cameras=tuple(CameraSpec(c, CAM_H, CAM_W) for c in CAMERAS),
                state=StateSpec(
                    fields=(
                        StateField("joint_pos", (13,), "rad+normalized"),
                        StateField("eef_pose", (6,), "m+rpy"),
                        StateField("hand_force", (6,), "raw"),
                    )
                ),
            ),
            control_hz=control_hz,
            capabilities=frozenset({SELF_PACED}),
            docs=DOCS,
            morphology="nero-inspire",
        )

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        if self.arm is None:
            self.arm = Arm(self.can, self.speed, self.can_interface)
            self.hand = Hand(self.hand_ip)
            self._watcher = threading.Thread(target=self._watch, daemon=True)
            self._watcher.start()
        self._stop, self._walls, self._still, self._skipping = None, {}, {}, False
        self._instruction = scene.instruction
        self._t = time.monotonic()
        return self._observe()

    def step(self, action: Action) -> StepResult:
        target = np.asarray(action.data, dtype=float)
        final = bool(action.meta.get("chunk_final"))
        if self._dead is not None:
            self._recover()
        with self._lock:
            if self._stop is not None:
                # The watcher stopped the arm: the rest of this motion is dropped (arm and hand), without its pacing,
                # and its last step's observation carries why, as a replan note the agent reads after the motion. The
                # agent marks every motion's last action chunk_final.
                self._skipping = True
            if not self._skipping:
                arm = target[:7].copy()
                for j, (at, push) in list(self._walls.items()):
                    if (arm[j] - at) * push > 0:
                        arm[j] = at  # not further into what stopped it
                    elif (at - arm[j]) * push > STALL_STILL_RAD:
                        del self._walls[j]  # commanded the other way: the wall is gone
                self.arm.move(arm)
        if not self._skipping:
            self.hand.move(target[7:])
            time.sleep(max(0.0, 1 / self.info.control_hz - (time.monotonic() - self._t)))
        self._t = time.monotonic()
        observation = self._observe()
        if self._skipping and final:
            reason, self._stop, self._skipping = self._stop, None, False
            observation = Observation(
                images=observation.images, state=observation.state, instruction=observation.instruction,
                image_times=observation.image_times, state_time=observation.state_time,
                extra={**observation.extra, "replan": {"trigger": "arm_stop", "reason": reason}},
            )
        return StepResult(observation=observation)

    def _watch(self) -> None:
        """Beside the control loop, every WATCH_S: hold the arm the moment a joint stalls or a driver flags trouble,
        and mark it dead when it can no longer move. Runs while the model thinks too, when nothing else would see a
        push."""
        while self.arm is not None:
            try:
                if self._dead is None:
                    self._check()
            except Exception as e:  # no feedback: the arm went quiet
                self._dead = self._dead or f"the arm stopped answering on CAN ({e})"
            time.sleep(WATCH_S)

    def _check(self) -> None:
        arm = self.arm
        here = arm.joints()
        found, dead = arm.trouble()
        if dead:
            self._dead = "; ".join(found)
            return
        self._alive_at = here  # the last pose read while powered: a dead joint can swing away from it (j7 on the hand's cable)
        fresh = tuple(found)
        if fresh and fresh != self._seen_trouble:
            with self._lock:
                arm.hold()
                self._stop = (f"the arm's drivers reported {'; '.join(found)}. The arm stopped and is holding where it is. "
                              "This comes before a fault that kills the arm: back off from whatever it is pressing into")
        self._seen_trouble = fresh
        target = arm.target
        now = time.monotonic()
        if target is None or now < self._grace_until:
            return
        for j in range(7):
            off = target[j] - here[j]
            if abs(off) <= STALL_RAD or j in self._walls:
                self._still.pop(j, None)
                continue
            at, since = self._still.get(j, (here[j], now))
            if abs(here[j] - at) > STALL_STILL_RAD:
                at, since = here[j], now
            self._still[j] = (at, since)
            if now - since < STALL_S:
                continue
            with self._lock:
                push = 1.0 if off > 0 else -1.0
                here = arm.joints()
                # Holding the pose it stalled at kept it pressing: on the arm (11-11, j7 into the hand's cable at 82.7 deg)
                # the joint sat at the 7 A drive cap and 11.5 Nm through the hold. Backing it off the stall unloads it.
                lo, hi = LIMITS[j]
                held = list(here)
                held[j] = min(max(here[j] - push * BACKOFF_RAD, lo), hi)
                arm.move(held)
                self._walls[j] = (held[j], push)
                self._still.clear()
                where = "at its joint limit" if min(abs(here[j] - lo), abs(here[j] - hi)) < STALL_RAD else "blocked"
                self._stop = (
                    f"j{j + 1} stalled {where}: it stopped at {here[j]:.3f} rad, {abs(target[j] - here[j]):.3f} rad short "
                    f"of its target {target[j]:.3f}, and did not move for {STALL_S:g} s. The arm backed j{j + 1} off to "
                    f"{held[j]:.3f} and is holding there, and j{j + 1} "
                    f"will not be driven past {held[j]:.3f} in the {'increasing' if push > 0 else 'decreasing'} direction "
                    f"until you command it the other way. Something is in the way: move j{j + 1} the other way "
                    f"({'decrease' if push > 0 else 'increase'} it), or move the other joints to clear what it is pressing into")
            return

    def _recover(self) -> None:
        """Reset a dead arm in place and tell the model; a reset that leaves it dead ends the trial with the arm's
        fault."""
        why = self._dead
        did = self.arm.recover(self.controller)  # raises when still dead: the rollout ends the trial as a fault
        here = self.arm.joints()
        # Unpowered on 11-11, j7 swung from 13 to 66 deg on the hand's cable and was held at 57 deg after the reset.
        moved = [f"j{j + 1} {a:.3f} -> {b:.3f} rad" for j, (a, b) in enumerate(zip(self._alive_at or here, here))
                 if abs(b - a) > STALL_RAD]
        self._walls, self._still, self._seen_trouble, self._dead = {}, {}, (), None
        self._grace_until = time.monotonic() + RESET_GRACE_S
        self._stop = (f"the arm died ({why}) and was reset ({'; '.join(did)}). It is powered again and holding where it "
                      "is now" + (f"; while unpowered it moved: {', '.join(moved)}" if moved else "") +
                      ". Plan from its current joints, and do not repeat the motion that killed it")

    def connect_operator_session(self, session) -> None:
        """Nothing here reads stdin; the framework console owns the episode end."""

    def close(self) -> None:
        arm, self.arm = self.arm, None  # stops the watcher
        if self._watcher is not None:
            self._watcher.join(timeout=1)
            self._watcher = None
        self.arm = arm
        for part in (self.arm, self.hand):
            if part is not None:
                part.close()
        self.arm = self.hand = None

    def _observe(self) -> Observation:
        now = time.monotonic()
        return Observation(
            images={c: cameras.frame(self.box_url, c) for c in CAMERAS},
            state={
                "joint_pos": np.array(self.arm.joints() + self.hand.angles()),
                "eef_pose": np.array(self.arm.flange_pose()),
                "hand_force": np.array(self.hand.forces()),
            },
            instruction=self._instruction,
            image_times=dict.fromkeys(CAMERAS, now),
            state_time=now,
        )
