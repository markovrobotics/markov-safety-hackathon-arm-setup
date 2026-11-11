"""Inspect Robots embodiment: NERO arm + Inspire hand + two ZED X Nano cameras.

With -E control=pose (the default) the action is the palm's pose then the hand: x, y, z (m) and roll, pitch, yaw (rad,
0 = the palm's orientation when the trial started) in the arm base frame, then 6 hand DOFs (0 closed .. 1 open); the
arm's joints come from inverse kinematics (kinematics.py). With -E control=joint it is 7 arm joint targets (rad) then
the hand. -E sim=true runs it all on a MuJoCo stand-in (sim.py) instead of the rig. Configure the rig with -E, e.g.
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
from inspect_robots_nero.kinematics import Unreachable, check_table, euler, ik, palm, rpy

HAND_LABELS = tuple(f"hand_{d}" for d in DOFS)
LABELS = tuple(f"j{i}" for i in range(1, 8)) + HAND_LABELS
POSE_LABELS = ("x", "y", "z", "roll", "pitch", "yaw") + HAND_LABELS
# The palm's box: walls at the arm's 0.7 m reach, a ceiling it can still reach under, and a floor at the lowest the palm
# can sit with the hand level and clear of the table (kinematics.hand_low, which refuses any pose that brings the
# open hand within CLEARANCE of the table: that check is the floor, this bound only tells the model the range)
WORKSPACE = ((-0.7, 0.7), (-0.7, 0.7), (0.09, 0.8))  # palm x, y, z bounds (m), arm base frame
# Rotations from the trial's start orientation about the base frame's x, y, z axes. Pitch stays clear of +-pi/2, where
# roll and yaw become the same turn.
ROTATION = ((-1.6, 1.6), (-1.3, 1.3), (-1.6, 1.6))
JUMP_RAD = 0.3  # an IK answer that moves a joint more than this in one step is a flip to another arm configuration
CAMERAS = ("wrist", "wrist_right", "wrist_depth", "tripod", "tripod_right", "tripod_depth")  # left eye, right eye, depth
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
fixed view of the table. Each is a stereo camera: "wrist" and "tripod" are its
left eye, "wrist_right" and "tripod_right" its right eye, a few cm to the side.
"wrist_depth" and "tripod_depth" are the depth each camera's left eye sees,
computed from its two eyes: grey, white at 0.1 m, black at 2.0 m and beyond,
linear between (each grey level about 7.5 mm), black also where it has no
depth (too near, occluded, shiny or textureless). The arm's base stands on the table: the table top is
about z = 0.03 in the arm base frame.
eef_pose is the palm's [x, y, z] in metres and [roll, pitch, yaw] in radians in
the arm base frame, then the 6 hand DOFs. The palm point is the middle of the
palm, on the hand's axis. roll, pitch, yaw turn the palm about the base frame's
x, y and z axes, about the palm point, measured from its orientation when the
trial started (0, 0, 0 = as it started).
"""
POSE_DOCS = """You command the palm: x, y, z, roll, pitch, yaw, and the hand DOFs. The arm's
joints follow by inverse kinematics; joint_pos shows where they went. A pose the
arm cannot reach (too far, past a joint limit, needing the arm to flip, or
bringing the open hand within 1 cm of the table) stops the motion where it got
to, and you are told why.
"""
JOINT_DOCS = """You command the arm's joints j1..j7 and the hand DOFs.
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
        control: str = "pose",
        sim: bool = False,
    ):
        if control not in ("pose", "joint"):
            raise ValueError(f"control must be pose or joint, got {control!r}")
        self.can, self.can_interface, self.hand_ip, self.box_url, self.speed = can, can_interface, hand_ip, box_url, speed
        self.pose = control == "pose"
        self.sim = sim if isinstance(sim, bool) else str(sim).lower() in ("1", "true", "yes")
        self._start_rot = np.eye(3)  # the palm's orientation when the trial started: rotation targets are from it
        self._rest = None
        self._frame = None  # camera name -> its latest frame
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
        ranges = (*WORKSPACE, *ROTATION) if self.pose else LIMITS
        low = [lo for lo, _ in ranges] + [0.0] * 6
        high = [hi for _, hi in ranges] + [1.0] * 6
        self.info = EmbodimentInfo(
            name="nero",
            action_space=Box(
                shape=(len(low),),
                low=np.array(low),
                high=np.array(high),
                semantics=ActionSemantics(control_mode="eef_abs_pose", dim_labels=POSE_LABELS) if self.pose
                else ActionSemantics(control_mode="joint_pos", dim_labels=LABELS),
            ),
            observation_space=ObservationSpace(
                cameras=tuple(CameraSpec(c, CAM_H, CAM_W) for c in CAMERAS),
                state=StateSpec(
                    fields=(
                        StateField("joint_pos", (13,), "rad+normalized"),
                        StateField("eef_pose", (12,), "m+rad+normalized"),
                        StateField("hand_force", (6,), "raw"),
                    )
                ),
            ),
            control_hz=control_hz,
            capabilities=frozenset({SELF_PACED}),
            docs=DOCS + (POSE_DOCS if self.pose else JOINT_DOCS),
            morphology="nero-inspire",
        )

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        if self.arm is None:
            if self.sim:
                from inspect_robots_nero.sim import Sim, SimArm, SimHand

                world = Sim()
                self.arm, self.hand, self._frame = SimArm(world), SimHand(world), world.frame
            else:
                self.arm = Arm(self.can, self.speed, self.can_interface)
                self.hand = Hand(self.hand_ip)
                self._frame = lambda c: cameras.frame(self.box_url, c)
            self._watcher = threading.Thread(target=self._watch, daemon=True)
            self._watcher.start()
        self._stop, self._walls, self._still, self._skipping = None, {}, {}, False
        self._instruction = scene.instruction
        self._rest = np.array(self.arm.joints())  # the posture the IK keeps the arm near: how it started the trial
        self._start_rot = palm(self._rest)[:3, :3]
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
            if not self._skipping and self.pose:
                try:
                    arm = self._joints_for(target[:6])
                except Unreachable as e:
                    # Nothing more of this motion is sent: the arm finishes the last reachable step and holds there.
                    at = self._observe_pose()
                    self._stop = (f"the palm stopped short of x={target[0]:.3f} y={target[1]:.3f} z={target[2]:.3f} "
                                  f"roll={target[3]:.2f} pitch={target[4]:.2f} yaw={target[5]:.2f}: {e}. The arm went "
                                  "as far as the motion stayed reachable and is holding there, at "
                                  + " ".join(f"{n}={v:.3f}" for n, v in zip(POSE_LABELS, at)) + ". Pick a nearer "
                                  "point or another orientation, and name all six of x, y, z, roll, pitch, yaw in the "
                                  "next move: one left out keeps the unreachable value it had here")
                    self._skipping = True
            elif not self._skipping:
                arm = target[:7].copy()
            if not self._skipping:
                for j, (at, push) in list(self._walls.items()):
                    if (arm[j] - at) * push > 0:
                        arm[j] = at  # not further into what stopped it
                    elif (at - arm[j]) * push > STALL_STILL_RAD:
                        del self._walls[j]  # commanded the other way: the wall is gone
                self.arm.move(arm)
        if not self._skipping:
            self.hand.move(target[-6:])
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

    def _joints_for(self, pose) -> np.ndarray:
        """The arm joints that put the palm at pose (x, y, z, then roll, pitch, yaw from the trial's start
        orientation), from inverse kinematics started at the last joint target. Raises Unreachable, also for a pose
        that brings the open hand nearer the table than it may go."""
        seed = np.array(self.arm.target if self.arm.target is not None else self.arm.joints())
        goal = np.eye(4)
        goal[:3, :3] = euler(pose[3:]) @ self._start_rot
        goal[:3, 3] = pose[:3]
        check_table(goal, palm(seed))
        q = ik(goal, seed, self._rest)
        jump = np.abs(q - seed)
        if jump.max() > JUMP_RAD:
            j = int(jump.argmax())
            raise Unreachable(f"getting there needs j{j + 1} to swing {jump[j]:.2f} rad at once, a flip to another arm "
                              "configuration")
        return q

    def _observe_pose(self, joints=None) -> list[float]:
        """The palm's x, y, z and roll, pitch, yaw from the trial's start orientation, from the measured joints."""
        here = palm(self.arm.joints() if joints is None else joints)
        return [*here[:3, 3], *rpy(here[:3, :3] @ self._start_rot.T)]

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
                    f"until it is commanded the other way. Something is in the way: "
                    + ("move the palm back the way it came, or around what it is pressing into" if self.pose else
                       f"move j{j + 1} the other way ({'decrease' if push > 0 else 'increase'} it), or move the other "
                       "joints to clear what it is pressing into"))
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
        joints, hand = self.arm.joints(), self.hand.angles()
        return Observation(
            images={c: self._frame(c) for c in CAMERAS},
            state={
                "joint_pos": np.array(joints + hand),
                "eef_pose": np.array([*self._observe_pose(joints), *hand]),
                "hand_force": np.array(self.hand.forces()),
            },
            instruction=self._instruction,
            image_times=dict.fromkeys(CAMERAS, now),
            state_time=now,
        )
