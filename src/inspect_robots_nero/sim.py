"""A MuJoCo stand-in for the rig: the NERO's joints from AgileX's URDF (as in kinematics.py) drawn as capsules, a
box-and-finger hand at the palm, a table at the arm base's height and a few objects on it. Arm and Hand here answer
what the real ones do, so the embodiment runs unchanged on it (-E sim=true): joints follow their targets at a capped
speed, nothing stalls or faults, and the cameras are rendered.
"""

import threading
import time

import mujoco
import numpy as np

from inspect_robots_nero.kinematics import _FLANGE, _JOINTS, LIMITS, PALM_Z

JOINT_SPEED = 1.0  # rad/s each joint moves toward its target
HAND_SPEED = 2.0  # of the 0..1 range per second
CAM_H, CAM_W = 360, 640

FINGERS = (  # name, y across the palm (m), the hand DOF that curls it
    ("pinky", -0.03, 0), ("ring", -0.01, 1), ("middle", 0.01, 2), ("index", 0.03, 3),
)


def _xml() -> str:
    bodies, close = [], []
    for i, ((x, y, z), (r, p, w)) in enumerate(_JOINTS):
        # the link drawn from this joint's origin back to its parent's, as a capsule in the parent frame
        if i and np.linalg.norm((x, y, z)) > 1e-6:
            bodies.append(f'<geom type="capsule" fromto="0 0 0 {x} {y} {z}" size="0.04" rgba=".85 .85 .88 1"/>')
        lo, hi = LIMITS[i]
        bodies.append(f'<body name="link{i + 1}" pos="{x} {y} {z}" euler="{r} {p} {w}">'
                      f'<joint name="j{i + 1}" axis="0 0 1" range="{lo} {hi}" limited="true"/>'
                      f'<geom type="cylinder" size="0.045 0.04" rgba=".3 .3 .33 1"/>')
        close.append("</body>")
    fz = np.arctan2(_FLANGE[1, 0], _FLANGE[0, 0])
    fingers = "".join(
        f'<body name="{n}" pos="0.01 {yy} 0.141"><joint name="{n}" axis="0 1 0" range="0 1.5"/>'
        f'<geom type="capsule" fromto="0 0 0 0 0 0.07" size="0.008" rgba=".15 .15 .15 1"/></body>'
        for n, yy, _ in FINGERS
    )
    hand = (f'<body name="flange" euler="0 0 {fz}">'
            f'<site name="palm" pos="0 0 {PALM_Z}"/>'
            '<geom type="cylinder" pos="0 0 0.01" size="0.026 0.01" rgba=".1 .1 .1 1"/>'
            f'<body name="hand" pos="0 0 0.020">'
            '<geom type="box" pos="0 0 0.07" size="0.025 0.045 0.07" rgba=".2 .2 .22 1"/>'
            f'{fingers}'
            '<body name="thumb" pos="0.02 0.045 0.07"><joint name="thumb_bend" axis="1 0 0" range="0 1"/>'
            '<joint name="thumb_rotation" axis="0 0 1" range="0 1.6"/>'
            '<geom type="capsule" fromto="0 0 0 0 0.03 0.04" size="0.009" rgba=".15 .15 .15 1"/></body>'
            '<camera name="wrist" pos="0.06 0 0.03" xyaxes="0 -1 0 -1 0 -0.15" fovy="80"/>'
            # each right eye 5 cm along its image's x axis: a stand-in for the ZED's baseline, not a measured one
            '<camera name="wrist_right" pos="0.06 -0.05 0.03" xyaxes="0 -1 0 -1 0 -0.15" fovy="80"/>'
            '</body></body>')
    return f"""
<mujoco model="nero_sim">
  <compiler angle="radian" eulerseq="XYZ"/>
  <visual><global offwidth="{CAM_W}" offheight="{CAM_H}"/></visual>
  <asset><texture name="grid" type="2d" builtin="checker" rgb1=".55 .45 .35" rgb2=".5 .4 .3" width="512" height="512"/>
    <material name="table" texture="grid" texrepeat="4 4"/></asset>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1" diffuse=".9 .9 .9"/>
    <geom name="table" type="box" pos="0 0 -0.02" size="0.9 0.9 0.02" material="table"/>
    <geom name="paper" type="box" pos="-0.35 -0.30 0.001" size="0.105 0.075 0.001" rgba=".95 .95 .95 1"/>
    <body name="bottle" pos="-0.40 0.10 0.05"><freejoint/><geom type="cylinder" size="0.025 0.05" rgba=".9 .5 .1 1"/></body>
    <body name="block" pos="-0.30 0.25 0.025"><freejoint/><geom type="box" size="0.025 0.025 0.025" rgba=".2 .4 .9 1"/></body>
    <camera name="tripod" pos="-1.1 0.0 0.75" xyaxes="0 -1 0 0.65 0 0.8" fovy="65"/>
    <camera name="tripod_right" pos="-1.1 -0.05 0.75" xyaxes="0 -1 0 0.65 0 0.8" fovy="65"/>
    <geom type="cylinder" pos="0 0 0.069" size="0.06 0.069" rgba=".3 .3 .33 1"/>
    {"".join(bodies)}{hand}{"".join(close)}
  </worldbody>
</mujoco>"""


class Sim:
    """One MuJoCo world shared by the sim arm, hand and cameras. Joints move kinematically: each read advances them
    toward their targets by the time since the last read."""

    def __init__(self, start=(0.0, 0.42, 0.0, 1.72, 0.0, 0.0, 0.0)):
        self.model = mujoco.MjModel.from_xml_string(_xml())
        self.data = mujoco.MjData(self.model)
        self._arm = [self.model.joint(f"j{i}").qposadr[0] for i in range(1, 8)]
        names = [n for n, _, _ in FINGERS] + ["thumb_bend", "thumb_rotation"]
        self._hand = [self.model.joint(n).qposadr[0] for n in names]
        self._hand_range = [self.model.joint(n).range[1] for n in names]
        self.q = np.array(start, dtype=float)
        self.q_target = self.q.copy()
        self.h = np.ones(6)
        self.h_target = self.h.copy()
        self._t = time.monotonic()
        self._lock = threading.Lock()
        self._renderers: dict[str, mujoco.Renderer] = {}
        self._advance()

    def _advance(self) -> None:
        with self._lock:
            now = time.monotonic()
            dt, self._t = now - self._t, now
            self.q += np.clip(self.q_target - self.q, -JOINT_SPEED * dt, JOINT_SPEED * dt)
            self.h += np.clip(self.h_target - self.h, -HAND_SPEED * dt, HAND_SPEED * dt)
            self.data.qpos[self._arm] = self.q
            # hand DOFs: 1 open, 0 closed; thumb rotation 0 tucked, 1 across the palm
            curl = [(1 - v) * r for v, r in zip(self.h[:5], self._hand_range[:5])] + [self.h[5] * self._hand_range[5]]
            self.data.qpos[self._hand] = curl
            mujoco.mj_forward(self.model, self.data)

    def frame(self, camera: str) -> np.ndarray:
        self._advance()
        with self._lock:
            r = self._renderers.get(camera)
            if r is None:
                r = self._renderers[camera] = mujoco.Renderer(self.model, CAM_H, CAM_W)
            r.update_scene(self.data, camera=camera)
            return r.render().copy()


class SimArm:
    """What Arm offers the embodiment, on the sim."""

    def __init__(self, sim: Sim):
        self.sim, self.enabled, self.target = sim, True, None

    def joints(self) -> list[float]:
        self.sim._advance()
        return self.sim.q.tolist()

    def move(self, joints) -> None:
        self.target = [float(q) for q in joints]
        self.sim.q_target = np.clip(self.target, LIMITS[:, 0], LIMITS[:, 1])

    def hold(self) -> None:
        self.move(self.joints())

    def trouble(self) -> tuple[list[str], bool]:
        return [], False

    def recover(self, controller) -> list[str]:
        return ["sim: nothing to reset"]

    def close(self) -> None:
        pass


class SimHand:
    """What Hand offers the embodiment, on the sim."""

    def __init__(self, sim: Sim):
        self.sim = sim

    def angles(self) -> list[float]:
        self.sim._advance()
        return self.sim.h.tolist()

    def forces(self) -> list[float]:
        return [0.0] * 6

    def move(self, angles) -> None:
        self.sim.h_target = np.clip(np.asarray(angles, dtype=float), 0, 1)

    def close(self) -> None:
        pass
