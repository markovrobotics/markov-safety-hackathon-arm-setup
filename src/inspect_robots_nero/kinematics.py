"""NERO forward and inverse kinematics for the palm, from AgileX's URDF.

The chain is AgileX's nero_description.urdf (agx_arm_urdf, MIT). Its link7 origin is where the firmware's
get_flange_pose puts the flange: over 266 poses read off the arm on 11-11 the two agree to 0.07 mm (median, 4.3 mm
worst, reading skew while moving), and the firmware's frame is link7 turned half a turn about its z, which then agrees
to 0.19 deg (median).

The palm point sits on the flange axis (link7 z, out of the flange face), PALM_Z out from the face: the AgileX flange
piece's face 12 mm out (flange.stl), the printed mount's 8 mm body to the hand's wrist face (mount_spigot37p8_screw5p0:
its spigot seat at 20 mm), and the middle of the palm 85 mm along the hand from the wrist face (the Inspire base link
runs 0 to 141 mm to the knuckles, dex-urdf inspire_hand_right).
"""

import numpy as np
from pyAgxArm import ArmModel
from pyAgxArm.api.agx_arm_factory import ROBOT_JOINT_LIMIT_PRESET

LIMITS = np.array(list(ROBOT_JOINT_LIMIT_PRESET[ArmModel.NERO].values()))  # rad, per joint (lo, hi): what pyAgxArm clamps to
# SPEC-AMBIGUITY: (2026-11-11 15:20 Kaya asked for "that better mode, especially given that we've already worked out
# some of the things from the urdf" and the node for "targeting the hand's palm" but this could mean
# 1. a point on the hand's axis in the middle of the palm, inside the hand
# 2. a point on the palm's surface, 25 mm off the axis toward the palm side
# 3. the fingertips' grasp centre
# Choice defaulted to: 1 because it is the same point however the hand is clocked on the flange, which no file gives
# with certainty, and rotations then turn the hand about its own middle
PALM_Z = 0.012 + 0.008 + 0.085  # m out from the flange face along its axis

_HALF = np.pi / 2


def euler(rpy) -> np.ndarray:
    """Rotation matrix of roll, pitch, yaw about the fixed x, y, z axes in turn (Rz @ Ry @ Rx)."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def rpy(m) -> np.ndarray:
    """Roll, pitch, yaw of a rotation matrix, the inverse of euler() for pitch inside (-pi/2, pi/2)."""
    return np.array([np.arctan2(m[2, 1], m[2, 2]), np.arcsin(np.clip(-m[2, 0], -1, 1)), np.arctan2(m[1, 0], m[0, 0])])


def rotvec(m) -> np.ndarray:
    """Axis times angle of a rotation matrix."""
    angle = np.arccos(np.clip((np.trace(m) - 1) / 2, -1, 1))
    axis = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]])
    if angle < 1e-9:
        return axis / 2
    if np.pi - angle < 1e-6:  # half a turn: the axis is the column of m + I with the largest norm
        b = m + np.eye(3)
        v = b[:, np.argmax(np.linalg.norm(b, axis=0))]
        return v / np.linalg.norm(v) * angle
    return axis / (2 * np.sin(angle)) * angle
# (xyz, rpy) of each joint's origin in its parent link, base to link7; every joint turns about its own z
_JOINTS = (
    ((0, 0, 0.138), (0, 0, 0)),
    ((0, 0, 0), (_HALF, np.pi, 0)),
    ((0, -0.31, 0), (-_HALF, 0, np.pi)),
    ((0, 0, 0), (-_HALF, 0, np.pi)),
    ((0, -0.27001, 0), (_HALF, -_HALF, 0)),
    ((0, 0, 0), (_HALF, -_HALF, 0)),
    ((0, -0.0235, 0), (_HALF, 0, 0)),
)


def _transform(xyz, rpy) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = euler(rpy)
    m[:3, 3] = xyz
    return m


_ORIGINS = [_transform(xyz, rpy) for xyz, rpy in _JOINTS]
_FLANGE = _transform((0, 0, 0), (0, 0, np.pi))  # link7 -> the firmware's flange frame
_PALM = _FLANGE @ _transform((0, 0, PALM_Z), (0, 0, 0))


# SPEC-AMBIGUITY: (2026-11-11 15:20 Kaya asked for the mode "bounded" by a workspace (node n039011cbfb) but where the
# table is could be
# 1. z = 0, the base's own bottom
# 2. a little above it, as the 11:36 hand-on-table reading gives (flange z 0.0595, flange level, the hand's side on
#    the table: 26-41 mm from the axis to the hand's surface)
# Choice defaulted to: 2, 0.03 m, from that reading; nobody has measured it
TABLE_Z = 0.03
# The hand as a cylinder about the palm frame's z (the flange axis): from the flange face to the open fingertips, and
# wide enough for the palm's 93 mm span (Inspire drawing) and the mount
HAND_BACK, HAND_FRONT, HAND_RADIUS = -PALM_Z, 0.125, 0.047
CLEARANCE = 0.01


def hand_low(pose: np.ndarray) -> float:
    """The lowest z any part of the hand (fingers open) reaches with the palm at pose (4x4)."""
    axis_z = pose[2, 2]
    ends = min(axis_z * HAND_BACK, axis_z * HAND_FRONT)
    return pose[2, 3] + ends - HAND_RADIUS * np.sqrt(max(0.0, 1 - axis_z**2))


def check_table(target: np.ndarray, here: np.ndarray) -> None:
    """Raise Unreachable when the palm at target (4x4) brings the open hand within CLEARANCE of the table and nearer it
    than at here: a hand already that low may always move up out of it."""
    low = hand_low(target)
    if low < TABLE_Z + CLEARANCE and low < hand_low(here) - 1e-4:
        raise Unreachable(f"the hand would go {(TABLE_Z + CLEARANCE - low) * 100:.1f} cm too low: with the fingers "
                          f"open its lowest point would be {(low - TABLE_Z) * 100:.1f} cm above the table, and it keeps "
                          f"{CLEARANCE * 100:.0f} cm. Raise the palm, or turn the fingers away from the table")


class Unreachable(Exception):
    """The arm cannot put the palm at a pose: the message says why, for the model."""


def _chain(q) -> tuple[np.ndarray, list[np.ndarray]]:
    """link7 in the base frame, and each joint's frame (its z is the joint axis)."""
    m, frames = np.eye(4), []
    for origin, qi in zip(_ORIGINS, q):
        m = m @ origin
        frames.append(m)
        c, s = np.cos(qi), np.sin(qi)
        m = m @ np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    return m, frames


def flange(q) -> np.ndarray:
    """The firmware's flange frame in the base frame, 4x4."""
    return _chain(q)[0] @ _FLANGE


def palm(q) -> np.ndarray:
    """The palm frame in the base frame, 4x4: the flange frame moved PALM_Z out along its axis."""
    return _chain(q)[0] @ _PALM


def ik(target: np.ndarray, seed, rest=None, *, iters: int = 100, pos_tol: float = 1e-3, rot_tol: float = np.radians(0.5)) -> np.ndarray:
    """Joints that put the palm at target (4x4), starting from seed and staying near it: damped least squares on the
    palm's position and orientation, inside the joint limits. The arm has one joint more than a pose needs: that
    spare freedom (the elbow swinging about the shoulder-wrist line) is pulled toward rest, when given, so the arm
    keeps its posture rather than drifting over many moves. Raises Unreachable when it cannot get within the
    tolerances."""
    q = np.clip(np.asarray(seed, dtype=float), LIMITS[:, 0], LIMITS[:, 1])
    for it in range(2 * iters):  # the second half without the posture pull, which can hold off the last millimetre
        link7, frames = _chain(q)
        here = link7 @ _PALM
        dp = target[:3, 3] - here[:3, 3]
        dr = rotvec(target[:3, :3] @ here[:3, :3].T)
        if np.linalg.norm(dp) < pos_tol and np.linalg.norm(dr) < rot_tol:
            return q
        jac = np.zeros((6, 7))
        for j, f in enumerate(frames):
            z = f[:3, 2]
            jac[:3, j] = np.cross(z, here[:3, 3] - f[:3, 3])
            jac[3:, j] = z
        err = np.r_[dp, dr * 0.2]  # 1 rad of orientation weighs as 0.2 m of position: about the hand's length
        jac[3:] *= 0.2
        damp = 1e-3
        pinv = jac.T @ np.linalg.inv(jac @ jac.T + damp * np.eye(6))
        step = pinv @ err
        if rest is not None and it < iters:  # a small drift per move: at most 0.01 rad a joint in one solve
            step += np.clip((np.eye(7) - pinv @ jac) @ (0.1 * (np.asarray(rest) - q)), -1e-4, 1e-4)
        q = np.clip(q + np.clip(step, -0.2, 0.2), LIMITS[:, 0], LIMITS[:, 1])
    at = [i + 1 for i in range(7) if min(q[i] - LIMITS[i, 0], LIMITS[i, 1] - q[i]) < 1e-3]
    raise Unreachable(
        f"the arm cannot put the palm there: the closest it gets is {np.linalg.norm(dp) * 100:.1f} cm and "
        f"{np.degrees(np.linalg.norm(dr)):.0f} deg off"
        + (f", with j{', j'.join(map(str, at))} at {'its limit' if len(at) == 1 else 'their limits'}" if at else "")
    )
