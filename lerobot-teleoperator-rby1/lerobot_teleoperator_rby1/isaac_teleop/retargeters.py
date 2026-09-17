"""Pure-numpy retargeting helpers for the RB-Y1 XR device.

Everything here is free of ``isaacteleop`` and ``rby1_sdk`` so it can be unit
tested offline: headset orientation -> head joints, body tracking -> torso
delta clamping, thumbsticks -> base velocity, SE3 -> LeRobot EE action keys.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from .xr_frame import BodyState

# OpenXR: a tracked head/view pose looks along its local -Z axis.
XR_HEAD_FORWARD_LOCAL = np.array([0.0, 0.0, -1.0])


def wrap_pi(angle: float) -> float:
    """Wrap an angle to ``(-pi, pi]``."""
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------


def head_yaw_pitch(R_head_robot: np.ndarray) -> tuple[float, float]:  # noqa: N803
    """Yaw / pitch (rad) of the headset look direction in the robot base frame.

    Uses the forward vector rather than an Euler decomposition, so roll is
    discarded and there is no gimbal / ordering ambiguity:
    ``yaw = atan2(f_y, f_x)`` (+ = left), ``pitch = atan2(f_z, |f_xy|)`` (+ = up).
    """
    f = np.asarray(R_head_robot, dtype=float)[:3, :3] @ XR_HEAD_FORWARD_LOCAL
    yaw = math.atan2(f[1], f[0])
    pitch = math.atan2(f[2], math.hypot(f[0], f[1]))
    return yaw, pitch


class HeadRetargeter:
    """Headset orientation -> ``[head_0 (yaw), head_1 (pitch)]`` joint targets.

    On :meth:`latch` the current look direction becomes the origin and the
    measured head joints become the offset, so the robot head does not move
    at latch time. :meth:`update` then applies the signed, gained delta,
    clips to the configured limits and low-pass filters the result.
    """

    def __init__(
        self,
        *,
        yaw_sign: float = 1.0,
        pitch_sign: float = -1.0,
        yaw_gain: float = 1.0,
        pitch_gain: float = 1.0,
        yaw_limit: float = math.radians(80.0),
        pitch_min: float = math.radians(-45.0),
        pitch_max: float = math.radians(80.0),
        smoothing: float = 0.3,
    ) -> None:
        self.yaw_sign = yaw_sign
        self.pitch_sign = pitch_sign
        self.yaw_gain = yaw_gain
        self.pitch_gain = pitch_gain
        self.yaw_limit = abs(yaw_limit)
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max
        self.smoothing = smoothing
        self._yaw0 = 0.0
        self._pitch0 = 0.0
        self._q0 = np.zeros(2)
        self._target: np.ndarray | None = None
        self._latched = False

    @property
    def latched(self) -> bool:
        return self._latched

    @property
    def target(self) -> np.ndarray | None:
        return None if self._target is None else self._target.copy()

    def hold(self, head_q: np.ndarray) -> None:
        """Set the held target (e.g. the measured head joints before any latch)."""
        self._target = np.asarray(head_q, dtype=float).copy()

    def latch(self, R_head_robot: np.ndarray, head_q_measured: np.ndarray) -> None:  # noqa: N803
        self._yaw0, self._pitch0 = head_yaw_pitch(R_head_robot)
        self._q0 = np.asarray(head_q_measured, dtype=float).copy()
        self._target = self._q0.copy()
        self._latched = True

    def update(self, R_head_robot: np.ndarray | None) -> np.ndarray | None:  # noqa: N803
        """Return the smoothed joint target; holds the previous one when no head pose."""
        if R_head_robot is None or not self._latched:
            return self.target
        yaw, pitch = head_yaw_pitch(R_head_robot)
        dyaw = wrap_pi(yaw - self._yaw0) * self.yaw_gain
        dpitch = (pitch - self._pitch0) * self.pitch_gain
        raw = np.array(
            [
                np.clip(self._q0[0] + self.yaw_sign * dyaw, -self.yaw_limit, self.yaw_limit),
                np.clip(self._q0[1] + self.pitch_sign * dpitch, self.pitch_min, self.pitch_max),
            ]
        )
        if self._target is None:
            self._target = raw
        else:
            self._target = self.smoothing * raw + (1.0 - self.smoothing) * self._target
        return self.target


# ---------------------------------------------------------------------------
# Torso
# ---------------------------------------------------------------------------


def chest_pose_from_body(
    body: BodyState | None,
    joint_index: int,
    required_indices: Sequence[int],
) -> np.ndarray | None:
    """4x4 pose of ``joint_index`` when every required joint is valid, else None."""
    if body is None:
        return None
    needed = list(required_indices) + [joint_index]
    if not all(bool(body.valid[i]) for i in needed):
        return None
    return body.joint_pose(joint_index)


def scale_clamp_delta(
    home_T: np.ndarray,  # noqa: N803
    target_T: np.ndarray,  # noqa: N803
    *,
    rot_scale: float = 1.0,
    z_scale: float = 1.0,
    use_xy: bool = False,
    max_rot: float = math.radians(35.0),
    max_z: float = 0.15,
) -> np.ndarray:
    """Scale and clamp the delta ``home -> target`` and return the clamped target.

    Rotation: the base-frame rotation vector of ``R_target R_home^T`` is scaled
    by ``rot_scale`` and its norm clamped to ``max_rot``. Translation: only the
    z component is kept (scaled by ``z_scale``, clamped to ``±max_z``) unless
    ``use_xy`` also passes x/y through unclamped-scaled.
    """
    home = np.asarray(home_T, dtype=float)
    target = np.asarray(target_T, dtype=float)
    dR = Rotation.from_matrix(target[:3, :3]) * Rotation.from_matrix(home[:3, :3]).inv()
    rotvec = dR.as_rotvec() * rot_scale
    norm = float(np.linalg.norm(rotvec))
    if norm > max_rot > 0.0:
        rotvec = rotvec * (max_rot / norm)
    dp = target[:3, 3] - home[:3, 3]
    if not use_xy:
        dp[:2] = 0.0
    dp[2] = float(np.clip(dp[2] * z_scale, -abs(max_z), abs(max_z)))
    out = np.eye(4)
    out[:3, :3] = (Rotation.from_rotvec(rotvec) * Rotation.from_matrix(home[:3, :3])).as_matrix()
    out[:3, 3] = home[:3, 3] + dp
    return out


# ---------------------------------------------------------------------------
# Mobile base
# ---------------------------------------------------------------------------


def _deadzone(v: float, deadzone: float) -> float:
    a = abs(v)
    if a <= deadzone:
        return 0.0
    scaled = (a - deadzone) / max(1.0 - deadzone, 1e-6)
    return math.copysign(min(scaled, 1.0), v)


def thumbsticks_to_base_vel(
    right_xy: Sequence[float] | None,
    left_xy: Sequence[float] | None,
    *,
    deadzone: float = 0.15,
    max_linear: float = 0.3,
    max_angular: float = 0.6,
) -> tuple[float, float, float]:
    """Map the thumbsticks to a body-frame ``(x, y, theta)`` velocity.

    Right stick: push forward (+y) -> +x (m/s), push right (+x) -> -y (m/s,
    i.e. rightwards). Left stick: push right (+x) -> -theta (rad/s, clockwise).
    """
    vx = vy = wz = 0.0
    if right_xy is not None:
        vx = _deadzone(float(right_xy[1]), deadzone) * max_linear
        vy = -_deadzone(float(right_xy[0]), deadzone) * max_linear
    if left_xy is not None:
        wz = -_deadzone(float(left_xy[0]), deadzone) * max_angular
    return vx, vy, wz


# ---------------------------------------------------------------------------
# Action encoding
# ---------------------------------------------------------------------------


def se3_to_ee_action(T: np.ndarray, prefix: str) -> dict[str, float]:  # noqa: N803
    """Encode a 4x4 pose as ``{prefix}.x/y/z`` (m) + ``{prefix}.wx/wy/wz`` (rotvec, rad).

    Same convention as ``lerobot_robot_rby1.command_builders.action_from_pose``.
    """
    T = np.asarray(T, dtype=float)
    rotvec = Rotation.from_matrix(T[:3, :3]).as_rotvec()
    return {
        f"{prefix}.x": float(T[0, 3]),
        f"{prefix}.y": float(T[1, 3]),
        f"{prefix}.z": float(T[2, 3]),
        f"{prefix}.wx": float(rotvec[0]),
        f"{prefix}.wy": float(rotvec[1]),
        f"{prefix}.wz": float(rotvec[2]),
    }
