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


def head_joints_to_gaze(head_q: np.ndarray, R_torso: np.ndarray | None = None) -> np.ndarray:  # noqa: N803
    """Unit look direction in the robot base frame for ``[head_0, head_1]``.

    URDF: ``link_torso_5 -> head_0 (pan, +z) -> head_1 (tilt, +y)``; the head
    looks along +x of the last link, so ``d_t5 = R_z(q0) R_y(q1) x̂``.
    ``R_torso`` is base->link_torso_5 (identity when omitted).
    """
    q0, q1 = float(head_q[0]), float(head_q[1])
    d = np.array([math.cos(q1) * math.cos(q0), math.cos(q1) * math.sin(q0), -math.sin(q1)])
    if R_torso is None:
        return d
    return np.asarray(R_torso, dtype=float)[:3, :3] @ d


def gaze_to_head_joints(d_base: np.ndarray, R_torso: np.ndarray | None = None) -> np.ndarray:  # noqa: N803
    """``[head_0, head_1]`` that point the head along ``d_base`` (base frame).

    Inverse of :func:`head_joints_to_gaze`; the 2-DoF head cannot roll, so
    only the direction is matched. Unclipped (callers apply the limits).
    """
    d = np.asarray(d_base, dtype=float)
    if R_torso is not None:
        d = np.asarray(R_torso, dtype=float)[:3, :3].T @ d
    n = float(np.linalg.norm(d))
    d = d / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
    return np.array([math.atan2(d[1], d[0]), -math.asin(float(np.clip(d[2], -1.0, 1.0)))])


class HeadRetargeter:
    """Headset orientation -> ``[head_0 (yaw), head_1 (pitch)]`` joint targets.

    The target is a **look direction in the robot base frame**: the headset
    yaw / pitch (operator frame) map onto "the joints the head would need with
    an upright torso", and every emitted target is re-solved for the torso's
    actual orientation (:meth:`target_for`). The headset not moving therefore
    means the robot's gaze does not move, whatever the torso does. With
    ``compensate_torso=False`` the joints are commanded directly (legacy).

    gaze = sign * gain * (headset yaw / pitch in the operator frame) + offset;
    :meth:`latch_offset` recomputes the offsets so that the current look
    direction maps onto given head joints (Right A / first action). The
    result is clipped to the configured limits and low-pass filtered.
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
        yaw_offset: float = 0.0,
        pitch_offset: float = 0.0,
        compensate_torso: bool = True,
    ) -> None:
        # "gaze" is expressed as the head joints for an upright torso (so
        # signs / offsets read like joint angles).
        self.yaw_offset = yaw_offset
        self.pitch_offset = pitch_offset
        self.yaw_sign = yaw_sign
        self.pitch_sign = pitch_sign
        self.yaw_gain = yaw_gain
        self.pitch_gain = pitch_gain
        self.yaw_limit = abs(yaw_limit)
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max
        self.smoothing = smoothing
        self.compensate_torso = compensate_torso
        self._gaze: np.ndarray | None = None    # smoothed gaze (upright-torso joints)
        self._target: np.ndarray | None = None  # last emitted joints
        self._latched = False

    # -- frame helpers -------------------------------------------------
    def _R(self, R_torso: np.ndarray | None) -> np.ndarray | None:  # noqa: N802,N803
        return R_torso if self.compensate_torso else None

    def _gaze_from_joints(self, head_q: np.ndarray, R_torso: np.ndarray | None) -> np.ndarray:  # noqa: N803
        """Measured joints under ``R_torso`` -> upright-torso joints (base gaze)."""
        return gaze_to_head_joints(head_joints_to_gaze(head_q, self._R(R_torso)))

    def _joints_from_gaze(self, gaze: np.ndarray, R_torso: np.ndarray | None) -> np.ndarray:  # noqa: N803
        q = gaze_to_head_joints(head_joints_to_gaze(gaze), self._R(R_torso))
        return np.array(
            [
                np.clip(q[0], -self.yaw_limit, self.yaw_limit),
                np.clip(q[1], self.pitch_min, self.pitch_max),
            ]
        )

    # -- state ---------------------------------------------------------
    @property
    def latched(self) -> bool:
        return self._latched

    @property
    def target(self) -> np.ndarray | None:
        """Last emitted joint target (see :meth:`target_for` for the current torso)."""
        return None if self._target is None else self._target.copy()

    @property
    def gaze(self) -> np.ndarray | None:
        """Current base-frame gaze as upright-torso head joints."""
        return None if self._gaze is None else self._gaze.copy()

    def target_for(self, R_torso: np.ndarray | None) -> np.ndarray | None:  # noqa: N803
        """Joint target that keeps the current gaze for the given torso orientation."""
        if self._gaze is None:
            return self.target
        self._target = self._joints_from_gaze(self._gaze, R_torso)
        return self.target

    def hold(self, head_q: np.ndarray, R_torso: np.ndarray | None = None) -> None:  # noqa: N803
        """Hold the gaze given by ``head_q`` under ``R_torso`` (e.g. the measured joints)."""
        q = np.asarray(head_q, dtype=float).copy()
        self._gaze = self._gaze_from_joints(q, R_torso)
        self._target = q

    def latch(
        self, R_head_robot: np.ndarray, head_q_measured: np.ndarray, R_torso: np.ndarray | None = None  # noqa: N803
    ) -> None:
        """Start tracking; the measured joints are held until the first update."""
        if self._gaze is None:
            self.hold(head_q_measured, R_torso)
        self._latched = True

    def latch_offset(
        self, R_head_robot: np.ndarray, head_q_target: np.ndarray, R_torso: np.ndarray | None = None  # noqa: N803
    ) -> None:
        """Make the current look direction map onto the gaze of ``head_q_target`` (under ``R_torso``).

        The yaw / pitch offsets are recomputed so that
        ``sign * gain * angle_now + offset == gaze``; the held gaze is set to
        it (the robot head is there or on its way there).
        """
        target = np.asarray(head_q_target, dtype=float).copy()
        g = self._gaze_from_joints(target, R_torso)
        yaw, pitch = head_yaw_pitch(R_head_robot)
        self.yaw_offset = float(g[0] - self.yaw_sign * yaw * self.yaw_gain)
        self.pitch_offset = float(g[1] - self.pitch_sign * pitch * self.pitch_gain)
        self.hold(target, R_torso)
        self._latched = True

    def update(self, R_head_robot: np.ndarray | None, R_torso: np.ndarray | None = None) -> np.ndarray | None:  # noqa: N803
        """Return the joint target for ``R_torso``; the gaze holds when there is no head pose."""
        if R_head_robot is None or not self._latched:
            return self.target_for(R_torso)
        yaw, pitch = head_yaw_pitch(R_head_robot)
        raw = np.array(
            [
                self.yaw_sign * yaw * self.yaw_gain + self.yaw_offset,
                self.pitch_sign * pitch * self.pitch_gain + self.pitch_offset,
            ]
        )
        # Keep the gaze itself inside the joint range of an upright torso so
        # the smoothed state cannot wind up far beyond the limits.
        raw = np.array(
            [np.clip(raw[0], -self.yaw_limit, self.yaw_limit), np.clip(raw[1], self.pitch_min, self.pitch_max)]
        )
        if self._gaze is None:
            self._gaze = raw
        else:
            self._gaze = self.smoothing * raw + (1.0 - self.smoothing) * self._gaze
        return self.target_for(R_torso)


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


def interpolate_pose(T0: np.ndarray, T1: np.ndarray, alpha: float) -> np.ndarray:  # noqa: N803
    """SE3 interpolation (position lerp, rotation slerp) for ``alpha`` in [0, 1]."""
    a = float(np.clip(alpha, 0.0, 1.0))
    T0 = np.asarray(T0, dtype=float)
    T1 = np.asarray(T1, dtype=float)
    r0 = Rotation.from_matrix(T0[:3, :3])
    r1 = Rotation.from_matrix(T1[:3, :3])
    delta = r1 * r0.inv()
    rot = Rotation.from_rotvec(delta.as_rotvec() * a) * r0
    out = np.eye(4)
    out[:3, :3] = rot.as_matrix()
    out[:3, 3] = (1.0 - a) * T0[:3, 3] + a * T1[:3, 3]
    return out


def smoothstep(alpha: float) -> float:
    """Ease-in / ease-out profile for ``alpha`` in [0, 1]."""
    a = float(np.clip(alpha, 0.0, 1.0))
    return a * a * (3.0 - 2.0 * a)


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
