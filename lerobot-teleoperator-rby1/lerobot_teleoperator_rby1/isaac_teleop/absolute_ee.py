"""Controller pose → absolute RB-Y1 end-effector target, referenced to the shoulder.

``arm_mode="ee_absolute"``: instead of a clutch (delta from the pose at
squeeze time), the hand position is taken **relative to the operator's
shoulder** (IOBT) and scaled by the robot / human reach ratio onto the
robot's shoulder. This keeps the commanded hand position and the IOBT-derived
elbow hint consistent (both come from the same body). The orientation is the
controller orientation times a fixed offset (latched on Right A or from the
config).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .arm_retargeter import robot_shoulder_position


class AbsoluteEeMapper:
    """Per-arm mapper with shoulder smoothing, reach scaling and rate limiting."""

    def __init__(
        self,
        side: str,
        *,
        robot_reach: float,
        human_reach: float | None = None,   # None = estimate from the body joints
        position_scale: float = 1.0,
        reach_max_ratio: float = 0.98,
        shoulder_smoothing: float = 0.2,
        reach_smoothing: float = 0.05,
        max_linear_vel: float = 1.0,
        max_angular_vel: float = 3.0,
        orientation_offset: np.ndarray | None = None,
    ) -> None:
        self.side = side
        self.robot_reach = robot_reach
        self._human_reach_cfg = human_reach
        self.position_scale = position_scale
        self.reach_max_ratio = reach_max_ratio
        self.shoulder_smoothing = shoulder_smoothing
        self.reach_smoothing = reach_smoothing
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        self._R_offset = np.eye(3) if orientation_offset is None else np.asarray(orientation_offset, dtype=float)
        self._shoulder: np.ndarray | None = None
        self._human_reach: float | None = None
        self._last_T: np.ndarray | None = None

    # ------------------------------------------------------------------
    @property
    def shoulder(self) -> np.ndarray | None:
        return None if self._shoulder is None else self._shoulder.copy()

    @property
    def human_reach(self) -> float | None:
        return self._human_reach_cfg if self._human_reach_cfg is not None else self._human_reach

    @property
    def orientation_offset(self) -> np.ndarray:
        return self._R_offset.copy()

    def latch_orientation_offset(self, R_ctrl: np.ndarray, R_ee_measured: np.ndarray) -> None:  # noqa: N803
        """Make the current controller orientation map onto the measured EE orientation."""
        self._R_offset = np.asarray(R_ctrl, dtype=float)[:3, :3].T @ np.asarray(R_ee_measured, dtype=float)[:3, :3]

    def reset_rate_limit(self, T: np.ndarray | None) -> None:  # noqa: N803
        self._last_T = None if T is None else np.asarray(T, dtype=float).copy()

    # ------------------------------------------------------------------
    def observe_body(self, shoulder: np.ndarray | None, elbow: np.ndarray | None, wrist: np.ndarray | None) -> None:
        """Update the smoothed shoulder position and the human reach estimate."""
        if shoulder is not None:
            s = np.asarray(shoulder, dtype=float)
            if self._shoulder is None:
                self._shoulder = s.copy()
            else:
                a = self.shoulder_smoothing
                self._shoulder = a * s + (1.0 - a) * self._shoulder
        if shoulder is not None and elbow is not None and wrist is not None:
            reach = float(np.linalg.norm(np.asarray(elbow) - np.asarray(shoulder)) + np.linalg.norm(np.asarray(wrist) - np.asarray(elbow)))
            if reach > 0.2:
                if self._human_reach is None:
                    self._human_reach = reach
                else:
                    b = self.reach_smoothing
                    self._human_reach = b * reach + (1.0 - b) * self._human_reach

    def target(self, ctrl_position: np.ndarray, ctrl_orientation_xyzw: np.ndarray, T_torso: np.ndarray) -> np.ndarray | None:  # noqa: N803
        """Absolute base-frame EE pose (4x4) for the current controller pose, or None."""
        if self._shoulder is None or self.human_reach is None:
            return None
        T_torso = np.asarray(T_torso, dtype=float)
        Rt = T_torso[:3, :3]
        d = np.asarray(ctrl_position, dtype=float) - self._shoulder
        scale = self.robot_reach / max(self.human_reach, 1e-3) * self.position_scale
        d_t = Rt.T @ d * scale
        max_len = self.reach_max_ratio * self.robot_reach
        n = float(np.linalg.norm(d_t))
        if n > max_len:
            d_t = d_t * (max_len / n)
        p = robot_shoulder_position(T_torso, self.side) + Rt @ d_t
        R = Rotation.from_quat(np.asarray(ctrl_orientation_xyzw, dtype=float)).as_matrix() @ self._R_offset
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        return T

    def rate_limit(self, T: np.ndarray, dt: float) -> np.ndarray:  # noqa: N803
        """Clamp the step from the last emitted pose to the velocity limits."""
        T = np.asarray(T, dtype=float)
        if self._last_T is None:
            self._last_T = T.copy()
            return T.copy()
        dt = max(dt, 1e-3)
        out = np.eye(4)
        dp = T[:3, 3] - self._last_T[:3, 3]
        n = float(np.linalg.norm(dp))
        max_dp = self.max_linear_vel * dt
        if n > max_dp:
            dp = dp * (max_dp / n)
        out[:3, 3] = self._last_T[:3, 3] + dp
        dR = Rotation.from_matrix(T[:3, :3]) * Rotation.from_matrix(self._last_T[:3, :3]).inv()
        rv = dR.as_rotvec()
        ang = float(np.linalg.norm(rv))
        max_ang = self.max_angular_vel * dt
        if ang > max_ang:
            rv = rv * (max_ang / ang)
        out[:3, :3] = (Rotation.from_rotvec(rv) * Rotation.from_matrix(self._last_T[:3, :3])).as_matrix()
        self._last_T = out.copy()
        return out


def rpy_deg_to_matrix(rpy_deg) -> np.ndarray:
    r, p, y = [float(v) for v in rpy_deg]
    return Rotation.from_euler("xyz", [r, p, y], degrees=True).as_matrix()


def is_finite_pose(T: np.ndarray) -> bool:  # noqa: N803
    return bool(np.all(np.isfinite(np.asarray(T, dtype=float))))

