"""Pure RB10E forward and inverse kinematics.

This module intentionally contains no robot communication. It converts an
RB10E Cartesian target into a six-joint target that can be emitted by a
LeRobot Teleoperator.

Unit conventions
----------------
* ``q_rad``: joint angles in radians.
* Target and FK translations: millimetres.
* Target and FK rotations: 3x3 rotation matrices.
* Jacobian translational rows: millimetres per radian.
* Returned joint targets: radians.

The FK and Jacobian equations are retained from the existing RB10E VR
teleoperation implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from .constants import (
    DEFAULT_IK_BASE_DAMPING,
    DEFAULT_IK_ERROR_DAMPING_GAIN,
    DEFAULT_IK_ITERATIONS,
    DEFAULT_READY_POSE_RAD,
    DOF,
    RB10E_JOINT_LIMITS_RAD,
)


FloatArray = NDArray[np.float64]

# cos(pi / 2), retained explicitly from the generated legacy equations.
_C90 = 6.123233995736766e-17


@dataclass(frozen=True)
class IKSolveInfo:
    """Diagnostics from the most recent IK solve."""

    iterations: int
    position_error_mm: float
    orientation_error: float
    total_error: float
    used_lstsq_fallback: bool


class RB10EKinematics:
    """RB10E numerical kinematics without control-box communication."""

    def __init__(
        self,
        *,
        initial_q_rad: Iterable[float] = DEFAULT_READY_POSE_RAD,
        joint_limits_rad: Iterable[Iterable[float]] = RB10E_JOINT_LIMITS_RAD,
        iterations: int = DEFAULT_IK_ITERATIONS,
        base_damping: float = DEFAULT_IK_BASE_DAMPING,
        error_damping_gain: float = DEFAULT_IK_ERROR_DAMPING_GAIN,
        max_iteration_step_rad: float | None = None,
    ) -> None:
        self._joint_limits = np.asarray(
            tuple(tuple(limit) for limit in joint_limits_rad),
            dtype=np.float64,
        )

        if self._joint_limits.shape != (DOF, 2):
            raise ValueError(
                "joint_limits_rad must have shape "
                f"({DOF}, 2), got {self._joint_limits.shape}."
            )

        if np.any(self._joint_limits[:, 0] > self._joint_limits[:, 1]):
            raise ValueError("Each lower joint limit must be <= its upper limit.")

        if iterations <= 0:
            raise ValueError("iterations must be greater than zero.")

        if base_damping < 0.0:
            raise ValueError("base_damping must be non-negative.")

        if error_damping_gain < 0.0:
            raise ValueError("error_damping_gain must be non-negative.")

        if (
            max_iteration_step_rad is not None
            and max_iteration_step_rad <= 0.0
        ):
            raise ValueError(
                "max_iteration_step_rad must be positive or None."
            )

        self._iterations = int(iterations)
        self._base_damping = float(base_damping)
        self._error_damping_gain = float(error_damping_gain)
        self._max_iteration_step_rad = max_iteration_step_rad

        self._identity = np.eye(DOF, dtype=np.float64)

        self._q_rad = self._validate_joint_vector(initial_q_rad)
        self._q_rad = np.clip(
            self._q_rad,
            self._joint_limits[:, 0],
            self._joint_limits[:, 1],
        )

        self._fk = np.eye(4, dtype=np.float64)
        self._jacobian = np.zeros((DOF, DOF), dtype=np.float64)
        self._transforms = tuple(
            np.eye(4, dtype=np.float64) for _ in range(DOF)
        )
        self._last_info = IKSolveInfo(
            iterations=0,
            position_error_mm=0.0,
            orientation_error=0.0,
            total_error=0.0,
            used_lstsq_fallback=False,
        )

        self.compute_fk_and_jacobian(self._q_rad)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def q_rad(self) -> FloatArray:
        """Return the most recently solved joint vector in radians."""

        return self._q_rad.copy()

    @property
    def fk(self) -> FloatArray:
        """Return the latest 4x4 end-effector pose in millimetres."""

        return self._fk.copy()

    @property
    def jacobian(self) -> FloatArray:
        """Return the latest 6x6 geometric Jacobian."""

        return self._jacobian.copy()

    @property
    def transforms(self) -> tuple[FloatArray, ...]:
        """Return base-to-link transforms T0 through T5."""

        return tuple(transform.copy() for transform in self._transforms)

    @property
    def joint_limits_rad(self) -> FloatArray:
        return self._joint_limits.copy()

    @property
    def last_info(self) -> IKSolveInfo:
        return self._last_info

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self, q_rad: Iterable[float] = DEFAULT_READY_POSE_RAD) -> None:
        """Reset the IK seed to a known joint pose."""

        q = self._validate_joint_vector(q_rad)
        self._q_rad = np.clip(
            q,
            self._joint_limits[:, 0],
            self._joint_limits[:, 1],
        )
        self.compute_fk_and_jacobian(self._q_rad)

    def set_seed(self, q_rad: Iterable[float]) -> None:
        """Set the joint seed used by the next IK call."""

        self.reset(q_rad)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_joint_vector(q_rad: Iterable[float]) -> FloatArray:
        q = np.asarray(tuple(q_rad), dtype=np.float64)

        if q.shape != (DOF,):
            raise ValueError(
                f"Expected a ({DOF},) joint vector, got {q.shape}."
            )

        if not np.all(np.isfinite(q)):
            raise ValueError(f"Joint vector contains NaN or Inf: {q}")

        return q.copy()

    @staticmethod
    def _validate_pose(target_pose_mm: FloatArray) -> FloatArray:
        target = np.asarray(target_pose_mm, dtype=np.float64)

        if target.shape != (4, 4):
            raise ValueError(
                f"Expected a (4, 4) target pose, got {target.shape}."
            )

        if not np.all(np.isfinite(target)):
            raise ValueError("Target pose contains NaN or Inf.")

        if not np.allclose(
            target[3],
            np.array([0.0, 0.0, 0.0, 1.0]),
            atol=1e-6,
        ):
            raise ValueError(
                "Target pose must be a homogeneous transform with "
                "last row [0, 0, 0, 1]."
            )

        rotation = target[:3, :3]
        should_be_identity = rotation.T @ rotation

        if not np.allclose(
            should_be_identity,
            np.eye(3),
            atol=1e-4,
        ):
            raise ValueError("Target rotation matrix is not orthonormal.")

        determinant = float(np.linalg.det(rotation))
        if not np.isclose(determinant, 1.0, atol=1e-4):
            raise ValueError(
                "Target rotation matrix must have determinant +1, "
                f"got {determinant:.6f}."
            )

        return target.copy()

    # ------------------------------------------------------------------
    # Forward kinematics and Jacobian
    # ------------------------------------------------------------------

    def forward_kinematics(
        self,
        q_rad: Iterable[float] | None = None,
    ) -> FloatArray:
        """Compute and return the RB10E TCP pose.

        The returned translation is in millimetres.
        """

        if q_rad is None:
            q = self._q_rad
        else:
            q = self._validate_joint_vector(q_rad)

        fk, _ = self.compute_fk_and_jacobian(q)
        return fk

    def compute_fk_and_jacobian(
        self,
        q_rad: Iterable[float],
    ) -> tuple[FloatArray, FloatArray]:
        """Compute the TCP pose and geometric Jacobian."""

        q = self._validate_joint_vector(q_rad)

        c0, c1, c2, c3, c4, c5 = np.cos(q)
        s0, s1, s2, s3, s4, s5 = np.sin(q)

        t0 = np.array(
            [
                [c0, -_C90 * s0, -s0, 0.0],
                [s0, _C90 * c0, c0, 0.0],
                [0.0, -1.0, _C90, 197.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        t1 = t0 @ np.array(
            [
                [c1, -_C90 * s1, s1, 0.0],
                [s1, _C90 * c1, -c1, 0.0],
                [0.0, 1.0, _C90, -187.5],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        t2 = t1 @ np.array(
            [
                [c2, -_C90 * s2, s2, 0.0],
                [
                    _C90 * s2,
                    (_C90 * _C90) * c2 + 1.0,
                    _C90 - _C90 * c2,
                    148.4,
                ],
                [
                    -s2,
                    _C90 - _C90 * c2,
                    c2 + (_C90 * _C90),
                    612.7,
                ],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        t3 = t2 @ np.array(
            [
                [c3, -_C90 * s3, s3, 0.0],
                [
                    _C90 * s3,
                    (_C90 * _C90) * c3 + 1.0,
                    _C90 - _C90 * c3,
                    -117.15,
                ],
                [
                    -s3,
                    _C90 - _C90 * c3,
                    c3 + (_C90 * _C90),
                    570.15,
                ],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        t4 = t3 @ np.array(
            [
                [c4, -_C90 * s4, -s4, 0.0],
                [s4, _C90 * c4, c4, 0.0],
                [0.0, -1.0, _C90, 117.15],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        t5 = t4 @ np.array(
            [
                [c5, -_C90 * s5, s5, 0.0],
                [s5, _C90 * c5, -c5, 0.0],
                [0.0, 1.0, _C90, -259.3],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        fk = t5.copy()

        jacobian = np.zeros((DOF, DOF), dtype=np.float64)

        axis_0 = -t0[:3, 1]
        axis_1 = t1[:3, 1]
        axis_2 = t2[:3, 1]
        axis_3 = t3[:3, 1]
        axis_4 = -t4[:3, 1]
        axis_5 = t5[:3, 1]

        origins = (
            t0[:3, 3],
            t1[:3, 3],
            t2[:3, 3],
            t3[:3, 3],
            t4[:3, 3],
            t5[:3, 3],
        )
        axes = (
            axis_0,
            axis_1,
            axis_2,
            axis_3,
            axis_4,
            axis_5,
        )

        tcp_position = fk[:3, 3]

        for index, (axis, origin) in enumerate(zip(axes, origins)):
            jacobian[:3, index] = np.cross(
                axis,
                tcp_position - origin,
            )
            jacobian[3:, index] = axis

        self._fk = fk.copy()
        self._jacobian = jacobian.copy()
        self._transforms = (
            t0.copy(),
            t1.copy(),
            t2.copy(),
            t3.copy(),
            t4.copy(),
            t5.copy(),
        )

        return fk.copy(), jacobian.copy()

    # ------------------------------------------------------------------
    # Pose error
    # ------------------------------------------------------------------

    @staticmethod
    def pose_error(
        target_pose_mm: FloatArray,
        current_pose_mm: FloatArray,
    ) -> FloatArray:
        """Return the legacy six-dimensional RB10E IK error vector."""

        target = RB10EKinematics._validate_pose(target_pose_mm)
        current = RB10EKinematics._validate_pose(current_pose_mm)

        target_rotation = target[:3, :3]
        current_rotation = current[:3, :3]

        relative_rotation = target_rotation @ current_rotation.T

        error = np.empty(DOF, dtype=np.float64)
        error[:3] = target[:3, 3] - current[:3, 3]

        # Retain the orientation error definition used by the existing code.
        error[3] = (
            relative_rotation[2, 1] - relative_rotation[1, 2]
        )
        error[4] = (
            relative_rotation[0, 2] - relative_rotation[2, 0]
        )
        error[5] = (
            relative_rotation[1, 0] - relative_rotation[0, 1]
        )

        return error

    # ------------------------------------------------------------------
    # Inverse kinematics
    # ------------------------------------------------------------------

    def solve(
        self,
        target_pose_mm: FloatArray,
        *,
        initial_q_rad: Iterable[float] | None = None,
        iterations: int | None = None,
    ) -> FloatArray:
        """Solve a Cartesian target and return six joint angles in radians.

        Parameters
        ----------
        target_pose_mm:
            Homogeneous 4x4 target pose. Translation must be in millimetres.
        initial_q_rad:
            Optional IK seed. When omitted, the previous solution is used.
        iterations:
            Optional number of solver iterations. When omitted, the value
            configured at construction is used.
        """

        target = self._validate_pose(target_pose_mm)

        if initial_q_rad is None:
            q = self._q_rad.copy()
        else:
            q = self._validate_joint_vector(initial_q_rad)
            q = np.clip(
                q,
                self._joint_limits[:, 0],
                self._joint_limits[:, 1],
            )

        iteration_count = (
            self._iterations if iterations is None else int(iterations)
        )

        if iteration_count <= 0:
            raise ValueError("iterations must be greater than zero.")

        used_lstsq_fallback = False

        for _ in range(iteration_count):
            fk, jacobian = self.compute_fk_and_jacobian(q)
            error = self.pose_error(target, fk)

            quadratic_error = 0.5 * float(error @ error)
            gradient = jacobian.T @ error

            damping = (
                self._base_damping
                + self._error_damping_gain * quadratic_error
            )

            hessian_approx = (
                jacobian.T @ jacobian
                + damping * self._identity
            )

            try:
                delta_q = np.linalg.solve(
                    hessian_approx,
                    gradient,
                )
            except np.linalg.LinAlgError:
                # The damping should normally make H invertible, but keep a
                # deterministic fallback so a singular numerical case does
                # not crash the teleoperation loop.
                delta_q = np.linalg.lstsq(
                    hessian_approx,
                    gradient,
                    rcond=None,
                )[0]
                used_lstsq_fallback = True

            if self._max_iteration_step_rad is not None:
                delta_q = np.clip(
                    delta_q,
                    -self._max_iteration_step_rad,
                    self._max_iteration_step_rad,
                )

            q = np.clip(
                q + delta_q,
                self._joint_limits[:, 0],
                self._joint_limits[:, 1],
            )

        final_fk, _ = self.compute_fk_and_jacobian(q)
        final_error = self.pose_error(target, final_fk)

        position_error_mm = float(np.linalg.norm(final_error[:3]))
        orientation_error = float(np.linalg.norm(final_error[3:]))
        total_error = float(np.linalg.norm(final_error))

        self._q_rad = q.copy()
        self._last_info = IKSolveInfo(
            iterations=iteration_count,
            position_error_mm=position_error_mm,
            orientation_error=orientation_error,
            total_error=total_error,
            used_lstsq_fallback=used_lstsq_fallback,
        )

        return q.copy()
