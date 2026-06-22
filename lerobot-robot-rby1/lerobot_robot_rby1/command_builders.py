"""Builders for the ``rby1_sdk`` command objects sent to the robot.

Each function receives the SDK module (``rby``) plus already-computed joint
or Cartesian targets and returns a ready-to-send command builder.  Isolating
the (verbose) command DSL here keeps :class:`~lerobot_robot_rby1.rby1.Rby1`
focused on high-level control flow.

Joint-mode actions are executed via :func:`build_body_command`; EE-mode
actions (``action_mode="ee"``) are decoded with :func:`pose_from_action` and
executed via :func:`build_whole_body_command` or
:func:`build_per_component_command`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from .constants import (
    ARM_DOF,
    EE_SUFFIXES,
    LEFT_ARM_NAMES,
    LEFT_ARM_Q_MAX,
    LEFT_ARM_Q_MIN,
    RIGHT_ARM_NAMES,
    RIGHT_ARM_Q_MAX,
    RIGHT_ARM_Q_MIN,
    TORSO_DOF,
)

# Stiffness / torque-limit arrays are laid out as
# [torso (6) | right arm (7) | left arm (7)].
_RIGHT_ARM_SLICE = slice(TORSO_DOF, TORSO_DOF + ARM_DOF)          # 6:13
_LEFT_ARM_SLICE = slice(TORSO_DOF + ARM_DOF, TORSO_DOF + 2 * ARM_DOF)  # 13:20

# Joint commands keep holding for this long (seconds) if the stream stalls.
_CONTROL_HOLD_TIME = 600.0


# ---------------------------------------------------------------------------
# EE (Cartesian impedance) targets
# ---------------------------------------------------------------------------

@dataclass
class CartesianTargets:
    """Target end-effector poses (4×4 SE3 in the robot base frame)."""

    torso: np.ndarray
    right_arm: np.ndarray
    left_arm: np.ndarray


@dataclass
class ResetFlags:
    """Per-component ``set_reset_reference`` flags for the next command."""

    torso: bool = False
    right_arm: bool = False
    left_arm: bool = False

    @property
    def any(self) -> bool:
        return self.torso or self.right_arm or self.left_arm


def pose_from_action(action: dict[str, Any], prefix: str) -> np.ndarray:
    """Decode ``{prefix}.x/y/z/wx/wy/wz`` action keys into a 4×4 SE3 matrix.

    Position is in metres, ``wx/wy/wz`` is a rotation vector in radians
    (the LeRobot EE convention).
    """
    x, y, z, wx, wy, wz = (float(action[f"{prefix}.{s}"]) for s in EE_SUFFIXES)
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec([wx, wy, wz]).as_matrix()
    T[:3, 3] = [x, y, z]
    return T


def action_from_pose(T: np.ndarray, prefix: str) -> dict[str, float]:
    """Encode a 4×4 SE3 pose as ``{prefix}.x/y/z/wx/wy/wz`` keys.

    Inverse of :func:`pose_from_action`: position in metres plus a rotation
    vector in radians (the LeRobot EE convention).
    """
    rotvec = R.from_matrix(T[:3, :3]).as_rotvec()
    values = (*T[:3, 3], *rotvec)
    return {f"{prefix}.{s}": float(v) for s, v in zip(EE_SUFFIXES, values)}


def _ee_command_header(rby: Any, hold_time: float) -> Any:
    return rby.CommandHeaderBuilder().set_control_hold_time(hold_time)


def _apply_joint_limits(builder: Any, limits: dict[str, tuple[float, float]]) -> Any:
    for joint_name, (lo, hi) in limits.items():
        builder.add_joint_limit(joint_name, lo, hi)
    return builder


def build_whole_body_command(
    rby: Any,
    cfg: Any,
    targets: CartesianTargets,
    reset: ResetFlags,
) -> Any:
    """Single-solver Cartesian impedance command over the full 20-DOF body."""
    dt = cfg.ee_dt
    builder = (
        rby.CartesianImpedanceControlCommandBuilder()
        .set_command_header(_ee_command_header(rby, cfg.ee_hold_time))
        .set_minimum_time(dt * cfg.min_time_factor_wb)
        .set_joint_stiffness(cfg.wb_joint_stiffness)
        .set_joint_torque_limit(cfg.wb_joint_torque_limit)
        .set_stop_joint_position_tracking_error(0)
        .set_stop_orientation_tracking_error(0)
        .set_reset_reference(reset.any)
    )
    _apply_joint_limits(builder, cfg.wb_joint_limits)

    if cfg.use_torso:
        builder.add_target(
            "base", "link_torso_5", targets.torso,
            *cfg.torso_target_gains_wb,
        )
    if cfg.use_right_arm:
        builder.add_target(
            "base", "link_right_arm_6", targets.right_arm,
            *cfg.arm_target_gains_wb,
        )
    if cfg.use_left_arm:
        builder.add_target(
            "base", "link_left_arm_6", targets.left_arm,
            *cfg.arm_target_gains_wb,
        )
    return builder


def build_per_component_command(
    rby: Any,
    cfg: Any,
    targets: CartesianTargets,
    reset: ResetFlags,
    null_right: np.ndarray,
    null_left: np.ndarray,
) -> Any:
    """Separate torso / right-arm / left-arm solvers (more stable than WB)."""
    dt = cfg.ee_dt
    body = rby.BodyComponentBasedCommandBuilder()

    if cfg.use_torso:
        torso_builder = (
            rby.CartesianImpedanceControlCommandBuilder()
            .set_command_header(_ee_command_header(rby, cfg.ee_hold_time))
            .set_minimum_time(dt * cfg.min_time_factor_pc)
            .set_joint_stiffness(cfg.torso_stiffness)
            .set_joint_torque_limit(cfg.torso_torque_limit)
            .set_stop_joint_position_tracking_error(0)
            .set_stop_orientation_tracking_error(0)
            .set_joint_damping_ratio(cfg.torso_damping_ratio)
            .set_reset_reference(reset.torso)
        )
        _apply_joint_limits(torso_builder, cfg.torso_joint_limits)
        torso_builder.add_target(
            "base", "link_torso_5", targets.torso,
            *cfg.torso_target_gains_pc,
        )
        body.set_torso_command(torso_builder)

    if cfg.use_right_arm:
        right_builder = (
            rby.CartesianImpedanceControlCommandBuilder()
            .set_command_header(_ee_command_header(rby, cfg.ee_hold_time))
            .set_minimum_time(dt * cfg.min_time_factor_pc)
            .set_joint_stiffness(cfg.right_arm_stiffness)
            .set_joint_torque_limit(cfg.right_arm_torque_limit)
            .set_nullspace_joint_target(
                null_right,
                np.asarray(cfg.nullspace_weight),
                cfg.nullspace_stiffness,
                cfg.nullspace_damping_ratio,
            )
            .set_stop_joint_position_tracking_error(0)
            .set_stop_orientation_tracking_error(0)
            .set_joint_damping_ratio(cfg.right_arm_damping_ratio)
            .set_reset_reference(reset.right_arm)
        )
        _apply_joint_limits(right_builder, cfg.right_arm_joint_limits)
        right_builder.add_target(
            "base", "link_right_arm_6", targets.right_arm,
            *cfg.arm_target_gains_pc,
        )
        body.set_right_arm_command(right_builder)

    if cfg.use_left_arm:
        left_builder = (
            rby.CartesianImpedanceControlCommandBuilder()
            .set_command_header(_ee_command_header(rby, cfg.ee_hold_time))
            .set_minimum_time(dt * cfg.min_time_factor_pc)
            .set_joint_stiffness(cfg.left_arm_stiffness)
            .set_joint_torque_limit(cfg.left_arm_torque_limit)
            .set_nullspace_joint_target(
                null_left,
                np.asarray(cfg.nullspace_weight),
                cfg.nullspace_stiffness,
                cfg.nullspace_damping_ratio,
            )
            .set_stop_joint_position_tracking_error(0)
            .set_stop_orientation_tracking_error(0)
            .set_joint_damping_ratio(cfg.left_arm_damping_ratio)
            .set_reset_reference(reset.left_arm)
        )
        _apply_joint_limits(left_builder, cfg.left_arm_joint_limits)
        left_builder.add_target(
            "base", "link_left_arm_6", targets.left_arm,
            *cfg.arm_target_gains_pc,
        )
        body.set_left_arm_command(left_builder)

    return body


def build_limb_command(
    rby: Any,
    *,
    position: np.ndarray,
    velocity_limit: np.ndarray,
    acceleration_limit: np.ndarray,
    minimum_time: float,
    use_impedance: bool,
    stiffness: Any = None,
    torque_limit: Any = None,
    hold_time: float = _CONTROL_HOLD_TIME,
) -> Any:
    """Build a single-limb joint command (impedance or plain position)."""
    header = rby.CommandHeaderBuilder().set_control_hold_time(hold_time)
    if use_impedance:
        return (
            rby.JointImpedanceControlCommandBuilder()
            .set_command_header(header)
            .set_minimum_time(minimum_time)
            .set_position(position)
            .set_velocity_limit(velocity_limit)
            .set_acceleration_limit(acceleration_limit)
            .set_stiffness(stiffness)
            .set_torque_limit(torque_limit)
        )
    return (
        rby.JointPositionCommandBuilder()
        .set_command_header(header)
        .set_minimum_time(minimum_time)
        .set_position(position)
        .set_velocity_limit(velocity_limit)
        .set_acceleration_limit(acceleration_limit)
    )


def build_body_command(
    rby: Any,
    cfg: Any,
    action: dict[str, Any],
    model: Any,
    max_qdot: np.ndarray,
    max_qddot: np.ndarray,
    minimum_time: float,
) -> tuple[Any, bool]:
    """Assemble a body command for the enabled arms.

    Disabled limbs are omitted entirely so the SDK holds them at their current
    position.  The torso is observation-only and is never commanded here.
    Joint targets are clipped to the URDF limits before being sent.

    Returns
    -------
    (body, has_command)
        ``body`` is a ``BodyComponentBasedCommandBuilder``; ``has_command`` is
        True when at least one arm command was added (so the caller can skip an
        empty body).
    """
    stiffness = cfg.impedance_stiffness
    torque_limit = cfg.impedance_torque_limit
    acc_scale = cfg.acceleration_limit_scale

    body = rby.BodyComponentBasedCommandBuilder()
    has_command = False

    if cfg.use_right_arm:
        right_q = np.clip(
            np.array([action[name] for name in RIGHT_ARM_NAMES]),
            RIGHT_ARM_Q_MIN,
            RIGHT_ARM_Q_MAX,
        )
        body.set_right_arm_command(
            build_limb_command(
                rby,
                position=right_q,
                velocity_limit=max_qdot[model.right_arm_idx],
                acceleration_limit=max_qddot[model.right_arm_idx] * acc_scale,
                minimum_time=minimum_time,
                use_impedance=cfg.use_impedance,
                stiffness=stiffness[_RIGHT_ARM_SLICE],
                torque_limit=torque_limit[_RIGHT_ARM_SLICE],
            )
        )
        has_command = True

    if cfg.use_left_arm:
        left_q = np.clip(
            np.array([action[name] for name in LEFT_ARM_NAMES]),
            LEFT_ARM_Q_MIN,
            LEFT_ARM_Q_MAX,
        )
        body.set_left_arm_command(
            build_limb_command(
                rby,
                position=left_q,
                velocity_limit=max_qdot[model.left_arm_idx],
                acceleration_limit=max_qddot[model.left_arm_idx] * acc_scale,
                minimum_time=minimum_time,
                use_impedance=cfg.use_impedance,
                stiffness=stiffness[_LEFT_ARM_SLICE],
                torque_limit=torque_limit[_LEFT_ARM_SLICE],
            )
        )
        has_command = True

    return body, has_command


def build_mobility_command(
    rby: Any,
    linear: np.ndarray,
    angular: float,
    minimum_time: float,
    hold_time: float = _CONTROL_HOLD_TIME,
) -> Any:
    """Build an SE2 base-velocity command for the omnidirectional platform.

    ``linear`` is a 2-vector (x, y) in m/s (robot frame); ``angular`` is the
    yaw rate in rad/s.
    """
    return (
        rby.SE2VelocityCommandBuilder()
        .set_command_header(
            rby.CommandHeaderBuilder().set_control_hold_time(hold_time)
        )
        .set_velocity(linear, angular)
        .set_minimum_time(minimum_time)
    )


def build_ready_pose_command(
    rby: Any,
    body_position: np.ndarray,
    head_position: np.ndarray,
    minimum_time: float,
    hold_time: float,
) -> Any:
    """Build a joint-position command that drives the body and head to a pose.

    Returns a ``ComponentBasedCommandBuilder``; the caller wraps it in a
    ``RobotCommandBuilder`` before sending.
    """
    return (
        rby.ComponentBasedCommandBuilder()
        .set_body_command(
            rby.JointPositionCommandBuilder()
            .set_command_header(
                rby.CommandHeaderBuilder().set_control_hold_time(hold_time)
            )
            .set_position(body_position)
            .set_minimum_time(minimum_time)
        )
        .set_head_command(
            rby.JointPositionCommandBuilder()
            .set_command_header(
                rby.CommandHeaderBuilder().set_control_hold_time(hold_time)
            )
            .set_position(head_position)
            .set_minimum_time(minimum_time)
        )
    )
