"""Builders for the ``rby1_sdk`` command objects sent to the robot.

Each function receives the SDK module (``rby``) plus already-computed joint
or Cartesian targets and returns a ready-to-send command builder.  Isolating
the (verbose) command DSL here keeps :class:`~lerobot_robot_rby1.rby1.Rby1`
focused on high-level control flow.

Joint-mode actions are executed via :func:`build_body_command`; EE-mode
actions (``action_mode="ee"``) are decoded with :func:`pose_from_action` and
executed via :func:`build_ee_command`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from .constants import (
    EE_SUFFIXES,
    LEFT_ARM_NAMES,
    LEFT_ARM_Q_MAX,
    LEFT_ARM_Q_MIN,
    POS_SUFFIX,
    RIGHT_ARM_NAMES,
    RIGHT_ARM_Q_MAX,
    RIGHT_ARM_Q_MIN,
    TORSO_NAMES,
    TORSO_Q_MAX,
    TORSO_Q_MIN,
)

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
    """Per-solver (torso / right arm / left arm) ``set_reset_reference`` flags for the next command."""

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


def build_ee_command(
    rby: Any,
    cfg: Any,
    targets: CartesianTargets,
    reset: ResetFlags,
    null_right: np.ndarray,
    null_left: np.ndarray,
) -> Any:
    """Cartesian impedance command with separate torso / right-arm / left-arm solvers."""
    dt = cfg.ee_action_period
    body = rby.BodyComponentBasedCommandBuilder()

    if cfg.use_torso:
        torso_builder = (
            rby.CartesianImpedanceControlCommandBuilder()
            .set_command_header(_ee_command_header(rby, cfg.ee_hold_time))
            .set_minimum_time(dt * cfg.ee_min_time_factor)
            .set_joint_stiffness(cfg.ee_torso_stiffness)
            .set_joint_torque_limit(cfg.ee_torso_torque_limit)
            .set_stop_joint_position_tracking_error(0)
            .set_stop_orientation_tracking_error(0)
            .set_joint_damping_ratio(cfg.ee_torso_damping_ratio)
            .set_reset_reference(reset.torso)
        )
        _apply_joint_limits(torso_builder, cfg.ee_torso_joint_limits)
        torso_builder.add_target(
            "base", "link_torso_5", targets.torso,
            *cfg.ee_torso_target_gains,
        )
        body.set_torso_command(torso_builder)

    if cfg.use_right_arm:
        right_builder = (
            rby.CartesianImpedanceControlCommandBuilder()
            .set_command_header(_ee_command_header(rby, cfg.ee_hold_time))
            .set_minimum_time(dt * cfg.ee_min_time_factor)
            .set_joint_stiffness(cfg.ee_arm_stiffness)
            .set_joint_torque_limit(cfg.ee_arm_torque_limit)
            .set_nullspace_joint_target(
                null_right,
                np.asarray(cfg.ee_nullspace_weight),
                cfg.ee_nullspace_stiffness,
                cfg.ee_nullspace_damping_ratio,
            )
            .set_stop_joint_position_tracking_error(0)
            .set_stop_orientation_tracking_error(0)
            .set_joint_damping_ratio(cfg.ee_arm_damping_ratio)
            .set_reset_reference(reset.right_arm)
        )
        _apply_joint_limits(right_builder, cfg.ee_right_arm_joint_limits)
        right_builder.add_target(
            "base", "link_right_arm_6", targets.right_arm,
            *cfg.ee_arm_target_gains,
        )
        body.set_right_arm_command(right_builder)

    if cfg.use_left_arm:
        left_builder = (
            rby.CartesianImpedanceControlCommandBuilder()
            .set_command_header(_ee_command_header(rby, cfg.ee_hold_time))
            .set_minimum_time(dt * cfg.ee_min_time_factor)
            .set_joint_stiffness(cfg.ee_arm_stiffness)
            .set_joint_torque_limit(cfg.ee_arm_torque_limit)
            .set_nullspace_joint_target(
                null_left,
                np.asarray(cfg.ee_nullspace_weight),
                cfg.ee_nullspace_stiffness,
                cfg.ee_nullspace_damping_ratio,
            )
            .set_stop_joint_position_tracking_error(0)
            .set_stop_orientation_tracking_error(0)
            .set_joint_damping_ratio(cfg.ee_arm_damping_ratio)
            .set_reset_reference(reset.left_arm)
        )
        _apply_joint_limits(left_builder, cfg.ee_left_arm_joint_limits)
        left_builder.add_target(
            "base", "link_left_arm_6", targets.left_arm,
            *cfg.ee_arm_target_gains,
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
    damping_ratio: float = 1.0,
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
            .set_damping_ratio(damping_ratio)
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


def clip_joint_targets(
    action: dict[str, Any],
    names: list[str],
    q_min: np.ndarray,
    q_max: np.ndarray,
) -> np.ndarray:
    """Joint targets for one group from ``<name>.pos`` keys, clipped to the URDF limits."""
    return np.clip(np.array([action[f"{name}{POS_SUFFIX}"] for name in names]), q_min, q_max)


def _joint_groups(cfg: Any) -> list[tuple[str, list[str], np.ndarray, np.ndarray]]:
    """``(group, joint names, q_min, q_max)`` of the groups joint mode commands, in action order."""
    groups = []
    if cfg.use_torso:
        groups.append(("torso", TORSO_NAMES, TORSO_Q_MIN, TORSO_Q_MAX))
    if cfg.use_right_arm:
        groups.append(("right_arm", RIGHT_ARM_NAMES, RIGHT_ARM_Q_MIN, RIGHT_ARM_Q_MAX))
    if cfg.use_left_arm:
        groups.append(("left_arm", LEFT_ARM_NAMES, LEFT_ARM_Q_MIN, LEFT_ARM_Q_MAX))
    return groups


def clipped_joint_action(cfg: Any, action: dict[str, Any]) -> dict[str, float]:
    """The ``<joint>.pos`` targets ``build_body_command`` sends, after clipping."""
    sent: dict[str, float] = {}
    for _, names, q_min, q_max in _joint_groups(cfg):
        q = clip_joint_targets(action, names, q_min, q_max)
        sent.update({f"{n}{POS_SUFFIX}": float(v) for n, v in zip(names, q)})
    return sent


def build_body_command(
    rby: Any,
    cfg: Any,
    action: dict[str, Any],
    model: Any,
    max_qdot: np.ndarray,
    max_qddot: np.ndarray,
    minimum_time: float,
) -> tuple[Any, bool]:
    """Assemble a joint-mode body command for the enabled groups.

    The torso (when ``use_torso``), right arm and left arm each get their own
    joint position / joint impedance command.  Disabled groups are omitted
    entirely so the SDK holds them at their current position.  Joint targets
    are clipped to the URDF limits before being sent.

    Returns
    -------
    (body, has_command)
        ``body`` is a ``BodyComponentBasedCommandBuilder``; ``has_command`` is
        True when at least one group command was added (so the caller can skip
        an empty body).
    """
    acc_scale = cfg.joint_acceleration_limit_scale
    body = rby.BodyComponentBasedCommandBuilder()
    has_command = False

    for group, names, q_min, q_max in _joint_groups(cfg):
        idx = getattr(model, f"{group}_idx")
        kind = "torso" if group == "torso" else "arm"
        limb = build_limb_command(
            rby,
            position=clip_joint_targets(action, names, q_min, q_max),
            velocity_limit=max_qdot[idx],
            acceleration_limit=max_qddot[idx] * acc_scale,
            minimum_time=minimum_time,
            use_impedance=cfg.use_joint_impedance,
            stiffness=getattr(cfg, f"joint_impedance_{kind}_stiffness"),
            torque_limit=getattr(cfg, f"joint_impedance_{kind}_torque_limit"),
            damping_ratio=cfg.joint_impedance_damping_ratio,
        )
        getattr(body, f"set_{group}_command")(limb)
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
