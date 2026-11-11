"""Shared constants for the RB-Y1 teleoperator package.

This module holds values that are *physical* properties of the robot (joint
counts, joint names, URDF position limits) and therefore should not live in
user-tunable config dataclasses.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Degrees of freedom
# ---------------------------------------------------------------------------

TORSO_DOF = 6
ARM_DOF = 7
TOTAL_BODY_DOF = TORSO_DOF + 2 * ARM_DOF  # 20

# ---------------------------------------------------------------------------
# Joint names — must match `lerobot_robot_rby1.Rby1.action_features`.
# ---------------------------------------------------------------------------

TORSO_NAMES: list[str] = [f"torso_{i}" for i in range(TORSO_DOF)]
RIGHT_ARM_NAMES: list[str] = [f"right_arm_{i}" for i in range(ARM_DOF)]
LEFT_ARM_NAMES: list[str] = [f"left_arm_{i}" for i in range(ARM_DOF)]
GRIPPER_NAMES: list[str] = ["right_gripper_0", "left_gripper_0"]
HEAD_NAMES: list[str] = ["head_0", "head_1"]  # pan (yaw), tilt (pitch)

# LeRobot key convention: joint / gripper positions carry a ".pos" suffix,
# matching `lerobot_robot_rby1.constants.POS_SUFFIX`. lerobot-rollout filters
# the robot features by this suffix, so bare names would be dropped.
POS_SUFFIX = ".pos"
# Nullspace-hint keys emitted by the Isaac teleop device (`<joint>.null`, rad);
# consumed by the follower's Cartesian solver, recorded only on request.
NULL_SUFFIX = ".null"
# One-shot joint-space "return to the ready pose" request to the follower
# (value = minimum motion time, s); matches lerobot_robot_rby1.constants.
RESET_CMD_KEY = "reset.cmd"

# Mobile-base velocity action keys (body frame), matching the follower robot
# and the LeRobot LeKiwi convention: linear x / y (m/s) and yaw rate (rad/s).
BASE_VEL_NAMES: list[str] = ["x.vel", "y.vel", "theta.vel"]

# End-effector action keys emitted by the VR teleoperator (and consumed by
# the follower robot in action_mode="ee"): pose of each component in the
# robot base frame, following the LeRobot EE convention — position in metres
# plus a rotation vector in radians.
EE_SUFFIXES: list[str] = ["x", "y", "z", "wx", "wy", "wz"]
TORSO_EE_NAMES: list[str] = [f"torso_ee.{s}" for s in EE_SUFFIXES]
RIGHT_EE_NAMES: list[str] = [f"right_ee.{s}" for s in EE_SUFFIXES]
LEFT_EE_NAMES: list[str] = [f"left_ee.{s}" for s in EE_SUFFIXES]

# ---------------------------------------------------------------------------
# Version-specific leader-arm init (ready) pose, in degrees (7 DOF per arm).
#
# The single source of truth for the ready pose is the follower robot package
# (lerobot_robot_rby1.constants.ready_pose_for_version). The leader arm output
# adds a per-side wrist offset on the last joint before sending it to the robot
# (see Rby1LeaderArm.get_action), so the leader init pose subtracts that offset
# on joint 6 — guaranteeing the robot lands exactly on its ready pose.
# ---------------------------------------------------------------------------

# Supported leader-arm firmware versions (matches the robot package).
SUPPORTED_LEADER_VERSIONS: tuple[str, ...] = ("1.0", "1.1", "1.2", "1.3")


def leader_init_pose_for_version(
    version: str,
    right_wrist_offset_deg: float = 0.0,
    left_wrist_offset_deg: float = 0.0,
) -> tuple[list[float], list[float]]:
    """Return ``(right_init_q_deg, left_init_q_deg)`` for the given version.

    The pose is taken from the follower robot's ready pose
    (:func:`lerobot_robot_rby1.constants.ready_pose_for_version`) so the two
    packages never drift apart. The leader output applies a wrist offset on the
    last joint, so it is subtracted here so the robot reaches its ready pose.
    """
    from lerobot_robot_rby1.constants import ready_pose_for_version

    _, right_ready, left_ready, _ = ready_pose_for_version(version)
    right_deg = list(np.rad2deg(right_ready))
    left_deg = list(np.rad2deg(left_ready))
    right_deg[6] -= right_wrist_offset_deg
    left_deg[6] -= left_wrist_offset_deg
    return [float(v) for v in right_deg], [float(v) for v in left_deg]


def leader_torso_target_for_version(version: str) -> list[float]:
    """Torso joints (degrees) of the follower robot's ready pose for ``version``."""
    from lerobot_robot_rby1.constants import ready_pose_for_version

    body, _, _, _ = ready_pose_for_version(version)
    return [float(v) for v in np.rad2deg(body[:TORSO_DOF])]
