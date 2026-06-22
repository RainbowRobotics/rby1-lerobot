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
# Physical joint limits from the rby1m URDF (radians).
# Used to clip outgoing actions so the leader-arm output cannot exceed the
# follower robot's mechanical range.
# ---------------------------------------------------------------------------

TORSO_Q_MIN = np.array(
    [-0.261799388, -0.523598776, -2.617993878, -0.785398163, -0.523598776, -2.356194490]
)
TORSO_Q_MAX = np.array(
    [ 0.261799388,  1.570796327,  1.570796327,  1.570796327,  0.523598776,  2.356194490]
)

RIGHT_ARM_Q_MIN = np.array(
    [-3.141592654, -3.141592654, -3.141592654, -2.617993878, -3.141592654, -1.570796327, -2.705260340]
)
RIGHT_ARM_Q_MAX = np.array(
    [ 3.141592654,  0.017453293,  3.141592654,  0.017453293,  3.141592654,  1.919862177,  2.705260340]
)

LEFT_ARM_Q_MIN = np.array(
    [-3.141592654, -0.017453293, -3.141592654, -2.617993878, -3.141592654, -1.570796327, -2.705260340]
)
LEFT_ARM_Q_MAX = np.array(
    [ 3.141592654,  3.141592654,  3.141592654,  0.017453293,  3.141592654,  1.919862177,  2.705260340]
)

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
