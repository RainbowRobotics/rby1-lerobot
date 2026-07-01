"""Shared constants for the RB-Y1 robot package.

These describe *physical* properties of the robot — joint counts, joint names,
URDF position limits, the default ready pose, and the Dynamixel gripper bus
parameters.  They are deliberately kept out of the user-tunable
:class:`~lerobot_robot_rby1.config_rby1.Rby1Config` because they describe the
hardware itself rather than a user preference.

The joint names mirror ``lerobot_teleoperator_rby1.constants`` so that the
teleoperator action dict and this robot's ``action_features`` stay aligned.
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
# Joint names — must match the teleoperator output and the dataset keys.
# ---------------------------------------------------------------------------

TORSO_NAMES: list[str] = [f"torso_{i}" for i in range(TORSO_DOF)]
RIGHT_ARM_NAMES: list[str] = [f"right_arm_{i}" for i in range(ARM_DOF)]
LEFT_ARM_NAMES: list[str] = [f"left_arm_{i}" for i in range(ARM_DOF)]
GRIPPER_NAMES: list[str] = ["right_gripper_0", "left_gripper_0"]

# Mobile-base velocity action keys (body frame), following the LeRobot
# convention used by LeKiwi: linear x / y (m/s) and yaw rate (rad/s).
BASE_VEL_NAMES: list[str] = ["x.vel", "y.vel", "theta.vel"]

# Mobile-base pose observation keys, flattened from the (3, 3) SE(2)
# homogeneous odometry matrix reported by rby1-sdk (RobotState.odometry):
# planar position x / y (m) and heading theta (rad) in the odometry frame.
BASE_POSE_NAMES: list[str] = ["base_x.pos", "base_y.pos", "base_theta.pos"]

# End-effector action keys (``action_mode="ee"``): pose of each enabled
# component in the robot base frame, following the LeRobot EE convention —
# position in metres plus a rotation vector in radians.
EE_SUFFIXES: list[str] = ["x", "y", "z", "wx", "wy", "wz"]
TORSO_EE_NAMES: list[str] = [f"torso_ee.{s}" for s in EE_SUFFIXES]
RIGHT_EE_NAMES: list[str] = [f"right_ee.{s}" for s in EE_SUFFIXES]
LEFT_EE_NAMES: list[str] = [f"left_ee.{s}" for s in EE_SUFFIXES]

# Body joints in command order: torso (6) | right arm (7) | left arm (7).
ALL_JOINT_NAMES: list[str] = TORSO_NAMES + RIGHT_ARM_NAMES + LEFT_ARM_NAMES

# ---------------------------------------------------------------------------
# Physical joint limits from the rby1m URDF (radians).
# Order: torso (6) | right_arm (7) | left_arm (7).
# Outgoing position commands are clipped to these bounds.
# ---------------------------------------------------------------------------

TORSO_Q_MIN = np.array(
    [-0.261799388, -0.523598776, -2.617993878, -0.785398163, -0.523598776, -2.356194490]
)
TORSO_Q_MAX = np.array(
    [0.261799388, 1.570796327, 1.570796327, 1.570796327, 0.523598776, 2.356194490]
)

RIGHT_ARM_Q_MIN = np.array(
    [-3.141592654, -3.141592654, -3.141592654, -2.617993878, -3.141592654, -1.570796327, -2.705260340]
)
RIGHT_ARM_Q_MAX = np.array(
    [3.141592654, 0.017453293, 3.141592654, 0.017453293, 3.141592654, 1.919862177, 2.705260340]
)

LEFT_ARM_Q_MIN = np.array(
    [-3.141592654, -0.017453293, -3.141592654, -2.617993878, -3.141592654, -1.570796327, -2.705260340]
)
LEFT_ARM_Q_MAX = np.array(
    [3.141592654, 3.141592654, 3.141592654, 0.017453293, 3.141592654, 1.919862177, 2.705260340]
)

# ---------------------------------------------------------------------------
# Ready pose (radians) — a safe, known configuration the robot moves to on
# connect and (optionally per-arm) between record episodes.
#
# RB-Y1 v1.2 and v1.3 differ in the joint configuration near the wrist, so each
# version needs its own arm ready pose. The v1.3 pose zeroes the three wrist
# joints (arm_4, arm_5, arm_6). The torso and head poses are version-independent.
# ---------------------------------------------------------------------------

READY_TORSO = np.deg2rad([0.0, 0.0, 0.0, 20.0, 0.0, 0.0])
READY_HEAD = np.deg2rad([0.0, 49.0])  # head_0, head_1

# v1.2 (and earlier) arm ready pose.
READY_RIGHT = np.deg2rad([15.0, -65.0, -15.0, -115.0, 75.0, -65.0, -5.0])
READY_LEFT = np.deg2rad([15.0, 65.0, 15.0, -115.0, -75.0, -65.0, -5.0])
READY_POSE = np.concatenate([READY_TORSO, READY_RIGHT, READY_LEFT])  # (20,)

# v1.3 arm ready pose: wrist joints (arm_4, arm_5, arm_6) are zeroed.
READY_RIGHT_V13 = np.deg2rad([15.0, -65.0, -15.0, -115.0, 0.0, 0.0, 0.0])
READY_LEFT_V13 = np.deg2rad([15.0, 65.0, 15.0, -115.0, 0.0, 0.0, 0.0])
READY_POSE_V13 = np.concatenate([READY_TORSO, READY_RIGHT_V13, READY_LEFT_V13])  # (20,)


def ready_pose_for_version(version: str):
    """Return ``(body_pose, right_arm, left_arm, head)`` for a firmware version.

    ``body_pose`` is the 20-DOF [torso | right arm | left arm] vector. v1.3 uses
    the wrist-zeroed arm pose; every other version uses the v1.2 pose.
    """
    if (version or "").strip() == "1.3":
        return READY_POSE_V13, READY_RIGHT_V13, READY_LEFT_V13, READY_HEAD
    return READY_POSE, READY_RIGHT, READY_LEFT, READY_HEAD


# ---------------------------------------------------------------------------
# Dynamixel gripper bus (two motors on /dev/rby1_gripper).
# Motor ID 0 = right hand, ID 1 = left hand.
# ---------------------------------------------------------------------------

GRIPPER_BAUD_RATE = 2_000_000
GRIPPER_IDS = [0, 1]
GRIPPER_HOMING_TORQUE = 0.46     # Nm, applied during the homing sweeps
GRIPPER_HOMING_STEPS = 30        # 0.1 s x 30 = 3 s per direction
GRIPPER_POSITION_TORQUE = 0.46   # Nm, max torque in position mode
