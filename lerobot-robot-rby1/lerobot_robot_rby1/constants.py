"""Shared constants for the RB-Y1 robot package.

These describe *physical* properties of the robot — joint counts, joint names,
URDF position limits, the Cartesian impedance solver defaults, and the
Dynamixel gripper bus parameters.  They are deliberately kept out of the
user-tunable :class:`~lerobot_robot_rby1.config_rby1.Rby1Config` because they
describe the hardware itself rather than a user preference.

The joint names mirror ``lerobot_teleoperator_rby1.constants`` so that the
teleoperator action dict and this robot's ``action_features`` stay aligned.
"""

from __future__ import annotations

import math

import numpy as np

# Joint-limit dictionary type alias: joint name -> (lower, upper) in radians.
JointLimits = dict[str, tuple[float, float]]

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

# LeRobot key convention: joint / gripper positions carry a ".pos" suffix
# (see lerobot.robots.lekiwi). lerobot-rollout filters the robot features by
# this suffix, so bare names are dropped from the policy-facing tensors. The
# names above stay bare — they are the base for the ".pos", ".vel" and
# ".torque" observation channels.
POS_SUFFIX = ".pos"

# Mobile-base velocity action keys (body frame), following the LeRobot
# convention used by LeKiwi: linear x / y (m/s) and yaw rate (rad/s).
BASE_VEL_NAMES: list[str] = ["x.vel", "y.vel", "theta.vel"]

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
# EE action mode defaults (Cartesian impedance).
# ---------------------------------------------------------------------------

# Nullspace arm targets (degrees) used by the EE arm solvers.
DEFAULT_NULL_RIGHT_DEG = [40.0, -30.0, -5.0, -135.0, -10.0, 20.0,  40.0]
DEFAULT_NULL_LEFT_DEG  = [40.0,  30.0, -5.0, -135.0,  10.0, 20.0, -40.0]

# Joint limits enforced by the Cartesian impedance solvers (radians).
DEFAULT_TORSO_JOINT_LIMITS: JointLimits = {
    "torso_1": (-0.523598776, 1.6),
    "torso_2": (-2.617993878, -0.2),
}

DEFAULT_RIGHT_ARM_JOINT_LIMITS: JointLimits = {"right_arm_3": (-2.6, -0.5)}
DEFAULT_LEFT_ARM_JOINT_LIMITS: JointLimits  = {"left_arm_3":  (-2.6, -0.5)}

# Cartesian target limit tuples, passed positionally to rby1_sdk
# CartesianImpedanceControlCommandBuilder.add_target(ref, link, T, ...):
#   (linear_velocity_limit [m/s], angular_velocity_limit [rad/s],
#    linear_acceleration_limit [m/s^2], angular_acceleration_limit [rad/s^2])
_PI = math.pi
TORSO_TARGET_GAINS = (1.0, _PI * 0.5, 20.0, _PI * 40.0)
ARM_TARGET_GAINS   = (3.0, _PI * 2.0, 150.0, _PI * 80.0)

# ---------------------------------------------------------------------------
# Dynamixel gripper bus (two motors on /dev/rby1_gripper).
# Motor ID 0 = right hand, ID 1 = left hand.
# ---------------------------------------------------------------------------

GRIPPER_BAUD_RATE = 2_000_000
GRIPPER_IDS = [0, 1]
GRIPPER_HOMING_TORQUE = 0.46     # Nm, applied during the homing sweeps
GRIPPER_HOMING_STEPS = 30        # 0.1 s x 30 = 3 s per direction
GRIPPER_POSITION_TORQUE = 0.46   # Nm, max torque in position mode
GRIPPER_PING_TIMEOUT_S = 10.0    # wait for the motors to boot after flange power-on

# ---------------------------------------------------------------------------
# Ready pose (radians) — a safe, known configuration the robot moves to on
# connect.
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
