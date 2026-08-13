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
# EE action mode defaults (Cartesian impedance).
# ---------------------------------------------------------------------------

# Nullspace arm targets (degrees) used by the per-component Cartesian solver.
DEFAULT_NULL_RIGHT_DEG = [40.0, -30.0, -5.0, -135.0, -10.0, 20.0,  40.0]
DEFAULT_NULL_LEFT_DEG  = [40.0,  30.0, -5.0, -135.0,  10.0, 20.0, -40.0]

# Joint limits enforced by the Cartesian impedance solvers (radians).
DEFAULT_WB_JOINT_LIMITS: JointLimits = {
    "right_arm_3": (-2.6, -0.5),
    "right_arm_5": (0.2, 1.9),
    "left_arm_3":  (-2.6, -0.5),
    "left_arm_5":  (0.2, 1.9),
    "torso_1":     (-0.523598776, 1.3),
    "torso_2":     (-2.617993878, -0.2),
}

DEFAULT_TORSO_JOINT_LIMITS: JointLimits = {
    "torso_1": (-0.523598776, 1.6),
    "torso_2": (-2.617993878, -0.2),
}

DEFAULT_RIGHT_ARM_JOINT_LIMITS: JointLimits = {"right_arm_3": (-2.6, -0.5)}
DEFAULT_LEFT_ARM_JOINT_LIMITS: JointLimits  = {"left_arm_3":  (-2.6, -0.5)}

# Cartesian add_target gain tuples for rby1_sdk add_target(...):
# (linear_acc_limit, linear_vel_limit, angular_acc_limit, angular_vel_limit).
_PI = math.pi
TORSO_TARGET_GAINS_WB = (1.0, _PI * 0.5, 10.0, _PI * 20.0)
TORSO_TARGET_GAINS_PC = (1.0, _PI * 0.5, 20.0, _PI * 40.0)
ARM_TARGET_GAINS_WB   = (2.0, _PI * 2.0,  20.0, _PI * 80.0)
ARM_TARGET_GAINS_PC   = (3.0, _PI * 2.0, 150.0, _PI * 80.0)

# ---------------------------------------------------------------------------
# Dynamixel gripper bus (two motors on /dev/rby1_gripper).
# Motor ID 0 = right hand, ID 1 = left hand.
# ---------------------------------------------------------------------------

GRIPPER_BAUD_RATE = 2_000_000
GRIPPER_IDS = [0, 1]
GRIPPER_HOMING_TORQUE = 0.46     # Nm, applied during the homing sweeps
GRIPPER_HOMING_STEPS = 30        # 0.1 s x 30 = 3 s per direction
GRIPPER_POSITION_TORQUE = 0.46   # Nm, max torque in position mode
