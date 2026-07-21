"""LeRobot Robot implementation for the Rainbow Robotics RB-Series cobots.

Exposes the :class:`Rb10` follower robot (registered with LeRobot as
``--robot.type rb10``), its :class:`Rb10Config` configuration, the shared
:class:`RbCobot` implementation used by every RB-Series model, the
software-only ``rb_auto`` teleoperator, and the gripper drivers.
"""

from .auto_teleop import RbAutoTeleop, RbAutoTeleopConfig
from .config_rb import RbCobotConfig
from .gripper import RbDynamixelGripper, RbGripperBase, make_gripper
from .models import (
    DOF,
    EE_NAMES,
    GRIPPER_NAME,
    JOINT_NAMES,
    MODEL_SPECS,
    RbModelSpec,
)
from .rb10 import Rb10, Rb10Config
from .rb_cobot import (
    RbCobot,
    clamp_ee_target,
    clamp_target,
    kinematics_estop_only,
    wrap_deg,
)

__all__ = [
    "DOF",
    "EE_NAMES",
    "GRIPPER_NAME",
    "JOINT_NAMES",
    "MODEL_SPECS",
    "RbAutoTeleop",
    "RbAutoTeleopConfig",
    "Rb10",
    "Rb10Config",
    "RbCobot",
    "RbCobotConfig",
    "RbDynamixelGripper",
    "RbGripperBase",
    "RbModelSpec",
    "clamp_ee_target",
    "clamp_target",
    "kinematics_estop_only",
    "make_gripper",
    "wrap_deg",
]
