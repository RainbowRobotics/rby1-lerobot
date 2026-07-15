"""LeRobot Robot implementation for the Rainbow Robotics RB-Series cobots.

Exposes the :class:`Rb10` follower robot (registered with LeRobot as
``--robot.type rb10``), its :class:`Rb10Config` configuration, the shared
:class:`RbCobot` implementation used by every RB-Series model, and the
gripper drivers.
"""

from .config_rb import RbCobotConfig
from .gripper import RbDynamixelGripper, RbGripperBase, make_gripper
from .models import DOF, GRIPPER_NAME, JOINT_NAMES, MODEL_SPECS, RbModelSpec
from .rb10 import Rb10, Rb10Config
from .rb_cobot import RbCobot, clamp_target

__all__ = [
    "DOF",
    "GRIPPER_NAME",
    "JOINT_NAMES",
    "MODEL_SPECS",
    "Rb10",
    "Rb10Config",
    "RbCobot",
    "RbCobotConfig",
    "RbDynamixelGripper",
    "RbGripperBase",
    "RbModelSpec",
    "clamp_target",
    "make_gripper",
]
