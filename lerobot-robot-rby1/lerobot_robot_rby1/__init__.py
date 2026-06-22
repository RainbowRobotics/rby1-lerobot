"""LeRobot Robot implementation for the Rainbow Robotics RB-Y1.

Exposes the :class:`Rby1` follower robot (registered with LeRobot as
``--robot.type rby1``), its :class:`Rby1Config` configuration, and the
:class:`Rby1Gripper` Dynamixel driver (also used by the RB-Y1 teleoperator).
"""

from .config_rby1 import Rby1Config
from .gripper import Rby1Gripper
from .model_probe import detect_model_and_version, resolve_model_version
from .rby1 import Rby1

__all__ = [
    "Rby1",
    "Rby1Config",
    "Rby1Gripper",
    "detect_model_and_version",
    "resolve_model_version",
]
