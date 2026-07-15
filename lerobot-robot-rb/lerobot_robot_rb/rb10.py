"""Registration shim exposing the RB10 as ``--robot.type=rb10``.

To support another RB-Series model, copy this module (e.g. ``rb5.py``),
change the registered name and the ``model`` default, add a matching entry
to :data:`~lerobot_robot_rb.models.MODEL_SPECS`, and export the new classes
from ``__init__.py``.  The shared implementation in ``rb_cobot.py`` needs no
changes.
"""

from dataclasses import dataclass

from lerobot.robots.config import RobotConfig

from .config_rb import RbCobotConfig
from .rb_cobot import RbCobot


@RobotConfig.register_subclass("rb10")
@dataclass
class Rb10Config(RbCobotConfig):
    model: str = "rb10"


class Rb10(RbCobot):
    config_class = Rb10Config
    name = "rb10"
