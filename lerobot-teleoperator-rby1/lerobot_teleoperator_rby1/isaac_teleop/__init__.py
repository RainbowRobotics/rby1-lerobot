"""NVIDIA Isaac Teleop devices for the RB-Y1, exposed as LeRobot teleoperators.

Mirrors the layout of the upstream LeRobot example
``examples/isaac_teleop_to_so101/isaac_teleop``:

* :mod:`base` — :class:`IsaacTeleopTeleoperator`, the shared CloudXR /
  ``TeleopSession`` lifecycle.
* :mod:`clutch` — :class:`Clutch`, the engage-relative pose clutch.
* :mod:`config_isaac_teleop` — the config dataclasses; :class:`Rby1XRConfig`
  registers ``--teleop.type=rby1_isaac`` in the *global* LeRobot registry so the
  device works with the stock ``lerobot-teleoperate`` / ``lerobot-record`` CLIs.
* :mod:`teleop_rby1_xr` — :class:`Rby1XR`, the RB-Y1 device (both
  controllers → arm EE poses, headset → head joints, body tracking → torso).

``isaacteleop`` is optional: every module here imports without it; constructing
a device fails fast with install instructions.
"""

from .base import IsaacTeleopTeleoperator
from .clutch import Clutch
from .config_isaac_teleop import IsaacTeleopConfig, Rby1XRConfig
from .teleop_rby1_xr import Rby1XR, Rby1XRTeleop

__all__ = [
    "Clutch",
    "IsaacTeleopConfig",
    "IsaacTeleopTeleoperator",
    "Rby1XRConfig",
    "Rby1XR",
    "Rby1XRTeleop",
]
