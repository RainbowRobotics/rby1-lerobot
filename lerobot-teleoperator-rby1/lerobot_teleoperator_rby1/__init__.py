"""LeRobot Teleoperator implementations for the Rainbow Robotics RB-Y1.

This package provides three teleoperators registered with LeRobot:

    * :class:`Rby1LeaderArm` — a 14-DOF master-arm bilateral teleoperator.
    * :class:`Rby1VR`        — a Meta-Quest teleoperator that emits
                                end-effector pose actions (executed by the
                                follower robot in ``action_mode="ee"``),
                                with optional mobile-base control.
    * :class:`Rby1XR`  — an NVIDIA Isaac Teleop (CloudXR / OpenXR)
                                device: both controllers → arm EE poses,
                                headset → head joints, body tracking → torso.

Use them via the standard LeRobot config registration, e.g. ``--teleop.type
rby1_isaac``, ``--teleop.type rby1_vr`` or ``--teleop.type rby1_leader_arm``.
"""

from .config_rby1_leader_arm import Rby1LeaderArmConfig
from .config_rby1_vr import Rby1VRConfig
from .isaac_teleop import Rby1XRConfig, Rby1XR
from .rby1_leader_arm import Rby1LeaderArm
from .rby1_vr import Rby1VR
from .vr_receiver import VRReceiver

__all__ = [
    "Rby1LeaderArm",
    "Rby1LeaderArmConfig",
    "Rby1VR",
    "Rby1VRConfig",
    "Rby1XRConfig",
    "Rby1XR",
    "VRReceiver",
]
