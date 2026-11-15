"""LeRobot Teleoperator implementations for the Rainbow Robotics RB-Y1.

This package provides three teleoperators registered with LeRobot:

    * :class:`Rby1LeaderArm` — a 14-DOF master-arm bilateral teleoperator.
    * :class:`Rby1VR`        — a Meta-Quest teleoperator that emits
                                end-effector pose actions (executed by the
                                follower robot in ``action_mode="ee"``),
                                with optional mobile-base control.
    * :class:`Rby1VRWholebody` — a Meta-Quest body + hand tracking
                                teleoperator driving torso, arms, head,
                                grippers and base together (follower
                                ``action_mode="wholebody"``).

Use them via the standard LeRobot config registration, e.g. ``--teleop.type
rby1_vr``, ``rby1_vr_wholebody`` or ``rby1_leader_arm``.
"""

from .config_rby1_leader_arm import Rby1LeaderArmConfig
from .config_rby1_vr import Rby1VRConfig
from .config_rby1_vr_wholebody import Rby1VRWholebodyConfig
from .rby1_leader_arm import Rby1LeaderArm
from .rby1_vr import Rby1VR
from .rby1_vr_wholebody import Rby1VRWholebody
from .vr_receiver import VRReceiver

__all__ = [
    "Rby1LeaderArm",
    "Rby1LeaderArmConfig",
    "Rby1VR",
    "Rby1VRConfig",
    "Rby1VRWholebody",
    "Rby1VRWholebodyConfig",
    "VRReceiver",
]
