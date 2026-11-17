"""What the teleoperation core needs to know about the robot."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RobotState:
    """Measured poses (4x4, base frame) and joints (rad) at one instant."""

    torso: np.ndarray      # base -> link_torso_5
    right_ee: np.ndarray   # base -> link_right_arm_6
    left_ee: np.ndarray    # base -> link_left_arm_6
    head_q: np.ndarray     # [head_0, head_1]
    right_q: np.ndarray | None = None  # (7,) arm joints, for the posture hints
    left_q: np.ndarray | None = None
    # Base odometry as a 3x3 SE(2) homogeneous transform (None when the robot reports none).
    odometry: np.ndarray | None = None


@dataclass(frozen=True)
class RobotInfo:
    """Facts about the robot that do not change during a session."""

    version: str = "1.3"     # "1.2" | "1.3": selects the arm geometry
    holonomic: bool = True   # False for a differential base (Model A): no lateral velocity
    # Shoulder -> link_<side>_arm_6 distance of the straight arm (m) per side, as
    # measured on the robot; None uses the built-in constant for ``version``.
    arm_reach: Mapping[str, float] | None = None
