"""Shared constants for the RB Meta Quest teleoperator.

The action feature names are imported from ``lerobot_robot_rb`` so that the
teleoperator and follower robot cannot silently drift to different key names.

Unit conventions
----------------
* Joint angles: radians at the LeRobot boundary.
* Cartesian positions used by the legacy RB10E IK: millimetres.
* Rotation matrices: dimensionless 3x3 matrices.
* Meta Quest trigger/grip values: normalised to [0, 1].
"""

from __future__ import annotations

import math
from typing import Final

from lerobot_robot_rb.models import (
    DOF,
    EE_NAMES,
    GRIPPER_NAME,
    JOINT_NAMES,
)


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

DEG2RAD: Final[float] = math.pi / 180.0
RAD2DEG: Final[float] = 180.0 / math.pi


# ---------------------------------------------------------------------------
# LeRobot feature names
# ---------------------------------------------------------------------------

JOINT_ACTION_NAMES: Final[tuple[str, ...]] = tuple(JOINT_NAMES)
EE_ACTION_NAMES: Final[tuple[str, ...]] = tuple(EE_NAMES)

if len(JOINT_ACTION_NAMES) != DOF:
    raise RuntimeError(
        f"Expected {DOF} RB joint names, got {len(JOINT_ACTION_NAMES)}: "
        f"{JOINT_ACTION_NAMES}"
    )


# ---------------------------------------------------------------------------
# Existing RB10E teleoperation defaults
# ---------------------------------------------------------------------------

# Existing startup / IK seed pose used by RB10E_TELEOP_VIZ.py.
DEFAULT_READY_POSE_DEG: Final[tuple[float, ...]] = (
    40.0,
    -70.0,
    -100.0,
    160.0,
    -60.0,
    0.0,
)

DEFAULT_READY_POSE_RAD: Final[tuple[float, ...]] = tuple(
    value * DEG2RAD for value in DEFAULT_READY_POSE_DEG
)

# Joint limits copied from the existing RB10E IK implementation.
#
# These limits belong to the numerical IK solver. The paired RbCobot follower
# applies its own model-specific limits again before sending a command.
RB10E_JOINT_LIMITS_DEG: Final[tuple[tuple[float, float], ...]] = (
    (-360.0, 360.0),
    (-180.0, 180.0),
    (-154.0, 154.0),
    (-360.0, 360.0),
    (-360.0, 360.0),
    (-360.0, 360.0),
)

RB10E_JOINT_LIMITS_RAD: Final[tuple[tuple[float, float], ...]] = tuple(
    (
        lower_deg * DEG2RAD,
        upper_deg * DEG2RAD,
    )
    for lower_deg, upper_deg in RB10E_JOINT_LIMITS_DEG
)


# ---------------------------------------------------------------------------
# VR controls
# ---------------------------------------------------------------------------

RIGHT_HAND: Final[str] = "right"
LEFT_HAND: Final[str] = "left"

PRIMARY_BUTTON: Final[str] = "primaryButton"
SECONDARY_BUTTON: Final[str] = "secondaryButton"
TRIGGER_BUTTON: Final[str] = "trigger"
GRIP_BUTTON: Final[str] = "grip"


# ---------------------------------------------------------------------------
# IK defaults
# ---------------------------------------------------------------------------

DEFAULT_IK_ITERATIONS: Final[int] = 3

# The existing implementation used:
#
# H = J.T @ J + quadratic_error * I + 2.0 * I
#
# These constants retain that behaviour while making it configurable.
DEFAULT_IK_BASE_DAMPING: Final[float] = 2.0
DEFAULT_IK_ERROR_DAMPING_GAIN: Final[float] = 1.0


def make_joint_action_features(
    *,
    use_gripper: bool,
) -> dict[str, type]:
    """Return the LeRobot action schema emitted by the VR teleoperator."""

    features: dict[str, type] = {
        name: float for name in JOINT_ACTION_NAMES
    }

    if use_gripper:
        features[GRIPPER_NAME] = float

    return features
