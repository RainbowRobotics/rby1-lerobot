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
# VR home pose
# ---------------------------------------------------------------------------

# Pose the arm moves to when the operator presses the primary (A) button.
#
# ORIGIN: hand-guided on the real robot, then snapped. Read back live from
# the control box on 2026-09-16 with program_mode == 0:
#
#     jnt_ang  = (-172.1388, -3.8135, 141.3375, 43.9742, 278.3751, 182.6252)
#     tcp_pos  = (-475.1793, 109.8054, 268.0233) mm
#     tcp Euler= (1.4627, 2.9902, -90.2875) deg
#
# The teach fixes WHERE the arm works. The orientation it landed on was
# rough: the control box's rx / ry are the tilt of the tool x-y plane out of
# the base x-y plane, and 1.46 / 2.99 deg of it is 3.21 deg of total tilt,
# with the yaw 0.29 deg off a round -90. The values below are the IK
# solution for that SAME end-effector position with the tilt taken to zero
# and the yaw snapped to exactly -90 deg. No joint moves more than 2.63 deg.
# Do not re-derive the position from geometry; it encodes where the operator
# physically put the arm.
#
# THE CONTROL BOX AND THIS MODEL AGREE ON ORIENTATION AND DISAGREE ON
# POSITION, which is worth knowing before comparing numbers with the
# pendant. Feeding the tcp Euler above through Rz @ Ry @ Rx reproduces
# get_fk()[:3, :3] to 0.15 deg, so the rotation matrix here IS the control
# box's tool rotation. The positions differ by 143.44 mm along the tool +Y
# column: this model carries the gripper inside its last link (259.3 mm from
# the wrist, against the bare flange's 115.9 mm), so get_fk() reports the
# GRIPPER TIP and tcp_pos reports the FLANGE. Snapping rotates about the tip,
# so the flange moves 3.93 mm while the tip stays put to 0.0002 mm.
#
# The end-effector orientation is exactly the base frame yawed -90 deg:
#
#     position = (-618.5657, 110.4523, 264.0060) mm    the taught tip
#     rotation = [[0,  1, 0],                    integer to within 5.8e-16
#                 [-1, 0, 0],
#                 [0,  0, 1]]
#     EE +X = base -Y,  EE +Y = base +X,  EE +Z = base +Z
#     tool +Y  = fk[:3, 1] = (1, 0, 0)           exactly base +X, zero offset
#     gripper  = (-1, 0, 0)                      horizontal, along base -X
#
# Zero tilt is the load-bearing half: the EE x-y plane is exactly the base
# x-y plane, so the operator's horizontal hand motion stays horizontal for
# the arm. The remaining -90 deg of yaw is bookkeeping, and it is what
# frame_transforms.R_OPERATOR_TO_BASE == Rz(-90 deg) cancels to give the
# operator
#
#     operator +X (forward)  ->  home EE -Y   (the gripper approach)
#     operator +Y (left)     ->  home EE +X
#     operator +Z (up)       ->  home EE +Z
#
# Change this orientation and that constant has to be re-derived with it.
#
# THREE THINGS ARE LOAD-BEARING, and they are NOT the ones the previous home
# pose used. This teach is WRIST-FLIPPED -- j5 sits at 180 deg, not 0 -- and
# that changes which combinations pin the orientation:
#
#     j1 + j2 + j3 == 180     the flange pitch
#     j0 - j4     == -450     the flange yaw  (-90 mod 360)
#     j5          == 180      the wrist flip
#
# Verified by perturbation: moving j1 against j2, or j0 WITH j4, leaves the
# rotation unchanged to 3e-16, while moving j0 against j4 -- the rule the
# previous, j5 == 0 home obeyed -- tilts it. Joints 1, 2 and 3 are a planar
# chain, so their SUM sets the pitch while their individual values slide the
# wrist along that plane.
#
# If you re-tune this pose, solve j3 and j4 LAST, from the other four:
#
#     j3 = 180 - j1 - j2
#     j4 = j0 + 450
#
# Rounding all six independently breaks both relations.
#
# The gripper approach direction is ``-fk[:3, 1]``, NOT the Z column. This
# model puts every joint axis on its frame's Y column and the tool transform
# is identity, so that convention reaches get_fk() unchanged. See the frame
# convention note in ``rb10e_kinematics.RB10E.update_fk_and_jcbn``.
#
# This matters more than it looks. RbVr anchors on the Grip rising edge and
# then rotates AWAY FROM the anchor orientation by a scaled fraction of the
# hand's rotation, so the home pose's orientation is the reference every
# stroke is measured from. An axis-aligned reference keeps the operator's
# mental model simple and keeps the arithmetic exact.
#
# WATCH JOINT 2, NOT THE WRIST. At 141.89 deg it has 12.11 deg of room to
# the +-154 deg IK limit in RB10E_JOINT_LIMITS_DEG, and a 30 deg tool
# rotation about base Y spends 6.8 deg of that in one gesture. IKLM clips at
# the limit, so a stroke that asks for more saturates rather than tracks.
# The wrist is comfortable by comparison: j4 == 277.47 deg sits 82.5 deg
# from the j4 == 360 deg singularity and 97.5 deg from the 180 deg one (the
# wrist is singular at BOTH). A 30 deg rotation about base Z is what eats
# wrist margin fastest, and only in one direction: -30 deg takes j4 to
# 326.12, i.e. 33.9 deg from singular, while +30 deg opens it up.
#
# Whole-pose conditioning, linear Jacobian block in metres as in
# test_vr_home_pose_is_not_near_a_singularity: smallest singular value
# 0.212, condition number 9.32.
#
# Tracking from here, at RbVrConfig.ik_iterations == 5. A 100 mm step leaves
# 0.0095 mm on the FIRST tick along base +X, 0.18 mm along +Y and 0.18 mm
# along +Z. Rotation is cheapest about base X -- that is the flange spin
# axis fk[:3, 1] at this pose, so 18 deg leaves 1.4 percent after one tick
# -- and costs about 30 percent about base Y or base Z.
#
# These no longer resemble MODEL_SPECS["rb10"].default_ready_pose_deg in
# lerobot_robot_rb.models, and must not be replaced by it. The VR home has
# requirements -- zero tilt against the base x-y plane, away from
# singularities, facing the cell's work area, at the taught working height
# -- that the model field carries no obligation to preserve.
#
# NOTE: this is deliberately a DIFFERENT pose from DEFAULT_READY_POSE_DEG
# above, whose comment incorrectly claims to be the IK seed.
VR_HOME_POSE_DEG: Final[tuple[float, ...]] = (
    -172.5253,
    -3.3101,
    141.8867,
    41.4234,
    277.4747,
    180.0,
)

VR_HOME_POSE_RAD: Final[tuple[float, ...]] = tuple(
    value * DEG2RAD for value in VR_HOME_POSE_DEG
)

if len(VR_HOME_POSE_DEG) != DOF:
    raise RuntimeError(
        f"VR_HOME_POSE_DEG must have {DOF} entries, "
        f"got {len(VR_HOME_POSE_DEG)}."
    )

for _index, (_value, (_lower, _upper)) in enumerate(
    zip(VR_HOME_POSE_RAD, RB10E_JOINT_LIMITS_RAD)
):
    if not _lower <= _value <= _upper:
        raise RuntimeError(
            f"VR_HOME_POSE_DEG[{_index}] = "
            f"{VR_HOME_POSE_DEG[_index]} deg lies outside the RB10E "
            "joint limits."
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
