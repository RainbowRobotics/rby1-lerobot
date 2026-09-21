"""Coordinate transforms for RB10E Meta Quest teleoperation.

This module preserves the transform convention used by the original,
hardware-tested RB10E VR teleoperation implementation.

Coordinate pipeline
-------------------
Quest pose
    position: metres
    quaternion: [x, y, z, w]

        ↓ pose_to_se3()

Quest SE(3)
    translation: millimetres

        ↓ T_conv.T @ pose @ T_conv

RB-oriented controller/head pose

Head pose:
    head_rb @ T_for_head
        → torso pose

Controller pose:
    inv(torso_pose) @ controller_rb @ T_for_RB10E
        → user scale
        → target Z offset
        → RB10E IK target

All matrices are 4x4 homogeneous transforms.
Translations are expressed in millimetres.

Which of these RbVr actually uses
---------------------------------
RbVr no longer commands the absolute pipeline above. It uses an anchored
delta: position 1:1, orientation scaled by RbVrConfig.orientation_scale.

    controller_translation_delta_mm()   inv(torso_a) applied to both poses
    controller_rotation_delta()         same, for the rotation blocks
    scale_rotation()                    fraction of the delta angle
    build_delta_target()                anchor EE pose + both deltas

    build_position_delta_target()       build_delta_target with no rotation

``controller_pose_to_rb10e_target``, ``compute_user_scale``, ``apply_scale``
and ``build_rb10e_target_from_quest`` are retained as the executable record
of the hardware-tested absolute convention, and because
``controller_pose_to_rb10e_target`` is what
``tests/test_frame_transforms.py`` compares the delta law against to prove
the two share the same axis mapping.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Original RB10E coordinate conversion matrices
# ---------------------------------------------------------------------------

T_conv = np.array(
    [
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


# Head-to-torso correction used by the original RB10E VR implementation.
#
# First:
#     rotation around Y by -45 degrees
#
# Then:
#     rotation around Z by +90 degrees
#     translation X = -120 mm
_HEAD_ANGLE_Y_RAD = math.radians(-45.0)
_HEAD_ANGLE_Z_RAD = math.radians(90.0)

_HEAD_ROT_Y = np.array(
    [
        [
            math.cos(_HEAD_ANGLE_Y_RAD),
            0.0,
            math.sin(_HEAD_ANGLE_Y_RAD),
            0.0,
        ],
        [0.0, 1.0, 0.0, 0.0],
        [
            -math.sin(_HEAD_ANGLE_Y_RAD),
            0.0,
            math.cos(_HEAD_ANGLE_Y_RAD),
            0.0,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

_HEAD_ROT_Z_AND_OFFSET = np.array(
    [
        [
            math.cos(_HEAD_ANGLE_Z_RAD),
            -math.sin(_HEAD_ANGLE_Z_RAD),
            0.0,
            -120.0,
        ],
        [
            math.sin(_HEAD_ANGLE_Z_RAD),
            math.cos(_HEAD_ANGLE_Z_RAD),
            0.0,
            0.0,
        ],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

T_for_head = _HEAD_ROT_Y @ _HEAD_ROT_Z_AND_OFFSET


# Controller orientation correction for the RB10E TCP.
#
# Rotation around Z by +90 degrees.
_RB10E_ANGLE_Z_RAD = math.radians(90.0)

T_for_RB10E = np.array(
    [
        [
            math.cos(_RB10E_ANGLE_Z_RAD),
            -math.sin(_RB10E_ANGLE_Z_RAD),
            0.0,
            0.0,
        ],
        [
            math.sin(_RB10E_ANGLE_Z_RAD),
            math.cos(_RB10E_ANGLE_Z_RAD),
            0.0,
            0.0,
        ],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
) @ np.array([        
    [           1,            0, 0, 0],
    [           0, np.cos(_RB10E_ANGLE_Z_RAD), -np.sin(_RB10E_ANGLE_Z_RAD), 0],
    [           0, np.sin(_RB10E_ANGLE_Z_RAD),  np.cos(_RB10E_ANGLE_Z_RAD), 0],
    [           0,            0, 0, 1]
])


# PEP 8 aliases for new code.
T_CONV = T_conv
T_FOR_HEAD = T_for_head
T_FOR_RB10E = T_for_RB10E


# ---------------------------------------------------------------------------
# Operator frame -> robot base frame
# ---------------------------------------------------------------------------

# Cell-specific correction for where the operator stands relative to the arm,
# applied to the anchored controller delta.
#
# With the identity here the torso-frame pipeline lands on, for an operator
# whose own right-handed frame is heading = +X, left = +Y, up = +Z,
#
#     operator +X (forward)  ->  base -Y
#     operator +Y (left)     ->  base +X
#     operator +Z (up)       ->  base +Z
#
# The operator asks for the mapping stated against the end effector at home
# instead, so that pushing the hand forward drives the gripper the way it
# points:
#
#     operator +X (forward)  ->  home EE -Y  ==  the gripper approach axis
#     operator +Y (left)     ->  home EE +X
#     operator +Z (up)       ->  home EE +Z
#
# constants.VR_HOME_POSE_DEG parks the EE frame on the base frame yawed
# -90 deg (EE +X = base -Y, EE +Y = base +X, EE +Z = base +Z), so those three
# lines read, in base terms,
#
#     operator +X (forward)  ->  base -X
#     operator +Y (left)     ->  base -Y
#     operator +Z (up)       ->  base +Z
#
# which is Rz(-90 deg) on top of the identity mapping above.
#
# THIS CONSTANT AND THE HOME ORIENTATION MOVE TOGETHER. It is a FIXED
# base-frame rotation: the "home EE" column only holds while the anchor
# orientation matches the home pose, and it does not follow the wrist. If
# VR_HOME_POSE_DEG is re-tuned, re-derive this from its FK -- that is what
# test_operator_to_base_maps_operator_axes_onto_the_home_ee_axes checks, by
# reading the home rotation out of the kinematics rather than restating it.
#
# Note there is NO handedness bug to compensate for: T_conv above already has
# determinant -1, so quest_pose_to_rb converts Unity's left-handed frame to a
# right-handed one. The observed operator->flange map also has determinant
# +1, i.e. it is a rotation and not a mirror.
#
# Re-derive this if the operator's station moves relative to the robot base.
# It must stay a proper rotation (determinant +1), or the 1:1 mapping and the
# handedness of the operator's motion would both break.
R_OPERATOR_TO_BASE = np.array(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _as_transform(
    transform: np.ndarray | Sequence[Sequence[float]],
    *,
    name: str,
) -> np.ndarray:
    """Convert and validate a homogeneous transform."""

    matrix = np.asarray(
        transform,
        dtype=np.float64,
    )

    if matrix.shape != (4, 4):
        raise ValueError(
            f"{name} must have shape (4, 4), got {matrix.shape}."
        )

    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            f"{name} contains non-finite values."
        )

    return matrix


def _as_vector(
    values: Sequence[float] | np.ndarray,
    *,
    size: int,
    name: str,
) -> np.ndarray:
    """Convert and validate a fixed-size vector."""

    vector = np.asarray(
        values,
        dtype=np.float64,
    )

    if vector.shape != (size,):
        raise ValueError(
            f"{name} must have shape ({size},), got {vector.shape}."
        )

    if not np.all(np.isfinite(vector)):
        raise ValueError(
            f"{name} contains non-finite values."
        )

    return vector


def _as_rotation(
    matrix: np.ndarray | Sequence[Sequence[float]],
    *,
    name: str,
) -> np.ndarray:
    """Convert and validate a 3x3 rotation matrix.

    Stricter than ``_as_transform`` on purpose. ``Rotation.from_matrix``
    silently ORTHONORMALISES a drifted proper matrix and raises a bare
    "Non-positive determinant in rotation matrix 0" for a reflected one,
    naming neither the caller nor the defect. A silent repair is exactly
    how a frame error hides for a month, so reject both here instead.
    """

    rotation = np.asarray(
        matrix,
        dtype=np.float64,
    )

    if rotation.shape != (3, 3):
        raise ValueError(
            f"{name} must have shape (3, 3), got {rotation.shape}."
        )

    if not np.all(np.isfinite(rotation)):
        raise ValueError(
            f"{name} contains non-finite values."
        )

    orthonormality_error = float(
        np.max(
            np.abs(
                rotation.T @ rotation
                - np.eye(3)
            )
        )
    )

    if orthonormality_error > 1.0e-6:
        raise ValueError(
            f"{name} is not orthonormal: max|R.T @ R - I| is "
            f"{orthonormality_error:.3e}."
        )

    if np.linalg.det(rotation) <= 0.0:
        raise ValueError(
            f"{name} is a reflection, not a rotation: det is "
            f"{float(np.linalg.det(rotation)):.6f}."
        )

    return rotation


# ---------------------------------------------------------------------------
# Original transform functions
# ---------------------------------------------------------------------------


def invert_se3(
    transform: np.ndarray | Sequence[Sequence[float]],
) -> np.ndarray:
    """Invert an SE(3) homogeneous transform.

    This uses the same rigid-transform inverse as the original implementation:

        R_inv = R.T
        t_inv = -R.T @ t
    """

    matrix = _as_transform(
        transform,
        name="transform",
    )

    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]

    inverse = np.eye(
        4,
        dtype=np.float64,
    )
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation

    return inverse


def pose_to_se3(
    position: Sequence[float] | np.ndarray,
    rotation_quat: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Convert one Meta Quest pose into an SE(3) matrix.

    Parameters
    ----------
    position:
        Quest position in metres: ``[x, y, z]``.

    rotation_quat:
        Quest quaternion in SciPy order: ``[x, y, z, w]``.

    Returns
    -------
    numpy.ndarray
        4x4 homogeneous transform with translation in millimetres.
    """

    position_m = _as_vector(
        position,
        size=3,
        name="position",
    )
    quaternion = _as_vector(
        rotation_quat,
        size=4,
        name="rotation_quat",
    )

    quaternion_norm = float(
        np.linalg.norm(quaternion)
    )

    if quaternion_norm <= 1.0e-12:
        raise ValueError(
            "rotation_quat must not be a zero quaternion."
        )

    # Rotation.from_quat() accepts a non-unit quaternion, but explicitly
    # normalising here makes replayed and live packets deterministic.
    quaternion = quaternion / quaternion_norm

    transform = np.eye(
        4,
        dtype=np.float64,
    )
    transform[:3, :3] = Rotation.from_quat(
        quaternion
    ).as_matrix()

    # Original code converts Quest metres into RB millimetres.
    transform[:3, 3] = position_m * 1000.0

    return transform


def apply_scale(
    transform: np.ndarray,
    scale: float,
    *,
    target_z_offset_mm: float = 300.0,
    copy: bool = False,
) -> np.ndarray:
    """Scale target translation and add the RB10E Z offset.

    The original implementation modifies the matrix in place. Therefore,
    ``copy=False`` is the default for parity.

    Rotation is not modified.

    Parameters
    ----------
    transform:
        Target transform whose translation is expressed in millimetres.

    scale:
        User reach scale.

    target_z_offset_mm:
        Constant added to target Z after scaling. Original value: 300 mm.

    copy:
        If True, return a modified copy. If False, modify the supplied
        ndarray in place.
    """

    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            f"scale must be finite and > 0, got {scale}."
        )

    if (
        not math.isfinite(target_z_offset_mm)
    ):
        raise ValueError(
            "target_z_offset_mm must be finite, "
            f"got {target_z_offset_mm}."
        )

    matrix = _as_transform(
        transform,
        name="transform",
    )

    if copy:
        matrix = matrix.copy()

    matrix[0, 3] *= scale
    matrix[1, 3] *= scale
    matrix[2, 3] *= scale

    # Original RB10E target height correction.
    # matrix[2, 3] += target_z_offset_mm

    return matrix


# ---------------------------------------------------------------------------
# High-level RB10E transform pipeline
# ---------------------------------------------------------------------------


def quest_pose_to_rb(
    position: Sequence[float] | np.ndarray,
    rotation_quat: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Convert a raw Quest pose into the original RB-oriented frame."""

    quest_transform = pose_to_se3(
        position,
        rotation_quat,
    )

    return T_conv.T @ quest_transform @ T_conv


def head_pose_to_torso(
    head_pose_rb: np.ndarray,
) -> np.ndarray:
    """Compute the torso reference pose from the converted head pose."""

    head_pose = _as_transform(
        head_pose_rb,
        name="head_pose_rb",
    )

    return head_pose @ T_for_head


def controller_pose_in_torso(
    controller_pose_rb: np.ndarray,
    torso_pose_rb: np.ndarray,
) -> np.ndarray:
    """Express one converted controller pose in the torso frame.

    This function does not apply the RB10E TCP orientation correction,
    user scaling, or target Z offset.
    """

    controller_pose = _as_transform(
        controller_pose_rb,
        name="controller_pose_rb",
    )
    torso_pose = _as_transform(
        torso_pose_rb,
        name="torso_pose_rb",
    )

    return invert_se3(torso_pose) @ controller_pose


def controller_translation_delta_mm(
    controller_pose_rb: np.ndarray,
    anchor_controller_pose_rb: np.ndarray,
    reference_torso_pose_rb: np.ndarray,
) -> np.ndarray:
    """Controller translation delta in the robot base frame.

    Both controller poses are expressed in the SAME torso frame, so the
    torso translation cancels exactly and only its rotation contributes.
    The result is then rotated from the operator's frame into the robot
    base frame::

        delta = R_OPERATOR_TO_BASE @ R_torso.T @ (p_current - p_anchor)

    ``reference_torso_pose_rb`` is meant to be captured once, when the
    operator engages the clutch, and held constant for the whole stroke.
    Recomputing it every tick would make head rotation move the robot even
    with a perfectly still hand.

    Both factors are rotations, so the length of the delta is preserved and
    the controller-to-end-effector mapping stays 1:1.

    All translations are in millimetres. Returns a ``(3,)`` vector.
    """

    current = controller_pose_in_torso(
        controller_pose_rb,
        reference_torso_pose_rb,
    )
    anchor = controller_pose_in_torso(
        anchor_controller_pose_rb,
        reference_torso_pose_rb,
    )

    delta_operator = current[:3, 3] - anchor[:3, 3]

    return np.asarray(
        R_OPERATOR_TO_BASE @ delta_operator,
        dtype=np.float64,
    )


def controller_rotation_delta(
    controller_pose_rb: np.ndarray,
    anchor_controller_pose_rb: np.ndarray,
    reference_torso_pose_rb: np.ndarray,
) -> np.ndarray:
    """Controller rotation delta in the robot base frame.

    The rotation counterpart of ``controller_translation_delta_mm``, built
    from the same ``controller_pose_in_torso`` calls so the two deltas can
    never drift onto different frames::

        delta = R_OPERATOR_TO_BASE @ (R_current @ R_anchor.T) @ R_OPERATOR_TO_BASE.T

    Returns a ``(3, 3)`` proper rotation.

    This is a WORLD (extrinsic) delta, so it is meant to be PRE-multiplied
    onto the anchor end-effector rotation. That matches the IK: the angular
    Jacobian rows hold base-frame joint axes and ``update_error_vector``
    uses ``R_target @ R_current.T``, so RB10E's orientation error is a
    base-frame left rotation too.

    Three properties worth not re-deriving:

    * The torso TRANSLATION cancels identically -- only the rotation blocks
      are read -- so the operator may lean or walk mid-stroke, exactly as
      ``controller_translation_delta_mm`` promises for translation.
    * The torso ROTATION does not cancel. It appears as the similarity
      ``R_torso.T @ ... @ R_torso``, which is what puts this delta in the
      same frozen operator frame the translation delta lives in.
      ``reference_torso_pose_rb`` must therefore be the pose captured when
      the operator engaged the clutch, held for the whole stroke.
    * A constant hand-to-flange offset CANCELS. Replacing both controller
      poses by ``T_ctrl @ T_const`` leaves the delta unchanged, because
      ``R_now @ T @ T.T @ R_anchor.T == R_now @ R_anchor.T``. This is why
      the absolute path's ``T_FOR_RB10E`` correction is not applied here
      and must not be added.

    Both factors are rotations, so the delta ANGLE is preserved 1:1. Any
    scaling is the caller's job -- see ``scale_rotation``.

    ``R_OPERATOR_TO_BASE`` is applied as a similarity ``R X R.T`` rather
    than a left multiply, because a rotation is a tensor, not a vector. The
    two spellings agree only when the constant is the identity, which it is
    not: it is Rz(-90 deg) for the current home pose.
    """

    current = controller_pose_in_torso(
        controller_pose_rb,
        reference_torso_pose_rb,
    )
    anchor = controller_pose_in_torso(
        anchor_controller_pose_rb,
        reference_torso_pose_rb,
    )

    rotation_current = _as_rotation(
        current[:3, :3],
        name="controller_pose_rb rotation",
    )
    rotation_anchor = _as_rotation(
        anchor[:3, :3],
        name="anchor_controller_pose_rb rotation",
    )

    delta_operator = rotation_current @ rotation_anchor.T

    return np.asarray(
        R_OPERATOR_TO_BASE
        @ delta_operator
        @ R_OPERATOR_TO_BASE.T,
        dtype=np.float64,
    )


def scale_rotation(
    rotation: np.ndarray,
    gain: float,
) -> np.ndarray:
    """Return ``rotation`` scaled to ``gain`` of its angle, same axis.

    Equivalent to a SLERP from the identity towards ``rotation``: the
    rotation vector is scaled, so the axis is exactly preserved and the
    angle is exactly multiplied.

    ``gain`` is not required to be positive. ``0.0`` is the meaningful
    "off" value and returns EXACTLY ``eye(3)`` -- bit-for-bit, not merely
    within tolerance -- so a caller can reproduce a no-rotation control law
    through this function without any numerical drift.

    Note the discontinuity at theta = pi. ``as_rotvec`` maps 179.999 deg to
    ``+179.999 deg * axis`` and 180.001 deg to ``-179.999 deg * axis``, so a
    scaled command flips by ``2 * gain * pi`` across that point. Nothing
    here can fix it -- the shortest-path rotation between two frames a half
    turn apart is genuinely ambiguous. ``RbVr._limit_joint_step`` is what
    keeps the flip from being a lurch.
    """

    if not math.isfinite(gain):
        raise ValueError(
            f"gain must be finite, got {gain}."
        )

    rotation_vector = Rotation.from_matrix(
        _as_rotation(
            rotation,
            name="rotation",
        )
    ).as_rotvec()

    return Rotation.from_rotvec(
        gain * rotation_vector
    ).as_matrix()


def build_delta_target(
    ee_anchor_pose: np.ndarray,
    translation_delta_mm: Sequence[float] | np.ndarray,
    *,
    rotation_delta: np.ndarray | None = None,
) -> np.ndarray:
    """Build an RB10E IK target that displaces the anchor EE pose.

    Translation is the anchor translation plus ``translation_delta_mm``,
    in millimetres, in the robot base frame. There is deliberately no
    translation scale parameter: the controller-to-end-effector POSITION
    mapping is 1:1 because no knob exists to make it anything else.

    ``rotation_delta`` is PRE-multiplied onto the anchor rotation::

        target_R = rotation_delta @ anchor_R

    ``None`` means "hold the anchor orientation" and is bit-for-bit the
    pre-orientation control law.

    Pre-multiplication is not interchangeable with post-multiplication.
    ``controller_rotation_delta`` returns a delta already expressed in the
    BASE frame; post-multiplying would apply it in the tool frame instead,
    and the hand-to-flange rotation map would silently stop matching the
    hand-to-flange translation map.

    ``rotation_delta`` is keyword-only so a stray third positional
    argument cannot be mistaken for a second translation.
    """

    anchor = _as_transform(
        ee_anchor_pose,
        name="ee_anchor_pose",
    )
    delta = _as_vector(
        translation_delta_mm,
        size=3,
        name="translation_delta_mm",
    )

    target = np.eye(4, dtype=np.float64)
    target[:3, 3] = anchor[:3, 3] + delta

    if rotation_delta is None:
        target[:3, :3] = anchor[:3, :3]
    else:
        target[:3, :3] = (
            _as_rotation(
                rotation_delta,
                name="rotation_delta",
            )
            @ anchor[:3, :3]
        )

    return target


def build_position_delta_target(
    ee_anchor_pose: np.ndarray,
    translation_delta_mm: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Build an RB10E IK target that translates the anchor EE pose.

    Orientation is copied from ``ee_anchor_pose`` and held constant.
    Translation is the anchor translation plus ``translation_delta_mm``.
    Both are in millimetres, in the robot base frame.

    Equivalent to ``build_delta_target(..., rotation_delta=None)``. Kept as
    the explicit spelling of the frozen-orientation path, and as what the
    clutch's zero-delta IK probe asks for: that probe exists to prove the
    solver round-trips FK at the anchor, so it should name the narrowest
    target it can rather than route through the orientation feature.
    """

    return build_delta_target(
        ee_anchor_pose,
        translation_delta_mm,
    )


def controller_pose_to_rb10e_target(
    controller_pose_rb: np.ndarray,
    torso_pose_rb: np.ndarray,
    user_scale: float,
    *,
    target_x_offset_mm: float = 300.0,
    target_y_offset_mm: float = 500.0,
    target_z_offset_mm: float = 600.0,
) -> np.ndarray:
    """Convert an RB-frame Quest controller pose to an RB10E IK target."""

    target_pose = controller_pose_in_torso(
        controller_pose_rb,
        torso_pose_rb,
    )

    target_pose = (
        target_pose
        @ T_FOR_RB10E
    )

    target_pose = apply_scale(
        target_pose,
        user_scale,
    )

    target_pose[0, 3] += float(
        target_x_offset_mm
    )
    target_pose[1, 3] += float(
        target_y_offset_mm
    )
    target_pose[2, 3] += float(
        target_z_offset_mm
    )

    return target_pose


def compute_user_scale(
    controller_pose_rb: np.ndarray,
    torso_pose_rb: np.ndarray,
    *,
    reference_reach_mm: float = 1300.0,
    minimum_reach_mm: float = 1.0,
) -> float:
    """Compute the original user reach scale.

    Original calculation:

        user_scale = 1300 / ||controller_position_in_torso||

    This helper performs the torso inverse before calculating the norm,
    ensuring the value is computed from the current packet rather than a
    stale inverse transform.
    """

    if (
        not math.isfinite(reference_reach_mm)
        or reference_reach_mm <= 0.0
    ):
        raise ValueError(
            "reference_reach_mm must be finite and > 0, "
            f"got {reference_reach_mm}."
        )

    if (
        not math.isfinite(minimum_reach_mm)
        or minimum_reach_mm <= 0.0
    ):
        raise ValueError(
            "minimum_reach_mm must be finite and > 0, "
            f"got {minimum_reach_mm}."
        )

    relative_pose = controller_pose_in_torso(
        controller_pose_rb,
        torso_pose_rb,
    )

    reach_mm = float(
        np.linalg.norm(relative_pose[:3, 3])
    )

    if reach_mm < minimum_reach_mm:
        raise ValueError(
            "Controller reach is too small for user-scale calibration: "
            f"{reach_mm:.3f} mm."
        )

    return reference_reach_mm / reach_mm


def build_rb10e_target_from_quest(
    controller_position: Sequence[float] | np.ndarray,
    controller_rotation_quat: Sequence[float] | np.ndarray,
    head_position: Sequence[float] | np.ndarray,
    head_rotation_quat: Sequence[float] | np.ndarray,
    user_scale: float,
    *,
    target_z_offset_mm: float = 300.0,
) -> np.ndarray:
    """Run the complete Quest-to-RB10E target transform pipeline."""

    controller_pose_rb = quest_pose_to_rb(
        controller_position,
        controller_rotation_quat,
    )
    head_pose_rb = quest_pose_to_rb(
        head_position,
        head_rotation_quat,
    )
    torso_pose_rb = head_pose_to_torso(
        head_pose_rb
    )

    return controller_pose_to_rb10e_target(
        controller_pose_rb,
        torso_pose_rb,
        user_scale,
        target_z_offset_mm=target_z_offset_mm,
    )
