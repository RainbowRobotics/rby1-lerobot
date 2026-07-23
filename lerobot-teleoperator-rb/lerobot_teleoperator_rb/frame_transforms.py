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
)


# PEP 8 aliases for new code.
T_CONV = T_conv
T_FOR_HEAD = T_for_head
T_FOR_RB10E = T_for_RB10E


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
    matrix[2, 3] += target_z_offset_mm

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


def controller_pose_to_rb10e_target(
    controller_pose_rb: np.ndarray,
    torso_pose_rb: np.ndarray,
    user_scale: float,
    *,
    target_x_offset_mm: float = 300.0,
    target_y_offset_mm: float = 200.0,
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
