"""Meta Quest to RB10E coordinate transformations.

This module preserves the coordinate mapping used by the existing RB10E VR
teleoperation program while separating it from UDP communication, inverse
kinematics, and robot control.

Coordinate flow
---------------
1. Meta Quest reports position in metres and orientation as an XYZW quaternion.
2. ``pose_to_se3()`` converts the pose into a 4x4 transform with millimetres.
3. ``quest_pose_to_rb_frame()`` changes the Quest basis into the RB basis.
4. The controller pose is expressed relative to the transformed head/torso pose.
5. Translation is scaled to the user's reach and shifted by the configured
   RB10E workspace offset.
6. The result is a Cartesian target for ``RB10EKinematics.solve()``.

Unit conventions
----------------
* Raw Meta Quest position: metres.
* Internal and output translation: millimetres.
* Quaternion order: [x, y, z, w].
* Rotation: 3x3 rotation matrix.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


FloatArray = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Legacy RB10E transform constants
# ---------------------------------------------------------------------------

# Quest basis conversion retained from RB10E_utils.py.
#
# This matrix changes handedness, so its 3x3 determinant is -1. It is used
# through conjugation:
#
#     T_rb = T_CONV.T @ T_quest @ T_CONV
#
# The resulting pose rotation remains a proper rotation with determinant +1.
T_CONV: FloatArray = np.array(
    [
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# Headset pose to the torso reference used by the existing implementation.
_head_pitch_rad = np.deg2rad(-45.0)
_head_yaw_rad = np.deg2rad(90.0)

T_FOR_HEAD: FloatArray = (
    np.array(
        [
            [
                np.cos(_head_pitch_rad),
                0.0,
                np.sin(_head_pitch_rad),
                0.0,
            ],
            [0.0, 1.0, 0.0, 0.0],
            [
                -np.sin(_head_pitch_rad),
                0.0,
                np.cos(_head_pitch_rad),
                0.0,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    @ np.array(
        [
            [
                np.cos(_head_yaw_rad),
                -np.sin(_head_yaw_rad),
                0.0,
                -120.0,
            ],
            [
                np.sin(_head_yaw_rad),
                np.cos(_head_yaw_rad),
                0.0,
                0.0,
            ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
)

# Controller orientation to the RB10E TCP orientation.
_tcp_yaw_rad = np.deg2rad(90.0)

T_FOR_RB10E: FloatArray = np.array(
    [
        [
            np.cos(_tcp_yaw_rad),
            -np.sin(_tcp_yaw_rad),
            0.0,
            0.0,
        ],
        [
            np.sin(_tcp_yaw_rad),
            np.cos(_tcp_yaw_rad),
            0.0,
            0.0,
        ],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# Prevent accidental mutation of module-level calibration matrices.
T_CONV.setflags(write=False)
T_FOR_HEAD.setflags(write=False)
T_FOR_RB10E.setflags(write=False)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _as_vector(
    value: Iterable[float],
    *,
    size: int,
    name: str,
) -> FloatArray:
    array = np.asarray(tuple(value), dtype=np.float64)

    if array.shape != (size,):
        raise ValueError(
            f"{name} must have shape ({size},), got {array.shape}."
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf: {array}")

    return array.copy()


def validate_se3(
    pose: FloatArray,
    *,
    name: str = "pose",
    atol: float = 1e-5,
) -> FloatArray:
    """Validate and return a copy of a proper homogeneous transform."""

    transform = np.asarray(pose, dtype=np.float64)

    if transform.shape != (4, 4):
        raise ValueError(
            f"{name} must have shape (4, 4), got {transform.shape}."
        )

    if not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} contains NaN or Inf.")

    if not np.allclose(
        transform[3],
        np.array([0.0, 0.0, 0.0, 1.0]),
        atol=atol,
    ):
        raise ValueError(
            f"{name} must end with homogeneous row [0, 0, 0, 1]."
        )

    rotation = transform[:3, :3]

    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        atol=atol,
    ):
        raise ValueError(
            f"{name} rotation matrix is not orthonormal."
        )

    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, atol=atol):
        raise ValueError(
            f"{name} rotation determinant must be +1, "
            f"got {determinant:.8f}."
        )

    return transform.copy()


# ---------------------------------------------------------------------------
# Basic SE(3) operations
# ---------------------------------------------------------------------------

def pose_to_se3(
    position_m: Iterable[float],
    quaternion_xyzw: Iterable[float],
) -> FloatArray:
    """Convert a Meta Quest pose into a 4x4 transform.

    Parameters
    ----------
    position_m:
        Quest position [x, y, z] in metres.
    quaternion_xyzw:
        Quest quaternion [x, y, z, w].

    Returns
    -------
    np.ndarray
        4x4 homogeneous transform with translation in millimetres.
    """

    position = _as_vector(
        position_m,
        size=3,
        name="position_m",
    )
    quaternion = _as_vector(
        quaternion_xyzw,
        size=4,
        name="quaternion_xyzw",
    )

    quaternion_norm = float(np.linalg.norm(quaternion))
    if quaternion_norm < 1e-8:
        raise ValueError("Quaternion norm is too close to zero.")

    # Quest packets should already contain a unit quaternion, but normalising
    # prevents small transmission/serialization errors from affecting the
    # rotation matrix.
    quaternion /= quaternion_norm

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    transform[:3, 3] = position * 1000.0

    return transform


def invert_se3(pose: FloatArray) -> FloatArray:
    """Return the rigid-body inverse of a homogeneous transform."""

    transform = validate_se3(pose)

    rotation = transform[:3, :3]
    translation = transform[:3, 3]

    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation

    return inverse


def relative_pose(
    reference_pose: FloatArray,
    target_pose: FloatArray,
) -> FloatArray:
    """Express ``target_pose`` in the coordinate frame of ``reference_pose``."""

    reference = validate_se3(
        reference_pose,
        name="reference_pose",
    )
    target = validate_se3(
        target_pose,
        name="target_pose",
    )

    relative = invert_se3(reference) @ target
    return validate_se3(relative, name="relative_pose")


def scale_translation(
    pose: FloatArray,
    *,
    scale: float,
    z_offset_mm: float = 0.0,
) -> FloatArray:
    """Scale only the translation component of a pose.

    Unlike the original ``apply_scale()``, this function does not mutate the
    input matrix.
    """

    transform = validate_se3(pose)

    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            f"scale must be finite and positive, got {scale}."
        )

    if not np.isfinite(z_offset_mm):
        raise ValueError(
            f"z_offset_mm must be finite, got {z_offset_mm}."
        )

    output = transform.copy()
    output[:3, 3] *= float(scale)
    output[2, 3] += float(z_offset_mm)

    return output


# ---------------------------------------------------------------------------
# Quest to RB coordinate conversion
# ---------------------------------------------------------------------------

def quest_pose_to_rb_frame(
    position_m: Iterable[float],
    quaternion_xyzw: Iterable[float],
) -> FloatArray:
    """Convert one raw Quest pose into the legacy RB coordinate frame."""

    quest_pose = pose_to_se3(
        position_m,
        quaternion_xyzw,
    )

    # T_CONV is orthogonal but includes a handedness conversion. Conjugating
    # the complete transform preserves a proper output rotation.
    rb_pose = T_CONV.T @ quest_pose @ T_CONV

    return validate_se3(rb_pose, name="rb_pose")


def head_pose_to_torso_pose(
    head_pose_rb: FloatArray,
) -> FloatArray:
    """Convert a transformed Quest head pose into the torso reference pose."""

    head = validate_se3(
        head_pose_rb,
        name="head_pose_rb",
    )

    torso = head @ T_FOR_HEAD
    return validate_se3(torso, name="torso_pose_rb")


def compute_user_scale(
    controller_pose_rb: FloatArray,
    torso_pose_rb: FloatArray,
    *,
    reference_reach_mm: float = 1300.0,
    minimum_reach_mm: float = 10.0,
) -> float:
    """Compute the legacy user reach scale.

    The original implementation used:

        user_scale = 1300 mm / measured controller reach

    where controller reach is measured relative to the torso reference.
    """

    if not np.isfinite(reference_reach_mm) or reference_reach_mm <= 0.0:
        raise ValueError(
            "reference_reach_mm must be finite and positive."
        )

    if not np.isfinite(minimum_reach_mm) or minimum_reach_mm <= 0.0:
        raise ValueError(
            "minimum_reach_mm must be finite and positive."
        )

    controller_relative = relative_pose(
        torso_pose_rb,
        controller_pose_rb,
    )
    measured_reach_mm = float(
        np.linalg.norm(controller_relative[:3, 3])
    )

    if measured_reach_mm < minimum_reach_mm:
        raise ValueError(
            "Controller is too close to the torso reference to compute "
            f"a stable user scale: reach={measured_reach_mm:.3f} mm."
        )

    return float(reference_reach_mm / measured_reach_mm)


def controller_pose_to_rb10e_target(
    controller_pose_rb: FloatArray,
    torso_pose_rb: FloatArray,
    *,
    user_scale: float,
    position_scale: float = 1.0,
    x_offset_mm: float = 300.0,
    y_offset_mm: float = 200.0,
    z_offset_mm: float = 600.0,
) -> FloatArray:
    """Convert an RB-frame controller pose into an RB10E IK target.

    This preserves the existing transformation:

        inverse(torso) @ controller @ T_FOR_RB10E

    followed by translation scaling and the RB10E workspace Z offset.
    """

    controller = validate_se3(
        controller_pose_rb,
        name="controller_pose_rb",
    )
    torso = validate_se3(
        torso_pose_rb,
        name="torso_pose_rb",
    )

    if not np.isfinite(position_scale) or position_scale <= 0.0:
        raise ValueError(
            "position_scale must be finite and positive."
        )

    total_scale = float(user_scale) * float(position_scale)

    relative_controller = (
        invert_se3(torso)
        @ controller
        @ T_FOR_RB10E
    )
    relative_controller = validate_se3(
        relative_controller,
        name="relative_controller_pose",
    )

    return scale_translation(
        relative_controller,
        scale=total_scale,
        z_offset_mm=z_offset_mm,
    )


def quest_sample_to_rb10e_target(
    *,
    head_position_m: Iterable[float],
    head_quaternion_xyzw: Iterable[float],
    controller_position_m: Iterable[float],
    controller_quaternion_xyzw: Iterable[float],
    user_scale: float,
    position_scale: float = 1.0,
    x_offset_mm: float = 300.0,
    y_offset_mm: float = 200.0,
    z_offset_mm: float = 600.0,
) -> FloatArray:
    """Convert one Quest head/controller sample directly to an RB10E target."""

    head_pose_rb = quest_pose_to_rb_frame(
        head_position_m,
        head_quaternion_xyzw,
    )
    controller_pose_rb = quest_pose_to_rb_frame(
        controller_position_m,
        controller_quaternion_xyzw,
    )
    torso_pose_rb = head_pose_to_torso_pose(head_pose_rb)

    return controller_pose_to_rb10e_target(
        controller_pose_rb,
        torso_pose_rb,
        user_scale=user_scale,
        position_scale=position_scale,
        x_offset_mm=x_offset_mm,
        y_offset_mm=y_offset_mm,
        z_offset_mm=z_offset_mm,
    )
