"""SE3 helpers and VR-to-robot frame conversions.

The Meta Quest reports controller poses in its own world coordinate frame
(``+X`` right, ``+Y`` up, ``+Z`` back).  The RB-Y1 base frame uses a different
convention (``+X`` forward, ``+Y`` left, ``+Z`` up).  ``T_VR_TO_ROBOT`` is the
similarity transform between the two frames.

``T_HAND_OFFSET`` accounts for any mechanical offset between the URDF
``link_*_arm_6`` frame and the physical hand grasp point (identity by default).

``T_GLOBAL_TO_START`` rotates a controller delta into the EE start frame so
that motion of the controller along its forward axis maps to motion of the
hand along *its* forward axis, regardless of the hand's current orientation.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R

# ---------------------------------------------------------------------------
# Static transforms
# ---------------------------------------------------------------------------

# VR world frame → robot base frame (similarity transform).
T_VR_TO_ROBOT = np.array(
    [
        [0, -1, 0, 0],
        [0,  0, 1, 0],
        [1,  0, 0, 0],
        [0,  0, 0, 1],
    ],
    dtype=float,
)

# Mechanical offset from link_*_arm_6 to the physical grasp point.
T_HAND_OFFSET: np.ndarray = np.eye(4)
T_HAND_OFFSET_INV: np.ndarray = np.linalg.inv(T_HAND_OFFSET)

# Rotation applied to controller deltas before composing with the EE start
# pose. A +90° rotation about the Y axis aligns the controller's "push
# forward" with the hand's "reach forward" direction.
T_GLOBAL_TO_START: np.ndarray = np.eye(4)
T_GLOBAL_TO_START[:3, :3] = R.from_euler("y", 90, degrees=True).as_matrix()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pose_to_se3(position: Sequence[float], rotation_quat: Sequence[float]) -> np.ndarray:
    """Build a 4×4 SE3 matrix from ``[x, y, z]`` and ``[qx, qy, qz, qw]``."""
    T = np.eye(4)
    T[:3, :3] = R.from_quat(rotation_quat).as_matrix()
    T[:3, 3] = position
    return T


def se3_to_pose_action(T: np.ndarray, prefix: str) -> dict[str, float]:
    """Encode a 4×4 SE3 pose as LeRobot EE action keys.

    Returns ``{prefix}.x/y/z`` (metres) and ``{prefix}.wx/wy/wz`` (rotation
    vector, radians), following the LeRobot EE convention.
    """
    rotvec = R.from_matrix(T[:3, :3]).as_rotvec()
    return {
        f"{prefix}.x": float(T[0, 3]),
        f"{prefix}.y": float(T[1, 3]),
        f"{prefix}.z": float(T[2, 3]),
        f"{prefix}.wx": float(rotvec[0]),
        f"{prefix}.wy": float(rotvec[1]),
        f"{prefix}.wz": float(rotvec[2]),
    }


def vr_to_robot(position: Sequence[float], rotation_quat: Sequence[float]) -> np.ndarray:
    """Convert a VR-frame controller pose into the robot base frame."""
    T_vr = pose_to_se3(position, rotation_quat)
    return T_VR_TO_ROBOT.T @ T_vr @ T_VR_TO_ROBOT


def cartesian_delta(
    controller_start: np.ndarray,
    controller_current: np.ndarray,
    ee_start: np.ndarray,
) -> np.ndarray:
    """Map a controller delta onto a target EE pose.

    ``delta``         = ``inv(controller_start) @ controller_current``
    ``delta_global``  = ``T_GLOBAL_TO_START @ delta @ T_GLOBAL_TO_START.T``
    ``target``        = ``ee_start @ delta_global``
    """
    diff = np.linalg.inv(controller_start) @ controller_current
    diff_global = T_GLOBAL_TO_START @ diff @ T_GLOBAL_TO_START.T
    return ee_start @ diff_global
