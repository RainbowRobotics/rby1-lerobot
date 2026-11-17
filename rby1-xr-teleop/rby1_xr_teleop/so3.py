"""Small SO(3) helpers in plain NumPy / ``math`` for the per-tick code paths.

``scipy.spatial.transform.Rotation`` is convenient but creating one object
costs tens of microseconds, and the teleoperator needs a few hundred tiny
rotations per tick (the arm posture solver alone about thirty per arm). These
functions return the same values as the SciPy calls they replace — same
quaternion extraction, same small-angle series — so results agree to rounding
error; ``tests/test_isaac_so3.py`` checks that against SciPy.

Conventions: matrices are 3x3 ``float64``, quaternions are ``xyzw`` (SciPy
order), rotation vectors are axis * angle in radians.
"""

from __future__ import annotations

import math

import numpy as np

_SMALL_ANGLE = 1e-3


def rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_axis_angle(axis, angle: float) -> np.ndarray:
    """Rotation by ``angle`` about ``axis``; equals ``Rotation.from_rotvec(axis * angle)``.

    ``axis`` need not be a unit vector: like the rotation vector ``axis * angle``,
    its length scales the angle.
    """
    x, y, z = float(axis[0]), float(axis[1]), float(axis[2])
    n = math.sqrt(x * x + y * y + z * z)
    if n < 1e-300:
        return np.eye(3)
    x, y, z = x / n, y / n, z / n
    theta = angle * n
    c, s = math.cos(theta), math.sin(theta)
    k = 1.0 - c
    return np.array(
        [
            [c + x * x * k, x * y * k - z * s, x * z * k + y * s],
            [y * x * k + z * s, c + y * y * k, y * z * k - x * s],
            [z * x * k - y * s, z * y * k + x * s, c + z * z * k],
        ]
    )


def matrix_from_rotvec(rotvec) -> np.ndarray:
    """``Rotation.from_rotvec(rotvec).as_matrix()``."""
    x, y, z = float(rotvec[0]), float(rotvec[1]), float(rotvec[2])
    angle = math.sqrt(x * x + y * y + z * z)
    if angle <= _SMALL_ANGLE:
        a2 = angle * angle
        scale = 0.5 - a2 / 48.0 + a2 * a2 / 3840.0
    else:
        scale = math.sin(angle / 2.0) / angle
    return _matrix_from_unit_quat(scale * x, scale * y, scale * z, math.cos(angle / 2.0))


def matrix_from_quat(quat_xyzw) -> np.ndarray:
    """``Rotation.from_quat(quat_xyzw).as_matrix()`` (the quaternion is normalised)."""
    x, y, z, w = float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2]), float(quat_xyzw[3])
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0.0:
        raise ValueError("Found zero norm quaternion.")
    return _matrix_from_unit_quat(x / n, y / n, z / n, w / n)


def _matrix_from_unit_quat(x: float, y: float, z: float, w: float) -> np.ndarray:
    x2, y2, z2, w2 = x * x, y * y, z * z, w * w
    xy, zw, xz, yw, yz, xw = x * y, z * w, x * z, y * w, y * z, x * w
    return np.array(
        [
            [x2 - y2 - z2 + w2, 2.0 * (xy - zw), 2.0 * (xz + yw)],
            [2.0 * (xy + zw), -x2 + y2 - z2 + w2, 2.0 * (yz - xw)],
            [2.0 * (xz - yw), 2.0 * (yz + xw), -x2 - y2 + z2 + w2],
        ]
    )


def quat_from_matrix(R) -> np.ndarray:  # noqa: N803
    """``Rotation.from_matrix(R).as_quat()`` (xyzw, same sign as SciPy returns)."""
    x, y, z, w = _quat_from_matrix(R)
    return np.array([x, y, z, w])


def _quat_from_matrix(R) -> tuple[float, float, float, float]:  # noqa: N803
    m00, m01, m02 = float(R[0][0]), float(R[0][1]), float(R[0][2])
    m10, m11, m12 = float(R[1][0]), float(R[1][1]), float(R[1][2])
    m20, m21, m22 = float(R[2][0]), float(R[2][1]), float(R[2][2])
    trace = m00 + m11 + m22
    # Largest of (m00, m11, m22, trace) picks the numerically safest branch,
    # first maximum on ties — as numpy.argmax does in SciPy.
    best, choice = m00, 0
    if m11 > best:
        best, choice = m11, 1
    if m22 > best:
        best, choice = m22, 2
    if trace > best:
        choice = 3
    if choice == 0:
        x, y, z, w = 1.0 - trace + 2.0 * m00, m10 + m01, m20 + m02, m21 - m12
    elif choice == 1:
        x, y, z, w = m01 + m10, 1.0 - trace + 2.0 * m11, m21 + m12, m02 - m20
    elif choice == 2:
        x, y, z, w = m02 + m20, m12 + m21, 1.0 - trace + 2.0 * m22, m10 - m01
    else:
        x, y, z, w = m21 - m12, m02 - m20, m10 - m01, 1.0 + trace
    n = math.sqrt(x * x + y * y + z * z + w * w)
    return x / n, y / n, z / n, w / n


def renormalize(R) -> np.ndarray:  # noqa: N803
    """Re-orthonormalise a rotation matrix through its quaternion.

    Use it on a rotation that is carried from tick to tick as a product of
    matrices, so rounding errors do not accumulate. For a rotation matrix it
    equals ``Rotation.from_matrix(R).as_matrix()`` up to rounding.
    """
    return _matrix_from_unit_quat(*_quat_from_matrix(R))


def matrix_from_rpy_deg(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """``Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=True).as_matrix()`` (fixed axes)."""
    return rot_z(math.radians(yaw)) @ rot_y(math.radians(pitch)) @ rot_x(math.radians(roll))


def rotvec_from_matrix(R) -> np.ndarray:  # noqa: N803
    """``Rotation.from_matrix(R).as_rotvec()`` (angle in [0, pi])."""
    x, y, z, w = _quat_from_matrix(R)
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    angle = 2.0 * math.atan2(math.sqrt(x * x + y * y + z * z), w)
    if angle <= _SMALL_ANGLE:
        a2 = angle * angle
        scale = 2.0 + a2 / 12.0 + 7.0 * a2 * a2 / 2880.0
    else:
        scale = angle / math.sin(angle / 2.0)
    return np.array([scale * x, scale * y, scale * z])
