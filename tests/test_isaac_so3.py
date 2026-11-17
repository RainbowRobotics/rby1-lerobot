"""The NumPy SO(3) helpers return what the SciPy calls they replace return."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lerobot_teleoperator_rby1.isaac_teleop import so3


def _rotvecs() -> list[np.ndarray]:
    rng = np.random.default_rng(7)
    out = [np.zeros(3), np.array([1e-9, 0.0, 0.0]), np.array([0.0, 5e-4, 0.0]), np.array([0.0, 0.0, 1.0001e-3])]
    for angle in (1e-6, 1e-3, 0.3, 1.0, math.pi / 2, 2.5, math.pi - 1e-3, math.pi - 1e-7):
        for _ in range(12):
            axis = rng.normal(size=3)
            out.append(axis / np.linalg.norm(axis) * angle)
    return out


def test_matrix_from_rotvec_and_axis_angle_match_scipy() -> None:
    for v in _rotvecs():
        expected = Rotation.from_rotvec(v).as_matrix()
        np.testing.assert_allclose(so3.matrix_from_rotvec(v), expected, atol=1e-13)
        angle = float(np.linalg.norm(v))
        if angle > 0.0:
            np.testing.assert_allclose(so3.rot_axis_angle(v / angle, angle), expected, atol=1e-13)
            # A non-unit axis scales the angle, as the rotation vector axis * angle does.
            np.testing.assert_allclose(so3.rot_axis_angle(2.0 * v / angle, angle / 2.0), expected, atol=1e-13)
    np.testing.assert_array_equal(so3.rot_axis_angle([0.0, 0.0, 0.0], 1.0), np.eye(3))


def test_principal_axis_rotations() -> None:
    for angle in (-2.0, -0.3, 0.0, 1e-5, 0.7, math.pi):
        np.testing.assert_allclose(so3.rot_x(angle), Rotation.from_euler("x", angle).as_matrix(), atol=1e-15)
        np.testing.assert_allclose(so3.rot_y(angle), Rotation.from_euler("y", angle).as_matrix(), atol=1e-15)
        np.testing.assert_allclose(so3.rot_z(angle), Rotation.from_euler("z", angle).as_matrix(), atol=1e-15)


def test_rotvec_and_quat_from_matrix_match_scipy() -> None:
    for v in _rotvecs():
        R = Rotation.from_rotvec(v).as_matrix()
        np.testing.assert_allclose(so3.rotvec_from_matrix(R), Rotation.from_matrix(R).as_rotvec(), atol=1e-9)
        np.testing.assert_allclose(so3.quat_from_matrix(R), Rotation.from_matrix(R).as_quat(), atol=1e-12)
        # Round trip.
        np.testing.assert_allclose(so3.matrix_from_rotvec(so3.rotvec_from_matrix(R)), R, atol=1e-9)


def test_half_turns_agree_with_scipy_as_rotations() -> None:
    # At exactly pi the rotation vector's sign is arbitrary: compare the rotations.
    for axis in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.6, 0.0, 0.8], [1.0, 1.0, 1.0]):
        axis = np.asarray(axis) / np.linalg.norm(axis)
        R = Rotation.from_rotvec(axis * math.pi).as_matrix()
        v = so3.rotvec_from_matrix(R)
        assert np.linalg.norm(v) == pytest.approx(math.pi, abs=1e-9)
        np.testing.assert_allclose(so3.matrix_from_rotvec(v), R, atol=1e-9)


def test_matrix_from_quat_matches_scipy_and_normalises() -> None:
    rng = np.random.default_rng(3)
    for _ in range(50):
        q = rng.normal(size=4) * rng.uniform(0.1, 5.0)
        np.testing.assert_allclose(so3.matrix_from_quat(q), Rotation.from_quat(q).as_matrix(), atol=1e-13)
    np.testing.assert_array_equal(so3.matrix_from_quat([0.0, 0.0, 0.0, 1.0]), np.eye(3))
    with pytest.raises(ValueError):
        so3.matrix_from_quat([0.0, 0.0, 0.0, 0.0])


def test_renormalize_and_fixed_axis_euler_match_scipy() -> None:
    rng = np.random.default_rng(11)
    for _ in range(30):
        R = Rotation.from_rotvec(rng.normal(size=3)).as_matrix()
        np.testing.assert_allclose(so3.renormalize(R), Rotation.from_matrix(R).as_matrix(), atol=1e-13)
        # A matrix that drifted off SO(3) comes back as a proper rotation close to it.
        fixed = so3.renormalize(R + rng.normal(scale=1e-6, size=(3, 3)))
        np.testing.assert_allclose(fixed @ fixed.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(fixed) == pytest.approx(1.0, abs=1e-12)
        np.testing.assert_allclose(fixed, R, atol=1e-5)
        rpy = rng.uniform(-170.0, 170.0, size=3)
        expected = Rotation.from_euler("xyz", rpy, degrees=True).as_matrix()
        np.testing.assert_allclose(so3.matrix_from_rpy_deg(*rpy), expected, atol=1e-13)
