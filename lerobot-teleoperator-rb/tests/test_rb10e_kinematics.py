import numpy as np
import pytest

from lerobot_teleoperator_rb.constants import (
    DEFAULT_READY_POSE_RAD,
)
from lerobot_teleoperator_rb.rb10e_kinematics import (
    RB10EKinematics,
)


def test_initial_fk_and_jacobian_are_valid():
    kinematics = RB10EKinematics()

    assert kinematics.q_rad.shape == (6,)
    assert kinematics.fk.shape == (4, 4)
    assert kinematics.jacobian.shape == (6, 6)

    assert np.all(np.isfinite(kinematics.fk))
    assert np.all(np.isfinite(kinematics.jacobian))

    assert np.allclose(
        kinematics.fk[3],
        [0.0, 0.0, 0.0, 1.0],
    )
    assert np.isclose(
        np.linalg.det(kinematics.fk[:3, :3]),
        1.0,
        atol=1e-6,
    )


def test_current_fk_target_keeps_same_joint_pose():
    kinematics = RB10EKinematics()

    q_before = kinematics.q_rad
    target = kinematics.fk

    q_after = kinematics.solve(target)

    assert np.allclose(q_after, q_before, atol=1e-10)
    assert kinematics.last_info.position_error_mm == pytest.approx(0.0)
    assert kinematics.last_info.orientation_error == pytest.approx(0.0)


def test_offset_target_converges():
    kinematics = RB10EKinematics()

    target = kinematics.fk
    target[0, 3] += 10.0

    q_after = kinematics.solve(
        target,
        iterations=10,
    )

    solved_pose = kinematics.fk

    assert np.all(np.isfinite(q_after))
    assert np.allclose(
        solved_pose[:3, 3],
        target[:3, 3],
        atol=1e-3,
    )
    assert kinematics.last_info.position_error_mm < 1e-3


def test_solution_respects_joint_limits():
    kinematics = RB10EKinematics()

    target = kinematics.fk
    target[1, 3] += 50.0

    q_after = kinematics.solve(
        target,
        iterations=10,
    )
    limits = kinematics.joint_limits_rad

    assert np.all(q_after >= limits[:, 0])
    assert np.all(q_after <= limits[:, 1])


def test_seed_is_clipped_to_joint_limits():
    kinematics = RB10EKinematics()

    kinematics.set_seed(
        np.full(6, 1000.0, dtype=np.float64)
    )

    limits = kinematics.joint_limits_rad

    assert np.all(kinematics.q_rad <= limits[:, 1])
    assert np.all(kinematics.q_rad >= limits[:, 0])


def test_invalid_target_pose_is_rejected():
    kinematics = RB10EKinematics()

    invalid_target = np.eye(4)
    invalid_target[3, 3] = 0.0

    with pytest.raises(ValueError):
        kinematics.solve(invalid_target)


def test_default_ready_pose_is_used():
    kinematics = RB10EKinematics()

    assert np.allclose(
        kinematics.q_rad,
        np.asarray(DEFAULT_READY_POSE_RAD),
    )
