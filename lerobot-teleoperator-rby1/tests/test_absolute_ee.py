import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lerobot_teleoperator_rby1.isaac_teleop.absolute_ee import AbsoluteEeMapper, rpy_deg_to_matrix

IDENT = np.array([0.0, 0.0, 0.0, 1.0])


def _torso(z=1.0):
    T = np.eye(4)
    T[:3, 3] = [0, 0, z]
    return T


def test_target_scales_hand_offset_from_human_shoulder_to_robot_shoulder():
    m = AbsoluteEeMapper("right", robot_reach=0.6, human_reach=1.2, shoulder_smoothing=1.0)
    assert m.target([0, 0, 0], IDENT, _torso()) is None  # no shoulder yet
    m.observe_body(np.array([0.0, -0.3, 1.5]), None, None)
    T = m.target(np.array([0.6, -0.3, 1.5]), IDENT, _torso())  # 60 cm in front of the shoulder
    np.testing.assert_allclose(T[:3, 3], [0.30, -0.22, 1.080073451539], atol=1e-6)  # half (reach ratio 0.5)


def test_reach_clamp_and_position_scale():
    m = AbsoluteEeMapper("left", robot_reach=0.6, human_reach=0.6, reach_max_ratio=0.5, position_scale=2.0, shoulder_smoothing=1.0)
    m.observe_body(np.zeros(3), None, None)
    T = m.target(np.array([1.0, 0, 0]), IDENT, _torso(0.0))
    np.testing.assert_allclose(T[:3, 3] - [0, 0.22, 0.080073452], [0.3, 0, 0], atol=1e-6)


def test_human_reach_estimated_from_body():
    m = AbsoluteEeMapper("right", robot_reach=0.6, human_reach=None, reach_smoothing=1.0)
    assert m.human_reach is None
    m.observe_body(np.zeros(3), np.array([0, 0, -0.3]), np.array([0, 0, -0.6]))
    assert m.human_reach == pytest.approx(0.6)


def test_orientation_offset_latch():
    m = AbsoluteEeMapper("right", robot_reach=0.6, human_reach=0.6, shoulder_smoothing=1.0)
    m.observe_body(np.zeros(3), None, None)
    R_ctrl = Rotation.from_euler("z", 40, degrees=True).as_matrix()
    R_meas = Rotation.from_euler("x", -90, degrees=True).as_matrix()
    m.latch_orientation_offset(R_ctrl, R_meas)
    T = m.target(np.array([0.2, 0, 0]), Rotation.from_matrix(R_ctrl).as_quat(), _torso(0.0))
    np.testing.assert_allclose(T[:3, :3], R_meas, atol=1e-9)


def test_rate_limit_and_rpy():
    m = AbsoluteEeMapper("right", robot_reach=0.6, human_reach=0.6, max_linear_vel=1.0, max_angular_vel=1.0)
    T0 = np.eye(4)
    m.reset_rate_limit(T0)
    T1 = np.eye(4)
    T1[:3, 3] = [1.0, 0, 0]
    T1[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    out = m.rate_limit(T1, dt=0.1)
    assert out[0, 3] == pytest.approx(0.1)
    assert Rotation.from_matrix(out[:3, :3]).magnitude() == pytest.approx(0.1)
    np.testing.assert_allclose(rpy_deg_to_matrix([0, 0, 90]), Rotation.from_euler("z", 90, degrees=True).as_matrix())
