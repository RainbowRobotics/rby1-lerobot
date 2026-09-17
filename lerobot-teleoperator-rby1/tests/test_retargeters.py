import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lerobot_teleoperator_rby1.isaac_teleop.retargeters import (
    HeadRetargeter,
    chest_pose_from_body,
    head_yaw_pitch,
    scale_clamp_delta,
    se3_to_ee_action,
    thumbsticks_to_base_vel,
    wrap_pi,
)
from lerobot_teleoperator_rby1.isaac_teleop.xr_frame import BodyJointIndex, BodyState

# base_T_anchor: OpenXR (X right, Y up, Z back) -> robot (X fwd, Y left, Z up).
BASE_T_ANCHOR = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=float)


def _head_R(yaw_deg=0.0, pitch_deg=0.0):
    """Head rotation in the robot frame for a look direction (yaw left+, pitch up+)."""
    # In the anchor frame the head looks along -Z; yaw about +Y (left = +), pitch about +X (up = +).
    R_anchor = Rotation.from_euler("YX", [yaw_deg, pitch_deg], degrees=True).as_matrix()
    return BASE_T_ANCHOR @ R_anchor


def test_identity_head_looks_forward():
    yaw, pitch = head_yaw_pitch(_head_R())
    assert abs(yaw) < 1e-9 and abs(pitch) < 1e-9


def test_yaw_left_and_pitch_up_signs():
    yaw, _ = head_yaw_pitch(_head_R(yaw_deg=30))
    assert yaw == pytest.approx(math.radians(30))
    _, pitch = head_yaw_pitch(_head_R(pitch_deg=20))
    assert pitch == pytest.approx(math.radians(20))


def test_wrap_pi():
    assert wrap_pi(math.pi + 0.1) == pytest.approx(-math.pi + 0.1)
    assert wrap_pi(-math.pi - 0.1) == pytest.approx(math.pi - 0.1)


def test_head_retargeter_latch_update_and_limits():
    h = HeadRetargeter(yaw_sign=1.0, pitch_sign=-1.0, smoothing=1.0, yaw_limit=math.radians(40))
    assert h.update(_head_R()) is None  # not latched yet -> no target
    h.latch(_head_R(yaw_deg=10), np.array([0.0, 0.85]))
    np.testing.assert_allclose(h.update(_head_R(yaw_deg=10)), [0.0, 0.85])
    q = h.update(_head_R(yaw_deg=40))  # +30 deg left of the latch
    assert q[0] == pytest.approx(math.radians(30))
    q = h.update(_head_R(yaw_deg=10, pitch_deg=-20))  # look down -> head_1 increases
    assert q[1] == pytest.approx(0.85 + math.radians(20))
    q = h.update(_head_R(yaw_deg=80))  # clipped to the 40 deg yaw limit
    assert q[0] == pytest.approx(math.radians(40))


def test_head_retargeter_smoothing_and_hold():
    h = HeadRetargeter(smoothing=0.5)
    h.latch(_head_R(), np.zeros(2))
    q1 = h.update(_head_R(yaw_deg=20))
    assert q1[0] == pytest.approx(0.5 * math.radians(20))
    np.testing.assert_allclose(h.update(None), q1)  # no head -> hold


def _body(valid=None):
    n = 24
    pos = np.zeros((n, 3))
    pos[BodyJointIndex.SPINE3] = [0.1, 0.0, 1.2]
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
    if valid is None:
        valid = np.ones(n, bool)
    return BodyState(pos, quat, valid)


def test_chest_pose_requires_valid_joints():
    req = [BodyJointIndex.PELVIS, BodyJointIndex.NECK]
    T = chest_pose_from_body(_body(), BodyJointIndex.SPINE3, req)
    np.testing.assert_allclose(T[:3, 3], [0.1, 0.0, 1.2])
    valid = np.ones(24, bool)
    valid[BodyJointIndex.PELVIS] = False
    assert chest_pose_from_body(_body(valid), BodyJointIndex.SPINE3, req) is None
    assert chest_pose_from_body(None, BodyJointIndex.SPINE3, req) is None


def test_scale_clamp_delta_rotation_and_z():
    home = np.eye(4)
    home[:3, 3] = [0, 0, 1.0]
    target = np.eye(4)
    target[:3, :3] = Rotation.from_euler("y", 60, degrees=True).as_matrix()
    target[:3, 3] = [0.3, 0.2, 0.5]  # dx, dy ignored; dz = -0.5 clamped to -0.15
    out = scale_clamp_delta(home, target, max_rot=math.radians(35), max_z=0.15)
    ang = Rotation.from_matrix(out[:3, :3]).magnitude()
    assert ang == pytest.approx(math.radians(35))
    np.testing.assert_allclose(out[:3, 3], [0.0, 0.0, 0.85])
    out_xy = scale_clamp_delta(home, target, use_xy=True, rot_scale=0.5, z_scale=0.2, max_z=1.0)
    np.testing.assert_allclose(out_xy[:3, 3], [0.3, 0.2, 0.9])
    assert Rotation.from_matrix(out_xy[:3, :3]).magnitude() == pytest.approx(math.radians(30))


def test_thumbsticks_deadzone_and_signs():
    assert thumbsticks_to_base_vel((0.1, 0.1), (0.1, 0.0), deadzone=0.15) == (0.0, 0.0, 0.0)
    vx, vy, wz = thumbsticks_to_base_vel((1.0, 1.0), (1.0, 0.0), deadzone=0.15, max_linear=0.3, max_angular=0.6)
    assert vx == pytest.approx(0.3) and vy == pytest.approx(-0.3) and wz == pytest.approx(-0.6)
    vx, vy, wz = thumbsticks_to_base_vel(None, None)
    assert (vx, vy, wz) == (0.0, 0.0, 0.0)


def test_se3_to_ee_action_keys_and_rotvec():
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec([0.1, 0.2, 0.3]).as_matrix()
    T[:3, 3] = [1, 2, 3]
    a = se3_to_ee_action(T, "right_ee")
    assert list(a) == ["right_ee.x", "right_ee.y", "right_ee.z", "right_ee.wx", "right_ee.wy", "right_ee.wz"]
    assert (a["right_ee.x"], a["right_ee.y"], a["right_ee.z"]) == (1.0, 2.0, 3.0)
    np.testing.assert_allclose([a["right_ee.wx"], a["right_ee.wy"], a["right_ee.wz"]], [0.1, 0.2, 0.3])
