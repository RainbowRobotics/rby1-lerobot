import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lerobot_teleoperator_rby1.isaac_teleop.retargeters import (
    HeadRetargeter,
    chest_pose_from_body,
    head_yaw_pitch,
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
    h.latch_offset(_head_R(yaw_deg=10), np.array([0.0, 0.85]))  # looking 10 deg left = joints [0, 0.85]
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


def test_head_retargeter_latch_offset():
    from lerobot_teleoperator_rby1.isaac_teleop.retargeters import HeadRetargeter
    from scipy.spatial.transform import Rotation

    R_fwd = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)  # headset -Z -> robot +X

    def R(yaw_deg, pitch_deg):
        return R_fwd @ Rotation.from_euler("y", yaw_deg, degrees=True).as_matrix() @ Rotation.from_euler("x", pitch_deg, degrees=True).as_matrix()

    target = np.array([0.1, 0.8])
    h = HeadRetargeter(smoothing=1.0, yaw_offset=0.5, pitch_offset=0.5)
    h.latch_offset(R(30, -20), target)
    np.testing.assert_allclose(h.update(R(30, -20)), target, atol=1e-9)
    q = h.update(R(40, -20))  # +10 deg yaw -> head_0 + 10 deg
    np.testing.assert_allclose(q, [target[0] + math.radians(10), target[1]], atol=1e-9)
    q = h.update(R(40, -10))  # +10 deg pitch (up), pitch_sign -1 -> head_1 - 10 deg
    np.testing.assert_allclose(q, [target[0] + math.radians(10), target[1] - math.radians(10)], atol=1e-9)


def test_head_gaze_joint_round_trip_and_sign():
    from lerobot_teleoperator_rby1.isaac_teleop.retargeters import gaze_to_head_joints, head_joints_to_gaze

    # Upright torso: head_1 > 0 looks down (READY_HEAD pitch +49 deg), head_0 > 0 looks left.
    d = head_joints_to_gaze(np.array([0.0, math.radians(49)]))
    assert d[2] < 0 and d[0] > 0
    d = head_joints_to_gaze(np.array([math.radians(30), 0.0]))
    assert d[1] > 0
    rng = np.random.default_rng(3)
    for _ in range(50):
        q = np.array([rng.uniform(-1.4, 1.4), rng.uniform(-1.2, 1.2)])
        R = Rotation.random(random_state=int(rng.integers(1 << 30))).as_matrix()
        np.testing.assert_allclose(gaze_to_head_joints(head_joints_to_gaze(q, R), R), q, atol=1e-9)


def test_head_retargeter_compensates_torso_rotation():
    from lerobot_teleoperator_rby1.isaac_teleop.retargeters import head_joints_to_gaze

    h = HeadRetargeter(smoothing=1.0, yaw_limit=math.radians(80))
    R_up = np.eye(3)
    h.latch_offset(_head_R(), np.array([0.0, 0.85]), R_up)
    q_up = h.update(_head_R(), R_up)
    np.testing.assert_allclose(q_up, [0.0, 0.85])
    # Torso yawed 30 deg left: the headset did not move, so the base gaze must not.
    R_yaw = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    q = h.update(_head_R(), R_yaw)
    assert q[0] == pytest.approx(math.radians(-30))
    assert q[1] == pytest.approx(0.85)
    np.testing.assert_allclose(head_joints_to_gaze(q, R_yaw), head_joints_to_gaze(q_up, R_up), atol=1e-9)
    # Torso pitched 20 deg forward (about +y): the head tilts back up by 20 deg.
    R_pitch = Rotation.from_euler("y", 20, degrees=True).as_matrix()
    q = h.update(_head_R(), R_pitch)
    assert q[1] == pytest.approx(0.85 - math.radians(20))
    # Held gaze (no headset) is compensated as well.
    q = h.update(None, R_yaw)
    assert q[0] == pytest.approx(math.radians(-30))
    # Legacy behaviour: joints ride with the torso.
    h2 = HeadRetargeter(smoothing=1.0, compensate_torso=False)
    h2.latch_offset(_head_R(), np.array([0.0, 0.85]), R_up)
    np.testing.assert_allclose(h2.update(_head_R(), R_yaw), [0.0, 0.85])


def test_head_hold_and_latch_offset_under_rotated_torso():
    # Joints measured under a yawed torso define a base gaze; back upright the
    # head must turn by the torso yaw to keep looking there.
    R_yaw = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    h = HeadRetargeter(smoothing=1.0)
    h.hold(np.array([0.0, 0.5]), R_yaw)
    np.testing.assert_allclose(h.target_for(R_yaw), [0.0, 0.5], atol=1e-9)
    assert h.target_for(np.eye(3))[0] == pytest.approx(math.radians(30))
    h.latch_offset(_head_R(), np.array([0.0, 0.5]), R_yaw)
    np.testing.assert_allclose(h.update(_head_R(), R_yaw), [0.0, 0.5], atol=1e-9)
    assert h.update(_head_R(), np.eye(3))[0] == pytest.approx(math.radians(30))


def test_clamp_pose_box_components():
    from lerobot_teleoperator_rby1.isaac_teleop.retargeters import clamp_pose_box
    from scipy.spatial.transform import Rotation

    pos_min, pos_max = [-0.15, -0.2, 0.8], [0.45, 0.2, 1.2]
    rpy_min, rpy_max = np.radians([-15, -20, -45]), np.radians([15, 50, 45])
    T = np.eye(4)
    T[:3, 3] = [0.1, 0.0, 1.0]
    T[:3, :3] = Rotation.from_euler("xyz", [5, 30, -10], degrees=True).as_matrix()
    out, clipped = clamp_pose_box(T, pos_min, pos_max, rpy_min, rpy_max)
    assert clipped == [] and np.allclose(out, T)
    # Outside in x (+), z (-), pitch (+) and yaw (-): only those components move.
    T2 = np.eye(4)
    T2[:3, 3] = [0.7, 0.1, 0.5]
    T2[:3, :3] = Rotation.from_euler("xyz", [5, 70, -60], degrees=True).as_matrix()
    out, clipped = clamp_pose_box(T2, pos_min, pos_max, rpy_min, rpy_max)
    assert clipped == ["x+", "z-", "pitch+", "yaw-"]
    np.testing.assert_allclose(out[:3, 3], [0.45, 0.1, 0.8])
    np.testing.assert_allclose(Rotation.from_matrix(out[:3, :3]).as_euler("xyz", degrees=True), [5, 50, -45], atol=1e-9)
