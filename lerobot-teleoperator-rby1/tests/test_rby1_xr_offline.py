"""Offline end-to-end test of Rby1XR with a scripted fake session."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lerobot_teleoperator_rby1.isaac_teleop import teleop_rby1_xr as mod
from lerobot_teleoperator_rby1.isaac_teleop.config_isaac_teleop import Rby1XRConfig
from lerobot_teleoperator_rby1.isaac_teleop.teleop_rby1_xr import Rby1XR
from lerobot_teleoperator_rby1.isaac_teleop.xr_frame import (
    OUT_BODY,
    OUT_CONTROLLER_LEFT,
    OUT_CONTROLLER_RIGHT,
    OUT_HEAD,
    BodyJointIndex,
)
from tests import fakes

# Robot-frame head rotation looking straight ahead (anchor -Z -> robot +X).
R_HEAD_FWD = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)


def _head_quat(yaw_deg=0.0, pitch_deg=0.0):
    R = R_HEAD_FWD @ Rotation.from_euler("y", yaw_deg, degrees=True).as_matrix() @ Rotation.from_euler("x", pitch_deg, degrees=True).as_matrix()
    return Rotation.from_matrix(R).as_quat()


def _frame(right=None, left=None, head=None, body=None):
    return {
        OUT_CONTROLLER_RIGHT: right if right is not None else fakes.ABSENT,
        OUT_CONTROLLER_LEFT: left if left is not None else fakes.ABSENT,
        OUT_HEAD: head if head is not None else fakes.ABSENT,
        OUT_BODY: body if body is not None else fakes.ABSENT,
    }


@pytest.fixture
def stubbed_pipeline(monkeypatch):
    monkeypatch.setattr(mod, "build_pipeline", lambda *, with_body: ("pipeline", True))
    monkeypatch.setattr(mod, "build_external_inputs", lambda anchor: {"base_T_anchor": anchor})
    monkeypatch.setattr(mod, "print_xr_connect_help", lambda: None)


def make_teleop(session, reader_holder, **cfg_overrides):
    # Most scenarios pre-date the shoulder-line reference and the engage-time
    # position offset; they opt back into the legacy behaviour (dedicated
    # tests cover the defaults).
    cfg_overrides.setdefault("reference_source", "auto")
    cfg_overrides.setdefault("ee_position_latch_on_a", False)
    cfg_overrides.setdefault("ready_return_mode", "ee")
    cfg = Rby1XRConfig(auto_launch_cloudxr=False, head_smoothing=1.0, **cfg_overrides)

    def reader_factory(address, model):
        reader_holder.append(fakes.FakeStateReader(address, model))
        return reader_holder[-1]

    return Rby1XR(
        cfg, session_factory=lambda pipeline: session, state_reader_factory=reader_factory
    )


def test_connect_waits_for_tracking_and_freezes_until_squeeze(stubbed_pipeline):
    session = fakes.FakeSession()
    session.push(_frame())  # nothing tracked yet
    session.push(_frame(right=fakes.controller((1, 1, 1)), left=fakes.controller((1, -1, 1))))
    readers: list = []
    t = make_teleop(session, readers, torso_source="head")
    t.connect()
    assert t.is_connected and session.entered and readers[0].connected
    assert session.steps == 2  # waited through the untracked frame

    a = t.get_action()
    assert set(a) == set(t.action_features)
    r = readers[0]
    assert a["right_ee.x"] == pytest.approx(r.right_ee[0, 3])
    assert a["left_ee.y"] == pytest.approx(r.left_ee[1, 3])
    assert a["torso_ee.z"] == pytest.approx(r.torso[2, 3])
    assert a["right_gripper_0.pos"] == 1.0 and a["x.vel"] == 0.0
    np.testing.assert_allclose([a["head_0.pos"], a["head_1.pos"]], r.head_q)
    t.disconnect()
    assert session.exited and not readers[0].connected and not t.is_connected



# Absolute arm mapping fixtures: human shoulder at (0, -0.3, 1.5), reach 0.6 m
# (config), robot shoulder = torso + [0, -0.22, 0.08]; hand offsets scale by
# robot_reach / 0.6 onto the robot shoulder.
ABS_SHOULDER = np.array([0.0, -0.3, 1.5])
ABS_KW = dict(engage_ramp_s=0.05, ee_max_linear_vel=100.0, ee_max_angular_vel=1000.0, shoulder_smoothing=1.0,
              arm_length_source="config", human_arm_length_m=0.6)


def _abs_body(side="right"):
    S = ABS_SHOULDER * (np.array([1, -1, 1]) if side == "left" else 1)
    return _body_with_arm(side, S, S + [0, 0, -0.3], S + [0, 0, -0.6])


def _abs_hand(dx=0.0, dy=0.0, dz=0.0):
    return tuple((ABS_SHOULDER + [dx, dy, dz]).tolist())


def _abs_goal(reader, dx=0.0, dy=0.0, dz=0.0, side="right"):
    from lerobot_teleoperator_rby1.isaac_teleop.arm_retargeter import robot_shoulder_position

    k = mod.robot_reach("1.3") / 0.6
    return robot_shoulder_position(reader.torso, side) + np.array([dx, dy, dz]) * k


def test_deadman_follows_holds_and_gripper(stubbed_pipeline, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock[0])
    body = _abs_body()
    session = fakes.FakeSession()
    session.push(_frame(right=fakes.controller(_abs_hand(0.3)), body=body))
    readers: list = []
    t = make_teleop(session, readers, torso_source="none", use_torso=False, use_left_arm=False, **ABS_KW)
    t.connect()
    t.get_action()
    r = readers[0]
    x0 = r.right_ee[0, 3]

    session.push(_frame(right=fakes.controller(_abs_hand(0.3), squeeze=0.9), body=body))  # engage edge
    a = t.get_action()
    assert a["right_ee.x"] == pytest.approx(x0)  # engage frame: ramp starts at the held target, no jump
    clock[0] += 0.1  # ramp done
    session.push(_frame(right=fakes.controller(_abs_hand(0.4, 0.0, 0.05), squeeze=0.9, trigger=0.75), body=body))
    a = t.get_action()
    goal = _abs_goal(r, 0.4, 0.0, 0.05)
    assert a["right_ee.x"] == pytest.approx(goal[0])
    assert a["right_ee.z"] == pytest.approx(goal[2])
    assert a["right_gripper_0.pos"] == pytest.approx(0.25)

    r.right_ee[:3, 3] = goal  # the robot followed
    session.push(_frame(right=fakes.controller(_abs_hand(1.0), squeeze=0.1), body=body))  # released: hold
    a = t.get_action()
    assert a["right_ee.x"] == pytest.approx(goal[0])
    assert "left_ee.x" not in a and "torso_ee.x" not in a

    # Re-engage: the ramp starts from the held target (no jump to the new hand).
    session.push(_frame(right=fakes.controller(_abs_hand(1.0), squeeze=0.9), body=body))
    a = t.get_action()
    assert a["right_ee.x"] == pytest.approx(goal[0])


def test_stop_and_resume(stubbed_pipeline, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock[0])
    body = _abs_body()
    session = fakes.FakeSession()
    session.push(_frame(right=fakes.controller(_abs_hand(), squeeze=0.9, thumb=(0, 1.0)), body=body))
    readers: list = []
    t = make_teleop(session, readers, torso_source="none", use_left_arm=False, ready_return_duration_s=0.1, **ABS_KW)
    t.connect()
    a = t.get_action()
    assert a["x.vel"] == pytest.approx(0.3)
    x0 = a["right_ee.x"]

    session.push(_frame(right=fakes.controller(_abs_hand(0.2), squeeze=0.9, thumb=(0, 1.0), secondary=True), body=body))
    a = t.get_action()  # Right B: stop
    assert t.is_stopped and a["x.vel"] == 0.0 and a["right_ee.x"] == pytest.approx(x0)
    clock[0] += 0.1
    session.push(_frame(right=fakes.controller(_abs_hand(0.5), squeeze=0.9, thumb=(0, 1.0)), body=body))
    a = t.get_action()  # still stopped: nothing moves
    assert a["right_ee.x"] == pytest.approx(x0) and a["x.vel"] == 0.0

    session.push(_frame(right=fakes.controller(_abs_hand(0.5), squeeze=0.9, primary=True), body=body))
    a = t.get_action()  # Right A: resume + return-to-start motion (already there)
    assert not t.is_stopped and a["right_ee.x"] == pytest.approx(x0)
    clock[0] += 0.15  # let the (0.1 s) return motion finish
    session.push(_frame(right=fakes.controller(_abs_hand(0.5), squeeze=0.9), body=body))
    a = t.get_action()  # motion done -> re-engage: ramp starts at the start pose, no jump
    assert a["right_ee.x"] == pytest.approx(x0)
    clock[0] += 0.1
    session.push(_frame(right=fakes.controller(_abs_hand(0.5), squeeze=0.9), body=body))
    assert t.get_action()["right_ee.x"] == pytest.approx(_abs_goal(readers[0], 0.5)[0])


def test_head_latches_then_follows_yaw(stubbed_pipeline):
    session = fakes.FakeSession()
    session.push(_frame(right=fakes.controller(), head=fakes.head(quat=_head_quat(0))))
    readers: list = []
    t = make_teleop(session, readers, torso_source="none")
    t.connect()
    a = t.get_action()  # latch frame
    assert a["head_0.pos"] == pytest.approx(0.0)
    session.push(_frame(right=fakes.controller(), head=fakes.head(quat=_head_quat(25))))
    a = t.get_action()
    assert a["head_0.pos"] == pytest.approx(math.radians(25))
    assert a["head_1.pos"] == pytest.approx(readers[0].head_q[1])
    session.push(_frame(right=fakes.controller()))  # head lost: hold
    assert t.get_action()["head_0.pos"] == pytest.approx(math.radians(25))


def test_torso_follows_body_only_when_both_arms_clutched(stubbed_pipeline):
    pos = np.zeros((24, 3), np.float32)
    pos[BodyJointIndex.SPINE3] = [0, 0, 1.2]
    quat = np.tile(np.array([0, 0, 0, 1], np.float32), (24, 1))
    body0 = fakes.body(positions=pos, orientations=quat)
    session = fakes.FakeSession()
    session.push(_frame(right=fakes.controller(squeeze=0.9), left=fakes.controller(squeeze=0.9), body=body0))
    readers: list = []
    t = make_teleop(session, readers, torso_source="body", torso_max_z_delta_m=0.15, torso_max_rot_delta_deg=35.0, torso_use_xy=False)
    t.connect()
    a = t.get_action()  # both arms engage; torso engages on the same tick
    z0 = readers[0].torso[2, 3]
    assert a["torso_ee.z"] == pytest.approx(z0)

    pos2 = pos.copy()
    pos2[BodyJointIndex.SPINE3] = [0.3, 0.0, 0.9]  # squat 0.3 m (clamped to 0.15), xy ignored
    quat2 = np.tile(Rotation.from_euler("y", 10, degrees=True).as_quat().astype(np.float32), (24, 1))
    session.push(_frame(right=fakes.controller(squeeze=0.9), left=fakes.controller(squeeze=0.9), body=fakes.body(positions=pos2, orientations=quat2)))
    a = t.get_action()
    assert a["torso_ee.z"] == pytest.approx(z0 - 0.15)
    assert a["torso_ee.x"] == pytest.approx(readers[0].torso[0, 3])
    assert abs(a["torso_ee.wy"]) == pytest.approx(math.radians(10))

    # Release one arm -> torso freezes even though the body keeps moving.
    readers[0].torso[2, 3] = z0 - 0.15  # robot has reached the commanded height
    pos3 = pos2.copy()
    pos3[BodyJointIndex.SPINE3] = [0.0, 0.0, 1.2]
    session.push(_frame(right=fakes.controller(squeeze=0.9), left=fakes.controller(squeeze=0.0), body=fakes.body(positions=pos3, orientations=quat2)))
    a = t.get_action()
    assert a["torso_ee.z"] == pytest.approx(z0 - 0.15)

    # Invalid body -> torso holds; valid again -> re-latches without a jump.
    # (The fake robot does not move by itself; emulate it having reached the
    # commanded torso height, otherwise the drift re-sync would snap the held
    # target back to the measured pose.)
    readers[0].torso[2, 3] = z0 - 0.15
    valid = np.ones(24, np.uint8)
    valid[BodyJointIndex.PELVIS] = 0
    session.push(_frame(right=fakes.controller(squeeze=0.9), left=fakes.controller(squeeze=0.9), body=fakes.body(positions=pos3, orientations=quat2, valid=valid)))
    a = t.get_action()
    assert a["torso_ee.z"] == pytest.approx(z0 - 0.15)
    session.push(_frame(right=fakes.controller(squeeze=0.9), left=fakes.controller(squeeze=0.9), body=fakes.body(positions=pos3, orientations=quat2)))
    a = t.get_action()
    assert a["torso_ee.z"] == pytest.approx(z0 - 0.15)


def test_first_action_session_start(stubbed_pipeline):
    session = fakes.FakeSession()
    session.push(_frame(right=fakes.controller()))
    readers: list = []
    t = make_teleop(session, readers, torso_source="none", session_start="first_action")
    t.connect()
    assert t.is_connected and not session.entered
    t.get_action()
    assert session.entered
    t.disconnect()
    assert session.exited


def test_worker_exception_propagates(stubbed_pipeline):
    session = fakes.FakeSession()
    session.push(_frame(right=fakes.controller()))
    readers: list = []
    t = make_teleop(session, readers, torso_source="none")
    t.connect()
    session.last_step_info = fakes.FakeStepInfo(worker_exception=ValueError("boom"))
    with pytest.raises(RuntimeError):
        t.get_action()


def test_config_validation():
    with pytest.raises(ValueError):
        Rby1XRConfig(torso_source="hips")
    with pytest.raises(ValueError):
        Rby1XRConfig(session_start="later")
    with pytest.raises(ValueError):
        Rby1XRConfig(torso_body_joint="CHEST")
    assert Rby1XRConfig(torso_body_joint="SPINE2").torso_body_joint == "SPINE2"


@pytest.mark.parametrize("policy,squeeze_left,expect_follow", [("both_arms", 0.0, False), ("any_arm", 0.0, True), ("always", 0.0, True)])
def test_torso_engage_policy(stubbed_pipeline, policy, squeeze_left, expect_follow):
    session = fakes.FakeSession()
    session.push(_frame(right=fakes.controller(squeeze=0.9), left=fakes.controller(squeeze=squeeze_left), head=fakes.head(quat=_head_quat(0))))
    readers: list = []
    t = make_teleop(session, readers, torso_source="head", torso_engage=policy, use_head=False)
    t.connect()
    t.get_action()  # latch
    z0 = readers[0].torso[2, 3]
    session.push(_frame(right=fakes.controller(squeeze=0.9), left=fakes.controller(squeeze=squeeze_left), head=fakes.head(pos=(0, 0, -0.1), quat=_head_quat(0))))
    a = t.get_action()
    if expect_follow:
        assert a["torso_ee.z"] == pytest.approx(z0 - 0.1)
        assert t._torso_hold_reason == "following"
    else:
        assert a["torso_ee.z"] == pytest.approx(z0)
        assert "arms not clutched" in t._torso_hold_reason


def test_first_action_resyncs_to_ready_pose_reached_after_connect(stubbed_pipeline):
    session = fakes.FakeSession()
    session.push(_frame(right=fakes.controller(), head=fakes.head(quat=_head_quat(0))))
    readers: list = []
    t = make_teleop(session, readers, torso_source="none")
    t.connect()  # latched at the pre-ready pose
    r = readers[0]
    # The follower now moves to a very different ready pose (after teleop.connect()).
    r.right_ee[:3, 3] = [0.2, -0.3, 0.6]
    r.left_ee[:3, 3] = [0.2, 0.3, 0.6]
    r.torso[2, 3] = 0.7
    r.head_q = np.array([0.0, 0.85])
    a = t.get_action()
    assert a["right_ee.x"] == pytest.approx(0.2) and a["right_ee.z"] == pytest.approx(0.6)
    assert a["left_ee.y"] == pytest.approx(0.3)
    assert a["torso_ee.z"] == pytest.approx(0.7)
    assert a["head_1.pos"] == pytest.approx(0.85)


def test_disengaged_components_follow_robot_after_reset(stubbed_pipeline, monkeypatch):
    monkeypatch.setattr(mod.time, "monotonic", lambda: 100.0)  # frozen: the engage ramp does not advance
    session = fakes.FakeSession()
    session.push(_frame(right=fakes.controller()))
    readers: list = []
    t = make_teleop(session, readers, torso_source="none", use_head=False)
    t.connect()
    t.get_action()  # nothing clutched
    r = readers[0]
    left_before = t.get_action()["left_ee.x"]
    # A record reset moved the LEFT arm by 20 cm while nothing is clutched: target follows.
    r.left_ee[0, 3] += 0.2
    a = t.get_action()
    assert a["left_ee.x"] == pytest.approx(left_before + 0.2)
    # A small sag (below threshold) does not move the held target.
    r.left_ee[0, 3] += 0.01
    assert t.get_action()["left_ee.x"] == pytest.approx(left_before + 0.2)
    # While the RIGHT arm is clutched nothing is re-synced: the torso carries
    # the free arm along and that must not overwrite its held target.
    session.push(_frame(right=fakes.controller(_abs_hand(), squeeze=0.9), body=_abs_body()))
    a = t.get_action()
    right_before, left_held = a["right_ee.x"], a["left_ee.x"]
    r.right_ee[0, 3] += 0.5
    r.left_ee[0, 3] += 0.5
    a = t.get_action()
    assert a["right_ee.x"] == pytest.approx(right_before)
    assert a["left_ee.x"] == pytest.approx(left_held)


def test_right_a_returns_to_start_pose(stubbed_pipeline, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock[0])
    body = _abs_body()
    session = fakes.FakeSession()
    session.push(_frame(right=fakes.controller(_abs_hand(), squeeze=0.9), head=fakes.head(quat=_head_quat(0)), body=body))
    readers: list = []
    t = make_teleop(session, readers, torso_source="none", ready_return_duration_s=2.0, **ABS_KW)
    t.connect()
    a0 = t.get_action()  # start pose recorded here (first action); dead-man engaged -> ramp
    x_start, head_start = a0["right_ee.x"], a0["head_1.pos"]
    clock[0] += 0.1  # ramp done
    # Move the hand 30 cm forward and the head by yawing.
    session.push(_frame(right=fakes.controller(_abs_hand(0.3), squeeze=0.9), head=fakes.head(quat=_head_quat(30)), body=body))
    a = t.get_action()
    x_fwd = _abs_goal(readers[0], 0.3)[0]
    assert a["right_ee.x"] == pytest.approx(x_fwd)
    assert a["head_0.pos"] == pytest.approx(math.radians(30))
    # Right A: the dead-man is released and the targets interpolate back. (The
    # operator faces forward again here so the yaw reference stays unchanged.)
    session.push(_frame(right=fakes.controller(_abs_hand(0.3), squeeze=0.9, primary=True), head=fakes.head(quat=_head_quat(0)), body=body))
    a = t.get_action()
    assert a["right_ee.x"] == pytest.approx(x_fwd)  # alpha = 0
    clock[0] += 1.0  # halfway (smoothstep(0.5) = 0.5)
    session.push(_frame(right=fakes.controller(_abs_hand(0.9), squeeze=0.9), head=fakes.head(quat=_head_quat(30)), body=body))
    a = t.get_action()
    assert a["right_ee.x"] == pytest.approx(x_start + 0.5 * (x_fwd - x_start))  # squeeze is ignored while returning
    assert a["head_0.pos"] == pytest.approx(math.radians(15))
    clock[0] += 1.5
    a = t.get_action()
    assert a["right_ee.x"] == pytest.approx(x_start)
    # The head origin was latched when A was pressed (headset at 0 deg ↦ start
    # pose); the headset is still at 30 deg, so after arrival the head follows.
    assert a["head_1.pos"] == pytest.approx(head_start) and a["head_0.pos"] == pytest.approx(math.radians(30))
    # After arrival a squeeze re-engages with a ramp from the start pose (no jump).
    readers[0].right_ee[0, 3] = x_start
    session.push(_frame(right=fakes.controller(_abs_hand(0.5), squeeze=0.9), body=body))
    assert t.get_action()["right_ee.x"] == pytest.approx(x_start)
    clock[0] += 0.1
    session.push(_frame(right=fakes.controller(_abs_hand(0.5), squeeze=0.9), body=body))
    assert t.get_action()["right_ee.x"] == pytest.approx(_abs_goal(readers[0], 0.5)[0])


def test_right_a_rereferences_operator_yaw(stubbed_pipeline, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock[0])
    session = fakes.FakeSession()
    # Operator initially faces robot +X (head looks along anchor -Z -> robot +X).
    session.push(_frame(right=fakes.controller(_abs_hand()), head=fakes.head(quat=_head_quat(0)), body=_abs_body()))
    readers: list = []
    t = make_teleop(session, readers, torso_source="none", use_head=False, ready_return_duration_s=0.5, **ABS_KW)
    t.connect()
    t.get_action()
    # The operator turns 90 deg to the left (now facing robot +Y) and presses A.
    # Nothing in the body frame is used for the reference here (no shoulders) -> head gaze.
    session.push(_frame(right=fakes.controller(_abs_hand(), primary=True), head=fakes.head(quat=_head_quat(90))))
    t.get_action()
    assert t._yaw_correction == pytest.approx(-math.pi / 2)
    clock[0] += 1.0  # return motion done
    t.get_action()
    # Squeeze with the hand 0.3 m "forward" for the operator = raw robot +Y of the shoulder.
    body = _abs_body()
    hand = tuple((ABS_SHOULDER + [0.0, 0.3, 0.0]).tolist())
    session.push(_frame(right=fakes.controller(hand, squeeze=0.9), head=fakes.head(quat=_head_quat(90)), body=body))
    t.get_action()
    clock[0] += 0.1  # ramp done
    session.push(_frame(right=fakes.controller(hand, squeeze=0.9), head=fakes.head(quat=_head_quat(90)), body=body))
    a = t.get_action()
    goal = _abs_goal(readers[0], 0.3)  # forward for the operator -> robot +X from the robot shoulder
    assert a["right_ee.x"] == pytest.approx(goal[0])
    assert a["right_ee.y"] == pytest.approx(goal[1])


def _body_with_arm(side, shoulder, elbow, wrist):
    pos = np.zeros((24, 3), np.float32)
    idx = {
        "right": (BodyJointIndex.RIGHT_SHOULDER, BodyJointIndex.RIGHT_ELBOW, BodyJointIndex.RIGHT_WRIST),
        "left": (BodyJointIndex.LEFT_SHOULDER, BodyJointIndex.LEFT_ELBOW, BodyJointIndex.LEFT_WRIST),
    }[side]
    for i, p in zip(idx, (shoulder, elbow, wrist)):
        pos[int(i)] = p
    return fakes.body(positions=pos)


def test_absolute_mode_deadman_ramp_track_hold(stubbed_pipeline, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock[0])
    # Human: shoulder at (0,-0.3,1.5), arm straight down (elbow / wrist below), reach 0.6.
    body = _body_with_arm("right", [0, -0.3, 1.5], [0, -0.3, 1.2], [0, -0.3, 0.9])
    hand0 = (0.0, -0.3, 0.9)
    session = fakes.FakeSession()
    session.push(_frame(right=fakes.controller(hand0), body=body))
    readers: list = []
    t = make_teleop(
        session, readers, torso_source="none", use_torso=False,
        use_left_arm=False, use_head=False, engage_ramp_s=1.0, ee_max_linear_vel=100.0,
        shoulder_smoothing=1.0, arm_length_source="config", human_arm_length_m=0.6,
    )
    t.connect()
    a = t.get_action()  # first action: start pose + orientation offset latched
    x0 = readers[0].right_ee[0, 3]
    assert a["right_ee.x"] == pytest.approx(x0)  # dead-man not held -> hold
    reach = mod.robot_reach("1.3")
    # Hand 0.3 m in front of the human shoulder -> robot: shoulder + 0.3 * (reach / 0.6) along x.
    hand = (0.3, -0.3, 1.5)
    goal_x = readers[0].torso[0, 3] + 0.0 + 0.3 * reach / 0.6
    session.push(_frame(right=fakes.controller(hand, squeeze=0.9), body=body))
    a = t.get_action()  # engage edge: ramp starts at the current target
    assert a["right_ee.x"] == pytest.approx(x0)
    clock[0] += 0.5
    a = t.get_action()  # halfway (smoothstep(0.5) = 0.5)
    assert a["right_ee.x"] == pytest.approx(x0 + 0.5 * (goal_x - x0))
    clock[0] += 0.6
    a = t.get_action()  # ramp done -> tracking the absolute target
    assert a["right_ee.x"] == pytest.approx(goal_x)
    assert t._abs_state["right"] == "track"
    # Move the hand: target follows absolutely (no clutch delta).
    hand2 = (0.15, -0.3, 1.5)
    clock[0] += 0.02
    session.push(_frame(right=fakes.controller(hand2, squeeze=0.9), body=body))
    a = t.get_action()
    assert a["right_ee.x"] == pytest.approx(readers[0].torso[0, 3] + 0.15 * reach / 0.6)
    # Release: hold, hand keeps moving.
    session.push(_frame(right=fakes.controller((0.5, -0.3, 1.5), squeeze=0.0), body=body))
    held = t.get_action()["right_ee.x"]
    assert held == pytest.approx(readers[0].torso[0, 3] + 0.15 * reach / 0.6)
    assert t._abs_state["right"] == "hold"


def test_absolute_mode_holds_without_body(stubbed_pipeline, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock[0])
    session = fakes.FakeSession()
    session.push(_frame(right=fakes.controller((0.3, -0.3, 1.5), squeeze=0.9)))
    readers: list = []
    t = make_teleop(session, readers, torso_source="none", use_torso=False, use_left_arm=False, use_head=False)
    t.connect()
    a = t.get_action()
    assert a["right_ee.x"] == pytest.approx(readers[0].right_ee[0, 3])
    assert "hold" in t._abs_state["right"]


def test_posture_hint_keys_and_recording(stubbed_pipeline, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock[0])
    q_true = np.array([0.3, -0.4, 0.2, -1.2])
    S, E, W = mod.forward_points(q_true, "right") if hasattr(mod, "forward_points") else (None, None, None)
    from lerobot_teleoperator_rby1.isaac_teleop.arm_retargeter import forward_points

    S, E, W = forward_points(q_true, "right")
    T = np.eye(4)  # torso at the base origin -> body points are directly in the torso frame
    body = _body_with_arm("right", S, E, W)
    session = fakes.FakeSession()
    session.push(_frame(right=fakes.controller(W, squeeze=0.9), body=body))  # hints only while engaged
    readers: list = []
    t = make_teleop(session, readers, torso_source="none", use_torso=False, use_left_arm=False, use_head=False,
                    arm_posture_hint=True, posture_hint_smoothing=1.0, posture_hint_max_vel=1000.0)
    t.connect()
    readers[0].torso = T
    a = t.get_action()
    for i in range(4):
        assert a[f"right_arm_{i}.null"] == pytest.approx(q_true[i], abs=1e-3)
    assert "right_arm_0.null" not in t.action_features  # not recorded by default
    # Recording flag exposes the keys as features.
    t2 = make_teleop(fakes.FakeSession(), [], torso_source="none", use_torso=False, use_left_arm=False, use_head=False,
                     arm_posture_hint=True, record_posture_hint=True)
    assert "right_arm_3.null" in t2.action_features and "left_arm_0.null" not in t2.action_features
    # Body lost while engaged (beyond hold_s): the last hint is kept — the
    # solver must not fall back to its default nullspace pose mid-motion.
    session.push(_frame(right=fakes.controller(W, squeeze=0.9)))
    clock[0] += 5.0
    a = t.get_action()
    assert a["right_arm_0.null"] == pytest.approx(q_true[0], abs=1e-3)


def test_posture_hint_seeded_from_measured_joints_and_ramped(stubbed_pipeline, monkeypatch):
    """The first hint starts at the robot's actual arm_0..3 and ramps at max_vel."""
    clock = [100.0]
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock[0])
    from lerobot_teleoperator_rby1.isaac_teleop.arm_retargeter import forward_points

    q_robot = np.array([0.26, -0.17, -0.26, -2.0, 0, 0, 0])  # ready-like
    q_human = np.array([0.3, -0.4, 0.2, -1.2])
    S, E, W = forward_points(q_human, "right")
    session = fakes.FakeSession()
    session.push(_frame(right=fakes.controller(W, squeeze=0.9), body=_body_with_arm("right", S, E, W)))
    readers: list = []
    t = make_teleop(session, readers, torso_source="none", use_torso=False, use_left_arm=False, use_head=False,
                    arm_posture_hint=True, posture_hint_smoothing=1.0, posture_hint_max_vel=1.0)
    t.connect()
    readers[0].torso = np.eye(4)
    readers[0].right_q = q_robot
    a = t.get_action()  # first tick: dt defaults to 1/60 s -> at most 1/60 rad away from the seed
    for i in range(4):
        assert abs(a[f"right_arm_{i}.null"] - q_robot[i]) <= 1.0 / 60.0 + 1e-9
    # Converges towards the human posture over time.
    for _ in range(120):
        clock[0] += 0.02
        session.push(_frame(right=fakes.controller(W, squeeze=0.9), body=_body_with_arm("right", S, E, W)))
        a = t.get_action()
    for i in range(4):
        assert a[f"right_arm_{i}.null"] == pytest.approx(q_human[i], abs=1e-3)


def test_recorded_hint_without_body_is_measured_posture_not_zeros(stubbed_pipeline):
    """record_posture_hint=True with no hint available must not emit q = 0."""
    q_robot = np.array([0.26, -0.17, -0.26, -2.0, 0, 0, 0])
    session = fakes.FakeSession()
    session.push(_frame(right=fakes.controller((0.3, -0.2, 0.9), squeeze=0.0)))
    readers: list = []
    t = make_teleop(session, readers, torso_source="none", use_torso=False, use_left_arm=False, use_head=False,
                    arm_posture_hint=True, record_posture_hint=True)
    t.connect()
    readers[0].right_q = q_robot
    a = t.get_action()
    for i in range(4):
        assert a[f"right_arm_{i}.null"] == pytest.approx(q_robot[i], abs=1e-9)


def test_posture_hint_dropped_on_right_a(stubbed_pipeline, monkeypatch):
    """Right A (re-reference + return motion) discards hints from the old frame."""
    clock = [100.0]
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock[0])
    from lerobot_teleoperator_rby1.isaac_teleop.arm_retargeter import forward_points

    q_a = np.array([0.3, -0.4, 0.2, -1.2])
    S, E, W = forward_points(q_a, "right")
    body = _body_with_arm("right", S, E, W)
    session = fakes.FakeSession()
    session.push(_frame(right=fakes.controller(W, squeeze=0.9), body=body))
    readers: list = []
    t = make_teleop(session, readers, torso_source="none", use_torso=False, use_left_arm=False, use_head=False,
                    arm_posture_hint=True, posture_hint_smoothing=1.0, posture_hint_max_vel=1000.0)
    t.connect()
    readers[0].torso = np.eye(4)
    a = t.get_action()
    assert "right_arm_0.null" in a
    clock[0] += 0.02
    session.push(_frame(right=fakes.controller(W, squeeze=0.9, primary=True), body=body))
    a = t.get_action()
    assert "right_arm_0.null" not in a


def _shoulders_body(right_xy, left_xy, z=1.5):
    pos = np.zeros((24, 3), np.float32)
    pos[BodyJointIndex.RIGHT_SHOULDER] = [right_xy[0], right_xy[1], z]
    pos[BodyJointIndex.LEFT_SHOULDER] = [left_xy[0], left_xy[1], z]
    return fakes.body(positions=pos)


def test_reference_prefers_shoulder_line_over_head(stubbed_pipeline):
    session = fakes.FakeSession()
    # Shoulders: right at (0,-0.2), left at (0,+0.2) -> facing +X, even though the head looks 60 deg left.
    session.push(_frame(right=fakes.controller(), head=fakes.head(quat=_head_quat(60)), body=_shoulders_body((0, -0.2), (0, 0.2))))
    readers: list = []
    # Body tracking is only in the pipeline when something needs it (torso / hint / absolute mode).
    t = make_teleop(session, readers, torso_source="none", use_torso=False, use_head=False, arm_posture_hint=True)
    t.connect()
    t.get_action()
    assert t._yaw_correction == pytest.approx(0.0, abs=1e-6)
    # Operator turns 90 deg left (shoulders now along -x .. +x): facing +Y.
    session.push(_frame(right=fakes.controller(primary=True), body=_shoulders_body((0.2, 0), (-0.2, 0))))
    t.get_action()
    assert t._yaw_correction == pytest.approx(-math.pi / 2, abs=1e-6)


def test_right_a_latches_orientation_against_start_pose(stubbed_pipeline, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock[0])
    body = _body_with_arm("right", [0, -0.3, 1.5], [0, -0.3, 1.2], [0, -0.3, 0.9])
    session = fakes.FakeSession()
    session.push(_frame(right=fakes.controller((0.3, -0.3, 1.5)), body=body))
    readers: list = []
    t = make_teleop(
        session, readers, torso_source="none", use_torso=False,
        use_left_arm=False, use_head=False, engage_ramp_s=0.5, ee_max_linear_vel=100.0, ee_max_angular_vel=100.0,
        shoulder_smoothing=1.0, arm_length_source="config", human_arm_length_m=0.6, ready_return_duration_s=0.5,
    )
    t.connect()
    t.get_action()  # start pose recorded, offset latched (controller identity -> start orientation)
    start_R = readers[0].right_ee[:3, :3].copy()
    # The robot is now somewhere else (arm moved), the operator presses A while the
    # controller is rotated 30 deg about z: after the return, squeezing with the
    # controller in that same orientation must reproduce the START orientation.
    readers[0].right_ee[:3, :3] = Rotation.from_euler("x", 45, degrees=True).as_matrix()
    q30 = Rotation.from_euler("z", 30, degrees=True).as_quat()
    session.push(_frame(right=fakes.controller((0.3, -0.3, 1.5), quat=q30, primary=True), body=body))
    t.get_action()
    clock[0] += 1.0
    session.push(_frame(right=fakes.controller((0.3, -0.3, 1.5), quat=q30), body=body))
    t.get_action()  # return motion finished
    readers[0].right_ee[:3, :3] = start_R  # robot is back at the start orientation
    session.push(_frame(right=fakes.controller((0.3, -0.3, 1.5), quat=q30, squeeze=0.9), body=body))
    t.get_action()  # engage (ramp from start)
    clock[0] += 1.0
    a = t.get_action()
    R_cmd = Rotation.from_rotvec([a["right_ee.wx"], a["right_ee.wy"], a["right_ee.wz"]]).as_matrix()
    np.testing.assert_allclose(R_cmd, start_R, atol=1e-6)


# ---------------------------------------------------------------------------
# Neck wear mode
# ---------------------------------------------------------------------------


def test_neck_mode_head_fixed_and_torso_follows_headset(stubbed_pipeline):
    session = fakes.FakeSession()
    session.push(_frame(right=fakes.controller(squeeze=0.9), head=fakes.head((0.0, 0.0, 1.5), quat=_head_quat(0))))
    readers: list = []
    t = make_teleop(session, readers, wear_mode="neck", torso_source="body", torso_engage="any_arm",
                    neck_torso_smoothing=1.0, use_left_arm=False, torso_use_xy=True, torso_max_rot_delta_deg=35.0)
    t.connect()
    a = t.get_action()  # torso engages on the headset pose
    head_q = readers[0].head_q.copy()
    z0 = readers[0].torso[2, 3]
    np.testing.assert_allclose([a["head_0.pos"], a["head_1.pos"]], head_q)
    assert t._torso_hold_reason == "following"
    # Headset (hanging from the neck) moves down 10 cm and yaws 30 deg: the head
    # joints stay put, the torso follows the headset pose.
    session.push(_frame(right=fakes.controller(squeeze=0.9), head=fakes.head((0.0, 0.0, 1.4), quat=_head_quat(30))))
    a = t.get_action()
    np.testing.assert_allclose([a["head_0.pos"], a["head_1.pos"]], head_q)
    assert a["torso_ee.z"] == pytest.approx(z0 - 0.1)
    assert abs(a["torso_ee.wz"]) == pytest.approx(math.radians(30), abs=1e-6)


def test_neck_mode_reference_from_controllers_when_no_body(stubbed_pipeline):
    session = fakes.FakeSession()
    # Headset at the origin looking "down" (gaze is meaningless), controllers held out along +Y.
    q_down = Rotation.from_matrix(R_HEAD_FWD @ Rotation.from_euler("x", -80, degrees=True).as_matrix()).as_quat()
    session.push(_frame(right=fakes.controller((0.1, 0.5, 1.0)), left=fakes.controller((-0.1, 0.5, 1.0)), head=fakes.head((0, 0, 1.3), quat=q_down)))
    readers: list = []
    t = make_teleop(session, readers, wear_mode="neck", torso_source="none", use_torso=False)
    t.connect()
    t.get_action()
    assert t._yaw_correction == pytest.approx(-math.pi / 2, abs=1e-6)  # facing +Y -> rotated to +X


def test_neck_mode_absolute_ee_uses_headset_shoulder_estimate(stubbed_pipeline, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock[0])
    session = fakes.FakeSession()
    head_pos = (0.0, 0.0, 1.6)
    # Both controllers held symmetrically (their midpoint sets the facing direction: +X).
    left_ctrl = fakes.controller((0.3, 0.2, 1.45))
    session.push(_frame(right=fakes.controller((0.3, -0.2, 1.45)), left=left_ctrl, head=fakes.head(head_pos, quat=_head_quat(0))))
    readers: list = []
    t = make_teleop(
        session, readers, wear_mode="neck", torso_source="none", use_torso=False,
        use_left_arm=False, use_head=False, engage_ramp_s=0.5, ee_max_linear_vel=100.0,
        arm_length_source="config", human_arm_length_m=0.6, neck_shoulder_offset=[-0.05, 0.20, -0.15],
    )
    t.connect()
    t.get_action()
    # Right shoulder estimate = head + (-0.05, -0.20, -0.15) = (-0.05, -0.20, 1.45); hand 0.35 m ahead of it.
    session.push(_frame(right=fakes.controller((0.30, -0.2, 1.45), squeeze=0.9), left=left_ctrl, head=fakes.head(head_pos, quat=_head_quat(0))))
    t.get_action()
    clock[0] += 1.0
    a = t.get_action()
    reach = mod.robot_reach("1.3")
    expected_x = readers[0].torso[0, 3] + 0.35 * reach / 0.6
    assert a["right_ee.x"] == pytest.approx(expected_x, abs=1e-6)
    assert t._abs_state["right"] == "track"


def test_viz_enabled_uses_televiz_owned_session(stubbed_pipeline):
    from tests.test_viz_panels import FakeTeleviz

    session = fakes.FakeSession()
    session.push(_frame(right=fakes.controller()))
    readers: list = []
    cfg = dict(torso_source="none", use_torso=False, use_head=False, viz_enabled=True,
               viz_cameras=["front", "left", "right"], viz_offsets_x=[0.0, -1.1, 1.1])

    def reader_factory(address, model):
        readers.append(fakes.FakeStateReader(address, model))
        return readers[-1]

    t = mod.Rby1XR(
        Rby1XRConfig(auto_launch_cloudxr=False, head_smoothing=1.0, **cfg),
        session_factory=lambda pipeline: session,
        state_reader_factory=reader_factory,
        viz_module=FakeTeleviz,
        viz_uploader=lambda f: f,
    )
    t.connect()
    viz_session = FakeTeleviz.VizSession.last
    assert viz_session is not None and t._viz is not None
    a = t.get_action()
    assert "right_ee.x" in a
    import time as _time

    _time.sleep(0.05)
    assert viz_session.renders > 0
    t.disconnect()
    assert viz_session.destroyed and session.exited


def test_config_wear_and_viz_validation():
    with pytest.raises(ValueError):
        Rby1XRConfig(wear_mode="chest")
    with pytest.raises(ValueError):
        Rby1XRConfig(viz_enabled=True, viz_cameras=["front"], viz_offsets_x=[0.0, 1.0])
    assert Rby1XRConfig(viz_enabled=True, viz_cameras=["front"], viz_offsets_x=[0.0]).viz_lock_mode == "gimbal"


def test_posture_hint_frozen_while_released(stubbed_pipeline, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock[0])
    from lerobot_teleoperator_rby1.isaac_teleop.arm_retargeter import forward_points

    q_a = np.array([0.3, -0.4, 0.2, -1.2])
    q_b = np.array([0.8, -0.9, 0.5, -0.6])
    S, E, W = forward_points(q_a, "right")
    session = fakes.FakeSession()
    session.push(_frame(right=fakes.controller(W, squeeze=0.9), body=_body_with_arm("right", S, E, W)))
    readers: list = []
    t = make_teleop(session, readers, torso_source="none", use_torso=False, use_left_arm=False, use_head=False,
                    arm_posture_hint=True, posture_hint_smoothing=1.0, posture_hint_max_vel=100.0)
    t.connect()
    readers[0].torso = np.eye(4)
    a = t.get_action()  # engaged: hint follows the arm
    assert a["right_arm_0.null"] == pytest.approx(q_a[0], abs=1e-5)
    # Release the squeeze and move the arm: the hint must not change.
    S2, E2, W2 = forward_points(q_b, "right")
    clock[0] += 0.02
    session.push(_frame(right=fakes.controller(W2, squeeze=0.0), body=_body_with_arm("right", S2, E2, W2)))
    a = t.get_action()
    assert a["right_arm_0.null"] == pytest.approx(q_a[0], abs=1e-5)
    assert a["right_arm_3.null"] == pytest.approx(q_a[3], abs=1e-5)
    # Squeeze again: the hint follows the new posture.
    clock[0] += 0.02
    session.push(_frame(right=fakes.controller(W2, squeeze=0.9), body=_body_with_arm("right", S2, E2, W2)))
    a = t.get_action()
    assert a["right_arm_0.null"] == pytest.approx(q_b[0], abs=1e-3)


def test_head_absolute_mode_follows_headset_orientation(stubbed_pipeline):
    session = fakes.FakeSession()
    session.push(_frame(right=fakes.controller(), head=fakes.head(quat=_head_quat(0))))
    readers: list = []
    readers_head = np.array([0.3, 0.85])  # robot head starts somewhere else (ready pitch)
    t = make_teleop(session, readers, torso_source="none", use_torso=False, head_pitch_offset_deg=10.0,
                    head_latch_on_a=False)  # legacy: pure absolute mapping with the fixed offsets
    t.connect()
    readers[0].head_q = readers_head
    a = t.get_action()  # first frame: latch marks only; the held (measured) value is emitted
    np.testing.assert_allclose([a["head_0.pos"], a["head_1.pos"]], readers_head)
    a = t.get_action()  # straight ahead -> yaw 0, pitch offset 10 deg (absolute, not the measured 0.85)
    assert a["head_0.pos"] == pytest.approx(0.0, abs=1e-6)
    assert a["head_1.pos"] == pytest.approx(math.radians(10.0), abs=1e-6)
    session.push(_frame(right=fakes.controller(), head=fakes.head(quat=_head_quat(-40))))
    a = t.get_action()
    assert a["head_0.pos"] == pytest.approx(math.radians(-40), abs=1e-6)
    # Right A does not re-centre the head in absolute mode: the same headset yaw
    # keeps giving the same joint angle (only the operator-frame reference matters).
    session.push(_frame(right=fakes.controller(primary=True), head=fakes.head(quat=_head_quat(-40))))
    t.get_action()
    session.push(_frame(right=fakes.controller(), head=fakes.head(quat=_head_quat(-40))))
    assert abs(t.get_action()["head_0.pos"]) <= math.radians(40) + 1e-6


def test_head_absolute_latches_look_direction_on_first_action_and_right_a(stubbed_pipeline, monkeypatch):
    """Right A: the look direction of that moment ↦ the start-pose head joints (like the EE)."""
    clock = [100.0]
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock[0])
    session = fakes.FakeSession()
    # Not looking straight ahead at the first action: yaw 20 deg, pitch -15 deg (down).
    session.push(_frame(right=fakes.controller(), head=fakes.head(quat=_head_quat(20, -15))))
    readers: list = []
    head_start = np.array([0.1, 0.85])
    t = make_teleop(session, readers, torso_source="none", use_torso=False, ready_return_duration_s=1.0,
                    head_pitch_offset_deg=10.0)  # the fixed offsets are irrelevant once latched
    t.connect()
    readers[0].head_q = head_start
    a = t.get_action()  # first action: latch (the operator frame is referenced from the gaze -> yaw 0)
    np.testing.assert_allclose([a["head_0.pos"], a["head_1.pos"]], head_start)
    a = t.get_action()  # same view -> still the start joints, not sign*pitch + offset
    np.testing.assert_allclose([a["head_0.pos"], a["head_1.pos"]], head_start, atol=1e-6)
    # Look 10 deg further up: pitch_sign=-1 -> head_1 decreases by 10 deg from the start value.
    session.push(_frame(right=fakes.controller(), head=fakes.head(quat=_head_quat(20, -5))))
    a = t.get_action()
    assert a["head_1.pos"] == pytest.approx(head_start[1] - math.radians(10), abs=1e-6)
    assert a["head_0.pos"] == pytest.approx(head_start[0], abs=1e-6)
    # Turn the head 30 deg left and pitch -25 deg, then press Right A there. The shoulder
    # line (body) defines the operator frame: facing +X, so the gaze yaw does not cancel out.
    body = _shoulders_body((0, -0.2), (0, 0.2))
    session.push(_frame(right=fakes.controller(primary=True), head=fakes.head(quat=_head_quat(30, -25)), body=body))
    t.get_action()
    clock[0] += 1.5  # return finished
    session.push(_frame(right=fakes.controller(), head=fakes.head(quat=_head_quat(30, -25)), body=body))
    a = t.get_action()
    a = t.get_action()
    np.testing.assert_allclose([a["head_0.pos"], a["head_1.pos"]], head_start, atol=1e-6)
    # From there, yaw +10 deg (left) -> head_0 = start + 10 deg (yaw_sign=+1).
    session.push(_frame(right=fakes.controller(), head=fakes.head(quat=_head_quat(40, -25)), body=body))
    a = t.get_action()
    assert a["head_0.pos"] == pytest.approx(head_start[0] + math.radians(10), abs=1e-6)


def test_head_latch_waits_for_tracked_head(stubbed_pipeline):
    session = fakes.FakeSession()
    session.push(_frame(right=fakes.controller()))  # no head yet
    readers: list = []
    t = make_teleop(session, readers, torso_source="none", use_torso=False)
    t.connect()
    readers[0].head_q = np.array([0.2, 0.7])
    a = t.get_action()
    np.testing.assert_allclose([a["head_0.pos"], a["head_1.pos"]], [0.2, 0.7])
    assert t._head_latch_target is not None  # pending
    session.push(_frame(right=fakes.controller(), head=fakes.head(quat=_head_quat(35, 10))))
    a = t.get_action()
    assert t._head_latch_target is None
    np.testing.assert_allclose([a["head_0.pos"], a["head_1.pos"]], [0.2, 0.7], atol=1e-6)


def test_camera_panel_grab_holds_gripper_and_blocks_clutch(stubbed_pipeline, monkeypatch, tmp_path):
    """A free hand pointing at the handle bar: trigger = grab (gripper held, no clutch), layout saved."""
    import time as _time

    from tests.test_viz_panels import FakeTeleviz

    frames = {"front": (np.zeros((480, 640, 3), np.uint8), 0.0, 1)}
    monkeypatch.setattr(mod, "bus_frame_source", lambda n: frames.get(n))
    layout_file = tmp_path / "viz_layout.json"
    # XR anchor frame: viewer at (0, 1.6, 0) (FakeVizSession head), handle bar centre at (0, 2.035, -1.5).
    M = np.asarray(Rby1XRConfig.__dataclass_fields__["base_T_anchor"].default_factory(), float)[:3, :3]

    def ctrl(target_xr, **kw):
        d = np.asarray(target_xr, float) - np.array([0.0, 1.6, 0.0])
        pitch = math.atan2(d[1], -d[2])
        yaw = math.atan2(-d[0], -d[2])
        R_xr = Rotation.from_euler("y", yaw) * Rotation.from_euler("x", pitch)  # -Z -> d
        R_robot = Rotation.from_matrix(M @ R_xr.as_matrix())
        return fakes.controller(pos=M @ np.array([0.0, 1.6, 0.0]), quat=R_robot.as_quat(), **kw)

    session = fakes.FakeSession()
    session.push(_frame(right=ctrl((0, 2.035, -1.5))))
    readers: list = []
    cfg = dict(torso_source="none", use_torso=False, use_head=False, viz_enabled=True, viz_cameras=["front"],
               viz_offsets_x=[0.0], viz_frame_bridge="identity", viz_layout_file=str(layout_file))

    def reader_factory(address, model):
        readers.append(fakes.FakeStateReader(address, model))
        return readers[-1]

    t = mod.Rby1XR(Rby1XRConfig(auto_launch_cloudxr=False, head_smoothing=1.0, **cfg),
                   session_factory=lambda pipeline: session, state_reader_factory=reader_factory,
                   viz_module=FakeTeleviz, viz_uploader=lambda f: f)
    t.connect()
    try:
        a = t.get_action()
        assert a["right_gripper_0.pos"] == 1.0

        def wait(cond):
            deadline = _time.time() + 2.0
            while _time.time() < deadline and not cond():
                t.get_action()
                _time.sleep(0.01)
            return cond()

        assert wait(lambda: t._viz.hover()["right"] == "front")
        # Trigger while hovering: grab, gripper stays open.
        session.push(_frame(right=ctrl((0, 2.035, -1.5), trigger=0.9)))
        a = t.get_action()
        assert a["right_gripper_0.pos"] == 1.0
        assert wait(lambda: t._viz.grabbing()["right"] == "front")
        # Squeezing while grabbing does not clutch the arm; the gripper is still held.
        session.push(_frame(right=ctrl((0.3, 2.035, -1.5), trigger=0.9, squeeze=0.9)))
        a = t.get_action()
        assert not t._clutch["right"].engaged and a["right_gripper_0.pos"] == 1.0
        assert wait(lambda: abs(t._viz.layout("front").offset_x - 0.3) < 0.03)
        # Release: layout persisted.
        session.push(_frame(right=ctrl((0.3, 2.035, -1.5), trigger=0.0)))
        assert wait(lambda: t._viz.grabbing()["right"] is None and layout_file.exists())
        import json

        assert abs(json.loads(layout_file.read_text())["front"]["offset_x"] - 0.3) < 0.03
        # Gripper works again once the ray leaves the bar.
        session.push(_frame(right=ctrl((0, 0.5, -1.5), trigger=1.0)))
        assert wait(lambda: t._viz.hover()["right"] is None)
        assert t.get_action()["right_gripper_0.pos"] == 0.0
    finally:
        t.disconnect()
    # A new teleop restores the saved layout.
    session2 = fakes.FakeSession()
    session2.push(_frame(right=fakes.controller()))
    t2 = mod.Rby1XR(Rby1XRConfig(auto_launch_cloudxr=False, head_smoothing=1.0, **cfg),
                    session_factory=lambda pipeline: session2, state_reader_factory=reader_factory,
                    viz_module=FakeTeleviz, viz_uploader=lambda f: f)
    t2.connect()
    try:
        assert abs(t2._viz.layout("front").offset_x - 0.3) < 0.03
    finally:
        t2.disconnect()


def test_head_gaze_fixed_in_base_frame_when_torso_moves(stubbed_pipeline):
    session = fakes.FakeSession()
    session.push(_frame(right=fakes.controller(), head=fakes.head(quat=_head_quat(0))))
    readers: list = []
    t = make_teleop(session, readers, torso_source="none", use_torso=False)
    t.connect()
    readers[0].head_q = np.array([0.0, 0.85])
    t.get_action()  # latch
    a = t.get_action()
    np.testing.assert_allclose([a["head_0.pos"], a["head_1.pos"]], [0.0, 0.85], atol=1e-6)
    # The torso turns 25 deg left (measured) while the headset stays still.
    readers[0].torso = fakes.se3((0, 0, 1.0), Rotation.from_euler("z", 25, degrees=True).as_matrix())
    session.push(_frame(right=fakes.controller(), head=fakes.head(quat=_head_quat(0))))
    a = t.get_action()
    assert a["head_0.pos"] == pytest.approx(math.radians(-25), abs=1e-6)
    assert a["head_1.pos"] == pytest.approx(0.85, abs=1e-6)
    # Headset lost: the held gaze is still compensated for a further torso motion.
    readers[0].torso = fakes.se3((0, 0, 1.0), Rotation.from_euler("z", 40, degrees=True).as_matrix())
    session.push(_frame(right=fakes.controller()))
    a = t.get_action()
    assert a["head_0.pos"] == pytest.approx(math.radians(-40), abs=1e-6)
    # Legacy frame: joints do not react to the torso.
    session2 = fakes.FakeSession()
    session2.push(_frame(right=fakes.controller(), head=fakes.head(quat=_head_quat(0))))
    readers2: list = []
    t2 = make_teleop(session2, readers2, torso_source="none", use_torso=False, head_gaze_frame="torso")
    t2.connect()
    readers2[0].head_q = np.array([0.0, 0.85])
    t2.get_action()
    readers2[0].torso = fakes.se3((0, 0, 1.0), Rotation.from_euler("z", 25, degrees=True).as_matrix())
    session2.push(_frame(right=fakes.controller(), head=fakes.head(quat=_head_quat(0))))
    a = t2.get_action()
    assert a["head_0.pos"] == pytest.approx(0.0, abs=1e-6)


def test_neck_mode_head_holds_gaze_against_torso(stubbed_pipeline):
    session = fakes.FakeSession()
    session.push(_frame(right=fakes.controller(), head=fakes.head(quat=_head_quat(0))))
    readers: list = []
    t = make_teleop(session, readers, torso_source="none", use_torso=False, wear_mode="neck")
    t.connect()
    readers[0].head_q = np.array([0.1, 0.85])
    a = t.get_action()
    np.testing.assert_allclose([a["head_0.pos"], a["head_1.pos"]], [0.1, 0.85])
    session.push(_frame(right=fakes.controller(), head=fakes.head(quat=_head_quat(40))))  # headset turns: ignored
    readers[0].torso = fakes.se3((0, 0, 1.0), Rotation.from_euler("z", -15, degrees=True).as_matrix())
    a = t.get_action()
    assert a["head_0.pos"] == pytest.approx(0.1 + math.radians(15), abs=1e-6)


def test_reference_shoulders_waits_for_body_then_uses_shoulder_line(stubbed_pipeline, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock[0])
    session = fakes.FakeSession()
    # Head looks 60 deg left, no body yet: with reference_source="shoulders" the frame is not referenced.
    session.push(_frame(right=fakes.controller(_abs_hand(), squeeze=0.9), head=fakes.head(quat=_head_quat(60))))
    readers: list = []
    t = make_teleop(session, readers, torso_source="none", use_torso=False, use_head=False, use_left_arm=False,
                    reference_source="shoulders", **ABS_KW)
    t.connect()
    a = t.get_action()
    assert t._needs_reference and t._yaw_correction == 0.0
    assert a["right_ee.x"] == pytest.approx(readers[0].right_ee[0, 3])
    assert t._abs_state["right"] == "hold (waiting for shoulders)"
    # Shoulders appear (facing +X while the head still looks 60 deg left): shoulder line wins.
    body = _shoulders_body((0, -0.2), (0, 0.2))
    session.push(_frame(right=fakes.controller(_abs_hand(), squeeze=0.9), head=fakes.head(quat=_head_quat(60)), body=body))
    t.get_action()
    assert not t._needs_reference and t._yaw_correction == pytest.approx(0.0)
    # Right A without shoulders falls back to the gaze (with a warning) instead of waiting.
    session.push(_frame(right=fakes.controller(_abs_hand(), primary=True), head=fakes.head(quat=_head_quat(60))))
    t.get_action()
    assert t._yaw_correction == pytest.approx(-math.radians(60))


def test_ee_offsets_latched_at_first_engage_not_at_button(stubbed_pipeline, monkeypatch):
    """Squeezing without moving the hand leaves the arm at the start pose; the
    hand posture used to press Right A does not enter the mapping."""
    clock = [100.0]
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock[0])
    body = _abs_body()
    q_tilt = Rotation.from_euler("x", 40, degrees=True).as_quat()
    session = fakes.FakeSession()
    session.push(_frame(right=fakes.controller(_abs_hand(0.2, 0.1, -0.1), quat=q_tilt), body=body))
    readers: list = []
    r_start = fakes.se3((0.4, -0.2, 0.9), Rotation.from_euler("z", 30, degrees=True).as_matrix())
    t = make_teleop(session, readers, torso_source="none", use_torso=False, use_head=False, use_left_arm=False,
                    ready_return_duration_s=0.5, ee_position_latch_on_a=True, **ABS_KW)
    t.connect()
    readers[0].right_ee = r_start.copy()
    t.get_action()  # start pose recorded
    # First squeeze with the hand somewhere and the controller tilted: offsets latch -> target == start pose.
    session.push(_frame(right=fakes.controller(_abs_hand(0.2, 0.1, -0.1), quat=q_tilt, squeeze=0.9), body=body))
    a = t.get_action()
    clock[0] += 0.1
    session.push(_frame(right=fakes.controller(_abs_hand(0.2, 0.1, -0.1), quat=q_tilt, squeeze=0.9), body=body))
    a = t.get_action()
    np.testing.assert_allclose([a["right_ee.x"], a["right_ee.y"], a["right_ee.z"]], r_start[:3, 3], atol=1e-9)
    np.testing.assert_allclose([a["right_ee.wx"], a["right_ee.wy"], a["right_ee.wz"]],
                               Rotation.from_matrix(r_start[:3, :3]).as_rotvec(), atol=1e-6)
    # Move the hand 0.1 m forward and yaw the controller 20 deg: start + k*0.1, orientation + 20 deg.
    k = mod.robot_reach("1.3") / 0.6
    q2 = (Rotation.from_euler("z", 20, degrees=True) * Rotation.from_quat(q_tilt)).as_quat()
    session.push(_frame(right=fakes.controller(_abs_hand(0.3, 0.1, -0.1), quat=q2, squeeze=0.9), body=body))
    a = t.get_action()
    assert a["right_ee.x"] == pytest.approx(r_start[0, 3] + 0.1 * k)
    R_out = Rotation.from_rotvec([a["right_ee.wx"], a["right_ee.wy"], a["right_ee.wz"]]).as_matrix()
    np.testing.assert_allclose(R_out, Rotation.from_euler("z", 50, degrees=True).as_matrix(), atol=1e-6)
    # Right A pressed with a weird posture (hand far away, controller upside down) -> return.
    q_weird = Rotation.from_euler("y", 170, degrees=True).as_quat()
    session.push(_frame(right=fakes.controller(_abs_hand(0.5, -0.3, 0.3), quat=q_weird, squeeze=0.9, primary=True), body=body))
    t.get_action()
    session.push(_frame(right=fakes.controller(_abs_hand(0.5, -0.3, 0.3), quat=q_weird, squeeze=0.0), body=body))  # let go
    clock[0] += 1.0
    t.get_action()  # return finished, nothing engaged -> offsets still pending
    assert t._needs_offset_latch["right"]
    # Squeeze again from yet another posture: the arm stays at the start pose (offsets from THIS posture).
    q3 = Rotation.from_euler("y", -30, degrees=True).as_quat()
    session.push(_frame(right=fakes.controller(_abs_hand(-0.1, 0.2, 0.0), quat=q3, squeeze=0.9), body=body))
    t.get_action()
    clock[0] += 0.1
    session.push(_frame(right=fakes.controller(_abs_hand(-0.1, 0.2, 0.0), quat=q3, squeeze=0.9), body=body))
    a = t.get_action()
    np.testing.assert_allclose([a["right_ee.x"], a["right_ee.y"], a["right_ee.z"]], r_start[:3, 3], atol=1e-9)
    np.testing.assert_allclose([a["right_ee.wx"], a["right_ee.wy"], a["right_ee.wz"]],
                               Rotation.from_matrix(r_start[:3, :3]).as_rotvec(), atol=1e-6)
    assert not t._needs_offset_latch["right"]


def test_right_a_joint_return_emits_reset_cmd_and_holds_start_pose(stubbed_pipeline, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock[0])
    body = _abs_body()
    session = fakes.FakeSession()
    session.push(_frame(right=fakes.controller(_abs_hand(), squeeze=0.9, thumb=(0, 1.0)), head=fakes.head(quat=_head_quat(0)), body=body))
    readers: list = []
    t = make_teleop(session, readers, torso_source="none", ready_return_duration_s=3.0, ready_return_mode="joint", **ABS_KW)
    t.connect()
    a0 = t.get_action()  # start pose recorded; dead-man engaged (ramp)
    assert "reset.cmd" not in a0 and "reset.cmd" not in t.action_features
    x_start, head_start = a0["right_ee.x"], a0["head_1.pos"]
    clock[0] += 0.1
    session.push(_frame(right=fakes.controller(_abs_hand(0.3), squeeze=0.9, thumb=(0, 1.0)), head=fakes.head(quat=_head_quat(30)), body=body))
    a = t.get_action()
    assert a["right_ee.x"] != pytest.approx(x_start) and a["x.vel"] == pytest.approx(0.3)
    # Right A: one reset.cmd tick, every target already at the start pose, base zeroed.
    session.push(_frame(right=fakes.controller(_abs_hand(0.3), squeeze=0.9, thumb=(0, 1.0), primary=True), head=fakes.head(quat=_head_quat(0)), body=body))
    a = t.get_action()
    assert a["reset.cmd"] == pytest.approx(3.0)
    assert a["right_ee.x"] == pytest.approx(x_start) and a["head_1.pos"] == pytest.approx(head_start)
    assert a["x.vel"] == 0.0 and t._returning
    # Next tick (the follower blocked for the motion): finished, no reset key, targets still at start.
    clock[0] += 3.0
    session.push(_frame(right=fakes.controller(_abs_hand(0.3), squeeze=0.0, thumb=(0, 1.0)), head=fakes.head(quat=_head_quat(0)), body=body))
    a = t.get_action()
    assert "reset.cmd" not in a and not t._returning
    assert a["right_ee.x"] == pytest.approx(x_start) and a["x.vel"] == pytest.approx(0.3)
