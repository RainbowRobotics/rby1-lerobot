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


def _head_quat(yaw_deg=0.0):
    R = R_HEAD_FWD @ Rotation.from_euler("y", yaw_deg, degrees=True).as_matrix()
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


def test_clutch_moves_holds_and_gripper(stubbed_pipeline):
    session = fakes.FakeSession()
    session.push(_frame(right=fakes.controller((1, 1, 1))))
    readers: list = []
    t = make_teleop(session, readers, torso_source="none", use_torso=False, use_left_arm=False)
    t.connect()
    t.get_action()
    x0 = readers[0].right_ee[0, 3]

    session.push(_frame(right=fakes.controller((1, 1, 1), squeeze=0.9)))  # engage edge
    a = t.get_action()
    assert a["right_ee.x"] == pytest.approx(x0)  # engage frame: no jump
    session.push(_frame(right=fakes.controller((1.1, 1.0, 1.05), squeeze=0.9, trigger=0.75)))
    a = t.get_action()
    assert a["right_ee.x"] == pytest.approx(x0 + 0.1)
    assert a["right_ee.z"] == pytest.approx(readers[0].right_ee[2, 3] + 0.05)
    assert a["right_gripper_0.pos"] == pytest.approx(0.25)

    session.push(_frame(right=fakes.controller((2, 2, 2), squeeze=0.1)))  # released: hold
    a = t.get_action()
    assert a["right_ee.x"] == pytest.approx(x0 + 0.1)
    assert "left_ee.x" not in a and "torso_ee.x" not in a

    # Re-engage after the robot "sagged": home comes from the measured pose.
    readers[0].right_ee[2, 3] -= 0.03
    session.push(_frame(right=fakes.controller((2, 2, 2), squeeze=0.9)))
    a = t.get_action()
    assert a["right_ee.z"] == pytest.approx(readers[0].right_ee[2, 3])


def test_stop_and_resume(stubbed_pipeline):
    session = fakes.FakeSession()
    session.push(_frame(right=fakes.controller((0, 0, 0), squeeze=0.9, thumb=(0, 1.0))))
    readers: list = []
    t = make_teleop(session, readers, torso_source="none", use_left_arm=False, ready_return_duration_s=0.1)
    t.connect()
    a = t.get_action()
    assert a["x.vel"] == pytest.approx(0.3)
    x0 = a["right_ee.x"]

    session.push(_frame(right=fakes.controller((0.2, 0, 0), squeeze=0.9, thumb=(0, 1.0), secondary=True)))
    a = t.get_action()  # Right B: stop
    assert t.is_stopped and a["x.vel"] == 0.0 and a["right_ee.x"] == pytest.approx(x0)
    session.push(_frame(right=fakes.controller((0.5, 0, 0), squeeze=0.9, thumb=(0, 1.0))))
    a = t.get_action()  # still stopped: nothing moves
    assert a["right_ee.x"] == pytest.approx(x0) and a["x.vel"] == 0.0

    session.push(_frame(right=fakes.controller((0.5, 0, 0), squeeze=0.9, primary=True)))
    a = t.get_action()  # Right A: resume + return-to-start motion (already there)
    assert not t.is_stopped and a["right_ee.x"] == pytest.approx(x0)
    import time as _time

    _time.sleep(0.15)  # let the (0.1 s) return motion finish
    session.push(_frame(right=fakes.controller((0.5, 0, 0), squeeze=0.9)))
    a = t.get_action()  # motion done -> re-engage at the measured pose, no jump
    assert a["right_ee.x"] == pytest.approx(x0)
    session.push(_frame(right=fakes.controller((0.6, 0, 0), squeeze=0.9)))
    assert t.get_action()["right_ee.x"] == pytest.approx(x0 + 0.1)


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
        Rby1XRConfig(latch_orientation="both")
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


def test_disengaged_components_follow_robot_after_reset(stubbed_pipeline):
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
    session.push(_frame(right=fakes.controller(squeeze=0.9)))
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
    session = fakes.FakeSession()
    session.push(_frame(right=fakes.controller((0, 0, 0), squeeze=0.9), head=fakes.head(quat=_head_quat(0))))
    readers: list = []
    t = make_teleop(session, readers, torso_source="none", ready_return_duration_s=2.0)
    t.connect()
    a0 = t.get_action()  # start pose recorded here (first action)
    x_start, head_start = a0["right_ee.x"], a0["head_1.pos"]
    # Move the right arm 30 cm and the head by yawing.
    session.push(_frame(right=fakes.controller((0.3, 0, 0), squeeze=0.9), head=fakes.head(quat=_head_quat(30))))
    a = t.get_action()
    assert a["right_ee.x"] == pytest.approx(x_start + 0.3)
    assert a["head_0.pos"] == pytest.approx(math.radians(30))
    # Right A: the clutch is released and the targets interpolate back. (The
    # operator faces forward again here so the yaw reference stays unchanged.)
    session.push(_frame(right=fakes.controller((0.3, 0, 0), squeeze=0.9, primary=True), head=fakes.head(quat=_head_quat(0))))
    a = t.get_action()
    assert a["right_ee.x"] == pytest.approx(x_start + 0.3)  # alpha = 0
    clock[0] += 1.0  # halfway (smoothstep(0.5) = 0.5)
    session.push(_frame(right=fakes.controller((0.9, 0, 0), squeeze=0.9), head=fakes.head(quat=_head_quat(30))))
    a = t.get_action()
    assert a["right_ee.x"] == pytest.approx(x_start + 0.15)  # squeeze is ignored while returning
    assert a["head_0.pos"] == pytest.approx(math.radians(15))
    clock[0] += 1.5
    a = t.get_action()
    assert a["right_ee.x"] == pytest.approx(x_start)
    assert a["head_1.pos"] == pytest.approx(head_start) and a["head_0.pos"] == pytest.approx(0.0)
    # After arrival a squeeze re-engages from the measured pose (no jump).
    readers[0].right_ee[0, 3] = x_start
    session.push(_frame(right=fakes.controller((0.9, 0, 0), squeeze=0.9)))
    t.get_action()
    session.push(_frame(right=fakes.controller((1.0, 0, 0), squeeze=0.9)))
    assert t.get_action()["right_ee.x"] == pytest.approx(x_start + 0.1)


def test_right_a_rereferences_operator_yaw(stubbed_pipeline, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock[0])
    session = fakes.FakeSession()
    # Operator initially faces robot +X (head looks along anchor -Z -> robot +X).
    session.push(_frame(right=fakes.controller((0, 0, 0)), head=fakes.head(quat=_head_quat(0))))
    readers: list = []
    t = make_teleop(session, readers, torso_source="none", use_head=False, ready_return_duration_s=0.5)
    t.connect()
    t.get_action()
    x0, y0 = readers[0].right_ee[0, 3], readers[0].right_ee[1, 3]
    # The operator turns 90 deg to the left (now facing robot +Y) and presses A.
    session.push(_frame(right=fakes.controller((0, 0, 0), primary=True), head=fakes.head(quat=_head_quat(90))))
    t.get_action()
    assert t._yaw_correction == pytest.approx(-math.pi / 2)
    clock[0] += 1.0  # return motion done
    t.get_action()
    # Squeeze, then push the controller "forward" for the operator = raw robot +Y.
    session.push(_frame(right=fakes.controller((0, 0, 0), squeeze=0.9), head=fakes.head(quat=_head_quat(90))))
    t.get_action()
    session.push(_frame(right=fakes.controller((0, 0.2, 0), squeeze=0.9), head=fakes.head(quat=_head_quat(90))))
    a = t.get_action()
    assert a["right_ee.x"] == pytest.approx(x0 + 0.2)  # forward for the operator -> robot +X
    assert a["right_ee.y"] == pytest.approx(y0)


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
        session, readers, arm_mode="ee_absolute", torso_source="none", use_torso=False,
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
    t = make_teleop(session, readers, arm_mode="ee_absolute", torso_source="none", use_torso=False, use_left_arm=False, use_head=False)
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
    session.push(_frame(right=fakes.controller(W), body=body))
    readers: list = []
    t = make_teleop(session, readers, torso_source="none", use_torso=False, use_left_arm=False, use_head=False,
                    arm_posture_hint=True, posture_hint_smoothing=1.0)
    t.connect()
    readers[0].torso = T
    a = t.get_action()
    for i in range(4):
        assert a[f"right_arm_{i}.null"] == pytest.approx(q_true[i], abs=1e-5)
    assert "right_arm_0.null" not in t.action_features  # not recorded by default
    # Recording flag exposes the keys as features.
    t2 = make_teleop(fakes.FakeSession(), [], torso_source="none", use_torso=False, use_left_arm=False, use_head=False,
                     arm_posture_hint=True, record_posture_hint=True)
    assert "right_arm_3.null" in t2.action_features and "left_arm_0.null" not in t2.action_features
    # No body -> no hint keys (unless recorded).
    session.push(_frame(right=fakes.controller(W)))
    clock[0] += 5.0
    a = t.get_action()
    assert "right_arm_0.null" not in a
