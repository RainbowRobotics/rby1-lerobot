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
    t = make_teleop(session, readers, torso_source="none", use_left_arm=False)
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
    a = t.get_action()  # Right A: resume -> re-engage at the measured pose, no jump
    assert not t.is_stopped and a["right_ee.x"] == pytest.approx(x0)
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
    t = make_teleop(session, readers, torso_source="body", torso_max_z_delta_m=0.15)
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
    pos3 = pos2.copy()
    pos3[BodyJointIndex.SPINE3] = [0.0, 0.0, 1.2]
    session.push(_frame(right=fakes.controller(squeeze=0.9), left=fakes.controller(squeeze=0.0), body=fakes.body(positions=pos3, orientations=quat2)))
    a = t.get_action()
    assert a["torso_ee.z"] == pytest.approx(z0 - 0.15)

    # Invalid body -> torso holds; valid again -> re-latches without a jump.
    valid = np.ones(24, np.uint8)
    valid[BodyJointIndex.PELVIS] = 0
    session.push(_frame(right=fakes.controller(squeeze=0.9), left=fakes.controller(squeeze=0.9), body=fakes.body(positions=pos3, orientations=quat2, valid=valid)))
    a = t.get_action()
    assert a["torso_ee.z"] == pytest.approx(z0 - 0.15)
    readers[0].torso[2, 3] = z0 - 0.15
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
