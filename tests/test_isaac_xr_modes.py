"""Rby1XR modes: start on Right A, arm reach calibration, body drive, grip-gated head and base."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import isaac_fakes as fakes
from isaac_xr_helpers import (
    ABS_KW,
    ABS_SHOULDER,
    _abs_body,
    _abs_hand,
    _frame,
    _head_quat,
    make_teleop,
)
from lerobot_teleoperator_rby1.isaac_teleop import teleop_rby1_xr as mod
from lerobot_teleoperator_rby1.isaac_teleop.config_isaac_teleop import Rby1XRConfig
from lerobot_teleoperator_rby1.isaac_teleop.xr_frame import (
    OUT_CONTROLLER_RIGHT,
    BodyJointIndex,
)


def test_nothing_follows_until_the_first_right_a(stubbed_pipeline, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock[0])
    body = _abs_body()
    head = fakes.head((0.0, 0.0, 1.6))
    session = fakes.FakeSession()
    session.push(_frame(right=fakes.controller(_abs_hand(0.3)), head=head, body=body))
    readers: list = []
    t = make_teleop(session, readers, torso_source="head", torso_engage="always", use_left_arm=False,
                    start_on_right_a=True, **ABS_KW)
    t.connect()
    a0 = t.get_action()
    r = readers[0]
    assert not t.is_started

    # Squeeze, trigger, thumbstick, a turned and moved head: everything is ignored.
    turned = fakes.head((0.2, 0.0, 1.5), Rotation.from_euler("z", 40, degrees=True).as_quat())
    busy = fakes.controller(_abs_hand(0.5, 0.0, 0.1), squeeze=0.9, trigger=1.0, thumb=(0.0, 1.0))
    for _ in range(3):
        clock[0] += 0.1
        session.push(_frame(right=busy, head=turned, body=body))
        a = t.get_action()
        assert a == a0
    assert a["right_gripper_0.pos"] == 1.0 and a["x.vel"] == 0.0
    assert a["right_ee.x"] == pytest.approx(r.right_ee[0, 3]) and a["torso_ee.z"] == pytest.approx(r.torso[2, 3])
    np.testing.assert_allclose([a["head_0.pos"], a["head_1.pos"]], r.head_q)
    assert not t._core._clutch["right"].engaged and not t._core._clutch["torso"].engaged

    # First Right A: starts, but does not run the return-to-start motion.
    clock[0] += 0.1
    session.push(_frame(right=fakes.controller(_abs_hand(0.3), primary=True), head=turned, body=body))
    a = t.get_action()
    assert t.is_started and "reset.cmd" not in a and not t._core._returning
    assert a["right_ee.x"] == pytest.approx(a0["right_ee.x"])
    np.testing.assert_allclose([a["head_0.pos"], a["head_1.pos"]], r.head_q, atol=1e-6)  # no head jump

    # Now the trigger drives the gripper and a squeeze engages the arm.
    clock[0] += 0.1
    session.push(_frame(right=fakes.controller(_abs_hand(0.3), squeeze=0.9, trigger=0.75), head=turned, body=body))
    a = t.get_action()
    assert a["right_gripper_0.pos"] == pytest.approx(0.25)
    assert t._core._clutch["right"].engaged and t._core._clutch["torso"].engaged
    np.testing.assert_allclose([a["head_0.pos"], a["head_1.pos"]], r.head_q, atol=1e-6)  # same look direction

    # A later Right A is the normal return-to-start.
    clock[0] += 0.1
    session.push(_frame(right=fakes.controller(_abs_hand(0.3)), head=turned, body=body))
    t.get_action()
    clock[0] += 0.1
    session.push(_frame(right=fakes.controller(_abs_hand(0.3), primary=True), head=turned, body=body))
    assert "reset.cmd" in t.get_action()


# ── Arm reach calibration on Right A ────────────────────────────────────────
CAL_KW = {**ABS_KW, "arm_length_source": "calibrate", "start_on_right_a": True, "torso_source": "none",
          "use_torso": False, "use_head": False}
CAL_LEFT_SHOULDER = ABS_SHOULDER * np.array([1, -1, 1])


def _cal_body():
    """Both shoulders valid (elbow / wrist joints are irrelevant for the calibration)."""
    pos = np.zeros((24, 3), np.float32)
    pos[int(BodyJointIndex.RIGHT_SHOULDER)] = ABS_SHOULDER
    pos[int(BodyJointIndex.LEFT_SHOULDER)] = CAL_LEFT_SHOULDER
    return fakes.body(positions=pos)


def _cal_frame(right=None, left=None, primary=False, squeeze=0.0):
    """Controllers ``right`` / ``left`` metres straight in front of their shoulders."""
    def ctrl(shoulder, reach, a=False):
        return None if reach is None else fakes.controller(
            tuple((shoulder + [reach, 0.0, 0.0]).tolist()), primary=a, squeeze=squeeze)
    return _frame(right=ctrl(ABS_SHOULDER, right, primary), left=ctrl(CAL_LEFT_SHOULDER, left), body=_cal_body())


def _cal_hold(t, session, clock, n=8, **reach):
    for _ in range(n):
        clock[0] += 0.05
        session.push(_cal_frame(**reach))
        t.get_action()


def _cal_press(t, session, clock, **reach):
    clock[0] += 0.05
    session.push(_cal_frame(primary=True, **reach))
    return t.get_action()


def _cal_teleop(monkeypatch, **cfg):
    clock = [100.0]
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock[0])
    session = fakes.FakeSession()
    session.push(_cal_frame(right=0.3, left=0.3))
    readers: list = []
    t = make_teleop(session, readers, **{**CAL_KW, **cfg})
    t.connect()
    t.get_action()
    return t, session, clock, readers[0]


def test_right_a_calibrates_the_arm_reach_and_starts(stubbed_pipeline, monkeypatch, caplog):
    t, session, clock, reader = _cal_teleop(monkeypatch)
    assert t._core._abs["right"].robot_reach == reader.reach["right"]  # shoulder -> link_arm_6 from the robot
    assert t._core._abs["right"].calibrated_reach is None and "reach=uncalibrated" in t._core._reach_status()

    _cal_hold(t, session, clock, right=0.62, left=0.60)
    with caplog.at_level("INFO", logger="rby1_xr_teleop"):
        _cal_press(t, session, clock, right=0.62, left=0.60)
    assert t.is_started and "Arm reach calibrated" in caplog.text
    assert t._core._abs["right"].human_reach == pytest.approx(0.62) and t._core._abs["left"].human_reach == pytest.approx(0.60)
    assert t._core._reach_status() == " reach=R0.62/L0.60 (x1.06/1.10)"

    # Fixed for the session: moving the hand maps with that scale, whatever the body joints say.
    scale = reader.reach["right"] / 0.62
    hand = fakes.controller(tuple((ABS_SHOULDER + [0.3, 0.0, 0.0]).tolist()), squeeze=0.9)
    clock[0] += 0.05
    session.push(_frame(right=hand, body=_abs_body()))  # elbow / wrist joints present now: ignored
    t.get_action()
    clock[0] += 0.2  # engage ramp done
    session.push(_frame(right=fakes.controller(tuple((ABS_SHOULDER + [0.4, 0.0, 0.0]).tolist()), squeeze=0.9),
                        body=_abs_body()))
    a = t.get_action()
    assert t._core._abs["right"].human_reach == pytest.approx(0.62)
    # Absolute mapping (no position latch in these fixtures): 0.4 m in front of the
    # operator's shoulder is 0.4 * scale in front of the robot's.
    from lerobot_teleoperator_rby1.isaac_teleop.arm_retargeter import robot_shoulder_position

    assert a["right_ee.x"] == pytest.approx(robot_shoulder_position(reader.torso, "right")[0] + 0.4 * scale)


def test_first_right_a_without_a_usable_reach_does_not_start(stubbed_pipeline, monkeypatch, caplog):
    t, session, clock, _ = _cal_teleop(monkeypatch)
    # Arms not extended (0.30 m is below the accepted range).
    _cal_hold(t, session, clock, right=0.30, left=0.30)
    with caplog.at_level("WARNING", logger="rby1_xr_teleop"):
        _cal_press(t, session, clock, right=0.30, left=0.30)
    assert not t.is_started and t._core._abs["right"].calibrated_reach is None
    assert "Hold both arms straight out" in caplog.text and "0.300 m" in caplog.text

    # Too few samples in the window (the press came right after extending the arms).
    clock[0] += 1.0
    session.push(_cal_frame(right=0.6, left=0.6))
    t.get_action()
    _cal_press(t, session, clock, right=0.6, left=0.6)
    assert not t.is_started

    # Held long enough: accepted.
    _cal_hold(t, session, clock, right=0.6, left=0.6)
    _cal_press(t, session, clock, right=0.6, left=0.6)
    assert t.is_started and t._core._abs["left"].human_reach == pytest.approx(0.6)


def test_arm_reach_asymmetry_and_single_arm(stubbed_pipeline, monkeypatch, caplog):
    # One arm clearly shorter (not fully extended): the longer value is used for both.
    t, session, clock, _ = _cal_teleop(monkeypatch)
    _cal_hold(t, session, clock, right=0.64, left=0.50)
    with caplog.at_level("WARNING", logger="rby1_xr_teleop"):
        _cal_press(t, session, clock, right=0.64, left=0.50)
    assert t.is_started and "differs between the arms" in caplog.text
    assert t._core._abs["right"].human_reach == pytest.approx(0.64) and t._core._abs["left"].human_reach == pytest.approx(0.64)

    # Only one arm usable (the other controller is not tracked): its value is shared.
    t, session, clock, _ = _cal_teleop(monkeypatch)
    _cal_hold(t, session, clock, right=0.58, left=None)
    _cal_press(t, session, clock, right=0.58, left=None)
    assert t.is_started
    assert t._core._abs["right"].human_reach == pytest.approx(0.58) and t._core._abs["left"].human_reach == pytest.approx(0.58)


def test_later_right_a_recalibrates_only_with_extended_arms(stubbed_pipeline, monkeypatch, caplog):
    t, session, clock, _ = _cal_teleop(monkeypatch)
    _cal_hold(t, session, clock, right=0.60, left=0.60)
    _cal_press(t, session, clock, right=0.60, left=0.60)
    assert t.is_started

    # Right A with the arms hanging / bent: the return-to-start still runs, the reach is kept.
    _cal_hold(t, session, clock, right=0.50, left=0.50)
    with caplog.at_level("INFO", logger="rby1_xr_teleop"):
        a = _cal_press(t, session, clock, right=0.50, left=0.50)
    assert "reset.cmd" in a and "Arm reach kept" in caplog.text
    assert t._core._abs["right"].human_reach == pytest.approx(0.60)
    t._core._return_motion = None  # skip the return motion bookkeeping

    # Out of range entirely: kept as well.
    _cal_hold(t, session, clock, right=0.20, left=0.20)
    assert "reset.cmd" in _cal_press(t, session, clock, right=0.20, left=0.20)
    assert t._core._abs["right"].human_reach == pytest.approx(0.60)
    t._core._return_motion = None

    # Extended again, a little longer: replaced, and the offsets are latched anew.
    t._core._needs_offset_latch = {"right": False, "left": False}
    _cal_hold(t, session, clock, right=0.63, left=0.62)
    _cal_press(t, session, clock, right=0.63, left=0.62)
    assert t._core._abs["right"].human_reach == pytest.approx(0.63) and t._core._abs["left"].human_reach == pytest.approx(0.62)
    assert t._core._needs_offset_latch == {"right": True, "left": True}


def test_arm_calibration_config_and_profile(stubbed_pipeline, monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="start_on_right_a"):
        Rby1XRConfig(arm_length_source="calibrate", start_on_right_a=False)
    with pytest.raises(ValueError, match="arm_calibration_min_m"):
        Rby1XRConfig(arm_calibration_min_m=0.9, arm_calibration_max_m=0.8)
    assert Rby1XRConfig().arm_length_source == "calibrate"

    import json

    path = tmp_path / "rby1_profile.json"
    t, session, clock, reader = _cal_teleop(monkeypatch, profile_out=str(path))
    assert "arm_calibration" not in json.loads(path.read_text())["teleop"]
    _cal_hold(t, session, clock, right=0.60, left=0.60)
    _cal_press(t, session, clock, right=0.60, left=0.60)
    teleop = json.loads(path.read_text())["teleop"]
    assert teleop["type"] == "rby1_isaac" and teleop["config"]["arm_length_source"] == "calibrate"
    right = teleop["arm_calibration"]["right"]
    assert right["operator_reach_m"] == pytest.approx(0.60) and right["robot_reach_m"] == reader.reach["right"]
    assert right["scale"] == pytest.approx(reader.reach["right"] / 0.60)


# ── Body drive: Left X held, the base follows the operator ───────────────────
DRIVE_KW = dict(torso_source="head", torso_engage="always", arm_length_source="config", human_arm_length_m=0.6,
                shoulder_smoothing=1.0, engage_ramp_s=0.05, ee_max_linear_vel=100.0, ee_max_angular_vel=1000.0,
                body_drive_kp_linear=1.0, body_drive_kp_angular=1.0, body_drive_smoothing=1.0,
                body_drive_deadband_m=0.0, body_drive_deadband_deg=0.0, head_latch_on_a=False)


def _operator(xy=(0.0, 0.0), body_deg=0.0, neck_deg=0.0, *, left_x=False, squeeze=0.0, shoulders=True, hand=None):
    """A frame of an operator standing at ``xy`` facing ``body_deg``, the head turned a further ``neck_deg``.

    The right hand is ``hand`` (default 0.3 m in front of the right shoulder) in the body frame.
    """
    Rz = Rotation.from_euler("z", body_deg, degrees=True).as_matrix()
    origin = np.array([xy[0], xy[1], 0.0])
    pos = np.zeros((24, 3), np.float32)
    valid = np.zeros(24, np.uint8)
    right_shoulder = origin + Rz @ [0.0, -0.2, 1.4]
    if shoulders:
        pos[int(BodyJointIndex.RIGHT_SHOULDER)] = right_shoulder
        pos[int(BodyJointIndex.LEFT_SHOULDER)] = origin + Rz @ [0.0, 0.2, 1.4]
        valid[[int(BodyJointIndex.RIGHT_SHOULDER), int(BodyJointIndex.LEFT_SHOULDER)]] = 1
    hand_pos = right_shoulder + Rz @ np.asarray([0.3, 0.0, 0.0] if hand is None else hand, dtype=float)
    return _frame(
        right=fakes.controller(tuple(hand_pos.tolist()), Rotation.from_matrix(Rz).as_quat(), squeeze=squeeze),
        left=fakes.controller(tuple((origin + Rz @ [0.3, 0.2, 1.0]).tolist()), primary=left_x),
        head=fakes.head(tuple((origin + [0.0, 0.0, 1.6]).tolist()), _head_quat(body_deg + neck_deg)),
        body=fakes.body(positions=pos, valid=valid),
    )


def _drive_teleop(monkeypatch, **cfg):
    clock = [100.0]
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock[0])
    session = fakes.FakeSession()
    session.push(_operator())
    readers: list = []
    t = make_teleop(session, readers, **{**DRIVE_KW, **cfg})
    t.connect()
    t.get_action()

    def step(**operator):
        clock[0] += 0.05
        session.push(_operator(**operator))
        return t.get_action()

    return t, readers[0], step


def _se2(x, y, yaw_deg):
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]])


def test_body_drive_follows_steps_and_turns(stubbed_pipeline, monkeypatch):
    t, reader, step = _drive_teleop(monkeypatch)
    reader.odometry = _se2(3.0, -1.0, 90.0)  # wherever the robot's odometry happens to be
    a0 = step()
    assert (a0["x.vel"], a0["y.vel"], a0["theta.vel"]) == (0.0, 0.0, 0.0)

    # Pressing Left X moves nothing by itself; the torso target is frozen from here on.
    a = step(left_x=True)
    assert (a["x.vel"], a["y.vel"], a["theta.vel"]) == (0.0, 0.0, 0.0)
    assert t._core._body_drive is not None and not t._core._clutch["torso"].engaged
    torso = {k: v for k, v in a.items() if k.startswith("torso_ee.")}

    # One step forward and a little to the left: the base is asked to follow.
    a = step(xy=(0.2, 0.1), left_x=True)
    assert (a["x.vel"], a["y.vel"]) == pytest.approx((0.2, 0.1)) and a["theta.vel"] == pytest.approx(0.0)
    assert {k: v for k, v in a.items() if k.startswith("torso_ee.")} == torso  # the head moved 0.2 m: ignored
    # The robot got there (odometry, in its own frame turned 90 deg): nothing left.
    reader.odometry = _se2(3.0, -1.0, 90.0) @ _se2(0.2, 0.1, 0.0)
    a = step(xy=(0.2, 0.1), left_x=True)
    assert (a["x.vel"], a["y.vel"], a["theta.vel"]) == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)

    # Turning the body 30 deg to the left turns the base; the robot head stays (same neck angle).
    head = (a["head_0.pos"], a["head_1.pos"])
    a = step(xy=(0.2, 0.1), body_deg=30.0, left_x=True)
    assert a["theta.vel"] == pytest.approx(math.radians(30.0))
    assert (a["head_0.pos"], a["head_1.pos"]) == pytest.approx(head, abs=1e-6)
    # Turning only the head pans the robot head and leaves the base command alone.
    b = step(xy=(0.2, 0.1), body_deg=30.0, neck_deg=20.0, left_x=True)
    assert b["theta.vel"] == pytest.approx(math.radians(30.0))
    assert b["head_0.pos"] == pytest.approx(head[0] + math.radians(20.0), abs=1e-6)

    # The base turned 30 deg: done. Releasing Left X stops and nothing jumps.
    reader.odometry = _se2(3.0, -1.0, 90.0) @ _se2(0.2, 0.1, 30.0)
    a = step(xy=(0.2, 0.1), body_deg=30.0, left_x=True)
    assert (a["x.vel"], a["y.vel"], a["theta.vel"]) == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)
    a = step(xy=(0.2, 0.1), body_deg=30.0)
    assert t._core._body_drive is None and (a["x.vel"], a["y.vel"], a["theta.vel"]) == (0.0, 0.0, 0.0)
    assert (a["head_0.pos"], a["head_1.pos"]) == pytest.approx(head, abs=1e-6)
    assert t._core._frame.yaw == pytest.approx(math.radians(-30.0))  # the operator frame stays with the body


def test_body_drive_keeps_a_clutched_arm_where_it_is(stubbed_pipeline, monkeypatch):
    t, reader, step = _drive_teleop(monkeypatch)
    step(squeeze=0.9)  # engage
    for _ in range(3):
        a = step(squeeze=0.9)
    arm = {k: v for k, v in a.items() if k.startswith("right_ee.")}
    step(squeeze=0.9, left_x=True)
    # Walk and turn with the arm held still relative to the body: the arm target does not move.
    for xy, deg in (((0.05, 0.0), 10.0), ((0.1, 0.05), 25.0), ((0.1, 0.05), 40.0)):
        a = step(xy=xy, body_deg=deg, squeeze=0.9, left_x=True)
        assert {k: v for k, v in a.items() if k.startswith("right_ee.")} == pytest.approx(arm, abs=1e-6)
    assert t._core._clutch["right"].engaged and a["theta.vel"] > 0.0


def test_body_drive_stops_on_lost_tracking_and_large_error(stubbed_pipeline, monkeypatch, caplog):
    t, reader, step = _drive_teleop(monkeypatch)
    step(left_x=True)
    assert step(xy=(0.1, 0.0), left_x=True)["x.vel"] == pytest.approx(0.1)
    # Shoulders lost: stop, and stay stopped until Left X is released and pressed again.
    with caplog.at_level("WARNING", logger="rby1_xr_teleop"):
        a = step(xy=(0.1, 0.0), left_x=True, shoulders=False)
    assert a["x.vel"] == 0.0 and t._core._body_drive is None and "press it again" in caplog.text
    assert step(xy=(0.3, 0.0), left_x=True)["x.vel"] == 0.0
    step(xy=(0.3, 0.0))
    step(xy=(0.3, 0.0), left_x=True)  # new origin here
    assert step(xy=(0.35, 0.0), left_x=True)["x.vel"] == pytest.approx(0.05)

    # A jump beyond body_drive_max_error_m (tracking glitch): stop as well.
    a = step(xy=(1.5, 0.0), left_x=True)
    assert a["x.vel"] == 0.0 and t._core._body_drive is None and t._core._body_drive_blocked

    # Right B stops everything; the mode does not come back while stopped.
    step(xy=(1.5, 0.0))
    step(xy=(1.5, 0.0), left_x=True)
    assert t._core._body_drive is not None
    t._core._stopped = True
    assert step(xy=(1.7, 0.0), left_x=True)["x.vel"] == 0.0 and t._core._body_drive is None


def test_body_drive_needs_odometry_mobile_base_and_a_started_session(stubbed_pipeline, monkeypatch):
    t, reader, step = _drive_teleop(monkeypatch)
    reader.odometry = None
    step(left_x=True)
    assert step(xy=(0.2, 0.0), left_x=True)["x.vel"] == 0.0 and t._core._body_drive is None

    t, reader, step = _drive_teleop(monkeypatch, body_drive_enabled=False)
    step(left_x=True)
    assert step(xy=(0.2, 0.0), left_x=True)["x.vel"] == 0.0

    t, reader, step = _drive_teleop(monkeypatch, start_on_right_a=True)
    step(left_x=True)
    assert not t.is_started and step(xy=(0.2, 0.0), left_x=True)["x.vel"] == 0.0

    t, reader, step = _drive_teleop(monkeypatch, use_mobile_base=False)
    step(left_x=True)
    assert "x.vel" not in step(xy=(0.2, 0.0), left_x=True) and t._core._body_drive is None

    with pytest.raises(ValueError, match="body_drive_kp_linear"):
        Rby1XRConfig(body_drive_kp_linear=0.0)
    with pytest.raises(ValueError, match="body_drive_smoothing"):
        Rby1XRConfig(body_drive_smoothing=0.0)


def test_head_and_base_hold_while_no_grip_is_squeezed(stubbed_pipeline, monkeypatch):
    t, reader, step = _drive_teleop(monkeypatch, head_base_require_grip=True, head_latch_on_a=True,
                                    torso_source="none", use_torso=False)
    session = t._session

    def stick(thumb=(0.0, 1.0), **operator):
        """Like step(), with the right thumbstick pushed."""
        frame = _operator(**operator)
        hand = frame[OUT_CONTROLLER_RIGHT]
        frame[OUT_CONTROLLER_RIGHT] = fakes.controller(
            tuple(np.asarray(hand[fakes.ControllerInputIndex.GRIP_POSITION]).tolist()),
            squeeze=operator.get("squeeze", 0.0), thumb=thumb)
        session.push(frame)
        return t.get_action()

    a0 = step()
    head0 = (a0["head_0.pos"], a0["head_1.pos"])
    # No grip squeezed: the thumbstick, Left X body drive and a turned head change nothing.
    a = stick(neck_deg=30.0)
    assert (a["x.vel"], a["y.vel"], a["theta.vel"]) == (0.0, 0.0, 0.0)
    assert (a["head_0.pos"], a["head_1.pos"]) == pytest.approx(head0)
    step(left_x=True, neck_deg=30.0)
    a = step(xy=(0.2, 0.0), left_x=True, neck_deg=30.0)
    assert a["x.vel"] == 0.0 and t._core._body_drive is None
    step(xy=(0.2, 0.0), neck_deg=30.0)

    # Squeezing: the head does not jump to the 30 deg it is turned by — it starts from here.
    a = step(xy=(0.2, 0.0), neck_deg=30.0, squeeze=0.9)
    assert (a["head_0.pos"], a["head_1.pos"]) == pytest.approx(head0, abs=1e-6)
    a = step(xy=(0.2, 0.0), neck_deg=40.0, squeeze=0.9)
    assert a["head_0.pos"] == pytest.approx(head0[0] + math.radians(10.0), abs=1e-6)
    # … and the base is live: thumbstick, then body drive.
    assert stick(xy=(0.2, 0.0), neck_deg=40.0, squeeze=0.9)["x.vel"] > 0.0
    step(xy=(0.2, 0.0), neck_deg=40.0, squeeze=0.9, left_x=True)
    assert step(xy=(0.3, 0.0), neck_deg=40.0, squeeze=0.9, left_x=True)["x.vel"] == pytest.approx(0.1)

    # Releasing the grip stops the base at once (body drive ends) and holds the head.
    a = step(xy=(0.3, 0.0), neck_deg=40.0, left_x=True)
    held = (a["head_0.pos"], a["head_1.pos"])
    assert a["x.vel"] == 0.0 and t._core._body_drive is None
    reader.head_q = np.array(held)  # the (fake) robot followed the head command
    a = stick(xy=(0.3, 0.0), neck_deg=0.0)
    assert a["x.vel"] == 0.0 and (a["head_0.pos"], a["head_1.pos"]) == pytest.approx(held)


def test_body_drive_on_a_differential_base_has_no_lateral_velocity(stubbed_pipeline, monkeypatch):
    import dataclasses

    t, reader, step = _drive_teleop(monkeypatch)
    t._core.robot = dataclasses.replace(t._core.robot, holonomic=False)
    step(left_x=True)
    a = step(xy=(0.1, 0.1), left_x=True)
    assert a["x.vel"] == pytest.approx(0.1) and a["y.vel"] == 0.0
