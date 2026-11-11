"""Leader arm: the output is clipped to the joint limits read from the robot."""

from __future__ import annotations

import logging

import numpy as np
import pytest
import rby1_sdk as rby
from lerobot_teleoperator_rby1 import Rby1LeaderArm, Rby1LeaderArmConfig

ADDRESS = "stub.invalid:1"
FAST = {"robot_address": ADDRESS, "init_duration": 0.1, "startup_wait_time": 0.1,
        "leader_arm_model_path": "unused.urdf", "connect_timeout_sec": 0.2}


def _with_limits(**kw) -> Rby1LeaderArm:
    """A leader arm holding the robot's limits, without the leader-arm device."""
    leader = Rby1LeaderArm(Rby1LeaderArmConfig(**{**FAST, **kw}))
    leader._limits = leader._read_robot_limits(leader._config)
    leader._is_connected = True  # get_action only reads the shared state
    return leader


def test_connect_reads_the_limits_and_clips_the_output() -> None:
    leader = Rby1LeaderArm(Rby1LeaderArmConfig(**FAST))
    leader.connect()
    try:
        assert leader._limits.right_arm[1][3] == rby.JOINT_Q_LIMITS["right_arm_3"][1]
        leader._leader_arm.stop_control()  # the control loop no longer writes the state
        leader._right_q = np.full(7, 9.0)
        leader._left_q = np.full(7, -9.0)
        action = leader.get_action()
    finally:
        leader.disconnect()
    assert [action[f"right_arm_{i}.pos"] for i in range(7)] == [rby.JOINT_Q_LIMITS[f"right_arm_{i}"][1] for i in range(7)]
    assert [action[f"left_arm_{i}.pos"] for i in range(7)] == [rby.JOINT_Q_LIMITS[f"left_arm_{i}"][0] for i in range(7)]
    assert leader._limits is None


def test_output_follows_the_connected_robot(monkeypatch) -> None:
    """M v1.3 has a narrower wrist than the older robots."""
    monkeypatch.setenv("RBY1_FAKE_MODEL", "M")
    monkeypatch.setenv("RBY1_FAKE_VERSION", "1.3")
    monkeypatch.setitem(rby.JOINT_Q_LIMITS, "right_arm_5", (-0.872664626, 0.872664626))
    leader = _with_limits(version="1.3")
    leader._right_q = np.array([0.0, -0.5, 0.0, -1.0, 0.0, 1.5, 0.0])
    action = leader.get_action()
    assert action["right_arm_5.pos"] == 0.872664626
    assert action["right_arm_3.pos"] == -1.0


def test_unreachable_robot_fails_connect(monkeypatch) -> None:
    monkeypatch.setenv("RBY1_FAKE_CONNECT_FAIL", "1")
    leader = Rby1LeaderArm(Rby1LeaderArmConfig(**FAST))
    n = len(rby.CALL_LOG)
    with pytest.raises(RuntimeError, match="require_robot_joint_limits=false"):
        leader.connect()
    assert not leader.is_connected
    # Stopped before the leader-arm device was opened.
    assert "upc.initialize_device" not in [call[0] for call in rby.CALL_LOG[n:]]


def test_without_a_robot_the_output_is_not_clipped(monkeypatch, caplog) -> None:
    monkeypatch.setenv("RBY1_FAKE_CONNECT_FAIL", "1")
    leader = Rby1LeaderArm(Rby1LeaderArmConfig(**FAST, require_robot_joint_limits=False))
    with caplog.at_level(logging.WARNING):
        leader.connect()
    try:
        assert "NOT clipped" in caplog.text
        assert leader._limits is None
        leader._leader_arm.stop_control()  # the control loop no longer writes the state
        leader._right_q = np.full(7, 9.0)
        action = leader.get_action()
        assert action["right_arm_0.pos"] == 9.0
    finally:
        leader.disconnect()


def test_torso_target_is_checked_against_the_robot() -> None:
    # 85 deg on torso_2: accepted as a config value, outside the robot's 80 deg limit.
    cfg = Rby1LeaderArmConfig(**FAST, use_torso=True, torso_target_deg=[0.0, 0.0, 85.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match=r"torso_target_deg\[2\] = 85.0 .*\[-140.0, 80.0\]"):
        Rby1LeaderArm(cfg).connect()
    assert _with_limits(use_torso=True)._limits is not None  # the default target is inside


def test_version_mismatch_is_a_warning(monkeypatch, caplog) -> None:
    monkeypatch.setenv("RBY1_FAKE_MODEL", "M")
    monkeypatch.setenv("RBY1_FAKE_VERSION", "1.3")
    with caplog.at_level(logging.WARNING):
        _with_limits(version="1.2")
    assert "reports version '1.3' but teleop.version is '1.2'" in caplog.text


def test_connect_works_on_model_m(monkeypatch) -> None:
    """The 12V check must use a handle of the robot's model, not Model A."""
    monkeypatch.setenv("RBY1_FAKE_MODEL", "M")
    leader = Rby1LeaderArm(Rby1LeaderArmConfig(**FAST))
    n = len(rby.CALL_LOG)
    leader.connect()
    leader.disconnect()
    assert "power_on" in [call[0] for call in rby.CALL_LOG[n:]]


def test_button_press_does_not_step_the_output() -> None:
    """The arm was pushed 1 rad off its hold target; pressing the button
    brings the output there at resync_speed, then the output follows the arm."""
    import time
    from types import SimpleNamespace
    from threading import Lock
    from lerobot_teleoperator_rby1.leader_arm_controller import LeaderArmControlLoop

    teleop = SimpleNamespace(_lock=Lock(), _right_q=None, _left_q=None, _init_done=False,
                             _right_trigger=0.0, _left_trigger=0.0)
    speed = 2.0  # rad/s
    loop = LeaderArmControlLoop(
        teleop=teleop, rby=rby, right_init_q=np.zeros(7), left_init_q=np.zeros(7),
        min_q=np.full(14, -3.0), max_q=np.full(14, 3.0), torque_limit=np.ones(14),
        viscous_gain=np.zeros(14), barrier=0.0, gravity_scale=1.0, init_duration=0.0,
        resync_speed=speed,
    )
    q = np.zeros(14)
    q[3] = 1.0  # right arm, pushed while the button is released

    def tick(button: int) -> np.ndarray:
        state = rby.upc.LeaderArm.State(q)
        state.button_right.button = button
        loop(state)
        return teleop._right_q.copy()

    assert tick(0)[3] == 0.0  # hold mode: the output is the hold target
    outputs, t0 = [tick(0)], time.time()
    while outputs[-1][3] < 1.0:
        assert time.time() - t0 < 2.0
        time.sleep(0.005)
        outputs.append(tick(1))
    elapsed = time.time() - t0
    steps = np.diff([o[3] for o in outputs])
    assert len(outputs) > 20 and steps.max() < 0.1  # no step
    assert 0.4 < elapsed < 1.0  # 1 rad at 2 rad/s
    q[3] = 1.3  # caught up: the output follows the arm directly
    assert tick(1)[3] == 1.3
    assert tick(0)[3] == 1.3 and not loop.right_state.resync


def test_return_moves_the_arm_to_the_zero_position_without_a_step() -> None:
    """After request_return the arm goes from where it is to the zero position,
    the buttons are ignored and the teleoperator output stays where it was."""
    import time
    from types import SimpleNamespace
    from threading import Lock
    from lerobot_teleoperator_rby1.leader_arm_controller import LeaderArmControlLoop

    teleop = SimpleNamespace(_lock=Lock(), _right_q=None, _left_q=None, _init_done=False,
                             _right_trigger=0.0, _left_trigger=0.0)
    speed = 2.0  # rad/s
    init = np.full(7, 0.2)
    loop = LeaderArmControlLoop(
        teleop=teleop, rby=rby, right_init_q=init, left_init_q=init,
        min_q=np.full(14, -3.0), max_q=np.full(14, 3.0), torque_limit=np.ones(14),
        viscous_gain=np.zeros(14), barrier=0.0, gravity_scale=1.0, init_duration=0.0,
        resync_speed=1.0, return_speed=speed,
    )
    q = np.full(14, 0.2)
    q[3] = 0.8  # right arm, moved with the button pressed

    def tick(button: int):
        state = rby.upc.LeaderArm.State(q)
        state.button_right.button = button
        return loop(state)

    tick(1)
    tick(1)
    assert teleop._right_q[3] == 0.8
    loop.request_return()
    targets, times = [], []
    while not loop.return_done.is_set():
        assert len(targets) < 1000
        cmd = tick(1)  # still pressed: ignored
        assert (cmd.target_operating_mode == rby.DynamixelBus.CurrentBasedPositionControlMode).all()
        targets.append(cmd.target_position.copy())
        times.append(time.time())
        q[:] = cmd.target_position  # the arm follows its target
        time.sleep(0.005)
    targets = np.array(targets)
    assert abs(targets[0][3] - 0.8) < 0.01  # starts where the arm is
    np.testing.assert_allclose(targets[-1], 0.0)
    assert np.all(np.diff(targets[:, 3]) <= 1e-12)  # monotonic
    rate = np.abs(np.diff(targets[:, 3]) / np.diff(times))
    assert rate.max() < 1.3 * speed  # peak speed (timing jitter allowed)
    assert 0.6 < loop.return_duration < 0.65  # pi/2 * 0.8 rad / 2 rad/s
    assert teleop._right_q[3] == 0.8  # output frozen: the robot does not follow


def test_disconnect_returns_to_the_zero_position_then_stops(caplog) -> None:
    leader = Rby1LeaderArm(Rby1LeaderArmConfig(**FAST, return_speed_deg_s=300.0))
    leader.connect()
    arm, loop = leader._leader_arm, leader._control_loop
    assert np.abs(arm._q).max() > 0.5  # at the init pose
    with caplog.at_level(logging.INFO):
        leader.disconnect()
    assert loop.return_done.is_set() and "returning to its zero position" in caplog.text
    np.testing.assert_allclose(arm._q, 0.0, atol=1e-9)
    assert arm._stop.is_set() and not leader.is_connected


def test_disconnect_without_return_stops_where_the_arm_is() -> None:
    leader = Rby1LeaderArm(Rby1LeaderArmConfig(**FAST, return_to_zero_on_disconnect=False))
    leader.connect()
    arm, loop = leader._leader_arm, leader._control_loop
    leader.disconnect()
    assert arm._stop.is_set() and not loop.return_requested.is_set()


def test_disconnect_stops_when_the_control_loop_is_dead(caplog) -> None:
    leader = Rby1LeaderArm(Rby1LeaderArmConfig(**FAST))
    leader.connect()
    leader._leader_arm.stop_control()
    with caplog.at_level(logging.WARNING):
        leader.disconnect()
    assert "control loop is not running" in caplog.text and not leader.is_connected
