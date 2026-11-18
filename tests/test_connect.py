"""connect()/disconnect() robustness against the fake SDK."""

from __future__ import annotations

import numpy as np
import pytest
import rby1_sdk as rby
from lerobot_robot_rby1 import Rby1, Rby1Config
from lerobot_robot_rby1 import gripper as gripper_mod


def _calls_since(n: int) -> list[str]:
    return [c[0] for c in rby.CALL_LOG[n:]]


def _robot(**kw) -> Rby1:
    return Rby1(Rby1Config(address="stub.invalid:1", model="a", version="1.2", **kw))


def test_failed_gripper_ping_rolls_back(monkeypatch) -> None:
    monkeypatch.setattr(rby.DynamixelBus, "ping", lambda self, dev_id: False)
    monkeypatch.setattr(gripper_mod, "GRIPPER_PING_TIMEOUT_S", 0.3)
    robot = _robot()
    n = len(rby.CALL_LOG)
    with pytest.raises(RuntimeError, match="ping"):
        robot.connect()
    calls = _calls_since(n)
    assert "disable_control_manager" in calls
    assert ("set_tool_flange_output_voltage", ("right", 0), {}) in rby.CALL_LOG[n:]
    assert not robot.is_connected
    assert robot._robot is None


def test_gripper_ping_is_retried(monkeypatch) -> None:
    attempts = {"n": 0}

    def flaky_ping(self, dev_id):
        attempts["n"] += 1
        return attempts["n"] > 3  # boots after a few tries

    monkeypatch.setattr(rby.DynamixelBus, "ping", flaky_ping)
    robot = _robot(move_to_ready_on_connect=False)
    robot.connect()
    robot.disconnect()
    assert attempts["n"] > 3


def test_disconnect_cancels_stream_before_disabling(monkeypatch) -> None:
    robot = _robot(move_to_ready_on_connect=False, use_gripper=False)
    robot.connect()
    order: list[str] = []
    monkeypatch.setattr(robot._stream, "cancel", lambda: order.append("cancel"))
    real_disable = robot._robot.disable_control_manager
    monkeypatch.setattr(
        robot._robot, "disable_control_manager", lambda: (order.append("disable"), real_disable())[1]
    )
    robot.disconnect()
    assert order == ["cancel", "disable"]


def test_disconnect_waits_for_the_cancelled_command_before_disabling(caplog) -> None:
    """The core refuses to disable while a command is active; a cancelled stream ends a moment later."""
    robot = _robot(move_to_ready_on_connect=False, use_gripper=False)
    robot.connect()
    fake = robot._robot
    robot.send_action(dict(robot.get_observation()))  # a streamed command: an active control
    assert fake.get_control_manager_state().control_state == rby.ControlManagerState.ControlState.Executing
    assert fake.disable_control_manager() is False  # what asking straight away gets
    n = len(rby.CALL_LOG)
    robot.disconnect()
    assert fake.get_control_manager_state().state == rby.ControlManagerState.State.Idle
    assert _calls_since(n).count("disable_control_manager") == 1  # asked once, when it could succeed
    assert "refused to disable" not in caplog.text


def test_disconnect_reports_a_control_manager_that_stays_enabled(monkeypatch, caplog) -> None:
    robot = _robot(move_to_ready_on_connect=False, use_gripper=False)
    robot.connect()
    fake = robot._robot
    monkeypatch.setattr(fake, "disable_control_manager", lambda: False)
    robot.disconnect()
    assert "refused to disable" in caplog.text
    assert fake.get_control_manager_state().state == rby.ControlManagerState.State.Enabled


def test_ready_pose_failure_fails_connect(monkeypatch) -> None:
    monkeypatch.setattr(rby.FakeRobot, "wait_for_control_ready", lambda self, timeout_ms: False)
    robot = _robot(use_gripper=False)
    with pytest.raises(RuntimeError, match="move_to_ready_pose"):
        robot.connect()
    assert not robot.is_connected


def test_gripper_read_failure_keeps_last_position(monkeypatch) -> None:
    robot = _robot(move_to_ready_on_connect=False)
    robot.connect()
    try:
        good = robot._gripper.get_positions()
        monkeypatch.setattr(rby.DynamixelBus, "group_fast_sync_read_encoder", lambda self, ids: None)
        np.testing.assert_array_equal(robot._gripper.get_positions(), good)
    finally:
        robot.disconnect()
