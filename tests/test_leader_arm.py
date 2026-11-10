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
