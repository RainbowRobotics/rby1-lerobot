"""hold(): lerobot's command hold re-sends the last targets and stops the base."""

from __future__ import annotations

import numpy as np
import pytest
from lerobot_robot_rby1 import Rby1, Rby1Config

FAST = {"address": "stub.invalid:1", "model": "a", "version": "1.2", "move_to_ready_on_connect": False}


def _connect(**kw) -> Rby1:
    robot = Rby1(Rby1Config(**{**FAST, **kw}))
    robot.connect()
    return robot


def _last_command(robot: Rby1):
    """The last stream command's (body, mobility) builders from the fake."""
    cmd = robot._robot.last_stream_command
    cbc = cmd.get("set_command")[0]
    body = cbc.get("set_body_command")
    mobility = cbc.get("set_mobility_command")
    return (body[0] if body else None), (mobility[0] if mobility else None)


def _joint_action(robot: Rby1, **override) -> dict[str, float]:
    obs = robot.get_observation()
    action = {k: float(obs[k]) for k in robot.action_features if k in obs}
    action.update({k: 0.0 for k in ("x.vel", "y.vel", "theta.vel") if k in robot.action_features})
    action.update(override)
    return action


def test_declares_command_hold() -> None:
    assert Rby1.supports_command_hold is True


def test_hold_resends_last_joint_targets() -> None:
    robot = _connect(use_gripper=False)
    try:
        action = _joint_action(robot, **{"right_arm_0.pos": 0.4})
        robot.send_action(action)
        n = robot._robot.n_stream_commands
        robot.hold()
        assert robot._robot.n_stream_commands == n + 1
        body, _ = _last_command(robot)
        right = body.get("set_right_arm_command")[0].get("set_position")[0]
        assert right[0] == pytest.approx(0.4)
    finally:
        robot.disconnect()


def test_hold_zeroes_base_velocity() -> None:
    robot = _connect(use_gripper=False, use_mobile_base=True)
    try:
        robot.send_action(_joint_action(robot, **{"x.vel": 0.3, "theta.vel": -0.5}))
        _, mobility = _last_command(robot)
        linear, angular = mobility.get("set_velocity")
        assert linear[0] == pytest.approx(0.3) and angular == pytest.approx(-0.5)
        robot.hold()
        _, mobility = _last_command(robot)
        linear, angular = mobility.get("set_velocity")
        assert np.allclose(linear, 0.0) and angular == 0.0
        assert robot._last_sent["x.vel"] == 0.0
    finally:
        robot.disconnect()


def test_hold_before_any_action_uses_current_pose() -> None:
    robot = _connect(use_gripper=False)
    try:
        n = robot._robot.n_stream_commands
        robot.hold()
        assert robot._robot.n_stream_commands == n + 1
        body, _ = _last_command(robot)
        right = body.get("set_right_arm_command")[0].get("set_position")[0]
        obs = robot.get_observation()
        assert np.allclose(right, [obs[f"right_arm_{i}.pos"] for i in range(7)])
    finally:
        robot.disconnect()


def test_hold_in_ee_mode_resends_pose_without_reset() -> None:
    robot = _connect(use_gripper=False, action_mode="ee", use_torso=True)
    try:
        obs = robot.get_observation()
        action = {k: float(obs[k]) for k in robot.action_features if k in obs}
        robot.send_action(action)
        n = robot._robot.n_stream_commands
        robot.hold()
        assert robot._robot.n_stream_commands == n + 1
        body, _ = _last_command(robot)
        right = body.get("set_right_arm_command")[0]
        assert right.get("set_reset_reference") == (False,)
    finally:
        robot.disconnect()


@pytest.mark.parametrize(
    "overrides",
    [{}, {"use_mobile_base": True}, {"action_mode": "ee", "use_torso": True}],
    ids=["joint", "joint+base", "ee"],
)
def test_lerobot_wrapper_configures_command_hold(overrides: dict) -> None:
    """lerobot 0.7.0's ThreadSafeRobot must accept every RB-Y1 action mode."""
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot

    robot = _connect(use_gripper=False, **overrides)
    try:
        wrapper = ThreadSafeRobot(robot)
        wrapper.configure_hold()  # raises for robots without a hold contract
        assert wrapper.supports_hold
        n = robot._robot.n_stream_commands
        wrapper.hold()
        assert robot._robot.n_stream_commands == n + 1
    finally:
        robot.disconnect()
