"""send_action contract used by upstream rollout's position hold."""

from __future__ import annotations

import math

from lerobot_robot_rby1.constants import RIGHT_ARM_Q_MAX


def _current_action(robot) -> dict[str, float]:
    obs = robot.get_observation()
    return {key: float(obs[key]) for key in robot.action_features}


def test_returns_every_actuator_target(connected_robot) -> None:
    sent = connected_robot.send_action(_current_action(connected_robot))
    assert set(sent) == set(connected_robot.action_features)
    assert all(math.isfinite(v) for v in sent.values())


def test_returns_clipped_arm_targets(connected_robot) -> None:
    action = _current_action(connected_robot)
    action["right_arm_0.pos"] = 99.0
    sent = connected_robot.send_action(action)
    assert sent["right_arm_0.pos"] == float(RIGHT_ARM_Q_MAX[0])


def test_missing_gripper_key_holds_position(connected_robot) -> None:
    action = _current_action(connected_robot)
    action["left_gripper_0.pos"] = 0.25
    connected_robot.send_action(action)
    held = connected_robot.get_observation()["left_gripper_0.pos"]

    del action["left_gripper_0.pos"]
    sent = connected_robot.send_action(action)
    assert sent["left_gripper_0.pos"] == held
    assert connected_robot.get_observation()["left_gripper_0.pos"] == held
