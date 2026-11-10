"""send_action contract used by upstream rollout's position hold."""

from __future__ import annotations

import math

import pytest
from lerobot_robot_rby1.constants import RIGHT_ARM_Q_MAX, TORSO_Q_MAX


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


# ── Joint mode with the torso (use_torso=True) ──────────────────────────────
@pytest.fixture(scope="module")
def torso_robot():
    from lerobot_robot_rby1 import Rby1, Rby1Config

    robot = Rby1(
        Rby1Config(address="stub.invalid:1", model="a", version="1.2", use_torso=True,
                   use_gripper=False, move_to_ready_on_connect=False)
    )
    robot.connect()
    yield robot
    robot.disconnect()


def _body(robot):
    return robot._robot.last_stream_command.get("set_command")[0].get("set_body_command")[0]


def test_joint_mode_commands_the_torso(torso_robot) -> None:
    assert list(torso_robot.action_features)[:6] == [f"torso_{i}.pos" for i in range(6)]
    action = _current_action(torso_robot)
    action["torso_3.pos"] = 0.3
    action["torso_1.pos"] = 99.0  # clipped to the URDF limit
    sent = torso_robot.send_action(action)
    assert set(sent) == set(torso_robot.action_features)
    assert sent["torso_3.pos"] == pytest.approx(0.3)
    assert sent["torso_1.pos"] == float(TORSO_Q_MAX[1])

    torso = _body(torso_robot).get("set_torso_command")[0]
    assert type(torso).__name__ == "JointPositionCommandBuilder"
    assert list(torso.get("set_position")[0]) == pytest.approx([sent[f"torso_{i}.pos"] for i in range(6)])
    # The torso follows the startup ramp like the arms.
    arm = _body(torso_robot).get("set_right_arm_command")[0]
    assert torso.get("set_minimum_time") == arm.get("set_minimum_time")
    assert torso_robot.get_observation()["torso_3.pos"] == pytest.approx(0.3)


def test_hold_resends_the_torso_target(torso_robot) -> None:
    action = _current_action(torso_robot)
    action["torso_3.pos"] = 0.25
    torso_robot.send_action(action)
    n = torso_robot._robot.n_stream_commands
    torso_robot.hold()
    assert torso_robot._robot.n_stream_commands == n + 1
    assert _body(torso_robot).get("set_torso_command")[0].get("set_position")[0][3] == pytest.approx(0.25)


def test_joint_impedance_uses_the_torso_gains() -> None:
    import numpy as np
    import rby1_sdk as rby
    from lerobot_robot_rby1 import Rby1Config
    from lerobot_robot_rby1 import command_builders as cb

    cfg = Rby1Config(
        address="stub.invalid:1", use_torso=True, use_joint_impedance=True,
        joint_impedance_torso_stiffness=[300.0] * 6, joint_impedance_torso_torque_limit=[200.0] * 6,
    )
    model = rby.FakeRobot("stub.invalid:1", "a").model()
    action = {key: 0.0 for key in cb.clipped_joint_action(cfg, _zeros(cfg))}
    limits = np.ones(len(model.robot_joint_names))
    body, has_command = cb.build_body_command(rby, cfg, action, model, limits, limits, 0.07)
    assert has_command
    torso = body.get("set_torso_command")[0]
    assert type(torso).__name__ == "JointImpedanceControlCommandBuilder"
    assert torso.get("set_stiffness") == ([300.0] * 6,)
    assert torso.get("set_torque_limit") == ([200.0] * 6,)
    arm = body.get("set_left_arm_command")[0]
    assert arm.get("set_stiffness") == ([150.0] * 7,)


def _zeros(cfg) -> dict[str, float]:
    from lerobot_robot_rby1 import Rby1

    return {key: 0.0 for key in Rby1(cfg).action_features}
