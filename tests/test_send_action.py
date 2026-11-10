"""send_action contract used by upstream rollout's position hold."""

from __future__ import annotations

import math

import pytest
import rby1_sdk as rby


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
    assert sent["right_arm_0.pos"] == rby.JOINT_Q_LIMITS["right_arm_0"][1]


def test_clips_to_the_limits_the_robot_reports(monkeypatch) -> None:
    """The limits are read at connect, so another robot's (M v1.3 wrist) are followed."""
    from lerobot_robot_rby1 import Rby1, Rby1Config

    wrist = {"right_arm_5": (-0.872664626, 0.872664626), "right_arm_6": (-1.5707963268, 1.5707963268)}
    for name, limits in wrist.items():
        monkeypatch.setitem(rby.JOINT_Q_LIMITS, name, limits)
    robot = Rby1(Rby1Config(address="stub.invalid:1", model="m", version="1.3", use_gripper=False,
                            move_to_ready_on_connect=False))
    robot.connect()
    try:
        action = _current_action(robot)
        action["right_arm_5.pos"] = 1.5  # inside the v1.2 limit (110 deg), outside v1.3 (50 deg)
        action["right_arm_6.pos"] = -2.0
        sent = robot.send_action(action)
        assert sent["right_arm_5.pos"] == wrist["right_arm_5"][1]
        assert sent["right_arm_6.pos"] == wrist["right_arm_6"][0]
        arm = _body(robot).get("set_right_arm_command")[0]
        assert list(arm.get("set_position")[0])[5:] == [wrist["right_arm_5"][1], wrist["right_arm_6"][0]]
    finally:
        robot.disconnect()


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
    action["torso_1.pos"] = 99.0  # clipped to the robot's limit
    action["torso_2.pos"] = math.radians(85.0)  # the URDF limit is 80 deg
    sent = torso_robot.send_action(action)
    assert set(sent) == set(torso_robot.action_features)
    assert sent["torso_3.pos"] == pytest.approx(0.3)
    assert sent["torso_1.pos"] == rby.JOINT_Q_LIMITS["torso_1"][1]
    assert sent["torso_2.pos"] == pytest.approx(math.radians(80.0))

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
    from lerobot_robot_rby1 import Rby1Config
    from lerobot_robot_rby1 import command_builders as cb

    cfg = Rby1Config(
        address="stub.invalid:1", use_torso=True, use_joint_impedance=True,
        joint_impedance_torso_stiffness=[300.0] * 6, joint_impedance_torso_torque_limit=[200.0] * 6,
    )
    model = rby.FakeRobot("stub.invalid:1", "a").model()
    action = {key: 0.0 for key in _zeros(cfg) if "gripper" not in key}
    limits = np.ones(len(model.robot_joint_names))
    body, has_command, sent = cb.build_body_command(rby, cfg, action, model, -limits, limits, limits, limits, 0.07)
    assert has_command
    assert sent == action
    torso = body.get("set_torso_command")[0]
    assert type(torso).__name__ == "JointImpedanceControlCommandBuilder"
    assert torso.get("set_stiffness") == ([300.0] * 6,)
    assert torso.get("set_torque_limit") == ([200.0] * 6,)
    arm = body.get("set_left_arm_command")[0]
    assert arm.get("set_stiffness") == ([150.0] * 7,)


def _zeros(cfg) -> dict[str, float]:
    from lerobot_robot_rby1 import Rby1

    return {key: 0.0 for key in Rby1(cfg).action_features}
