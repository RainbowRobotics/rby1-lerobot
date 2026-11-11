"""Robot features used by the rby1_isaac teleoperator: head, posture hints, reset.cmd, frame bus."""

from __future__ import annotations

import logging

import numpy as np
import pytest
from lerobot_robot_rby1 import Rby1, Rby1Config, frame_bus
from lerobot_robot_rby1.constants import RESET_CMD_KEY

FAST = {"address": "stub.invalid:1", "model": "a", "version": "1.2", "move_to_ready_on_connect": False,
        "use_gripper": False}
HEAD = ("head_0.pos", "head_1.pos")
RIGHT_HINT = tuple(f"right_arm_{i}.null" for i in range(4))
LEFT_HINT = tuple(f"left_arm_{i}.null" for i in range(4))


def _connect(**kw) -> Rby1:
    robot = Rby1(Rby1Config(**{**FAST, **kw}))
    robot.connect()
    return robot


def _cbc(robot: Rby1):
    return robot._robot.last_stream_command.get("set_command")[0]


def _action(robot: Rby1, **override) -> dict[str, float]:
    obs = robot.get_observation()
    action = {k: float(obs[k]) for k in robot.action_features if k in obs}
    action.update(override)
    return action


def _nullspace(robot: Rby1, side: str):
    """(target, weight) of the last EE command's nullspace term for one arm."""
    arm = _cbc(robot).get("set_body_command")[0].get(f"set_{side}_arm_command")[0]
    target, weight, *_ = arm.get("set_nullspace_joint_target")
    return np.asarray(target), np.asarray(weight)


# ── Head ────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("mode", ["joint", "ee"])
def test_head_is_commanded_and_clipped(mode: str) -> None:
    robot = _connect(use_head=True, action_mode=mode)
    try:
        assert set(HEAD) <= set(robot.action_features) and set(HEAD) <= set(robot.observation_features)
        hi = robot._q_upper[robot._model.head_idx]
        sent = robot.send_action(_action(robot, **{"head_0.pos": 0.2, "head_1.pos": 99.0}))
        assert set(sent) == set(robot.action_features)
        assert sent["head_0.pos"] == pytest.approx(0.2) and sent["head_1.pos"] == pytest.approx(hi[1])
        head = _cbc(robot).get("set_head_command")[0].get("set_position")[0]
        assert list(head) == pytest.approx([0.2, hi[1]])
        assert robot.get_observation()["head_0.pos"] == pytest.approx(0.2)

        # hold() re-sends the head target.
        robot.hold()
        assert list(_cbc(robot).get("set_head_command")[0].get("set_position")[0]) == pytest.approx([0.2, hi[1]])

        # Without head keys the head is not commanded and the last target is reported.
        action = _action(robot)
        for key in HEAD:
            del action[key]
        sent = robot.send_action(action)
        assert _cbc(robot).get("set_head_command") is None
        assert sent["head_0.pos"] == pytest.approx(0.2)
    finally:
        robot.disconnect()


def test_head_keys_are_ignored_without_use_head() -> None:
    robot = _connect()
    try:
        assert not set(HEAD) & set(robot.action_features)
        sent = robot.send_action(_action(robot, **{"head_0.pos": 0.2, "head_1.pos": 0.1}))
        assert _cbc(robot).get("set_head_command") is None and not set(HEAD) & set(sent)
    finally:
        robot.disconnect()


# ── Posture hints ───────────────────────────────────────────────────────────
def test_posture_hint_changes_nullspace_target_and_weight() -> None:
    robot = _connect(action_mode="ee", ee_posture_hint_weight=7.0)
    try:
        cfg = robot._config
        assert not any(k.endswith(".null") for k in robot.action_features)
        base = _action(robot)
        robot.send_action(base)
        target, weight = _nullspace(robot, "right")
        assert target == pytest.approx(robot._null_right) and weight == pytest.approx(cfg.ee_nullspace_weight)

        hint = dict(zip(RIGHT_HINT, (0.1, -0.2, 0.3, -0.4)))
        sent = robot.send_action({**base, **hint})
        assert set(sent) == set(robot.action_features)  # hints are not recorded by default
        target, weight = _nullspace(robot, "right")
        assert target[:4] == pytest.approx([0.1, -0.2, 0.3, -0.4])
        assert target[4:] == pytest.approx(robot._null_right[4:])
        assert weight[:4] == pytest.approx([7.0] * 4)
        assert weight[4:] == pytest.approx(cfg.ee_nullspace_weight[4:])
        # The other arm keeps the config values.
        target, weight = _nullspace(robot, "left")
        assert target == pytest.approx(robot._null_left) and weight == pytest.approx(cfg.ee_nullspace_weight)

        # hold() keeps the hinted posture although the hint is not an action feature.
        robot.hold()
        target, weight = _nullspace(robot, "right")
        assert target[:4] == pytest.approx([0.1, -0.2, 0.3, -0.4]) and weight[0] == pytest.approx(7.0)

        # An incomplete hint is ignored.
        robot.send_action({**base, RIGHT_HINT[0]: 0.5})
        target, weight = _nullspace(robot, "right")
        assert target == pytest.approx(robot._null_right) and weight == pytest.approx(cfg.ee_nullspace_weight)
        robot.hold()
        assert _nullspace(robot, "right")[0] == pytest.approx(robot._null_right)
    finally:
        robot.disconnect()


def test_recorded_posture_hint_is_an_action_feature() -> None:
    robot = _connect(action_mode="ee", record_posture_hint=True)
    try:
        assert set(RIGHT_HINT + LEFT_HINT) <= set(robot.action_features)
        base = _action(robot)
        # No hint in the action: the default nullspace target is reported.
        sent = robot.send_action(base)
        assert set(sent) == set(robot.action_features)
        assert [sent[k] for k in LEFT_HINT] == pytest.approx(list(robot._null_left[:4]))
        sent = robot.send_action({**base, **dict(zip(LEFT_HINT, (0.1, 0.2, 0.3, 0.4)))})
        assert [sent[k] for k in LEFT_HINT] == pytest.approx([0.1, 0.2, 0.3, 0.4])
        assert [sent[k] for k in RIGHT_HINT] == pytest.approx(list(robot._null_right[:4]))
    finally:
        robot.disconnect()


def test_hints_do_not_exist_in_joint_mode() -> None:
    robot = Rby1(Rby1Config(address="stub.invalid:1", record_posture_hint=True))
    assert not any(k.endswith(".null") for k in robot.action_features)


# ── reset.cmd ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize("mode", ["joint", "ee"])
def test_reset_cmd_moves_to_ready_pose(mode: str, monkeypatch) -> None:
    robot = _connect(action_mode=mode, use_head=True, use_mobile_base=True)
    try:
        assert RESET_CMD_KEY not in robot.action_features
        action = _action(robot, **{"x.vel": 0.3})
        robot.send_action(action)
        calls = []
        real = robot.move_to_ready_pose
        monkeypatch.setattr(robot, "move_to_ready_pose", lambda **kw: (calls.append(kw), real(**kw))[1])
        n = robot._robot.n_stream_commands

        sent = robot.send_action({**action, RESET_CMD_KEY: 2.0})
        assert calls == [{"minimum_time": 2.0}]
        assert robot._robot.n_stream_commands == n  # no streamed command on the reset tick
        assert set(sent) == set(robot.action_features)
        assert RESET_CMD_KEY not in sent and RESET_CMD_KEY not in robot._last_sent
        assert sent["x.vel"] == 0.0
        q = np.asarray(robot._robot.get_state().position)
        assert q[robot._model.right_arm_idx] == pytest.approx(robot._ready_right)
        if mode == "joint":
            assert sent["right_arm_3.pos"] == pytest.approx(robot._ready_right[3])
        assert robot._last_ee_targets is None  # the next EE command re-references the solver

        # reset.cmd = 0 is a normal action.
        robot.send_action({**action, RESET_CMD_KEY: 0.0})
        assert calls == [{"minimum_time": 2.0}] and robot._robot.n_stream_commands == n + 1
    finally:
        robot.disconnect()


# ── Ready torso ─────────────────────────────────────────────────────────────
def test_ee_ready_torso_deg_overrides_the_ready_torso() -> None:
    torso = [0.0, 45.0, -90.0, 45.0, 0.0, 0.0]
    robot = _connect(action_mode="ee", use_torso=True, ee_ready_torso_deg=torso)
    try:
        assert robot._ready_body[:6] == pytest.approx(np.deg2rad(torso))
        assert robot._ready_body[6:13] == pytest.approx(robot._ready_right)
    finally:
        robot.disconnect()
    # Joint mode ignores it.
    robot = _connect(ee_ready_torso_deg=torso)
    try:
        assert robot._ready_body[:6] != pytest.approx(np.deg2rad(torso))
    finally:
        robot.disconnect()


def test_ee_ready_torso_deg_length_is_checked() -> None:
    with pytest.raises(ValueError, match="ee_ready_torso_deg"):
        Rby1Config(address="stub.invalid:1", ee_ready_torso_deg=[0.0, 45.0])


def test_ready_torso_outside_ee_limits_warns(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        robot = _connect(action_mode="ee", use_torso=True)
        robot.disconnect()
    assert any("ee_ready_torso_deg" in r.getMessage() for r in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        robot = _connect(action_mode="ee", use_torso=True, ee_ready_torso_deg=[0, 45, -90, 45, 0, 0])
        robot.disconnect()
    assert not any("ee_ready_torso_deg" in r.getMessage() for r in caplog.records)


# ── Robot log stream ────────────────────────────────────────────────────────
class _Log:
    def __init__(self, level: str, message: str) -> None:
        self.level, self.message = level, message


def test_robot_logs_are_forwarded(caplog) -> None:
    robot = _connect()
    try:
        sdk = robot._robot
        assert sdk.log_stream_cb is not None
        with caplog.at_level(logging.INFO):
            sdk.log_stream_cb([_Log("Level.Info", "hello"), _Log("Level.Warn", "careful"), _Log("Level.Error", "bad")])
        messages = [r.getMessage() for r in caplog.records]
        assert "[robot] careful" in messages and "[robot] bad" in messages
        assert "[robot] hello" not in messages  # forward_robot_logs_info is off
    finally:
        robot.disconnect()
    assert sdk.log_stream_cb is None


def test_robot_log_stream_can_be_disabled() -> None:
    robot = _connect(forward_robot_logs=False)
    try:
        assert getattr(robot._robot, "log_stream_cb", None) is None
    finally:
        robot.disconnect()


# ── Frame bus ───────────────────────────────────────────────────────────────
def test_frame_bus_publish_latest_and_sequence() -> None:
    bus = frame_bus.FrameBus()
    assert bus.latest("front") is None and bus.names() == []
    f1 = np.zeros((2, 2, 3), np.uint8)
    bus.publish("front", f1, t=1.0)
    frame, t, seq = bus.latest("front")
    assert frame is f1 and t == 1.0 and seq == 1
    bus.publish("front", np.ones((2, 2, 3), np.uint8))
    assert bus.latest("front")[2] == 2
    bus.publish("left", f1)
    assert sorted(bus.names()) == ["front", "left"]
    bus.clear()
    assert bus.names() == []


# ── Expired command stream ──────────────────────────────────────────────────
def test_expired_stream_is_retried_once_while_enabled() -> None:
    robot = _connect()
    try:
        action = _action(robot)
        robot.send_action(action)
        robot._stream._done = True  # the robot ended the streamed command
        n = robot._robot.n_stream_commands
        robot.send_action(action)
        assert robot._robot.n_stream_commands == n + 1
    finally:
        robot.disconnect()


def test_expired_stream_with_a_faulted_control_manager_raises() -> None:
    import rby1_sdk as rby

    robot = _connect()
    try:
        action = _action(robot)
        robot.send_action(action)
        robot._stream._done = True
        robot._robot._cm = rby.ControlManagerState.State.MajorFault
        n = robot._robot.n_stream_commands
        with pytest.raises(RuntimeError, match="control manager state"):
            robot.send_action(action)
        assert robot._robot.n_stream_commands == n
    finally:
        robot._robot._cm = rby.ControlManagerState.State.Enabled
        robot.disconnect()
