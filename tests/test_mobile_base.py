"""Mobile-base safety: short hold, clamped velocities, stream re-creation."""

from __future__ import annotations

import time

import pytest
from lerobot_robot_rby1 import Rby1, Rby1Config


@pytest.fixture(scope="module")
def base_robot():
    robot = Rby1(Rby1Config(address="stub.invalid:1", model="a", version="1.2", use_mobile_base=True))
    robot.connect()
    yield robot
    robot.disconnect()


def _action(robot, **base) -> dict[str, float]:
    obs = robot.get_observation()
    action = {k: float(obs[k]) for k in robot.action_features if k in obs}
    action.update({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0, **base})
    return action


def test_base_command_uses_short_hold_time(base_robot) -> None:
    import rby1_sdk as rby

    builder = base_robot._mobility_command(rby, {"x.vel": 0.1}, 0.07)
    header = builder.get("set_command_header")[0]
    assert header.get("set_control_hold_time") == (base_robot._config.mobile_base_hold_time,)


def test_base_velocity_is_clamped(base_robot) -> None:
    sent = base_robot.send_action(_action(base_robot, **{"x.vel": 3.0, "y.vel": 4.0, "theta.vel": -9.0}))
    cfg = base_robot._config
    assert (sent["x.vel"] ** 2 + sent["y.vel"] ** 2) ** 0.5 == pytest.approx(cfg.mobile_base_max_linear_velocity)
    assert sent["x.vel"] / sent["y.vel"] == pytest.approx(3.0 / 4.0)
    assert sent["theta.vel"] == -cfg.mobile_base_max_angular_velocity


def test_stream_is_recreated_after_a_stall(base_robot) -> None:
    base_robot.send_action(_action(base_robot, **{"x.vel": 0.1}))
    old_stream = base_robot._stream
    time.sleep(base_robot._config.mobile_base_hold_time + 0.2)  # the base command finishes
    base_robot.send_action(_action(base_robot))
    assert base_robot._stream is not old_stream
