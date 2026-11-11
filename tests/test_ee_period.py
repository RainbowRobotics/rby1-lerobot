"""EE mode: minimum_time follows the measured interval between actions."""

from __future__ import annotations

import pytest
from lerobot_robot_rby1 import Rby1, Rby1Config
from lerobot_robot_rby1.action_period import MAX_PERIOD, MIN_PERIOD, ActionPeriodEstimator

FACTOR = 1.02


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def ee_robot():
    robots = []

    def make(**kw) -> tuple[Rby1, FakeClock]:
        robot = Rby1(
            Rby1Config(address="stub.invalid:1", model="a", version="1.2", action_mode="ee",
                       use_gripper=False, move_to_ready_on_connect=False, **kw)
        )
        robot.connect()
        robot._clock = FakeClock()
        robots.append(robot)
        return robot, robot._clock

    yield make
    for robot in robots:
        robot.disconnect()


def _step(robot: Rby1, clock: FakeClock, dt: float) -> float:
    """Send the current pose ``dt`` seconds after the previous action; return the sent minimum_time."""
    clock.now += dt
    obs = robot.get_observation()
    robot.send_action({key: float(obs[key]) for key in robot.action_features})
    body = robot._robot.last_stream_command.get("set_command")[0].get("set_body_command")[0]
    times = {body.get(f"set_{group}_command")[0].get("set_minimum_time")[0] for group in ("right_arm", "left_arm")}
    assert len(times) == 1  # both solvers get the same minimum_time
    return times.pop()


def test_first_command_uses_the_configured_period(ee_robot) -> None:
    robot, clock = ee_robot()
    assert _step(robot, clock, 0.0) == pytest.approx(FACTOR / 15)


def test_minimum_time_converges_to_the_action_rate(ee_robot) -> None:
    robot, clock = ee_robot()
    for _ in range(40):
        minimum_time = _step(robot, clock, 1 / 30)
    assert minimum_time == pytest.approx(FACTOR / 30, rel=0.01)


def test_a_pause_does_not_disturb_the_estimate(ee_robot) -> None:
    robot, clock = ee_robot()
    for _ in range(40):
        before = _step(robot, clock, 1 / 30)
    assert _step(robot, clock, 5.0) == pytest.approx(before)  # e.g. saving an episode
    assert _step(robot, clock, 1 / 30) == pytest.approx(before, rel=0.01)


def test_ready_pose_motion_restarts_from_the_configured_period(ee_robot) -> None:
    robot, clock = ee_robot()
    for _ in range(40):
        _step(robot, clock, 1 / 30)
    robot.move_to_ready_pose(minimum_time=0.1)
    assert _step(robot, clock, 1 / 30) == pytest.approx(FACTOR / 15)


def test_fixed_period_when_auto_is_off(ee_robot) -> None:
    robot, clock = ee_robot(ee_auto_action_period=False, ee_action_period=0.05)
    for _ in range(20):
        assert _step(robot, clock, 1 / 30) == pytest.approx(0.05 * FACTOR)


def test_estimator_bounds_and_lasting_rate_change() -> None:
    est = ActionPeriodEstimator(1 / 15)
    t = 0.0
    for _ in range(200):  # 1 kHz caller: clamped to the lower bound
        t += 0.001
        est.update(t)
    assert est.period == pytest.approx(MIN_PERIOD)
    for _ in range(10):  # a much slower rate that stays is adopted, within the upper bound
        t += 1.0
        est.update(t)
    assert est.period == pytest.approx(MAX_PERIOD)
    est.update(t)  # same timestamp: ignored
    assert est.period == pytest.approx(MAX_PERIOD)
