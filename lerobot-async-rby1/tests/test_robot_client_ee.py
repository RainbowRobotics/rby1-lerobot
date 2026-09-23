from __future__ import annotations

import importlib.util
import sys
import threading
import time
import types
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).parents[1] / "lerobot_async_inference" / "robot_client_ee.py"
SPEC = importlib.util.spec_from_file_location("robot_client_ee_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

SERVER_DESCRIPTOR = MODULE.SERVER_DESCRIPTOR
AsyncEEController = MODULE.AsyncEEController
EEClientConfig = MODULE.EEClientConfig
JOINT_KEYS = MODULE.JOINT_KEYS
_ActionChunk = MODULE._ActionChunk
_validate_descriptor = MODULE._validate_descriptor
_with_inference_safe_start = MODULE._with_inference_safe_start


class FakeRobotConfig:
    model = "rb10"
    first_state_timeout_s = 0.1
    control_rate_hz = 10.0


class FakeClock:
    def __init__(self, value: float = 10.0):
        self.value = value

    def __call__(self) -> float:
        return self.value


class FakeRobot:
    def __init__(self):
        self.observation = {key: 0.0 for key in JOINT_KEYS}
        self.observation.update(
            {
                "gripper_0": 1.0,
                "wrist": np.zeros((2, 2, 3), dtype=np.uint8),
            }
        )
        self.get_calls: list[int] = []
        self.send_calls: list[tuple[int, dict[str, float]]] = []
        self.enable_calls: list[int] = []
        self.disable_calls: list[int] = []

    def get_observation(self):
        self.get_calls.append(threading.get_ident())
        return dict(self.observation)

    def get_joint_positions(self, *, measured=True):
        assert measured
        return np.asarray([self.observation[key] for key in JOINT_KEYS])

    def get_reference_joint_positions(self):
        return self.get_joint_positions().copy()

    def send_action(self, action):
        self.send_calls.append((threading.get_ident(), dict(action)))
        return action

    def send_cartesian_action(self, pose_m_rad, gripper=None):
        self.send_calls.append((threading.get_ident(), {"pose": np.array(pose_m_rad), "gripper": gripper}))
        return {"gripper_0": gripper}

    def enable_servo_commands(self):
        self.enable_calls.append(threading.get_ident())

    def disable_servo_commands(self):
        self.disable_calls.append(threading.get_ident())


class FakeAdapter:
    def __init__(self):
        self.build_calls: list[tuple[int, float]] = []
        self.kinematics = types.SimpleNamespace(forward=self.validate_joints)

    @staticmethod
    def validate_joints(q):
        assert np.asarray(q).shape == (6,)
        if not np.all(np.isfinite(q)) or np.any(np.abs(q) > 2 * np.pi):
            raise ValueError("invalid joint position")
        return np.zeros(6)

    def validate_robot(self, robot):
        return None

    def build_observation(self, raw, task):
        self.build_calls.append((threading.get_ident(), raw["joint_0"]))
        return {"raw": raw, "task": task}

    def joint_action(self, action7, current_q6, *, max_joint_delta, previous_q=None, max_tracking_error=None):
        return {
            **{key: float(action7[index]) for index, key in enumerate(JOINT_KEYS)},
            "gripper_0": float(action7[6]),
        }


class IdleClient:
    def __init__(self):
        self.closed = False

    def describe(self):
        return dict(SERVER_DESCRIPTOR)

    def get_action(self, observation):
        return {"actions": np.zeros((1, 7))}

    def close(self):
        self.closed = True


def make_config(**overrides):
    values = {
        "robot": FakeRobotConfig(),
        "task": "test task",
        "control_rate_hz": 10.0,
        "max_joint_velocity_rad_s": 1.0,
        "max_joint_delta_cap_rad": 0.1,
        "max_joint_tracking_error_rad": 0.1,
        "timeout_ms": 50,
        "reconnect_delay_s": 0.01,
        "dry_run": True,
    }
    values.update(overrides)
    return EEClientConfig(**values)


def make_controller(config=None, *, robot=None, adapter=None, factory=None, clock=None):
    config = config or make_config()
    robot = robot or FakeRobot()
    adapter = adapter or FakeAdapter()
    factory = factory or (lambda address, timeout: IdleClient())
    controller = AsyncEEController(
        config,
        robot,
        adapter,
        client_factory=factory,
        parse_actions=lambda response: np.asarray(response["actions"], dtype=np.float64),
        clock=clock or FakeClock(),
    )
    return controller, robot, adapter


def wait_for(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not reached before timeout")


def install_chunk(controller, clock, actions, *, generation=None, observed_at=None):
    controller._set_chunk(
        _ActionChunk(
            generation=controller.generation if generation is None else generation,
            observed_at=clock() if observed_at is None else observed_at,
            received_at=clock(),
            actions=np.asarray(actions, dtype=np.float64),
        )
    )


def mark_network_ready(controller):
    with controller._state_lock:
        controller._network_ready = True


def test_config_and_descriptor_fail_closed_before_hardware():
    class StaleConfig(FakeRobotConfig):
        first_state_timeout_s = 0.251

    with pytest.raises(ValueError, match="stale joint state"):
        make_config(robot=StaleConfig())
    with pytest.raises(ValueError, match="kinematics_model"):
        make_config(kinematics_model="rby1")
    with pytest.raises(ValueError, match=r"\[1, 1000\]"):
        make_config(timeout_ms=1001)
    with pytest.raises(ValueError, match="action_log_interval_s"):
        make_config(action_log_interval_s=-1)

    _validate_descriptor(dict(SERVER_DESCRIPTOR))
    wrong = dict(SERVER_DESCRIPTOR, euler_convention="RxRyRz")
    with pytest.raises(ValueError, match="descriptor"):
        _validate_descriptor(wrong)


@pytest.mark.parametrize(
    "options",
    [{"dry_run": True}, {"dry_run": False, "move_to_ready_on_stop": False}],
)
def test_startup_never_actuates_before_f_even_after_network_connects(options):
    controller, robot, _ = make_controller(make_config(**options))
    controller.initialize_robot_safety()
    mark_network_ready(controller)
    for _ in range(10):
        assert controller.tick() is None
    controller.handle_key("s")
    assert controller.tick() is None
    assert controller.paused
    assert not robot.enable_calls
    assert not robot.send_calls


class FollowingRobot(FakeRobot):
    """Measured joints equal the last command plus a constant controller bias."""

    def __init__(self, bias=0.0):
        super().__init__()
        self.bias = np.full(6, float(bias))
        self.reference = np.zeros(6)
        for index, key in enumerate(JOINT_KEYS):
            self.observation[key] = float(self.reference[index] + self.bias[index])

    def get_reference_joint_positions(self):
        return self.reference.copy()

    def send_action(self, action):
        result = super().send_action(action)
        self._follow(np.asarray([action[key] for key in JOINT_KEYS], dtype=np.float64))
        return result

    def send_cartesian_action(self, pose_m_rad, gripper=None):
        # With CartesianAdapter's identity FK, joints equal the commanded pose.
        result = super().send_cartesian_action(pose_m_rad, gripper)
        self._follow(np.asarray(pose_m_rad, dtype=np.float64))
        return result

    def _follow(self, reference):
        self.reference = reference
        for index, key in enumerate(JOINT_KEYS):
            self.observation[key] = float(self.reference[index] + self.bias[index])


def test_activation_anchors_controller_reference_and_preserves_measurement():
    class BiasedRobot(FakeRobot):
        def get_reference_joint_positions(self):
            return np.full(6, -0.02)

    class HoldAdapter(FakeAdapter):
        def joint_action(self, action7, current_q6, *, max_joint_delta, previous_q, max_tracking_error):
            # Measured joints (0) corrected by the activation bias (+0.02)
            # coincide with the anchor: a resting controller has zero tracking error.
            np.testing.assert_allclose(current_q6, previous_q)
            np.testing.assert_allclose(current_q6, -0.02)
            assert max_joint_delta == pytest.approx(0.005)
            assert max_tracking_error == 0.05
            return dict(zip(JOINT_KEYS, previous_q, strict=True)) | {"gripper_0": 1.0}

    clock = FakeClock()
    cfg = make_config(dry_run=False, max_joint_velocity_rad_s=0.05, max_joint_tracking_error_rad=0.05)
    controller, robot, _ = make_controller(cfg, robot=BiasedRobot(), adapter=HoldAdapter(), clock=clock)
    controller.initialize_robot_safety()
    mark_network_ready(controller)
    assert controller.activate()
    np.testing.assert_allclose(controller._last_command_q, -0.02)
    assert not robot.send_calls
    install_chunk(controller, clock, np.zeros((2, 7)))
    command = controller.tick()
    assert command is not None
    np.testing.assert_allclose([command[key] for key in JOINT_KEYS], -0.02)


@pytest.mark.parametrize("reference", [np.full(6, 0.11), np.full(6, np.nan)])
def test_bad_controller_reference_fails_closed_on_activation(reference):
    robot = FakeRobot()
    robot.get_reference_joint_positions = lambda: reference
    controller, _, _ = make_controller(make_config(dry_run=False), robot=robot)
    controller.initialize_robot_safety()
    mark_network_ready(controller)
    assert not controller.activate()
    assert controller.paused
    assert controller._last_command_q is None
    assert not robot.send_calls
    assert len(robot.disable_calls) >= 2


def test_excessive_tracking_error_stops_without_resetting_anchor():
    clock = FakeClock()
    controller, robot, _ = make_controller(make_config(dry_run=False), clock=clock)
    controller.initialize_robot_safety()
    mark_network_ready(controller)
    assert controller.activate()
    install_chunk(controller, clock, np.zeros((2, 7)))
    robot.observation["joint_0"] = 0.11
    assert controller.tick() is None
    assert controller.paused
    assert not robot.send_calls
    assert "tracking error" in controller.last_pause_reason


def test_tracking_error_is_measured_against_the_activation_offset():
    # Controller bias 0.04 rad (measured above reference) with a 0.05 limit:
    # a raw comparison would leave 0.01 rad of headroom; the corrected one
    # tolerates the full limit and rejects beyond it.
    clock = FakeClock()
    robot = FollowingRobot(bias=0.04)
    cfg = make_config(dry_run=False, max_joint_tracking_error_rad=0.05)
    controller, _, _ = make_controller(cfg, robot=robot, clock=clock)
    controller.initialize_robot_safety()
    mark_network_ready(controller)
    assert controller.activate()
    np.testing.assert_allclose(controller._tracking_offset, 0.04)
    install_chunk(controller, clock, np.zeros((3, 7)))

    robot.observation["joint_0"] = 0.04 + 0.03  # corrected error 0.03 < limit
    assert controller.tick() is not None
    assert len(robot.send_calls) == 1

    robot.observation["joint_0"] = 0.04 + 0.06  # corrected error 0.06 > limit
    assert controller.tick() is None
    assert controller.paused
    assert "tracking error" in controller.last_pause_reason
    assert controller._tracking_offset is None


def test_ik_budget_starts_after_observation_capture():
    clock = FakeClock()

    class BlockingCameraRobot(FakeRobot):
        def get_observation(self):
            clock.value += 0.09  # two cameras waiting for fresh frames
            return super().get_observation()

    class SlowishAdapter(FakeAdapter):
        def joint_action(self, action7, current_q6, *, max_joint_delta, previous_q=None, max_tracking_error=None):
            clock.value += 0.05  # IK within one 0.1 s control period
            return super().joint_action(action7, current_q6, max_joint_delta=max_joint_delta)

    robot = BlockingCameraRobot()
    controller, _, _ = make_controller(
        make_config(dry_run=False), robot=robot, adapter=SlowishAdapter(), clock=clock
    )
    controller.initialize_robot_safety()
    mark_network_ready(controller)
    assert controller.activate()
    install_chunk(controller, clock, np.zeros((5, 7)))
    assert controller.tick() is not None
    assert len(robot.send_calls) == 1
    assert not controller.paused


def test_reference_anchor_does_not_relax_command_velocity_limit():
    clock = FakeClock()
    cfg = make_config(dry_run=False, max_joint_velocity_rad_s=0.05, max_joint_tracking_error_rad=0.05)
    controller, robot, _ = make_controller(cfg, clock=clock)
    controller.initialize_robot_safety()
    mark_network_ready(controller)
    assert controller.activate()
    install_chunk(controller, clock, [np.r_[np.full(6, 0.02), 1.0]])
    assert controller.tick() is None
    assert controller.paused
    assert "command-to-command" in controller.last_pause_reason
    assert not robot.send_calls


@pytest.mark.parametrize("dry_run", [False, True])
def test_run_client_starts_without_actuation_and_cleans_up(monkeypatch, dry_run):
    @dataclass
    class Config:
        model: str = "rb10"
        first_state_timeout_s: float = 0.1
        control_rate_hz: float = 10.0
        inference_safe_start: bool = False

    class Adapter(FakeAdapter):
        def __init__(self, *args, **kwargs):
            super().__init__()

    class Keyboard:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def poll(self):
            return "q"

    events = []
    robot = FakeRobot()
    robot.connect = lambda: events.append("connect")
    robot.disconnect = lambda: events.append("disconnect")
    def factory(config):
        assert config.inference_safe_start
        return robot

    monkeypatch.setattr(MODULE, "_Keyboard", Keyboard)
    monkeypatch.setattr(MODULE, "_load_policy_api", lambda: (
        Adapter, lambda *args, **kwargs: IdleClient(), lambda value: value["actions"],
    ))
    MODULE.run_client(
        make_config(robot=Config(), dry_run=dry_run, move_to_ready_on_start=False),
        robot_factory=factory,
    )
    assert events == ["connect", "disconnect"]
    assert not robot.enable_calls
    assert robot.disable_calls
    assert not robot.send_calls


def test_run_client_starts_ready_move_in_real_mode_and_q_aborts_it(monkeypatch):
    @dataclass
    class Config:
        model: str = "rb10"
        first_state_timeout_s: float = 0.1
        control_rate_hz: float = 10.0
        inference_safe_start: bool = False

    class Adapter(FakeAdapter):
        def __init__(self, *args, **kwargs):
            super().__init__()

    class Keyboard:
        def __init__(self):
            self.polls = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def poll(self):
            self.polls += 1
            return None if self.polls < 3 else "q"

    robot = FollowingRobot()
    robot.connect = lambda: None
    robot.disconnect = lambda: None
    monkeypatch.setattr(MODULE, "_Keyboard", Keyboard)
    monkeypatch.setattr(MODULE, "_load_policy_api", lambda: (
        Adapter, lambda *args, **kwargs: IdleClient(), lambda value: value["actions"],
    ))
    MODULE.run_client(make_config(robot=Config(), dry_run=False), robot_factory=lambda config: robot)
    # Startup enabled the gate and streamed ready commands (gripper open) until q.
    assert robot.enable_calls
    assert 1 <= len(robot.send_calls) <= 2
    assert all(action["gripper_0"] == 1.0 for _, action in robot.send_calls)
    assert robot.disable_calls[-1]


def ready_config(**overrides):
    values = {
        "dry_run": False,
        "ready_pose_rad": [0.3, -0.2, 0.1, 0.0, 0.25, -0.15],
        "ready_move_duration_s": 1.0,
        "ready_settle_s": 0.2,
        "ready_max_joint_speed_rad_s": 1.0,
    }
    values.update(overrides)
    return make_config(**values)


def run_ready_move(controller, robot, clock, *, ticks=20):
    commands = []
    for _ in range(ticks):
        clock.value += 0.1
        command = controller.tick()
        if command is not None:
            commands.append(np.asarray([command[key] for key in JOINT_KEYS]))
    return commands


def test_ready_move_profiles_to_target_then_pauses_until_f():
    clock = FakeClock()
    robot = FollowingRobot(bias=0.02)
    controller, _, _ = make_controller(ready_config(), robot=robot, clock=clock)
    controller.initialize_robot_safety()
    assert controller.start_ready_move("startup")
    assert robot.enable_calls
    assert controller.paused  # inference stays paused while the arm moves

    commands = run_ready_move(controller, robot, clock)
    target = np.asarray(controller.config.ready_pose_rad)
    assert commands, "ready move sent no commands"
    np.testing.assert_allclose(commands[-1], target, atol=1e-9)
    steps = np.diff(np.vstack([np.zeros(6), *commands]), axis=0)
    assert np.max(np.abs(steps)) <= controller.config.max_joint_delta + 1e-12
    # The gripper opens (dataset 1 = open) with every ready command.
    assert all(action["gripper_0"] == 1.0 for _, action in robot.send_calls)
    assert controller._ready_move is None
    assert controller.last_pause_reason.startswith("ready pose reached")
    assert robot.disable_calls[-1]

    # The ready move never auto-starts inference; f anchors from the new reference.
    mark_network_ready(controller)
    assert controller.activate()
    np.testing.assert_allclose(controller._last_command_q, target)
    np.testing.assert_allclose(controller._tracking_offset, 0.02)


def test_ready_move_speed_cap_lengthens_long_moves_and_gripper_can_stay_untouched():
    clock = FakeClock()
    robot = FollowingRobot()
    cfg = ready_config(ready_max_joint_speed_rad_s=0.15, ready_gripper_open=False)
    controller, _, _ = make_controller(cfg, robot=robot, clock=clock)
    controller.initialize_robot_safety()
    assert controller.start_ready_move("operator stop")
    # 0.3 rad furthest joint at a 0.15 rad/s cap: pi * 0.3 / (2 * 0.15) = 3.14 s > 1 s minimum.
    assert controller._ready_move.duration == pytest.approx(np.pi * 0.3 / (2 * 0.15))
    clock.value += 0.1
    command = controller.tick()
    assert command is not None and "gripper_0" not in command
    assert all("gripper_0" not in action for _, action in robot.send_calls)


def test_s_returns_to_ready_from_inference_and_s_again_aborts():
    clock = FakeClock()
    robot = FollowingRobot()
    controller, _, _ = make_controller(ready_config(), robot=robot, clock=clock)
    controller.initialize_robot_safety()
    mark_network_ready(controller)
    assert controller.activate()
    install_chunk(controller, clock, np.zeros((5, 7)))
    assert controller.tick() is not None

    controller.handle_key("s")
    assert controller._ready_move is not None
    assert controller.paused
    sent_before = len(robot.send_calls)
    clock.value += 0.1
    assert controller.tick() is not None
    assert len(robot.send_calls) == sent_before + 1

    # f is ignored mid-move; s aborts without chaining another move.
    assert not controller.activate()
    controller.handle_key("s")
    assert controller._ready_move is None
    assert controller.paused
    sent_after = len(robot.send_calls)
    for _ in range(5):
        clock.value += 0.1
        assert controller.tick() is None
    assert len(robot.send_calls) == sent_after
    assert robot.disable_calls[-1]


class CartesianAdapter(FakeAdapter):
    """FK that maps joints to a pose directly: xyz = q[:3], euler = q[3:]."""

    def __init__(self):
        super().__init__()
        self.kinematics = types.SimpleNamespace(forward=self.forward)

    @staticmethod
    def forward(q):
        q = np.asarray(q, dtype=np.float64)
        assert q.shape == (6,)
        if not np.all(np.isfinite(q)):
            raise ValueError("invalid joint position")
        return q.copy()

    def joint_action(self, *args, **kwargs):  # pragma: no cover - must not be used
        raise AssertionError("cartesian mode must not call joint IK")


def cartesian_config(**overrides):
    values = {
        "dry_run": False,
        "servo_mode": "cartesian",
        "max_tcp_speed_m_s": 0.1,  # 0.01 m per 0.1 s tick
        "max_tcp_angular_speed_rad_s": 1.0,
        "max_tcp_tracking_error_m": 0.02,
        "max_tcp_tracking_error_rad": 0.1,
        "move_to_ready_on_stop": False,
    }
    values.update(overrides)
    return make_config(**values)


def test_cartesian_mode_streams_bounded_tcp_waypoints_without_client_ik():
    clock = FakeClock()
    robot = FollowingRobot(bias=0.01)
    controller, _, _ = make_controller(cartesian_config(), robot=robot, adapter=CartesianAdapter(), clock=clock)
    controller.initialize_robot_safety()
    mark_network_ready(controller)
    assert controller.activate()
    np.testing.assert_allclose(controller._last_command_pose, np.zeros(6))

    target = np.array([0.03, 0.04, 0.0, 0.0, 0.0, 0.0, 0.25])
    install_chunk(controller, clock, [target] * 20)
    poses = []
    for _ in range(8):
        command = controller.tick()
        assert command is not None, controller.last_pause_reason
        poses.append(np.array([command[k] for k in ("x", "y", "z", "rx", "ry", "rz")]))
        assert command["gripper_0"] == 0.25
        clock.value += 0.1
    poses = np.vstack(poses)
    steps = np.diff(np.vstack([np.zeros(6), poses]), axis=0)
    # Straight line towards the target at the capped speed, then hold.
    assert np.all(np.linalg.norm(steps[:, :3], axis=1) <= 0.01 + 1e-9)
    directions = steps[:4, :3] / np.linalg.norm(steps[:4, :3], axis=1, keepdims=True)
    np.testing.assert_allclose(directions, np.tile([0.6, 0.8, 0.0], (4, 1)), atol=1e-9)
    np.testing.assert_allclose(poses[-1, :3], target[:3], atol=1e-9)
    assert all("pose" in action for _, action in robot.send_calls)
    assert not controller.paused


def test_cartesian_mode_faults_on_tcp_tracking_loss_and_never_sends_in_dry_run():
    clock = FakeClock()
    robot = FollowingRobot()
    controller, _, _ = make_controller(cartesian_config(), robot=robot, adapter=CartesianAdapter(), clock=clock)
    controller.initialize_robot_safety()
    mark_network_ready(controller)
    assert controller.activate()
    install_chunk(controller, clock, np.zeros((5, 7)))
    assert controller.tick() is not None
    robot.observation["joint_0"] += 0.05  # TCP pushed 5 cm off its command
    clock.value += 0.1
    assert controller.tick() is None
    assert controller.paused
    assert "TCP" in controller.last_pause_reason
    assert controller._last_command_pose is None

    clock = FakeClock()
    robot = FollowingRobot()
    controller, _, _ = make_controller(
        cartesian_config(dry_run=True), robot=robot, adapter=CartesianAdapter(), clock=clock
    )
    controller.initialize_robot_safety()
    mark_network_ready(controller)
    assert controller.activate()
    install_chunk(controller, clock, [[0.01, 0, 0, 0, 0, 0, 1]])
    assert controller.tick() is not None
    assert not robot.send_calls


def test_ready_move_is_skipped_in_dry_run_and_faults_on_tracking_loss():
    clock = FakeClock()
    controller, robot, _ = make_controller(ready_config(dry_run=True), clock=clock)
    controller.initialize_robot_safety()
    assert not controller.start_ready_move("startup")
    assert controller._ready_move is None
    assert not robot.enable_calls and not robot.send_calls

    clock = FakeClock()
    robot = FollowingRobot()
    controller, _, _ = make_controller(ready_config(), robot=robot, clock=clock)
    controller.initialize_robot_safety()
    assert controller.start_ready_move("operator stop")
    clock.value += 0.1
    assert controller.tick() is not None
    robot.observation["joint_2"] += 0.5  # arm stalled / pushed away from its command
    clock.value += 0.1
    assert controller.tick() is None
    assert controller._ready_move is None
    assert controller.paused
    assert "tracking error" in controller.last_pause_reason
    assert robot.disable_calls[-1]


def test_paused_and_dry_run_never_call_send_action_including_gripper():
    clock = FakeClock()
    controller, robot, _ = make_controller(clock=clock)
    controller.initialize_robot_safety()

    assert controller.paused
    assert controller.tick() is None
    assert robot.get_calls == []
    assert robot.send_calls == []

    mark_network_ready(controller)
    assert controller.handle_key("f")
    install_chunk(controller, clock, [[0, 0, 0, 0, 0, 0, 0.25]])
    command = controller.tick()
    assert command["gripper_0"] == pytest.approx(0.25)
    assert robot.send_calls == []
    assert robot.enable_calls == []

    controller.handle_key("s")
    assert controller.paused
    assert controller.tick() is None
    assert robot.send_calls == []


@pytest.mark.parametrize("interval,expected", [(0.0, 0), (1.0, 1)])
def test_client_logs_selected_action_and_command_fk_delta(caplog, interval, expected):
    clock = FakeClock()
    controller, robot, adapter = make_controller(make_config(action_log_interval_s=interval), clock=clock)
    adapter.kinematics.forward = lambda q: np.r_[q[:3], np.zeros(3)]
    controller.initialize_robot_safety()
    mark_network_ready(controller)
    controller.activate()
    install_chunk(controller, clock, [[0.01, -0.02, 0.03, 0, 0, 0, 1]])
    with caplog.at_level("INFO"):
        controller.tick()
        controller.tick()
    messages = [record.message for record in caplog.records if "command_fk_delta=" in record.message]
    assert len(messages) == expected
    if expected:
        assert "EE action[0] frame=link0 unit=m dry_run=True" in messages[0]
        assert "target_delta=[0.01, -0.02, 0.03]" in messages[0]
        assert "command_fk_delta=[0.01, -0.02, 0.03]" in messages[0]
    assert robot.send_calls == []

def test_generation_rejects_inflight_response_after_pause():
    clock = FakeClock()
    controller, robot, _ = make_controller(clock=clock)
    controller.initialize_robot_safety()
    mark_network_ready(controller)
    controller.activate()
    old_generation = controller.generation

    controller.pause("test pause")
    install_chunk(
        controller,
        clock,
        [[0, 0, 0, 0, 0, 0, 0]],
        generation=old_generation,
    )
    assert controller._action_chunk is None
    assert controller.tick() is None
    assert robot.send_calls == []


def test_expired_chunk_is_discarded_by_monotonic_horizon_index():
    clock = FakeClock(5.0)
    controller, robot, _ = make_controller(clock=clock)
    controller.initialize_robot_safety()
    mark_network_ready(controller)
    controller.activate()
    actions = np.zeros((2, 7))
    install_chunk(controller, clock, actions, observed_at=4.79)

    assert controller.tick() is None
    assert controller._action_chunk is None
    assert controller.paused
    assert controller.last_pause_reason == "action chunk horizon expired"
    assert robot.send_calls == []


def test_command_to_command_jump_faults_and_requires_explicit_rearm():
    clock = FakeClock()
    config = make_config(dry_run=False)
    controller, robot, _ = make_controller(config, clock=clock)
    controller.initialize_robot_safety()
    mark_network_ready(controller)
    controller.activate()

    install_chunk(controller, clock, [[0, 0, 0, 0, 0, 0, 1]])
    controller.tick()
    assert len(robot.send_calls) == 1

    install_chunk(controller, clock, [[0.2, 0, 0, 0, 0, 0, 0]])
    assert controller.tick() is None
    assert controller.paused
    assert len(robot.send_calls) == 1
    assert robot.disable_calls

    assert controller.tick() is None
    assert len(robot.send_calls) == 1
    assert controller.activate()
    assert not controller.paused


def test_timeout_pauses_reconnects_and_does_not_resume_without_f():
    class TimeoutClient(IdleClient):
        def __init__(self):
            super().__init__()
            self.called = threading.Event()

        def get_action(self, observation):
            self.called.set()
            raise TimeoutError("scripted timeout")

    clients = deque([TimeoutClient(), IdleClient()])
    created_threads = []

    def factory(address, timeout):
        created_threads.append(threading.get_ident())
        if clients:
            return clients.popleft()
        return IdleClient()

    controller, robot, adapter = make_controller(factory=factory)
    main_thread = threading.get_ident()
    controller.initialize_robot_safety()
    controller.start_network_worker()
    try:
        wait_for(lambda: controller.network_ready)
        assert controller.activate()
        controller.tick()

        wait_for(lambda: controller.paused and len(created_threads) >= 2 and controller.network_ready)
        controller.tick()  # services the pending main-thread servo disable
        assert controller.last_pause_reason.startswith("network/policy outage")
        assert robot.send_calls == []
        assert all(thread_id != main_thread for thread_id in created_threads)
        assert all(thread_id == main_thread for thread_id, _ in adapter.build_calls)
        assert all(thread_id == main_thread for thread_id in robot.get_calls)
        assert all(thread_id == main_thread for thread_id in robot.disable_calls)

        get_count = len(robot.get_calls)
        controller.tick()
        assert len(robot.get_calls) == get_count
        assert controller.activate()
        controller.tick()
        wait_for(lambda: controller._action_chunk is not None)
    finally:
        controller.pause("test cleanup")
        controller.stop_network_worker()


def test_describe_is_mandatory_before_network_becomes_ready():
    attempts = []

    class NoDescribeClient:
        def close(self):
            return None

    def factory(address, timeout):
        attempts.append(threading.get_ident())
        return NoDescribeClient()

    controller, _, _ = make_controller(factory=factory)
    controller.start_network_worker()
    try:
        wait_for(lambda: len(attempts) >= 2)
        assert not controller.network_ready
        assert controller.paused
    finally:
        controller.stop_network_worker()


def test_camera_buffer_is_snapshotted_before_network_handoff():
    received = []
    received_event = threading.Event()

    class CapturingClient(IdleClient):
        def get_action(self, observation):
            received.append(observation)
            received_event.set()
            return {"actions": np.zeros((1, 7))}

    robot = FakeRobot()
    controller, _, _ = make_controller(robot=robot, factory=lambda address, timeout: CapturingClient())
    controller.initialize_robot_safety()
    controller.start_network_worker()
    try:
        wait_for(lambda: controller.network_ready)
        controller.activate()
        controller.tick()
        robot.observation["wrist"][:] = 255
        assert received_event.wait(1.0)
        assert np.all(received[0]["raw"]["wrist"] == 0)
    finally:
        controller.pause("test cleanup")
        controller.stop_network_worker()


def test_action_that_becomes_obsolete_during_ik_is_never_sent():
    clock = FakeClock()

    class SlowAdapter(FakeAdapter):
        def joint_action(self, action7, current_q6, *, max_joint_delta, previous_q=None, max_tracking_error=None):
            clock.value += 0.101
            return super().joint_action(action7, current_q6, max_joint_delta=max_joint_delta)

    config = make_config(dry_run=False)
    controller, robot, _ = make_controller(config, adapter=SlowAdapter(), clock=clock)
    controller.initialize_robot_safety()
    mark_network_ready(controller)
    controller.activate()
    install_chunk(controller, clock, [[0, 0, 0, 0, 0, 0, 1]])

    assert controller.tick() is None
    assert robot.send_calls == []
    assert controller.paused
    assert controller.last_pause_reason == "action became obsolete during IK"


def test_ik_may_cross_action_index_boundary_within_tick_budget():
    clock = FakeClock(10.099)

    class FiveMillisecondAdapter(FakeAdapter):
        def joint_action(self, action7, current_q6, *, max_joint_delta, previous_q=None, max_tracking_error=None):
            clock.value += 0.005
            return super().joint_action(action7, current_q6, max_joint_delta=max_joint_delta)

    config = make_config(dry_run=False)
    controller, robot, _ = make_controller(config, adapter=FiveMillisecondAdapter(), clock=clock)
    controller.initialize_robot_safety()
    mark_network_ready(controller)
    controller.activate()
    install_chunk(
        controller,
        clock,
        np.zeros((3, 7)),
        observed_at=10.0,
    )

    command = controller.tick()
    assert command is not None
    assert len(robot.send_calls) == 1
    assert not controller.paused


def test_send_action_exception_faults_and_closes_servo_gate():
    class FailingRobot(FakeRobot):
        def send_action(self, action):
            self.send_calls.append((threading.get_ident(), dict(action)))
            raise RuntimeError("scripted ServoJ rejection")

    clock = FakeClock()
    config = make_config(dry_run=False)
    robot = FailingRobot()
    controller, _, _ = make_controller(config, robot=robot, clock=clock)
    controller.initialize_robot_safety()
    mark_network_ready(controller)
    controller.activate()
    install_chunk(controller, clock, [[0, 0, 0, 0, 0, 0, 1]])

    assert controller.tick() is None
    assert controller.paused
    assert controller.last_pause_reason.startswith("robot send_action fault")
    assert len(robot.send_calls) == 1
    assert robot.disable_calls[-1] == threading.get_ident()


def test_failed_servo_enable_never_marks_controller_active():
    class GateFailRobot(FakeRobot):
        def enable_servo_commands(self):
            super().enable_servo_commands()
            raise RuntimeError("scripted gate failure")

    config = make_config(dry_run=False)
    robot = GateFailRobot()
    controller, _, _ = make_controller(config, robot=robot)
    controller.initialize_robot_safety()
    mark_network_ready(controller)

    assert not controller.activate()
    assert controller.paused
    assert controller.last_pause_reason.startswith("servo gate activation fault")
    assert robot.send_calls == []


def test_real_activation_opens_generation_only_after_enable_succeeds():
    class InspectingRobot(FakeRobot):
        controller = None
        state_during_enable = None

        def enable_servo_commands(self):
            self.state_during_enable = (self.controller.generation, self.controller.paused)
            super().enable_servo_commands()

    config = make_config(dry_run=False)
    robot = InspectingRobot()
    controller, _, _ = make_controller(config, robot=robot)
    robot.controller = controller
    controller.initialize_robot_safety()
    mark_network_ready(controller)
    generation_before = controller.generation

    assert controller.activate()
    assert robot.state_during_enable == (generation_before, True)
    assert controller.generation == generation_before + 1
    assert not controller.paused


def test_timeout_and_reconnect_during_enable_cannot_activate_generation():
    class RacingRobot(FakeRobot):
        controller = None

        def enable_servo_commands(self):
            super().enable_servo_commands()
            self.controller._pause_from_worker("network/policy outage: timeout during preparation")
            with self.controller._state_lock:
                self.controller._network_ready = True

    config = make_config(dry_run=False)
    robot = RacingRobot()
    controller, _, _ = make_controller(config, robot=robot)
    robot.controller = controller
    controller.initialize_robot_safety()
    mark_network_ready(controller)
    generation_before = controller.generation

    assert not controller.activate()
    assert controller.paused
    assert controller.network_ready
    assert controller.generation == generation_before + 1
    assert controller.last_pause_reason == "network/policy outage: timeout during preparation"
    assert len(robot.enable_calls) == 1
    assert robot.disable_calls[-1] == threading.get_ident()


def test_runtime_config_forces_safe_start_without_mutating_cli_config():
    @dataclass
    class SafeStartRobotConfig:
        model: str = "rb10"
        first_state_timeout_s: float = 0.1
        control_rate_hz: float = 10.0
        inference_safe_start: bool = False

    parsed = make_config(robot=SafeStartRobotConfig())
    effective = _with_inference_safe_start(parsed)

    assert parsed.robot.inference_safe_start is False
    assert effective is not parsed
    assert effective.robot is not parsed.robot
    assert effective.robot.inference_safe_start is True


def test_main_registers_plugins_before_invoking_draccus_parser(monkeypatch):
    calls = []
    log_options = {}

    def configure_logging(**kwargs):
        log_options.update(kwargs)
        calls.append("logging")

    monkeypatch.setattr(MODULE.logging, "basicConfig", configure_logging)
    lerobot = types.ModuleType("lerobot")
    cameras = types.ModuleType("lerobot.cameras")
    opencv = types.ModuleType("lerobot.cameras.opencv")
    realsense = types.ModuleType("lerobot.cameras.realsense")
    utils = types.ModuleType("lerobot.utils")
    import_utils = types.ModuleType("lerobot.utils.import_utils")
    robot_plugin = types.ModuleType("lerobot_robot_rb")
    rb10 = types.ModuleType("lerobot_robot_rb.rb10")
    opencv.OpenCVCameraConfig = object
    realsense.RealSenseCameraConfig = object
    import_utils.register_third_party_plugins = lambda: calls.append("register")
    rb10.Rb10Config = object
    for name, module in {
        "lerobot": lerobot,
        "lerobot.cameras": cameras,
        "lerobot.cameras.opencv": opencv,
        "lerobot.cameras.realsense": realsense,
        "lerobot.utils": utils,
        "lerobot.utils.import_utils": import_utils,
        "lerobot_robot_rb": robot_plugin,
        "lerobot_robot_rb.rb10": rb10,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(MODULE, "async_ee_client", lambda: calls.append("parse"))

    MODULE.main()
    assert calls == ["logging", "register", "parse"]
    assert log_options["level"] == MODULE.logging.INFO
    assert log_options["force"] is True


def test_latest_observation_slot_drops_intermediate_frames():
    first_started = threading.Event()
    release_first = threading.Event()
    seen = []

    class BlockingClient(IdleClient):
        def get_action(self, observation):
            seen.append(observation["raw"]["joint_0"])
            if len(seen) == 1:
                first_started.set()
                assert release_first.wait(1.0)
            return {"actions": np.zeros((1, 7))}

    robot = FakeRobot()
    controller, _, _ = make_controller(robot=robot, factory=lambda address, timeout: BlockingClient())
    controller.initialize_robot_safety()
    controller.start_network_worker()
    try:
        wait_for(lambda: controller.network_ready)
        controller.activate()
        robot.observation["joint_0"] = 1.0
        controller.tick()
        assert first_started.wait(1.0)

        robot.observation["joint_0"] = 2.0
        controller.tick()
        robot.observation["joint_0"] = 3.0
        controller.tick()
        release_first.set()
        wait_for(lambda: len(seen) >= 2)
        assert seen[:2] == [1.0, 3.0]
    finally:
        release_first.set()
        controller.pause("test cleanup")
        controller.stop_network_worker()
