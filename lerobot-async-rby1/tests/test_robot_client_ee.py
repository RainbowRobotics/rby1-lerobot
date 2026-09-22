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

    def send_action(self, action):
        self.send_calls.append((threading.get_ident(), dict(action)))
        return action

    def enable_servo_commands(self):
        self.enable_calls.append(threading.get_ident())

    def disable_servo_commands(self):
        self.disable_calls.append(threading.get_ident())


class FakeAdapter:
    def __init__(self):
        self.build_calls: list[tuple[int, float]] = []

    def validate_robot(self, robot):
        return None

    def build_observation(self, raw, task):
        self.build_calls.append((threading.get_ident(), raw["joint_0"]))
        return {"raw": raw, "task": task}

    def joint_action(self, action7, current_q6, *, max_joint_delta):
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

    _validate_descriptor(dict(SERVER_DESCRIPTOR))
    wrong = dict(SERVER_DESCRIPTOR, euler_convention="RxRyRz")
    with pytest.raises(ValueError, match="descriptor"):
        _validate_descriptor(wrong)


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
        def joint_action(self, action7, current_q6, *, max_joint_delta):
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
        def joint_action(self, action7, current_q6, *, max_joint_delta):
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
    assert calls == ["register", "parse"]


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
