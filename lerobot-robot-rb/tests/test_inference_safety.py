"""Hardware-free tests for the opt-in inference safety gate."""

from __future__ import annotations

import importlib
import math
import sys
import types
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).parents[1]


def _install_lerobot_stubs() -> None:
    lerobot = types.ModuleType("lerobot")
    lerobot.__path__ = []
    sys.modules["lerobot"] = lerobot

    cameras = types.ModuleType("lerobot.cameras")

    class CameraConfig:
        pass

    class Cv2Rotation(Enum):
        NO_ROTATION = 0
        ROTATE_90 = 1

    cameras.CameraConfig = CameraConfig
    cameras.Cv2Rotation = Cv2Rotation
    cameras.make_cameras_from_configs = lambda configs: {}
    sys.modules["lerobot.cameras"] = cameras

    realsense = types.ModuleType(
        "lerobot.cameras.realsense.configuration_realsense"
    )

    @dataclass
    class RealSenseCameraConfig(CameraConfig):
        serial_number_or_name: str
        fps: int
        width: int
        height: int
        rotation: Cv2Rotation

    realsense.RealSenseCameraConfig = RealSenseCameraConfig
    sys.modules[
        "lerobot.cameras.realsense.configuration_realsense"
    ] = realsense

    robot_config = types.ModuleType("lerobot.robots.config")

    class RobotConfig:
        def __post_init__(self) -> None:
            pass

    robot_config.RobotConfig = RobotConfig
    sys.modules["lerobot.robots.config"] = robot_config

    robot_module = types.ModuleType("lerobot.robots.robot")

    class Robot:
        def __init__(self, config: Any) -> None:
            self.config = config

    robot_module.Robot = Robot
    sys.modules["lerobot.robots.robot"] = robot_module

    errors = types.ModuleType("lerobot.utils.errors")

    class DeviceAlreadyConnectedError(RuntimeError):
        pass

    class DeviceNotConnectedError(RuntimeError):
        pass

    errors.DeviceAlreadyConnectedError = DeviceAlreadyConnectedError
    errors.DeviceNotConnectedError = DeviceNotConnectedError
    sys.modules["lerobot.utils.errors"] = errors


_install_lerobot_stubs()
package = types.ModuleType("lerobot_robot_rb")
package.__path__ = [str(PROJECT_ROOT / "lerobot_robot_rb")]
sys.modules["lerobot_robot_rb"] = package

config_module = importlib.import_module("lerobot_robot_rb.config_rb")
rb_module = importlib.import_module("lerobot_robot_rb.rb_cobot")

RbCobotConfig = config_module.RbCobotConfig
RbCobot = rb_module.RbCobot
GRIPPER_NAME = rb_module.GRIPPER_NAME
JOINT_NAMES = rb_module.JOINT_NAMES


class _FakeCobot:
    instances: list[_FakeCobot] = []

    def __init__(self, **kwargs: Any) -> None:
        self.events: list[str] = []
        self.servo_commands: list[list[float]] = []
        self.instances.append(self)

    def ConnectToCB(self) -> bool:
        self.events.append("connect")
        return True

    def wait_for_first_state(self, timeout_s: float) -> bool:
        self.events.append("first_state")
        return True

    def set_ff_gain_off(self) -> bool:
        self.events.append("ff_off")
        return True

    def set_joint_space_impedance(self) -> bool:
        self.events.append("impedance")
        return True

    def SetBaseSpeed(self, speed: float) -> bool:
        self.events.append("speed")
        return True

    def GetLatestState(self, timeout_s: float) -> Any:
        return types.SimpleNamespace(program_mode=0)

    def ServoJ(self, *, joints_deg: Any, **kwargs: Any) -> bool:
        self.events.append("servo")
        self.servo_commands.append(list(joints_deg))
        return True

    def DisConnectToCB(self) -> bool:
        self.events.append("disconnect")
        return True


class _FakeGripper:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.positions: list[float] = []
        self.connected = False

    def connect(self) -> None:
        self.events.append("gripper_connect")
        self.connected = True

    def disconnect(self) -> None:
        if self.connected:
            self.events.append("gripper_disconnect")
        self.connected = False

    def set_position(self, normalized: float) -> None:
        assert self.connected
        self.events.append("gripper_action")
        self.positions.append(normalized)

    def get_position(self) -> float:
        # Before deferred initialization this is the driver's hypothetical
        # cached open state, not a hardware measurement.
        return 0.0


@pytest.fixture
def robot_factory(monkeypatch: pytest.MonkeyPatch):
    _FakeCobot.instances.clear()
    grippers: list[_FakeGripper] = []

    monkeypatch.setattr(rb_module, "Cobot", _FakeCobot)
    monkeypatch.setattr(rb_module.time, "sleep", lambda seconds: None)

    def make_fake_gripper(config: Any, cobot: _FakeCobot):
        if config.gripper_type == "none":
            return None
        gripper = _FakeGripper(cobot.events)
        grippers.append(gripper)
        return gripper

    monkeypatch.setattr(rb_module, "make_gripper", make_fake_gripper)

    def make(*, safe: bool = True, gripper: bool = True):
        config = RbCobotConfig(
            cameras={},
            gripper_type="rby1_dynamixel" if gripper else "none",
            inference_safe_start=safe,
        )
        robot = RbCobot(config)
        robot.connect()
        return robot, _FakeCobot.instances[-1], (
            grippers[-1] if gripper else None
        )

    return make


def _action(
    joints: list[float] | None = None,
    gripper: float = 1.0,
) -> dict[str, float]:
    values = joints if joints is not None else [0.0] * len(JOINT_NAMES)
    action = dict(zip(JOINT_NAMES, values, strict=True))
    action[GRIPPER_NAME] = gripper
    return action


def test_safe_connect_is_observation_only(robot_factory) -> None:
    robot, cobot, gripper = robot_factory()

    assert not robot.servo_enabled
    assert gripper is not None and not gripper.connected
    assert cobot.events == ["connect", "first_state"]


def test_first_enable_prepares_once_before_opening_gate(robot_factory) -> None:
    robot, cobot, gripper = robot_factory()

    robot.enable_servo_commands()
    assert robot.servo_enabled
    assert gripper is not None and gripper.connected
    assert cobot.events == [
        "connect",
        "first_state",
        "gripper_connect",
        "ff_off",
        "impedance",
    ]

    robot.disable_servo_commands()
    robot.enable_servo_commands()
    assert cobot.events.count("gripper_connect") == 1
    assert cobot.events.count("ff_off") == 1
    assert cobot.events.count("impedance") == 1


def test_failed_preparation_keeps_servo_gate_closed(robot_factory) -> None:
    robot, cobot, _ = robot_factory()

    def fail_impedance() -> bool:
        cobot.events.append("impedance_failed")
        raise RuntimeError("fake impedance failure")

    cobot.set_joint_space_impedance = fail_impedance

    with pytest.raises(RuntimeError, match="fake impedance failure"):
        robot.set_servo_enabled(True)

    assert not robot.servo_enabled
    assert "servo" not in cobot.events


def test_safe_mode_gates_arm_and_gripper_during_pause(robot_factory) -> None:
    robot, cobot, gripper = robot_factory()

    robot.send_action(_action(gripper=0.0))
    assert "servo" not in cobot.events
    assert gripper is not None and gripper.positions == []

    robot.enable_servo_commands()
    robot.send_action(_action(gripper=0.0))
    assert len(cobot.servo_commands) == 1
    assert gripper.positions == [1.0]

    robot.disable_servo_commands()
    robot.send_action(_action(gripper=1.0))
    assert len(cobot.servo_commands) == 1
    assert gripper.positions == [1.0]


@pytest.mark.parametrize(
    "action",
    [
        _action(gripper=math.nan),
        _action(joints=[3.141, 0.0, 0.0, 0.0, 0.0, 0.0]),
        _action(joints=[0.0, 0.0, math.radians(165.1), 0.0, 0.0, 0.0]),
    ],
)
def test_bad_action_is_rejected_before_arm_and_gripper(
    robot_factory,
    action: dict[str, float],
) -> None:
    robot, cobot, gripper = robot_factory()
    robot.enable_servo_commands()
    cobot.servo_commands.clear()
    assert gripper is not None
    gripper.positions.clear()

    with pytest.raises(ValueError):
        robot.send_action(action)

    assert cobot.servo_commands == []
    assert gripper.positions == []


def test_legacy_connect_and_gripper_gate_behavior_is_unchanged(
    robot_factory,
) -> None:
    robot, cobot, gripper = robot_factory(safe=False)

    assert robot.servo_enabled
    assert gripper is not None and gripper.connected
    assert cobot.events == [
        "connect",
        "first_state",
        "ff_off",
        "impedance",
        "gripper_connect",
    ]

    robot.disable_servo_commands()
    robot.send_action(_action(gripper=0.0))
    assert cobot.servo_commands == []
    assert gripper.positions == [1.0]
