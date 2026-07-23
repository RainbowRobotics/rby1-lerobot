"""Configuration for Rainbow Robotics RB-Series LeRobot adapters.

The Robot adapter uses the same low-level TCP protocol and ServoJ command
parameters as the original hardware-tested RB10E teleoperation code.

Responsibilities in this configuration are intentionally narrow:

- RB control-box TCP connection
- Latest-state reception
- 35 Hz joint ServoJ command parameters
- Optional camera and gripper configuration

The Robot adapter does not automatically:

- switch Real/Simulation mode
- initialize or power the robot
- move to a ready pose
- reset the robot between episodes
- interpolate commands in a separate 200 Hz worker

Those operations remain under the control of the teaching pendant or the
teleoperator state machine.
"""

from __future__ import annotations

import ipaddress
import math
from dataclasses import dataclass, field

from lerobot.robots.config import RobotConfig

from lerobot.cameras import CameraConfig, Cv2Rotation
from lerobot.cameras.realsense.configuration_realsense import (
    RealSenseCameraConfig,
)

from .models import MODEL_SPECS



def _default_cameras() -> dict[str, CameraConfig]:
    return {
        "front": RealSenseCameraConfig(
            serial_number_or_name="409122274689",
            fps=30,
            width=480,
            height=640,
            rotation=Cv2Rotation.ROTATE_90,
        ),
    }


@dataclass
class RbCobotConfig(RobotConfig):
    """Shared configuration for RB-Series collaborative robots.

    This base configuration is not registered directly with the LeRobot CLI.
    Model-specific shims such as ``Rb10Config`` set ``model="rb10"`` and
    register the public CLI type ``--robot.type=rb10``.

    LeRobot boundary units
    ----------------------
    Joint observations and actions use radians.

    Control-box boundary units
    --------------------------
    The low-level Cobot client receives and transmits joint angles in
    degrees. Conversion between radians and degrees is handled by
    ``RbCobot``.
    """

    # Key in MODEL_SPECS. Model-specific shims override this value.
    model: str = "rb10"

    # RB control-box network configuration.
    ip: str = "10.0.2.7"
    command_port: int = 5000
    data_port: int = 5001

    # Socket timeout used by the low-level command/data TCP sockets.
    socket_timeout_s: float = 1.0

    # Maximum wait for the first valid 580-byte state packet during connect.
    first_state_timeout_s: float = 2.0

    # State request frequency for the background data receiver.
    #
    # This is independent from the LeRobot control frequency. The receiver
    # keeps only the latest state, while get_observation() reads that latest
    # state without accumulating a queue.
    data_request_hz: float = 500.0

    # The original RB10E VR controller operates at 35 Hz.
    #
    # lerobot-teleoperate should also be executed with:
    #     --fps=35
    control_rate_hz: float = 35.0

    # Original move_servo_j parameters:
    #
    #     t1    = 1 / control_rate_hz
    #     t2    = servo_hold_multiplier * t1
    #     gain  = 1.0
    #     alpha = 0.1
    #
    # t1 and t2 are exposed below as calculated properties so they cannot
    # accidentally become inconsistent with the selected control rate.
    servo_hold_multiplier: float = 3.0
    servo_gain: float = 1.0
    servo_alpha: float = 0.1

    # Optional global speed-bar update.
    #
    # Disabled by default so connecting LeRobot does not silently alter the
    # speed selected on the teaching pendant. Enable explicitly when needed.
    set_speed_bar_on_connect: bool = False
    speed_bar: float = 0.3

    # Optional gripper.
    #
    # "none":
    #     No gripper feature.
    #
    # "rby1_dynamixel":
    #     Dynamixel gripper connected to the control PC.
    gripper_type: str = "none"
    gripper_port: str = "/dev/rby1_gripper"
    gripper_ids: list[int] = field(default_factory=lambda: [0])
    gripper_invert: bool = False

    # Optional LeRobot cameras. Empty by default.
    cameras: dict[str, CameraConfig] = field(default_factory=_default_cameras)

    @property
    def servo_t1(self) -> float:
        """ServoJ arrival time in seconds."""

        return 1.0 / self.control_rate_hz

    @property
    def servo_t2(self) -> float:
        """ServoJ hold time in seconds."""

        return self.servo_hold_multiplier * self.servo_t1

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.model not in MODEL_SPECS:
            raise ValueError(
                "RbCobotConfig.model must be one of "
                f"{sorted(MODEL_SPECS)}, got {self.model!r}."
            )

        try:
            ipaddress.IPv4Address(self.ip)
        except ipaddress.AddressValueError as exc:
            raise ValueError(
                f"RbCobotConfig.ip must be a valid IPv4 address, got {self.ip!r}."
            ) from exc

        for name, value in (
            ("command_port", self.command_port),
            ("data_port", self.data_port),
        ):
            if not isinstance(value, int):
                raise TypeError(
                    f"RbCobotConfig.{name} must be an integer, got {type(value).__name__}."
                )

            if not 0 < value <= 65535:
                raise ValueError(
                    f"RbCobotConfig.{name} must be in [1, 65535], got {value}."
                )

        for name, value in (
            ("socket_timeout_s", self.socket_timeout_s),
            ("first_state_timeout_s", self.first_state_timeout_s),
            ("data_request_hz", self.data_request_hz),
            ("control_rate_hz", self.control_rate_hz),
            ("servo_hold_multiplier", self.servo_hold_multiplier),
            ("servo_gain", self.servo_gain),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"RbCobotConfig.{name} must be finite and > 0, got {value}."
                )

        if not math.isfinite(self.servo_alpha):
            raise ValueError(
                "RbCobotConfig.servo_alpha must be finite, "
                f"got {self.servo_alpha}."
            )

        if not 0.0 < self.servo_alpha < 1.0:
            raise ValueError(
                "RbCobotConfig.servo_alpha must be in (0, 1), "
                f"got {self.servo_alpha}."
            )

        if not math.isfinite(self.speed_bar):
            raise ValueError(
                f"RbCobotConfig.speed_bar must be finite, got {self.speed_bar}."
            )

        if not 0.0 < self.speed_bar <= 1.0:
            raise ValueError(
                "RbCobotConfig.speed_bar must be in (0, 1], "
                f"got {self.speed_bar}."
            )

        if self.gripper_type not in ("none", "rby1_dynamixel"):
            raise ValueError(
                "RbCobotConfig.gripper_type must be "
                f"'none' or 'rby1_dynamixel', got {self.gripper_type!r}."
            )

        if self.gripper_type != "none" and not self.gripper_ids:
            raise ValueError(
                "RbCobotConfig.gripper_ids must not be empty "
                "when a gripper is enabled."
            )

        if len(set(self.gripper_ids)) != len(self.gripper_ids):
            raise ValueError(
                "RbCobotConfig.gripper_ids must not contain duplicates, "
                f"got {self.gripper_ids}."
            )

        if any(
            not isinstance(device_id, int) or device_id < 0
            for device_id in self.gripper_ids
        ):
            raise ValueError(
                "RbCobotConfig.gripper_ids must contain non-negative integers, "
                f"got {self.gripper_ids}."
            )
