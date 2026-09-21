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
    """Cameras recorded into the dataset, keyed by name.

    The dict key becomes the dataset feature name, i.e. ``front`` is stored
    as ``observation.images.front``. Renaming a camera makes new recordings
    incompatible with existing datasets, so treat these names as frozen.

    ``width``/``height`` are the **output** dimensions, after ``rotation``.
    RealSenseCamera swaps them back before asking the sensor when the
    rotation is +/-90 degrees, so a portrait entry MUST be paired with
    ``ROTATE_90``:

        480x640 + ROTATE_90     -> sensor is asked for 640x480   (valid)
        480x640 + NO_ROTATION   -> sensor is asked for 480x640   (D405 has
                                   no such mode; the pipeline fails to open)

    Serial numbers are the librealsense ones. They are NOT the serials that
    ``lsusb`` or sysfs report for the same devices. Confirm with the
    pyrealsense2 enumeration in RUN.md before changing them.
    """

    return {
        # Fixed workspace view, camera mounted on its side.
        # "front": RealSenseCameraConfig(
        #     serial_number_or_name="315122271025",
        #     fps=30,
        #     width=480,
        #     height=640,
        #     rotation=Cv2Rotation.ROTATE_90,
        # ),
        # Gripper-mounted close-up view.
        #
        # Landscape and unrotated. If this camera is physically mounted on
        # its side, switch to width=480, height=640 and ROTATE_90 to match
        # "front" -- do not change only one of the two.
        "wrist": RealSenseCameraConfig(
            serial_number_or_name="262622274852",
            fps=30,
            width=640,
            height=480,
            rotation=Cv2Rotation.NO_ROTATION,
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
    control_rate_hz: float = 30.0

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

    # LeRobot cameras, keyed by name. Defaults to the two RealSense D405s in
    # _default_cameras(). Pass --robot.cameras='{}' to record without any.
    #
    # Camera names share one flat namespace with joint_0..joint_5 and
    # gripper_0 in observation_features, so do not reuse those names.
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
            ) 

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
