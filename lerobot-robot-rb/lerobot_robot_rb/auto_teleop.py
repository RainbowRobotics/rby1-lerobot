"""Software-only gentle trajectory source for RB-Series recording tests.

``RbAutoTeleop`` captures the connected RB robot's current pose once and
generates a small sinusoidal absolute target on one axis — a joint
(``space="joint"``) or a TCP axis (``space="ee"``, for robots configured
with ``action_space="ee"``). It is intended for short ``lerobot-record``
smoke tests when no physical teleoperation device is available.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.types import RobotAction
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .models import DOF, EE_NAMES, GRIPPER_NAME, JOINT_NAMES
from .rb_cobot import get_active_rb_cobot

logger = logging.getLogger(__name__)


@TeleoperatorConfig.register_subclass("rb_auto")
@dataclass(kw_only=True)
class RbAutoTeleopConfig(TeleoperatorConfig):
    """Configuration for the software-only RB gentle-motion teleoperator."""

    # Must match the robot's action_space ("joint" or "ee").
    space: str = "joint"

    # Axis to move: joint index for "joint"; 0..5 = x,y,z,rx,ry,rz for "ee".
    axis: int = 5

    # Sine amplitude: degrees for joints and rotation axes, millimetres for
    # EE position axes.
    amplitude: float = 0.5

    period_s: float = 8.0

    # During normal recording this stays unset so the connected robot's
    # current observation becomes the centre and the first target cannot
    # jump. An explicit centre (dataset units: radians, or metres/radians
    # for "ee") is useful for hardware-free tests.
    center: list[float] | None = None

    def __post_init__(self) -> None:
        if self.space not in ("joint", "ee"):
            raise ValueError(
                f'space must be "joint" or "ee", got "{self.space}".'
            )
        if not 0 <= self.axis < DOF:
            raise ValueError(
                f"axis must be in [0, {DOF - 1}], got {self.axis}."
            )
        if not math.isfinite(self.amplitude) or self.amplitude <= 0:
            raise ValueError(
                f"amplitude must be finite and > 0, got {self.amplitude}."
            )
        if not math.isfinite(self.period_s) or self.period_s <= 0:
            raise ValueError(f"period_s must be finite and > 0, got {self.period_s}.")
        if self.center is not None:
            if len(self.center) != DOF:
                raise ValueError(
                    f"center must contain {DOF} values, got {len(self.center)}."
                )
            if not np.isfinite(np.asarray(self.center, dtype=np.float64)).all():
                raise ValueError("center must contain only finite values.")


class RbAutoTeleop(Teleoperator):
    """Generate a smooth, low-amplitude target without extra hardware."""

    config_class = RbAutoTeleopConfig
    name = "rb_auto"

    def __init__(self, config: RbAutoTeleopConfig) -> None:
        super().__init__(config)
        self.config = config
        self._names = EE_NAMES if config.space == "ee" else JOINT_NAMES
        self._connected = False
        self._center: np.ndarray | None = None
        self._start_time: float | None = None

        # Amplitude in dataset units: mm -> m for EE position axes,
        # degrees -> radians everywhere else.
        if config.space == "ee" and config.axis < 3:
            self._amplitude = config.amplitude / 1000.0
        else:
            self._amplitude = math.radians(config.amplitude)

    @property
    def action_features(self) -> dict[str, type]:
        return {name: float for name in self._names}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        if self.config.center is not None:
            center = np.asarray(self.config.center, dtype=np.float64)
        else:
            robot = get_active_rb_cobot()
            if robot is None:
                raise DeviceNotConnectedError(
                    "rb_auto requires a connected RB robot in the same process. "
                    "Use it with `lerobot-record --robot.type=rb10`."
                )
            robot_space = robot._config.action_space
            if robot_space != self.config.space:
                raise ValueError(
                    f"rb_auto space={self.config.space!r} does not match the "
                    f"robot's action_space={robot_space!r}; pass "
                    f"`--teleop.space={robot_space}`."
                )
            if GRIPPER_NAME in robot.action_features:
                raise ValueError(
                    "rb_auto currently supports `--robot.gripper_type=none` only."
                )
            observation = robot.get_observation()
            center = np.asarray(
                [float(observation[name]) for name in self._names],
                dtype=np.float64,
            )

        if center.shape != (DOF,) or not np.isfinite(center).all():
            raise ValueError(
                "Could not obtain a finite six-value centre pose for rb_auto."
            )

        self._center = center
        self._start_time = time.perf_counter()
        self._connected = True
        logger.info(
            "RB auto teleop (%s) centred at %s; axis %d moves ±%s every %.2f s.",
            self.config.space,
            center.round(5).tolist(),
            self.config.axis,
            self.config.amplitude,
            self.config.period_s,
        )

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> RobotAction:
        if not self.is_connected or self._center is None or self._start_time is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        elapsed = time.perf_counter() - self._start_time
        phase = 2.0 * math.pi * elapsed / self.config.period_s
        target = self._center.copy()
        target[self.config.axis] += self._amplitude * math.sin(phase)

        return {
            name: float(value)
            for name, value in zip(self._names, target, strict=True)
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:  # noqa: ARG002
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

    def disconnect(self) -> None:
        self._connected = False
        self._center = None
        self._start_time = None
