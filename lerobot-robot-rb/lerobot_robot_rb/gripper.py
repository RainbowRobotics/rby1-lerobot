"""RB-Series gripper drivers.

The RB Dynamixel gripper is controlled through the existing RB control-box
connection. This module does not open a serial port or create another Cobot
connection.

Hardware convention
-------------------
0.0 = fully open
1.0 = fully closed

The Robot adapter converts this to/from the LeRobot dataset convention.
"""

from __future__ import annotations

import abc
import logging
from typing import Protocol

import numpy as np


logger = logging.getLogger(__name__)


DEFAULT_OPEN_CURRENT_MA = -50
DEFAULT_CLOSE_CURRENT_MA = 50


class CobotGripperProtocol(Protocol):
    """Control-box gripper methods required by this module."""

    def gripper_dxl_xm_initialization(
        self,
        mode: int,
    ) -> object:
        ...

    def gripper_dxl_xm_set_target_current(
        self,
        current_mA: int,
    ) -> object:
        ...


class RbGripperBase(abc.ABC):
    """Minimal gripper interface used by RbCobot."""

    @abc.abstractmethod
    def connect(self) -> None:
        ...

    @abc.abstractmethod
    def disconnect(self) -> None:
        ...

    @abc.abstractmethod
    def set_position(
        self,
        normalized: float,
    ) -> None:
        """Set hardware position: 0.0=open, 1.0=closed."""

    @abc.abstractmethod
    def get_position(self) -> float:
        """Return hardware position: 0.0=open, 1.0=closed."""


class RbDynamixelGripper(RbGripperBase):
    """Current-controlled Dynamixel gripper through the RB control box."""

    def __init__(
        self,
        cobot: CobotGripperProtocol,
        *,
        mode: int = 0,
        open_current_mA: int = DEFAULT_OPEN_CURRENT_MA,
        close_current_mA: int = DEFAULT_CLOSE_CURRENT_MA,
        invert: bool = False,
    ) -> None:
        self._cobot = cobot
        self._mode = int(mode)

        open_current = int(open_current_mA)
        close_current = int(close_current_mA)

        if invert:
            open_current, close_current = (
                close_current,
                open_current,
            )

        self._open_current_mA = open_current
        self._close_current_mA = close_current

        self._is_connected = False

        # Hardware convention:
        #   0.0 = open
        #   1.0 = closed
        self._current_position = 0.0

    def connect(self) -> None:
        if self._is_connected:
            return

        self._cobot.gripper_dxl_xm_initialization(
            self._mode
        )

        # Preserve the proposed control style: initialize in the open state.
        self._cobot.gripper_dxl_xm_set_target_current(
            self._open_current_mA
        )

        self._current_position = 0.0
        self._is_connected = True

        logger.info(
            "RB Dynamixel gripper initialized "
            "(mode=%d, open=%d mA, close=%d mA).",
            self._mode,
            self._open_current_mA,
            self._close_current_mA,
        )

    def disconnect(self) -> None:
        if not self._is_connected:
            return

        # Release applied current on shutdown.
        try:
            self._cobot.gripper_dxl_xm_set_target_current(
                0
            )
        except Exception:
            logger.exception(
                "Failed to release gripper current during disconnect."
            )

        self._is_connected = False

        logger.info(
            "RB Dynamixel gripper disconnected."
        )

    def set_position(
        self,
        normalized: float,
    ) -> None:
        """Set binary gripper state.

        Values below 0.5 select OPEN.
        Values equal to or above 0.5 select CLOSED.
        """

        if not self._is_connected:
            raise RuntimeError(
                "RB Dynamixel gripper is not connected."
            )

        value = float(
            np.clip(
                float(normalized),
                0.0,
                1.0,
            )
        )

        target_position = (
            1.0
            if value >= 0.5
            else 0.0
        )

        # Avoid sending the same control-box command every LeRobot tick.
        if target_position == self._current_position:
            return

        if target_position == 0.0:
            target_current = (
                self._open_current_mA
            )
        else:
            target_current = (
                self._close_current_mA
            )

        self._cobot.gripper_dxl_xm_set_target_current(
            target_current
        )

        self._current_position = target_position

    def get_position(self) -> float:
        """Return the last commanded binary gripper state."""

        return float(
            self._current_position
        )


def make_gripper(
    config,
    cobot: CobotGripperProtocol | None = None,
) -> RbGripperBase | None:
    """Create the gripper configured for an RB robot."""

    if config.gripper_type == "none":
        return None

    if config.gripper_type == "rby1_dynamixel":
        if cobot is None:
            raise ValueError(
                "A connected Cobot instance is required for "
                "gripper_type='rby1_dynamixel'."
            )

        return RbDynamixelGripper(
            cobot,
            mode=0,
            invert=config.gripper_invert,
        )

    raise ValueError(
        f"Unknown gripper_type: {config.gripper_type!r}."
    )
