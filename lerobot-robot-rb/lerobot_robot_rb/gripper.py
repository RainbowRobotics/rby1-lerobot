"""Gripper drivers for the RB-Series cobots.

The robot exposes a single normalised gripper channel (``gripper_0``).  The
hardware convention throughout this module is ``0.0 = fully open`` and
``1.0 = fully closed``; :class:`~lerobot_robot_rb.rb_cobot.RbCobot` flips to
the dataset convention (1.0 = open) at the observation/action boundary, the
same as the RB-Y1 package.

Gripper selection is config-driven (``RbCobotConfig.gripper_type``) through
:func:`make_gripper`.  To add a new gripper (e.g. the RH-P12-RN through the
rbpodo built-in ``gripper_rts_rhp12rn_*`` API), implement
:class:`RbGripperBase` and add a branch to the factory.
"""

from __future__ import annotations

import abc
import logging
import time

import numpy as np

logger = logging.getLogger(__name__)

# Dynamixel bus parameters, shared with the RB-Y1 gripper hardware.
GRIPPER_BAUD_RATE = 2_000_000
GRIPPER_HOMING_TORQUE = 0.46     # Nm, applied during the homing sweeps
GRIPPER_HOMING_STEPS = 30        # 0.1 s x 30 = 3 s per direction
GRIPPER_POSITION_TORQUE = 0.46   # Nm, max torque in position mode


class RbGripperBase(abc.ABC):
    """Minimal gripper interface consumed by :class:`RbCobot`.

    Positions are normalised at the hardware level: 0.0 = fully open,
    1.0 = fully closed.
    """

    @abc.abstractmethod
    def connect(self) -> None: ...

    @abc.abstractmethod
    def disconnect(self) -> None: ...

    @abc.abstractmethod
    def set_position(self, normalized: float) -> None: ...

    @abc.abstractmethod
    def get_position(self) -> float: ...


class RbDynamixelGripper(RbGripperBase):
    """Dynamixel gripper driven through rby1_sdk's DynamixelBus.

    Adapted from ``lerobot_robot_rby1.gripper.Rby1Gripper`` (two mirrored
    motors on the RB-Y1 UPC) for a single motor on a configurable serial
    port.  The encoder range is discovered by the same fixed-duration
    current-sweep homing; ``invert`` flips which end of the range is "open"
    (depends on the mounting orientation).

    Requires the optional ``rby1-sdk`` dependency
    (``pip install lerobot-robot-rb[gripper-rby1]``).
    """

    def __init__(self, rbpodo_cobot) -> None:
        n = len(self._ids)
        # Encoder values (radians) discovered during homing.
        self._open_mA: int = -50
        self._close_mA: int = 50
        self._homed: bool = False
        self._robot = rbpodo_cobot
        self._robot.gripper_dxl_xm_initialization(0)
        self._current_updated_value = 0 # 0: opened, 1: closed

    # ------------------------------------------------------------------ #
    #  Homing                                                              #
    # ------------------------------------------------------------------ #

    def _home(self) -> None:
        """Sweep the motor to its mechanical limits to map the encoder range.

        direction 0 applies ``+GRIPPER_HOMING_TORQUE`` for
        ``GRIPPER_HOMING_STEPS`` steps (~3 s); direction 1 applies the same
        torque in reverse.  The min / max encoder values seen over the full
        run define the open / closed range (which end is which is selected
        by ``invert``).
        """
        self._robot.gripper_dxl_xm_set_target_current(self._open_mA)
        self._current_updated_value = 0
        self._homed = True

    # ------------------------------------------------------------------ #
    #  I/O                                                                 #
    # ------------------------------------------------------------------ #

    def set_gripper(self, open_close: int) -> None:
        """Send a goal current (0.0 = open, 1.0 = closed) to all motors."""
        if not self._homed:
            return
        if open_close == 0:
            self._robot.gripper_dxl_xm_set_target_current(self._open_mA)
            self._current_updated_value = 0
        elif open_close == 1:
            self._robot.gripper_dxl_xm_set_target_current(self._close_mA)
            self._current_updated_value = 1

    def get_position(self) -> float:
        """Read the current normalised position (0.0 = open, 1.0 = closed).

        With multiple motors on the bus the mean position is returned.
        """
        if not self._homed:
            return 0.0
        return float(self._current_updated_value)


def make_gripper(config, rbpodo_cobot = None) -> RbGripperBase | None:
    """Instantiate the gripper selected by ``config.gripper_type`` (or None)."""
    if config.gripper_type == "none":
        return None
    if config.gripper_type == "rby1_dynamixel":
        return RbDynamixelGripper(
            rbpodo_cobot=rbpodo_cobot,
        )
    raise ValueError(f'Unknown gripper_type "{config.gripper_type}".')
