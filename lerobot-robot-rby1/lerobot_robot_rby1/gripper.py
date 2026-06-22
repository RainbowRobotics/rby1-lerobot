"""Dynamixel gripper driver for the RB-Y1.

:class:`Rby1Gripper` drives the two Dynamixel motors on ``/dev/rby1_gripper``
(ID 0 = right hand, ID 1 = left hand).  Positions are normalised so that
``0.0`` is fully open and ``1.0`` is fully closed; the physical encoder range
is discovered automatically by the homing sweep in :meth:`Rby1Gripper.connect`.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from .constants import (
    GRIPPER_BAUD_RATE,
    GRIPPER_HOMING_STEPS,
    GRIPPER_HOMING_TORQUE,
    GRIPPER_IDS,
    GRIPPER_POSITION_TORQUE,
)

logger = logging.getLogger(__name__)


class Rby1Gripper:
    """Two Dynamixel gripper motors on ``/dev/rby1_gripper``.

    Motor ID 0 drives the right-hand gripper and ID 1 the left-hand gripper.
    Positions are normalised: ``0.0`` = fully open, ``1.0`` = fully closed.
    """

    def __init__(self) -> None:
        self._bus = None
        # Encoder values (radians) discovered during homing.
        self._open_rad: np.ndarray = np.zeros(2)
        self._close_rad: np.ndarray = np.zeros(2)
        self._homed: bool = False

    # ------------------------------------------------------------------ #
    #  Connection                                                          #
    # ------------------------------------------------------------------ #

    def connect(self) -> None:
        try:
            import rby1_sdk as rby
            import rby1_sdk.upc as upc
        except ImportError as e:
            raise ImportError(
                "rby1_sdk is required for Rby1Gripper. Install it from the RB-Y1 SDK."
            ) from e

        self._bus = rby.DynamixelBus(upc.GripperDeviceName)

        if not self._bus.open_port():
            raise RuntimeError(f"Failed to open gripper port: {upc.GripperDeviceName}")

        # Set the USB latency_timer to 1 ms (the 16 ms default caps the bus at ~20 Hz).
        upc.initialize_device(upc.GripperDeviceName)
        logger.info("Gripper USB latency_timer set to 1ms.")

        if not self._bus.set_baud_rate(GRIPPER_BAUD_RATE):
            raise RuntimeError("Failed to set gripper baud rate.")
        self._bus.set_torque_constant([1.0, 1.0])

        for dev_id in GRIPPER_IDS:
            if not self._bus.ping(dev_id):
                raise RuntimeError(f"Gripper motor {dev_id} did not respond to ping.")

        self._home()
        logger.info("Rby1Gripper connected and homed.")

    def disconnect(self) -> None:
        if self._bus is not None:
            try:
                self._bus.group_sync_write_torque_enable(
                    [(dev_id, 0) for dev_id in GRIPPER_IDS]
                )
            except Exception:
                pass
            self._bus = None
            self._homed = False
        logger.info("Rby1Gripper disconnected.")

    # ------------------------------------------------------------------ #
    #  Homing                                                              #
    # ------------------------------------------------------------------ #

    def _home(self) -> None:
        """Sweep both motors to their mechanical limits to map the encoder range.

        direction 0 applies ``+GRIPPER_HOMING_TORQUE`` for ``GRIPPER_HOMING_STEPS``
        steps (~3 s); direction 1 applies the same torque in reverse.  The min /
        max encoder values seen over the full run define the open / closed range.

        This is a fixed-duration sweep (not stall detection): the step counter
        always advances so the motors never stay pressed against the closed limit
        indefinitely.
        """
        import rby1_sdk as rby

        def _set_mode(mode: int) -> None:
            self._bus.group_sync_write_torque_enable(
                [(dev_id, rby.DynamixelBus.TorqueDisable) for dev_id in GRIPPER_IDS]
            )
            self._bus.group_sync_write_operating_mode(
                [(dev_id, mode) for dev_id in GRIPPER_IDS]
            )
            self._bus.group_sync_write_torque_enable(
                [(dev_id, rby.DynamixelBus.TorqueEnable) for dev_id in GRIPPER_IDS]
            )

        logger.info("Homing grippers (3 s per direction) ...")
        _set_mode(rby.DynamixelBus.CurrentControlMode)

        q = np.zeros(2, dtype=np.float64)
        min_q = np.full(2, np.inf)
        max_q = np.full(2, -np.inf)

        for direction in range(2):
            torque = GRIPPER_HOMING_TORQUE if direction == 0 else -GRIPPER_HOMING_TORQUE
            for _ in range(GRIPPER_HOMING_STEPS):
                self._bus.group_sync_write_send_torque(
                    [(dev_id, torque) for dev_id in GRIPPER_IDS]
                )
                rv = self._bus.group_fast_sync_read_encoder(GRIPPER_IDS)
                if rv is not None:
                    for dev_id, enc in rv:
                        q[dev_id] = enc
                min_q = np.minimum(min_q, q)
                max_q = np.maximum(max_q, q)
                time.sleep(0.1)

        # Stop the homing current and switch to position control.
        self._bus.group_sync_write_send_torque(
            [(dev_id, 0.0) for dev_id in GRIPPER_IDS]
        )
        _set_mode(rby.DynamixelBus.CurrentBasedPositionControlMode)
        self._bus.group_sync_write_send_torque(
            [(dev_id, GRIPPER_POSITION_TORQUE) for dev_id in GRIPPER_IDS]
        )

        # The motors are mirror-mounted, so the encoder direction differs per side.
        #   Right (ID 0): high encoder = CLOSED, low encoder = OPEN
        #   Left  (ID 1): low encoder  = CLOSED, high encoder = OPEN
        # Mapping the normalised range so that 0.0 = open and 1.0 = closed for both:
        #   open_rad  = [min_q[0] (OPEN),   max_q[1] (CLOSED)]
        #   close_rad = [max_q[0] (CLOSED), min_q[1] (OPEN)]
        self._open_rad = np.array([min_q[0], max_q[1]])
        self._close_rad = np.array([max_q[0], min_q[1]])
        self._homed = True

        # Sanity check: warn if a side did not travel a meaningful distance.
        for i, side in enumerate(("right", "left")):
            span = abs(self._close_rad[i] - self._open_rad[i])
            if span < 0.01:
                logger.warning(
                    f"[Gripper] {side} motor range is very small ({span:.4f} rad). "
                    "Homing may not have reached the physical limits."
                )
            else:
                logger.info(
                    f"[Gripper] {side} (ID={i}): "
                    f"open_rad={self._open_rad[i]:.4f}, "
                    f"close_rad={self._close_rad[i]:.4f}, "
                    f"span={span:.4f} rad"
                )

    # ------------------------------------------------------------------ #
    #  I/O                                                                 #
    # ------------------------------------------------------------------ #

    def set_positions(self, normalized: np.ndarray) -> None:
        """Send goal positions to both motors.

        Parameters
        ----------
        normalized : np.ndarray, shape (2,)
            ``[right, left]`` in ``[0.0 = open, 1.0 = closed]``.
        """
        if not self._homed:
            return
        normalized = np.clip(normalized, 0.0, 1.0)
        target_rad = self._open_rad + normalized * (self._close_rad - self._open_rad)
        logger.debug(
            f"[Gripper] set_positions: normalized={normalized.round(3)}, "
            f"target_rad={target_rad.round(4)}"
        )
        self._bus.group_sync_write_send_position(
            [(dev_id, float(r)) for dev_id, r in zip(GRIPPER_IDS, target_rad)]
        )

    def get_positions(self) -> np.ndarray:
        """Read the current normalised positions.

        Returns
        -------
        np.ndarray, shape (2,)
            ``[right, left]`` in ``[0.0 = open, 1.0 = closed]``.
        """
        if not self._homed:
            return np.zeros(2)
        result = self._bus.group_fast_sync_read_encoder(GRIPPER_IDS)
        if result is None:
            return np.zeros(2)
        enc_map = {dev_id: enc for dev_id, enc in result}
        current_rad = np.array([enc_map[dev_id] for dev_id in GRIPPER_IDS])

        span = self._close_rad - self._open_rad
        # Guard against a zero span if homing failed to find a range.
        safe_span = np.where(np.abs(span) > 1e-6, span, 1.0)
        return np.clip((current_rad - self._open_rad) / safe_span, 0.0, 1.0)
