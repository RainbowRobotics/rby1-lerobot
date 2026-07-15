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

    def __init__(self, port: str, ids: list[int], invert: bool = False) -> None:
        self._port = port
        self._ids = list(ids)
        self._invert = invert
        self._bus = None
        n = len(self._ids)
        # Encoder values (radians) discovered during homing.
        self._open_rad: np.ndarray = np.zeros(n)
        self._close_rad: np.ndarray = np.zeros(n)
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
                "rby1_sdk is required for gripper_type='rby1_dynamixel'. "
                "Install it with `pip install lerobot-robot-rb[gripper-rby1]`."
            ) from e

        self._bus = rby.DynamixelBus(self._port)

        if not self._bus.open_port():
            raise RuntimeError(f"Failed to open gripper port: {self._port}")

        # Set the USB latency_timer to 1 ms (the 16 ms default caps the bus
        # at ~20 Hz). Best-effort: needs udev/root permissions on some hosts.
        try:
            upc.initialize_device(self._port)
            logger.info("Gripper USB latency_timer set to 1ms.")
        except Exception as exc:
            logger.warning(
                f"Could not set the gripper USB latency_timer ({exc}); "
                "the gripper will work but with reduced bus rate."
            )

        if not self._bus.set_baud_rate(GRIPPER_BAUD_RATE):
            raise RuntimeError("Failed to set gripper baud rate.")
        self._bus.set_torque_constant([1.0] * len(self._ids))

        for dev_id in self._ids:
            if not self._bus.ping(dev_id):
                raise RuntimeError(f"Gripper motor {dev_id} did not respond to ping.")

        self._home()
        logger.info("RbDynamixelGripper connected and homed.")

    def disconnect(self) -> None:
        if self._bus is not None:
            try:
                self._bus.group_sync_write_torque_enable(
                    [(dev_id, 0) for dev_id in self._ids]
                )
            except Exception:
                pass
            self._bus = None
            self._homed = False
        logger.info("RbDynamixelGripper disconnected.")

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
        import rby1_sdk as rby

        def _set_mode(mode: int) -> None:
            self._bus.group_sync_write_torque_enable(
                [(dev_id, rby.DynamixelBus.TorqueDisable) for dev_id in self._ids]
            )
            self._bus.group_sync_write_operating_mode(
                [(dev_id, mode) for dev_id in self._ids]
            )
            self._bus.group_sync_write_torque_enable(
                [(dev_id, rby.DynamixelBus.TorqueEnable) for dev_id in self._ids]
            )

        logger.info("Homing gripper (3 s per direction) ...")
        _set_mode(rby.DynamixelBus.CurrentControlMode)

        n = len(self._ids)
        idx = {dev_id: i for i, dev_id in enumerate(self._ids)}
        q = np.zeros(n, dtype=np.float64)
        min_q = np.full(n, np.inf)
        max_q = np.full(n, -np.inf)

        for direction in range(2):
            torque = GRIPPER_HOMING_TORQUE if direction == 0 else -GRIPPER_HOMING_TORQUE
            for _ in range(GRIPPER_HOMING_STEPS):
                self._bus.group_sync_write_send_torque(
                    [(dev_id, torque) for dev_id in self._ids]
                )
                rv = self._bus.group_fast_sync_read_encoder(self._ids)
                if rv is not None:
                    for dev_id, enc in rv:
                        q[idx[dev_id]] = enc
                min_q = np.minimum(min_q, q)
                max_q = np.maximum(max_q, q)
                time.sleep(0.1)

        # Stop the homing current and switch to position control.
        self._bus.group_sync_write_send_torque(
            [(dev_id, 0.0) for dev_id in self._ids]
        )
        _set_mode(rby.DynamixelBus.CurrentBasedPositionControlMode)
        self._bus.group_sync_write_send_torque(
            [(dev_id, GRIPPER_POSITION_TORQUE) for dev_id in self._ids]
        )

        # Default: low encoder = OPEN, high encoder = CLOSED; ``invert``
        # swaps the two (mirror-mounted grippers).
        if self._invert:
            self._open_rad, self._close_rad = max_q.copy(), min_q.copy()
        else:
            self._open_rad, self._close_rad = min_q.copy(), max_q.copy()
        self._homed = True

        # Sanity check: warn if a motor did not travel a meaningful distance.
        for i, dev_id in enumerate(self._ids):
            span = abs(self._close_rad[i] - self._open_rad[i])
            if span < 0.01:
                logger.warning(
                    f"[Gripper] motor {dev_id} range is very small ({span:.4f} rad). "
                    "Homing may not have reached the physical limits."
                )
            else:
                logger.info(
                    f"[Gripper] motor {dev_id}: "
                    f"open_rad={self._open_rad[i]:.4f}, "
                    f"close_rad={self._close_rad[i]:.4f}, "
                    f"span={span:.4f} rad"
                )

    # ------------------------------------------------------------------ #
    #  I/O                                                                 #
    # ------------------------------------------------------------------ #

    def set_position(self, normalized: float) -> None:
        """Send a goal position (0.0 = open, 1.0 = closed) to all motors."""
        if not self._homed:
            return
        normalized = float(np.clip(normalized, 0.0, 1.0))
        target_rad = self._open_rad + normalized * (self._close_rad - self._open_rad)
        self._bus.group_sync_write_send_position(
            [(dev_id, float(r)) for dev_id, r in zip(self._ids, target_rad)]
        )

    def get_position(self) -> float:
        """Read the current normalised position (0.0 = open, 1.0 = closed).

        With multiple motors on the bus the mean position is returned.
        """
        if not self._homed:
            return 0.0
        result = self._bus.group_fast_sync_read_encoder(self._ids)
        if result is None:
            return 0.0
        enc_map = {dev_id: enc for dev_id, enc in result}
        current_rad = np.array(
            [enc_map.get(dev_id, self._open_rad[i]) for i, dev_id in enumerate(self._ids)]
        )
        span = self._close_rad - self._open_rad
        # Guard against a zero span if homing failed to find a range.
        safe_span = np.where(np.abs(span) > 1e-6, span, 1.0)
        normalized = np.clip((current_rad - self._open_rad) / safe_span, 0.0, 1.0)
        return float(normalized.mean())


def make_gripper(config) -> RbGripperBase | None:
    """Instantiate the gripper selected by ``config.gripper_type`` (or None)."""
    if config.gripper_type == "none":
        return None
    if config.gripper_type == "rby1_dynamixel":
        return RbDynamixelGripper(
            port=config.gripper_port,
            ids=config.gripper_ids,
            invert=config.gripper_invert,
        )
    raise ValueError(f'Unknown gripper_type "{config.gripper_type}".')
