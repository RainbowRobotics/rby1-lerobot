"""LeRobot Teleoperator for the RB-Y1 mobile base driven by a UDP joystick.

The joystick is attached to a Legion Go running ``joystick_sender.py`` (pygame
→ UDP). This teleoperator, running on the robot-side PC, *receives* those UDP
packets over WiFi and maps the analog sticks to mobile-base SE(2) velocities,
replacing the previous keyboard input backend. The registration name stays
``rby1_keyboard`` for backward compatibility (``--teleop.type rby1_keyboard``).

Wire format (from ``joystick_sender.py``)
-----------------------------------------
Each packet is ``struct.pack(f"{N}f{M}B4B", *axes, *buttons, *hat)`` in native
byte order: ``N`` float axes in ``[-1, 1]`` first, then ``M`` button bytes,
then 4 hat bytes. Because the axes lead the packet, we only need to unpack the
first ``max(axis index)+1`` floats — the axis/button counts need not be known.

Control flow
------------
1. :meth:`connect` binds the UDP socket and starts a background reader thread
   that keeps the latest axis snapshot (and its arrival time) under a lock.
2. Each call to :meth:`get_action`:

   a. Reads the latest axes; if the last packet is older than
      ``command_timeout`` (WiFi drop / sender crash), the target is zeroed.
   b. Maps the configured axis indices to a target ``(x, y, yaw)``.
   c. Low-pass-filters the commanded velocity toward that target for smooth
      starts/stops.
   d. Returns the velocity as a LeRobot action dict.

Following the LeRobot convention, this teleoperator only *produces* actions
and never commands the robot. The paired ``lerobot_robot_rby1.Rby1`` follower
executes the ``x.vel`` / ``y.vel`` / ``theta.vel`` velocities via
``send_action`` (its onboard mobile-base SE2 velocity command), exactly as the
VR teleoperator's mobile-base path does.

Action keys
-----------
    x.vel       body-frame linear x velocity (m/s)
    y.vel       body-frame linear y velocity (m/s)
    theta.vel   yaw rate (rad/s)
    head_1      head-pitch target position (rad); a fixed neutral angle — the
                joystick does not steer the head, so the follower holds its pose.
"""

from __future__ import annotations

import logging
import socket
import struct
import threading
import time
from typing import Any

import numpy as np

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot_robot_rby1 import endpoint_state

from .config_rby1_keyboard import Rby1KeyboardConfig
from .constants import BASE_VEL_NAMES, HEAD_PITCH_NAME

logger = logging.getLogger(__name__)


class Rby1Keyboard(Teleoperator):
    """UDP joystick teleoperator emitting mobile-base velocity actions for the RB-Y1."""

    config_class = Rby1KeyboardConfig
    name = "rby1_keyboard"

    def __init__(self, config: Rby1KeyboardConfig) -> None:
        super().__init__(config)
        self._config = config
        self._is_connected = False

        # Background UDP joystick receiver (opened on connect).
        self._receiver: _JoystickReceiver | None = None

        # Commanded (low-pass-filtered) body-frame velocity: [x, y, yaw].
        self._vel = np.zeros(3)

        # Commanded head-pitch target position (rad) — fixed for the joystick.
        self._head_pitch = config.head_pitch_init

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def action_features(self) -> dict[str, type]:
        return {name: float for name in (*BASE_VEL_NAMES, HEAD_PITCH_NAME)}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    # ------------------------------------------------------------------ #
    # Connection
    # ------------------------------------------------------------------ #

    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002
        if self._is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected.")

        # Start clean: endpoint-state toggle back to 0 for the new session.
        endpoint_state.reset()
        self._receiver = _JoystickReceiver(
            self._config.udp_ip,
            self._config.udp_port,
            self._config.num_axes,
            self._config.toggle_button_index,
            endpoint_state.set,
            self._config.log_buttons,
        )
        self._receiver.open()
        self._receiver.start()

        self._vel = np.zeros(3)
        self._head_pitch = self._config.head_pitch_init
        self._is_connected = True
        logger.info(
            "%s listening for joystick UDP on %s:%d. Base stops after %.2fs "
            "without a packet.",
            self,
            self._config.udp_ip,
            self._config.udp_port,
            self._config.command_timeout,
        )

    def disconnect(self) -> None:
        if not self._is_connected:
            return
        if self._receiver is not None:
            self._receiver.stop()
            self._receiver = None
        self._is_connected = False
        logger.info(f"{self} disconnected.")

    # ------------------------------------------------------------------ #
    # Calibration / Configuration
    # ------------------------------------------------------------------ #

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    # Reset (for lerobot-record)
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Reset commanded state (e.g. between recorded episodes).

        Zeroes the base velocity, returns the head pitch to its init angle, and
        clears the endpoint-state toggle back to 0 so every episode starts at 0.
        """
        self._vel = np.zeros(3)
        self._head_pitch = self._config.head_pitch_init
        if self._receiver is not None:
            self._receiver.reset_toggle()
        else:
            endpoint_state.reset()

    # ------------------------------------------------------------------ #
    # get_action — main per-tick entry point
    # ------------------------------------------------------------------ #

    def get_action(self) -> dict[str, Any]:
        if not self._is_connected or self._receiver is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Mobile base — low-pass filter toward the stick-derived velocity target
        # for smooth accel/decel.
        target = self._compute_target()
        alpha = self._config.smoothing
        self._vel += alpha * (target - self._vel)

        return {
            "x.vel": float(self._vel[0]),
            "y.vel": float(self._vel[1]),
            "theta.vel": float(self._vel[2]),
            HEAD_PITCH_NAME: float(self._head_pitch),
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _compute_target(self) -> np.ndarray:
        """Map the latest joystick packet to a target ``[x, y, yaw]``.

        Returns a zero target when the last packet is stale (no fresh input
        within ``command_timeout``) so the base coasts to a stop on WiFi loss.
        """
        cfg = self._config
        assert self._receiver is not None

        axes, age = self._receiver.snapshot()
        if axes is None or age > cfg.command_timeout:
            return np.zeros(3)

        stick_x = self._deadzone(axes[cfg.axis_x_index])
        stick_y = self._deadzone(axes[cfg.axis_y_index])
        stick_yaw = self._deadzone(axes[cfg.axis_yaw_index])

        if cfg.invert_x_axis:
            stick_x = -stick_x
        if cfg.invert_y_axis:
            stick_y = -stick_y
        if cfg.invert_yaw_axis:
            stick_yaw = -stick_yaw

        return np.array(
            [
                stick_x * cfg.max_linear_speed,
                stick_y * cfg.max_linear_speed,
                stick_yaw * cfg.max_angular_speed,
            ]
        )

    def _deadzone(self, value: float) -> float:
        """Zero out |value| below the configured deadzone."""
        return 0.0 if abs(value) < self._config.joystick_deadzone else float(value)


class _JoystickReceiver(threading.Thread):
    """Background UDP reader for the joystick axis snapshot and toggle button.

    ``joystick_sender.py`` streams packets (~60 Hz) laid out as ``num_axes``
    floats, then one byte per button, then 4 hat bytes. This thread:

    * stores the most recent axis tuple plus its arrival time under a lock
      (:meth:`snapshot` returns them and the packet age); and
    * edge-detects the toggle button (``button_data[toggle_button_index]``) and
      flips a 0/1 state on each 0->1 rising edge, publishing it via ``on_toggle``.

    Edge detection lives in this thread so a quick tap is never missed between
    ``get_action`` calls.
    """

    def __init__(
        self,
        udp_ip: str,
        udp_port: int,
        num_axes: int,
        toggle_button_index: int,
        on_toggle: Any,
        log_buttons: bool = False,
    ) -> None:
        super().__init__(daemon=True)
        self._udp_ip = udp_ip
        self._udp_port = udp_port
        self._num_axes = num_axes
        self._axes_fmt = f"{num_axes}f"
        self._axes_nbytes = num_axes * 4
        # Byte offset of the toggle button (buttons follow all axis floats).
        self._toggle_button_index = toggle_button_index
        self._toggle_byte_offset = (
            num_axes * 4 + toggle_button_index if toggle_button_index >= 0 else -1
        )
        self._on_toggle = on_toggle
        self._log_buttons = log_buttons
        self._logged_layout = False
        self._prev_buttons: tuple[int, ...] = ()
        self._sock: socket.socket | None = None
        self._running = False
        self._lock = threading.Lock()
        self._axes: tuple[float, ...] | None = None
        self._stamp: float = 0.0
        self._toggle = 0
        self._prev_button = 0

    def open(self) -> None:
        """Bind the UDP socket (raises on bind failure, e.g. port in use)."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self._udp_ip, self._udp_port))
        # Time out recv so the loop can observe stop() promptly.
        sock.settimeout(0.2)
        self._sock = sock

    def run(self) -> None:
        assert self._sock is not None
        self._running = True
        while self._running:
            try:
                data, _ = self._sock.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError:
                # Socket closed on stop() — end the thread quietly.
                break
            if len(data) < self._axes_nbytes:
                continue
            try:
                axes = struct.unpack(self._axes_fmt, data[: self._axes_nbytes])
            except struct.error:
                continue
            with self._lock:
                self._axes = axes
                self._stamp = time.monotonic()
            self._log_diagnostics(data)
            self._update_toggle(data)

    def _log_diagnostics(self, data: bytes) -> None:
        """Log the packet layout and pressed button indices (diagnostic only).

        Buttons occupy ``data[num_axes*4 : len-4]`` (the final 4 bytes are the
        hat block). If the implied button count is wrong or the A button never
        shows up at the expected index, ``num_axes`` is likely off.
        """
        if not self._log_buttons:
            return
        if not self._logged_layout:
            self._logged_layout = True
            implied_buttons = len(data) - self._axes_nbytes - 4
            offset = self._toggle_byte_offset
            byte_val = data[offset] if 0 <= offset < len(data) else -1
            logger.info(
                "[joystick] packet=%d bytes; num_axes=%d -> implied %d button "
                "bytes + 4 hat bytes. toggle reads byte %d (button_data[%d])=%d. "
                "hex=%s",
                len(data), self._num_axes, implied_buttons, offset,
                self._toggle_button_index, byte_val, data.hex(),
            )
        start, end = self._axes_nbytes, len(data) - 4
        if 0 <= start <= end <= len(data):
            buttons = tuple(data[start:end])
            if buttons != self._prev_buttons:
                self._prev_buttons = buttons
                pressed = [i for i, b in enumerate(buttons) if b]
                logger.info("[joystick] button_data pressed indices=%s", pressed)

    def _update_toggle(self, data: bytes) -> None:
        """Flip the 0/1 toggle on a rising edge of the toggle button."""
        if self._toggle_byte_offset < 0 or len(data) <= self._toggle_byte_offset:
            return
        button = 1 if data[self._toggle_byte_offset] else 0
        with self._lock:
            rising = self._prev_button == 0 and button == 1
            self._prev_button = button
            if rising:
                self._toggle ^= 1
                toggle = self._toggle
        if rising and self._on_toggle is not None:
            self._on_toggle(float(toggle))

    def reset_toggle(self) -> None:
        """Clear the toggle back to 0 and republish it (e.g. new episode)."""
        with self._lock:
            self._toggle = 0
            self._prev_button = 0
        if self._on_toggle is not None:
            self._on_toggle(0.0)

    def snapshot(self) -> tuple[tuple[float, ...] | None, float]:
        """Return ``(latest_axes, age_seconds)``; axes is ``None`` before any packet."""
        with self._lock:
            axes = self._axes
            stamp = self._stamp
        age = time.monotonic() - stamp if axes is not None else float("inf")
        return axes, age

    def stop(self) -> None:
        self._running = False
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
