"""LeRobot Teleoperator for the RB-Y1 mobile base driven by the keyboard.

This merges the two reference ROS scripts (a keyboard ``/cmd_vel`` publisher
and an SE2-velocity mobile-base controller) into a single LeRobot plugin,
dropping the ROS layer entirely.

Control flow
------------
1. :meth:`connect` puts the controlling terminal into cbreak mode so key
   presses can be read one character at a time without waiting for Enter.
2. Each call to :meth:`get_action`:

   a. Drains all pending keystrokes from stdin (non-blocking).
   b. Maps the pressed keys to a target body-frame velocity
      ``(x, y, yaw)``.
   c. Low-pass-filters the commanded velocity toward that target for smooth
      starts/stops (mirrors the reference publisher's LPF).
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

Key mapping (body frame)
------------------------
    w / s   linear x  + / -   (forward / backward)
    a / d   linear y  - / +   (left / right)
    q / e   yaw rate  + / -   (turn left / right)
"""

from __future__ import annotations

import logging
import select
import sys
from typing import Any

import numpy as np

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_rby1_keyboard import Rby1KeyboardConfig
from .constants import BASE_VEL_NAMES

logger = logging.getLogger(__name__)


class Rby1Keyboard(Teleoperator):
    """Keyboard teleoperator emitting mobile-base velocity actions for the RB-Y1."""

    config_class = Rby1KeyboardConfig
    name = "rby1_keyboard"

    def __init__(self, config: Rby1KeyboardConfig) -> None:
        super().__init__(config)
        self._config = config
        self._is_connected = False

        # Saved terminal attributes, restored on disconnect.
        self._old_term: Any = None

        # Commanded (low-pass-filtered) body-frame velocity: [x, y, yaw].
        self._vel = np.zeros(3)

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
        return {name: float for name in BASE_VEL_NAMES}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    # ------------------------------------------------------------------ #
    # Connection
    # ------------------------------------------------------------------ #

    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002
        if self._is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected.")

        self._enter_cbreak()
        self._vel = np.zeros(3)
        self._is_connected = True
        logger.info(
            "%s connected. Keyboard mobile-base control:\n"
            "  w/s: linear x +/-\n"
            "  a/d: linear y -/+\n"
            "  q/e: yaw rate +/-\n"
            "Release keys to coast to a stop.",
            self,
        )

    def disconnect(self) -> None:
        if not self._is_connected:
            return
        self._restore_terminal()
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
        """Zero the commanded velocity (e.g. between recorded episodes)."""
        self._vel = np.zeros(3)

    # ------------------------------------------------------------------ #
    # get_action — main per-tick entry point
    # ------------------------------------------------------------------ #

    def get_action(self) -> dict[str, Any]:
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        target = self._compute_target(self._drain_keys())

        # Low-pass filter toward the target for smooth accel/decel.
        alpha = self._config.smoothing
        self._vel += alpha * (target - self._vel)

        return {
            "x.vel": float(self._vel[0]),
            "y.vel": float(self._vel[1]),
            "theta.vel": float(self._vel[2]),
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _compute_target(self, keys: set[str]) -> np.ndarray:
        """Map the set of keys read this tick to a target ``[x, y, yaw]``."""
        cfg = self._config
        target = np.zeros(3)

        if "w" in keys:
            target[0] += cfg.max_linear_speed
        if "s" in keys:
            target[0] -= cfg.max_linear_speed
        if "d" in keys:
            target[1] += cfg.max_linear_speed
        if "a" in keys:
            target[1] -= cfg.max_linear_speed
        if "q" in keys:
            target[2] += cfg.max_angular_speed
        if "e" in keys:
            target[2] -= cfg.max_angular_speed

        return target

    def _drain_keys(self) -> set[str]:
        """Return every character currently buffered on stdin (non-blocking).

        Terminals emit no key-release events, so a held key arrives as an
        auto-repeat stream. Reading every pending character each tick lets
        simultaneously-held keys (e.g. w+d for a diagonal) register together;
        ticks with no keypress yield an empty set, so the LPF coasts the
        velocity back toward zero.
        """
        keys: set[str] = set()
        stdin = sys.stdin
        try:
            while select.select([stdin], [], [], 0)[0]:
                ch = stdin.read(1)
                if not ch:
                    break
                keys.add(ch.lower())
        except (OSError, ValueError):
            # stdin closed / not readable — treat as no input this tick.
            pass
        return keys

    def _enter_cbreak(self) -> None:
        """Put the terminal into cbreak mode, saving the previous attributes.

        No-op when stdin is not an interactive TTY (e.g. piped input), so the
        teleoperator still constructs in headless/test contexts — it just
        won't receive keystrokes.
        """
        if not sys.stdin.isatty():
            logger.warning(
                "%s: stdin is not a TTY; keyboard input will be unavailable.", self
            )
            return
        import termios
        import tty

        self._old_term = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

    def _restore_terminal(self) -> None:
        if self._old_term is None:
            return
        import termios

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_term)
        self._old_term = None
