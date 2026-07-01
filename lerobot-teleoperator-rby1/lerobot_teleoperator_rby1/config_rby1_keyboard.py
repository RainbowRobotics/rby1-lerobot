"""Dataclass-based configuration for the keyboard mobile-base teleoperator."""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("rby1_keyboard")
@dataclass
class Rby1KeyboardConfig(TeleoperatorConfig):
    """Configuration for the RB-Y1 keyboard mobile-base teleoperator.

    This teleoperator only *produces* mobile-base velocity actions
    (``x.vel`` / ``y.vel`` / ``theta.vel``) from keyboard presses read off the
    controlling terminal. Following the LeRobot convention it never commands
    the robot itself — the paired ``lerobot_robot_rby1.Rby1`` follower executes
    the velocities via ``send_action``.

    Key mapping (body frame)::

        8 / 2   linear x  + / -   (forward / backward)
        4 / 6   linear y  + / -   (strafe left / right)
        s / f   yaw rate  + / -   (turn left / right)
    """

    # ── Speed limits ─────────────────────────────────────────────────
    # Target body-frame linear speed while a direction key is held (m/s).
    max_linear_speed: float = 0.8
    # Target yaw rate while a rotation key is held (rad/s).
    max_angular_speed: float = 1.0

    # ── Smoothing ────────────────────────────────────────────────────
    # Low-pass-filter coefficient in (0, 1]. Each tick the commanded velocity
    # moves this fraction of the way toward the key-derived target: higher is
    # snappier, lower is smoother. 1.0 disables smoothing.
    smoothing: float = 0.1

    def __post_init__(self) -> None:
        if self.max_linear_speed < 0:
            raise ValueError("max_linear_speed must be >= 0")
        if self.max_angular_speed < 0:
            raise ValueError("max_angular_speed must be >= 0")
        if not 0.0 < self.smoothing <= 1.0:
            raise ValueError("smoothing must be in (0, 1]")
