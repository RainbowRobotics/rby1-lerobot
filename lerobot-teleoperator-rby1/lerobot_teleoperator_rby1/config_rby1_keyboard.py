"""Dataclass-based configuration for the joystick mobile-base teleoperator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lerobot.teleoperators.config import TeleoperatorConfig

# Head-pitch (head_1) ready angle, in radians — the fixed commanded pitch so
# the head holds a neutral angle (mirrors ``lerobot_robot_rby1.constants.
# READY_HEAD`` pitch of 49 deg). This joystick teleop does not steer the head;
# it emits this constant so the follower's head stays put instead of jumping.
_READY_HEAD_PITCH_RAD: float = float(np.deg2rad(25.0))


@TeleoperatorConfig.register_subclass("rby1_keyboard")
@dataclass
class Rby1KeyboardConfig(TeleoperatorConfig):
    """Configuration for the RB-Y1 joystick mobile-base teleoperator.

    This teleoperator receives joystick data over UDP from the Legion Go's
    ``joystick_sender.py`` (pygame axes packed as ``{N}f{M}B{4}B`` — N float
    axes, M button bytes, 4 hat bytes) and only *produces* mobile-base velocity
    actions (``x.vel`` / ``y.vel`` / ``theta.vel``) plus a fixed head-pitch
    position (``head_1``, radians). Following the LeRobot convention it never
    commands the robot itself — the paired ``lerobot_robot_rby1.Rby1`` follower
    executes the velocities via ``send_action``.

    Axis mapping (body frame) — the pygame axis *index* for each is
    configurable below, since the layout is controller-specific::

        axis[axis_x_index]     linear x   (forward / backward)
        axis[axis_y_index]     linear y   (strafe left / right)
        axis[axis_yaw_index]   yaw rate   (turn left / right)

    The registration name stays ``rby1_keyboard`` for backward compatibility
    (``--teleop.type rby1_keyboard``) even though the input is now a joystick.
    """

    # ── UDP receiver ─────────────────────────────────────────────────
    # Address/port to bind for incoming joystick packets. "0.0.0.0" listens on
    # every interface; the port must match joystick_sender.py's UDP_PORT.
    udp_ip: str = "0.0.0.0"
    udp_port: int = 5005

    # Stop the base if no packet arrives within this many seconds (WiFi drop /
    # sender crash safety). The head still holds its fixed angle.
    command_timeout: float = 0.5

    # ── Axis index mapping (pygame axis order) ───────────────────────
    # Which pygame axis index drives each body-frame component. Defaults match
    # the Legion Go's observed pygame layout:
    #   0=left-X, 1=left-Y, 2=LEFT TRIGGER, 3=right-X, 4=right-Y, 5=RIGHT TRIGGER
    # WARNING: the trigger axes (2 and 5) rest at -1.0, not 0.0 — never map a
    # motion axis onto a trigger or the base runs at full speed while idle.
    # Verify against the axis_data printed by joystick_sender.py.
    axis_x_index: int = 4    # right stick vertical   -> forward/backward
    axis_y_index: int = 3    # right stick horizontal -> strafe left/right
    axis_yaw_index: int = 0  # left stick horizontal  -> turn left/right

    # Total number of float axes in each packet (joystick_sender.py sends one
    # float per pygame axis, before the button/hat bytes). Needed to locate the
    # button bytes that follow the axes; must exceed every axis index above.
    # The Legion Go reports 6 axes.
    num_axes: int = 6

    # ── Endpoint-state toggle button ─────────────────────────────────
    # Button index (into joystick_sender.py's button_data) whose rising edge
    # toggles the endpoint-state observation between 0 and 1. The Legion Go A
    # button is the first entry, button_data[0]. Set to -1 to disable. The value
    # is published to lerobot_robot_rby1.endpoint_state and recorded as the
    # "endpoint_state" observation when the robot's use_endpoint_state is enabled.
    toggle_button_index: int = 0

    # Diagnostic: log the joystick packet layout on the first packet and the
    # pressed button indices whenever they change. Use this to verify num_axes
    # and find the real toggle_button_index on your controller, then turn it
    # off. Leave False for normal use.
    log_buttons: bool = False

    # Normalized stick deadzone in [0, 1). Axis magnitudes below this (pygame
    # already reports [-1, 1]) are treated as zero to reject resting drift.
    joystick_deadzone: float = 0.08

    # Per-axis sign inversion, applied after the deadzone. Flip these on the
    # real hardware if a stick drives the base the wrong way (e.g. pygame
    # reports stick-up as negative, so forward often needs invert_x_axis=True).
    invert_x_axis: bool = True
    invert_y_axis: bool = True
    invert_yaw_axis: bool = True

    # ── Speed limits ─────────────────────────────────────────────────
    # Body-frame linear speed at full stick deflection (m/s).
    max_linear_speed: float = 0.7
    # Yaw rate at full stick deflection (rad/s).
    max_angular_speed: float = 1.0

    # ── Smoothing ────────────────────────────────────────────────────
    # Low-pass-filter coefficient in (0, 1]. Each tick the commanded velocity
    # moves this fraction of the way toward the stick-derived target: higher is
    # snappier, lower is smoother. 1.0 disables smoothing.
    smoothing: float = 0.1

    # ── Head pitch (head_1 joint, radians) ───────────────────────────
    # The joystick does not steer the head; head_1 is emitted at this fixed
    # angle every tick so the follower holds a neutral head pose.
    head_pitch_init: float = _READY_HEAD_PITCH_RAD

    def __post_init__(self) -> None:
        if self.max_linear_speed < 0:
            raise ValueError("max_linear_speed must be >= 0")
        if self.max_angular_speed < 0:
            raise ValueError("max_angular_speed must be >= 0")
        if not 0.0 < self.smoothing <= 1.0:
            raise ValueError("smoothing must be in (0, 1]")
        if not 0.0 <= self.joystick_deadzone < 1.0:
            raise ValueError("joystick_deadzone must be in [0, 1)")
        if not (0 < self.udp_port < 65536):
            raise ValueError("udp_port must be in (0, 65536)")
        if self.command_timeout <= 0:
            raise ValueError("command_timeout must be > 0")
        if min(self.axis_x_index, self.axis_y_index, self.axis_yaw_index) < 0:
            raise ValueError("axis indices must be >= 0")
        if self.num_axes <= max(
            self.axis_x_index, self.axis_y_index, self.axis_yaw_index
        ):
            raise ValueError("num_axes must exceed every axis index")
