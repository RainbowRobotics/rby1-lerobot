"""100 Hz background control loop for the RB-Y1 leader arm.

Encapsulates the cosine-interpolated init trajectory, gravity compensation,
joint-limit barriers, viscous damping, and button-driven free/hold mode
switching that runs inside the rby1_sdk control callback.  State that used
to be carried via one-element-list closures in ``rby1_leader_arm.connect()``
is now plain instance attributes on this class.

Thread-safety
-------------
Reads/writes to the *teleop-shared* state (``teleop._right_q``,
``teleop._left_q``, triggers, ``_init_done``, ``_reset_requested``) go
through ``teleop._lock``.  All other state in this class is touched only by
the SDK's control thread, which calls :meth:`__call__` serially.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import numpy as np
import time

if TYPE_CHECKING:  # pragma: no cover
    from .rby1_leader_arm import Rby1LeaderArm


@dataclass
class _ArmTrajState:
    """Mutable per-arm bookkeeping for the init trajectory and hold pose."""

    init_start_q: np.ndarray | None = None     # Initial joint state at t=0
    init_target_q: np.ndarray | None = None    # Target joint state at t=duration
    hold_q: np.ndarray | None = None           # Latest "hold" command
    reset_anchor_q: np.ndarray | None = None   # Anchor pose for skip-reset arms


@dataclass
class LeaderArmControlLoop:
    """Stateful callable plugged into ``MasterArm.start_control(...)``."""

    teleop: "Rby1LeaderArm"
    rby: Any  # rby1_sdk module

    # Pre-converted numerical parameters (radians, etc.)
    right_init_q: np.ndarray
    left_init_q: np.ndarray
    min_q: np.ndarray
    max_q: np.ndarray
    torque_limit: np.ndarray
    viscous_gain: np.ndarray
    barrier: float
    gravity_scale: float
    init_duration: float

    # Mutable runtime state
    init_t0: float | None = None
    right_state: _ArmTrajState = field(default_factory=_ArmTrajState)
    left_state: _ArmTrajState = field(default_factory=_ArmTrajState)
    reset_right_active: bool = True
    reset_left_active: bool = True
    init_done: bool = False

    # ------------------------------------------------------------------ #

    def __call__(self, state: Any) -> Any:
        """Single tick of the leader-arm control loop."""
        ma_input = self.rby.upc.MasterArm.ControlInput()
        teleop = self.teleop

        # 1. Pick up a pending reset request from the main thread.
        with teleop._lock:
            reset_requested = teleop._reset_requested
            if reset_requested:
                self.reset_right_active = teleop._reset_right_arm_requested
                self.reset_left_active = teleop._reset_left_arm_requested
                self.right_state.reset_anchor_q = teleop._right_q.copy()
                self.left_state.reset_anchor_q = teleop._left_q.copy()
                teleop._reset_requested = False

        if reset_requested:
            self._begin_init_trajectory()
            teleop._init_done = False

        # 2. Lazily latch the init-trajectory starting point.
        if self.init_t0 is None:
            self._latch_init_start(state)

        # 3. Either run the cosine init or the steady-state controller.
        elapsed = time.time() - self.init_t0
        if elapsed < self.init_duration:
            self._fill_init_command(ma_input, state, elapsed)
        else:
            self._fill_steady_command(ma_input, state)

        return ma_input

    # ------------------------------------------------------------------ #
    # Init trajectory
    # ------------------------------------------------------------------ #

    def _begin_init_trajectory(self) -> None:
        self.init_t0 = None
        self.right_state.init_start_q = None
        self.left_state.init_start_q = None
        self.right_state.hold_q = None
        self.left_state.hold_q = None
        self.init_done = False

    def _latch_init_start(self, state: Any) -> None:
        self.init_t0 = time.time()
        self.right_state.init_target_q = self.right_init_q.copy()
        self.left_state.init_target_q = self.left_init_q.copy()
        self.right_state.init_start_q = self._pick_start_q(
            state.q_joint[0:7], self.reset_right_active, self.right_state.reset_anchor_q
        )
        self.left_state.init_start_q = self._pick_start_q(
            state.q_joint[7:14], self.reset_left_active, self.left_state.reset_anchor_q
        )

    @staticmethod
    def _pick_start_q(current_q: np.ndarray, reset_active: bool, anchor_q: np.ndarray | None) -> np.ndarray:
        """Choose where the init trajectory starts from."""
        if reset_active or anchor_q is None:
            return current_q.copy()
        return anchor_q.copy()

    def _fill_init_command(self, ma_input: Any, state: Any, elapsed: float) -> None:
        s = 0.5 * (1.0 - np.cos(np.pi * elapsed / self.init_duration))
        right_target = (
            self.right_state.init_target_q
            if self.reset_right_active
            else self.right_state.init_start_q
        )
        left_target = (
            self.left_state.init_target_q
            if self.reset_left_active
            else self.left_state.init_start_q
        )
        right_q_cmd = self.right_state.init_start_q + s * (right_target - self.right_state.init_start_q)
        left_q_cmd = self.left_state.init_start_q + s * (left_target - self.left_state.init_start_q)

        ma_input.target_operating_mode.fill(
            self.rby.DynamixelBus.CurrentBasedPositionControlMode
        )
        ma_input.target_position[0:7] = right_q_cmd
        ma_input.target_position[7:14] = left_q_cmd
        ma_input.target_torque[:] = self.torque_limit

        with self.teleop._lock:
            self.teleop._right_q = right_q_cmd.copy()
            self.teleop._left_q = left_q_cmd.copy()
            self.teleop._right_trigger = float(state.button_right.trigger)
            self.teleop._left_trigger = float(state.button_left.trigger)

    # ------------------------------------------------------------------ #
    # Steady-state controller (after init)
    # ------------------------------------------------------------------ #

    def _fill_steady_command(self, ma_input: Any, state: Any) -> None:
        if not self.init_done:
            self.right_state.hold_q = self._pick_hold_q(
                state.q_joint[0:7], self.reset_right_active,
                self.right_state.reset_anchor_q, self.right_init_q,
            )
            self.left_state.hold_q = self._pick_hold_q(
                state.q_joint[7:14], self.reset_left_active,
                self.left_state.reset_anchor_q, self.left_init_q,
            )
            self.init_done = True
            self.teleop._init_done = True

        # Triggers (always read).
        with self.teleop._lock:
            self.teleop._right_trigger = float(state.button_right.trigger)
            self.teleop._left_trigger = float(state.button_left.trigger)

        # Joint torques: gravity + barrier + viscous damping.
        torque = (
            state.gravity_term
            + self.barrier * (
                np.maximum(self.min_q - state.q_joint, 0)
                + np.minimum(self.max_q - state.q_joint, 0)
            )
            + self.viscous_gain * state.qvel_joint
        )
        torque = np.clip(torque, -self.torque_limit, self.torque_limit)

        self._fill_arm_command(ma_input, state, torque, side="right", slice_=slice(0, 7))
        self._fill_arm_command(ma_input, state, torque, side="left", slice_=slice(7, 14))

        with self.teleop._lock:
            self.teleop._right_q = self.right_state.hold_q.copy()
            self.teleop._left_q = self.left_state.hold_q.copy()

    @staticmethod
    def _pick_hold_q(
        current_q: np.ndarray,
        reset_active: bool,
        anchor_q: np.ndarray | None,
        init_q: np.ndarray,
    ) -> np.ndarray:
        if reset_active:
            return init_q.copy()
        if anchor_q is not None:
            return anchor_q.copy()
        return current_q.copy()

    def _fill_arm_command(
        self,
        ma_input: Any,
        state: Any,
        torque: np.ndarray,
        side: str,
        slice_: slice,
    ) -> None:
        button = state.button_right.button if side == "right" else state.button_left.button
        arm_state = self.right_state if side == "right" else self.left_state

        if button == 1:
            # Free mode: gravity compensation only.
            ma_input.target_operating_mode[slice_].fill(
                self.rby.DynamixelBus.CurrentControlMode
            )
            ma_input.target_torque[slice_] = torque[slice_] * self.gravity_scale
            arm_state.hold_q = state.q_joint[slice_].copy()
        else:
            # Hold mode: keep the last commanded position.
            ma_input.target_operating_mode[slice_].fill(
                self.rby.DynamixelBus.CurrentBasedPositionControlMode
            )
            ma_input.target_torque[slice_] = self.torque_limit[slice_]
            ma_input.target_position[slice_] = arm_state.hold_q
