"""The XR teleoperation logic of the RB-Y1, one tick at a time.

:class:`XrTeleopCore` turns the operator's XR input into robot targets. It
knows nothing about where the input comes from or where the targets go: a
host feeds it one :class:`~rby1_xr_teleop.xr_frame.XRFrame` and the robot's
measured :class:`~rby1_xr_teleop.state.RobotState` per tick and receives a
:class:`~rby1_xr_teleop.command.TeleopCommand`.

    core = XrTeleopCore(settings, RobotInfo(version="1.3"))
    core.reset(state)                      # targets = where the robot is
    command = core.step(frame, state, now)

Inputs: the controllers move the arms (absolute, shoulder-relative mapping
with a grip dead-man), the body the torso, the headset the head, the
thumbsticks or the body (Left X) the base. Right A starts, re-references and
returns to the start pose; Right B stops. See
:class:`~rby1_xr_teleop.settings.XrTeleopSettings` for every option.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import replace

import numpy as np

from .absolute_ee import AbsoluteEeMapper, rpy_deg_to_matrix
from .arm_retargeter import ArmPostureRetargeter, robot_reach
from .clutch import Clutch, PoseHold
from .command import HINT_JOINTS, TeleopCommand
from .gating import Gates, Mode, compute_gates
from .operator_frame import OperatorFrame, shoulder_line_yaw
from .retargeters import (
    HeadRetargeter,
    body_drive_velocity,
    chest_pose_from_body,
    clamp_pose_box,
    interpolate_pose,
    se2_relative,
    smoothstep,
    thumbsticks_to_base_vel,
    wrap_pi,
)
from .settings import XrTeleopSettings
from .so3 import quat_from_matrix, rotvec_from_matrix
from .state import RobotInfo, RobotState
from .xr_frame import BodyJointIndex, ButtonEdge, ControllerState, XRFrame

logger = logging.getLogger(__name__)

# Fewest shoulder -> controller samples an arm-reach calibration accepts.
_CALIBRATION_MIN_SAMPLES = 5
# A later Right A replaces the reach only by a value at least this fraction of
# the current one: an arm that is not extended can only measure short.
_CALIBRATION_KEEP_RATIO = 0.95


# shoulder_source → the inputs an arm's operator shoulder may come from, in order of preference.
_SHOULDER_SOURCES = {"auto": ("body", "headset"), "body": ("body",), "headset": ("headset",)}
_SHOULDER_SOURCE_NAMES = {"body": "body tracking", "headset": "the headset estimate"}


class XrTeleopCore:
    """XR input + measured robot state -> robot targets, one :meth:`step` per tick."""

    def __init__(
        self,
        settings: XrTeleopSettings,
        robot: RobotInfo | None = None,
        *,
        on_arm_calibrated: Callable[[dict[str, dict[str, float]]], None] | None = None,
    ) -> None:
        self.settings = settings
        self.robot = robot or RobotInfo()
        self._on_arm_calibrated = on_arm_calibrated
        self._now = 0.0
        self._ui_active: Mapping[str, bool] = {}
        self._last_frame: XRFrame | None = None
        self._edges = ButtonEdge()

        # Arm targets (absolute mapping) and the torso's engage-relative clutch.
        self._clutch: dict[str, Clutch | PoseHold | None] = {"right": None, "left": None, "torso": None}
        self._torso_target: np.ndarray | None = None
        self._head = self._make_head_retargeter()
        self._gripper = {"right": 1.0, "left": 1.0}  # dataset convention: 1 = open
        self._base_vel = (0.0, 0.0, 0.0)
        self._stopped = False
        # start_on_right_a: nothing follows the headset until the first Right A.
        self._started = not settings.start_on_right_a
        self._warned_no_body = False
        self._torso_hold_reason = "not started"
        self._needs_initial_resync = True
        # Start pose (first action) and the interpolated return motion (Right A).
        self._start_snapshot: RobotState | None = None
        self._return_motion: _ReturnMotion | None = None
        # Which way the operator faces: yaw correction applied to every incoming pose.
        self._frame = OperatorFrame(reference_source=settings.reference_source, wear_mode=settings.wear_mode)

        self._torso_joint = int(BodyJointIndex[settings.torso_body_joint])
        self._torso_required = [int(BodyJointIndex[n]) for n in settings.body_required_joints]

        # Absolute arm mapping / posture hints, per enabled arm.
        self._abs: dict[str, AbsoluteEeMapper] = {}
        self._abs_ramp: dict[str, tuple[float, np.ndarray, float] | None] = {"right": None, "left": None}
        self._abs_state: dict[str, str] = {"right": "hold", "left": "hold"}
        self._abs_last_body_t: dict[str, float | None] = {"right": None, "left": None}
        # Where each arm's operator shoulder comes from ("body" / "headset").
        # _shoulder_fed: the source that last updated it. _shoulder_locked: the
        # source the arm engaged with; from then on only that one is used, until
        # the operator frame is re-referenced (Right A). Switching sources moves
        # the shoulder, and with it the hand target, by several centimetres.
        self._shoulder_fed: dict[str, str | None] = {"right": None, "left": None}
        self._shoulder_locked: dict[str, str | None] = {"right": None, "left": None}
        self._posture: dict[str, ArmPostureRetargeter] = {}
        self._hints: dict[str, np.ndarray | None] = {"right": None, "left": None}
        # head_base_require_grip: a grip is squeezed (dead-man of head and base).
        self._grip_held = False
        # Body drive (Left X held): entry state, or None. _body_drive_blocked
        # holds it off after an abort until Left X is released.
        self._body_drive: _BodyDrive | None = None
        self._body_drive_blocked = False
        # arm_length_source="calibrate": recent (time, |controller - shoulder|) per arm.
        self._reach_samples: dict[str, deque[tuple[float, float]]] = {"right": deque(), "left": deque()}
        self._last_snapshot: RobotState | None = None
        # Per arm: latch the controller↦gripper offsets at the next engage.
        self._needs_offset_latch: dict[str, bool] = {"right": True, "left": True}
        # Head joints the current look direction must map onto (latched on the
        # next tracked head frame; None = nothing pending).
        # (head joints, base->torso_5 pose those joints were measured / defined under)
        self._head_latch_target: tuple[np.ndarray, np.ndarray] | None = None
        self._last_tick_t: float | None = None
        # Neck mode: smoothed headset pose driving the torso.
        self._neck_driver: np.ndarray | None = None
        self._warned_neck_torso = False
        self._init_arm_mappers()

    def _init_arm_mappers(self) -> None:
        cfg = self.settings
        version = self.robot.version if self.robot.version in ("1.2", "1.3") else "1.3"
        reach = robot_reach(version)
        calibrate = cfg.arm_length_source == "calibrate"
        offset = rpy_deg_to_matrix(cfg.ee_orientation_offset_rpy_deg)
        for side, enabled in (("right", cfg.use_right_arm), ("left", cfg.use_left_arm)):
            if not enabled:
                continue
            side_reach = reach
            # The calibrated operator reach is shoulder -> controller, so the
            # robot side is shoulder -> the commanded link (link_<side>_arm_6).
            if calibrate and self.robot.arm_reach is not None and side in self.robot.arm_reach:
                side_reach = float(self.robot.arm_reach[side])
            self._abs[side] = AbsoluteEeMapper(
                side,
                robot_reach=side_reach,
                estimate_reach=not calibrate,
                human_reach=cfg.human_arm_length_m if cfg.arm_length_source == "config" else None,
                position_scale=cfg.ee_position_scale,
                reach_max_ratio=cfg.ee_reach_max_ratio,
                shoulder_smoothing=cfg.shoulder_smoothing,
                max_linear_vel=cfg.ee_max_linear_vel,
                max_angular_vel=cfg.ee_max_angular_vel,
                orientation_offset=offset,
                fallback_reach=cfg.human_arm_length_m,
            )
            if cfg.arm_posture_hint:
                self._posture[side] = ArmPostureRetargeter(
                    side,
                    version=version,
                    smoothing=cfg.posture_hint_smoothing,
                    max_vel=cfg.posture_hint_max_vel,
                    hold_s=cfg.posture_hint_hold_s,
                )
        logger.info(
            "absolute arm mapping, posture_hint=%s (robot version %s, reach %s m, operator reach: %s)",
            cfg.arm_posture_hint, version,
            "/".join(f"{m.robot_reach:.3f}" for m in self._abs.values()) or f"{reach:.3f}",
            cfg.arm_length_source,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def started(self) -> bool:
        """False while waiting for the first Right A (``start_on_right_a``)."""
        return self._started

    @property
    def stopped(self) -> bool:
        """True after Right B until the next Right A."""
        return self._stopped

    @property
    def last_frame(self) -> XRFrame | None:
        """The input of the last :meth:`step`, in the operator frame."""
        return self._last_frame

    def arm_engaged(self, side: str) -> bool:
        """Whether that arm is following its controller."""
        hold = self._clutch.get(side)
        return hold is not None and hold.engaged

    def reset(self, state: RobotState) -> None:
        """Seed every target from the measured robot pose so nothing moves yet."""
        snap = state
        cfg = self.settings
        self._clutch["right"] = PoseHold(snap.right_ee) if cfg.use_right_arm else None
        self._clutch["left"] = PoseHold(snap.left_ee) if cfg.use_left_arm else None
        self._clutch["torso"] = Clutch(snap.torso) if cfg.use_torso else None
        self._torso_target = snap.torso.copy()
        self._head.hold(snap.head_q, snap.torso)
        self._gripper = {"right": 1.0, "left": 1.0}
        self._base_vel = (0.0, 0.0, 0.0)
        self._stopped = False
        self._started = not cfg.start_on_right_a
        self._body_drive = None
        self._body_drive_blocked = False
        self._grip_held = False
        self._shoulder_locked = {side: None for side in self._shoulder_locked}
        # The follower connects (and moves to its ready pose) after this, so
        # the targets are re-seeded again on the first get_action().
        self._needs_initial_resync = True
        self._frame.needs_reference = True
        logger.info(
            "targets latched to the current robot pose (%s).",
            ("hold both arms straight out in front and press Right A to start, then squeeze to follow"
             if cfg.arm_length_source == "calibrate"
             else "press Right A to start, then squeeze to follow")
            if cfg.start_on_right_a else "squeeze to follow",
        )

    def step(
        self,
        raw: XRFrame,
        state: RobotState,
        now: float,
        *,
        grabbing: Mapping[str, str | None] | None = None,
        ui_active: Mapping[str, bool] | None = None,
    ) -> TeleopCommand:
        """One tick: ``raw`` XR input (robot-base axes, before the operator frame), measured ``state``.

        ``now`` is a monotonic time in seconds. ``grabbing`` / ``ui_active``
        name the hands that currently operate a user-interface element (a
        camera panel) instead of the robot: a grabbing hand neither clutches
        nor drives the base, a UI-active free hand keeps its gripper value.
        """
        self._now = now
        self._ui_active = ui_active or {}
        snap = state
        events = self._edges.update(raw)

        # Operator frame: reference on the first tracked head frame and on Right A.
        if events["right_a"] or self._frame.needs_reference:
            if self._frame.reference(raw, on_button=bool(events["right_a"])):
                self._on_operator_frame_changed()

        # Engage edges latch from the measured state and disengaged components
        # are re-synchronised to it when the robot moved on its own.
        self._last_snapshot = snap
        self._resync_disengaged(snap)
        raw = mask_grabbing_hands(raw, grabbing or {})

        if any(events.values()):
            logger.info("xr button edge: %s", [k for k, v in events.items() if v])
        if events["right_b"]:
            logger.info("Right B — stopping: targets frozen, base zeroed.")
            self._stopped = True
            for clutch in self._clutch.values():
                if clutch is not None:
                    clutch.disengage()
        elif events["right_a"] and not self._started:
            if self._calibrate_arms():
                self._start(snap)
        elif events["right_a"]:
            self._calibrate_arms()
            if self._stopped:
                logger.info("Right A — resuming.")
            self._stopped = False
            self._start_return_motion(snap)
            if self.settings.use_head and self.settings.head_latch_on_a and self._start_snapshot is not None:
                start = self._start_snapshot
                self._head_latch_target = (start.head_q.copy(), start.torso.copy())

        self._update_grip_held(raw, snap)
        self._update_body_drive(raw, snap)  # may turn the operator frame with the body
        # Every pose in the operator frame, once, now that the frame is final for this tick.
        frame = self._frame.apply(raw)

        dt = 1.0 / 60.0 if self._last_tick_t is None else float(np.clip(now - self._last_tick_t, 1e-3, 0.1))
        self._last_tick_t = now

        if self._head_latch_target is not None:
            self._latch_head_offset(frame)

        self._advance_return_motion(frame, snap)
        # What may move this tick. Evaluated after the return motion advanced
        # (it ends on its second tick) and after the body drive settled.
        gates = self._gates()
        self._update_arm_absolute("right", frame, snap, now, dt, gates)
        self._update_arm_absolute("left", frame, snap, now, dt, gates)
        self._update_posture_hints(frame, snap, now, dt, gates)
        self._update_torso(frame, snap, gates)
        self._update_head(frame, snap, gates)
        self._update_grippers(frame, gates)
        self._update_base(frame, gates)
        self._last_frame = frame
        return self._command()

    def _command(self) -> TeleopCommand:
        cfg = self.settings
        command = TeleopCommand()
        motion = self._return_motion
        if motion is not None and motion.reset_emitted and not motion.finished:
            # One-shot request; the robot runs its own return-to-ready motion.
            command.reset_seconds = float(motion.duration)
        if cfg.use_torso:
            command.torso = self._torso_target
        if cfg.use_right_arm:
            command.right_ee = self._clutch["right"].last_commanded
        if cfg.use_left_arm:
            command.left_ee = self._clutch["left"].last_commanded
        if cfg.use_gripper:
            if cfg.use_right_arm:
                command.right_gripper = self._gripper["right"]
            if cfg.use_left_arm:
                command.left_gripper = self._gripper["left"]
        if cfg.use_mobile_base:
            command.base = self._base_vel
        snap = self._last_snapshot
        if cfg.use_head:
            # Re-solve the held / tracked gaze for the torso orientation of this tick.
            head_q = self._head.target_for(None if snap is None else snap.torso)
            command.head = np.zeros(2) if head_q is None else np.asarray(head_q, dtype=float)
        if cfg.arm_posture_hint:
            for side, h in self._hints.items():
                if side not in self._posture:
                    continue
                if h is None and cfg.record_posture_hint:
                    # Recorded features must always be present. A neutral value
                    # is the arm's measured posture (the solver holds what it
                    # has) — never zeros, which would drag the arm to q = 0.
                    if snap is not None:
                        h = (snap.right_q if side == "right" else snap.left_q)[:HINT_JOINTS]
                if h is not None:
                    command.hints[side] = np.asarray(h, dtype=float)[:HINT_JOINTS]
        return command

    def arm_calibration(self) -> dict[str, dict[str, float]]:
        """The fixed operator reach per arm with the resulting scale (``arm_length_source="calibrate"``)."""
        return {
            side: {
                "operator_reach_m": m.calibrated_reach,
                "robot_reach_m": m.robot_reach,
                "scale": m.robot_reach / m.calibrated_reach * m.position_scale,
            }
            for side, m in self._abs.items()
        }

    def status_line(self, *, headset: str | None = None, extra: str = "") -> str:
        """One line: what is tracked, what is clutched, why the torso holds.

        ``headset`` replaces the head-tracking flag (a host that knows the
        headset's link state passes it); ``extra`` is appended.
        """
        frame = self._last_frame if self._last_frame is not None else XRFrame()

        def ctrl(c: ControllerState | None) -> str:
            if c is None:
                return "-"
            return (
                f"sq={c.squeeze:.2f} tr={c.trigger:.2f} "
                f"stick=({c.thumbstick[0]:+.2f},{c.thumbstick[1]:+.2f}) "
                f"A={int(c.primary)} B={int(c.secondary)}"
            )

        def eng(side: str) -> str:
            c = self._clutch.get(side)
            if c is None:
                return "n/a"
            return self._abs_state[side].upper() if c.engaged else self._abs_state[side]

        def hint(side: str) -> str:
            h = self._hints.get(side)
            if not self._posture:
                return ""
            return f" hint{side[0].upper()}=" + ("-" if h is None else str(np.rad2deg(h).round(0).astype(int).tolist()))

        body = "-"
        if frame.body is not None:
            body = f"valid {int(frame.body.valid.sum())}/24"
        elif self.settings.torso_source == "body":
            body = "none"
        vx, vy, wz = self._base_vel
        return "xr status | right: %s [%s] | left: %s [%s] | head: %s | body: %s | torso: %s | base=(%.2f,%.2f,%.2f)%s%s%s" % (
            ctrl(frame.right),
            eng("right"),
            ctrl(frame.left),
            eng("left"),
            headset if headset is not None else ("ok" if frame.head is not None else "-"),
            body,
            self._torso_hold_reason if self._clutch.get("torso") is not None else "disabled",
            vx,
            vy,
            wz,
            hint("right"),
            hint("left"),
            self._reach_status()
            + self._body_drive_status()
            + (
                " | head/base hold (no grip)"
                if self._started and self.settings.head_base_require_grip and not self._grip_held
                else ""
            )
            + ("" if self._started else " | NOT STARTED (press Right A)")
            + (" | STOPPED" if self._stopped else "")
            + (f" | wear={self.settings.wear_mode}" if self.settings.wear_mode != "head" else "")
            + (f" | {extra}" if extra else ""),
        )

    # ------------------------------------------------------------------
    # Operator frame, mode and gates
    # ------------------------------------------------------------------

    def _on_operator_frame_changed(self, turned_by: float | None = None) -> None:
        """Keep what is derived from the operator frame consistent with it.

        ``turned_by`` (rad): the frame followed the body's turn (body drive) —
        the smoothed shoulders turn with it and everything else stays valid.
        ``None``: the frame was re-referenced — everything derived from it
        starts over.
        """
        if turned_by is not None:
            for mapper in self._abs.values():
                mapper.rotate_shoulder(turned_by)
            return
        for mapper in self._abs.values():
            mapper.reset_shoulder()
        self._shoulder_fed = {side: None for side in self._shoulder_fed}
        self._shoulder_locked = {side: None for side in self._shoulder_locked}
        for retargeter in self._posture.values():
            retargeter.reset()
        # Hints derived from the old frame must not be emitted any more.
        self._hints = {side: None for side in self._hints}
        self._needs_offset_latch = {side: True for side in self._needs_offset_latch}

    def _mode(self) -> Mode:
        return Mode(
            started=self._started,
            stopped=self._stopped,
            returning=self._returning,
            grip_held=self._grip_held,
            body_drive=self._body_drive is not None,
        )

    def _gates(self) -> Gates:
        """What may follow the operator right now (the one place that decides it: ``gating``)."""
        cfg = self.settings
        return compute_gates(
            self._mode(),
            head_base_require_grip=cfg.head_base_require_grip,
            use_mobile_base=cfg.use_mobile_base,
            body_drive_enabled=cfg.body_drive_enabled,
        )

    def _update_grip_held(self, frame: XRFrame, snap: RobotState) -> None:
        """Dead-man of the head and the base: at least one grip squeezed (``head_base_require_grip``)."""
        cfg = self.settings
        held = any(
            c is not None and c.squeeze > cfg.clutch_threshold for c in (frame.right, frame.left)
        )
        pressed = held and not self._grip_held
        self._grip_held = held
        if pressed and cfg.head_base_require_grip and self._mode().active:
            # The head held while the grips were released: start from where it is.
            self._relatch_head(snap)

    def _relatch_head(self, snap: RobotState) -> None:
        """Grip squeezed again: the head starts from where it is (``head_latch_on_a`` only)."""
        cfg = self.settings
        if not cfg.use_head or cfg.wear_mode == "neck" or not cfg.head_latch_on_a:
            return
        self._hold_head_at(snap.head_q, snap.torso)

    def _hold_head_at(self, head_q: np.ndarray, torso: np.ndarray, *, relatch: bool = True) -> None:
        """Hold the head at ``head_q`` (measured under the base->torso_5 pose ``torso``) with a fresh origin.

        With ``relatch`` and ``head_latch_on_a``, the look direction of the next
        tracked head frame is mapped onto these joints, so the head continues
        from here instead of jumping to the absolute headset direction.
        """
        self._head = self._make_head_retargeter()
        self._head.hold(head_q, torso)
        if relatch and self.settings.head_latch_on_a:
            self._head_latch_target = (head_q.copy(), torso.copy())

    # ------------------------------------------------------------------
    # Body drive: Left X held -> the base follows the operator 1:1
    # ------------------------------------------------------------------

    def _end_body_drive(self, reason: str, *, block: bool = False) -> None:
        if self._body_drive is None:
            return
        self._body_drive = None
        self._body_drive_blocked = block
        self._base_vel = (0.0, 0.0, 0.0)
        (logger.warning if block else logger.info)(
            "Base drive ended (%s)%s.", reason, " — release Left X and press it again to resume" if block else ""
        )

    def _update_body_drive(self, raw: XRFrame, snap: RobotState) -> None:
        """Run the body-drive mode for this tick (it may turn the operator frame).

        While Left X is held the base follows the operator's displacement
        (headset, horizontal) and turn (shoulder line) since the press, measured
        against the robot's odometry. The operator frame follows the body's
        turn, so head and arm commands stay relative to the body.
        """
        cfg = self.settings
        pressed = raw.left is not None and raw.left.primary
        if not pressed:
            self._end_body_drive("Left X released")
            self._body_drive_blocked = False
            return
        if self._body_drive_blocked:
            return
        if not self._gates().body_drive:
            self._end_body_drive("stopped, or no grip squeezed")
            return
        body_yaw = shoulder_line_yaw(raw)
        odometry = getattr(snap, "odometry", None)
        tracked = raw.head is not None and body_yaw is not None and odometry is not None
        drive = self._body_drive
        if drive is None:
            if not tracked:
                return  # entered on the first tick with headset, shoulders and odometry
            drive = self._body_drive = _BodyDrive(
                origin_xy=raw.head.position[:2].copy(),
                body_yaw=body_yaw,
                yaw_correction0=self._frame.yaw,
                odometry0=np.asarray(odometry, dtype=float).copy(),
            )
            torso = self._clutch.get("torso")
            if torso is not None:
                torso.disengage()
            logger.info("Base drive (Left X held): the torso holds, the base follows your steps and turns.")
        if not tracked:
            self._end_body_drive("headset, shoulders or odometry lost", block=True)
            return

        # Operator pose since the press, in the base frame of that moment (yaw unwrapped, lightly filtered).
        a = cfg.body_drive_smoothing
        drive.turn_raw += wrap_pi(body_yaw - drive.body_yaw)
        drive.body_yaw = body_yaw
        c0, s0 = math.cos(drive.yaw_correction0), math.sin(drive.yaw_correction0)
        d = raw.head.position[:2] - drive.origin_xy
        step = np.array([c0 * d[0] - s0 * d[1], s0 * d[0] + c0 * d[1]])
        drive.turn += a * (drive.turn_raw - drive.turn)
        drive.step += a * (step - drive.step)
        # Robot pose since the press, from its odometry (yaw unwrapped).
        x, y, yaw = se2_relative(drive.odometry0, odometry)
        drive.robot_turn += wrap_pi(yaw - drive.robot_yaw_wrapped)
        drive.robot_yaw_wrapped = yaw

        velocity, error = body_drive_velocity(
            drive.step, drive.turn, (x, y), drive.robot_turn,
            kp_linear=cfg.body_drive_kp_linear, kp_angular=cfg.body_drive_kp_angular,
            max_linear=cfg.body_drive_max_linear, max_angular=cfg.body_drive_max_angular,
            deadband_m=cfg.body_drive_deadband_m, deadband_rad=math.radians(cfg.body_drive_deadband_deg),
            holonomic=self.robot.holonomic,
        )
        drive.error = error
        if math.hypot(error[0], error[1]) > cfg.body_drive_max_error_m or abs(error[2]) > math.radians(
            cfg.body_drive_max_error_deg
        ):
            self._end_body_drive(
                f"the base is too far behind: {error[0]:+.2f}, {error[1]:+.2f} m, {math.degrees(error[2]):+.0f} deg",
                block=True,
            )
            return
        self._base_vel = velocity

        # The operator frame turns with the body: head = neck angle, arms relative to the body.
        correction = drive.yaw_correction0 - drive.turn
        delta = correction - self._frame.yaw
        if abs(delta) >= 1e-9:
            self._frame.yaw = correction
            self._on_operator_frame_changed(turned_by=delta)

    def _body_drive_status(self) -> str:
        drive = self._body_drive
        if drive is None:
            return ""
        ex, ey, eyaw = drive.error
        return f" | BASE DRIVE err=({ex:+.2f},{ey:+.2f},{math.degrees(eyaw):+.0f}°)"

    # ------------------------------------------------------------------
    # Arm reach calibration and start (Right A)
    # ------------------------------------------------------------------

    def _calibrate_arms(self) -> bool:
        """Fix the operator reach from the arms held straight out (``arm_length_source="calibrate"``).

        Called on every Right A. The reach of each arm is the median
        shoulder -> controller distance of the last ``arm_calibration_window_s``.
        Returns False only when nothing is calibrated yet and no arm gave a
        usable value: the first Right A then does not start teleoperation.
        Later presses never block; they replace the reach only by a value that
        is not clearly shorter (an arm that is not extended measures short).
        """
        cfg = self.settings
        if cfg.arm_length_source != "calibrate" or not self._abs:
            return True
        now = self._now
        measured: dict[str, float] = {}
        seen: dict[str, str] = {}
        for side in self._abs:
            values = [d for t, d in self._reach_samples[side] if now - t <= cfg.arm_calibration_window_s]
            if len(values) < _CALIBRATION_MIN_SAMPLES:
                seen[side] = f"{len(values)} samples"
                continue
            reach = float(np.median(values))
            seen[side] = f"{reach:.3f} m"
            if cfg.arm_calibration_min_m <= reach <= cfg.arm_calibration_max_m:
                measured[side] = reach
        current = {side: m.calibrated_reach for side, m in self._abs.items()}
        calibrated = all(v is not None for v in current.values())
        summary = ", ".join(f"{side} {text}" for side, text in seen.items())

        if not measured:
            if calibrated:
                logger.info("Arm reach kept (%s; not measured on this Right A: %s).", self._reach_status().strip(), summary)
                return True
            logger.warning(
                "Arm reach not measured (%s; accepted %.2f-%.2f m with controllers and shoulders tracked). "
                "Hold both arms straight out in front and press Right A again.",
                summary, cfg.arm_calibration_min_m, cfg.arm_calibration_max_m,
            )
            return False

        longest = max(measured.values())
        if len(measured) == 2 and (longest - min(measured.values())) / longest > cfg.arm_calibration_max_asymmetry:
            logger.warning("Arm reach differs between the arms (%s): using %.3f m for both.", summary, longest)
            measured = {side: longest for side in measured}
        # An arm without a usable value shares the other arm's.
        new = {side: measured.get(side, longest) for side in self._abs}

        if calibrated and any(new[side] < _CALIBRATION_KEEP_RATIO * current[side] for side in new):
            logger.info("Arm reach kept (%s; this Right A measured shorter: %s).", self._reach_status().strip(), summary)
            return True
        for side, mapper in self._abs.items():
            mapper.set_human_reach(new[side])
        # The controller -> gripper offsets belong to the old scale.
        self._needs_offset_latch = {side: True for side in self._needs_offset_latch}
        logger.info("Arm reach calibrated:%s (measured %s).", self._reach_status(), summary)
        if self._on_arm_calibrated is not None:
            self._on_arm_calibrated(self.arm_calibration())
        return True

    def _reach_status(self) -> str:
        """`` reach=R0.61/L0.60 (x0.96)`` for the status line; empty outside "calibrate"."""
        if self.settings.arm_length_source != "calibrate" or not self._abs:
            return ""
        if any(m.calibrated_reach is None for m in self._abs.values()):
            return " reach=uncalibrated"
        parts = "/".join(f"{side[0].upper()}{m.calibrated_reach:.2f}" for side, m in self._abs.items())
        scales = "/".join(f"{m.robot_reach / m.calibrated_reach * m.position_scale:.2f}" for m in self._abs.values())
        return f" reach={parts} (x{scales})"

    def _start(self, snap: RobotState) -> None:
        """First Right A: go live from the robot's present pose, without moving it.

        The operator frame was referenced on this press already (get_action);
        the look direction of this moment maps onto the measured head joints,
        and the controllers latch onto the grippers at the first squeeze.
        """
        logger.info("Right A — teleoperation started (squeeze to follow).")
        self._started = True
        self._stopped = False
        if self.settings.use_head:
            self._hold_head_at(snap.head_q, snap.torso)

    # ------------------------------------------------------------------
    # Re-synchronisation with the robot
    # ------------------------------------------------------------------

    def _resync_disengaged(self, snap: RobotState) -> None:
        """Re-seed held targets from the measured pose when the robot moved on its own.

        On the first tick this is unconditional: the follower reaches its ready
        pose only after ``teleop.connect()`` latched the targets, so commanding
        those would drag the robot straight back. Afterwards it triggers only
        while nothing is clutched, per component whose measured pose drifted
        past the ``resync_*`` thresholds (record reset, manual move).
        """
        cfg = self.settings
        force = self._needs_initial_resync
        self._needs_initial_resync = False
        if force:
            # The follower has just reached its ready pose: remember it as the
            # start pose Right A returns to.
            self._start_snapshot = snap
            if cfg.use_torso:
                _, clipped = self._clamp_torso_box(snap.torso)
                if clipped:
                    logger.warning(
                        "Start torso pose is outside torso_pos/rpy limits (%s): the torso will be "
                        "pulled into the box as soon as it engages. The default box is computed for the "
                        "crouched torso (--robot.ee_ready_torso_deg='[0,45,-90,45,0,0]'); use that ready "
                        "torso or adjust torso_pos_min/max, torso_rpy_*_deg.",
                        ",".join(clipped),
                    )
        # While any clutch is engaged the robot is being driven by us: the
        # torso carries the free arm along and the solvers lag behind the
        # targets, so drift is expected and must not be "corrected". The same
        # holds while (and shortly after) a Right-A return motion drives it.
        if not force and (
            any(c is not None and c.engaged for c in self._clutch.values())
            or (self._return_motion is not None and not self._return_motion.settled(self._now))
        ):
            return
        pos_thr = cfg.resync_position_threshold_m
        rot_thr = math.radians(cfg.resync_rotation_threshold_deg)
        synced: list[str] = []

        for side, measured in (("right", snap.right_ee), ("left", snap.left_ee)):
            clutch = self._clutch[side]
            if clutch is None or clutch.engaged:
                continue
            if force or _pose_drift(clutch.last_commanded, measured, pos_thr, rot_thr):
                clutch.hold_at(measured)
                synced.append(side)

        torso = self._clutch["torso"]
        if torso is not None and not torso.engaged and self._torso_target is not None:
            if force or _pose_drift(self._torso_target, snap.torso, pos_thr, rot_thr):
                torso.hold_at(snap.torso)
                self._torso_target = snap.torso.copy()
                synced.append("torso")

        # The head is re-seeded only on the first tick or together with an arm /
        # torso re-sync (a reset event): a plain lag between the commanded and
        # measured head joints during a fast head turn must not reset its origin.
        if cfg.use_head and (force or synced):
            target = self._head.target
            head_thr = math.radians(cfg.resync_head_threshold_deg)
            if force or (
                target is not None and float(np.max(np.abs(target - snap.head_q))) > head_thr
            ):
                # Hold the measured head joints; the look-direction origin is
                # re-latched on the next tracked head frame.
                self._hold_head_at(snap.head_q, snap.torso)
                synced.append("head")

        if synced:
            logger.info(
                "rby1_isaac: targets re-synced to the measured robot pose (%s)%s.",
                ", ".join(synced),
                " after connect / ready pose" if force else " — robot moved while not clutched",
            )

    # ------------------------------------------------------------------
    # Right A: return to the start pose
    # ------------------------------------------------------------------

    def _start_return_motion(self, snap: RobotState) -> None:
        """Release every clutch and ask the follower for its joint-space ready-pose motion."""
        if self._start_snapshot is None:
            logger.warning("Right A — no start pose recorded yet; nothing to return to.")
            return
        for clutch in self._clutch.values():
            if clutch is not None:
                clutch.disengage()
        # The arms go back to the start pose: drop the teleop posture hints so
        # the solver's nullspace target is not fighting the return motion.
        for retargeter in self._posture.values():
            retargeter.reset()
        self._hints = {side: None for side in self._hints}
        cfg = self.settings
        self._return_motion = _ReturnMotion(duration=max(cfg.ready_return_duration_s, 0.5))
        logger.info(
            "Right A — returning arms / torso / head to the start pose: follower joint-position "
            "motion over %.1fs (clutches released; squeeze again afterwards to follow).",
            self._return_motion.duration,
        )

    def _advance_return_motion(self, frame: XRFrame, snap: RobotState) -> None:
        """Hold every target at the start pose while the follower moves there.

        The follower performs the motion itself (blocking inside its
        send_action for ``duration``): ``reset.cmd`` is emitted once and the
        motion is considered finished on the next tick, so the first EE
        command afterwards is the pose the robot has just reached.
        """
        motion = self._return_motion
        if motion is None or motion.finished:
            return
        goal = self._start_snapshot if self._start_snapshot is not None else snap
        for side in ("right", "left"):
            clutch = self._clutch.get(side)
            if clutch is not None:
                clutch.hold_at(goal.right_ee if side == "right" else goal.left_ee)
        if self._clutch.get("torso") is not None:
            self._torso_target = goal.torso.copy()
            self._clutch["torso"].hold_at(goal.torso)
        if self.settings.use_head:
            self._head.hold(goal.head_q, goal.torso)
        if not motion.reset_emitted:
            motion.reset_emitted = True  # _build_action adds reset.cmd this tick
            return
        motion.finished = True
        motion.t_done = self._now
        if self.settings.use_head and not self.settings.head_latch_on_a:
            # Fresh head origin at the start pose: the current view becomes centre.
            self._hold_head_at(goal.head_q, goal.torso, relatch=False)
        logger.info("Start pose reached — squeeze to follow again.")

    def _make_head_retargeter(self) -> HeadRetargeter:
        cfg = self.settings
        return HeadRetargeter(
            yaw_sign=cfg.head_yaw_sign,
            pitch_sign=cfg.head_pitch_sign,
            yaw_gain=cfg.head_yaw_gain,
            pitch_gain=cfg.head_pitch_gain,
            yaw_limit=math.radians(cfg.head_yaw_limit_deg),
            pitch_min=math.radians(cfg.head_pitch_min_deg),
            pitch_max=math.radians(cfg.head_pitch_max_deg),
            smoothing=cfg.head_smoothing,
            yaw_offset=math.radians(cfg.head_yaw_offset_deg),
            pitch_offset=math.radians(cfg.head_pitch_offset_deg),
            compensate_torso=cfg.head_gaze_frame == "base",
        )

    @property
    def _returning(self) -> bool:
        return self._return_motion is not None and not self._return_motion.finished

    # ------------------------------------------------------------------
    # Arms: absolute mapping (dead-man) and posture hints
    # ------------------------------------------------------------------

    @staticmethod
    def _body_points(frame: XRFrame, side: str) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """IOBT shoulder / elbow / wrist positions of one side (None when invalid)."""
        body = frame.body
        if body is None:
            return None, None, None
        idx = {
            "right": (BodyJointIndex.RIGHT_SHOULDER, BodyJointIndex.RIGHT_ELBOW, BodyJointIndex.RIGHT_WRIST),
            "left": (BodyJointIndex.LEFT_SHOULDER, BodyJointIndex.LEFT_ELBOW, BodyJointIndex.LEFT_WRIST),
        }[side]
        out = []
        for i in idx:
            out.append(body.positions[int(i)].copy() if bool(body.valid[int(i)]) else None)
        return out[0], out[1], out[2]

    def _latch_ee_offsets(self, side: str, ctrl: ControllerState, snap: RobotState) -> None:
        """Controller now ↦ the START pose gripper (orientation and position offsets).

        Called at the first engage after connect / Right A, so the offsets come
        from the hand posture the operator actually works with, not from the
        posture used to press the button. Right A returns the arms to the
        start pose, so the controller is mapped onto that pose: squeezing
        without moving the hand leaves the arm where it is.
        """
        cfg = self.settings
        mapper = self._abs.get(side)
        if mapper is None:
            return
        ref = self._start_snapshot if self._start_snapshot is not None else snap
        target = ref.right_ee if side == "right" else ref.left_ee
        what = []
        if cfg.ee_orientation_latch_on_a:
            mapper.latch_orientation_offset(ctrl.pose[:3, :3], target[:3, :3])
            what.append("orientation")
        if cfg.ee_position_latch_on_a:
            if mapper.latch_position_offset(ctrl.position, snap.torso, target[:3, 3]):
                what.append("position")
            else:
                return  # shoulder not known yet: retry at the next engage
        self._needs_offset_latch[side] = False
        if what:
            logger.info("%s arm: controller ↦ start-pose gripper %s offset latched at engage.", side, " + ".join(what))

    def _latch_head_offset(self, frame: XRFrame) -> None:
        """Headset look direction now ↦ ``_head_latch_target`` (start / measured head joints).

        The head counterpart of :meth:`_latch_orientation_offsets`: after Right
        A the view at the moment of the press is the robot head's straight
        ahead, whatever the operator-frame yaw reference (shoulder line) is.
        Waits for a tracked head frame; nothing happens in neck mode.
        """
        target = self._head_latch_target
        if target is None:
            return
        if not self.settings.use_head or self.settings.wear_mode == "neck":
            self._head_latch_target = None
            return
        if frame.head is None:
            return  # retry on the next tracked head frame
        head_q, torso = target
        self._head.latch_offset(frame.head.pose, head_q, torso)
        self._head_latch_target = None
        logger.info("Head offset latched (headset look direction ↦ head joints %s).", np.round(head_q, 3).tolist())

    def _update_arm_absolute(
        self, side: str, frame: XRFrame, snap: RobotState, t: float, dt: float, gates: Gates
    ) -> None:
        clutch = self._clutch[side]
        mapper = self._abs.get(side)
        if clutch is None or mapper is None:
            return
        ctrl = frame.right if side == "right" else frame.left
        measured = snap.right_ee if side == "right" else snap.left_ee
        cfg = self.settings

        S, E, W = self._body_points(frame, side)
        # Before the arm's first engage "auto" takes the body shoulder when it
        # is valid and the headset estimate otherwise; after it, only the
        # source it engaged with (see _shoulder_locked).
        locked = self._shoulder_locked[side]
        sources = (locked,) if locked is not None else _SHOULDER_SOURCES[cfg.shoulder_source]
        if "body" in sources and S is not None:
            mapper.observe_body(S, E, W)
            self._shoulder_fed[side] = "body"
            self._abs_last_body_t[side] = t
        elif "headset" in sources and frame.head is not None:
            # Estimate the shoulder from the headset position (operator frame
            # is yaw-referenced: +x forward, +y left).
            ox, oy, oz = cfg.neck_shoulder_offset
            est = frame.head.position + np.array([ox, oy if side == "left" else -oy, oz])
            mapper.set_shoulder(est)
            self._shoulder_fed[side] = "headset"
            self._abs_last_body_t[side] = t
        body_ok = self._abs_last_body_t[side] is not None and t - self._abs_last_body_t[side] <= cfg.abs_hold_s
        if cfg.arm_length_source == "calibrate":
            samples = self._reach_samples[side]
            shoulder = mapper.shoulder
            if ctrl is not None and shoulder is not None and self._abs_last_body_t[side] == t:
                samples.append((t, float(np.linalg.norm(ctrl.position - shoulder))))
            while samples and t - samples[0][0] > cfg.arm_calibration_window_s:
                samples.popleft()

        def release(state: str) -> None:
            if clutch.engaged:
                clutch.disengage()
                logger.debug("%s arm released — holding.", side)
            self._abs_ramp[side] = None
            mapper.reset_rate_limit(None)
            self._abs_state[side] = state

        if not gates.arms:
            release(gates.arm_hold)
            return
        if ctrl is None:
            release("hold")
            return
        if self._frame.needs_reference and cfg.reference_source == "shoulders":
            release("hold (waiting for shoulders)")
            return
        if ctrl.squeeze <= cfg.clutch_threshold:
            release("hold")
            return
        if not body_ok:
            lost = "hold (no headset)" if locked == "headset" else "hold (no body)"
            if locked is not None and self._abs_state[side] != lost:
                logger.warning(
                    "%s arm holds: its shoulder comes from %s, which has been lost for more than %.1f s. "
                    "It follows again when that tracking returns; Right A starts over.",
                    side, _SHOULDER_SOURCE_NAMES[locked], cfg.abs_hold_s,
                )
            release(lost)
            return
        if not clutch.engaged and self._needs_offset_latch[side]:
            self._latch_ee_offsets(side, ctrl, snap)
        target = mapper.target(ctrl.position, ctrl.orientation, snap.torso)
        if target is None:
            release("hold (no shoulder)")
            return

        if not clutch.engaged:
            if locked is None and self._shoulder_fed[side] is not None:
                self._shoulder_locked[side] = self._shoulder_fed[side]
                logger.info(
                    "%s arm: operator shoulder from %s until the next Right A.",
                    side, _SHOULDER_SOURCE_NAMES[self._shoulder_fed[side]],
                )
            clutch.engage(measured)
            self._abs_ramp[side] = (t, clutch.last_commanded.copy(), max(cfg.engage_ramp_s, 0.05))
            mapper.reset_rate_limit(None)
            logger.debug("%s arm dead-man engaged — ramping to the absolute target.", side)

        ramp = self._abs_ramp[side]
        if ramp is not None:
            t0, T_start, dur = ramp
            a = smoothstep((t - t0) / dur)
            T = interpolate_pose(T_start, target, a)
            self._abs_state[side] = "ramp"
            if a >= 1.0:
                self._abs_ramp[side] = None
                mapper.reset_rate_limit(T)
                self._abs_state[side] = "track"
        else:
            T = mapper.rate_limit(target, dt)
            self._abs_state[side] = "track"
        clutch.set_commanded(T)

    def _update_posture_hints(
        self, frame: XRFrame, snap: RobotState, t: float, dt: float, gates: Gates
    ) -> None:
        """Retarget the arm posture only while that arm is engaged (squeeze held).

        A released arm must not move at all, so its nullspace hint is frozen
        at the last value together with its EE target. The retargeter is seeded
        with the robot's measured ``arm_0..arm_3`` so the first hint starts at
        the actual posture (and on the solver branch nearest to it) and ramps
        towards the operator's posture instead of jumping. Body-tracking
        dropouts while engaged keep the last hint (no fall-back to the robot's
        default nullspace pose mid-motion).
        """
        if not self._posture or not gates.posture_hints:
            return
        for side, retargeter in self._posture.items():
            clutch = self._clutch.get(side)
            if clutch is None or not clutch.engaged:
                continue
            S, E, Wb = self._body_points(frame, side)
            ctrl = frame.right if side == "right" else frame.left
            W = ctrl.position if (self.settings.hint_wrist_source == "controller" and ctrl is not None) else Wb
            seed = snap.right_q if side == "right" else snap.left_q
            h = retargeter.update(S, E, W, snap.torso, t, dt, seed=seed)
            if h is not None:
                self._hints[side] = h

    # ------------------------------------------------------------------
    # Torso, head, grippers, base
    # ------------------------------------------------------------------

    def _torso_engage_allowed(self) -> bool:
        """Apply ``torso_engage``: both / any enabled arm clutched, or always."""
        policy = self.settings.torso_engage
        if policy == "always":
            return True
        arms = [c for k, c in self._clutch.items() if k != "torso" and c is not None]
        if not arms:
            return False
        if policy == "any_arm":
            return any(c.engaged for c in arms)
        return all(c.engaged for c in arms)

    def _torso_driver_pose(self, frame: XRFrame) -> np.ndarray | None:
        src = self.settings.torso_source
        if self.settings.wear_mode == "neck":
            if src == "body" and not self._warned_neck_torso:
                logger.warning("wear_mode=neck: torso_source=body is replaced by the headset pose.")
                self._warned_neck_torso = True
            if frame.head is None:
                return None
            pose = frame.head.pose
            a = self.settings.neck_torso_smoothing
            if self._neck_driver is None or a >= 1.0:
                self._neck_driver = pose
            else:
                self._neck_driver = interpolate_pose(self._neck_driver, pose, a)
            return self._neck_driver
        if src == "body":
            pose = chest_pose_from_body(frame.body, self._torso_joint, self._torso_required)
            if pose is None and not self._warned_no_body:
                logger.warning(
                    "torso_source=body but no valid body-tracking frame yet — the torso "
                    "holds until body tracking is available (Quest 3: enable body "
                    "tracking in the headset; Pico: pair the motion trackers)."
                )
                self._warned_no_body = True
            return pose
        if src == "head":
            return frame.head.pose if frame.head is not None else None
        return None

    def _update_torso(self, frame: XRFrame, snap: RobotState, gates: Gates) -> None:
        clutch = self._clutch["torso"]
        if clutch is None:
            return
        driver = None
        if not gates.torso:
            self._torso_hold_reason = gates.torso_hold
        elif not self._torso_engage_allowed():
            self._torso_hold_reason = f"arms not clutched (torso_engage={self.settings.torso_engage})"
        else:
            driver = self._torso_driver_pose(frame)
            if driver is None:
                src = self.settings.torso_source
                if src == "body":
                    if frame.body is None:
                        self._torso_hold_reason = "no body-tracking frame"
                    else:
                        bad = [
                            BodyJointIndex(i).name
                            for i in [*self._torso_required, self._torso_joint]
                            if not bool(frame.body.valid[i])
                        ]
                        self._torso_hold_reason = f"body joints invalid: {bad}"
                elif src == "head" or self.settings.wear_mode == "neck":
                    self._torso_hold_reason = "no headset pose"
                else:
                    self._torso_hold_reason = "torso_source=none"
        if driver is None:
            clutch.disengage()
            return
        self._torso_hold_reason = "following"
        pos = driver[:3, 3]
        quat = quat_from_matrix(driver[:3, :3])
        if not clutch.engaged:
            clutch.engage(pos, quat, snap.torso, latch_orientation="measured")
        else:
            clutch.rebase(pos, quat)
        # Chest delta since engage applied 1:1, then the absolute workspace box.
        self._torso_target, clipped = self._clamp_torso_box(clutch.last_commanded)
        if clipped:
            self._torso_hold_reason = f"following, clamped[{','.join(clipped)}]"

    def _clamp_torso_box(self, T: np.ndarray) -> tuple[np.ndarray, list[str]]:  # noqa: N803
        cfg = self.settings
        return clamp_pose_box(
            T,
            cfg.torso_pos_min,
            cfg.torso_pos_max,
            np.radians(cfg.torso_rpy_min_deg),
            np.radians(cfg.torso_rpy_max_deg),
        )

    def _update_head(self, frame: XRFrame, snap: RobotState, gates: Gates) -> None:
        if not self.settings.use_head or not gates.head:
            return
        if self.settings.wear_mode == "neck":
            return  # the held (start / ready) gaze stays; torso compensation happens in _build_action
        if frame.head is None:
            return
        if not self._head.latched:
            self._head.latch(frame.head.pose, snap.head_q, snap.torso)
            logger.info("Head origin latched.")
            return
        self._head.update(frame.head.pose, snap.torso)

    def _update_grippers(self, frame: XRFrame, gates: Gates) -> None:
        # Trigger 1 (squeezed) -> closed. Dataset convention: 1.0 = open. A free
        # hand pointing at / holding a camera-panel handle keeps its last gripper
        # value: its trigger is a UI click, not a gripper command.
        if not gates.grippers:
            return  # keep the grippers where they are before the first Right A and while stopped
        ui = self._ui_active
        for side, ctrl in (("right", frame.right), ("left", frame.left)):
            if ctrl is None:
                continue
            clutch = self._clutch.get(side)
            free = clutch is None or not clutch.engaged
            if ui.get(side) and free:
                continue
            self._gripper[side] = 1.0 - float(np.clip(ctrl.trigger, 0.0, 1.0))

    def _update_base(self, frame: XRFrame, gates: Gates) -> None:
        cfg = self.settings
        if not gates.base:
            self._base_vel = (0.0, 0.0, 0.0)
            return
        if not gates.base_sticks:
            return  # body drive: _update_body_drive set the velocity; the thumbsticks are ignored
        self._base_vel = thumbsticks_to_base_vel(
            frame.right.thumbstick if frame.right is not None else None,
            frame.left.thumbstick if frame.left is not None else None,
            deadzone=cfg.thumbstick_deadzone,
            max_linear=cfg.base_max_linear,
            max_angular=cfg.base_max_angular,
        )


def mask_grabbing_hands(frame: XRFrame, grabbing: Mapping[str, str | None]) -> XRFrame:
    """Neutralise squeeze / trigger / thumbstick of hands that hold a camera panel."""
    if not any(grabbing.values()):
        return frame
    def mask(c: ControllerState | None) -> ControllerState | None:
        if c is None:
            return None
        return replace(c, squeeze=0.0, trigger=c.trigger, thumbstick=np.zeros(2))

    right = mask(frame.right) if grabbing.get("right") else frame.right
    left = mask(frame.left) if grabbing.get("left") else frame.left
    return XRFrame(right=right, left=left, head=frame.head, body=frame.body)


class _BodyDrive:
    """State of one body-drive episode (Left X held), all relative to the press."""

    def __init__(self, *, origin_xy: np.ndarray, body_yaw: float, yaw_correction0: float, odometry0: np.ndarray):
        self.origin_xy = origin_xy            # headset xy at the press (raw XR frame)
        self.body_yaw = body_yaw              # last shoulder-line yaw (raw XR frame)
        self.yaw_correction0 = yaw_correction0
        self.odometry0 = odometry0            # robot odometry at the press (3x3 SE(2))
        self.turn_raw = 0.0                   # operator turn since the press, unwrapped (rad)
        self.turn = 0.0                       # … filtered
        self.step = np.zeros(2)               # operator displacement since the press, start base frame (m), filtered
        self.robot_yaw_wrapped = 0.0
        self.robot_turn = 0.0                 # robot turn since the press, unwrapped (rad)
        self.error = (0.0, 0.0, 0.0)


class _ReturnMotion:
    """State of a Right-A return-to-start (follower joint-space motion)."""

    SETTLE_S = 2.0  # drift re-sync stays suppressed this long after arrival

    def __init__(self, *, duration: float) -> None:
        self.duration = duration      # minimum time of the follower's motion (reset.cmd value)
        self.reset_emitted = False
        self.finished = False
        self.t_done: float | None = None

    def settled(self, now: float) -> bool:
        return self.finished and self.t_done is not None and (now - self.t_done > self.SETTLE_S)


def _pose_drift(
    target_T: np.ndarray, measured_T: np.ndarray, pos_thr: float, rot_thr: float  # noqa: N803
) -> bool:
    """True when measured and target differ by more than the thresholds."""
    dpos = float(np.linalg.norm(measured_T[:3, 3] - target_T[:3, 3]))
    angle = float(np.linalg.norm(rotvec_from_matrix(measured_T[:3, :3] @ target_T[:3, :3].T)))
    return dpos > pos_thr or angle > rot_thr
