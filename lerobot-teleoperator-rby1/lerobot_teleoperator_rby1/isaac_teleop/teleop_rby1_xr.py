"""RB-Y1 XR device for NVIDIA Isaac Teleop, exposed to LeRobot as ``rby1_isaac``.

Control flow (one ``TeleopSession.step()`` per :meth:`get_action`)::

    controllers / head / body  ──►  XRFrame (robot base frame)
        right / left controller  ──► Clutch ──► right_ee.* / left_ee.*
        body chest (or head)     ──► Clutch + clamp ──► torso_ee.*
        headset look direction   ──► HeadRetargeter ──► head_0.pos / head_1.pos
        triggers                 ──► *_gripper_0.pos   (1.0 = open)
        thumbsticks              ──► x.vel / y.vel / theta.vel

Following the LeRobot convention this teleoperator only *produces* actions;
the paired ``lerobot_robot_rby1.Rby1`` follower (``action_mode="ee"``,
``use_head=True`` …) executes them. The robot link held here is read-only
(state + forward kinematics so the clutch latches on the *measured* pose).

Button mapping (Meta Quest naming)
----------------------------------
    Squeeze (grip) > threshold   Arm follows its controller (clutch engaged)
    Trigger                      Gripper (1 = closed)
    Right thumbstick             Base linear velocity; left thumbstick: yaw
    Right B                      Stop: freeze every target, zero the base
    Right A                      Release every clutch and return arms, torso and
                                 head to the start pose (measured right after the
                                 follower's ready-pose motion); re-reference the
                                 operator frame (current facing direction = robot
                                 +X); also resumes after a stop

Operator frame
--------------
OpenXR reports poses in a reference space whose forward is wherever the
headset looked when the session started. All poses are therefore rotated
about the robot z axis by ``-yaw(head)`` taken at the first tracked head
frame and again on every Right A, so "push the controller forward" always
means robot +X for the operator as they currently stand.
    Torso                        Follows the chest (body tracking) or the head
                                 while BOTH arms are clutched
"""

from __future__ import annotations

import logging
import math
import socket
import subprocess
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..constants import (
    BASE_VEL_NAMES,
    HEAD_NAMES,
    LEFT_EE_NAMES,
    POS_SUFFIX,
    RIGHT_EE_NAMES,
    TORSO_EE_NAMES,
)
from .base import IsaacTeleopTeleoperator
from .clutch import Clutch
from .config_isaac_teleop import Rby1XRConfig
from .retargeters import (
    HeadRetargeter,
    chest_pose_from_body,
    head_yaw_pitch,
    interpolate_pose,
    scale_clamp_delta,
    se3_to_ee_action,
    smoothstep,
    thumbsticks_to_base_vel,
)
from .robot_state import Rby1StateReader, RobotSnapshot
from .xr_frame import (
    BodyJointIndex,
    ButtonEdge,
    ControllerState,
    XRFrame,
    build_external_inputs,
    build_pipeline,
    frame_from_outputs,
    rotate_frame_about_z,
)

logger = logging.getLogger(__name__)

CLOUDXR_WEB_CLIENT_URL = "https://nvidia.github.io/IsaacTeleop/client"
CLOUDXR_WSS_PORT = 48322
_XR_CONNECT_REMINDER_S = 15.0
_SKIP_IFACE_PREFIXES = ("lo", "docker", "br-", "veth", "virbr", "l4tbr")


class Rby1XR(IsaacTeleopTeleoperator):
    """XR-controller / headset / body-tracking teleoperator for the RB-Y1.

    The class name must be ``Rby1XRConfig`` minus ``Config``: LeRobot's
    ``make_teleoperator_from_config`` resolves third-party device classes by
    that rule, importing them from the config's parent package.
    """

    config_class = Rby1XRConfig
    name = "rby1_isaac"

    def __init__(
        self,
        config: Rby1XRConfig,
        *,
        session_factory: Any | None = None,
        launcher_factory: Any | None = None,
        state_reader_factory: Callable[[str, str], Any] | None = None,
    ) -> None:
        super().__init__(config, session_factory=session_factory, launcher_factory=launcher_factory)
        self.config: Rby1XRConfig = config
        self._reader_factory = state_reader_factory or Rby1StateReader
        self._reader: Any = None
        self._connected = False

        self._external_inputs: dict[str, Any] | None = None
        self._body_transform: np.ndarray | None = None
        self._edges = ButtonEdge()

        self._clutch: dict[str, Clutch | None] = {"right": None, "left": None, "torso": None}
        self._torso_target: np.ndarray | None = None
        self._head = HeadRetargeter(
            yaw_sign=config.head_yaw_sign,
            pitch_sign=config.head_pitch_sign,
            yaw_gain=config.head_yaw_gain,
            pitch_gain=config.head_pitch_gain,
            yaw_limit=math.radians(config.head_yaw_limit_deg),
            pitch_min=math.radians(config.head_pitch_min_deg),
            pitch_max=math.radians(config.head_pitch_max_deg),
            smoothing=config.head_smoothing,
        )
        self._gripper = {"right": 1.0, "left": 1.0}  # dataset convention: 1 = open
        self._base_vel = (0.0, 0.0, 0.0)
        self._stopped = False
        self._tracking = False
        self._warned_no_body = False
        self._last_status_log = 0.0
        self._torso_hold_reason = "not started"
        self._needs_initial_resync = True
        # Start pose (first action) and the interpolated return motion (Right A).
        self._start_snapshot: RobotSnapshot | None = None
        self._return_motion: _ReturnMotion | None = None
        # Operator-frame yaw correction (rad about robot z) applied to every
        # incoming pose; taken from the raw head yaw at reference time.
        self._yaw_correction = 0.0
        self._needs_reference = True
        self._last_raw_frame: XRFrame | None = None

        self._torso_joint = int(BodyJointIndex[config.torso_body_joint])
        self._torso_required = [int(BodyJointIndex[n]) for n in config.body_required_joints]

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------

    @property
    def action_features(self) -> dict[str, type]:
        cfg = self.config
        names: list[str] = []
        if cfg.use_torso:
            names += TORSO_EE_NAMES
        if cfg.use_right_arm:
            names += RIGHT_EE_NAMES
        if cfg.use_left_arm:
            names += LEFT_EE_NAMES
        features: dict[str, type] = {n: float for n in names}
        if cfg.use_gripper:
            if cfg.use_right_arm:
                features[f"right_gripper_0{POS_SUFFIX}"] = float
            if cfg.use_left_arm:
                features[f"left_gripper_0{POS_SUFFIX}"] = float
        if cfg.use_mobile_base:
            for n in BASE_VEL_NAMES:
                features[n] = float
        if cfg.use_head:
            for n in HEAD_NAMES:
                features[f"{n}{POS_SUFFIX}"] = float
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_tracking(self) -> bool:
        """Whether the last frame carried at least one tracked controller."""
        return self._tracking

    @property
    def is_stopped(self) -> bool:
        return self._stopped

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002
        if self._connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected.")
        cfg = self.config
        logger.info("rby1_isaac: opening read-only robot link at %s …", cfg.robot_address)
        self._reader = self._reader_factory(cfg.robot_address, cfg.robot_model)
        try:
            self._reader.connect()
            if cfg.session_start == "connect":
                self._open_session()
            else:
                logger.info(
                    "rby1_isaac: session_start=first_action — the CloudXR/OpenXR session "
                    "opens on the first get_action() call."
                )
        except Exception:
            self._close_reader()
            raise
        self._connected = True
        logger.info("%s connected.", self)

    def disconnect(self) -> None:
        if not self._connected:
            return
        try:
            super().disconnect()
        finally:
            self._close_reader()
            self._external_inputs = None
            self._connected = False
            self._tracking = False
            logger.info("%s disconnected.", self)

    def _close_reader(self) -> None:
        if self._reader is not None:
            try:
                self._reader.close()
            finally:
                self._reader = None

    def _open_session(self) -> None:
        """Launch CloudXR + TeleopSession, wait for the headset, latch the start pose."""
        super().connect()
        try:
            self._external_inputs = build_external_inputs(self.config.base_T_anchor)
            self._wait_for_tracking()
            self._latch_initial()
        except Exception:
            # No half-state: a live session behind a raised connect would leak
            # the CloudXR runtime.
            super().disconnect()
            raise

    def _build_pipeline(self) -> Any:
        pipeline, body_in_graph = build_pipeline(with_body=self.config.torso_source == "body")
        self._body_transform = (
            None if body_in_graph else np.asarray(self.config.base_T_anchor, dtype=float)
        )
        return pipeline

    # ------------------------------------------------------------------
    # Frame acquisition
    # ------------------------------------------------------------------

    def _read_frame(self) -> XRFrame:
        outputs = self._step(
            execution_events=self._running_events(), external_inputs=self._external_inputs
        )
        raw = frame_from_outputs(
            outputs,
            want_body=self.config.torso_source == "body",
            body_transform=self._body_transform,
        )
        self._last_raw_frame = raw
        self._tracking = raw.any_controller
        return rotate_frame_about_z(raw, self._yaw_correction)

    def _set_reference_from_head(self, raw: XRFrame) -> bool:
        """Make the operator's current facing direction robot +X. Returns True if taken."""
        if raw.head is None:
            return False
        yaw, _ = head_yaw_pitch(raw.head.pose)
        self._yaw_correction = -yaw
        self._needs_reference = False
        logger.info(
            "Operator frame re-referenced: head yaw %.1f deg in the XR anchor frame is now robot +X.",
            math.degrees(yaw),
        )
        return True

    def _wait_for_tracking(self) -> None:
        """Block until a controller is tracked (user-paced; Ctrl-C aborts)."""
        print_xr_connect_help()
        logger.info("Waiting for the headset controllers to start streaming …")
        t0 = time.monotonic()
        last_reminder = t0
        timeout = self.config.tracking_wait_timeout_s
        while True:
            frame = self._read_frame()
            if frame.any_controller:
                logger.info("Headset connected — controllers are streaming.")
                return
            now = time.monotonic()
            if now - last_reminder >= _XR_CONNECT_REMINDER_S:
                print_xr_connect_help()
                logger.info("… still waiting for the headset to connect (Ctrl-C to abort).")
                last_reminder = now
            if timeout > 0.0 and now - t0 > timeout:
                raise TimeoutError(
                    f"No XR controller tracked within {timeout:.0f}s "
                    f"(tracking_wait_timeout_s)."
                )
            time.sleep(0.01)

    def _latch_initial(self) -> None:
        """Seed every target from the measured robot pose so nothing moves yet."""
        snap: RobotSnapshot = self._reader.read()
        cfg = self.config
        self._clutch["right"] = Clutch(snap.right_ee) if cfg.use_right_arm else None
        self._clutch["left"] = Clutch(snap.left_ee) if cfg.use_left_arm else None
        self._clutch["torso"] = Clutch(snap.torso) if cfg.use_torso else None
        self._torso_target = snap.torso.copy()
        self._head.hold(snap.head_q)
        self._gripper = {"right": 1.0, "left": 1.0}
        self._base_vel = (0.0, 0.0, 0.0)
        self._stopped = False
        # The follower connects (and moves to its ready pose) after this, so
        # the targets are re-seeded again on the first get_action().
        self._needs_initial_resync = True
        self._needs_reference = True
        logger.info("rby1_isaac: targets latched to the current robot pose (squeeze to follow).")

    # ------------------------------------------------------------------
    # get_action — main per-tick entry point
    # ------------------------------------------------------------------

    def get_action(self) -> dict[str, Any]:
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if self._session is None:  # session_start == "first_action"
            self._open_session()

        frame = self._read_frame()
        events = self._edges.update(frame)
        raw = self._last_raw_frame

        # Operator frame: reference on the first tracked head frame and on Right A.
        if raw is not None and (events["right_a"] or self._needs_reference):
            if self._set_reference_from_head(raw):
                frame = rotate_frame_about_z(raw, self._yaw_correction)

        # One robot read per tick: engage edges latch from it and disengaged
        # components are re-synchronised to it when the robot moved on its own.
        snap = self._reader.read()

        def get_snap() -> RobotSnapshot:
            return snap

        self._resync_disengaged(snap)

        if any(events.values()):
            logger.info("xr button edge: %s", [k for k, v in events.items() if v])
        if events["right_b"]:
            logger.info("Right B — stopping: targets frozen, base zeroed.")
            self._stopped = True
            for clutch in self._clutch.values():
                if clutch is not None:
                    clutch.disengage()
        elif events["right_a"]:
            if self._stopped:
                logger.info("Right A — resuming.")
            self._stopped = False
            self._start_return_motion(snap)

        self._advance_return_motion(frame, snap)
        self._update_arm("right", frame.right, get_snap)
        self._update_arm("left", frame.left, get_snap)
        self._update_torso(frame, get_snap)
        self._update_head(frame, get_snap)
        self._update_grippers(frame)
        self._update_base(frame)
        self._log_status(frame)
        return self._build_action()

    # ------------------------------------------------------------------
    # Per-tick helpers
    # ------------------------------------------------------------------

    def _resync_disengaged(self, snap: RobotSnapshot) -> None:
        """Re-seed held targets from the measured pose when the robot moved on its own.

        On the first tick this is unconditional: the follower reaches its ready
        pose only after ``teleop.connect()`` latched the targets, so commanding
        those would drag the robot straight back. Afterwards it triggers only
        while nothing is clutched, per component whose measured pose drifted
        past the ``resync_*`` thresholds (record reset, manual move).
        """
        cfg = self.config
        force = self._needs_initial_resync
        self._needs_initial_resync = False
        if force:
            # The follower has just reached its ready pose: remember it as the
            # start pose Right A returns to.
            self._start_snapshot = snap
        # While any clutch is engaged the robot is being driven by us: the
        # torso carries the free arm along and the solvers lag behind the
        # targets, so drift is expected and must not be "corrected". The same
        # holds while (and shortly after) a Right-A return motion drives it.
        if not force and (
            any(c is not None and c.engaged for c in self._clutch.values())
            or (self._return_motion is not None and not self._return_motion.settled())
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
                self._head = self._make_head_retargeter()
                self._head.hold(snap.head_q)
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

    def _start_return_motion(self, snap: RobotSnapshot) -> None:
        """Release every clutch and interpolate all targets back to the start pose."""
        if self._start_snapshot is None:
            logger.warning("Right A — no start pose recorded yet; nothing to return to.")
            return
        for clutch in self._clutch.values():
            if clutch is not None:
                clutch.disengage()
        cfg = self.config
        goal = self._start_snapshot
        self._return_motion = _ReturnMotion(
            t0=time.monotonic(),
            duration=max(cfg.ready_return_duration_s, 0.1),
            right=(self._clutch["right"].last_commanded, goal.right_ee) if cfg.use_right_arm else None,
            left=(self._clutch["left"].last_commanded, goal.left_ee) if cfg.use_left_arm else None,
            torso=(self._torso_target, goal.torso) if cfg.use_torso else None,
            head=(self._head.target if self._head.target is not None else snap.head_q, goal.head_q)
            if cfg.use_head
            else None,
        )
        logger.info(
            "Right A — returning arms / torso / head to the start pose over %.1fs "
            "(clutches released; squeeze again afterwards to follow).",
            self._return_motion.duration,
        )

    def _advance_return_motion(self, frame: XRFrame, snap: RobotSnapshot) -> None:
        motion = self._return_motion
        if motion is None or motion.finished:
            return
        a = smoothstep((time.monotonic() - motion.t0) / motion.duration)
        if motion.right is not None:
            self._clutch["right"].hold_at(interpolate_pose(*motion.right, a))
        if motion.left is not None:
            self._clutch["left"].hold_at(interpolate_pose(*motion.left, a))
        if motion.torso is not None:
            T = interpolate_pose(*motion.torso, a)
            self._torso_target = T
            self._clutch["torso"].hold_at(T)
        if motion.head is not None:
            q0, q1 = motion.head
            self._head.hold((1.0 - a) * np.asarray(q0) + a * np.asarray(q1))
        if a >= 1.0:
            motion.finished = True
            motion.t_done = time.monotonic()
            if motion.head is not None:
                # Fresh head origin at the start pose: the current view becomes centre.
                self._head = self._make_head_retargeter()
                self._head.hold(np.asarray(motion.head[1]))
            logger.info("Start pose reached — squeeze to follow again.")

    def _make_head_retargeter(self) -> HeadRetargeter:
        cfg = self.config
        return HeadRetargeter(
            yaw_sign=cfg.head_yaw_sign,
            pitch_sign=cfg.head_pitch_sign,
            yaw_gain=cfg.head_yaw_gain,
            pitch_gain=cfg.head_pitch_gain,
            yaw_limit=math.radians(cfg.head_yaw_limit_deg),
            pitch_min=math.radians(cfg.head_pitch_min_deg),
            pitch_max=math.radians(cfg.head_pitch_max_deg),
            smoothing=cfg.head_smoothing,
        )

    @property
    def _returning(self) -> bool:
        return self._return_motion is not None and not self._return_motion.finished

    def _update_arm(
        self, side: str, ctrl: ControllerState | None, get_snap: Callable[[], RobotSnapshot]
    ) -> None:
        clutch = self._clutch[side]
        if clutch is None:
            return
        if self._stopped or self._returning or ctrl is None:
            clutch.disengage()
            return
        enabled = ctrl.squeeze > self.config.clutch_threshold
        if enabled and not clutch.engaged:
            snap = get_snap()
            measured = snap.right_ee if side == "right" else snap.left_ee
            clutch.engage(
                ctrl.position,
                ctrl.orientation,
                measured,
                latch_orientation=self.config.latch_orientation,
            )
            logger.debug("%s arm clutch engaged.", side)
        elif enabled:
            clutch.rebase(ctrl.position, ctrl.orientation)
        elif clutch.engaged:
            clutch.disengage()
            logger.debug("%s arm clutch released — holding.", side)

    def _torso_engage_allowed(self) -> bool:
        """Apply ``torso_engage``: both / any enabled arm clutched, or always."""
        policy = self.config.torso_engage
        if policy == "always":
            return True
        arms = [c for k, c in self._clutch.items() if k != "torso" and c is not None]
        if not arms:
            return False
        if policy == "any_arm":
            return any(c.engaged for c in arms)
        return all(c.engaged for c in arms)

    def _torso_driver_pose(self, frame: XRFrame) -> np.ndarray | None:
        src = self.config.torso_source
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

    def _update_torso(self, frame: XRFrame, get_snap: Callable[[], RobotSnapshot]) -> None:
        clutch = self._clutch["torso"]
        if clutch is None:
            return
        driver = None
        if self._stopped:
            self._torso_hold_reason = "stopped (Right B)"
        elif self._returning:
            self._torso_hold_reason = "returning to start pose (Right A)"
        elif not self._torso_engage_allowed():
            self._torso_hold_reason = f"arms not clutched (torso_engage={self.config.torso_engage})"
        else:
            driver = self._torso_driver_pose(frame)
            if driver is None:
                src = self.config.torso_source
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
                elif src == "head":
                    self._torso_hold_reason = "no head pose"
                else:
                    self._torso_hold_reason = "torso_source=none"
        if driver is None:
            clutch.disengage()
            return
        self._torso_hold_reason = "following"
        pos = driver[:3, 3]
        quat = Rotation.from_matrix(driver[:3, :3]).as_quat()
        if not clutch.engaged:
            clutch.engage(pos, quat, get_snap().torso, latch_orientation="measured")
        else:
            clutch.rebase(pos, quat)
        cfg = self.config
        self._torso_target = scale_clamp_delta(
            clutch.home,
            clutch.last_commanded,
            rot_scale=cfg.torso_rot_scale,
            z_scale=cfg.torso_z_scale,
            use_xy=cfg.torso_use_xy,
            max_rot=math.radians(cfg.torso_max_rot_delta_deg),
            max_z=cfg.torso_max_z_delta_m,
        )

    def _update_head(self, frame: XRFrame, get_snap: Callable[[], RobotSnapshot]) -> None:
        if not self.config.use_head or self._stopped or self._returning:
            return
        if frame.head is None:
            return
        if not self._head.latched:
            self._head.latch(frame.head.pose, get_snap().head_q)
            logger.info("Head origin latched.")
            return
        self._head.update(frame.head.pose)

    def _update_grippers(self, frame: XRFrame) -> None:
        # Trigger 1 (squeezed) -> closed. Dataset convention: 1.0 = open.
        if frame.right is not None:
            self._gripper["right"] = 1.0 - float(np.clip(frame.right.trigger, 0.0, 1.0))
        if frame.left is not None:
            self._gripper["left"] = 1.0 - float(np.clip(frame.left.trigger, 0.0, 1.0))

    def _update_base(self, frame: XRFrame) -> None:
        cfg = self.config
        if not cfg.use_mobile_base or self._stopped:
            self._base_vel = (0.0, 0.0, 0.0)
            return
        self._base_vel = thumbsticks_to_base_vel(
            frame.right.thumbstick if frame.right is not None else None,
            frame.left.thumbstick if frame.left is not None else None,
            deadzone=cfg.thumbstick_deadzone,
            max_linear=cfg.base_max_linear,
            max_angular=cfg.base_max_angular,
        )

    def _log_status(self, frame: XRFrame) -> None:
        """Periodic one-line status: what is tracked, what is clutched, why the torso holds."""
        period = self.config.status_log_period_s
        if period <= 0.0:
            return
        now = time.monotonic()
        if now - self._last_status_log < period:
            return
        self._last_status_log = now

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
            return "n/a" if c is None else ("ENGAGED" if c.engaged else "hold")

        body = "-"
        if frame.body is not None:
            body = f"valid {int(frame.body.valid.sum())}/24"
        elif self.config.torso_source == "body":
            body = "none"
        vx, vy, wz = self._base_vel
        logger.info(
            "xr status | right: %s [%s] | left: %s [%s] | head: %s | body: %s | torso: %s | "
            "base=(%.2f,%.2f,%.2f)%s",
            ctrl(frame.right),
            eng("right"),
            ctrl(frame.left),
            eng("left"),
            "ok" if frame.head is not None else "-",
            body,
            self._torso_hold_reason if self._clutch.get("torso") is not None else "disabled",
            vx,
            vy,
            wz,
            " | STOPPED" if self._stopped else "",
        )

    def _build_action(self) -> dict[str, Any]:
        cfg = self.config
        action: dict[str, Any] = {}
        if cfg.use_torso:
            action.update(se3_to_ee_action(self._torso_target, "torso_ee"))
        if cfg.use_right_arm:
            action.update(se3_to_ee_action(self._clutch["right"].last_commanded, "right_ee"))
        if cfg.use_left_arm:
            action.update(se3_to_ee_action(self._clutch["left"].last_commanded, "left_ee"))
        if cfg.use_gripper:
            if cfg.use_right_arm:
                action[f"right_gripper_0{POS_SUFFIX}"] = self._gripper["right"]
            if cfg.use_left_arm:
                action[f"left_gripper_0{POS_SUFFIX}"] = self._gripper["left"]
        if cfg.use_mobile_base:
            vx, vy, wz = self._base_vel
            action["x.vel"], action["y.vel"], action["theta.vel"] = vx, vy, wz
        if cfg.use_head:
            head_q = self._head.target
            for i, n in enumerate(HEAD_NAMES):
                action[f"{n}{POS_SUFFIX}"] = float(head_q[i]) if head_q is not None else 0.0
        return action


class _ReturnMotion:
    """Interpolation state of a Right-A return-to-start motion."""

    SETTLE_S = 2.0  # drift re-sync stays suppressed this long after arrival

    def __init__(self, *, t0: float, duration: float, right, left, torso, head) -> None:
        self.t0 = t0
        self.duration = duration
        self.right = right
        self.left = left
        self.torso = torso
        self.head = head
        self.finished = False
        self.t_done: float | None = None

    def settled(self) -> bool:
        return self.finished and self.t_done is not None and (
            time.monotonic() - self.t_done > self.SETTLE_S
        )


def _pose_drift(
    target_T: np.ndarray, measured_T: np.ndarray, pos_thr: float, rot_thr: float  # noqa: N803
) -> bool:
    """True when measured and target differ by more than the thresholds."""
    dpos = float(np.linalg.norm(measured_T[:3, 3] - target_T[:3, 3]))
    dR = Rotation.from_matrix(measured_T[:3, :3]) * Rotation.from_matrix(target_T[:3, :3]).inv()
    return dpos > pos_thr or float(dR.magnitude()) > rot_thr


# ---------------------------------------------------------------------------
# Headset connection help
# ---------------------------------------------------------------------------


def _primary_ipv4() -> str | None:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
        finally:
            s.close()
    except OSError:
        return None


def candidate_ipv4s() -> list[tuple[str, str]]:
    """``[(interface, ipv4), ...]`` the headset might reach this host at."""
    primary = _primary_ipv4()
    found: list[tuple[str, str]] = []
    try:
        out = subprocess.run(
            ["ip", "-4", "-o", "addr", "show"], capture_output=True, text=True, timeout=2.0
        ).stdout
        for line in out.splitlines():
            parts = line.split()
            if len(parts) < 4:
                continue
            iface, ip = parts[1], parts[3].split("/")[0]
            if iface.startswith(_SKIP_IFACE_PREFIXES) or ip.startswith(("127.", "169.254.")):
                continue
            found.append((iface, ip))
    except Exception:
        pass
    if not found and primary:
        found.append(("default", primary))
    found.sort(key=lambda t: t[1] != primary)
    return found


def print_xr_connect_help() -> None:
    ips = candidate_ipv4s()
    lines = [
        "=" * 76,
        "Connect your XR headset to this host over NVIDIA CloudXR:",
        f"  1. In the headset browser, open:  {CLOUDXR_WEB_CLIENT_URL}",
        "  2. Enter this host's IP address:",
    ]
    if ips:
        lines += [f"        {ip:<15}  ({iface})" for iface, ip in ips]
        if len(ips) > 1:
            lines.append("     (use the address on the same network as your headset)")
    else:
        lines.append("        <could not determine — check `hostname -I` / `ip addr`>")
    lines += [
        f"  3. Accept the self-signed cert at https://<that-ip>:{CLOUDXR_WSS_PORT}/ , then Connect.",
        "=" * 76,
    ]
    print("\n".join(lines), flush=True)


# Backward-compatible alias (pre-rename).
Rby1XRTeleop = Rby1XR
