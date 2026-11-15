"""LeRobot Teleoperator: RB-Y1 whole-body control from Meta Quest body + hand tracking.

The operator stands in front of the robot wearing a Quest running the
Meta_with_hands app (inside-out body tracking + hand tracking, no
controllers). The operator's upper body drives the torso, the hands drive
the arms (elbow flexion mapped 1:1 to the robot elbow), the gaze drives the
head, pinching drives the grippers and walking / turning drives the mobile
base (pelvis displacement tracked with the base odometry).

Control flow
------------
1. :meth:`connect` opens a read-only robot connection (state, odometry, FK),
   checks the bundled RB-Y1 M v1.3 arm model against the robot, and starts
   the headset stream and the keyboard reader.
2. The first :meth:`get_action` blocks until **Enter** is pressed in the
   terminal while the body is tracked: that instant becomes Δ = 0 between the
   operator and the robot (which the follower already moved to its ready
   pose — stand in a similar posture).
3. Each :meth:`get_action` then retargets the latest snapshot
   (:class:`~.wb_retarget.Retargeter`) and returns the whole-body action:
   torso / arm EE poses, arm nullspace joint targets (the IK solution, which
   places the elbow), head joints, grippers and base velocity.

Keys: **Enter** latch / re-latch and resume, **Space** hold (base stops, arms
and torso keep their last target).

Following the LeRobot convention this class never commands the robot: the
follower ``lerobot_robot_rby1.Rby1`` with ``action_mode="wholebody"``
executes the actions. Ported from the semi-humanoid teleoperation
``rby1_ces_elbow.py`` (CES branch).
"""

from __future__ import annotations

import logging
import math
import select
import sys
import threading
import time
from collections import deque
from typing import Any

import numpy as np

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_rby1_vr_wholebody import Rby1VRWholebodyConfig
from .constants import BASE_VEL_NAMES, LEFT_EE_NAMES, POS_SUFFIX, RIGHT_EE_NAMES, TORSO_EE_NAMES
from .frame_transforms import se3_to_pose_action
from .wb_kinematics import Rby1Arms
from .wb_retarget import BODY_TIMEOUT, Measured, Retargeter, Targets, Tuning, rotvec_of, valid_se2
from .wb_stream import SIDES, FilterTaus, Snapshot, StreamReceiver

logger = logging.getLogger(__name__)

ARM_DOF = 7
RIGHT_NULL_NAMES = [f"right_arm_{i}.null" for i in range(ARM_DOF)]
LEFT_NULL_NAMES = [f"left_arm_{i}.null" for i in range(ARM_DOF)]
HEAD_POS_NAMES = [f"head_{i}{POS_SUFFIX}" for i in range(2)]

# FK links, in index order for compute_transformation.
_LINKS = ["base", "link_torso_3", "link_torso_5", "link_right_arm_6", "link_left_arm_6"]


class _Keyboard:
    """Reads single keys from the controlling terminal (cbreak mode) in a thread."""

    def __init__(self) -> None:
        self.keys: deque[str] = deque()
        self.enabled = sys.stdin.isatty()
        self._old: Any = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.enabled:
            return
        import termios
        import tty

        fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        self._thread = threading.Thread(target=self._run, name="rby1-wb-keys", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                if select.select([sys.stdin], [], [], 0.2)[0]:
                    ch = sys.stdin.read(1)
                    if ch:
                        self.keys.append(ch)
            except (OSError, ValueError):
                return

    def drain(self) -> list[str]:
        out = []
        while self.keys:
            out.append(self.keys.popleft())
        return out

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._old is not None:
            import termios

            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old)
            self._old = None


class Rby1VRWholebody(Teleoperator):
    """Whole-body teleoperator (Quest body + hand tracking) for the RB-Y1."""

    config_class = Rby1VRWholebodyConfig
    name = "rby1_vr_wholebody"

    def __init__(self, config: Rby1VRWholebodyConfig) -> None:
        super().__init__(config)
        self._config = config
        self._is_connected = False
        self._robot: Any = None
        self._model: Any = None
        self._dyn: Any = None
        self._dyn_state: Any = None
        self._arm_idx: dict[str, np.ndarray] = {}
        self._head_idx: np.ndarray | None = None
        self._arms: Rby1Arms | None = None
        self._rt: Retargeter | None = None
        self._rx: StreamReceiver | None = None
        self._kb: _Keyboard | None = None
        self._state = "IDLE"          # IDLE (not latched) | RUN | HOLD
        self._last_tick: float | None = None
        self._last_status = 0.0
        self._last_targets: Targets | None = None

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
        cfg = self._config
        names = TORSO_EE_NAMES + RIGHT_EE_NAMES + LEFT_EE_NAMES + RIGHT_NULL_NAMES + LEFT_NULL_NAMES
        if cfg.use_head:
            names = names + HEAD_POS_NAMES
        features: dict[str, type] = {n: float for n in names}
        if cfg.use_gripper:
            features[f"right_gripper_0{POS_SUFFIX}"] = float
            features[f"left_gripper_0{POS_SUFFIX}"] = float
        if cfg.use_mobile_base:
            for n in BASE_VEL_NAMES:
                features[n] = float
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    # ------------------------------------------------------------------ #
    # Connection
    # ------------------------------------------------------------------ #

    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected.")
        try:
            import rby1_sdk as rby
        except ImportError as e:
            raise ImportError("rby1_sdk is required.") from e

        cfg = self._config
        try:
            self._init_robot(rby, cfg)
            self._arms = Rby1Arms()
            self._check_model()
            self._rt = Retargeter(self._tuning(), self._arms, self._arm_idx, self._head_idx)
            self._rx = StreamReceiver(
                cfg.headset_ip, cfg.headset_port,
                FilterTaus(hand=cfg.tau_hand, body=cfg.tau_body, heading=cfg.tau_heading, hmd=cfg.tau_hmd),
                cfg.body_allow_low_fidelity,
            )
            self._rx.start()
            self._kb = _Keyboard()
            self._kb.start()
        except BaseException:
            self._teardown()
            raise

        self._state = "IDLE"
        self._last_tick = None
        self._is_connected = True
        if cfg.profile_out:
            from lerobot_robot_rby1.profile import update_profile_section

            path = update_profile_section(cfg.profile_out, "teleop", {"type": self.name, "config": cfg})
            logger.info(f"{self} config recorded in {path}.")
        if not self._kb.enabled:
            logger.warning(f"{self}: stdin is not a terminal — keyboard latch/hold disabled"
                           + (f"; auto latch after {cfg.auto_latch_s:.1f}s" if cfg.auto_latch_s is not None else ""))
        logger.info(f"{self} connected. Stand in the robot's posture and press [Enter] to start "
                    "([Space] hold, [Enter] re-latch).")

    def _init_robot(self, rby: Any, cfg: Rby1VRWholebodyConfig) -> None:
        """Read-only connection: power, servos and commands belong to the follower."""
        from lerobot_robot_rby1 import model_probe

        model = cfg.robot_model.strip().lower()
        if model == "auto":
            model, _ = model_probe.resolve_model_version("auto", "auto", cfg.robot_address)
        if model not in ("a", "m"):
            raise RuntimeError(f"RB-Y1 model '{model}' is not supported by {self.name} (only A and M).")
        self._robot = rby.create_robot(cfg.robot_address, model)
        if not self._robot.connect():
            raise ConnectionError(f"Failed to connect to RB-Y1 at {cfg.robot_address}.")
        self._model = self._robot.model()
        self._arm_idx = {
            "right": np.array(self._model.right_arm_idx),
            "left": np.array(self._model.left_arm_idx),
        }
        self._head_idx = np.array(self._model.head_idx)
        self._dyn = self._robot.get_dynamics()
        self._dyn_state = self._dyn.make_state(_LINKS, self._model.robot_joint_names)

    def _check_model(self) -> None:
        """Compare the bundled v1.3 arm model with the robot's FK at the current pose."""
        cfg = self._config
        meas = self._measure()
        T_t5_inv = np.linalg.inv(meas.T_t5)
        worst = 0.0
        for side in SIDES:
            f = self._arms.arms[side].fk(meas.q[self._arm_idx[side]])
            T = T_t5_inv @ meas.T_arm6[side]
            dp = float(np.linalg.norm(T[:3, 3] - f["W"]))
            dr = float(np.linalg.norm(rotvec_of(T[:3, :3] @ f["R_ee"].T)))
            logger.info(f"Arm model check {side}: wrist dp={dp * 1000:.1f} mm, dR={math.degrees(dr):.2f} deg")
            worst = max(worst, dp)
        if worst > cfg.model_check_tol_m:
            msg = (f"Bundled RB-Y1 M v1.3 arm model differs from the robot by {worst * 1000:.1f} mm; "
                   "the hand mapping would be wrong")
            if not cfg.ignore_model_check:
                raise RuntimeError(msg + " (set ignore_model_check=true to proceed anyway).")
            logger.error(msg + " (ignored).")

    def _tuning(self) -> Tuning:
        c = self._config
        return Tuning(
            tau_heading=c.tau_heading,
            hand_slew_v=c.hand_slew_v, hand_slew_w=c.hand_slew_w, torso_slew_w=c.torso_slew_w,
            bleed_lin=c.bleed_lin, bleed_ang=c.bleed_ang, bleed_time=c.bleed_time,
            base_vmax=c.base_vmax, base_wmax=c.base_wmax, base_kp=c.base_kp, base_kp_yaw=c.base_kp_yaw,
            base_kff=c.base_kff, base_tau_ff=c.base_tau_ff,
            base_err_stop=c.base_err_stop, base_err_go=c.base_err_go,
            base_yaw_stop=math.radians(c.base_yaw_stop_deg), base_yaw_go=math.radians(c.base_yaw_go_deg),
            base_max_lag=c.base_max_lag, base_max_lag_yaw=math.radians(c.base_max_lag_yaw_deg),
            hand_map=c.hand_map, tau_elbow=c.tau_elbow, elbow_slew_w=math.radians(c.elbow_slew_deg),
            arm_scale=c.arm_scale, arm6_offset=math.radians(c.arm6_offset_deg),
            pinch_open_m=c.pinch_open_m, pinch_threshold=c.pinch_threshold, pinch_hyst=c.pinch_hyst,
            heading=c.heading, torso_height=c.torso_height, base_enabled=c.use_mobile_base,
        )

    def disconnect(self) -> None:
        if not self.is_connected:
            return
        self._teardown()
        logger.info(f"{self} disconnected.")

    def _teardown(self) -> None:
        if self._kb is not None:
            self._kb.stop()
            self._kb = None
        if self._rx is not None:
            self._rx.stop()
            self._rx = None
        if self._robot is not None:
            try:
                self._robot.disconnect()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"{self}: robot disconnect failed: {exc}")
            self._robot = None
        self._is_connected = False

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    # ------------------------------------------------------------------ #
    # Robot state
    # ------------------------------------------------------------------ #

    def _measure(self) -> Measured:
        state = self._robot.get_state()
        q = np.asarray(state.position, dtype=float).copy()
        self._dyn_state.set_q(q.copy())
        self._dyn.compute_forward_kinematics(self._dyn_state)
        T = [np.array(self._dyn.compute_transformation(self._dyn_state, 0, i)) for i in range(1, len(_LINKS))]
        odom = getattr(state, "odometry", None)
        odom = np.asarray(odom, dtype=float) if odom is not None else None
        fresh = odom is not None and valid_se2(odom)
        return Measured(q, T[1], T[0], {"right": T[2], "left": T[3]}, odom if fresh else None, fresh)

    # ------------------------------------------------------------------ #
    # get_action
    # ------------------------------------------------------------------ #

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self._state == "IDLE":
            self._wait_for_latch()

        now = time.monotonic()
        snap = self._rx.snapshot
        meas = self._measure()
        for key in self._kb.drain():
            if key == " " and self._state == "RUN":
                self._state = "HOLD"
                logger.info("HOLD — base stopped, arms/torso frozen; [Enter] to re-latch.")
            elif key in ("\n", "\r"):
                self._try_latch(snap, meas, now)

        dt = (now - self._last_tick) if self._last_tick is not None else 1.0 / 30.0
        self._last_tick = now
        if snap is None:  # cannot happen after a latch; keep the last targets
            tg = self._last_targets
        else:
            tg = self._rt.step(snap, meas, now, hold=(self._state == "HOLD"), dt=dt)
            self._last_targets = tg
        self._log_status(now, snap, tg)
        return self._build_action(tg)

    def _wait_for_latch(self) -> None:
        """Block until the operator latches (Enter, or auto_latch_s without a TTY)."""
        cfg = self._config
        last_print = 0.0
        tracked_since: float | None = None
        while self._state == "IDLE":
            now = time.monotonic()
            snap = self._rx.snapshot
            body_ok = snap is not None and snap.body.ok and now - snap.body.stamp < BODY_TIMEOUT
            tracked_since = (tracked_since or now) if body_ok else None
            keys = self._kb.drain()
            auto = (not self._kb.enabled and cfg.auto_latch_s is not None
                    and tracked_since is not None and now - tracked_since >= cfg.auto_latch_s)
            if auto or any(k in ("\n", "\r") for k in keys):
                self._try_latch(snap, self._measure(), now)
                if self._state != "IDLE":
                    return
            if now - last_print >= 2.0:
                last_print = now
                stream = "connected" if self._rx.connected else f"connecting to {cfg.headset_ip}:{cfg.headset_port}"
                logger.info(f"Waiting for [Enter] to latch — headset {stream}, "
                            f"body {'tracked' if body_ok else 'NOT tracked'}.")
            time.sleep(0.05)

    def _try_latch(self, snap: Snapshot | None, meas: Measured, now: float) -> None:
        body_ok = snap is not None and snap.body.ok and now - snap.body.stamp < BODY_TIMEOUT
        if not body_ok:
            logger.warning("Cannot latch: body not tracked.")
            return
        if self._config.use_mobile_base and not meas.odom_fresh:
            logger.warning("Cannot latch: no base odometry (set use_mobile_base=false to run without base).")
            return
        self._rt.latch(snap, meas, now)
        self._state = "RUN"
        self._last_tick = None
        logger.info(f"LATCHED psi0={math.degrees(self._rt.psi0):.0f} deg, hand_map={self._config.hand_map}, "
                    f"heading={self._config.heading}.")

    # ------------------------------------------------------------------ #
    # Action
    # ------------------------------------------------------------------ #

    def _build_action(self, tg: Targets) -> dict[str, Any]:
        cfg = self._config
        action: dict[str, Any] = {}
        action.update(se3_to_pose_action(tg.T_t5, "torso_ee"))
        action.update(se3_to_pose_action(tg.T_hand["right"], "right_ee"))
        action.update(se3_to_pose_action(tg.T_hand["left"], "left_ee"))
        action.update({k: float(v) for k, v in zip(RIGHT_NULL_NAMES, tg.q_ns["right"])})
        action.update({k: float(v) for k, v in zip(LEFT_NULL_NAMES, tg.q_ns["left"])})
        if cfg.use_head:
            action.update({k: float(v) for k, v in zip(HEAD_POS_NAMES, tg.head)})
        if cfg.use_gripper:
            # Dataset convention: 1.0 = open, 0.0 = closed.
            for side in ("right", "left"):
                value = 0.0 if tg.closed[side] else 1.0
                action[f"{side}_gripper_0{POS_SUFFIX}"] = 1.0 - value if cfg.gripper_invert else value
        if cfg.use_mobile_base:
            action["x.vel"] = float(tg.base_v[0])
            action["y.vel"] = float(tg.base_v[1])
            action["theta.vel"] = float(tg.base_w)
        return action

    def _log_status(self, now: float, snap: Snapshot | None, tg: Targets | None) -> None:
        period = self._config.status_log_period_s
        if period <= 0 or now - self._last_status < period or snap is None or tg is None:
            return
        self._last_status = now
        i = tg.info
        hands = " ".join(
            f"{s[0].upper()}={'ok' if snap.hands[s].ok else '--'}"
            f"/{'CLOSE' if tg.closed[s] else 'open'}"
            f"/err={self._rt.hands[s].palm_err * 1000:.0f}mm" for s in SIDES
        )
        y, p, r = i.get("torso_ypr", (0.0, 0.0, 0.0))
        logger.info(
            f"[{self._state}] pkt={snap.packets} age={now - snap.rx:.2f}s "
            f"body={'ok' if snap.body.ok else 'NO'} {hands} | base err={i.get('base_err', 0) * 100:.0f}cm "
            f"v=({tg.base_v[0]:+.2f},{tg.base_v[1]:+.2f}) w={tg.base_w:+.2f} "
            f"{'MOVE' if i.get('moving') else 'stop'} odom={'ok' if i.get('odom') else 'NO'} | "
            f"torso ypr=({math.degrees(y):.0f},{math.degrees(p):.0f},{math.degrees(r):.0f})"
            f"{'*' if i.get('torso_clip') else ''} | head=({math.degrees(tg.head[0]):.0f},"
            f"{math.degrees(tg.head[1]):.0f})"
        )
