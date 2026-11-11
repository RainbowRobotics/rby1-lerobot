"""LeRobot Teleoperator implementation for the RB-Y1 leader arm.

Overview
--------
The leader arm is a 14-DOF physical leader arm (7 per hand) with a button
and analog trigger per hand.  This teleoperator runs a 100 Hz background
control loop on the leader arm hardware that applies:

    * a smooth cosine init trajectory to the configured ready pose
    * gravity compensation, joint-limit barriers, viscous friction compensation
    * button-driven free / hold mode switching
    * trigger read-back for gripper control

:meth:`get_action` simply snapshots the latest joint positions and trigger
values written by the background thread and returns them in the LeRobot
action format.  It does NOT command the follower robot directly — that is
the follower's :meth:`send_action`'s job, per LeRobot convention.

Action keys
-----------
    torso_0.pos … torso_5.pos           (rad; only with use_torso, a fixed
                                         target: the leader arm has no torso)
    right_arm_0.pos … right_arm_6.pos   (rad)
    left_arm_0.pos  … left_arm_6.pos    (rad)
    right_gripper_0.pos                 (1.0 = open, 0.0 = closed)
    left_gripper_0.pos                  (1.0 = open, 0.0 = closed)

Button mapping
--------------
    Per-hand button   pressed → arm free (gravity comp); released → hold
    Per-hand trigger  analog [0, gripper_trigger_max] → normalised gripper
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

import numpy as np

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot_robot_rby1.joint_limits import RobotJointLimits, read_joint_limits
from lerobot_robot_rby1.model_probe import resolve_model_version

from .config_rby1_leader_arm import Rby1LeaderArmConfig
from .constants import (
    ARM_DOF,
    LEFT_ARM_NAMES,
    POS_SUFFIX,
    RIGHT_ARM_NAMES,
    TORSO_NAMES,
)
from .leader_arm_controller import LeaderArmControlLoop

logger = logging.getLogger(__name__)

_URDF_ENV_VAR = "RBY1_LEADER_ARM_MODEL"
# Source checkout of rby1-sdk in the home directory (the UPC default layout).
_HOME_SDK_URDF_PATH = "~/rby1-sdk/models/leader_arm/model.urdf"

# Settle time after switching the 12V rail on, before the Dynamixel bus is
# probed. Mirrors the capacitor-stabilisation sleep in Rby1.connect().
_POWER_SETTLE_TIME = 0.1
# Return to the zero position at disconnect: how long to wait for the control
# loop to start the motion, and extra time on top of its planned duration.
_RETURN_START_TIMEOUT = 1.0
_RETURN_MARGIN = 2.0


class Rby1LeaderArm(Teleoperator):
    """Teleoperator that reads joint positions from the RB-Y1 leader arm."""

    config_class = Rby1LeaderArmConfig
    name = "rby1_leader_arm"

    def __init__(self, config: Rby1LeaderArmConfig) -> None:
        super().__init__(config)
        self._config = config
        self._is_connected = False

        # rby1_sdk objects (set during connect).
        self._leader_arm = None
        self._control_loop: LeaderArmControlLoop | None = None
        # Joint limits read from the robot in connect(); None when they could
        # not be read and require_robot_joint_limits is False.
        self._limits: RobotJointLimits | None = None

        # Thread-safe shared state — written by the 100 Hz control loop,
        # read by get_action() on the main thread.
        self._lock = threading.Lock()
        self._right_q = np.zeros(ARM_DOF)
        self._left_q = np.zeros(ARM_DOF)
        self._right_trigger: float = 0.0
        self._left_trigger: float = 0.0
        self._init_done: bool = False

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True  # Leader arm uses absolute encoders.

    @property
    def action_features(self) -> dict[str, type]:
        cfg = self._config
        names: list[str] = []
        if cfg.use_torso:
            names += TORSO_NAMES
        if cfg.use_right_arm:
            names += RIGHT_ARM_NAMES
        if cfg.use_left_arm:
            names += LEFT_ARM_NAMES
        features: dict[str, type] = {f"{n}{POS_SUFFIX}": float for n in names}
        if cfg.use_gripper:
            if cfg.use_right_arm:
                features[f"right_gripper_0{POS_SUFFIX}"] = float
            if cfg.use_left_arm:
                features[f"left_gripper_0{POS_SUFFIX}"] = float
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    # ------------------------------------------------------------------ #
    # Connection
    # ------------------------------------------------------------------ #

    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002
        if self._is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected.")

        rby = self._import_sdk()
        cfg = self._config

        # 0. Power the 12V rail that feeds the leader-arm servos. This must
        #    happen before the Dynamixel bus is opened below.
        if cfg.auto_power_on_12v:
            self._power_on_12v(rby, cfg)

        # 0b. Joint limits of the robot the actions are meant for.
        self._limits = self._read_robot_limits(cfg)

        model_path = self._resolve_urdf_path(cfg)

        logger.info("Initialising leader-arm device …")
        rby.upc.initialize_device(rby.upc.LeaderArmDeviceName)
        self._leader_arm = rby.upc.LeaderArm(rby.upc.LeaderArmDeviceName)
        self._leader_arm.set_model_path(model_path)
        self._leader_arm.set_control_period(1.0 / cfg.control_frequency)

        active_ids = self._leader_arm.initialize(verbose=False)
        expected = rby.upc.LeaderArm.DeviceCount
        if len(active_ids) != expected:
            raise RuntimeError(
                f"Leader-arm device count mismatch: expected {expected}, "
                f"got {len(active_ids)} (active IDs: {active_ids}). "
                "The leader-arm servos are powered from the RB-Y1 12V rail — "
                "check that 12V is on, that the leader arm is plugged in, and "
                "that teleop.robot_address points at your robot."
            )

        right_init_q = np.deg2rad(cfg.right_init_q_deg)
        left_init_q = np.deg2rad(cfg.left_init_q_deg)
        self._right_q = right_init_q.copy()
        self._left_q = left_init_q.copy()

        self._control_loop = LeaderArmControlLoop(
            teleop=self,
            rby=rby,
            right_init_q=right_init_q,
            left_init_q=left_init_q,
            min_q=np.deg2rad(cfg.min_q_deg),
            max_q=np.deg2rad(cfg.max_q_deg),
            torque_limit=np.array(cfg.torque_limit),
            viscous_gain=np.array(cfg.viscous_gain),
            barrier=cfg.joint_limit_barrier,
            gravity_scale=cfg.gravity_comp_scale,
            init_duration=cfg.init_duration,
            resync_speed=np.deg2rad(cfg.resync_speed_deg_s),
            return_speed=np.deg2rad(cfg.return_speed_deg_s),
        )
        self._leader_arm.start_control(self._control_loop)

        self._is_connected = True
        if self._config.profile_out:
            from lerobot_robot_rby1.profile import update_profile_section

            path = update_profile_section(
                self._config.profile_out, "teleop", {"type": self.name, "config": self._config}
            )
            logger.info(f"{self} config recorded in {path}.")
        logger.info(
            f"{self} control loop started. Waiting {cfg.startup_wait_time:.1f}s "
            "for leader arm to reach ready pose …"
        )
        time.sleep(cfg.startup_wait_time)
        logger.info(
            f"{self} connected. Leader-arm control loop running at "
            f"{cfg.control_frequency} Hz."
        )

    def disconnect(self) -> None:
        if not self._is_connected:
            return
        if self._leader_arm is not None:
            if self._config.return_to_zero_on_disconnect and self._control_loop is not None:
                self._return_to_zero(self._control_loop)
            try:
                self._leader_arm.stop_control()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Error stopping leader arm control: {exc}")
            self._leader_arm = None
        self._control_loop = None
        self._limits = None
        self._is_connected = False
        logger.info(f"{self} disconnected.")

    @staticmethod
    def _return_to_zero(loop: LeaderArmControlLoop) -> None:
        """Bring the leader arm to its init pose and wait until it is there.

        Never raises: the caller stops the control loop afterwards in any
        case. Ctrl+C during the motion skips the rest of it.
        """
        try:
            loop.request_return()
            if not loop.return_started.wait(timeout=_RETURN_START_TIMEOUT):
                logger.warning(
                    "The leader-arm control loop is not running; the arm does not "
                    "return to its zero position."
                )
                return
            logger.info(
                f"Leader arm returning to its zero position ({loop.return_duration:.1f} s; "
                "Ctrl+C stops it where it is) ..."
            )
            if not loop.return_done.wait(timeout=loop.return_duration + _RETURN_MARGIN):
                logger.warning("The leader arm did not reach its init pose in time.")
        except KeyboardInterrupt:
            logger.warning("Leader-arm return to the zero position interrupted.")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Leader-arm return to the zero position failed: {exc}")

    # ------------------------------------------------------------------ #
    # Calibration / Configuration
    # ------------------------------------------------------------------ #

    def calibrate(self) -> None:
        pass  # Absolute encoders — no calibration needed.

    def configure(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    # get_action
    # ------------------------------------------------------------------ #

    def get_action(self) -> dict[str, Any]:
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        cfg = self._config

        with self._lock:
            right_q = self._right_q.copy()
            left_q = self._left_q.copy()
            right_trigger = self._right_trigger
            left_trigger = self._left_trigger

        action: dict[str, Any] = {}

        if cfg.use_torso:
            # No torso on the leader arm: a fixed target (see torso_target_deg).
            for name, deg in zip(TORSO_NAMES, cfg.torso_target_deg):
                action[f"{name}{POS_SUFFIX}"] = float(np.deg2rad(deg))

        if cfg.use_right_arm:
            right_q_out = right_q.copy()
            right_q_out[6] += np.deg2rad(cfg.right_wrist_offset_deg)
            if self._limits is not None:
                right_q_out = np.clip(right_q_out, *self._limits.right_arm)
            for i, name in enumerate(RIGHT_ARM_NAMES):
                action[f"{name}{POS_SUFFIX}"] = float(right_q_out[i])

        if cfg.use_left_arm:
            left_q_out = left_q.copy()
            left_q_out[6] += np.deg2rad(cfg.left_wrist_offset_deg)
            if self._limits is not None:
                left_q_out = np.clip(left_q_out, *self._limits.left_arm)
            for i, name in enumerate(LEFT_ARM_NAMES):
                action[f"{name}{POS_SUFFIX}"] = float(left_q_out[i])

        if cfg.use_gripper:
            tmax = cfg.gripper_trigger_max
            if cfg.use_right_arm:
                action[f"right_gripper_0{POS_SUFFIX}"] = float(
                    np.clip(right_trigger / tmax, 0.0, 1.0)
                )
            if cfg.use_left_arm:
                action[f"left_gripper_0{POS_SUFFIX}"] = float(
                    np.clip(left_trigger / tmax, 0.0, 1.0)
                )

        return action

    # ------------------------------------------------------------------ #
    # Feedback (no haptics)
    # ------------------------------------------------------------------ #

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _import_sdk() -> Any:
        try:
            import rby1_sdk as rby
        except ImportError as e:
            raise ImportError(
                "rby1_sdk is required for the RBY1 leader arm teleoperator. "
                "Install it from the RB-Y1 SDK."
            ) from e
        return rby

    @staticmethod
    def _read_robot_limits(cfg: Rby1LeaderArmConfig) -> RobotJointLimits | None:
        """Read the joint position limits from the robot at ``cfg.robot_address``.

        The arm targets are clipped to them in :meth:`get_action`, and the
        fixed torso target is checked against them here. Returns ``None``
        (after a warning) when the robot cannot be read and
        ``require_robot_joint_limits`` is False.
        """
        try:
            limits = read_joint_limits(cfg.robot_address, cfg.connect_timeout_sec)
        except Exception as exc:
            if cfg.require_robot_joint_limits:
                raise RuntimeError(
                    f"Could not read the joint limits from the RB-Y1 at '{cfg.robot_address}': {exc} "
                    "The leader arm clips its output to the robot's joint limits. Check "
                    "--teleop.robot_address and that the robot is on, or pass "
                    "--teleop.require_robot_joint_limits=false to use the leader arm without "
                    "a robot (its output is then not clipped)."
                ) from exc
            logger.warning(
                f"Could not read the joint limits from the RB-Y1 at '{cfg.robot_address}' ({exc}); "
                "the leader-arm output is NOT clipped to the robot's joint limits."
            )
            return None

        logger.info(f"Joint limits read from RB-Y1 model='{limits.model}' version='{limits.version}'.")
        if limits.version and limits.version != cfg.version:
            logger.warning(
                f"The robot reports version '{limits.version}' but teleop.version is '{cfg.version}': "
                "the leader-arm init pose is chosen by teleop.version and may not match the "
                "robot's ready pose."
            )
        if cfg.use_torso:
            lo, hi = (np.rad2deg(q) for q in limits.torso)
            for i, value in enumerate(cfg.torso_target_deg):
                if not lo[i] - 1e-6 <= value <= hi[i] + 1e-6:
                    raise ValueError(
                        f"Rby1LeaderArmConfig.torso_target_deg[{i}] = {value} is outside the robot's "
                        f"torso_{i} joint limit [{lo[i]:.1f}, {hi[i]:.1f}] deg."
                    )
        return limits

    @staticmethod
    def _power_on_12v(rby: Any, cfg: Rby1LeaderArmConfig) -> None:
        """Power the 12V rail that supplies the leader-arm Dynamixel servos.

        ``lerobot-teleoperate`` connects the teleoperator before the robot, so
        the follower's ``power_on(".*")`` has not run yet and the servos would
        not answer ``initialize()``. This mirrors the RB-Y1 SDK examples, which
        call ``power_on("12v")`` before touching the leader arm.

        Uses a short-lived A-template handle — the gRPC backend is the same for
        every model (see ``lerobot_robot_rby1.model_probe``), and the follower
        robot keeps ownership of servos, control manager and command streams.
        A connection failure is only a warning here: 12V may already be on.
        Reading the joint limits afterwards reports an unreachable robot.
        """
        timeout_ms = max(1, int(round(cfg.connect_timeout_sec * 1000.0)))
        unreachable = (
            f"Could not reach the RB-Y1 at '{cfg.robot_address}' to check the "
            "12V rail; continuing. Pass teleop.robot_address if the robot "
            "lives elsewhere."
        )

        # The handle must be of the robot's own model: a Model A handle on a
        # Model M robot fails in is_power_on() (joint state size mismatch).
        try:
            model, _ = resolve_model_version("auto", "auto", cfg.robot_address, cfg.connect_timeout_sec)
            robot = rby.create_robot(cfg.robot_address, model)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"{unreachable} ({exc})")
            return

        try:
            try:
                connected = robot.connect(max_retries=1, timeout_ms=timeout_ms)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"{unreachable} ({exc})")
                return
            if not connected:
                logger.warning(unreachable)
                return

            if robot.is_power_on("12v"):
                logger.info("RB-Y1 12V rail already on.")
                return

            logger.info("Powering on the RB-Y1 12V rail for the leader arm ...")
            if not robot.power_on("12v"):
                raise RuntimeError(
                    "Failed to power on the RB-Y1 12V rail; the leader-arm "
                    "servos will not respond."
                )
            time.sleep(_POWER_SETTLE_TIME)
        finally:
            try:
                robot.disconnect()
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _resolve_urdf_path(cfg: Rby1LeaderArmConfig) -> str:
        """Find the leader-arm URDF.

        Order: ``leader_arm_model_path`` → ``$RBY1_LEADER_ARM_MODEL`` → the
        ``models/`` directory next to an editable rby1_sdk install →
        ``~/rby1-sdk``. The PyPI rby1-sdk wheel does not ship the model.
        """
        if cfg.leader_arm_model_path is not None:
            return cfg.leader_arm_model_path
        env_path = os.environ.get(_URDF_ENV_VAR)
        if env_path:
            return env_path

        import importlib.resources

        sdk_path = str(importlib.resources.files("rby1_sdk"))
        candidates = [
            os.path.normpath(f"{sdk_path}/../models/leader_arm/model.urdf"),
            os.path.expanduser(_HOME_SDK_URDF_PATH),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(
            "Could not find the leader arm URDF (models/leader_arm/model.urdf from an "
            "rby1-sdk source checkout; the PyPI wheel does not include it). Looked in: "
            f"{', '.join(candidates)}. Pass --teleop.leader_arm_model_path=<path> or set "
            f"{_URDF_ENV_VAR}."
        )
