"""LeRobot Teleoperator implementation for the RB-Y1 leader arm.

Overview
--------
The leader arm is a 14-DOF physical leader arm (7 per hand) with a button
and analog trigger per hand.  This teleoperator runs a 100 Hz background
control loop on the leader arm hardware that applies:

    * a smooth cosine init trajectory to the configured ready pose
    * gravity compensation, joint-limit barriers, viscous damping
    * button-driven free / hold mode switching
    * trigger read-back for gripper control

:meth:`get_action` simply snapshots the latest joint positions and trigger
values written by the background thread and returns them in the LeRobot
action format.  It does NOT command the follower robot directly — that is
the follower's :meth:`send_action`'s job, per LeRobot convention.

Action keys
-----------
    right_arm_0 … right_arm_6   (rad)
    left_arm_0  … left_arm_6    (rad)
    right_gripper_0             (0.0 = open, 1.0 = closed)
    left_gripper_0              (0.0 = open, 1.0 = closed)

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

from .config_rby1_leader_arm import Rby1LeaderArmConfig
from .constants import (
    ARM_DOF,
    LEFT_ARM_NAMES,
    LEFT_ARM_Q_MAX,
    LEFT_ARM_Q_MIN,
    RIGHT_ARM_NAMES,
    RIGHT_ARM_Q_MAX,
    RIGHT_ARM_Q_MIN,
    TORSO_NAMES,
)
from .leader_arm_controller import LeaderArmControlLoop

logger = logging.getLogger(__name__)

_FALLBACK_URDF_PATH = "/home/nvidia/rby1-sdk/models/leader_arm/model.urdf"


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

        # Thread-safe shared state — written by the 100 Hz control loop,
        # read by get_action() / is_reset_complete() on the main thread.
        self._lock = threading.Lock()
        self._right_q = np.zeros(ARM_DOF)
        self._left_q = np.zeros(ARM_DOF)
        self._right_trigger: float = 0.0
        self._left_trigger: float = 0.0
        self._init_done: bool = False
        self._reset_requested: bool = False
        self._reset_right_arm_requested: bool = True
        self._reset_left_arm_requested: bool = True

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
        features: dict[str, type] = {n: float for n in names}
        if cfg.use_gripper:
            if cfg.use_right_arm:
                features["right_gripper_0"] = float
            if cfg.use_left_arm:
                features["left_gripper_0"] = float
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
                f"got {len(active_ids)} (active IDs: {active_ids})"
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
        )
        self._leader_arm.start_control(self._control_loop)

        self._is_connected = True
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
            try:
                self._leader_arm.stop_control()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Error stopping leader arm control: {exc}")
            self._leader_arm = None
        self._control_loop = None
        self._is_connected = False
        logger.info(f"{self} disconnected.")

    # ------------------------------------------------------------------ #
    # Calibration / Configuration
    # ------------------------------------------------------------------ #

    def calibrate(self) -> None:
        pass  # Absolute encoders — no calibration needed.

    def configure(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    # Reset (for lerobot-record)
    # ------------------------------------------------------------------ #

    def request_reset(self) -> None:
        """Trigger a non-blocking reset back to the ready pose."""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        reset_right = bool(self._config.reset_right_arm_on_record)
        reset_left = bool(self._config.reset_left_arm_on_record)
        if not reset_right and not reset_left:
            logger.info(
                f"{self} record reset disabled for both arms; keeping current pose."
            )
            return

        logger.info(f"{self} reset requested: moving leader arm to ready pose.")
        with self._lock:
            self._reset_right_arm_requested = reset_right
            self._reset_left_arm_requested = reset_left
            self._reset_requested = True
            self._init_done = False

    def is_reset_complete(self) -> bool:
        """Return ``True`` once the requested reset trajectory has finished."""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        with self._lock:
            return self._init_done and not self._reset_requested

    def reset(self) -> None:
        """Block until the leader arm has returned to the ready pose."""
        self.request_reset()

        timeout_s = max(float(self._config.init_duration) + 2.0, 2.0)
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.is_reset_complete():
                logger.info(f"{self} reset complete.")
                return
            time.sleep(0.05)

        logger.warning(
            f"{self} reset timed out after {timeout_s:.1f}s; continuing."
        )

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
            # Leader arm has no torso DOFs — emit zeros as placeholder so
            # action_features stay consistent with the follower robot.
            for name in TORSO_NAMES:
                action[name] = 0.0

        if cfg.use_right_arm:
            right_q_out = right_q.copy()
            right_q_out[6] += np.deg2rad(cfg.right_wrist_offset_deg)
            right_q_out = np.clip(right_q_out, RIGHT_ARM_Q_MIN, RIGHT_ARM_Q_MAX)
            for i, name in enumerate(RIGHT_ARM_NAMES):
                action[name] = float(right_q_out[i])

        if cfg.use_left_arm:
            left_q_out = left_q.copy()
            left_q_out[6] += np.deg2rad(cfg.left_wrist_offset_deg)
            left_q_out = np.clip(left_q_out, LEFT_ARM_Q_MIN, LEFT_ARM_Q_MAX)
            for i, name in enumerate(LEFT_ARM_NAMES):
                action[name] = float(left_q_out[i])

        if cfg.use_gripper:
            tmax = cfg.gripper_trigger_max
            if cfg.use_right_arm:
                action["right_gripper_0"] = float(
                    np.clip(right_trigger / tmax, 0.0, 1.0)
                )
            if cfg.use_left_arm:
                action["left_gripper_0"] = float(
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
    def _resolve_urdf_path(cfg: Rby1LeaderArmConfig) -> str:
        """Find the leader-arm URDF, honouring the user's override first."""
        if cfg.leader_arm_model_path is not None:
            return cfg.leader_arm_model_path

        import importlib.resources

        sdk_path = str(importlib.resources.files("rby1_sdk"))
        candidate = f"{sdk_path}/../models/leader_arm/model.urdf"
        if os.path.isfile(candidate):
            return candidate
        if os.path.isfile(_FALLBACK_URDF_PATH):
            return _FALLBACK_URDF_PATH
        raise FileNotFoundError(
            "Could not find leader arm URDF model. "
            "Set leader_arm_model_path in Rby1LeaderArmConfig."
        )
