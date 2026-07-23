"""Meta Quest VR teleoperator for Rainbow Robotics RB10E.

Control flow
------------
1. VRReceiver retains the latest Meta Quest head/controller state.
2. Right-A initializes control and optionally calibrates the user workspace scale.
3. Grip activates arm following.
4. The controller pose is transformed into an RB10E Cartesian target.
5. RB10EKinematics converts the Cartesian target into six joint angles.
6. get_action() returns joint_0 ... joint_5 in radians.
7. The paired RbCobot follower sends the action to the real robot.

This class never opens a command connection and never sends ServoJ directly.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.errors import (
    DeviceAlreadyConnectedError,
    DeviceNotConnectedError,
)

from lerobot_robot_rb.rb_cobot import get_active_rb_cobot

from .config_rb_vr import RbVrConfig
from .constants import (
    DEFAULT_READY_POSE_RAD,
    GRIPPER_NAME,
    JOINT_ACTION_NAMES,
    make_joint_action_features,
)
from .frame_transforms import (
    compute_user_scale,
    controller_pose_to_rb10e_target,
    head_pose_to_torso_pose,
)
from .rb10e_kinematics import RB10EKinematics
from .vr_receiver import ControllerSnapshot, VRReceiver, VRSnapshot


logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]


class RbVr(Teleoperator):
    """Meta Quest VR teleoperator emitting RB10E joint targets."""

    config_class = RbVrConfig
    name = "rb_vr"

    def __init__(self, config: RbVrConfig) -> None:
        super().__init__(config)

        self._config = config
        self._is_connected = False
        self._receiver: VRReceiver | None = None

        self._kinematics = RB10EKinematics(
            initial_q_rad=DEFAULT_READY_POSE_RAD,
            iterations=config.ik_iterations,
            base_damping=2.0,
        )

        self._robot_synced = False
        self._connected_robot: Any = None

        self._last_joint_target = np.asarray(
            DEFAULT_READY_POSE_RAD,
            dtype=np.float64,
        )
        self._last_gripper_target = 1.0

        self._user_scale = float(config.default_user_scale)

        self._is_initialized = not config.require_initialization_button
        self._is_stopped = False
        self._following = False

    # ------------------------------------------------------------------
    # LeRobot properties
    # ------------------------------------------------------------------

    @property
    def action_features(self) -> dict[str, type]:
        return make_joint_action_features(
            use_gripper=self._config.use_gripper,
        )

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return (
            self._is_connected
            and self._receiver is not None
            and self._receiver.is_running
        )

    @property
    def is_calibrated(self) -> bool:
        # User scale calibration is performed interactively with Right-A.
        # No persistent motor calibration is required by the VR device.
        return True

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002
        if self._is_connected:
            raise DeviceAlreadyConnectedError(
                f"{self} is already connected."
            )

        cfg = self._config

        receiver = VRReceiver(
            local_ip=cfg.local_ip,
            local_port=cfg.local_port,
            meta_quest_ip=cfg.meta_quest_ip,
            meta_quest_port=cfg.meta_quest_port,
            send_handshake=cfg.send_handshake,
        )

        try:
            receiver.start()
        except Exception:
            receiver.stop()
            raise

        self._receiver = receiver
        self._is_connected = True

        try:
            self.configure()
        except Exception:
            self.disconnect()
            raise

        logger.info(
            "%s connected. Waiting for Meta Quest packets on %s:%s.",
            self,
            cfg.local_ip,
            receiver.bound_port,
        )

        if cfg.require_initialization_button:
            logger.info(
                "Press the selected controller's primary button "
                "to initialize VR control."
            )

    def disconnect(self) -> None:
        if not self._is_connected and self._receiver is None:
            return

        receiver = self._receiver
        self._receiver = None

        if receiver is not None:
            receiver.stop()

        self._connected_robot = None
        self._robot_synced = False
        self._following = False
        self._is_connected = False

        logger.info("%s disconnected.", self)

    # ------------------------------------------------------------------
    # Calibration / configuration
    # ------------------------------------------------------------------

    def calibrate(self) -> None:
        # Runtime user-scale calibration is performed by Right-A.
        pass

    def configure(self) -> None:
        cfg = self._config

        if cfg.grip_threshold < 0.0 or cfg.grip_threshold > 1.0:
            raise ValueError("grip_threshold must be in [0, 1].")

        if cfg.tracking_timeout_s <= 0.0:
            raise ValueError("tracking_timeout_s must be positive.")

        if cfg.default_user_scale <= 0.0:
            raise ValueError("default_user_scale must be positive.")

        if cfg.reference_reach_mm <= 0.0:
            raise ValueError("reference_reach_mm must be positive.")

        if cfg.position_scale <= 0.0:
            raise ValueError("position_scale must be positive.")

        if cfg.max_joint_delta_rad <= 0.0:
            raise ValueError("max_joint_delta_rad must be positive.")

        if cfg.max_ik_position_error_mm <= 0.0:
            raise ValueError(
                "max_ik_position_error_mm must be positive."
            )

        if cfg.max_ik_orientation_error <= 0.0:
            raise ValueError(
                "max_ik_orientation_error must be positive."
            )

        self._robot_synced = False
        self._connected_robot = None

        self._last_joint_target = np.asarray(
            DEFAULT_READY_POSE_RAD,
            dtype=np.float64,
        )
        self._last_gripper_target = 1.0
        self._kinematics.reset(self._last_joint_target)

        self._user_scale = float(cfg.default_user_scale)
        self._is_initialized = not cfg.require_initialization_button
        self._is_stopped = False
        self._following = False

    # ------------------------------------------------------------------
    # Main LeRobot entry point
    # ------------------------------------------------------------------

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self} is not connected."
            )

        # lerobot-teleoperate connects teleop before robot, whereas
        # lerobot-record connects robot before teleop. At the first control
        # tick both are connected, so resolve the follower lazily here.
        self._sync_from_robot()

        assert self._receiver is not None

        events = self._receiver.consume_button_events()
        snapshot = self._receiver.get_state()

        primary_event, secondary_event = self._selected_button_events(
            events
        )

        # Secondary/B button always acts as an immediate software stop.
        if secondary_event:
            logger.info("VR secondary button pressed: following stopped.")
            self._is_stopped = True
            self._deactivate_following()
            return self._build_action()

        if snapshot is None:
            return self._build_action()

        if self._receiver.is_stale(
            self._config.tracking_timeout_s
        ):
            if self._following:
                logger.warning(
                    "VR packets are stale (age=%.3fs); holding position.",
                    self._receiver.age_s,
                )
            self._deactivate_following()
            return self._build_action()

        controller = self._selected_controller(snapshot)

        if (
            not snapshot.head_tracked
            or snapshot.head_pose_rb is None
            or not controller.tracked
            or controller.pose_rb is None
        ):
            if self._following:
                logger.warning(
                    "Head or selected controller tracking was lost; "
                    "holding position."
                )
            self._deactivate_following()
            return self._build_action()

        torso_pose = head_pose_to_torso_pose(
            snapshot.head_pose_rb
        )

        if primary_event:
            self._initialize_control(
                controller=controller,
                torso_pose=torso_pose,
            )

            # Return the freshly synchronized robot pose for this frame.
            # Actual VR following begins on the next frame.
            return self._build_action()

        if not self._is_initialized or self._is_stopped:
            return self._build_action()

        # Gripper trigger is independent from arm-following grip.
        if self._config.use_gripper:
            self._last_gripper_target = float(
                np.clip(1.0 - controller.trigger, 0.0, 1.0)
            )

        # Grip released: freeze the current target.
        if controller.grip <= self._config.grip_threshold:
            self._deactivate_following()
            return self._build_action()

        if not self._following:
            logger.info("VR arm following started.")
            self._following = True
            self._kinematics.set_seed(self._last_joint_target)

        try:
            target_pose = controller_pose_to_rb10e_target(
                controller.pose_rb,
                torso_pose,
                user_scale=self._user_scale,
                position_scale=self._config.position_scale,
                x_offset_mm=self._config.target_x_offset_mm,
                y_offset_mm=self._config.target_y_offset_mm,
                z_offset_mm=self._config.target_z_offset_mm,
            )

            solved_q = self._kinematics.solve(
                target_pose,
                initial_q_rad=self._last_joint_target,
                iterations=self._config.ik_iterations,
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            logger.warning(
                "VR target or IK calculation failed; holding position: %s",
                exc,
            )
            self._kinematics.set_seed(self._last_joint_target)
            return self._build_action()

        if not self._ik_solution_is_acceptable():
            info = self._kinematics.last_info
            logger.warning(
                "IK target rejected: position_error=%.1f mm, "
                "orientation_error=%.3f.",
                info.position_error_mm,
                info.orientation_error,
            )
            self._kinematics.set_seed(self._last_joint_target)
            return self._build_action()

        if not np.all(np.isfinite(solved_q)):
            logger.warning(
                "IK returned NaN or Inf; holding the previous target."
            )
            self._kinematics.set_seed(self._last_joint_target)
            return self._build_action()

        max_delta = self._config.max_joint_delta_rad
        command_q = np.clip(
            solved_q,
            self._last_joint_target - max_delta,
            self._last_joint_target + max_delta,
        )

        # Keep the numerical solver seeded with what was actually emitted,
        # rather than an unclamped target.
        self._last_joint_target = command_q.astype(
            np.float64,
            copy=True,
        )
        self._kinematics.set_seed(self._last_joint_target)

        return self._build_action()

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self} is not connected."
            )

        # Haptic feedback is not implemented yet.
        if feedback:
            logger.debug(
                "Ignoring unsupported VR feedback keys: %s",
                tuple(feedback),
            )

    # ------------------------------------------------------------------
    # Robot synchronization
    # ------------------------------------------------------------------

    def _sync_from_robot(self, *, force: bool = False) -> None:
        if self._robot_synced and not force:
            return

        robot = get_active_rb_cobot()

        if robot is None:
            raise RuntimeError(
                "No connected RB follower was found. The VR teleoperator "
                "must be used together with lerobot_robot_rb."
            )

        robot_features = set(robot.action_features)
        teleop_features = set(self.action_features)

        if robot_features != teleop_features:
            raise RuntimeError(
                "RB robot and VR teleoperator action features do not match.\n"
                f"Robot: {sorted(robot_features)}\n"
                f"VR:    {sorted(teleop_features)}\n"
                "Use robot.action_space=joint and make the robot gripper "
                "configuration match teleop.use_gripper."
            )

        observation = robot.get_observation()

        try:
            current_q = np.array(
                [
                    float(observation[name])
                    for name in JOINT_ACTION_NAMES
                ],
                dtype=np.float64,
            )
        except KeyError as exc:
            raise RuntimeError(
                f"RB observation is missing joint key {exc.args[0]!r}."
            ) from exc

        if not np.all(np.isfinite(current_q)):
            raise RuntimeError(
                f"RB joint observation contains NaN or Inf: {current_q}"
            )

        self._last_joint_target = current_q
        self._kinematics.set_seed(current_q)

        if (
            self._config.use_gripper
            and GRIPPER_NAME in observation
        ):
            self._last_gripper_target = float(
                np.clip(
                    float(observation[GRIPPER_NAME]),
                    0.0,
                    1.0,
                )
            )

        self._connected_robot = robot
        self._robot_synced = True

        logger.info(
            "VR teleoperator synchronized with current RB joints: %s",
            np.round(np.rad2deg(current_q), 2).tolist(),
        )

    # ------------------------------------------------------------------
    # VR state handling
    # ------------------------------------------------------------------

    def _selected_controller(
        self,
        snapshot: VRSnapshot,
    ) -> ControllerSnapshot:
        if self._config.controller_hand == "right":
            return snapshot.right
        return snapshot.left

    def _selected_button_events(
        self,
        events: dict[str, bool],
    ) -> tuple[bool, bool]:
        prefix = self._config.controller_hand

        return (
            bool(events.get(f"{prefix}_primary", False)),
            bool(events.get(f"{prefix}_secondary", False)),
        )

    def _initialize_control(
        self,
        *,
        controller: ControllerSnapshot,
        torso_pose: FloatArray,
    ) -> None:
        assert controller.pose_rb is not None

        if self._config.auto_user_scale:
            try:
                self._user_scale = compute_user_scale(
                    controller.pose_rb,
                    torso_pose,
                    reference_reach_mm=(
                        self._config.reference_reach_mm
                    ),
                )
            except ValueError as exc:
                logger.warning(
                    "User-scale calibration failed; "
                    "VR control remains uninitialized: %s",
                    exc,
                )
                return
        else:
            self._user_scale = float(
                self._config.default_user_scale
            )

        # Re-read the actual robot pose so initialization never causes a
        # jump from a stale internal joint target.
        self._sync_from_robot(force=True)

        self._is_initialized = True
        self._is_stopped = False
        self._following = False

        logger.info(
            "VR control initialized: user_scale=%.4f. "
            "Hold grip to move the arm.",
            self._user_scale,
        )

    def _deactivate_following(self) -> None:
        was_following = self._following
        self._following = False

        if was_following:
            logger.info("VR arm following stopped.")

        if (
            was_following
            and not self._config.hold_last_target
        ):
            try:
                self._sync_from_robot(force=True)
            except Exception as exc:
                logger.warning(
                    "Could not synchronize actual robot pose while "
                    "stopping VR following: %s",
                    exc,
                )

    def _ik_solution_is_acceptable(self) -> bool:
        info = self._kinematics.last_info

        return (
            info.position_error_mm
            <= self._config.max_ik_position_error_mm
            and info.orientation_error
            <= self._config.max_ik_orientation_error
        )

    # ------------------------------------------------------------------
    # Action construction
    # ------------------------------------------------------------------

    def _build_action(self) -> dict[str, Any]:
        action: dict[str, Any] = {
            name: float(value)
            for name, value in zip(
                JOINT_ACTION_NAMES,
                self._last_joint_target,
                strict=True,
            )
        }

        if self._config.use_gripper:
            action[GRIPPER_NAME] = float(
                self._last_gripper_target
            )

        return action
