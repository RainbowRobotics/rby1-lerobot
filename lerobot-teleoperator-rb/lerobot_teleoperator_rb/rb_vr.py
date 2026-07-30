"""Meta Quest VR teleoperator for Rainbow Robotics RB-Series arms.

This module adapts the original hardware-tested RB10E VR control flow to
the LeRobot Teleoperator interface.

Original control behaviour preserved
------------------------------------
1. Meta Quest controller/head poses arrive over UDP.
2. The selected controller is converted into an RB10E Cartesian target.
3. Pressing the primary button (Right-A by default):
   - computes the user's reach scale,
   - arms VR control.
4. Holding Grip:
   - first interpolates from the measured robot joints to the current IK
     solution for five seconds,
   - then continuously follows the VR target using the original IKLM solver.
5. Releasing Grip stops ServoJ transmission.
6. Pressing the secondary button (Right-B by default) disarms control and
   requires the primary button to be pressed again.

The Teleoperator never opens a second robot connection and never sends
ServoJ directly. It returns joint actions in radians. The paired RbCobot
Robot adapter converts radians to degrees and sends one ServoJ command per
LeRobot control tick.
"""

from __future__ import annotations

import logging
import math
import time
from enum import Enum
from typing import Any

import numpy as np

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.errors import (
    DeviceAlreadyConnectedError,
    DeviceNotConnectedError,
)

from lerobot_robot_rb.models import GRIPPER_NAME, JOINT_NAMES
from lerobot_robot_rb.rb_cobot import (
    RbCobot,
    get_active_rb_cobot,
)

from .config_rb_vr import RbVrConfig
from .frame_transforms import (
    compute_user_scale,
    controller_pose_to_rb10e_target,
    head_pose_to_torso,
)
from .rb10e_kinematics import RB10E
from .vr_receiver import (
    VRButtonEvents,
    VRControllerState,
    VRReceiver,
    VRState,
)


logger = logging.getLogger(__name__)


class _ControlState(Enum):
    """Internal VR arm state machine."""

    IDLE = "idle"
    STARTUP = "startup"
    FOLLOWING = "following"


class RbVr(Teleoperator):
    """Meta Quest teleoperator producing RB joint targets in radians."""

    config_class = RbVrConfig
    name = "rb_vr"

    # Original RB10E startup transition duration.
    STARTUP_DURATION_S = 5.0

    def __init__(self, config: RbVrConfig) -> None:
        super().__init__(config)

        self._config = config
        self._is_connected = False

        self._robot: RbCobot | None = None
        self._receiver: VRReceiver | None = None
        self._kinematics: RB10E | None = None

        self._initialized = (
            not config.require_initialization_button
        )
        self._control_state = _ControlState.IDLE

        self._user_scale = float(
            config.default_user_scale
        )

        # Last action returned to LeRobot, radians.
        self._last_action_rad = np.zeros(
            len(JOINT_NAMES),
            dtype=np.float64,
        )

        # Startup interpolation:
        # measured robot q -> current IK q.
        self._startup_q_start = np.zeros(
            len(JOINT_NAMES),
            dtype=np.float64,
        )
        self._startup_q_end = np.zeros(
            len(JOINT_NAMES),
            dtype=np.float64,
        )
        self._startup_time = 0.0

        self._last_waiting_log_time = 0.0
        self._last_tracking_warning_time = 0.0

    # ------------------------------------------------------------------
    # LeRobot properties
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def action_features(self) -> dict[str, type]:
        features = {
            name: float
            for name in JOINT_NAMES
        }

        if self._config.use_gripper:
            features[GRIPPER_NAME] = float

        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(
        self,
        calibrate: bool = True,  # noqa: ARG002
    ) -> None:
        """Start the Quest receiver.

        LeRobot connects the Teleoperator before the Robot:

            teleop.connect()
            robot.connect()

        Therefore, the RB robot is resolved lazily during the first
        get_action() call, after robot.connect() has completed.
        """

        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                f"{self} is already connected."
            )

        receiver = VRReceiver(
            local_ip=self._config.local_ip,
            local_port=self._config.local_port,
            meta_quest_ip=self._config.meta_quest_ip,
            meta_quest_port=self._config.meta_quest_port,
            send_handshake=self._config.send_handshake,
            tracking_timeout_s=(
                self._config.tracking_timeout_s
            ),
        )

        try:
            receiver.start()

            # Robot is connected after Teleoperator.connect() by the
            # standard LeRobot teleoperation script.
            self._robot = None
            self._receiver = receiver

            # Pure kinematics object; opens no robot connection.
            self._kinematics = RB10E()

            self._last_action_rad = np.zeros(
                len(JOINT_NAMES),
                dtype=np.float64,
            )

            self._startup_q_start = np.zeros(
                len(JOINT_NAMES),
                dtype=np.float64,
            )
            self._startup_q_end = np.zeros(
                len(JOINT_NAMES),
                dtype=np.float64,
            )
            self._startup_time = 0.0

            self._initialized = (
                not self._config.require_initialization_button
            )
            self._control_state = _ControlState.IDLE

            self._user_scale = float(
                self._config.default_user_scale
            )

            self._is_connected = True

        except Exception:
            receiver.stop()
            self._robot = None
            self._receiver = None
            self._kinematics = None
            self._is_connected = False
            raise

        logger.info(
            "%s connected. Waiting for Meta Quest packets on %s:%d.",
            self,
            self._config.local_ip,
            self._config.local_port,
        )

        logger.info(
            "RB robot will be synchronized after robot.connect() "
            "during the first teleoperation tick."
        )

        if self._config.require_initialization_button:
            logger.info(
                "Press the selected controller's primary button "
                "to initialize VR control."
            )
        else:
            logger.warning(
                "VR initialization-button requirement is disabled."
            )

    def disconnect(self) -> None:
        if (
            not self.is_connected
            and self._receiver is None
        ):
            return

        if self._robot is not None:
            try:
                self._robot.disable_servo_commands()
            except DeviceNotConnectedError:
                pass

        if self._receiver is not None:
            self._receiver.stop()

        self._receiver = None
        self._robot = None
        self._kinematics = None

        self._initialized = False
        self._control_state = _ControlState.IDLE
        self._is_connected = False

        logger.info(
            "%s disconnected.",
            self,
        )

    # ------------------------------------------------------------------
    # Calibration / configuration
    # ------------------------------------------------------------------

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    def send_feedback(
        self,
        feedback: dict[str, Any],
    ) -> None:
        """Receive optional feedback from the robot.

        RB VR teleoperation does not currently use haptic or visual
        feedback, so the feedback is intentionally ignored.
        """

        del feedback

    # ------------------------------------------------------------------
    # Main LeRobot tick
    # ------------------------------------------------------------------

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self} is not connected."
            )

        robot = self._resolve_robot()
        receiver = self._require_receiver()
        kinematics = self._require_kinematics()

        events = receiver.consume_button_events()
        vr_state = receiver.get_state()

        controller, primary_event, secondary_event = (
            self._selected_controller_and_events(
                vr_state,
                events,
            )
        )

        # B has priority over all other input.
        if secondary_event:
            self._stop_and_disarm(
                require_reinitialization=True,
            )

            logger.info(
                "VR secondary button pressed: "
                "control stopped; press the primary button to restart."
            )

            return self._build_action(
                self._last_action_rad,
                controller,
            )

        # A calculates the scale and arms control.
        if primary_event:
            initialized = self._initialize_control(
                vr_state,
                controller,
            )

            if not initialized:
                return self._build_action(
                    self._last_action_rad,
                    controller,
                )

        if not self._initialized:
            robot.disable_servo_commands()
            self._control_state = _ControlState.IDLE

            now = time.monotonic()
            if (
                now - self._last_waiting_log_time
                >= 2.0
            ):
                logger.info(
                    "Waiting for the selected controller's "
                    "primary button to initialize VR control."
                )
                self._last_waiting_log_time = now

            return self._build_action(
                self._last_action_rad,
                controller,
            )

        if not self._tracking_is_valid(
            vr_state,
            controller,
        ):
            self._stop_following_only()

            now = time.monotonic()
            if (
                now - self._last_tracking_warning_time
                >= 1.0
            ):
                logger.warning(
                    "VR controller/head tracking is missing or stale; "
                    "ServoJ transmission disabled."
                )
                self._last_tracking_warning_time = now

            return self._build_action(
                self._last_action_rad,
                controller,
            )

        torso_pose = head_pose_to_torso(
            vr_state.head.pose_rb
        )

        target_pose = controller_pose_to_rb10e_target(
            controller.pose_rb,
            torso_pose,
            self._user_scale
            * float(self._config.position_scale),
            target_x_offset_mm=(
                self._config.target_x_offset_mm
            ),
            target_y_offset_mm=(
                self._config.target_y_offset_mm
            ),
            target_z_offset_mm=(
                self._config.target_z_offset_mm
            ),
        )

        grip_pressed = (
            controller.buttons.grip
            > self._config.grip_threshold
        )

        # The original loop keeps the IK solution updated while Grip is not
        # pressed. That gives Startup a current target joint vector.
        if self._control_state != _ControlState.STARTUP:
            kinematics.IKLM(
                kinematics._q,
                target_pose,
                self._config.ik_iterations,
            )

        if not grip_pressed:
            if (
                self._control_state
                != _ControlState.IDLE
            ):
                logger.info(
                    "VR arm following stopped."
                )

            robot.disable_servo_commands()
            self._control_state = _ControlState.IDLE

            # Servo gate is OFF, but retain the current IK target as the
            # logical action for the next startup transition.
            self._last_action_rad = (
                robot.get_joint_positions(measured=True)
            )

            return self._build_action(
                self._last_action_rad,
                controller,
            )

        if self._control_state == _ControlState.IDLE:
            self._begin_startup_transition()

            return self._build_action(
                self._last_action_rad,
                controller,
            )

        if (
            self._control_state
            == _ControlState.STARTUP
        ):
            q_out = self._startup_action()
            self._last_action_rad = q_out

            return self._build_action(
                q_out,
                controller,
            )

        # FOLLOWING
        robot.enable_servo_commands()

        self._last_action_rad = (
            kinematics._q.copy()
        )

        return self._build_action(
            self._last_action_rad,
            controller,
        )

    # ------------------------------------------------------------------
    # Button / controller selection
    # ------------------------------------------------------------------

    def _selected_controller_and_events(
        self,
        vr_state: VRState,
        events: VRButtonEvents,
    ) -> tuple[
        VRControllerState,
        bool,
        bool,
    ]:
        if self._config.controller_hand == "right":
            return (
                vr_state.right,
                events.right_primary,
                events.right_secondary,
            )

        if self._config.controller_hand == "left":
            return (
                vr_state.left,
                events.left_primary,
                events.left_secondary,
            )

        raise ValueError(
            "controller_hand must be 'right' or 'left', "
            f"got {self._config.controller_hand!r}."
        )

    # ------------------------------------------------------------------
    # Initialization / stopping
    # ------------------------------------------------------------------

    def _initialize_control(
        self,
        vr_state: VRState,
        controller: VRControllerState,
    ) -> bool:
        robot = self._require_robot()

        if not self._tracking_is_valid(
            vr_state,
            controller,
        ):
            robot.disable_servo_commands()

            logger.warning(
                "Primary button received, but controller/head "
                "tracking is not valid. Initialization was not applied."
            )
            return False

        torso_pose = head_pose_to_torso(
            vr_state.head.pose_rb
        )

        if self._config.auto_user_scale:
            try:
                user_scale = compute_user_scale(
                    controller.pose_rb,
                    torso_pose,
                    reference_reach_mm=(
                        self._config.reference_reach_mm
                    ),
                )
            except ValueError as exc:
                robot.disable_servo_commands()

                logger.warning(
                    "VR user-scale initialization failed: %s",
                    exc,
                )
                return False

            self._user_scale = user_scale
        else:
            self._user_scale = float(
                self._config.default_user_scale
            )

        # Re-initialization always cancels active motion. The operator must
        # hold Grip again after the new scale has been applied.
        robot.disable_servo_commands()
        self._control_state = _ControlState.IDLE
        self._initialized = True

        logger.info(
            "VR control initialized: user_scale=%.4f. "
            "Hold grip to move the arm.",
            self._user_scale,
        )

        return True

    def _stop_following_only(self) -> None:
        robot = self._require_robot()

        robot.disable_servo_commands()
        self._control_state = _ControlState.IDLE

    def _stop_and_disarm(
        self,
        *,
        require_reinitialization: bool,
    ) -> None:
        self._stop_following_only()

        if require_reinitialization:
            self._initialized = False

    # ------------------------------------------------------------------
    # Startup interpolation
    # ------------------------------------------------------------------

    def _begin_startup_transition(self) -> None:
        robot = self._require_robot()
        kinematics = self._require_kinematics()

        q_start = robot.get_joint_positions(
            measured=True
        )
        kinematics.set_q(q_start)
        kinematics.IKLM(
            kinematics._q,
            controller_pose_to_rb10e_target(
                self._require_receiver().get_state().right.pose_rb,
                head_pose_to_torso(
                    self._require_receiver().get_state().head.pose_rb
                ),
                self._user_scale
                * float(self._config.position_scale),
                target_x_offset_mm=(
                    self._config.target_x_offset_mm
                ),
                target_y_offset_mm=(
                    self._config.target_y_offset_mm
                ),
                target_z_offset_mm=(
                    self._config.target_z_offset_mm
                ),
            ),
            self._config.ik_iterations,
        )

        q_end = kinematics._q.copy()

        self._validate_joint_vector(
            q_start,
            name="startup measured joints",
        )
        self._validate_joint_vector(
            q_end,
            name="startup IK joints",
        )

        self._startup_q_start = q_start
        self._startup_q_end = q_end
        self._startup_time = time.monotonic()

        self._last_action_rad = q_start.copy()
        self._control_state = _ControlState.STARTUP

        robot.enable_servo_commands()

        logger.info(
            "VR arm following started. "
            "Beginning %.1f s startup interpolation.",
            self.STARTUP_DURATION_S,
        )

    def _startup_action(self) -> np.ndarray:
        elapsed = (
            time.monotonic()
            - self._startup_time
        )

        phase = float(
            np.clip(
                elapsed / self.STARTUP_DURATION_S,
                0.0,
                1.0,
            )
        )

        # Original interpolation:
        #
        # (q_end - q_start)
        # * (1 - cos(t / 5 * pi)) / 2
        # + q_start
        blend = (
            1.0
            - math.cos(phase * math.pi)
        ) / 2.0

        q_out = (
            self._startup_q_start
            + (
                self._startup_q_end
                - self._startup_q_start
            )
            * blend
        )

        if phase >= 1.0:
            self._control_state = (
                _ControlState.FOLLOWING
            )

            logger.info(
                "VR startup interpolation completed."
            )

        return q_out

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------

    def _tracking_is_valid(
        self,
        vr_state: VRState,
        controller: VRControllerState,
    ) -> bool:
        receiver = self._require_receiver()

        if receiver.is_stale(
            state=vr_state
        ):
            return False

        if not controller.tracked:
            return False

        if not vr_state.head.tracked:
            return False

        return True

    # ------------------------------------------------------------------
    # Action conversion
    # ------------------------------------------------------------------

    def _build_action(
        self,
        joint_rad: np.ndarray,
        controller: VRControllerState,
    ) -> dict[str, Any]:
        self._validate_joint_vector(
            joint_rad,
            name="joint action",
        )

        action: dict[str, Any] = {
            name: float(joint_rad[index])
            for index, name in enumerate(
                JOINT_NAMES
            )
        }

        if self._config.use_gripper:
            # LeRobot convention:
            #   1.0 = open
            #
            # Quest trigger:
            #   0.0 = released
            #   1.0 = fully pressed
            action[GRIPPER_NAME] = (
                1.0
                - float(
                    np.clip(
                        controller.buttons.trigger,
                        0.0,
                        1.0,
                    )
                )
            )

        return action

    @staticmethod
    def _validate_joint_vector(
        joint_rad: np.ndarray,
        *,
        name: str,
    ) -> None:
        values = np.asarray(
            joint_rad,
            dtype=np.float64,
        )

        expected_shape = (
            len(JOINT_NAMES),
        )

        if values.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, "
                f"got {values.shape}."
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"{name} contains non-finite values: "
                f"{values.tolist()}."
            )

    # ------------------------------------------------------------------
    # Required-object helpers
    # ------------------------------------------------------------------

    def _resolve_robot(self) -> RbCobot:
        """Resolve and synchronize the RB robot lazily.

        The standard LeRobot script connects the Teleoperator first and
        the Robot second. By the first get_action() call, robot.connect()
        has completed and registered the active RbCobot instance.
        """

        if (
            self._robot is not None
            and self._robot.is_connected
        ):
            return self._robot

        self._robot = None

        robot = get_active_rb_cobot()

        if robot is None:
            raise DeviceNotConnectedError(
                "No connected RB robot is registered. "
                "RbVr expected robot.connect() to complete before "
                "the first get_action() call."
            )

        current_q = robot.get_joint_positions(
            measured=True
        )

        self._validate_joint_vector(
            current_q,
            name="current robot joints",
        )

        # Start by returning the physical robot's current pose.
        self._last_action_rad = current_q.copy()

        # Motion remains blocked until:
        #   A initialization
        #   + valid tracking
        #   + Grip
        robot.disable_servo_commands()

        self._robot = robot

        logger.info(
            "VR teleoperator synchronized with current RB joints: %s",
            np.round(
                np.rad2deg(current_q),
                3,
            ).tolist(),
        )

        return robot

    def _require_robot(self) -> RbCobot:
        if self._robot is None:
            raise DeviceNotConnectedError(
                "RB robot is not available."
            )

        return self._robot

    def _require_receiver(self) -> VRReceiver:
        if self._receiver is None:
            raise DeviceNotConnectedError(
                "VR receiver is not available."
            )

        return self._receiver

    def _require_kinematics(self) -> RB10E:
        if self._kinematics is None:
            raise DeviceNotConnectedError(
                "RB10E kinematics is not available."
            )

        return self._kinematics
