"""Meta Quest VR teleoperator for Rainbow Robotics RB-Series arms.

This module adapts the original hardware-tested RB10E VR control flow to
the LeRobot Teleoperator interface.

Control flow
------------
1. Meta Quest controller/head poses arrive over UDP.
2. Pressing the primary button (Right-A by default) arms VR control and
   interpolates the arm to the fixed home pose (constants.VR_HOME_POSE_DEG)
   over five seconds. Grip is ignored until the arm arrives.
   A also re-zeroes the operator reference frame to the headset's current
   pose. That frame defines the hand-to-robot axis mapping and is held
   until the next A press.
3. Pressing Grip captures a clutch anchor: the controller pose and the
   robot's end-effector pose at that instant.
4. While Grip is held, the controller's translation since the anchor is
   added 1:1 to the anchor end-effector position, in the robot base frame.
   The controller's rotation since the anchor is applied to the anchor
   orientation about the same axis, scaled by RbVrConfig.orientation_scale.
   There is no separate cap on the commanded rotation: the per-tick
   MAX_JOINT_STEP_RAD limit and RB10E's joint clipping are what bound it.
   Setting orientation_scale to 0.0 restores the frozen-orientation
   behaviour exactly.
5. Releasing Grip stops ServoJ transmission and discards the anchor, so the
   next press re-anchors from wherever the arm and hand then are.
6. Pressing the secondary button (Right-B by default) disarms control and
   requires the primary button to be pressed again.

The controller-to-end-effector position mapping is 1:1 and is measured in
the operator frame captured at the moment A is pressed. Freezing it means
head rotation moves the arm neither during a stroke nor between strokes;
only A re-zeroes it.

The ROTATION mapping is scaled rather than 1:1, because a wrist flick is
cheap for the operator and expensive for the arm. orientation_scale is the
ONLY thing bounding how far a stroke can rotate the tool, so it is also the
knob to reach for if a stroke stops tracking. At the current home pose the
first margin to run out is the ELBOW: joint 2 sits 12.11 deg from its
+-154 deg IK limit, and IKLM clips there, so the arm saturates rather than
follows. Rotations about base Y spend it fastest, roughly 6.8 deg for a
30 deg tool rotation. The wrist is the looser of the two: joint 4 parks at
277.47 deg, 82.5 deg from the joint 4 == 360 deg singularity, and base Z
rotation is what spends that one.

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
from .constants import VR_HOME_POSE_RAD
from .frame_transforms import (
    build_delta_target,
    build_position_delta_target,
    controller_rotation_delta,
    controller_translation_delta_mm,
    head_pose_to_torso,
    scale_rotation,
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
    HOMING = "homing"
    FOLLOWING = "following"


class RbVr(Teleoperator):
    """Meta Quest teleoperator producing RB joint targets in radians."""

    config_class = RbVrConfig
    name = "rb_vr"

    # Duration of the cosine ramp to the fixed home pose.
    HOMING_DURATION_S = 5.0

    # Grip is an analog axis. Without hysteresis a hand resting near the
    # threshold chatters between IDLE and FOLLOWING at tick rate, toggling
    # the servo gate and re-anchoring every tick.
    GRIP_RELEASE_HYSTERESIS = 0.1

    # Largest controller translation, in millimetres, that may be applied to
    # the anchor end-effector position. Bounds how far outside the workspace
    # the operator can drag the IK target.
    MAX_POSITION_DELTA_MM = 700.0

    # Position residual above which the IK solution is considered
    # non-convergent and the previous command is held instead.
    IK_RESIDUAL_LIMIT_MM = 50.0

    # Largest per-tick change of any single joint. ~2.9 deg/tick, which is
    # ~86 deg/s at 30 Hz. Neither RbCobot.send_action nor the RB control box
    # applies a rate limit, so this is the only one in the chain.
    MAX_JOINT_STEP_RAD = 0.05

    # Tolerance for "the first command after a transition must not jump".
    FIRST_COMMAND_TOLERANCE_RAD = math.radians(1.0)

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

        # Last action returned to LeRobot, radians.
        self._last_action_rad = np.zeros(
            len(JOINT_NAMES),
            dtype=np.float64,
        )

        # Homing interpolation:
        # measured robot q -> VR_HOME_POSE_RAD.
        self._homing_q_start = np.zeros(
            len(JOINT_NAMES),
            dtype=np.float64,
        )
        self._homing_q_end = np.zeros(
            len(JOINT_NAMES),
            dtype=np.float64,
        )
        self._homing_time = 0.0

        # Previous grip level, for rising-edge detection. The receiver
        # publishes grip as an analog level and emits no grip event.
        self._grip_pressed = False

        # Operator reference frame, captured from the headset when the
        # primary (A) button is pressed and held until the next A press.
        #
        # It deliberately outlives individual grip strokes: re-capturing it
        # per stroke would silently change the hand-to-robot axis mapping
        # whenever the operator turned between strokes. A is the explicit
        # "re-zero my frame to where I am standing now" gesture.
        self._operator_torso_pose: np.ndarray | None = None

        # Clutch anchor, captured when Grip is pressed. These stay None
        # while no anchor is held, so following without one raises instead
        # of silently commanding an identity target.
        self._anchor_controller_pose: np.ndarray | None = None
        self._anchor_ee_pose: np.ndarray | None = None
        self._anchor_q_rad: np.ndarray | None = None

        self._last_waiting_log_time = 0.0
        self._last_tracking_warning_time = 0.0
        self._last_clamp_log_time = 0.0
        self._last_residual_log_time = 0.0

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

            self._homing_q_start = np.zeros(
                len(JOINT_NAMES),
                dtype=np.float64,
            )
            self._homing_q_end = np.zeros(
                len(JOINT_NAMES),
                dtype=np.float64,
            )
            self._homing_time = 0.0

            self._grip_pressed = False
            self._operator_torso_pose = None
            self._clear_anchor()

            self._initialized = (
                not self._config.require_initialization_button
            )
            self._control_state = _ControlState.IDLE

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
        self._grip_pressed = False
        self._operator_torso_pose = None
        self._clear_anchor()
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

        events = receiver.consume_button_events()
        vr_state = receiver.get_state()

        controller, primary_event, secondary_event = (
            self._selected_controller_and_events(
                vr_state,
                events,
            )
        )

        tracking_ok = self._tracking_is_valid(
            vr_state,
            controller,
        )

        # The grip edge detector must run exactly once per tick in EVERY
        # state, including HOMING. Skipping it while homing would let a grip
        # held across the whole ramp produce a rising edge on arrival, and
        # following would start without the operator asking for it.
        grip_pressed, grip_rising = self._update_grip_state(
            controller,
            tracking_ok=tracking_ok,
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

        # Homing owns the whole tick: it is an open-loop joint trajectory to
        # a constant, so it consumes no VR input and ignores A and Grip.
        if self._control_state is _ControlState.HOMING:
            if primary_event:
                logger.debug(
                    "Primary button ignored: homing is in progress."
                )

            q_out = self._limit_joint_step(
                self._homing_action()
            )
            self._last_action_rad = q_out

            return self._build_action(
                q_out,
                controller,
            )

        # A arms control and sends the arm to the fixed home pose.
        if primary_event:
            self._begin_homing(vr_state, controller)

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

        if not tracking_ok:
            # Dropping the anchor here is what prevents a lunge when
            # tracking returns after the operator's hand has moved.
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

        if (
            self._control_state is _ControlState.FOLLOWING
            and not grip_pressed
        ):
            logger.info("VR arm following stopped.")

            self._stop_following_only()

            self._last_action_rad = (
                robot.get_joint_positions(measured=True)
            )

            return self._build_action(
                self._last_action_rad,
                controller,
            )

        if (
            self._control_state is _ControlState.IDLE
            and grip_rising
        ):
            if not self._begin_following(
                vr_state,
                controller,
            ):
                return self._build_action(
                    self._last_action_rad,
                    controller,
                )

            # The anchor delta is zero on this tick by construction, so the
            # first command is the measured pose. Emit it directly rather
            # than running IK for a target we know is a fixed point.
            return self._build_action(
                self._last_action_rad,
                controller,
            )

        if self._control_state is _ControlState.IDLE:
            robot.disable_servo_commands()

            # Keep the reported action on the physical arm while idle, so
            # the next anchor and the LeRobot dataset both see the truth.
            self._last_action_rad = (
                robot.get_joint_positions(measured=True)
            )

            return self._build_action(
                self._last_action_rad,
                controller,
            )

        # FOLLOWING
        q_out = self._limit_joint_step(
            self._following_action(controller)
        )

        robot.enable_servo_commands()
        self._last_action_rad = q_out

        return self._build_action(
            q_out,
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
    # Grip edge detection
    # ------------------------------------------------------------------

    def _update_grip_state(
        self,
        controller: VRControllerState,
        *,
        tracking_ok: bool,
    ) -> tuple[bool, bool]:
        """Return ``(grip_pressed, grip_rising_edge)``.

        The receiver publishes grip as an analog level and emits no grip
        event, so the rising edge is derived here.

        MUST be called exactly once per tick, in every state, before any
        state-dependent branching. Skipping it in some states would let a
        grip held across a state change produce a spurious rising edge.

        Invalid tracking reports "not pressed" but FREEZES the remembered
        level instead of clearing it. Clearing it would manufacture a
        falling edge, and the operator still holding Grip when tracking
        recovers would then look like a fresh press: the arm would resume
        moving on its own after a dropout. Freezing means an operator who
        keeps holding Grip gets no rising edge and no resume, while one who
        actually releases and presses again does.
        """

        if not tracking_ok:
            return False, False

        level = float(
            np.clip(
                controller.buttons.grip,
                0.0,
                1.0,
            )
        )

        press_level = float(
            self._config.grip_threshold
        )
        release_level = max(
            0.0,
            press_level - self.GRIP_RELEASE_HYSTERESIS,
        )

        pressed = level > (
            release_level
            if self._grip_pressed
            else press_level
        )
        rising = pressed and not self._grip_pressed

        self._grip_pressed = pressed

        return pressed, rising

    # ------------------------------------------------------------------
    # Stopping
    # ------------------------------------------------------------------

    def _stop_following_only(self) -> None:
        robot = self._require_robot()

        robot.disable_servo_commands()
        self._control_state = _ControlState.IDLE
        self._clear_anchor()

    def _stop_and_disarm(
        self,
        *,
        require_reinitialization: bool,
    ) -> None:
        self._stop_following_only()

        if require_reinitialization:
            self._initialized = False

            # Re-arming goes through A, which captures a fresh frame.
            self._operator_torso_pose = None

    # ------------------------------------------------------------------
    # Homing to the fixed VR home pose
    # ------------------------------------------------------------------

    def _begin_homing(
        self,
        vr_state: VRState,
        controller: VRControllerState,
    ) -> None:
        """Start the cosine ramp to the fixed home pose.

        Homing is an open-loop joint-space trajectory to a constant and
        consumes no VR data, so it is deliberately NOT aborted by tracking
        loss: a single dropped UDP packet would otherwise park the arm in an
        arbitrary intermediate pose. The B button always aborts it.

        A also re-zeroes the operator reference frame to the headset's
        current pose, so the hand-to-robot axis mapping is re-established
        for wherever the operator is now standing and facing.
        """

        robot = self._require_robot()
        kinematics = self._require_kinematics()

        q_start = robot.get_joint_positions(
            measured=True
        )
        self._validate_joint_vector(
            q_start,
            name="homing measured joints",
        )

        q_end = np.asarray(
            VR_HOME_POSE_RAD,
            dtype=np.float64,
        )
        self._validate_joint_vector(
            q_end,
            name="VR home pose",
        )

        self._homing_q_start = q_start.copy()
        self._homing_q_end = q_end.copy()
        self._homing_time = time.monotonic()

        self._clear_anchor()

        # Re-zero the operator frame. Only a tracked headset can define it;
        # if tracking is bad right now, keep whatever frame we already had
        # rather than silently adopting a garbage one.
        if self._tracking_is_valid(vr_state, controller):
            self._operator_torso_pose = head_pose_to_torso(
                vr_state.head.pose_rb
            )
            logger.info("Operator reference frame re-zeroed to the headset.")
        elif self._operator_torso_pose is None:
            logger.warning(
                "Homing without headset tracking: no operator reference "
                "frame captured. Press A again with tracking valid before "
                "using Grip."
            )
        else:
            logger.warning(
                "Homing without headset tracking: keeping the previous "
                "operator reference frame."
            )

        # Keep the solver on the physical arm.
        kinematics.set_q(q_start)

        # blend(0) == 0, so the first command emitted after the servo gate
        # opens is exactly the measured pose.
        self._last_action_rad = q_start.copy()
        self._initialized = True
        self._control_state = _ControlState.HOMING

        robot.enable_servo_commands()

        logger.info(
            "Homing to %s deg over %.1f s. "
            "Release and press Grip afterwards to follow.",
            np.round(
                np.rad2deg(q_end),
                3,
            ).tolist(),
            self.HOMING_DURATION_S,
        )

    def _homing_action(self) -> np.ndarray:
        elapsed = (
            time.monotonic()
            - self._homing_time
        )

        phase = float(
            np.clip(
                elapsed / self.HOMING_DURATION_S,
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
            self._homing_q_start
            + (
                self._homing_q_end
                - self._homing_q_start
            )
            * blend
        )

        arrived = bool(
            np.max(
                np.abs(q_out - self._homing_q_end)
            )
            <= self.FIRST_COMMAND_TOLERANCE_RAD
        )

        if phase >= 1.0 and arrived:
            # _limit_joint_step can make the emitted trajectory lag the
            # clock under a scheduler stall, so the clock alone is not
            # enough to declare arrival.
            self._require_kinematics().set_q(
                self._homing_q_end
            )
            self._last_action_rad = (
                self._homing_q_end.copy()
            )
            self._control_state = _ControlState.IDLE
            self._require_robot().disable_servo_commands()

            logger.info(
                "Homing complete. "
                "Press and hold Grip to follow."
            )

        elif elapsed > 2.0 * self.HOMING_DURATION_S:
            logger.error(
                "Homing did not converge within %.1f s; stopping.",
                2.0 * self.HOMING_DURATION_S,
            )
            self._stop_following_only()

        return q_out

    # ------------------------------------------------------------------
    # Clutch anchor and delta following
    # ------------------------------------------------------------------

    def _begin_following(
        self,
        vr_state: VRState,
        controller: VRControllerState,
    ) -> bool:
        """Capture the clutch anchor and engage delta following.

        Returns ``False``, leaving the servo gate closed, if the zero-delta
        self-check fails.
        """

        robot = self._require_robot()
        kinematics = self._require_kinematics()

        q_measured = robot.get_joint_positions(
            measured=True
        )
        self._validate_joint_vector(
            q_measured,
            name="anchor measured joints",
        )

        # set_q refreshes the cached FK, so get_fk() below is exactly
        # FK(q_measured).
        kinematics.set_q(q_measured)
        ee_anchor = kinematics.get_fk()

        if self._operator_torso_pose is None:
            # Only reachable when require_initialization_button is False, so
            # A was never pressed. Fall back to capturing the frame here so
            # that configuration keeps working.
            self._operator_torso_pose = head_pose_to_torso(
                vr_state.head.pose_rb
            )
            logger.info(
                "No operator reference frame yet; capturing it at this "
                "grip press. Press A to re-zero it deliberately."
            )

        controller_anchor = np.asarray(
            controller.pose_rb,
            dtype=np.float64,
        ).copy()

        # The target at delta zero must be a fixed point of the IK. If it is
        # not, a frame or unit error would show up as a lurch the moment the
        # gate opens, so refuse to follow instead.
        #
        # Deliberately the position-only target. Any rotation delta measured
        # at this instant is the identity by construction (the controller
        # pose IS the anchor), so routing the probe through the orientation
        # path could not exercise it and would only make this safety check
        # depend on orientation_scale. The probe therefore cannot catch a
        # rotation-frame error; _limit_joint_step and
        # tests/manual/replay_delta.py are what cover that, plus the
        # low-gain first-run procedure in RUN.md.
        zero_target = build_position_delta_target(
            ee_anchor,
            np.zeros(3),
        )
        q_check = kinematics.IKLM(
            kinematics.get_q(),
            zero_target,
            self._config.ik_iterations,
        )
        drift = float(
            np.max(np.abs(q_check - q_measured))
        )

        # Undo the probe either way.
        kinematics.set_q(q_measured)

        if drift > self.FIRST_COMMAND_TOLERANCE_RAD:
            robot.disable_servo_commands()

            logger.error(
                "Refusing to follow: zero-delta IK moved by %.3f deg. "
                "This indicates a frame or unit error.",
                math.degrees(drift),
            )
            return False

        self._anchor_controller_pose = controller_anchor
        self._anchor_ee_pose = ee_anchor
        self._anchor_q_rad = q_measured.copy()

        self._last_action_rad = q_measured.copy()
        self._control_state = _ControlState.FOLLOWING

        robot.enable_servo_commands()

        logger.info(
            "VR following engaged: anchored 1:1 position delta, "
            "orientation gain %.2f.",
            self._config.orientation_scale,
        )

        return True

    def _following_action(
        self,
        controller: VRControllerState,
    ) -> np.ndarray:
        kinematics = self._require_kinematics()

        (
            operator_torso,
            controller_anchor,
            ee_anchor,
        ) = self._require_anchor()

        delta_mm = controller_translation_delta_mm(
            controller.pose_rb,
            controller_anchor,
            operator_torso,
        )
        delta_mm = self._limit_position_delta(delta_mm)

        # None means "hold the anchor orientation", which is bit-for-bit
        # the pre-orientation control law. At gain 0 the rotation path is
        # not merely a no-op, it is not executed at all: with the feature
        # switched off, a malformed controller rotation cannot raise.
        rotation_delta = None
        gain = float(self._config.orientation_scale)

        if gain > 0.0:
            rotation_delta = scale_rotation(
                controller_rotation_delta(
                    controller.pose_rb,
                    controller_anchor,
                    operator_torso,
                ),
                gain,
            )

        target = build_delta_target(
            ee_anchor,
            delta_mm,
            rotation_delta=rotation_delta,
        )

        # Warm start from the previous solution, not from the measurement:
        # IKLM runs a fixed iteration count with no convergence test, and
        # re-seeding from a lagging measurement feeds servo tracking error
        # back into the command.
        q_out = kinematics.IKLM(
            kinematics.get_q(),
            target,
            self._config.ik_iterations,
        )

        # IKLM never signals failure: it clips to the joint limits every
        # iteration and always returns a vector. Check the residual instead.
        # update_error_vector returns its internal buffer by reference, so
        # reduce it immediately.
        #
        # Position rows ONLY, and that stays true now that orientation
        # varies. The orientation rows are 2*sin(theta)*axis: dimensionless
        # against millimetres, non-monotonic, peaking at 90 deg and back to
        # ZERO at 180 deg -- a fully inverted tool is indistinguishable from
        # a converged one, so no threshold on them can mean "did not
        # converge". Gating on them would also be worse than useless here:
        # the hold below returns the whole previous command, so a
        # wrist-limited tilt would stall translation too. Leaving them out
        # means an unreachable orientation degrades to "right position,
        # partial tilt", which is the behaviour we want.
        #
        # Two couplings worth knowing. IKLM folds 0.5*e.T@e into its LM
        # damping and the orientation rows now contribute to it, but at
        # ~1.0 against a base damping of 2.0 and a J.T@J whose leading
        # singular values are ~1e5 that is negligible.
        #
        # The one that matters: a commanded rotation of exactly 180 deg is
        # a STATIONARY POINT of this solver. The error rows vanish there,
        # so the arm stops rotating and holds the anchor orientation while
        # the position rows stay converged -- an orientation stall that
        # this check cannot see, and that looks like "rotation stopped
        # working". Reaching it needs orientation_scale == 1.0 and a 180
        # deg hand rotation; any gain below 1.0 caps the command short of
        # it.
        residual_mm = float(
            np.linalg.norm(
                kinematics.update_error_vector(
                    target,
                    kinematics.get_fk(),
                )[0:3]
            )
        )

        if residual_mm > self.IK_RESIDUAL_LIMIT_MM:
            now = time.monotonic()
            if now - self._last_residual_log_time >= 1.0:
                logger.warning(
                    "IK did not reach the target (%.1f mm residual); "
                    "holding the previous command. The target is likely "
                    "outside the workspace.",
                    residual_mm,
                )
                self._last_residual_log_time = now

            # Stay in FOLLOWING: the arm parks at the workspace boundary and
            # resumes smoothly when the hand comes back, with no re-clutch.
            return self._last_action_rad.copy()

        return q_out

    # ------------------------------------------------------------------
    # Anchor bookkeeping
    # ------------------------------------------------------------------

    def _clear_anchor(self) -> None:
        # The operator reference frame is NOT cleared here: it survives
        # grip releases and tracking dropouts, and is replaced only by the
        # next A press.
        self._anchor_controller_pose = None
        self._anchor_ee_pose = None
        self._anchor_q_rad = None

    def _require_anchor(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if (
            self._operator_torso_pose is None
            or self._anchor_controller_pose is None
            or self._anchor_ee_pose is None
        ):
            raise RuntimeError(
                "VR following requires an operator frame and a clutch "
                "anchor, but at least one is missing."
            )

        return (
            self._operator_torso_pose,
            self._anchor_controller_pose,
            self._anchor_ee_pose,
        )

    # ------------------------------------------------------------------
    # Safety limiters
    # ------------------------------------------------------------------

    def _limit_position_delta(
        self,
        delta_mm: np.ndarray,
    ) -> np.ndarray:
        """Bound the commanded translation to MAX_POSITION_DELTA_MM.

        The norm is scaled, never the individual axes: clamping per axis
        would rotate the commanded direction, so the arm would veer sideways
        at the limit instead of simply stopping short.
        """

        magnitude = float(np.linalg.norm(delta_mm))

        if magnitude <= self.MAX_POSITION_DELTA_MM:
            return delta_mm

        now = time.monotonic()
        if now - self._last_clamp_log_time >= 1.0:
            logger.warning(
                "Controller delta %.0f mm exceeds the %.0f mm limit; "
                "clamping.",
                magnitude,
                self.MAX_POSITION_DELTA_MM,
            )
            self._last_clamp_log_time = now

        return delta_mm * (
            self.MAX_POSITION_DELTA_MM / magnitude
        )

    def _limit_joint_step(
        self,
        q_out: np.ndarray,
    ) -> np.ndarray:
        """Bound the per-tick change of every joint.

        Neither RbCobot.send_action nor the RB control box rate-limits a
        ServoJ command, so this is the only rate limit in the chain. A
        tracking glitch that teleports the controller produces a large but
        reachable target, which neither the delta clamp nor the residual
        check catches.

        The whole step is scaled so the joint-space direction is preserved.
        """

        step = q_out - self._last_action_rad
        peak = float(np.max(np.abs(step)))

        if peak <= self.MAX_JOINT_STEP_RAD:
            return q_out

        return (
            self._last_action_rad
            + step * (self.MAX_JOINT_STEP_RAD / peak)
        )

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

        # Keep the solver seeded on the physical arm from the first tick.
        self._require_kinematics().set_q(current_q)

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
