"""LeRobot Teleoperator for the RB-Y1 driven by a Meta Quest headset.

Control flow
------------
1. The Meta Quest streams controller JSON over UDP to :class:`VRReceiver`.
2. Each call to :meth:`get_action`:

   a. Reads the current robot state (read-only), runs FK to obtain the
      torso / EE poses.
   b. Parses VR controller poses (converted to robot frame).
   c. ``grip > grip_threshold`` → latch start poses (controller & EE).
   d. Computes a Cartesian delta from the start poses to update the target
      EE pose of each tracked component; untracked components keep their
      last target (frozen) so the follower holds position.
   e. Returns the target EE poses — encoded with the LeRobot EE convention
      (``<group>_ee.x/y/z`` in metres plus ``<group>_ee.wx/wy/wz`` rotation
      vector in radians) — together with the gripper and mobile-base
      commands derived from the controllers.

Following the LeRobot convention, this teleoperator only *produces* actions
and never commands the robot.  The paired ``lerobot_robot_rby1.Rby1``
follower (configured with ``action_mode="ee"``) executes them via
``send_action``, driving the onboard Cartesian impedance solver, the gripper
bus and the mobile base.  The robot connection held by this class is
read-only (state + forward kinematics for latching); ready-pose motions are
also owned by the follower (``move_to_ready_on_connect`` / record resets).

Button mapping
--------------
    Right A      (Re-)initialise: latch targets to the current robot pose
    Right B      Stop following
    Grip > 0.8   Activate hand following for that arm
    Trigger      Open/close gripper (emitted as a gripper action)
    Thumbstick   Mobile base velocity (right: linear, left: angular)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_rby1_vr import Rby1VRConfig
from .constants import (
    BASE_VEL_NAMES,
    LEFT_EE_NAMES,
    RIGHT_EE_NAMES,
    TORSO_EE_NAMES,
)
from .frame_transforms import (
    T_HAND_OFFSET,
    cartesian_delta,
    se3_to_pose_action,
    vr_to_robot,
)
from .mobile_base import MobileBaseGains, MobileBaseIntegrator
from .vr_receiver import VRReceiver

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal state containers
# ---------------------------------------------------------------------------

@dataclass
class _FollowingState:
    """Per-component flags: are we actively tracking the controller?"""

    right_arm: bool = False
    left_arm: bool = False
    torso: bool = False


@dataclass
class _LatchedPoses:
    """SE3 latches captured the moment following is (re-)activated."""

    right_controller_start: np.ndarray = field(default_factory=lambda: np.eye(4))
    right_ee_start: np.ndarray = field(default_factory=lambda: np.eye(4))
    right_hand_locked: np.ndarray = field(default_factory=lambda: np.eye(4))

    left_controller_start: np.ndarray = field(default_factory=lambda: np.eye(4))
    left_ee_start: np.ndarray = field(default_factory=lambda: np.eye(4))
    left_hand_locked: np.ndarray = field(default_factory=lambda: np.eye(4))

    head_controller_start: np.ndarray = field(default_factory=lambda: np.eye(4))
    torso_start: np.ndarray = field(default_factory=lambda: np.eye(4))
    torso_locked: np.ndarray = field(default_factory=lambda: np.eye(4))


# ---------------------------------------------------------------------------
# Main teleoperator
# ---------------------------------------------------------------------------

class Rby1VR(Teleoperator):
    """VR teleoperator emitting end-effector pose actions for the RB-Y1."""

    config_class = Rby1VRConfig
    name = "rby1_vr"

    def __init__(self, config: Rby1VRConfig) -> None:
        super().__init__(config)
        self._config = config
        self._is_connected = False

        # rby1_sdk objects (initialised in connect). The robot connection is
        # read-only: state + forward kinematics, never commands.
        self._rby: Any = None
        self._robot: Any = None
        self._model: Any = None
        self._dyn_robot: Any = None
        self._dyn_state: Any = None

        # Link indices for FK (cached after connect).
        self._idx_base = 0
        self._idx_torso_5 = 1
        self._idx_right_arm6 = 2
        self._idx_left_arm6 = 3

        # VR receiver (background UDP thread).
        self._receiver: VRReceiver | None = None

        # High-level state.
        self._is_initialized = False
        self._is_stopped = False
        self._following = _FollowingState()
        self._poses = _LatchedPoses()

        # Mobile-base integrator.
        self._mobile_base = MobileBaseIntegrator(
            MobileBaseGains(
                linear_acc_gain=config.mobile_base_linear_acc_gain,
                angular_acc_gain=config.mobile_base_angular_acc_gain,
                linear_damping=config.mobile_base_linear_damping,
                angular_damping=config.mobile_base_angular_damping,
                output_scale=config.mobile_base_output_scale,
            )
        )

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
                features["right_gripper_0"] = float
            if cfg.use_left_arm:
                features["left_gripper_0"] = float
        if cfg.use_mobile_base:
            for name in BASE_VEL_NAMES:
                features[name] = float
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
        self._rby = rby

        cfg = self._config
        logger.info(
            f"VR Teleoperator connecting to robot at {cfg.robot_address} (read-only) …"
        )

        self._init_robot(rby, cfg)
        self._init_dynamics(rby)
        self._init_receiver(cfg)

        self._is_connected = True
        logger.info(f"{self} connected. Waiting for Right-A button to initialise …")

    def disconnect(self) -> None:
        if not self.is_connected:
            return

        if self._receiver is not None:
            self._receiver.stop()
            self._receiver = None

        if self._robot is not None:
            # Read-only connection: the follower robot owns power, servos and
            # the control manager, so just drop the gRPC link.
            self._robot.disconnect()
            self._robot = None

        self._is_connected = False
        logger.info(f"{self} disconnected.")

    # ------------------------------------------------------------------ #
    # Connect-time helpers
    # ------------------------------------------------------------------ #

    def _init_robot(self, rby: Any, cfg: Rby1VRConfig) -> None:
        """Open a read-only connection used for state reads and FK.

        Power, servos, the control manager and all command streams are owned
        by the follower robot (``lerobot_robot_rby1.Rby1``).
        """
        self._robot = rby.create_robot(cfg.robot_address, cfg.robot_model)
        if not self._robot.connect():
            raise ConnectionError(f"Failed to connect to RB-Y1 at {cfg.robot_address}.")

    def _init_dynamics(self, rby: Any) -> None:
        self._model = self._robot.model()
        self._dyn_robot = self._robot.get_dynamics()
        # Index order matches the link list below.
        self._dyn_state = self._dyn_robot.make_state(
            ["base", "link_torso_5", "link_right_arm_6", "link_left_arm_6"],
            self._model.robot_joint_names,
        )

    def _init_receiver(self, cfg: Rby1VRConfig) -> None:
        self._receiver = VRReceiver(
            local_ip=cfg.local_ip,
            local_port=cfg.local_port,
            meta_quest_ip=cfg.meta_quest_ip,
            meta_quest_port=cfg.meta_quest_port,
        )
        self._receiver.start()

    # ------------------------------------------------------------------ #
    # Calibration / Configuration
    # ------------------------------------------------------------------ #

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    # get_action — main per-tick entry point
    # ------------------------------------------------------------------ #

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Block until the user presses Right-A for the first time.
        if not self._is_initialized:
            self._wait_for_ready_press()

        state = self._robot.get_state()
        torso_T, right_ee_T, left_ee_T = self._compute_current_ee_poses(state)

        events = self._receiver.consume_button_events()
        vr = self._receiver.get_state()

        if events["right_a"]:
            self._initialise_tracking()
        elif events["right_b"]:
            logger.info("Right B pressed — stopping following.")
            self._is_stopped = True
            self._following = _FollowingState()
            self._mobile_base.reset()

        if not self._is_stopped:
            self._update_mobile_base(vr)
            self._update_following(vr, right_ee_T, left_ee_T, torso_T)
            self._update_targets(vr)

        return self._build_action(vr)

    # ------------------------------------------------------------------ #
    # Initialization handshake
    # ------------------------------------------------------------------ #

    def _wait_for_ready_press(self) -> None:
        """Block until the user presses Right-A at least once."""
        last_print = 0.0
        while not self._is_initialized:
            if self._receiver.consume_button_events()["right_a"]:
                logger.info("Right-A received — initialising.")
                self._initialise_tracking()
                return
            now = time.monotonic()
            if now - last_print >= 2.0:
                logger.info("Waiting for Right-A button to initialise …")
                last_print = now
            time.sleep(0.05)

    def _initialise_tracking(self) -> None:
        """(Re-)initialise tracking from the robot's current pose (Right-A).

        Latches the emitted targets to the current EE poses so the follower
        holds position until a grip is squeezed.  Moving the robot to its
        ready pose is owned by the follower robot
        (``move_to_ready_on_connect`` / record resets), not by this
        teleoperator.
        """
        logger.info("Right A pressed — latching targets to the current robot pose.")
        self._mobile_base.reset()
        self._following = _FollowingState()
        self._latch_locked_poses_to_current()
        self._is_initialized = True
        self._is_stopped = False

    def _latch_locked_poses_to_current(self) -> None:
        state = self._robot.get_state()
        self._dyn_state.set_q(state.position.copy())
        self._dyn_robot.compute_forward_kinematics(self._dyn_state)
        self._poses.torso_locked = self._dyn_robot.compute_transformation(
            self._dyn_state, self._idx_base, self._idx_torso_5
        )
        self._poses.right_hand_locked = (
            self._dyn_robot.compute_transformation(
                self._dyn_state, self._idx_base, self._idx_right_arm6
            ) @ T_HAND_OFFSET
        )
        self._poses.left_hand_locked = (
            self._dyn_robot.compute_transformation(
                self._dyn_state, self._idx_base, self._idx_left_arm6
            ) @ T_HAND_OFFSET
        )
        logger.info("Locked poses initialised to current EE positions.")

    # ------------------------------------------------------------------ #
    # Per-tick helpers
    # ------------------------------------------------------------------ #

    def _compute_current_ee_poses(
        self, state: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._dyn_state.set_q(state.position.copy())
        self._dyn_robot.compute_forward_kinematics(self._dyn_state)
        torso_T = self._dyn_robot.compute_transformation(
            self._dyn_state, self._idx_base, self._idx_torso_5
        )
        right_ee_T = (
            self._dyn_robot.compute_transformation(
                self._dyn_state, self._idx_base, self._idx_right_arm6
            ) @ T_HAND_OFFSET
        )
        left_ee_T = (
            self._dyn_robot.compute_transformation(
                self._dyn_state, self._idx_base, self._idx_left_arm6
            ) @ T_HAND_OFFSET
        )
        return torso_T, right_ee_T, left_ee_T

    def _update_mobile_base(self, vr: dict) -> None:
        cfg = self._config
        if not cfg.use_mobile_base:
            return
        hands = vr.get("hands", {})
        right_ctrl = hands.get("right", {})
        left_ctrl = hands.get("left", {})
        right_axis = (
            tuple(right_ctrl["buttons"].get("thumbstickAxis", [0.0, 0.0]))
            if right_ctrl else None
        )
        left_axis = (
            tuple(left_ctrl["buttons"].get("thumbstickAxis", [0.0, 0.0]))
            if left_ctrl else None
        )
        self._mobile_base.update(right_axis, left_axis)

    def _update_following(
        self,
        vr: dict,
        right_ee_T: np.ndarray,
        left_ee_T: np.ndarray,
        torso_T: np.ndarray,
    ) -> None:
        cfg = self._config
        hands = vr.get("hands", {})
        right_ctrl = hands.get("right", {})
        left_ctrl = hands.get("left", {})
        head_ctrl = vr.get("head", {})

        # Right arm
        if right_ctrl:
            right_pose = vr_to_robot(right_ctrl["position"], right_ctrl["rotation"])
            grip_r = right_ctrl["buttons"].get("grip", 0.0)
            if self._following.right_arm and grip_r <= cfg.grip_threshold:
                self._following.right_arm = False
            elif not self._following.right_arm and grip_r > cfg.grip_threshold:
                self._poses.right_controller_start = right_pose
                self._poses.right_ee_start = right_ee_T.copy()
                self._following.right_arm = True
        else:
            self._following.right_arm = False

        # Left arm
        if left_ctrl:
            left_pose = vr_to_robot(left_ctrl["position"], left_ctrl["rotation"])
            grip_l = left_ctrl["buttons"].get("grip", 0.0)
            if self._following.left_arm and grip_l <= cfg.grip_threshold:
                self._following.left_arm = False
            elif not self._following.left_arm and grip_l > cfg.grip_threshold:
                self._poses.left_controller_start = left_pose
                self._poses.left_ee_start = left_ee_T.copy()
                self._following.left_arm = True
        else:
            self._following.left_arm = False

        # Torso (only when head + both arms are tracking).
        if head_ctrl and right_ctrl and left_ctrl:
            head_pose = vr_to_robot(head_ctrl["position"], head_ctrl["rotation"])
            torso_tracking = self._following.right_arm and self._following.left_arm
            if self._following.torso and not torso_tracking:
                self._following.torso = False
            elif not self._following.torso and torso_tracking:
                self._poses.head_controller_start = head_pose
                self._poses.torso_start = torso_T.copy()
                self._following.torso = True
        else:
            self._following.torso = False

    def _update_targets(self, vr: dict) -> None:
        """Update the latched target poses from the current controller deltas.

        Components that are not being followed keep their last target, so the
        follower robot holds them in place.
        """
        hands = vr.get("hands", {})
        right_ctrl = hands.get("right", {})
        left_ctrl = hands.get("left", {})
        head_ctrl = vr.get("head", {})

        if self._following.right_arm and right_ctrl:
            right_pose = vr_to_robot(right_ctrl["position"], right_ctrl["rotation"])
            self._poses.right_hand_locked = cartesian_delta(
                self._poses.right_controller_start, right_pose, self._poses.right_ee_start
            )

        if self._following.left_arm and left_ctrl:
            left_pose = vr_to_robot(left_ctrl["position"], left_ctrl["rotation"])
            self._poses.left_hand_locked = cartesian_delta(
                self._poses.left_controller_start, left_pose, self._poses.left_ee_start
            )

        if self._following.torso and head_ctrl:
            head_pose = vr_to_robot(head_ctrl["position"], head_ctrl["rotation"])
            diff = np.linalg.inv(self._poses.head_controller_start) @ head_pose
            # Ignore translation — only the rotation is meaningful for the torso.
            diff[0, 3] = 0.0
            diff[1, 3] = 0.0
            self._poses.torso_locked = self._poses.torso_start @ diff

    # ------------------------------------------------------------------ #
    # Action / feedback
    # ------------------------------------------------------------------ #

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def _build_action(self, vr: dict) -> dict[str, Any]:
        """Build the LeRobot action dict from the latched target poses.

        The targets are the EE poses the follower's Cartesian impedance
        solver should track: while a component is followed they are updated
        from the controller delta each tick, otherwise they stay frozen so
        the robot holds position.  The gripper and mobile-base values are
        derived from the VR controllers; the follower robot's ``send_action``
        executes everything.
        """
        cfg = self._config
        action: dict[str, Any] = {}
        if cfg.use_torso:
            action.update(se3_to_pose_action(self._poses.torso_locked, "torso_ee"))
        if cfg.use_right_arm:
            action.update(se3_to_pose_action(self._poses.right_hand_locked, "right_ee"))
        if cfg.use_left_arm:
            action.update(se3_to_pose_action(self._poses.left_hand_locked, "left_ee"))

        # Gripper: trigger 1 (squeezed) -> closed.  The follower uses the
        # dataset convention 1.0 = open, 0.0 = closed, so emit 1 - trigger.
        if cfg.use_gripper:
            right_trigger, left_trigger = self._triggers_from_vr(vr)
            if cfg.use_right_arm:
                action["right_gripper_0"] = 1.0 - right_trigger
            if cfg.use_left_arm:
                action["left_gripper_0"] = 1.0 - left_trigger

        # Mobile base: body-frame velocity from the thumbstick integrator.
        if cfg.use_mobile_base:
            linear, angular = self._mobile_base.velocity
            action["x.vel"] = float(linear[0])
            action["y.vel"] = float(linear[1])
            action["theta.vel"] = float(angular)

        return action

    @staticmethod
    def _triggers_from_vr(vr: dict) -> tuple[float, float]:
        """Return the (right, left) analog trigger values, defaulting to 0.0."""
        hands = vr.get("hands", {})
        right_ctrl = hands.get("right", {})
        left_ctrl = hands.get("left", {})
        right_trigger = (
            float(right_ctrl["buttons"].get("trigger", 0.0)) if right_ctrl else 0.0
        )
        left_trigger = (
            float(left_ctrl["buttons"].get("trigger", 0.0)) if left_ctrl else 0.0
        )
        return right_trigger, left_trigger
