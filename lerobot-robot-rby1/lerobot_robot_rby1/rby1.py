"""LeRobot Robot interface for the Rainbow Robotics RB-Y1.

Implements the :class:`lerobot.robots.robot.Robot` abstract class so the RB-Y1
can be used with the standard LeRobot tools (data collection, teleoperation,
imitation-learning inference).

Joint layout (Model M, 26-DOF)::

    wheel_fr/fl/rr/rl   mobility            (excluded from obs/action)
    torso_0 .. torso_5  torso, 6 DOF        (observation only by default)
    right_arm_0 .. _6   right arm, 7 DOF    (obs + action)
    left_arm_0 .. _6    left arm, 7 DOF     (obs + action)
    head_0, head_1      head                (excluded from obs/action)
    + two Dynamixel gripper motors (right=0, left=1) on /dev/rby1_gripper

The two arm grippers are exposed as normalised scalars; see
:class:`~lerobot_robot_rby1.gripper.Rby1Gripper` for the hardware convention
(0.0 = open, 1.0 = closed) and the dataset convention used in observations and
actions (1.0 = open, 0.0 = closed).

Action modes
------------
``action_mode="joint"`` (default): actions are joint positions (radians) for
the enabled arms, executed as joint position / impedance commands.

``action_mode="ee"``: actions are end-effector poses in the base frame
(``torso_ee.* / right_ee.* / left_ee.*``, position in metres plus a rotation
vector in radians — the LeRobot EE convention) executed via the onboard
Cartesian impedance solver.  This is the mode produced by the
``rby1_vr`` teleoperator.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from . import command_builders as cb
from . import model_probe
from .config_rby1 import (
    READY_HEAD,
    READY_LEFT,
    READY_POSE,
    READY_RIGHT,
    Rby1Config,
    ready_pose_for_version,
)
from .constants import (
    ARM_DOF,
    BASE_VEL_NAMES,
    LEFT_ARM_NAMES,
    LEFT_EE_NAMES,
    RIGHT_ARM_NAMES,
    RIGHT_EE_NAMES,
    TORSO_DOF,
    TORSO_EE_NAMES,
    TORSO_NAMES,
    TOTAL_BODY_DOF,
)
from .gripper import Rby1Gripper

logger = logging.getLogger(__name__)

# Body-position slices within the 20-DOF [torso | right arm | left arm] vector.
_RIGHT_ARM_BODY_SLICE = slice(TORSO_DOF, TORSO_DOF + ARM_DOF)          # 6:13
_LEFT_ARM_BODY_SLICE = slice(TORSO_DOF + ARM_DOF, TOTAL_BODY_DOF)      # 13:20

# Link indices for forward kinematics, matching the order of the link list
# passed to ``make_state`` in connect() (EE-pose observations, EE mode only).
_FK_LINKS = ["base", "link_torso_5", "link_right_arm_6", "link_left_arm_6"]
_IDX_BASE, _IDX_TORSO_5, _IDX_RIGHT_ARM_6, _IDX_LEFT_ARM_6 = range(4)


class Rby1(Robot):
    """LeRobot Robot implementation for the Rainbow Robotics RB-Y1 (Model M).

    Observation features
    --------------------
    * Joint positions (radians) for each enabled group: ``torso_0..5`` (only
      when ``use_torso``), ``right_arm_0..6``, ``left_arm_0..6``.
    * Gripper positions (normalised, 1.0 = open) for each enabled arm.
    * Optional ``<joint>.vel`` and ``<joint>.torque`` channels for the 20 body
      joints when ``use_velocity`` / ``use_torque`` are set.
    * One ``(H, W, 3)`` image per configured camera.

    Action features
    ---------------
    In joint mode, joint positions (radians) for the enabled arms plus the
    enabled grippers; the torso is observation-only and is never commanded.
    In EE mode (``action_mode="ee"``), end-effector poses
    (``<group>_ee.x/y/z/wx/wy/wz``) for each enabled group — including the
    torso when ``use_torso`` is set — plus the enabled grippers.

    Example
    -------
    >>> cfg = Rby1Config(address="192.168.30.1:50051")
    >>> robot = Rby1(cfg)
    >>> robot.connect()
    >>> obs = robot.get_observation()
    >>> robot.send_action(obs)   # replicate the current pose
    >>> robot.disconnect()
    """

    config_class = Rby1Config
    name = "rby1"

    def __init__(self, config: Rby1Config) -> None:
        super().__init__(config)
        self._config = config
        self._robot = None
        self._model = None
        self._stream = None
        self._gripper: Rby1Gripper | None = None
        self._max_qdot: np.ndarray | None = None
        self._max_qddot: np.ndarray | None = None
        self._is_connected: bool = False
        self.cameras = make_cameras_from_configs(config.cameras)
        # Timestamp of the first send_action() call; the startup ramp begins
        # from that moment (None until the first command is sent).
        self._action_start_time: float | None = None
        # EE mode: last commanded Cartesian targets (None = next command
        # resets the impedance reference) and nullspace arm targets.
        self._last_ee_targets: cb.CartesianTargets | None = None
        self._null_right: np.ndarray = np.deg2rad(config.null_right_arm_deg)
        self._null_left: np.ndarray = np.deg2rad(config.null_left_arm_deg)
        # EE mode: dynamics model + FK state for end-effector observations.
        self._dyn_robot: Any = None
        self._fk_state: Any = None
        # Resolved model / version (set at connect() — possibly via auto-probe)
        # and the version-specific ready pose selected from them.
        self._resolved_model: str | None = None
        self._resolved_version: str | None = None
        self._ready_body: np.ndarray = READY_POSE
        self._ready_right: np.ndarray = READY_RIGHT
        self._ready_left: np.ndarray = READY_LEFT
        self._ready_head: np.ndarray = READY_HEAD

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @is_connected.setter
    def is_connected(self, value: bool) -> None:
        self._is_connected = value

    @property
    def is_calibrated(self) -> bool:
        # RB-Y1 uses absolute encoders - no calibration required.
        return True

    # ------------------------------------------------------------------ #
    #  Features                                                            #
    # ------------------------------------------------------------------ #

    @property
    def _motors_ft(self) -> dict[str, type]:
        names: list[str] = []
        if self._config.action_mode == "ee":
            # EE poses for each enabled group (incl. the torso, which is
            # commanded by the Cartesian solver in this mode).
            if self._config.use_torso:
                names += TORSO_EE_NAMES
            if self._config.use_right_arm:
                names += RIGHT_EE_NAMES
            if self._config.use_left_arm:
                names += LEFT_EE_NAMES
            return {name: float for name in names}
        # Joint mode: torso is observation-only and is intentionally excluded.
        if self._config.use_right_arm:
            names += RIGHT_ARM_NAMES
        if self._config.use_left_arm:
            names += LEFT_ARM_NAMES
        return {name: float for name in names}

    @property
    def _obs_ft(self) -> dict[str, type]:
        names: list[str] = []
        if self._config.use_torso:
            names += TORSO_NAMES
        if self._config.use_right_arm:
            names += RIGHT_ARM_NAMES
        if self._config.use_left_arm:
            names += LEFT_ARM_NAMES
        return {name: float for name in names}

    @property
    def _gripper_ft(self) -> dict[str, type]:
        if not self._config.use_gripper:
            return {}
        names: list[str] = []
        if self._config.use_right_arm:
            names.append("right_gripper_0")
        if self._config.use_left_arm:
            names.append("left_gripper_0")
        return {name: float for name in names}

    @property
    def _base_ft(self) -> dict[str, type]:
        if not self._config.use_mobile_base:
            return {}
        return {name: float for name in BASE_VEL_NAMES}

    @property
    def _ee_obs_ft(self) -> dict[str, type]:
        # In EE mode the observation additionally exposes the end-effector
        # pose of each enabled group (computed via forward kinematics).
        if self._config.action_mode != "ee":
            return {}
        names: list[str] = []
        if self._config.use_torso:
            names += TORSO_EE_NAMES
        if self._config.use_right_arm:
            names += RIGHT_EE_NAMES
        if self._config.use_left_arm:
            names += LEFT_EE_NAMES
        return {name: float for name in names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (
                self._config.cameras[cam].height,
                self._config.cameras[cam].width,
                3,
            )
            for cam in self.cameras
        }

    @property
    def observation_features(self) -> dict[str, Any]:
        features: dict[str, Any] = {**self._obs_ft, **self._gripper_ft}
        if self._config.use_velocity:
            for name in self._obs_ft:
                features[f"{name}.vel"] = float
        if self._config.use_torque:
            for name in self._obs_ft:
                features[f"{name}.torque"] = float
        features.update(self._ee_obs_ft)
        features.update(self._cameras_ft)
        return features

    @property
    def action_features(self) -> dict[str, Any]:
        return {**self._motors_ft, **self._gripper_ft, **self._base_ft}

    # ------------------------------------------------------------------ #
    #  Connection                                                          #
    # ------------------------------------------------------------------ #

    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        try:
            import rby1_sdk as rby
        except ImportError as e:
            raise ImportError("rby1_sdk is required. Install it from the RB-Y1 SDK.") from e

        logger.info(f"Connecting to RB-Y1 at {self._config.address} ...")

        # 0. Resolve the model and firmware version. When either is "auto" this
        #    probes the robot via rby1-sdk; the version selects the ready pose.
        self._resolved_model, self._resolved_version = model_probe.resolve_model_version(
            self._config.model,
            self._config.version,
            self._config.address,
            self._config.connect_timeout_sec,
        )
        (
            self._ready_body,
            self._ready_right,
            self._ready_left,
            self._ready_head,
        ) = ready_pose_for_version(self._resolved_version)
        logger.info(
            f"Using RB-Y1 model='{self._resolved_model}' "
            f"version='{self._resolved_version}'."
        )

        # 1. Create and connect.
        self._robot = rby.create_robot(self._config.address, self._resolved_model)
        if not self._robot.connect():
            raise ConnectionError(
                f"Failed to connect to RB-Y1 at {self._config.address}."
            )

        # 2. Power and servo on.
        if not self._robot.is_power_on(".*"):
            if not self._robot.power_on(".*"):
                raise RuntimeError("Failed to power on RB-Y1 actuators.")
        if not self._robot.is_servo_on(".*"):
            if not self._robot.servo_on(".*"):
                raise RuntimeError("Failed to servo on RB-Y1 motors.")

        # 3. Clear faults and enable the control manager.
        cm_state = self._robot.get_control_manager_state()
        if cm_state.state in (
            rby.ControlManagerState.State.MajorFault,
            rby.ControlManagerState.State.MinorFault,
        ):
            logger.warning("Clearing RB-Y1 control manager fault ...")
            self._robot.reset_fault_control_manager()
        # EE mode runs the Cartesian impedance solver at teleop speeds and
        # needs the unlimited mode (as the VR teleoperator previously used).
        self._robot.enable_control_manager(
            unlimited_mode_enabled=(self._config.action_mode == "ee")
        )

        # 4. Cache the model (joint index arrays) and URDF velocity / acceleration
        #    limits from the dynamics model.
        self._model = self._robot.model()
        dyn_model = self._robot.get_dynamics()
        dyn_state = dyn_model.make_state([], self._model.robot_joint_names)
        self._max_qdot = dyn_model.get_limit_qdot_upper(dyn_state)
        self._max_qddot = dyn_model.get_limit_qddot_upper(dyn_state)
        # The wrist (last arm joint) needs extra velocity headroom in impedance mode.
        if self._config.use_impedance:
            scale = self._config.wrist_velocity_limit_scale
            self._max_qdot[self._model.right_arm_idx[-1]] *= scale
            self._max_qdot[self._model.left_arm_idx[-1]] *= scale
        logger.info(
            f"URDF limits loaded: max_qdot range=[{self._max_qdot.min():.2f}, "
            f"{self._max_qdot.max():.2f}] rad/s, max_qddot range=["
            f"{self._max_qddot.min():.2f}, {self._max_qddot.max():.2f}] rad/s^2 "
            f"(acc scale={self._config.acceleration_limit_scale}x)"
        )

        # 4b. In EE mode, prepare a forward-kinematics state so the observation
        #     can report the end-effector pose of each group.
        if self._config.action_mode == "ee":
            self._dyn_robot = dyn_model
            self._fk_state = dyn_model.make_state(
                _FK_LINKS, self._model.robot_joint_names
            )

        # 5. Open the command stream (priority=1). This stream owns the arms,
        #    the gripper and the mobile base for both teleoperation (the
        #    teleoperator only produces actions) and replay / policy inference.
        self._stream = self._robot.create_command_stream(priority=1)
        self._last_ee_targets = None

        # 6. Power the tool flanges and home the Dynamixel grippers.
        if self._config.use_gripper:
            self._robot.set_tool_flange_output_voltage("right", 12)
            self._robot.set_tool_flange_output_voltage("left", 12)
            time.sleep(0.5)  # allow the capacitors to stabilise
            self._gripper = Rby1Gripper()
            self._gripper.connect()
        else:
            logger.info(
                "Gripper disabled (use_gripper=False) - skipping flange power and gripper init."
            )

        # 7. Connect cameras.
        for cam in self.cameras.values():
            cam.connect()

        self._is_connected = True
        self.configure()
        logger.info(f"{self} connected.")

        # 8. Move to the ready pose before inference begins.
        if self._config.move_to_ready_on_connect:
            self.move_to_ready_pose()

    def disconnect(self) -> None:
        if not self.is_connected:
            return

        if self._gripper is not None:
            self._gripper.disconnect()
            self._gripper = None

        for cam in self.cameras.values():
            cam.disconnect()

        if self._robot is not None:
            self._robot.disable_control_manager()
            if self._config.use_gripper:
                self._robot.set_tool_flange_output_voltage("right", 0)
                self._robot.set_tool_flange_output_voltage("left", 0)
            self._robot.disconnect()
            self._robot = None

        self._model = None
        self._stream = None
        self._max_qdot = None
        self._max_qddot = None
        self._is_connected = False
        self._action_start_time = None
        self._last_ee_targets = None
        self._dyn_robot = None
        self._fk_state = None
        self._resolved_model = None
        self._resolved_version = None
        logger.info(f"{self} disconnected.")

    # ------------------------------------------------------------------ #
    #  Calibration / configuration                                         #
    # ------------------------------------------------------------------ #

    def calibrate(self) -> None:
        # RB-Y1 uses absolute encoders; no software calibration needed.
        pass

    def configure(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        # Low-pass filter on joint position commands to reduce jitter.
        self._robot.set_parameter("joint_position_command.cutoff_frequency", "5")
        logger.info("RB-Y1 configured: cutoff_frequency=5 Hz.")

    def reset(self) -> None:
        """Return the enabled arms to the ready pose between record episodes.

        Called by ``lerobot-record``; mirrors ``unitree_g1.reset()``.  Arms whose
        ``reset_*_arm_on_record`` flag is False keep their current position.
        """
        if (
            not self._config.reset_right_arm_on_record
            and not self._config.reset_left_arm_on_record
        ):
            logger.info("Record reset disabled for both arms; keeping current robot pose.")
            return

        body_position, head_position = self._build_record_reset_targets()
        self.move_to_ready_pose(body_position=body_position, head_position=head_position)

    def move_to_ready_pose(
        self,
        minimum_time: float = 5.0,
        hold_time: float = 1.0,
        body_position: np.ndarray | None = None,
        head_position: np.ndarray | None = None,
    ) -> None:
        """Move the robot to a joint-position pose (the ready pose by default).

        Called automatically at the end of :meth:`connect` when
        ``config.move_to_ready_on_connect`` is True, so the robot is in a safe,
        known configuration before inference begins.

        Parameters
        ----------
        minimum_time : float
            Minimum trajectory duration in seconds.
        hold_time : float
            Control hold time in seconds.
        body_position, head_position : np.ndarray | None
            Override targets; default to the constant ready pose / head pitch.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        try:
            import rby1_sdk as rby
        except ImportError as e:
            raise ImportError("rby1_sdk is required.") from e

        logger.info("Moving to ready pose ...")
        try:
            ctrl_state = self._robot.get_control_manager_state().control_state
            if ctrl_state != rby.ControlManagerState.ControlState.Idle:
                self._robot.cancel_control()
                time.sleep(1.0)

            if not self._robot.wait_for_control_ready(1000):
                logger.error("wait_for_control_ready timed out - skipping ready pose motion.")
                return

            target_body = (
                self._ready_body if body_position is None
                else np.asarray(body_position, dtype=np.float64)
            )
            target_head = (
                self._ready_head if head_position is None
                else np.asarray(head_position, dtype=np.float64)
            )

            cbc = cb.build_ready_pose_command(
                rby, target_body, target_head, minimum_time, hold_time
            )
            self._robot.send_command(rby.RobotCommandBuilder().set_command(cbc)).get()
            logger.info("Ready pose reached.")

            # Re-create the command stream so inference can start immediately.
            # The next EE command must reset the impedance reference (the
            # robot likely moved away from the previously commanded target).
            self._stream = self._robot.create_command_stream(priority=1)
            self._last_ee_targets = None
        except Exception as exc:
            logger.error(f"move_to_ready_pose error: {exc}")

    def _build_record_reset_targets(self) -> tuple[np.ndarray, np.ndarray]:
        """Build (body, head) reset targets from the current state.

        Enabled arms are overwritten with the ready pose; disabled arms and the
        torso keep their measured positions.
        """
        state = self._robot.get_state()
        position = np.asarray(state.position, dtype=np.float64)
        model = self._model

        body_position = np.concatenate(
            [
                position[model.torso_idx],
                position[model.right_arm_idx],
                position[model.left_arm_idx],
            ]
        ).copy()
        head_position = np.asarray(position[model.head_idx], dtype=np.float64).copy()

        if self._config.reset_right_arm_on_record:
            body_position[_RIGHT_ARM_BODY_SLICE] = self._ready_right
        if self._config.reset_left_arm_on_record:
            body_position[_LEFT_ARM_BODY_SLICE] = self._ready_left

        return body_position, head_position

    # ------------------------------------------------------------------ #
    #  Observation                                                         #
    # ------------------------------------------------------------------ #

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        state = self._robot.get_state()
        model = self._model
        obs: dict[str, Any] = {}

        # Joint positions (and optional velocities / torques) per enabled group.
        self._read_group(obs, state.position, model, "")
        if self._config.use_velocity:
            self._read_group(obs, state.velocity, model, ".vel")
        if self._config.use_torque:
            self._read_group(obs, state.torque, model, ".torque")

        # End-effector poses (EE mode): forward kinematics of the enabled groups.
        self._read_ee_observation(obs, state)

        # Grippers. Dataset convention is 1.0 = open, 0.0 = closed; Rby1Gripper
        # reports 0 = open, 1 = closed, so both sides are flipped here.
        if self._config.use_gripper and self._gripper is not None:
            gripper_pos = self._gripper.get_positions()  # [right, left], 0=open 1=closed
            if self._config.use_right_arm:
                obs["right_gripper_0"] = 1.0 - float(gripper_pos[0])
            if self._config.use_left_arm:
                obs["left_gripper_0"] = 1.0 - float(gripper_pos[1])

        # Cameras.
        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.async_read()

        return obs

    def _read_group(
        self, obs: dict[str, Any], values: np.ndarray, model: Any, suffix: str
    ) -> None:
        """Copy the enabled torso / arm joints of ``values`` into ``obs``.

        ``suffix`` is appended to each key (``""`` for positions, ``".vel"`` /
        ``".torque"`` for the optional channels).
        """
        if self._config.use_torso:
            torso = values[model.torso_idx]
            for i, name in enumerate(TORSO_NAMES):
                obs[f"{name}{suffix}"] = float(torso[i])
        if self._config.use_right_arm:
            right = values[model.right_arm_idx]
            for i, name in enumerate(RIGHT_ARM_NAMES):
                obs[f"{name}{suffix}"] = float(right[i])
        if self._config.use_left_arm:
            left = values[model.left_arm_idx]
            for i, name in enumerate(LEFT_ARM_NAMES):
                obs[f"{name}{suffix}"] = float(left[i])

    def _read_ee_observation(self, obs: dict[str, Any], state: Any) -> None:
        """Add the end-effector pose of each enabled group to ``obs``.

        EE mode only: runs forward kinematics on the measured joint positions
        and encodes each pose as ``<group>_ee.{x,y,z,wx,wy,wz}`` (the same
        rotvec convention as the EE action).
        """
        if self._config.action_mode != "ee" or self._fk_state is None:
            return
        self._fk_state.set_q(np.asarray(state.position, dtype=np.float64).copy())
        self._dyn_robot.compute_forward_kinematics(self._fk_state)
        if self._config.use_torso:
            torso_T = self._dyn_robot.compute_transformation(
                self._fk_state, _IDX_BASE, _IDX_TORSO_5
            )
            obs.update(cb.action_from_pose(torso_T, "torso_ee"))
        if self._config.use_right_arm:
            right_T = self._dyn_robot.compute_transformation(
                self._fk_state, _IDX_BASE, _IDX_RIGHT_ARM_6
            )
            obs.update(cb.action_from_pose(right_T, "right_ee"))
        if self._config.use_left_arm:
            left_T = self._dyn_robot.compute_transformation(
                self._fk_state, _IDX_BASE, _IDX_LEFT_ARM_6
            )
            obs.update(cb.action_from_pose(left_T, "left_ee"))

    # ------------------------------------------------------------------ #
    #  Action                                                              #
    # ------------------------------------------------------------------ #

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        try:
            import rby1_sdk as rby
        except ImportError as e:
            raise ImportError("rby1_sdk is required.") from e

        if self._config.action_mode == "ee":
            self._send_ee_action(rby, action)
        else:
            self._send_joint_action(rby, action)
        self._send_gripper_action(action)
        return action

    def _send_joint_action(self, rby: Any, action: dict[str, Any]) -> None:
        """Execute a joint-position action (``action_mode="joint"``)."""
        # Start the startup ramp on the first command (after connect() and the
        # ready-pose motion are complete).
        now = time.monotonic()
        if self._action_start_time is None:
            self._action_start_time = now
            logger.info(
                f"Startup ramp started: minimum_time decreases from "
                f"{self._config.startup_min_time:.2f}s -> "
                f"{self._config.normal_min_time:.2f}s over "
                f"{self._config.startup_ramp_duration:.1f}s."
            )
        minimum_time = self._ramp_minimum_time(now)

        if self._config.use_impedance:
            logger.debug(
                f"[impedance] stiffness={self._config.impedance_stiffness}, "
                f"torque_limit={self._config.impedance_torque_limit}, "
                f"damping={self._config.impedance_damping_ratio}"
            )

        # Compose the per-tick command: enabled arms (held limbs are omitted)
        # plus the optional mobile-base velocity. Both ride the same priority-1
        # stream so they are applied atomically.
        body, has_body = cb.build_body_command(
            rby,
            self._config,
            action,
            self._model,
            self._max_qdot,
            self._max_qddot,
            minimum_time,
        )
        cbc = rby.ComponentBasedCommandBuilder()
        if has_body:
            cbc.set_body_command(body)
        if self._config.use_mobile_base:
            linear, angular = self._base_velocity_from_action(action)
            cbc.set_mobility_command(
                cb.build_mobility_command(rby, linear, angular, minimum_time)
            )
        self._stream.send_command(
            rby.RobotCommandBuilder().set_command(cbc)
        )

    def _send_ee_action(self, rby: Any, action: dict[str, Any]) -> None:
        """Execute an end-effector pose action (``action_mode="ee"``).

        Decodes the ``<group>_ee.*`` keys into SE3 targets and sends a
        Cartesian impedance command. When a target jumps further than the
        configured thresholds from the previous one (or this is the first
        command after connect / a ready-pose motion), ``set_reset_reference``
        re-references the solver to the robot's actual state.
        """
        cfg = self._config
        targets = cb.CartesianTargets(
            torso=cb.pose_from_action(action, "torso_ee")
            if cfg.use_torso else np.eye(4),
            right_arm=cb.pose_from_action(action, "right_ee")
            if cfg.use_right_arm else np.eye(4),
            left_arm=cb.pose_from_action(action, "left_ee")
            if cfg.use_left_arm else np.eye(4),
        )
        reset = self._detect_ee_reset(targets)
        if reset.any:
            logger.info(
                f"EE reference reset (torso={reset.torso}, "
                f"right={reset.right_arm}, left={reset.left_arm})."
            )

        if cfg.ee_whole_body:
            body = cb.build_whole_body_command(rby, cfg, targets, reset)
        else:
            body = cb.build_per_component_command(
                rby, cfg, targets, reset, self._null_right, self._null_left
            )
        cbc = rby.ComponentBasedCommandBuilder().set_body_command(body)
        if cfg.use_mobile_base:
            linear, angular = self._base_velocity_from_action(action)
            cbc.set_mobility_command(
                cb.build_mobility_command(
                    rby, linear, angular, cfg.ee_dt * cfg.min_time_factor_wb
                )
            )
        self._stream.send_command(
            rby.RobotCommandBuilder().set_command(cbc)
        )
        self._last_ee_targets = targets

    def _detect_ee_reset(self, targets: cb.CartesianTargets) -> cb.ResetFlags:
        """Flag components whose target jumped since the last command."""
        cfg = self._config
        last = self._last_ee_targets
        if last is None:
            return cb.ResetFlags(
                torso=cfg.use_torso,
                right_arm=cfg.use_right_arm,
                left_arm=cfg.use_left_arm,
            )
        return cb.ResetFlags(
            torso=cfg.use_torso and self._target_jumped(last.torso, targets.torso),
            right_arm=cfg.use_right_arm
            and self._target_jumped(last.right_arm, targets.right_arm),
            left_arm=cfg.use_left_arm
            and self._target_jumped(last.left_arm, targets.left_arm),
        )

    def _target_jumped(self, prev: np.ndarray, new: np.ndarray) -> bool:
        """True when two SE3 targets differ beyond the reset thresholds."""
        cfg = self._config
        dpos = float(np.linalg.norm(new[:3, 3] - prev[:3, 3]))
        dR = prev[:3, :3].T @ new[:3, :3]
        angle = float(np.arccos(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)))
        return (
            dpos > cfg.ee_reset_position_threshold
            or angle > cfg.ee_reset_rotation_threshold
        )

    def _send_gripper_action(self, action: dict[str, Any]) -> None:
        """Drive the Dynamixel grippers from the action dict.

        Action uses the dataset convention (1=open, 0=closed);
        Rby1Gripper.set_positions() expects the hardware convention
        (0=open, 1=closed).
        """
        if not self._config.use_gripper or self._gripper is None:
            return
        current_gripper = self._gripper.get_positions()
        gripper_target = np.array(
            [
                1.0 - action.get("right_gripper_0", 1.0)
                if self._config.use_right_arm
                else current_gripper[0],
                1.0 - action.get("left_gripper_0", 1.0)
                if self._config.use_left_arm
                else current_gripper[1],
            ]
        )
        self._gripper.set_positions(gripper_target)

    @staticmethod
    def _base_velocity_from_action(action: dict[str, Any]) -> tuple[np.ndarray, float]:
        """Extract the SE2 base command (linear xy, yaw rate) from an action."""
        linear = np.array(
            [action.get("x.vel", 0.0), action.get("y.vel", 0.0)], dtype=np.float64
        )
        angular = float(action.get("theta.vel", 0.0))
        return linear, angular


    def _ramp_minimum_time(self, now: float) -> float:
        """Interpolate ``minimum_time`` from the startup value down to normal.

        A large per-command ``minimum_time`` slows the robot at the start of a
        session so it converges smoothly to the first target instead of snapping
        to it.  Returns ``normal_min_time`` immediately when the ramp is disabled.
        """
        cfg = self._config
        if cfg.startup_ramp_duration <= 0.0:
            return cfg.normal_min_time
        elapsed = now - self._action_start_time
        progress = min(elapsed / cfg.startup_ramp_duration, 1.0)
        minimum_time = cfg.startup_min_time + progress * (
            cfg.normal_min_time - cfg.startup_min_time
        )
        logger.debug(
            f"[ramp] elapsed={elapsed:.2f}s progress={progress:.3f} "
            f"min_time={minimum_time:.3f}s"
        )
        return minimum_time
