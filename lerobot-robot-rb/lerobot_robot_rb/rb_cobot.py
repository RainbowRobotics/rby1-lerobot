"""LeRobot Robot adapter for Rainbow Robotics RB-Series cobots.

This implementation wraps the hardware-tested TCP Cobot client rather than
the rbpodo-based streaming worker.

Control flow
------------
LeRobot action boundary:
    joint_0 .. joint_5 in radians

RB control-box boundary:
    move_servo_j joint targets in degrees

One call to ``send_action()`` produces at most one ``ServoJ()`` command.
There is no additional 200 Hz worker, automatic interpolation, automatic
operation-mode change, or automatic ready-pose motion.

Servo command gate
------------------
The command gate is disabled when the robot connects. The VR Teleoperator
must explicitly enable it after its A-button initialisation and Grip state
permit motion.

While the gate is disabled, ``send_action()`` validates and returns the
action but does not transmit ``move_servo_j`` to the control box.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots.robot import Robot
from lerobot.utils.errors import (
    DeviceAlreadyConnectedError,
    DeviceNotConnectedError,
)

from .cobot import Cobot, systemSTAT
from .config_rb import RbCobotConfig
from .gripper import RbGripperBase, make_gripper
from .models import DOF, GRIPPER_NAME, JOINT_NAMES, MODEL_SPECS


logger = logging.getLogger(__name__)


# The standard LeRobot CLI connects the Robot before the Teleoperator.
# The VR Teleoperator uses this reference to:
#
# - read the current measured joints
# - enable/disable ServoJ transmission
#
# It does not open a second control-box connection.
_ACTIVE_RB_COBOT = None


def _set_active_rb_cobot(robot: RbCobot | None) -> None:
    global _ACTIVE_RB_COBOT
    _ACTIVE_RB_COBOT = robot


def get_active_rb_cobot() -> RbCobot | None:
    """Return the connected RB robot in this Python process."""

    robot = _ACTIVE_RB_COBOT

    if robot is None or not robot.is_connected:
        return None

    return robot


# ---------------------------------------------------------------------------
# Compatibility helpers
# ---------------------------------------------------------------------------
#
# These functions remain available because lerobot_robot_rb.__init__ and
# existing tests import them. The new Robot adapter does not use them in its
# command path.


def clamp_target(
    target_deg: np.ndarray,
    last_deg: np.ndarray,
    limits_deg: np.ndarray,
    max_delta_deg: float,
) -> np.ndarray:
    """Clip a joint target to limits and a maximum per-call change."""

    target = np.asarray(target_deg, dtype=np.float64)
    last = np.asarray(last_deg, dtype=np.float64)
    limits = np.asarray(limits_deg, dtype=np.float64)

    clipped = np.clip(
        target,
        limits[:, 0],
        limits[:, 1],
    )
    delta = np.clip(
        clipped - last,
        -float(max_delta_deg),
        float(max_delta_deg),
    )
    return last + delta


def wrap_deg(angles_deg: np.ndarray) -> np.ndarray:
    """Map degree angles into the interval [-180, 180]."""

    angles = np.asarray(angles_deg, dtype=np.float64)
    return angles - 360.0 * np.round(angles / 360.0)


def clamp_ee_target(
    target: np.ndarray,
    last: np.ndarray,
    max_pos_delta_mm: float,
    max_rot_delta_deg: float,
    bounds_mm: np.ndarray | None = None,
) -> np.ndarray:
    """Compatibility helper for the old Cartesian command implementation."""

    target = np.asarray(target, dtype=np.float64).copy()
    last = np.asarray(last, dtype=np.float64)

    if bounds_mm is not None:
        bounds = np.asarray(bounds_mm, dtype=np.float64)
        target[:3] = np.clip(
            target[:3],
            bounds[:, 0],
            bounds[:, 1],
        )

    result = last.copy()

    result[:3] = last[:3] + np.clip(
        target[:3] - last[:3],
        -float(max_pos_delta_mm),
        float(max_pos_delta_mm),
    )

    result[3:] = last[3:] + np.clip(
        wrap_deg(target[3:] - last[3:]),
        -float(max_rot_delta_deg),
        float(max_rot_delta_deg),
    )

    return result


def kinematics_estop_only(system_state: Any) -> bool:
    """Check whether only the kinematics EMS flag is active.

    This compatibility helper uses fields available in the original
    ``systemSTAT`` packet. It is not used automatically by the new Robot
    adapter.
    """

    if not int(system_state.op_stat_ems_flag) & 0x3F:
        return False

    if int(system_state.op_stat_collision_occur) & 0x3:
        return False

    if int(system_state.op_stat_self_collision) & 0x3:
        return False

    if int(system_state.op_stat_soft_estop_occur) & 0x3:
        return False

    if int(system_state.op_stat_sos_flag) & 0x3F:
        return False

    if int(system_state.is_freedrive_mode) & 0x3:
        return False

    if int(system_state.program_mode) != 0:
        return False

    if any(
        int(value) & 0xFF00
        for value in system_state.jnt_info
    ):
        return False

    if int(system_state.init_error) & 0xFFF:
        return False

    if int(system_state.init_state_info) != 6:
        return False

    return True


class RbCobot(Robot):
    """LeRobot Robot implementation shared by RB-Series cobots.

    Do not register this base class directly with the CLI. Use a model shim
    such as ``Rb10`` registered as ``--robot.type=rb10``.

    Observation features
    --------------------
    - ``joint_0`` .. ``joint_5`` in radians
    - optional ``gripper_0`` with 1.0 = open
    - configured camera images

    Action features
    ---------------
    - ``joint_0`` .. ``joint_5`` in radians
    - optional ``gripper_0`` with 1.0 = open
    """

    config_class = RbCobotConfig
    name = "rb_cobot"

    def __init__(self, config: RbCobotConfig) -> None:
        super().__init__(config)

        self._config = config
        self._spec = MODEL_SPECS[config.model]

        self._cobot: Cobot | None = None

        # Compatibility alias for older internal code that accessed
        # ``robot._robot``.
        self._robot: Cobot | None = None

        self._gripper: RbGripperBase | None = None
        self._is_connected = False

        self.cameras = make_cameras_from_configs(
            config.cameras
        )

        # Arm motion transmission is disabled until the Teleoperator enables
        # it after A-button initialisation and Grip activation.
        self._servo_enabled = False

        # Last requested LeRobot joint action, radians.
        self._last_action_rad: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @is_connected.setter
    def is_connected(self, value: bool) -> None:
        self._is_connected = bool(value)

    @property
    def is_calibrated(self) -> bool:
        # RB-Series robots use absolute encoders.
        return True

    @property
    def servo_enabled(self) -> bool:
        """Whether send_action() may transmit ServoJ commands."""

        return self._servo_enabled

    @property
    def low_level_cobot(self) -> Cobot:
        """Return the connected low-level TCP Cobot client."""

        if not self.is_connected or self._cobot is None:
            raise DeviceNotConnectedError(
                f"{self} is not connected."
            )

        return self._cobot

    # ------------------------------------------------------------------
    # LeRobot feature definitions
    # ------------------------------------------------------------------

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            name: float
            for name in JOINT_NAMES
        }

    @property
    def _gripper_ft(self) -> dict[str, type]:
        if self._config.gripper_type == "none":
            return {}

        return {
            GRIPPER_NAME: float,
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple[int, int, int]]:
        return {
            camera_name: (
                self._config.cameras[camera_name].height,
                self._config.cameras[camera_name].width,
                3,
            )
            for camera_name in self.cameras
        }

    @property
    def observation_features(self) -> dict[str, Any]:
        features: dict[str, Any] = {
            **self._motors_ft,
            **self._gripper_ft,
        }
        features.update(self._cameras_ft)
        return features

    @property
    def action_features(self) -> dict[str, Any]:
        return {
            **self._motors_ft,
            **self._gripper_ft,
        }

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(
        self,
        calibrate: bool = True,  # noqa: ARG002
    ) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                f"{self} is already connected."
            )

        cfg = self._config

        logger.info(
            "Connecting to %s at %s:%d/%d ...",
            cfg.model,
            cfg.ip,
            cfg.command_port,
            cfg.data_port,
        )

        cobot = Cobot(
            ip=cfg.ip,
            command_port=cfg.command_port,
            data_port=cfg.data_port,
            socket_timeout_s=cfg.socket_timeout_s,
            data_request_hz=cfg.data_request_hz,
        )

        try:
            if not cobot.ConnectToCB():
                raise ConnectionError(
                    "Failed to connect to RB control box at "
                    f"{cfg.ip}:{cfg.command_port}/{cfg.data_port}."
                )

            if not cobot.wait_for_first_state(
                cfg.first_state_timeout_s
            ):
                raise TimeoutError(
                    "Connected to the RB control box, but no valid "
                    f"state packet arrived within "
                    f"{cfg.first_state_timeout_s:.2f}s."
                )

            self._cobot = cobot
            self._robot = cobot

            if cfg.set_speed_bar_on_connect:
                cobot.SetBaseSpeed(cfg.speed_bar)
                logger.info(
                    "RB speed bar set to %.3f.",
                    cfg.speed_bar,
                )

            self._gripper = make_gripper(
                cfg,
                cobot,
            )
            if self._gripper is not None:
                self._gripper.connect()

            for camera in self.cameras.values():
                camera.connect()

            self._servo_enabled = False
            self._last_action_rad = None
            self._is_connected = True

            self.configure()
            _set_active_rb_cobot(self)

            state = cobot.GetLatestState(
                timeout_s=cfg.first_state_timeout_s
            )
            mode_name = (
                "real"
                if int(state.program_mode) == 0
                else "simulation"
            )

            logger.info(
                "%s connected "
                "(control-box mode=%s, ServoJ gate=OFF).",
                self,
                mode_name,
            )

        except Exception:
            self._cleanup_resources()
            raise

    def disconnect(self) -> None:
        if (
            not self.is_connected
            and self._cobot is None
            and self._gripper is None
        ):
            return

        self.disable_servo_commands()
        self._cleanup_resources()

        logger.info("%s disconnected.", self)

    def _cleanup_resources(self) -> None:
        if _ACTIVE_RB_COBOT is self:
            _set_active_rb_cobot(None)

        for camera in self.cameras.values():
            try:
                camera.disconnect()
            except Exception as exc:
                logger.warning(
                    "Camera disconnect failed: %s",
                    exc,
                )

        if self._gripper is not None:
            try:
                self._gripper.disconnect()
            except Exception as exc:
                logger.warning(
                    "Gripper disconnect failed: %s",
                    exc,
                )
            finally:
                self._gripper = None

        if self._cobot is not None:
            try:
                self._cobot.DisConnectToCB()
            except Exception as exc:
                logger.warning(
                    "RB socket disconnect failed: %s",
                    exc,
                )

        self._cobot = None
        self._robot = None
        self._servo_enabled = False
        self._last_action_rad = None
        self._is_connected = False

    # ------------------------------------------------------------------
    # Calibration / configuration / reset
    # ------------------------------------------------------------------

    def calibrate(self) -> None:
        # Absolute encoders; no LeRobot motor calibration is required.
        return

    def configure(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self} is not connected."
            )

        logger.info(
            "%s configured: control_rate=%.1f Hz, "
            "ServoJ(t1=%.4f, t2=%.4f, gain=%.3f, alpha=%.3f).",
            self._config.model,
            self._config.control_rate_hz,
            self._config.servo_t1,
            self._config.servo_t2,
            self._config.servo_gain,
            self._config.servo_alpha,
        )

    def reset(self) -> None:
        """Disable ServoJ transmission and retain the current pose.

        LeRobot recording may call reset() between episodes. This adapter
        deliberately performs no automatic ready-pose movement.
        """

        self.disable_servo_commands()
        self._last_action_rad = None

        logger.info(
            "RB reset: ServoJ gate disabled; "
            "current robot pose retained."
        )

    # ------------------------------------------------------------------
    # Servo command gate
    # ------------------------------------------------------------------

    def set_servo_enabled(self, enabled: bool) -> None:
        """Enable or disable transmission of arm ServoJ commands.

        The VR Teleoperator owns this state:

        - connect / B button / Grip release -> False
        - initialised A state + Grip active -> True
        """

        enabled = bool(enabled)

        if enabled and not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self} is not connected."
            )

        if self._servo_enabled == enabled:
            return

        self._servo_enabled = enabled

        logger.info(
            "RB ServoJ command gate %s.",
            "ENABLED" if enabled else "DISABLED",
        )

    def enable_servo_commands(self) -> None:
        self.set_servo_enabled(True)

    def disable_servo_commands(self) -> None:
        self.set_servo_enabled(False)

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    def get_system_state(self) -> systemSTAT:
        """Return the latest decoded RB control-box state."""

        return self.low_level_cobot.GetLatestState(
            timeout_s=self._config.first_state_timeout_s
        )

    @staticmethod
    def _extract_joint_deg(
        state: systemSTAT,
        *,
        measured: bool,
    ) -> np.ndarray:
        values = (
            state.jnt_ang
            if measured
            else state.jnt_ref
        )

        joint_deg = np.asarray(
            values,
            dtype=np.float64,
        )

        if joint_deg.shape[0] < DOF:
            source = "jnt_ang" if measured else "jnt_ref"
            raise RuntimeError(
                f"RB state field {source} contains "
                f"{joint_deg.shape[0]} values; expected at least {DOF}."
            )

        joint_deg = joint_deg[:DOF].copy()

        if not np.all(np.isfinite(joint_deg)):
            source = "jnt_ang" if measured else "jnt_ref"
            raise RuntimeError(
                f"RB state field {source} contains non-finite values: "
                f"{joint_deg.tolist()}."
            )

        return joint_deg

    def get_joint_positions(
        self,
        *,
        measured: bool = True,
    ) -> np.ndarray:
        """Return current joint positions in radians.

        Parameters
        ----------
        measured:
            True returns ``jnt_ang``. This is the source used by the original
            VR code when beginning the 5-second startup interpolation.

            False returns ``jnt_ref``.
        """

        state = self.get_system_state()
        joint_deg = self._extract_joint_deg(
            state,
            measured=measured,
        )
        return np.deg2rad(joint_deg)

    def get_reference_joint_positions(self) -> np.ndarray:
        """Return control-box joint references in radians."""

        return self.get_joint_positions(measured=False)

    def _observation_joint_positions(
        self,
        state: systemSTAT,
    ) -> np.ndarray:
        # The original control-box state reports:
        #
        #   program_mode == 0: real robot
        #   otherwise: simulation
        #
        # In simulation the measured angle may remain unchanged, so expose
        # jnt_ref. In real mode expose the physical jnt_ang measurement.
        measured = int(state.program_mode) == 0

        joint_deg = self._extract_joint_deg(
            state,
            measured=measured,
        )
        return np.deg2rad(joint_deg)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self} is not connected."
            )

        state = self.get_system_state()
        q_rad = self._observation_joint_positions(state)

        observation: dict[str, Any] = {
            name: float(q_rad[index])
            for index, name in enumerate(JOINT_NAMES)
        }

        # Dataset convention: 1.0 = open.
        # Gripper hardware convention: 0.0 = open.
        if self._gripper is not None:
            observation[GRIPPER_NAME] = (
                1.0 - self._gripper.get_position()
            )

        for camera_name, camera in self.cameras.items():
            observation[camera_name] = camera.async_read()

        return observation

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------

    def _joint_action_to_array(
        self,
        action: dict[str, Any],
    ) -> np.ndarray:
        missing = [
            name
            for name in JOINT_NAMES
            if name not in action
        ]

        if missing:
            raise KeyError(
                "RB joint action is missing keys: "
                + ", ".join(missing)
            )

        joint_rad = np.asarray(
            [
                float(action[name])
                for name in JOINT_NAMES
            ],
            dtype=np.float64,
        )

        if joint_rad.shape != (DOF,):
            raise ValueError(
                f"Expected RB joint action shape ({DOF},), "
                f"got {joint_rad.shape}."
            )

        if not np.all(np.isfinite(joint_rad)):
            raise ValueError(
                "RB joint action contains non-finite values: "
                f"{joint_rad.tolist()}."
            )

        return joint_rad

    def send_action(
        self,
        action: dict[str, Any],
    ) -> dict[str, Any]:
        """Send one joint ServoJ command when the command gate is enabled.

        ``lerobot-teleoperate --fps`` should match
        ``config.control_rate_hz``. With the default configuration this is
        35 Hz.
        """

        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self} is not connected."
            )

        joint_rad = self._joint_action_to_array(action)
        self._last_action_rad = joint_rad.copy()

        sent: dict[str, Any] = {
            name: float(joint_rad[index])
            for index, name in enumerate(JOINT_NAMES)
        }

        if self._servo_enabled:
            joint_deg = np.rad2deg(joint_rad)

            success = self.low_level_cobot.ServoJ(
                joints_deg=joint_deg,
                t1=self._config.servo_t1,
                t2=self._config.servo_t2,
                gain=self._config.servo_gain,
                alpha=self._config.servo_alpha,
            )

            if not success:
                raise RuntimeError(
                    "RB control box rejected the ServoJ command."
                )

        if self._gripper is not None and GRIPPER_NAME in action:
            # Dataset convention:
            #   1.0 = open
            #
            # Hardware driver convention:
            #   0.0 = open
            dataset_target = float(
                np.clip(
                    float(action[GRIPPER_NAME]),
                    0.0,
                    1.0,
                )
            )
            hardware_target = 1.0 - dataset_target

            self._gripper.set_position(hardware_target)
            sent[GRIPPER_NAME] = dataset_target

        return sent
