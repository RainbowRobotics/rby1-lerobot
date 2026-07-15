"""LeRobot Robot interface for the Rainbow Robotics RB-Series cobots.

Implements the :class:`lerobot.robots.robot.Robot` abstract class for the
RB-Series collaborative arms (RB3/RB5/RB10) through the ``rbpodo`` client
SDK, so the arms can be used with the standard LeRobot tools (data
collection, teleoperation replay, imitation-learning inference).

The control box exposes two TCP channels used by this class:

* command channel (``rbpodo.Cobot``, port 5000) — motion commands + ACKs.
* data channel (``rbpodo.CobotData``, port 5001) — single-shot state polls.

Unit conventions
----------------
rbpodo talks degrees; the LeRobot dataset boundary of this repository is
radians (matching ``lerobot_robot_rby1``), with joint keys ``joint_0`` ..
``joint_5`` (base to wrist).  All conversion happens inside this class.
The gripper channel ``gripper_0`` is normalised with 1.0 = open (dataset
convention; the hardware drivers use 0.0 = open, see ``gripper.py``).

Motion model
------------
``send_action()`` streams joint-position targets through ``move_servo_j``.
Streaming requires ACK-waiting to be disabled on the command channel, while
the blocking PTP moves (``move_j`` in ``move_to_ready_pose()`` / ``reset()``)
require it enabled; ``_enter_streaming()`` / ``_exit_streaming()`` keep the
two modes from ever mixing.  When streaming stops the robot simply holds the
last commanded target, which the per-step delta clamp keeps within
``max_joint_delta_deg`` of the measured position.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_rb import RbCobotConfig
from .gripper import RbGripperBase, make_gripper
from .models import DOF, GRIPPER_NAME, JOINT_NAMES, MODEL_SPECS

logger = logging.getLogger(__name__)

# Flush the command-channel response collector every N streamed commands
# (ACK-less streaming still queues event messages on the collector).
_RC_CLEAR_PERIOD = 100


def clamp_target(
    target_deg: np.ndarray,
    last_deg: np.ndarray,
    limits_deg: np.ndarray,
    max_delta_deg: float,
) -> np.ndarray:
    """Clamp a streamed joint target for safety.

    The target is first clipped to the joint limits, then the change from
    the previously commanded target is limited to ``max_delta_deg`` per
    call, so a large jump in the action stream ramps in over several steps
    instead of being executed at full speed.

    Pure function (unit-testable without rbpodo).

    Parameters
    ----------
    target_deg : np.ndarray, shape (DOF,)
        Requested joint target, degrees.
    last_deg : np.ndarray, shape (DOF,)
        Previously commanded joint target, degrees.
    limits_deg : np.ndarray, shape (DOF, 2)
        Per-joint (lower, upper) limits, degrees.
    max_delta_deg : float
        Maximum per-joint change from ``last_deg``, degrees.
    """
    target = np.clip(target_deg, limits_deg[:, 0], limits_deg[:, 1])
    delta = np.clip(target - last_deg, -max_delta_deg, max_delta_deg)
    return last_deg + delta


class RbCobot(Robot):
    """LeRobot Robot implementation shared by the RB-Series cobots.

    Do not instantiate through the CLI directly — use a registered
    per-model shim such as :class:`~lerobot_robot_rb.rb10.Rb10`
    (``--robot.type=rb10``).

    Observation features
    --------------------
    * Joint positions (radians): ``joint_0`` .. ``joint_5``.
    * Optional ``<joint>.vel`` (finite-differenced) and ``<joint>.current``
      (A) channels when ``use_velocity`` / ``use_current`` are set.
    * Gripper position (normalised, 1.0 = open) when a gripper is configured.
    * One ``(H, W, 3)`` image per configured camera.

    Action features
    ---------------
    Joint positions (radians) plus the normalised gripper when configured.

    Example
    -------
    >>> cfg = Rb10Config(ip="10.0.2.7", operation_mode="simulation")
    >>> robot = Rb10(cfg)
    >>> robot.connect()
    >>> obs = robot.get_observation()
    >>> robot.send_action(obs)   # replicate the current pose
    >>> robot.disconnect()
    """

    config_class = RbCobotConfig
    name = "rb_cobot"

    def __init__(self, config: RbCobotConfig) -> None:
        super().__init__(config)
        self._config = config
        self._spec = MODEL_SPECS[config.model]
        self._robot = None  # rbpodo.Cobot (command channel)
        self._rc = None     # rbpodo.ResponseCollector
        self._data = None   # rbpodo.CobotData (data channel)
        self._gripper: RbGripperBase | None = None
        self._is_connected: bool = False
        self.cameras = make_cameras_from_configs(config.cameras)

        # Streaming state machine (see module docstring).
        self._streaming: bool = False
        self._last_cmd_deg: np.ndarray | None = None
        self._tick: int = 0

        # Previous poll used to finite-difference "<joint>.vel".
        self._prev_q_rad: np.ndarray | None = None
        self._prev_time: float | None = None

        # Effective per-model safety constants (config overrides spec).
        self._joint_limits_deg = np.asarray(
            config.joint_limits_deg
            if config.joint_limits_deg is not None
            else self._spec.joint_limits_deg,
            dtype=np.float64,
        )
        self._ready_pose_deg = np.asarray(
            config.ready_pose_deg
            if config.ready_pose_deg is not None
            else self._spec.default_ready_pose_deg,
            dtype=np.float64,
        )

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
        # RB-Series arms use absolute encoders - no calibration required.
        return True

    # ------------------------------------------------------------------ #
    #  Features                                                            #
    # ------------------------------------------------------------------ #

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {name: float for name in JOINT_NAMES}

    @property
    def _gripper_ft(self) -> dict[str, type]:
        if self._config.gripper_type == "none":
            return {}
        return {GRIPPER_NAME: float}

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
        features: dict[str, Any] = {**self._motors_ft, **self._gripper_ft}
        if self._config.use_velocity:
            for name in JOINT_NAMES:
                features[f"{name}.vel"] = float
        if self._config.use_current:
            for name in JOINT_NAMES:
                features[f"{name}.current"] = float
        features.update(self._cameras_ft)
        return features

    @property
    def action_features(self) -> dict[str, Any]:
        return {**self._motors_ft, **self._gripper_ft}

    # ------------------------------------------------------------------ #
    #  Connection                                                          #
    # ------------------------------------------------------------------ #

    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        try:
            import rbpodo as rb
        except ImportError as e:
            raise ImportError(
                "rbpodo is required. Install it with `pip install rbpodo`."
            ) from e

        cfg = self._config
        logger.info(f"Connecting to {cfg.model} at {cfg.ip} ...")

        # 1. Command channel + response collector, data channel.
        self._robot = rb.Cobot(cfg.ip, cfg.command_port)
        self._rc = rb.ResponseCollector()
        self._data = rb.CobotData(cfg.ip, cfg.data_port)

        # 2. Operation mode (simulation = control-box simulator, no motion)
        #    and the global speed override.
        mode = (
            rb.OperationMode.Real
            if cfg.operation_mode == "real"
            else rb.OperationMode.Simulation
        )
        self._robot.set_operation_mode(self._rc, mode)
        self._robot.set_speed_bar(self._rc, cfg.speed_bar)
        self._rc.error().throw_if_not_empty()
        self._rc.clear()

        # 3. Verify the data channel with one poll.
        if self._data.request_data(cfg.data_timeout_sec) is None:
            raise ConnectionError(
                f"No state data from {cfg.ip}:{cfg.data_port} within "
                f"{cfg.data_timeout_sec}s - check the robot's data channel."
            )

        # 4. Gripper (config-selected; None when gripper_type="none").
        self._gripper = make_gripper(cfg)
        if self._gripper is not None:
            self._gripper.connect()

        # 5. Cameras.
        for cam in self.cameras.values():
            cam.connect()

        self._is_connected = True
        self.configure()
        logger.info(f"{self} connected (operation_mode={cfg.operation_mode}).")

        # 6. Optionally move to the ready pose before inference begins.
        if cfg.move_to_ready_on_connect:
            self.move_to_ready_pose()

    def disconnect(self) -> None:
        if not self.is_connected:
            return

        # Stop streaming; move_servo_j holds its last target, which the
        # delta clamp keeps close to the measured position - no halt needed.
        self._exit_streaming()

        if self._gripper is not None:
            self._gripper.disconnect()
            self._gripper = None

        for cam in self.cameras.values():
            cam.disconnect()

        # rbpodo has no explicit close; the sockets close with the objects.
        self._robot = None
        self._rc = None
        self._data = None
        self._prev_q_rad = None
        self._prev_time = None
        self._tick = 0
        self._is_connected = False
        logger.info(f"{self} disconnected.")

    # ------------------------------------------------------------------ #
    #  Calibration / configuration                                         #
    # ------------------------------------------------------------------ #

    def calibrate(self) -> None:
        # RB-Series arms use absolute encoders; no software calibration needed.
        pass

    def configure(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        logger.info(
            f"{self._config.model} configured: speed_bar={self._config.speed_bar}, "
            f"max_joint_delta_deg={self._config.max_joint_delta_deg}."
        )

    def reset(self) -> None:
        """Return to the ready pose between record episodes.

        Called by ``lerobot-record``; disabled unless ``reset_on_record``
        is set (the robot then keeps its current pose).
        """
        if not self._config.reset_on_record:
            logger.info("Record reset disabled; keeping current robot pose.")
            return
        self.move_to_ready_pose()

    def move_to_ready_pose(self, start_timeout_sec: float = 5.0) -> None:
        """Blocking PTP move (``move_j``) to the configured ready pose.

        Exits streaming mode first so the move waits on ACKs.  Errors are
        logged rather than raised so a failed reset does not kill a
        recording session (same policy as the RB-Y1 package).
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._exit_streaming()
        logger.info(f"Moving to ready pose {self._ready_pose_deg.tolist()} (deg) ...")
        try:
            self._robot.move_j(
                self._rc,
                self._ready_pose_deg,
                self._config.ptp_speed,
                self._config.ptp_acc,
            )
            if self._robot.wait_for_move_started(
                self._rc, start_timeout_sec
            ).is_success():
                self._robot.wait_for_move_finished(self._rc)
                logger.info("Ready pose reached.")
            else:
                logger.warning(
                    "Ready-pose move did not start within "
                    f"{start_timeout_sec}s - skipping."
                )
            self._rc.clear()
        except Exception as exc:
            logger.error(f"move_to_ready_pose error: {exc}")

    # ------------------------------------------------------------------ #
    #  Streaming state machine                                             #
    # ------------------------------------------------------------------ #

    def _enter_streaming(self) -> None:
        """Switch the command channel to ACK-less servo streaming.

        Seeds the delta-clamp reference from the robot's current reference
        position so the first commands ramp from where the robot actually
        is instead of jumping to the first action.
        """
        data = self._data.request_data(self._config.data_timeout_sec)
        if data is None:
            raise DeviceNotConnectedError(
                f"{self} data channel timed out while entering streaming mode."
            )
        self._last_cmd_deg = np.asarray(
            data.sdata.jnt_ref, dtype=np.float64
        )[:DOF].copy()
        self._rc.clear()
        self._robot.disable_waiting_ack(self._rc)
        self._streaming = True
        logger.info("Servo streaming started (ACK waiting disabled).")

    def _exit_streaming(self) -> None:
        """Re-enable ACK waiting after servo streaming (idempotent)."""
        if not self._streaming:
            return
        try:
            self._robot.enable_waiting_ack(self._rc)
        except Exception as exc:
            logger.warning(f"enable_waiting_ack failed: {exc}")
        self._rc.clear()
        self._streaming = False
        self._last_cmd_deg = None
        logger.info("Servo streaming stopped (ACK waiting re-enabled).")

    # ------------------------------------------------------------------ #
    #  Observation                                                         #
    # ------------------------------------------------------------------ #

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        data = self._data.request_data(self._config.data_timeout_sec)
        if data is None:
            # One retry; the data channel is a single-shot poll.
            data = self._data.request_data(self._config.data_timeout_sec)
        if data is None:
            raise RuntimeError(
                f"{self} data channel timed out twice "
                f"({self._config.data_timeout_sec}s each)."
            )

        obs: dict[str, Any] = {}
        q_rad = np.deg2rad(
            np.asarray(data.sdata.jnt_ang, dtype=np.float64)[:DOF]
        )
        for i, name in enumerate(JOINT_NAMES):
            obs[name] = float(q_rad[i])

        if self._config.use_velocity:
            t = float(data.sdata.time)
            if self._prev_q_rad is None or self._prev_time is None or t <= self._prev_time:
                vel = np.zeros(DOF)
            else:
                vel = (q_rad - self._prev_q_rad) / (t - self._prev_time)
            self._prev_q_rad = q_rad
            self._prev_time = t
            for i, name in enumerate(JOINT_NAMES):
                obs[f"{name}.vel"] = float(vel[i])

        if self._config.use_current:
            cur = np.asarray(data.sdata.jnt_cur, dtype=np.float64)[:DOF]
            for i, name in enumerate(JOINT_NAMES):
                obs[f"{name}.current"] = float(cur[i])

        # Gripper. Dataset convention is 1.0 = open; the driver reports
        # 0.0 = open, so the value is flipped here.
        if self._gripper is not None:
            obs[GRIPPER_NAME] = 1.0 - self._gripper.get_position()

        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.async_read()

        return obs

    # ------------------------------------------------------------------ #
    #  Action                                                              #
    # ------------------------------------------------------------------ #

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if not self._streaming:
            self._enter_streaming()

        target_deg = np.rad2deg(
            np.array([float(action[name]) for name in JOINT_NAMES])
        )
        cmd_deg = clamp_target(
            target_deg,
            self._last_cmd_deg,
            self._joint_limits_deg,
            self._config.max_joint_delta_deg,
        )
        self._robot.move_servo_j(
            self._rc,
            cmd_deg,
            self._config.servo_t1,
            self._config.servo_t2,
            self._config.servo_gain,
            self._config.servo_alpha,
        )
        self._last_cmd_deg = cmd_deg

        # The collector still queues event messages while ACKs are disabled.
        self._tick += 1
        if self._tick % _RC_CLEAR_PERIOD == 0:
            self._rc.clear()

        # Return what was actually commanded (post-clamp), so datasets
        # record the executed action when a safety clamp engages.
        sent: dict[str, Any] = {
            name: float(v) for name, v in zip(JOINT_NAMES, np.deg2rad(cmd_deg))
        }

        if self._gripper is not None and GRIPPER_NAME in action:
            # Dataset 1.0 = open -> hardware 0.0 = open.
            hw_target = float(np.clip(1.0 - float(action[GRIPPER_NAME]), 0.0, 1.0))
            self._gripper.set_position(hw_target)
            sent[GRIPPER_NAME] = 1.0 - hw_target

        return sent
