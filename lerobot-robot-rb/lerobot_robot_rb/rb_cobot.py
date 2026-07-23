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
rbpodo talks degrees / millimetres; the LeRobot dataset boundary of this
repository is radians / metres. Joint keys are ``joint_0`` .. ``joint_5``
(base to wrist); EE keys (``action_space="ee"``) are ``ee_x`` .. ``ee_rz``.
The gripper channel ``gripper_0`` is normalised with 1.0 = open (dataset
convention; the hardware drivers use 0.0 = open, see ``gripper.py``).

Motion model
------------
``send_action()`` updates a streamed target; a background worker linearly
interpolates from the previous target and refreshes ``move_servo_j`` (joint
mode) or ``move_servo_l`` (EE mode, control-box IK) every
``servo_command_period_s``. The robot therefore receives a continuous
reference instead of action-rate steps — stepped references combined with a
small ``servo_t1`` produce stop-and-go torque spikes that the control box
misreads as external collisions.

Streaming requires ACK-waiting to be disabled on the command channel, while
the blocking PTP moves (``move_j`` in ``move_to_ready_pose()`` / ``reset()``)
require it enabled; ``_enter_streaming()`` / ``_exit_streaming()`` keep the
two modes from ever mixing. The exit path sends a zero-speed command before
restoring ACKs.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_rb import RbCobotConfig
from .gripper import RbGripperBase, make_gripper
from .models import DOF, EE_NAMES, GRIPPER_NAME, JOINT_NAMES, MODEL_SPECS

logger = logging.getLogger(__name__)

# Flush the command-channel response collector every N streamed commands
# (ACK-less streaming still queues event messages on the collector).
_RC_CLEAR_PERIOD = 100

# rbpodo SystemState bit definitions used by the real-mode safety gate.
_ARM_POWER_BIT = 6
_JOINT_ERROR_MASK = 0xFF00
_INIT_STAGE_MASK = 0x3F
_INIT_ERROR_MASK = 0xFFF
_ACTIVATION_DONE_STAGE = 6

# The record CLI connects the robot before the teleoperator. Keep the active
# RB instance so the software-only ``rb_auto`` teleoperator can capture the
# current pose without opening another rbpodo data connection.
_ACTIVE_RB_COBOT = None


def _set_active_rb_cobot(robot: RbCobot | None) -> None:
    global _ACTIVE_RB_COBOT
    _ACTIVE_RB_COBOT = robot


def get_active_rb_cobot() -> RbCobot | None:
    """Return the connected RB robot in this process, if one exists."""
    robot = _ACTIVE_RB_COBOT
    if robot is None or not robot.is_connected:
        return None
    return robot


def arm_power_is_on(system_state: Any) -> bool:
    """Return whether the control box reports power on the robot arm."""
    return bool((int(system_state.information_chunk_1) >> _ARM_POWER_BIT) & 1)


def activation_stage(system_state: Any) -> int:
    """Return the documented low-six-bit arm activation stage."""
    return int(system_state.init_state_info) & _INIT_STAGE_MASK


def real_state_issues(
    system_state: Any,
    *,
    require_collision_detection: bool,
) -> list[str]:
    """Return control-box conditions that make real motion unsafe to start."""
    issues: list[str] = []
    if int(system_state.real_vs_simulation_mode) & 0xF:
        issues.append("control box is not in Real Robot mode")
    if int(system_state.is_freedrive_mode) & 0x3:
        issues.append("freedrive mode is active")
    if int(system_state.op_stat_collision_occur) & 0x3:
        issues.append("external collision is reported")
    if int(system_state.op_stat_self_collision) & 0x3:
        issues.append("self-collision is reported")
    if int(system_state.op_stat_soft_estop_occur) & 0x3:
        issues.append("pause/soft-estop state is active")

    ems_code = int(system_state.op_stat_ems_flag) & 0x3F
    if ems_code:
        issues.append(f"kinematics emergency-stop code={ems_code}")
    sos_code = int(system_state.op_stat_sos_flag) & 0x3F
    if sos_code:
        issues.append(f"robot-arm device error code={sos_code}")

    joint_errors = [
        index
        for index, value in enumerate(system_state.jnt_info)
        if int(value) & _JOINT_ERROR_MASK
    ]
    if joint_errors:
        issues.append(f"joint error bits are set for joints {joint_errors}")

    init_error = int(system_state.init_error) & _INIT_ERROR_MASK
    if init_error:
        issues.append(f"arm activation error={init_error}")
    if not arm_power_is_on(system_state):
        issues.append("robot-arm power is off")
    stage = activation_stage(system_state)
    if stage != _ACTIVATION_DONE_STAGE:
        issues.append(
            f"arm activation is incomplete (stage={stage}); "
            "power on and initialize the arm from the pendant"
        )
    if (
        require_collision_detection
        and int(system_state.collision_detect_onoff) != 1
    ):
        issues.append("collision detection is disabled")
    return issues


def kinematics_estop_only(system_state: Any) -> bool:
    """True iff a kinematics EMS is latched and nothing else is wrong.

    A kinematics emergency-stop (``op_stat_ems_flag``) means the control
    box could not solve the requested motion (singularity / out of reach)
    and halted — it is not a physical-safety event. This predicate is
    deliberately strict: it returns False if *any* collision, self-
    collision, soft-estop, device-error (SOS), freedrive, joint-error,
    power, activation, or simulation-mode condition is also present, so an
    auto-clear built on it can never mask a real safety stop.
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
    if int(system_state.real_vs_simulation_mode) & 0xF:
        return False
    if any(int(v) & _JOINT_ERROR_MASK for v in system_state.jnt_info):
        return False
    if not arm_power_is_on(system_state):
        return False
    if activation_stage(system_state) != _ACTIVATION_DONE_STAGE:
        return False
    return True


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


def wrap_deg(angles_deg: np.ndarray) -> np.ndarray:
    """Map angles (degrees) into [-180, 180] (either sign at the boundary)."""
    return angles_deg - 360.0 * np.round(angles_deg / 360.0)


def clamp_ee_target(
    target: np.ndarray,
    last: np.ndarray,
    max_pos_delta_mm: float,
    max_rot_delta_deg: float,
    bounds_mm: np.ndarray | None = None,
) -> np.ndarray:
    """Clamp a streamed EE target ([x,y,z] mm + [rx,ry,rz] deg) for safety.

    Positions are optionally clipped to a workspace box, then the per-step
    change from ``last`` is limited. Orientation deltas are computed
    wrap-aware so a target crossing the ±180° seam does not produce a full
    revolution; the returned angles stay continuous with ``last``.

    Pure function (unit-testable without rbpodo).
    """
    target = np.asarray(target, dtype=np.float64).copy()
    last = np.asarray(last, dtype=np.float64)

    if bounds_mm is not None:
        target[:3] = np.clip(target[:3], bounds_mm[:, 0], bounds_mm[:, 1])

    out = last.copy()
    pos_delta = np.clip(target[:3] - last[:3], -max_pos_delta_mm, max_pos_delta_mm)
    out[:3] = last[:3] + pos_delta
    rot_delta = np.clip(
        wrap_deg(target[3:] - last[3:]), -max_rot_delta_deg, max_rot_delta_deg
    )
    out[3:] = last[3:] + rot_delta
    return out


class RbCobot(Robot):
    """LeRobot Robot implementation shared by the RB-Series cobots.

    Do not instantiate through the CLI directly — use a registered
    per-model shim such as :class:`~lerobot_robot_rb.rb10.Rb10`
    (``--robot.type=rb10``).

    Observation features
    --------------------
    * Joint positions (radians): ``joint_0`` .. ``joint_5``.
    * EE (TCP) pose ``ee_x`` .. ``ee_rz`` (metres / radians) when
      ``action_space="ee"``.
    * Optional ``<joint>.vel`` (finite-differenced) and ``<joint>.current``
      (A) channels when ``use_velocity`` / ``use_current`` are set.
    * Gripper position (normalised, 1.0 = open) when a gripper is configured.
    * One ``(H, W, 3)`` image per configured camera.

    Action features
    ---------------
    Joint positions (radians) or the EE pose (``action_space="ee"``), plus
    the normalised gripper when configured.

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

        # Streaming state machine (see module docstring). The worker thread
        # interpolates _interp_prev -> _interp_target over _interp_dur and
        # refreshes move_servo_j/l; all four fields live under _command_lock.
        self._streaming: bool = False
        self._command_lock = threading.Lock()
        self._interp_prev: np.ndarray | None = None
        self._interp_target: np.ndarray | None = None
        self._interp_t0: float = 0.0
        self._interp_dur: float = config.interp_min_s
        self._action_dt_ema: float | None = None
        self._last_action_time: float | None = None
        self._stream_stop = threading.Event()
        self._stream_thread: threading.Thread | None = None
        self._stream_error: Exception | None = None
        self._cmd_error_count: int = 0
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
        self._ee_bounds_mm = (
            np.asarray(config.ee_bounds_mm, dtype=np.float64)
            if config.ee_bounds_mm is not None
            else None
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
    def _ee_ft(self) -> dict[str, type]:
        if self._config.action_space != "ee":
            return {}
        return {name: float for name in EE_NAMES}

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
        features: dict[str, Any] = {
            **self._motors_ft,
            **self._ee_ft,
            **self._gripper_ft,
        }
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
        if self._config.action_space == "ee":
            return {**self._ee_ft, **self._gripper_ft}
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
        mode_result = self._robot.set_operation_mode(self._rc, mode)
        self._robot.set_speed_bar(self._rc, cfg.speed_bar)
        self._rc.error().throw_if_not_empty()
        self._rc.clear()
        if not mode_result.is_success():
            raise RuntimeError(
                f"rbpodo did not acknowledge the {cfg.operation_mode} "
                "operation-mode request."
            )

        # 3. Wait until the data channel reflects the requested mode, then
        #    gate real mode on a safe control-box state. Arm power/servo
        #    activation is never automatic — use the pendant.
        data = self._wait_for_operation_mode(cfg.operation_mode)
        if cfg.operation_mode == "real":
            self._assert_real_state(data.sdata)

        # 4. Gripper (config-selected; None when gripper_type="none").
        self._gripper = make_gripper(cfg, rb.Cobot(cfg.ip, cfg.command_port))
        if self._gripper is not None:
            self._gripper.connect()

        # 5. Cameras.
        for cam in self.cameras.values():
            cam.connect()

        self._is_connected = True
        self.configure()
        _set_active_rb_cobot(self)
        logger.info(f"{self} connected (operation_mode={cfg.operation_mode}).")

        # 6. Optionally move to the ready pose before inference begins.
        if cfg.move_to_ready_on_connect:
            self.move_to_ready_pose()

    def _wait_for_operation_mode(self, operation_mode: str) -> Any:
        """Wait until the data channel reflects the requested mode."""
        expected = 0 if operation_mode == "real" else 1
        deadline = time.monotonic() + self._config.operation_mode_timeout_sec
        last_mode = None

        while time.monotonic() < deadline:
            data = self._data.request_data(self._config.data_timeout_sec)
            if data is None:
                continue
            last_mode = int(data.sdata.real_vs_simulation_mode) & 0xF
            if last_mode == expected:
                return data
            time.sleep(0.05)

        if last_mode is None:
            raise ConnectionError(
                f"No state data from {self._config.ip}:{self._config.data_port} "
                f"within {self._config.operation_mode_timeout_sec}s - check "
                "the robot's data channel."
            )
        raise TimeoutError(
            "Timed out waiting for control-box operation mode "
            f"{operation_mode!r} (last mode value={last_mode})."
        )

    def _assert_real_state(self, system_state: Any) -> None:
        issues = real_state_issues(
            system_state,
            require_collision_detection=(
                self._config.require_collision_detection
            ),
        )
        if issues:
            raise RuntimeError(
                "RB real-mode safety check failed: " + "; ".join(issues)
            )

    def try_clear_kinematics_estop(self) -> bool:
        """Clear a latched kinematics EMS, only if it is the sole fault.

        Streaming Cartesian targets toward a singularity or out of reach
        makes the control box latch a kinematics emergency-stop and refuse
        further motion. That is not a physical-safety event, so this method
        offers a programmatic recovery: it stops streaming, verifies via
        :func:`kinematics_estop_only` that nothing else is wrong (a real
        collision / soft-estop returns False and is never touched), issues
        ``task_stop`` to clear the latch, and re-checks. Returns True only
        if the EMS actually cleared.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._exit_streaming()
        data = self._data.request_data(self._config.data_timeout_sec)
        if data is None or not kinematics_estop_only(data.sdata):
            return False

        self._robot.task_stop(self._rc, 3.0)
        self._rc.clear()
        time.sleep(0.3)

        data = self._data.request_data(self._config.data_timeout_sec)
        if data is None:
            return False
        cleared = not int(data.sdata.op_stat_ems_flag) & 0x3F
        if cleared:
            logger.warning(
                "Kinematics EMS auto-cleared (IK failure, not a safety "
                "stop). Re-latch and resume from a non-singular pose."
            )
        return cleared

    def disconnect(self) -> None:
        if not self.is_connected:
            return

        # Stop the servo worker and issue a zero-speed command before closing.
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
        if _ACTIVE_RB_COBOT is self:
            _set_active_rb_cobot(None)
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
            f"{self._config.model} configured: "
            f"action_space={self._config.action_space}, "
            f"speed_bar={self._config.speed_bar}."
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

    def _current_pose_native(self, sdata: Any) -> np.ndarray:
        """Streamed-target seed in native units (deg, or mm+deg for ee).

        ``jnt_ang``/``tcp_pos`` do not change in control-box simulation
        mode; the ``*_ref`` fields are the simulated state that follows
        motion commands.
        """
        sim = self._config.operation_mode == "simulation"
        if self._config.action_space == "ee":
            raw = sdata.tcp_ref if sim else sdata.tcp_pos
        else:
            raw = sdata.jnt_ref if sim else sdata.jnt_ang
        return np.asarray(raw, dtype=np.float64)[:DOF].copy()

    def _enter_streaming(self) -> None:
        """Switch the command channel to ACK-less servo streaming.

        Seeds the interpolator from the current state, then starts the
        high-rate worker. rbpodo's reference implementation refreshes
        ``move_servo_j`` every 5 ms with a continuously advancing target;
        replicating that signal shape (instead of action-rate steps) is
        what keeps the servo reference — and the joint torques — smooth.
        """
        data = self._data.request_data(self._config.data_timeout_sec)
        if data is None:
            raise DeviceNotConnectedError(
                f"{self} data channel timed out while entering streaming mode."
            )
        if self._config.operation_mode == "real":
            self._assert_real_state(data.sdata)

        seed = self._current_pose_native(data.sdata)
        now = time.perf_counter()
        with self._command_lock:
            self._interp_prev = seed.copy()
            self._interp_target = seed.copy()
            self._interp_t0 = now
            self._interp_dur = self._config.interp_min_s
        self._action_dt_ema = None
        self._last_action_time = None

        self._rc.clear()
        self._robot.disable_waiting_ack(self._rc)
        self._stream_stop.clear()
        self._stream_error = None
        self._cmd_error_count = 0
        self._tick = 0
        self._streaming = True
        self._stream_thread = threading.Thread(
            target=self._stream_commands,
            name=f"{self.name}-servo",
            daemon=True,
        )
        self._stream_thread.start()
        logger.info(
            "Servo streaming started (%s mode, %.0f Hz refresh, "
            "ACK waiting disabled).",
            self._config.action_space,
            1.0 / self._config.servo_command_period_s,
        )

    def _stream_commands(self) -> None:
        """Refresh the interpolated servo target at the worker rate."""
        cfg = self._config
        period = cfg.servo_command_period_s
        send_ee = cfg.action_space == "ee"
        deadline = time.perf_counter()

        try:
            while not self._stream_stop.is_set():
                now = time.perf_counter()
                with self._command_lock:
                    if self._interp_prev is None or self._interp_target is None:
                        return
                    phase = min(1.0, (now - self._interp_t0) / self._interp_dur)
                    cmd = (
                        self._interp_prev
                        + (self._interp_target - self._interp_prev) * phase
                    )

                if send_ee:
                    self._robot.move_servo_l(
                        self._rc, cmd, cfg.servo_t1, cfg.servo_t2,
                        cfg.servo_gain, cfg.servo_alpha,
                    )
                else:
                    self._robot.move_servo_j(
                        self._rc, cmd, cfg.servo_t1, cfg.servo_t2,
                        cfg.servo_gain, cfg.servo_alpha,
                    )

                self._tick += 1
                if self._tick % _RC_CLEAR_PERIOD == 0:
                    # Surface rejected commands (e.g. "unsorvable"/
                    # "armstratch" when EE streaming hits an IK failure or
                    # a stretched-arm singularity) instead of silently
                    # discarding them with the routine event messages.
                    errors = list(self._rc.error())
                    if errors:
                        self._cmd_error_count += len(errors)
                        logger.warning(
                            "Control box rejected streamed commands "
                            "(%d total): %s",
                            self._cmd_error_count,
                            errors[-1],
                        )
                    self._rc.clear()

                deadline += period
                remaining = deadline - time.perf_counter()
                if remaining > 0:
                    self._stream_stop.wait(remaining)
                else:
                    # Do not burst-send commands after a scheduling delay.
                    deadline = time.perf_counter()
        except Exception as exc:
            self._stream_error = exc
            self._stream_stop.set()

    def _exit_streaming(self) -> None:
        """Stop servo motion and re-enable ACK waiting (idempotent)."""
        if not self._streaming:
            return

        self._stream_stop.set()
        worker_stopped = True
        if self._stream_thread is not None:
            self._stream_thread.join(timeout=1.0)
            worker_stopped = not self._stream_thread.is_alive()
            if not worker_stopped:
                logger.error(
                    "Servo streaming thread did not stop within one second."
                )

        if self._stream_error is not None:
            logger.error(
                "Servo streaming worker stopped with an error: %s",
                self._stream_error,
            )

        try:
            if worker_stopped:
                # Match rbpodo's move_servo_j example: zero joint velocity
                # ends servo motion before ACK waiting is restored.
                self._robot.move_speed_j(
                    self._rc,
                    np.zeros(DOF, dtype=np.float64),
                    self._config.servo_t1,
                    self._config.servo_t2,
                    self._config.servo_gain,
                    self._config.servo_alpha,
                )
            self._robot.enable_waiting_ack(self._rc)
            if worker_stopped:
                result = self._robot.wait_for_move_finished(self._rc, 1.0)
                if not result.is_success():
                    logger.warning(
                        "Servo stop was not confirmed within one second."
                    )
        except Exception as exc:
            logger.warning(f"Servo streaming stop failed: {exc}")
        self._rc.clear()
        self._streaming = False
        with self._command_lock:
            self._interp_prev = None
            self._interp_target = None
        self._stream_thread = None
        self._stream_error = None
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

        if self._config.operation_mode == "real":
            self._assert_real_state(data.sdata)

        sim = self._config.operation_mode == "simulation"
        obs: dict[str, Any] = {}
        q_deg = data.sdata.jnt_ref if sim else data.sdata.jnt_ang
        q_rad = np.deg2rad(np.asarray(q_deg, dtype=np.float64)[:DOF])
        for i, name in enumerate(JOINT_NAMES):
            obs[name] = float(q_rad[i])

        if self._config.action_space == "ee":
            # tcp pose is [x, y, z] mm + [rx, ry, rz] deg -> m / rad.
            tcp = np.asarray(
                data.sdata.tcp_ref if sim else data.sdata.tcp_pos,
                dtype=np.float64,
            )[:DOF]
            ee = np.concatenate((tcp[:3] / 1000.0, np.deg2rad(tcp[3:])))
            for i, name in enumerate(EE_NAMES):
                obs[name] = float(ee[i])

        if self._config.use_velocity:
            t = float(data.sdata.time)
            if (
                self._prev_q_rad is None
                or self._prev_time is None
                or t <= self._prev_time
            ):
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

    def _next_interp_duration(self, now: float) -> float:
        """Track the send_action() call interval with a light EMA."""
        if self._last_action_time is not None:
            dt = now - self._last_action_time
            if self._action_dt_ema is None:
                self._action_dt_ema = dt
            else:
                self._action_dt_ema += 0.2 * (dt - self._action_dt_ema)
        self._last_action_time = now
        estimate = (
            self._action_dt_ema
            if self._action_dt_ema is not None
            else self._config.interp_min_s
        )
        return float(
            np.clip(estimate, self._config.interp_min_s, self._config.interp_max_s)
        )

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        stream_error = self._stream_error
        if stream_error is not None:
            raise RuntimeError("Servo streaming worker failed.") from stream_error

        if not self._streaming:
            self._enter_streaming()

        ee_mode = self._config.action_space == "ee"
        if ee_mode:
            # Dataset m / rad -> control box mm / deg.
            raw = np.array([float(action[name]) for name in EE_NAMES])
            target_native = np.concatenate(
                (raw[:3] * 1000.0, np.rad2deg(raw[3:]))
            )
        else:
            target_native = np.rad2deg(
                np.array([float(action[name]) for name in JOINT_NAMES])
            )

        now = time.perf_counter()
        duration = self._next_interp_duration(now)
        with self._command_lock:
            if self._interp_prev is None or self._interp_target is None:
                raise RuntimeError("Servo streaming target was not initialised.")
            # Continue from the point the worker has interpolated to, and
            # clamp the new target against the previous one so per-step
            # speed limits hold regardless of the action rate.
            phase = min(1.0, (now - self._interp_t0) / self._interp_dur)
            current_cmd = (
                self._interp_prev
                + (self._interp_target - self._interp_prev) * phase
            )
            if ee_mode:
                cmd_native = clamp_ee_target(
                    target_native,
                    self._interp_target,
                    self._config.max_ee_pos_delta_mm,
                    self._config.max_ee_rot_delta_deg,
                    self._ee_bounds_mm,
                )
            else:
                cmd_native = clamp_target(
                    target_native,
                    self._interp_target,
                    self._joint_limits_deg,
                    self._config.max_joint_delta_deg,
                )
            self._interp_prev = current_cmd
            self._interp_target = cmd_native
            self._interp_t0 = now
            self._interp_dur = duration

        # Return what was actually commanded (post-clamp), so datasets
        # record the executed action when a safety clamp engages.
        if ee_mode:
            sent_vals = np.concatenate(
                (cmd_native[:3] / 1000.0, np.deg2rad(cmd_native[3:]))
            )
            sent: dict[str, Any] = {
                name: float(v) for name, v in zip(EE_NAMES, sent_vals)
            }
        else:
            sent = {
                name: float(v)
                for name, v in zip(JOINT_NAMES, np.deg2rad(cmd_native))
            }

        if self._gripper is not None and GRIPPER_NAME in action:
            # Dataset 1.0 = open -> hardware 0.0 = open.
            hw_target = float(
                np.clip(1.0 - float(action[GRIPPER_NAME]), 0.0, 1.0)
            )
            self._gripper.set_position(hw_target)
            sent[GRIPPER_NAME] = 1.0 - hw_target

        return sent
