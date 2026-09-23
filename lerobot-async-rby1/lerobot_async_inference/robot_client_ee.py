"""Dedicated asynchronous end-effector inference client for RB10/RB10-E.

The LeRobot robot adapter is always ``rb10`` because both physical variants
share the RB ServoJ driver.  ``kinematics_model`` selects the model used by
the policy adapter for FK/IK.

The client defaults to dry-run mode.  In real mode it first moves to the
dataset ready pose (a profiled joint move), then waits paused.  Press ``f``
to start inference, ``s`` to stop inference and return to the ready pose
(or to abort a ready move in progress), or ``q`` to quit.
"""

import logging
import math
import select
import sys
import termios
import threading
import time
import tty
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from pprint import pformat
from queue import Empty, SimpleQueue
from typing import Any, Protocol

import draccus
import numpy as np
from scipy.spatial.transform import Rotation

try:
    # The controller/state-machine module remains importable in hardware-free
    # test environments where LeRobot is not installed.  A real CLI run gets
    # the registered polymorphic config type here.
    from lerobot.robots.config import RobotConfig
except ModuleNotFoundError:  # pragma: no cover - exercised by minimal test envs
    class RobotConfig:  # type: ignore[no-redef]
        pass

LOGGER = logging.getLogger(__name__)
JOINT_KEYS = tuple(f"joint_{index}" for index in range(6))

# Ready pose = mean of the first-frame joint state over all 197 episodes of
# rainbowrobotics/rb10_iros_double_trim (revision f7e8b828, joint_0..5, rad).
# Per-joint spread across episodes is 1.7-4.9 deg (std), so the mean is a
# representative recorded start pose, not an arbitrary preset.  Degrees:
# [-174.79, -2.42, 143.55, 40.28, 275.53, 179.77].
DEFAULT_READY_POSE_RAD: tuple[float, ...] = (
    -3.0506391525268555,
    -0.04220624640583992,
    2.505460500717163,
    0.7030301690101624,
    4.8088812828063965,
    3.1375038623809814,
)

# Mandatory server contract checked by the describe RPC before readiness.
# Each action/state is a Cartesian pose (XYZ + extrinsic XYZ Euler angles,
# composed as RzRyRx) and one
# normalized gripper value.  Chunks are never averaged, especially not their
# Euler components.
SERVER_DESCRIPTOR: dict[str, Any] = {
    "protocol": "rb10-ee-v1",
    "state_dim": 7,
    "action_dim": 7,
    "position_unit": "m",
    "angle_unit": "rad",
    "euler_convention": "RzRyRx",
    "base_frame": "link0",
    "target_frame": "tcp",
    "gripper_open": 1.0,
}


class RobotLike(Protocol):
    def get_joint_positions(self, *, measured: bool = True) -> np.ndarray: ...

    def get_reference_joint_positions(self) -> np.ndarray: ...

    def get_observation(self) -> dict[str, Any]: ...

    def send_action(self, action: dict[str, float]) -> dict[str, Any]: ...

    def send_cartesian_action(
        self, pose_m_rad: np.ndarray, gripper: float | None = None
    ) -> dict[str, Any]: ...

    def enable_servo_commands(self) -> None: ...

    def disable_servo_commands(self) -> None: ...


class AdapterLike(Protocol):
    kinematics: Any

    def validate_robot(self, robot: RobotLike) -> None: ...

    def build_observation(self, raw: dict[str, Any], task: str) -> dict[str, Any]: ...

    def joint_action(
        self,
        action7: np.ndarray,
        current_q6: np.ndarray,
        *,
        max_joint_delta: float,
        previous_q: np.ndarray | None = None,
        max_tracking_error: float | None = None,
    ) -> dict[str, float]: ...


class ClientLike(Protocol):
    def get_action(self, observation: dict[str, Any]) -> Any: ...

    def close(self) -> None: ...


@dataclass
class EEClientConfig:
    """CLI configuration for the dedicated RB end-effector client."""

    robot: RobotConfig
    kinematics_model: str = field(
        default="rb10",
        metadata={"help": "Physical kinematics model: rb10 or rb10e"},
    )
    task: str = ""
    server_address: str = "127.0.0.1:5555"
    wrist_camera_key: str = "wrist"
    front_camera_key: str = "front"
    control_rate_hz: float = 30.0
    max_joint_velocity_rad_s: float = 0.5
    max_joint_delta_cap_rad: float = 0.05
    max_joint_tracking_error_rad: float = 0.05
    timeout_ms: int = 1000
    reconnect_delay_s: float = 0.25
    dry_run: bool = True
    action_log_interval_s: float = 1.0
    # "joint": client IK + ServoJ (default).  "cartesian": bounded TCP
    # waypoints sent as ServoL; the control box solves IK with its own TCP
    # setting, so joint-envelope checks apply only to measured joints.
    servo_mode: str = "joint"
    max_tcp_speed_m_s: float = 0.15
    max_tcp_angular_speed_rad_s: float = 0.5
    # |previous TCP command - offset-corrected measured TCP| bounds.
    max_tcp_tracking_error_m: float = 0.03
    max_tcp_tracking_error_rad: float = 0.15
    # Profiled joint move to the recorded start pose at startup and after
    # ``s``.  Skipped in dry-run (no actuation) or when disabled.
    move_to_ready_on_start: bool = True
    move_to_ready_on_stop: bool = True
    ready_pose_rad: list[float] = field(default_factory=lambda: list(DEFAULT_READY_POSE_RAD))
    ready_move_duration_s: float = 5.0
    ready_settle_s: float = 0.5
    # Blind joint-space move: cap its peak joint speed well below the
    # inference speed limit so servo lag plus impedance sag stays inside the
    # tracking bound on long moves (a 44 deg move at 5 s tripped it).
    ready_max_joint_speed_rad_s: float = 0.15
    # The recorded start pose has the gripper open in every episode; open it
    # as the ready move begins (startup and after ``s``).
    ready_gripper_open: bool = True

    def __post_init__(self) -> None:
        if self.kinematics_model not in {"rb10", "rb10e"}:
            raise ValueError("kinematics_model must be 'rb10' or 'rb10e'")
        if getattr(self.robot, "model", None) != "rb10":
            raise ValueError("robot must use the rb10 ServoJ adapter for RB10 and RB10-E")
        state_timeout = float(getattr(self.robot, "first_state_timeout_s", math.inf))
        if not math.isfinite(state_timeout) or not 0.0 < state_timeout <= 0.25:
            raise ValueError(
                "robot.first_state_timeout_s must be finite and in (0, 0.25] "
                "so stale joint state fails closed"
            )
        if not self.server_address:
            raise ValueError("server_address cannot be empty")
        if not self.task.strip():
            raise ValueError("task cannot be empty")
        if (
            not self.wrist_camera_key
            or not self.front_camera_key
            or self.wrist_camera_key == self.front_camera_key
        ):
            raise ValueError("camera keys must be nonempty and distinct")
        for name in (
            "control_rate_hz",
            "max_joint_velocity_rad_s",
            "max_joint_delta_cap_rad",
            "max_joint_tracking_error_rad",
            "reconnect_delay_s",
            "ready_move_duration_s",
            "ready_max_joint_speed_rad_s",
            "max_tcp_speed_m_s",
            "max_tcp_angular_speed_rad_s",
            "max_tcp_tracking_error_m",
            "max_tcp_tracking_error_rad",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0, got {value}")
        if not math.isfinite(self.ready_settle_s) or self.ready_settle_s < 0.0:
            raise ValueError("ready_settle_s must be finite and >= 0")
        if self.servo_mode not in {"joint", "cartesian"}:
            raise ValueError("servo_mode must be 'joint' or 'cartesian'")
        ready = np.asarray(self.ready_pose_rad, dtype=np.float64)
        if ready.shape != (6,) or not np.all(np.isfinite(ready)):
            raise ValueError("ready_pose_rad must be six finite joint angles in radians")
        if not isinstance(self.timeout_ms, int) or not 0 < self.timeout_ms <= 1000:
            raise ValueError("timeout_ms must be an integer in [1, 1000]")
        if not math.isfinite(self.action_log_interval_s) or self.action_log_interval_s < 0:
            raise ValueError("action_log_interval_s must be finite and >= 0 (0 disables logging)")
        robot_rate = float(getattr(self.robot, "control_rate_hz", self.control_rate_hz))
        if not math.isfinite(robot_rate) or not math.isclose(
            robot_rate, self.control_rate_hz, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError("control_rate_hz must match robot.control_rate_hz")

    @property
    def nominal_dt(self) -> float:
        return 1.0 / self.control_rate_hz

    @property
    def max_joint_delta(self) -> float:
        return min(
            self.max_joint_velocity_rad_s * self.nominal_dt,
            self.max_joint_delta_cap_rad,
        )


@dataclass(frozen=True)
class _ObservationRequest:
    sequence: int
    generation: int
    captured_at: float
    wire: dict[str, Any]


@dataclass(frozen=True)
class _ActionChunk:
    generation: int
    observed_at: float
    received_at: float
    actions: np.ndarray


@dataclass
class _ReadyMove:
    """A profiled joint trajectory streamed through the ServoJ gate.

    The trajectory is anchored to the controller reference (``jnt_ref``) and
    continues from the previous command, exactly like inference commands.
    ``offset`` is ``jnt_ang - jnt_ref`` at the start: the controller's steady
    tracking bias, which measured joints are corrected by before comparison.
    """

    reason: str
    start_q: np.ndarray
    target_q: np.ndarray
    last_q: np.ndarray
    offset: np.ndarray
    started_at: float
    last_tick_at: float
    duration: float
    arrived_at: float | None = None


class _LatestObservation:
    """A blocking, latest-only handoff from the control to network thread."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._value: _ObservationRequest | None = None

    def put(self, value: _ObservationRequest) -> None:
        with self._condition:
            self._value = value
            self._condition.notify()

    def get_after(
        self,
        sequence: int,
        stop_event: threading.Event,
        timeout: float = 0.1,
    ) -> _ObservationRequest | None:
        with self._condition:
            self._condition.wait_for(
                lambda: stop_event.is_set()
                or (self._value is not None and self._value.sequence > sequence),
                timeout=timeout,
            )
            if stop_event.is_set() or self._value is None or self._value.sequence <= sequence:
                return None
            return self._value

    def wake(self) -> None:
        with self._condition:
            self._condition.notify_all()


def _load_policy_api() -> tuple[type[Any], type[Any], Callable[[Any], np.ndarray]]:
    from .policy.rb10_ee import (  # Imported lazily so config validation stays hardware-free.
        EEPolicyAdapter,
        ReconnectingEEClient,
        parse_ee_actions,
    )

    return EEPolicyAdapter, ReconnectingEEClient, parse_ee_actions


def _validate_descriptor(descriptor: dict[str, Any]) -> None:
    if not isinstance(descriptor, dict):
        raise ValueError("EE server describe() must return a dict")
    mismatches = {
        key: (expected, descriptor.get(key))
        for key, expected in SERVER_DESCRIPTOR.items()
        if descriptor.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Incompatible EE server descriptor: {mismatches}")


class AsyncEEController:
    """Safety state machine with single-threaded robot ownership.

    ``tick`` and ``handle_key`` must be called by the thread that owns the
    robot.  The background thread owns the policy client's entire lifetime.
    """

    def __init__(
        self,
        config: EEClientConfig,
        robot: RobotLike,
        adapter: AdapterLike,
        *,
        client_factory: Callable[[str, int], ClientLike] | None = None,
        parse_actions: Callable[[Any], np.ndarray] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.robot = robot
        self.adapter = adapter
        self._clock = clock
        if client_factory is None or parse_actions is None:
            _, default_client, default_parser = _load_policy_api()
            client_factory = client_factory or (
                lambda address, timeout: default_client(address, timeout_ms=timeout)
            )
            parse_actions = parse_actions or default_parser
        self._client_factory = client_factory
        self._parse_actions = parse_actions

        self._state_lock = threading.RLock()
        self._generation = 0
        self._sequence = 0
        self._paused = True
        self._network_ready = False
        self._quit = False
        self._last_command_q: np.ndarray | None = None
        # Controller steady-state bias (jnt_ang - jnt_ref) captured at
        # activation.  Measured joints are corrected by it before tracking
        # comparisons, so a constant impedance/gravity offset is not counted
        # as tracking error and does not shrink the IK search box one-sidedly.
        self._tracking_offset: np.ndarray | None = None
        # Cartesian mode: last commanded TCP pose [x, y, z, rx, ry, rz].
        self._last_command_pose: np.ndarray | None = None
        self._action_chunk: _ActionChunk | None = None
        self._disable_requested = True
        self._last_pause_reason = "startup"
        self._last_action_log_at = -math.inf
        self._ready_move: _ReadyMove | None = None
        self._chunk_wait_logged_generation = -1

        self._observations = _LatestObservation()
        self._network_messages: SimpleQueue[tuple[str, str]] = SimpleQueue()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None

    @property
    def paused(self) -> bool:
        with self._state_lock:
            return self._paused

    @property
    def network_ready(self) -> bool:
        with self._state_lock:
            return self._network_ready

    @property
    def generation(self) -> int:
        with self._state_lock:
            return self._generation

    @property
    def should_quit(self) -> bool:
        with self._state_lock:
            return self._quit

    @property
    def last_pause_reason(self) -> str:
        with self._state_lock:
            return self._last_pause_reason

    def initialize_robot_safety(self) -> None:
        """Close the adapter's ServoJ gate after connect, from the main thread."""
        self.robot.disable_servo_commands()
        with self._state_lock:
            self._disable_requested = False

    def start_network_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop_event.clear()
        self._worker = threading.Thread(
            target=self._network_loop,
            name="rb10-ee-network",
            daemon=True,
        )
        self._worker.start()

    def stop_network_worker(self) -> None:
        self._stop_event.set()
        self._observations.wake()
        worker = self._worker
        if worker is not None:
            worker.join(timeout=max(1.2, self.config.timeout_ms / 1000.0 + 0.2))
            if worker.is_alive():
                LOGGER.warning("EE network worker did not stop within its request timeout")
        self._worker = None

    def _invalidate_locked(self, reason: str) -> None:
        self._generation += 1
        self._paused = True
        self._action_chunk = None
        self._last_command_q = None
        self._last_command_pose = None
        self._tracking_offset = None
        self._ready_move = None
        self._disable_requested = True
        self._last_pause_reason = reason

    def _pause_from_worker(self, reason: str) -> None:
        # This changes only software state.  The next main-thread tick performs
        # the robot gate call; no worker thread ever accesses the robot.
        with self._state_lock:
            self._network_ready = False
            self._invalidate_locked(reason)

    def _service_main_thread_safety(self) -> bool:
        with self._state_lock:
            disable = self._disable_requested
        if disable:
            try:
                self.robot.disable_servo_commands()
            except Exception:
                LOGGER.exception("Failed to close the RB ServoJ command gate")
            else:
                with self._state_lock:
                    self._disable_requested = False

        while True:
            try:
                level, message = self._network_messages.get_nowait()
            except Empty:
                break
            getattr(LOGGER, level)(message)
        with self._state_lock:
            return not self._disable_requested

    def activate(self) -> bool:
        """Arm inference after an explicit ``f``; never auto-resume motion."""
        if not self._service_main_thread_safety():
            return False
        with self._state_lock:
            if self._ready_move is not None:
                LOGGER.warning("Ready movement in progress; s aborts it, f is ignored")
                return False
            if not self._network_ready:
                self._last_pause_reason = "server not connected; press f after reconnect"
                return False
            self._paused = True
            self._action_chunk = None
            self._last_command_q = None
            self._tracking_offset = None
            generation_before_prepare = self._generation

        try:
            if self.config.dry_run:
                self.robot.disable_servo_commands()
            else:
                LOGGER.warning(
                    "First real activation may initialize impedance control and open the configured gripper"
                )
                self.robot.enable_servo_commands()
            # Preserve the controller's equilibrium/reference. Re-anchoring each
            # tick on a compliant measured pose can turn tracking bias into drift.
            reference, measured, offset = self._read_anchor_and_offset()
            LOGGER.info(
                "EE command anchor jnt_ref_deg=%s jnt_ang_deg=%s offset_deg=%s tracking_limit_rad=%.6f",
                np.round(np.rad2deg(reference), 5).tolist(),
                np.round(np.rad2deg(measured), 5).tolist(),
                np.round(np.rad2deg(offset), 4).tolist(),
                self.config.max_joint_tracking_error_rad,
            )
        except Exception as exc:
            with self._state_lock:
                self._invalidate_locked(f"servo gate activation fault: {exc}")
            self._service_main_thread_safety()
            LOGGER.exception("Failed to configure the RB ServoJ gate during activation")
            return False

        with self._state_lock:
            activated = (
                self._generation == generation_before_prepare
                and self._network_ready
                and self._paused
                and not self._disable_requested
            )
            if activated:
                self._generation += 1
                generation = self._generation
                self._last_command_q = reference.copy()
                self._last_command_pose = np.asarray(
                    self.adapter.kinematics.forward(reference), dtype=np.float64
                )
                self._tracking_offset = offset.copy()
                self._paused = False
            else:
                # Preparation completed after a safety transition.  Even if
                # the worker has already reconnected, require another f and
                # close any gate that preparation may have opened.
                self._paused = True
                self._disable_requested = True
        if not activated:
            self._service_main_thread_safety()
            return False
        LOGGER.info("EE inference active at generation %d (dry_run=%s)", generation, self.config.dry_run)
        return True

    def pause(self, reason: str = "operator pause") -> None:
        with self._state_lock:
            self._invalidate_locked(reason)
        self._service_main_thread_safety()

    def handle_key(self, key: str) -> bool:
        key = key.lower()
        if key == "f":
            return self.activate()
        if key == "s":
            with self._state_lock:
                aborting_ready = self._ready_move is not None
            if aborting_ready:
                # Never chain a second automatic move after an operator abort.
                self.pause("operator aborted ready movement")
                return True
            self.pause("operator pause")
            if self.config.move_to_ready_on_stop:
                self.start_ready_move("operator stop")
            return True
        if key == "q":
            self.pause("operator quit")
            with self._state_lock:
                self._quit = True
            return True
        return False

    def _fault(self, reason: str, exc: Exception | None = None) -> None:
        detail = reason if exc is None else f"{reason}: {exc}"
        with self._state_lock:
            self._invalidate_locked(detail)
        self._service_main_thread_safety()
        LOGGER.error("EE motion fault; explicit f required to rearm: %s", detail)

    def _read_anchor_and_offset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read controller reference and measured joints; validate their gap.

        Returns ``(reference, measured, offset)`` with ``offset = measured -
        reference``.  A gap beyond the tracking limit means the controller has
        not settled (or is not tracking its own reference), so it fails closed.
        """
        reference = np.asarray(self.robot.get_reference_joint_positions(), dtype=np.float64)
        measured = np.asarray(self.robot.get_joint_positions(measured=True), dtype=np.float64)
        self.adapter.kinematics.forward(reference)
        self.adapter.kinematics.forward(measured)
        gap = np.abs(reference - measured)
        if np.any(gap > self.config.max_joint_tracking_error_rad):
            worst = int(np.argmax(gap))
            raise ValueError(
                "controller reference exceeds the measured tracking error limit: "
                f"joint_{worst} |jnt_ref - jnt_ang| = {np.rad2deg(gap[worst]):.3f} deg "
                f"> {np.rad2deg(self.config.max_joint_tracking_error_rad):.3f} deg; "
                f"jnt_ref_deg={np.round(np.rad2deg(reference), 3).tolist()} "
                f"jnt_ang_deg={np.round(np.rad2deg(measured), 3).tolist()}. "
                "The control box reference has not settled on the measured pose "
                "(e.g. after freedrive, a manual jog or an emergency stop); re-sync it "
                "(a short pendant/jog move) before activating"
            )
        return reference, measured, measured - reference

    def start_ready_move(self, reason: str) -> bool:
        """Begin a profiled joint move to the ready pose; the caller owns the robot.

        Inference is paused first.  The move streams ServoJ commands from the
        controller reference towards ``ready_pose_rad`` with a cosine profile
        over at least ``ready_move_duration_s``, then settles and pauses.
        It is skipped in dry-run mode because it actuates the arm.
        """
        with self._state_lock:
            if self._ready_move is not None:
                return True
            self._invalidate_locked(f"ready movement: {reason}")
        if not self._service_main_thread_safety():
            return False
        if self.config.dry_run:
            LOGGER.info("Ready movement skipped in dry-run (%s)", reason)
            return False
        try:
            target = np.asarray(self.config.ready_pose_rad, dtype=np.float64)
            self.adapter.kinematics.forward(target)
            LOGGER.warning(
                "Ready movement (%s) to %s deg over >= %.1f s; clear the workspace. "
                "s aborts. Servo preparation may open the gripper.",
                reason, np.round(np.rad2deg(target), 3).tolist(), self.config.ready_move_duration_s,
            )
            self.robot.enable_servo_commands()
            # Actuator preparation can block; anchor only after it returns.
            reference, measured, offset = self._read_anchor_and_offset()
            speed = min(
                self.config.ready_max_joint_speed_rad_s,
                self.config.max_joint_delta / self.config.nominal_dt,
            )
            # A cosine profile peaks at pi/2 times the mean speed.
            duration = max(
                self.config.ready_move_duration_s,
                math.pi * float(np.max(np.abs(target - reference))) / (2.0 * speed),
            )
            now = self._clock()
            LOGGER.info(
                "Ready movement anchor jnt_ref_deg=%s jnt_ang_deg=%s offset_deg=%s duration_s=%.2f",
                np.round(np.rad2deg(reference), 5).tolist(),
                np.round(np.rad2deg(measured), 5).tolist(),
                np.round(np.rad2deg(offset), 4).tolist(),
                duration,
            )
            move = _ReadyMove(
                reason=reason,
                start_q=reference.copy(),
                target_q=target,
                last_q=reference.copy(),
                offset=offset,
                started_at=now,
                last_tick_at=now,
                duration=duration,
            )
            with self._state_lock:
                if self._disable_requested or self._quit:
                    raise RuntimeError("safety transition during ready preparation")
                self._ready_move = move
            return True
        except Exception as exc:
            self._fault("ready movement fault", exc)
            return False

    def _tick_ready_move(self, move: _ReadyMove) -> dict[str, float] | None:
        try:
            now = self._clock()
            if now - move.started_at > 2.0 * move.duration + 10.0:
                raise RuntimeError("ready pose arrival timed out")
            measured = np.asarray(self.robot.get_joint_positions(measured=True), dtype=np.float64)
            self.adapter.kinematics.forward(measured)
            current_ref = measured - move.offset
            if np.any(np.abs(move.last_q - current_ref) > self.config.max_joint_tracking_error_rad):
                raise RuntimeError("measured joints exceeded the ready movement tracking error limit")
            phase = min(1.0, max(0.0, (now - move.started_at) / move.duration))
            desired = move.start_q + (move.target_q - move.start_q) * (1 - math.cos(math.pi * phase)) / 2
            dt = min(self.config.nominal_dt, max(0.0, now - move.last_tick_at))
            step = min(self.config.max_joint_delta, self.config.max_joint_velocity_rad_s * dt)
            lower = np.maximum(current_ref - self.config.max_joint_tracking_error_rad, move.last_q - step)
            upper = np.minimum(current_ref + self.config.max_joint_tracking_error_rad, move.last_q + step)
            if np.any(lower > upper):
                raise RuntimeError("ready movement has no safe joint step")
            command_q = np.clip(desired, lower, upper)
            self.adapter.kinematics.forward(command_q)
            if phase >= 1.0 and move.arrived_at is None:
                move.arrived_at = now
            if move.arrived_at is not None and now - move.arrived_at >= self.config.ready_settle_s:
                residual = np.rad2deg(measured - move.target_q)
                self.pause("ready pose reached; press f to start inference")
                LOGGER.info(
                    "Ready pose reached (%s); measured_minus_target_deg=%s. Inference paused until f",
                    move.reason, np.round(residual, 3).tolist(),
                )
                return None
            if self._clock() - now > self.config.nominal_dt:
                raise RuntimeError("ready movement joint state became obsolete")
            command = dict(zip(JOINT_KEYS, map(float, command_q), strict=True))
            if self.config.ready_gripper_open:
                # Dataset convention 1 = open; the driver only forwards a change.
                command["gripper_0"] = 1.0
            with self._state_lock:
                if self._ready_move is not move or self._disable_requested:
                    return None
                self.robot.send_action(command)
                move.last_q = command_q.copy()
                move.last_tick_at = now
            return command
        except Exception as exc:
            self._fault("ready movement fault", exc)
            return None

    @staticmethod
    def _joint_vector(raw: dict[str, Any]) -> np.ndarray:
        q = np.asarray([raw[key] for key in JOINT_KEYS], dtype=np.float64)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            raise ValueError(f"invalid measured joints: {q.tolist()}")
        return q

    def _publish_observation(self, wire: dict[str, Any], now: float, generation: int) -> None:
        with self._state_lock:
            self._sequence += 1
            sequence = self._sequence
        self._observations.put(
            _ObservationRequest(
                sequence=sequence,
                generation=generation,
                captured_at=now,
                wire=wire,
            )
        )

    @staticmethod
    def _snapshot_observation(raw: dict[str, Any]) -> dict[str, Any]:
        """Own camera buffers before the capture call can reuse them."""
        return {
            key: np.array(value, copy=True, order="C") if isinstance(value, np.ndarray) else value
            for key, value in raw.items()
        }

    def tick(self) -> dict[str, float] | None:
        """Run one main-thread control tick and return the selected command."""
        self._service_main_thread_safety()
        with self._state_lock:
            ready_move = self._ready_move
        if ready_move is not None:
            return self._tick_ready_move(ready_move)
        with self._state_lock:
            if self._paused or self._quit:
                return None
            generation = self._generation
            offset = np.zeros(6) if self._tracking_offset is None else self._tracking_offset.copy()

        tick_started = self._clock()
        try:
            raw = self._snapshot_observation(self.robot.get_observation())
            current_q = self._joint_vector(raw)
            wire = self.adapter.build_observation(raw, self.config.task)
        except Exception as exc:
            self._fault("observation/FK fault", exc)
            return None
        # Camera reads block until each camera delivers a new frame, so the
        # observation is only "acquired" here.  Chunk indexing and the IK
        # budget are measured from this instant, not from the tick start.
        now = self._clock()
        observation_ms = (now - tick_started) * 1e3

        self._publish_observation(wire, now, generation)

        with self._state_lock:
            chunk = self._action_chunk
        if chunk is None or chunk.generation != generation:
            if self._chunk_wait_logged_generation != generation:
                self._chunk_wait_logged_generation = generation
                LOGGER.info("EE generation %d waiting for its first action chunk", generation)
            return None

        action_index = int(max(0.0, now - chunk.observed_at) / self.config.nominal_dt)
        if action_index >= len(chunk.actions):
            self._fault("action chunk horizon expired")
            return None

        # Measured joints corrected by the activation-time controller bias.
        # Tracking limits compare this against commands; the raw measurement
        # is still what the model observed.
        current_ref = current_q - offset
        action7 = chunk.actions[action_index].copy()
        if self.config.servo_mode == "cartesian":
            return self._cartesian_step(
                chunk, action_index, action7, current_q, current_ref, offset, now, observation_ms
            )
        try:
            with self._state_lock:
                previous_q = None if self._last_command_q is None else self._last_command_q.copy()
            if previous_q is None:
                raise ValueError("missing controller reference; explicit f required")
            if np.any(np.abs(previous_q - current_ref) > self.config.max_joint_tracking_error_rad):
                raise ValueError("measured joints exceeded the previous-command tracking error limit")
            command = self.adapter.joint_action(
                action7,
                current_ref,
                max_joint_delta=self.config.max_joint_delta,
                previous_q=previous_q,
                max_tracking_error=self.config.max_joint_tracking_error_rad,
            )
            target_q = np.asarray([command[key] for key in JOINT_KEYS], dtype=np.float64)
            command_values = np.asarray(list(command.values()), dtype=np.float64)
            if target_q.shape != (6,) or not np.all(np.isfinite(command_values)):
                raise ValueError("IK returned a malformed or non-finite joint action")
            if np.any(np.abs(target_q - current_ref) > self.config.max_joint_tracking_error_rad + 1e-12):
                raise ValueError("measured-to-command tracking error exceeds the configured limit")
            if previous_q is not None and np.any(
                np.abs(target_q - previous_q) > self.config.max_joint_delta + 1e-12
            ):
                raise ValueError("command-to-command joint delta exceeds the configured limit")
        except Exception as exc:
            tracking_error = None if previous_q is None else np.rad2deg(previous_q - current_ref)
            LOGGER.error(
                "Rejected EE action[%d]: target_xyz_m=%s measured_q_rad=%s "
                "previous_minus_corrected_measured_deg=%s offset_deg=%s",
                action_index, action7[:3].tolist(), current_q.tolist(),
                None if tracking_error is None else np.round(tracking_error, 4).tolist(),
                np.round(np.rad2deg(offset), 4).tolist(),
            )
            self._fault("IK/action safety fault", exc)
            return None
        ik_ms = (self._clock() - now) * 1e3

        if (
            self.config.action_log_interval_s > 0
            and now - self._last_action_log_at >= self.config.action_log_interval_s
        ):
            try:
                measured_xyz = np.asarray(self.adapter.kinematics.forward(current_q))[:3]
                command_xyz = np.asarray(self.adapter.kinematics.forward(target_q))[:3]
                anchor_xyz = np.asarray(self.adapter.kinematics.forward(previous_q))[:3]
                LOGGER.info(
                    "EE action[%d] frame=link0 unit=m dry_run=%s current_xyz=%s "
                    "target_xyz=%s target_delta=%s command_fk_delta=%s command_from_anchor_delta=%s "
                    "measured_q_deg=%s previous_command_q_deg=%s planned_command_q_deg=%s "
                    "tracking_error_deg=%s observation_ms=%.1f ik_ms=%.1f",
                    action_index, self.config.dry_run,
                    np.round(measured_xyz, 6).tolist(), np.round(action7[:3], 6).tolist(),
                    np.round(action7[:3] - measured_xyz, 6).tolist(),
                    np.round(command_xyz - measured_xyz, 6).tolist(),
                    np.round(command_xyz - anchor_xyz, 6).tolist(),
                    np.round(np.rad2deg(current_q), 5).tolist(),
                    np.round(np.rad2deg(previous_q), 5).tolist(),
                    np.round(np.rad2deg(target_q), 5).tolist(),
                    np.round(np.rad2deg(previous_q - current_ref), 4).tolist(),
                    observation_ms, ik_ms,
                )
                self._last_action_log_at = now
            except Exception as exc:
                self._fault("command FK diagnostic fault", exc)
                return None

        # Network phase can place this tick arbitrarily close to the next
        # nominal action-index boundary.  Give IK one full control period from
        # observation acquisition, while never extending validity beyond the
        # complete chunk horizon.
        chunk_deadline = chunk.observed_at + len(chunk.actions) * self.config.nominal_dt
        command_deadline = min(now + self.config.nominal_dt, chunk_deadline)
        if self._clock() > command_deadline:
            LOGGER.error(
                "IK overran its budget: observation_ms=%.1f ik_ms=%.1f budget_ms=%.1f",
                observation_ms, ik_ms, self.config.nominal_dt * 1e3,
            )
            self._fault("action became obsolete during IK")
            return None

        # Hold the state lock across the final gate check and send.  A worker
        # timeout first invalidates the generation under this same lock, so a
        # response from an old generation cannot race into the actuator path.
        try:
            with self._state_lock:
                if self._paused or self._generation != generation:
                    return None
                self._last_command_q = target_q.copy()
                if not self.config.dry_run:
                    self.robot.send_action(command)
        except Exception as exc:
            self._fault("robot send_action fault", exc)
            return None
        return command

    def _cartesian_step(
        self,
        chunk: _ActionChunk,
        action_index: int,
        action7: np.ndarray,
        current_q: np.ndarray,
        current_ref: np.ndarray,
        offset: np.ndarray,
        now: float,
        observation_ms: float,
    ) -> dict[str, float] | None:
        """ServoL variant of the command phase: bounded TCP waypoint, no client IK.

        The waypoint continues from the previous TCP command (anchored to the
        controller reference at activation) along the straight line and
        shortest rotation towards the model target, with one common fraction
        so the TCP direction is preserved.  Tracking compares the previous
        command with the offset-corrected measured TCP pose.
        """
        cfg = self.config
        try:
            with self._state_lock:
                previous = None if self._last_command_pose is None else self._last_command_pose.copy()
            if previous is None:
                raise ValueError("missing controller reference; explicit f required")
            measured_pose = np.asarray(self.adapter.kinematics.forward(current_ref), dtype=np.float64)
            previous_rotation = Rotation.from_euler("xyz", previous[3:])
            measured_rotation = Rotation.from_euler("xyz", measured_pose[3:])
            tracking_position = float(np.linalg.norm(previous[:3] - measured_pose[:3]))
            tracking_rotation = float(
                np.linalg.norm((previous_rotation * measured_rotation.inv()).as_rotvec())
            )
            if tracking_position > cfg.max_tcp_tracking_error_m or tracking_rotation > cfg.max_tcp_tracking_error_rad:
                raise ValueError(
                    "measured TCP exceeded the previous-command tracking error limit "
                    f"(position={tracking_position:.4f} m, rotation={tracking_rotation:.4f} rad)"
                )
            target = np.asarray(action7[:6], dtype=np.float64)
            if target.shape != (6,) or not np.all(np.isfinite(action7)):
                raise ValueError("EE action is malformed or non-finite")
            target_rotation = Rotation.from_euler("xyz", target[3:])
            delta_position = target[:3] - previous[:3]
            delta_rotvec = (target_rotation * previous_rotation.inv()).as_rotvec()
            position_norm = float(np.linalg.norm(delta_position))
            rotation_norm = float(np.linalg.norm(delta_rotvec))
            fraction = 1.0
            if position_norm > 0.0:
                fraction = min(fraction, cfg.max_tcp_speed_m_s * cfg.nominal_dt / position_norm)
            if rotation_norm > 0.0:
                fraction = min(fraction, cfg.max_tcp_angular_speed_rad_s * cfg.nominal_dt / rotation_norm)
            planned_rotation = Rotation.from_rotvec(fraction * delta_rotvec) * previous_rotation
            planned = np.concatenate(
                (previous[:3] + fraction * delta_position, planned_rotation.as_euler("xyz"))
            )
            if not np.all(np.isfinite(planned)):
                raise ValueError("planned TCP waypoint is non-finite")
            gripper = float(np.clip(action7[6], 0.0, 1.0))
        except Exception as exc:
            LOGGER.error(
                "Rejected EE action[%d] (cartesian): target_pose=%s measured_q_rad=%s offset_deg=%s",
                action_index, action7[:6].tolist(), current_q.tolist(),
                np.round(np.rad2deg(offset), 4).tolist(),
            )
            self._fault("TCP/action safety fault", exc)
            return None
        plan_ms = (self._clock() - now) * 1e3

        if cfg.action_log_interval_s > 0 and now - self._last_action_log_at >= cfg.action_log_interval_s:
            measured_xyz = np.asarray(self.adapter.kinematics.forward(current_q))[:3]
            LOGGER.info(
                "EE action[%d] mode=cartesian frame=link0 unit=m dry_run=%s current_xyz=%s "
                "target_xyz=%s target_delta=%s command_fk_delta=%s command_from_anchor_delta=%s "
                "measured_q_deg=%s planned_pose=%s fraction=%.3f "
                "tracking_error_m=%.4f tracking_error_rad=%.4f observation_ms=%.1f plan_ms=%.1f",
                action_index, cfg.dry_run,
                np.round(measured_xyz, 6).tolist(), np.round(target[:3], 6).tolist(),
                np.round(target[:3] - measured_xyz, 6).tolist(),
                np.round(planned[:3] - measured_xyz, 6).tolist(),
                np.round(planned[:3] - previous[:3], 6).tolist(),
                np.round(np.rad2deg(current_q), 5).tolist(),
                np.round(planned, 6).tolist(), fraction,
                tracking_position, tracking_rotation, observation_ms, plan_ms,
            )
            self._last_action_log_at = now

        chunk_deadline = chunk.observed_at + len(chunk.actions) * cfg.nominal_dt
        if self._clock() > min(now + cfg.nominal_dt, chunk_deadline):
            self._fault("action became obsolete during planning")
            return None

        command = {
            **dict(zip(("x", "y", "z", "rx", "ry", "rz"), map(float, planned), strict=True)),
            "gripper_0": gripper,
        }
        try:
            with self._state_lock:
                if self._paused or self._generation != chunk.generation:
                    return None
                self._last_command_pose = planned.copy()
                if not cfg.dry_run:
                    self.robot.send_cartesian_action(planned, gripper)
        except Exception as exc:
            self._fault("robot send_cartesian_action fault", exc)
            return None
        return command

    def _set_chunk(self, chunk: _ActionChunk) -> None:
        with self._state_lock:
            if not self._paused and chunk.generation == self._generation:
                self._action_chunk = chunk

    def _network_loop(self) -> None:
        client: ClientLike | None = None
        last_sequence = -1
        while not self._stop_event.is_set():
            if client is None:
                try:
                    client = self._client_factory(self.config.server_address, self.config.timeout_ms)
                    describe = getattr(client, "describe", None)
                    if not callable(describe):
                        raise ValueError("EE server client must provide describe() for protocol preflight")
                    _validate_descriptor(describe())
                    with self._state_lock:
                        self._network_ready = True
                    self._network_messages.put(("info", "Connected to RB10 EE inference server; press f to activate"))
                except Exception as exc:
                    if client is not None:
                        try:
                            client.close()
                        except Exception:
                            pass
                    client = None
                    with self._state_lock:
                        self._network_ready = False
                    self._network_messages.put(("warning", f"EE server reconnect failed: {exc}"))
                    self._stop_event.wait(self.config.reconnect_delay_s)
                    continue

            request = self._observations.get_after(last_sequence, self._stop_event)
            if request is None:
                continue
            last_sequence = request.sequence

            with self._state_lock:
                valid = not self._paused and request.generation == self._generation
            if not valid:
                continue

            try:
                response = client.get_action(request.wire)
                actions = np.asarray(self._parse_actions(response), dtype=np.float64)
                if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) == 0:
                    raise ValueError(f"expected a non-empty (T, 7) EE action chunk, got {actions.shape}")
                if not np.all(np.isfinite(actions)):
                    raise ValueError("EE action chunk contains non-finite values")
                self._set_chunk(
                    _ActionChunk(
                        generation=request.generation,
                        observed_at=request.captured_at,
                        received_at=self._clock(),
                        actions=actions.copy(),
                    )
                )
            except Exception as exc:
                self._pause_from_worker(f"network/policy outage: {exc}")
                self._network_messages.put(
                    ("error", f"EE inference failed; motion paused and explicit f required after reconnect: {exc}")
                )
                try:
                    client.close()
                except Exception:
                    pass
                client = None

        if client is not None:
            try:
                client.close()
            except Exception:
                LOGGER.exception("Failed to close EE inference client")
        with self._state_lock:
            self._network_ready = False


class _Keyboard:
    def __init__(self) -> None:
        self._fd: int | None = None
        self._settings: list[Any] | None = None

    def __enter__(self) -> "_Keyboard":
        if sys.stdin.isatty():
            self._fd = sys.stdin.fileno()
            self._settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def poll(self) -> str | None:
        if self._fd is None:
            return None
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        return sys.stdin.read(1) if readable else None

    def __exit__(self, *_: Any) -> None:
        if self._fd is not None and self._settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._settings)


def _with_inference_safe_start(cfg: EEClientConfig) -> EEClientConfig:
    """Return an effective config that cannot initialize actuators on connect."""
    robot_cfg = cfg.robot
    if not is_dataclass(robot_cfg) or "inference_safe_start" not in {
        item.name for item in fields(robot_cfg)
    }:
        raise ValueError(
            "The RB10 adapter must provide RobotConfig.inference_safe_start before EE inference can run"
        )
    safe_robot_cfg = replace(robot_cfg, inference_safe_start=True)
    return replace(cfg, robot=safe_robot_cfg)


def run_client(
    cfg: EEClientConfig,
    *,
    robot_factory: Callable[[RobotConfig], RobotLike] | None = None,
) -> None:
    """Construct, run, and cleanly stop the dedicated EE client."""
    # Preserve the parsed config while enforcing a no-actuator connect path.
    # EEClientConfig validation reruns for the effective clone before the
    # first possible robot connection.
    cfg = _with_inference_safe_start(cfg)
    adapter_cls, _, _ = _load_policy_api()
    if robot_factory is None:
        from lerobot.robots import make_robot_from_config

        robot_factory = make_robot_from_config
    robot = robot_factory(cfg.robot)
    adapter = adapter_cls(
        cfg.kinematics_model,
        wrist_camera_key=cfg.wrist_camera_key,
        front_camera_key=cfg.front_camera_key,
    )
    adapter.validate_robot(robot)

    controller = AsyncEEController(cfg, robot, adapter)
    connected = False
    try:
        LOGGER.info("Connecting to robot and cameras (actuator initialization deferred)")
        robot.connect()  # type: ignore[attr-defined]
        connected = True
        controller.initialize_robot_safety()
        controller.start_network_worker()
        with _Keyboard() as keyboard:
            LOGGER.info("Keys: f=activate inference, s=stop and return to ready pose, q=quit")
            if cfg.move_to_ready_on_start:
                controller.start_ready_move("startup")
            else:
                LOGGER.info("Started paused without preset movement")
            while not controller.should_quit:
                started = time.monotonic()
                key = keyboard.poll()
                if key is not None:
                    controller.handle_key(key)
                controller.tick()
                time.sleep(max(0.0, cfg.nominal_dt - (time.monotonic() - started)))
    except KeyboardInterrupt:
        controller.pause("keyboard interrupt")
    finally:
        try:
            controller.pause("shutdown")
        except Exception:
            LOGGER.exception("Failed while pausing the EE controller during shutdown")
        try:
            controller.stop_network_worker()
        except Exception:
            LOGGER.exception("Failed while stopping the EE network worker")
        if connected:
            try:
                robot.disconnect()  # type: ignore[attr-defined]
            except Exception:
                LOGGER.exception("Failed while disconnecting the RB robot")


@draccus.wrap()
def async_ee_client(cfg: EEClientConfig) -> None:
    logging.info(pformat(asdict(cfg)))
    run_client(cfg)


def main() -> None:
    """Register polymorphic configs before draccus parses ``robot.type``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
    from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
    from lerobot.utils.import_utils import register_third_party_plugins
    from lerobot_robot_rb.rb10 import Rb10Config  # noqa: F401

    register_third_party_plugins()
    async_ee_client()


if __name__ == "__main__":
    main()
