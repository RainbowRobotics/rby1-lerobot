"""Dedicated asynchronous end-effector inference client for RB10/RB10-E.

The LeRobot robot adapter is always ``rb10`` because both physical variants
share the RB ServoJ driver.  ``kinematics_model`` selects the model used by
the policy adapter for FK/IK.

Safety defaults are deliberately conservative: the client starts paused and
in dry-run mode, and it never moves to a ready pose.  Press ``f`` to start
inference, ``s`` to pause and invalidate in-flight work, or ``q`` to quit.
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
    def get_observation(self) -> dict[str, Any]: ...

    def send_action(self, action: dict[str, float]) -> dict[str, Any]: ...

    def enable_servo_commands(self) -> None: ...

    def disable_servo_commands(self) -> None: ...


class AdapterLike(Protocol):
    def validate_robot(self, robot: RobotLike) -> None: ...

    def build_observation(self, raw: dict[str, Any], task: str) -> dict[str, Any]: ...

    def joint_action(
        self,
        action7: np.ndarray,
        current_q6: np.ndarray,
        *,
        max_joint_delta: float,
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
    timeout_ms: int = 1000
    reconnect_delay_s: float = 0.25
    dry_run: bool = True

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
            "reconnect_delay_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0, got {value}")
        if not isinstance(self.timeout_ms, int) or not 0 < self.timeout_ms <= 1000:
            raise ValueError("timeout_ms must be an integer in [1, 1000]")
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
        self._action_chunk: _ActionChunk | None = None
        self._disable_requested = True
        self._last_pause_reason = "startup"

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
            if not self._network_ready:
                self._last_pause_reason = "server not connected; press f after reconnect"
                return False
            self._paused = True
            self._action_chunk = None
            self._last_command_q = None
            generation_before_prepare = self._generation

        try:
            if self.config.dry_run:
                self.robot.disable_servo_commands()
            else:
                LOGGER.warning(
                    "First real activation may initialize impedance control and open the configured gripper"
                )
                self.robot.enable_servo_commands()
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
            self.pause("operator pause")
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
            if self._paused or self._quit:
                return None
            generation = self._generation

        now = self._clock()
        try:
            raw = self._snapshot_observation(self.robot.get_observation())
            current_q = self._joint_vector(raw)
            wire = self.adapter.build_observation(raw, self.config.task)
        except Exception as exc:
            self._fault("observation/FK fault", exc)
            return None

        self._publish_observation(wire, now, generation)

        with self._state_lock:
            chunk = self._action_chunk
        if chunk is None or chunk.generation != generation:
            return None

        action_index = int(max(0.0, now - chunk.observed_at) / self.config.nominal_dt)
        if action_index >= len(chunk.actions):
            self._fault("action chunk horizon expired")
            return None

        action7 = chunk.actions[action_index].copy()
        try:
            command = self.adapter.joint_action(
                action7,
                current_q,
                max_joint_delta=self.config.max_joint_delta,
            )
            target_q = np.asarray([command[key] for key in JOINT_KEYS], dtype=np.float64)
            command_values = np.asarray(list(command.values()), dtype=np.float64)
            if target_q.shape != (6,) or not np.all(np.isfinite(command_values)):
                raise ValueError("IK returned a malformed or non-finite joint action")
            with self._state_lock:
                previous_q = None if self._last_command_q is None else self._last_command_q.copy()
            if previous_q is not None and np.any(
                np.abs(target_q - previous_q) > self.config.max_joint_delta + 1e-12
            ):
                raise ValueError("command-to-command joint delta exceeds the configured limit")
        except Exception as exc:
            self._fault("IK/action safety fault", exc)
            return None

        # Network phase can place this tick arbitrarily close to the next
        # nominal action-index boundary.  Give IK one full control period from
        # this acquisition tick, while never extending validity beyond the
        # complete chunk horizon.
        chunk_deadline = chunk.observed_at + len(chunk.actions) * self.config.nominal_dt
        command_deadline = min(now + self.config.nominal_dt, chunk_deadline)
        if self._clock() > command_deadline:
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
        robot.connect()  # type: ignore[attr-defined]
        connected = True
        controller.initialize_robot_safety()
        controller.start_network_worker()
        LOGGER.info("RB10 EE client paused. Keys: f=activate, s=pause, q=quit")
        with _Keyboard() as keyboard:
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
    from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
    from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
    from lerobot.utils.import_utils import register_third_party_plugins
    from lerobot_robot_rb.rb10 import Rb10Config  # noqa: F401

    register_third_party_plugins()
    async_ee_client()


if __name__ == "__main__":
    main()
