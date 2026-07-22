"""Background UDP receiver for Meta Quest teleoperation.

The receiver owns only VR communication and parsing. It does not perform
inverse kinematics and never commands the robot.

Unlike the legacy multiprocessing Queue implementation, this receiver keeps
only the latest VR state. Old controller poses therefore cannot accumulate
and be replayed later when the consumer is slower than the Quest sender.

Expected packet shape
---------------------
{
    "timestamp": ...,
    "head": {
        "position": [x, y, z],
        "rotation": [qx, qy, qz, qw]
    },
    "hands": {
        "right": {
            "position": [x, y, z],
            "rotation": [qx, qy, qz, qw],
            "buttons": {
                "primaryButton": false,
                "secondaryButton": false,
                "trigger": 0.0,
                "grip": 0.0
            }
        },
        "left": {...}
    }
}

Quest positions are converted from metres into RB-frame SE(3) transforms
whose translations are in millimetres.
"""

from __future__ import annotations

import copy
import json
import logging
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from .constants import (
    GRIP_BUTTON,
    LEFT_HAND,
    PRIMARY_BUTTON,
    RIGHT_HAND,
    SECONDARY_BUTTON,
    TRIGGER_BUTTON,
)
from .frame_transforms import quest_pose_to_rb_frame


logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ControllerSnapshot:
    """Parsed state of one Meta Quest controller."""

    tracked: bool
    pose_rb: FloatArray | None
    primary_pressed: bool
    secondary_pressed: bool
    trigger: float
    grip: float


@dataclass(frozen=True)
class VRSnapshot:
    """Latest complete VR state received from Meta Quest."""

    received_at: float
    source_timestamp: float | None
    sender: tuple[str, int] | None

    head_tracked: bool
    head_pose_rb: FloatArray | None

    right: ControllerSnapshot
    left: ControllerSnapshot


def _empty_controller() -> ControllerSnapshot:
    return ControllerSnapshot(
        tracked=False,
        pose_rb=None,
        primary_pressed=False,
        secondary_pressed=False,
        trigger=0.0,
        grip=0.0,
    )


def _normalised_axis(value: Any, *, name: str) -> float:
    """Convert one analog button value into the [0, 1] range."""

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}.") from exc

    if not np.isfinite(result):
        raise ValueError(f"{name} contains NaN or Inf.")

    return float(np.clip(result, 0.0, 1.0))


def _parse_pose(entity: Mapping[str, Any], *, name: str) -> FloatArray:
    try:
        position = entity["position"]
        rotation = entity["rotation"]
    except KeyError as exc:
        raise ValueError(
            f"{name} must contain position and rotation."
        ) from exc

    return quest_pose_to_rb_frame(
        position,
        rotation,
    )


def _parse_controller(
    controller: Any,
    *,
    name: str,
) -> ControllerSnapshot:
    if controller is None:
        return _empty_controller()

    if not isinstance(controller, Mapping):
        raise ValueError(
            f"{name} controller must be an object or null."
        )

    pose = _parse_pose(controller, name=f"{name} controller")

    buttons = controller.get("buttons", {})
    if not isinstance(buttons, Mapping):
        raise ValueError(
            f"{name} controller buttons must be an object."
        )

    return ControllerSnapshot(
        tracked=True,
        pose_rb=pose,
        primary_pressed=bool(buttons.get(PRIMARY_BUTTON, False)),
        secondary_pressed=bool(buttons.get(SECONDARY_BUTTON, False)),
        trigger=_normalised_axis(
            buttons.get(TRIGGER_BUTTON, 0.0),
            name=f"{name} trigger",
        ),
        grip=_normalised_axis(
            buttons.get(GRIP_BUTTON, 0.0),
            name=f"{name} grip",
        ),
    )


def parse_vr_payload(
    payload: Mapping[str, Any],
    *,
    received_at: float | None = None,
    sender: tuple[str, int] | None = None,
) -> VRSnapshot:
    """Parse one decoded Meta Quest JSON payload.

    This is a pure function so packet parsing can be unit-tested without
    opening a UDP socket.
    """

    if not isinstance(payload, Mapping):
        raise ValueError("VR payload must be a JSON object.")

    if received_at is None:
        received_at = time.monotonic()

    source_timestamp_raw = payload.get("timestamp")
    source_timestamp: float | None

    if source_timestamp_raw is None:
        source_timestamp = None
    else:
        try:
            source_timestamp = float(source_timestamp_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "VR timestamp must be numeric or null."
            ) from exc

        if not np.isfinite(source_timestamp):
            raise ValueError("VR timestamp contains NaN or Inf.")

    hands = payload.get("hands", {})
    if hands is None:
        hands = {}

    if not isinstance(hands, Mapping):
        raise ValueError("VR hands field must be an object or null.")

    right = _parse_controller(
        hands.get(RIGHT_HAND),
        name=RIGHT_HAND,
    )
    left = _parse_controller(
        hands.get(LEFT_HAND),
        name=LEFT_HAND,
    )

    head_raw = payload.get("head")
    if head_raw is None:
        head_tracked = False
        head_pose = None
    else:
        if not isinstance(head_raw, Mapping):
            raise ValueError("VR head field must be an object or null.")

        head_pose = _parse_pose(head_raw, name="head")
        head_tracked = True

    return VRSnapshot(
        received_at=float(received_at),
        source_timestamp=source_timestamp,
        sender=sender,
        head_tracked=head_tracked,
        head_pose_rb=head_pose,
        right=right,
        left=left,
    )


def _copy_controller(
    controller: ControllerSnapshot,
) -> ControllerSnapshot:
    return ControllerSnapshot(
        tracked=controller.tracked,
        pose_rb=(
            None
            if controller.pose_rb is None
            else controller.pose_rb.copy()
        ),
        primary_pressed=controller.primary_pressed,
        secondary_pressed=controller.secondary_pressed,
        trigger=controller.trigger,
        grip=controller.grip,
    )


def copy_snapshot(snapshot: VRSnapshot) -> VRSnapshot:
    """Return a defensive copy of a VR snapshot."""

    return VRSnapshot(
        received_at=snapshot.received_at,
        source_timestamp=snapshot.source_timestamp,
        sender=snapshot.sender,
        head_tracked=snapshot.head_tracked,
        head_pose_rb=(
            None
            if snapshot.head_pose_rb is None
            else snapshot.head_pose_rb.copy()
        ),
        right=_copy_controller(snapshot.right),
        left=_copy_controller(snapshot.left),
    )


class VRReceiver:
    """Receive and retain the latest Meta Quest controller state."""

    def __init__(
        self,
        *,
        local_ip: str,
        local_port: int,
        meta_quest_ip: str | None = None,
        meta_quest_port: int = 6000,
        send_handshake: bool = True,
        socket_timeout_s: float = 0.1,
        receive_buffer_bytes: int = 65535,
    ) -> None:
        if not local_ip:
            raise ValueError("local_ip must not be empty.")

        if not 0 <= int(local_port) <= 65535:
            raise ValueError(
                f"local_port must be in [0, 65535], got {local_port}."
            )

        if not 0 <= int(meta_quest_port) <= 65535:
            raise ValueError(
                "meta_quest_port must be in [0, 65535], "
                f"got {meta_quest_port}."
            )

        if socket_timeout_s <= 0.0:
            raise ValueError("socket_timeout_s must be positive.")

        if receive_buffer_bytes <= 0:
            raise ValueError("receive_buffer_bytes must be positive.")

        if send_handshake and meta_quest_ip is None:
            raise ValueError(
                "meta_quest_ip is required when send_handshake=True."
            )

        if send_handshake and local_ip == "0.0.0.0":
            raise ValueError(
                "An actual LAN/Wi-Fi local_ip is required for the Quest "
                "handshake; 0.0.0.0 cannot be advertised to Meta Quest."
            )

        self._local_ip = local_ip
        self._local_port = int(local_port)
        self._meta_quest_ip = meta_quest_ip
        self._meta_quest_port = int(meta_quest_port)
        self._send_handshake = bool(send_handshake)
        self._socket_timeout_s = float(socket_timeout_s)
        self._receive_buffer_bytes = int(receive_buffer_bytes)

        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._lock = threading.Lock()
        self._latest_state: VRSnapshot | None = None

        self._button_down = {
            "right_primary": False,
            "right_secondary": False,
            "left_primary": False,
            "left_secondary": False,
        }
        self._pending_button_events = {
            "right_primary": False,
            "right_secondary": False,
            "left_primary": False,
            "left_secondary": False,
        }

        self._packet_count = 0
        self._parse_error_count = 0
        self._last_error: str | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return (
            thread is not None
            and thread.is_alive()
            and not self._stop_event.is_set()
        )

    @property
    def bound_port(self) -> int | None:
        sock = self._socket
        if sock is None:
            return None
        return int(sock.getsockname()[1])

    @property
    def packet_count(self) -> int:
        with self._lock:
            return self._packet_count

    @property
    def parse_error_count(self) -> int:
        with self._lock:
            return self._parse_error_count

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    @property
    def has_state(self) -> bool:
        with self._lock:
            return self._latest_state is not None

    @property
    def age_s(self) -> float:
        """Age of the latest valid packet, or infinity before first packet."""

        with self._lock:
            latest = self._latest_state

        if latest is None:
            return float("inf")

        return max(0.0, time.monotonic() - latest.received_at)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self.is_running:
            raise RuntimeError("VRReceiver is already running.")

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self._local_ip, self._local_port))
        sock.settimeout(self._socket_timeout_s)

        self._socket = sock
        self._stop_event.clear()

        if self._send_handshake:
            self._send_quest_handshake()

        self._thread = threading.Thread(
            target=self._receive_loop,
            name="rb-vr-receiver",
            daemon=True,
        )
        self._thread.start()

        logger.info(
            "VR UDP receiver listening on %s:%d.",
            self._local_ip,
            self.bound_port,
        )

    def stop(self) -> None:
        self._stop_event.set()

        sock = self._socket
        self._socket = None

        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)

            if thread.is_alive():
                logger.warning(
                    "VR receiver thread did not stop within one second."
                )

        self._thread = None
        logger.info("VR UDP receiver stopped.")

    def _send_quest_handshake(self) -> None:
        if self._socket is None:
            raise RuntimeError(
                "Cannot send Quest handshake before binding UDP socket."
            )

        if self._meta_quest_ip is None:
            raise RuntimeError(
                "meta_quest_ip is missing for Quest handshake."
            )

        target_info = {
            "ip": self._local_ip,
            "port": self.bound_port,
        }
        message = json.dumps(target_info).encode("utf-8")

        self._socket.sendto(
            message,
            (
                self._meta_quest_ip,
                self._meta_quest_port,
            ),
        )

        logger.info(
            "Sent PC endpoint to Meta Quest: %s",
            target_info,
        )

    # ------------------------------------------------------------------
    # Consumer API
    # ------------------------------------------------------------------

    def get_state(self) -> VRSnapshot | None:
        """Return the latest state without blocking."""

        with self._lock:
            latest = self._latest_state

            if latest is None:
                return None

            return copy_snapshot(latest)

    def is_stale(self, timeout_s: float) -> bool:
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive.")

        return self.age_s > timeout_s

    def wait_for_first_packet(
        self,
        *,
        timeout_s: float,
    ) -> bool:
        """Wait for initial VR data, returning False on timeout."""

        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive.")

        deadline = time.monotonic() + timeout_s

        while time.monotonic() < deadline:
            if self.has_state:
                return True

            if not self.is_running:
                return False

            time.sleep(0.01)

        return self.has_state

    def consume_button_events(self) -> dict[str, bool]:
        """Return and clear pending rising-edge button events."""

        with self._lock:
            result = copy.copy(self._pending_button_events)

            for name in self._pending_button_events:
                self._pending_button_events[name] = False

        return result

    # ------------------------------------------------------------------
    # Internal receive path
    # ------------------------------------------------------------------

    def _receive_loop(self) -> None:
        while not self._stop_event.is_set():
            sock = self._socket
            if sock is None:
                return

            try:
                data, sender = sock.recvfrom(
                    self._receive_buffer_bytes
                )
            except socket.timeout:
                continue
            except OSError as exc:
                if not self._stop_event.is_set():
                    logger.error("VR UDP receive error: %s", exc)
                return

            received_at = time.monotonic()

            try:
                decoded = data.decode("utf-8")
                payload = json.loads(decoded)

                snapshot = parse_vr_payload(
                    payload,
                    received_at=received_at,
                    sender=sender,
                )
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
                ValueError,
                TypeError,
            ) as exc:
                with self._lock:
                    self._parse_error_count += 1
                    self._last_error = str(exc)

                logger.warning(
                    "Discarded invalid VR packet from %s: %s",
                    sender,
                    exc,
                )
                continue

            self._accept_snapshot(snapshot)

    def _accept_snapshot(self, snapshot: VRSnapshot) -> None:
        current_buttons = {
            "right_primary": snapshot.right.primary_pressed,
            "right_secondary": snapshot.right.secondary_pressed,
            "left_primary": snapshot.left.primary_pressed,
            "left_secondary": snapshot.left.secondary_pressed,
        }

        with self._lock:
            for name, is_pressed in current_buttons.items():
                was_pressed = self._button_down[name]

                if is_pressed and not was_pressed:
                    self._pending_button_events[name] = True

                self._button_down[name] = is_pressed

            self._latest_state = copy_snapshot(snapshot)
            self._packet_count += 1
            self._last_error = None
