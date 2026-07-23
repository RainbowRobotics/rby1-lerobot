"""Meta Quest UDP receiver for RB-Series VR teleoperation.

This receiver preserves the JSON packet structure used by the original
hardware-tested RB10E teleoperation script:

{
    "hands": {
        "right": {
            "position": [x, y, z],
            "rotation": [qx, qy, qz, qw],
            "buttons": {
                "primaryButton": bool,
                "secondaryButton": bool,
                "trigger": float,
                "grip": float
            }
        },
        "left": {...}
    },
    "head": {
        "position": [x, y, z],
        "rotation": [qx, qy, qz, qw]
    }
}

Responsibilities
----------------
- Send the PC endpoint handshake to Meta Quest.
- Receive Quest JSON packets over UDP.
- Convert raw Quest poses into the original RB-oriented frame.
- Preserve button levels and expose rising-edge button events.
- Keep only the latest packet; no queue accumulation.
- Detect stale controller tracking.

User scaling, torso-relative conversion, IK, A-button initialisation and
Grip/ServoJ gating are handled by ``rb_vr.py``.
"""

from __future__ import annotations

import copy
import json
import logging
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .frame_transforms import quest_pose_to_rb


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State containers
# ---------------------------------------------------------------------------


@dataclass
class VRButtons:
    """Button levels from one Meta Quest controller."""

    primary: bool = False
    secondary: bool = False
    trigger: float = 0.0
    grip: float = 0.0


@dataclass
class VRControllerState:
    """Converted pose and buttons for one controller."""

    pose_rb: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float64)
    )
    buttons: VRButtons = field(default_factory=VRButtons)
    tracked: bool = False


@dataclass
class VRHeadState:
    """Converted Meta Quest headset pose."""

    pose_rb: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float64)
    )
    tracked: bool = False


@dataclass
class VRState:
    """Latest complete VR state snapshot."""

    right: VRControllerState = field(
        default_factory=VRControllerState
    )
    left: VRControllerState = field(
        default_factory=VRControllerState
    )
    head: VRHeadState = field(
        default_factory=VRHeadState
    )

    packet_monotonic_time: float = 0.0
    source_ip: str | None = None
    source_port: int | None = None
    packet_count: int = 0


@dataclass
class VRButtonEvents:
    """Rising-edge button events accumulated since the previous read."""

    right_primary: bool = False
    right_secondary: bool = False
    left_primary: bool = False
    left_secondary: bool = False


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _clamp_analog(value: Any) -> float:
    """Convert a controller analog value into [0, 1]."""

    try:
        converted = float(value)
    except (TypeError, ValueError):
        return 0.0

    if not np.isfinite(converted):
        return 0.0

    return float(np.clip(converted, 0.0, 1.0))


def _parse_buttons(raw: Any) -> VRButtons:
    """Parse the original Quest button mapping."""

    if not isinstance(raw, dict):
        return VRButtons()

    return VRButtons(
        primary=bool(
            raw.get(
                "primaryButton",
                False,
            )
        ),
        secondary=bool(
            raw.get(
                "secondaryButton",
                False,
            )
        ),
        trigger=_clamp_analog(
            raw.get(
                "trigger",
                0.0,
            )
        ),
        grip=_clamp_analog(
            raw.get(
                "grip",
                0.0,
            )
        ),
    )


def _parse_pose(
    raw: Any,
    *,
    name: str,
) -> np.ndarray:
    """Parse and convert one Quest pose into the RB-oriented frame."""

    if not isinstance(raw, dict):
        raise ValueError(
            f"{name} must be a JSON object."
        )

    if "position" not in raw:
        raise ValueError(
            f"{name} is missing 'position'."
        )

    if "rotation" not in raw:
        raise ValueError(
            f"{name} is missing 'rotation'."
        )

    return quest_pose_to_rb(
        position=raw["position"],
        rotation_quat=raw["rotation"],
    )


def _copy_controller(
    state: VRControllerState,
) -> VRControllerState:
    return VRControllerState(
        pose_rb=state.pose_rb.copy(),
        buttons=copy.copy(state.buttons),
        tracked=state.tracked,
    )


def _copy_state(
    state: VRState,
) -> VRState:
    return VRState(
        right=_copy_controller(state.right),
        left=_copy_controller(state.left),
        head=VRHeadState(
            pose_rb=state.head.pose_rb.copy(),
            tracked=state.head.tracked,
        ),
        packet_monotonic_time=state.packet_monotonic_time,
        source_ip=state.source_ip,
        source_port=state.source_port,
        packet_count=state.packet_count,
    )


# ---------------------------------------------------------------------------
# Receiver
# ---------------------------------------------------------------------------


class VRReceiver:
    """Latest-state Meta Quest UDP receiver."""

    def __init__(
        self,
        *,
        local_ip: str,
        local_port: int,
        meta_quest_ip: str | None,
        meta_quest_port: int = 6000,
        send_handshake: bool = True,
        tracking_timeout_s: float = 0.25,
        receive_buffer_bytes: int = 65535,
    ) -> None:
        if not local_ip:
            raise ValueError(
                "local_ip must not be empty."
            )

        if not 0 < int(local_port) <= 65535:
            raise ValueError(
                f"Invalid local_port: {local_port}."
            )

        if not 0 < int(meta_quest_port) <= 65535:
            raise ValueError(
                f"Invalid meta_quest_port: {meta_quest_port}."
            )

        if tracking_timeout_s <= 0.0:
            raise ValueError(
                "tracking_timeout_s must be positive."
            )

        if receive_buffer_bytes <= 0:
            raise ValueError(
                "receive_buffer_bytes must be positive."
            )

        if send_handshake:
            if not meta_quest_ip:
                raise ValueError(
                    "meta_quest_ip is required when "
                    "send_handshake=True."
                )

            if local_ip == "0.0.0.0":
                raise ValueError(
                    "A concrete local_ip is required when "
                    "send_handshake=True because Meta Quest must "
                    "receive the PC's reachable IP address."
                )

        self.local_ip = local_ip
        self.local_port = int(local_port)

        self.meta_quest_ip = meta_quest_ip
        self.meta_quest_port = int(
            meta_quest_port
        )

        self.send_handshake = bool(
            send_handshake
        )
        self.tracking_timeout_s = float(
            tracking_timeout_s
        )
        self.receive_buffer_bytes = int(
            receive_buffer_bytes
        )

        self._state = VRState()
        self._events = VRButtonEvents()

        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._first_packet_event = threading.Event()

        self._thread: threading.Thread | None = None
        self._socket: socket.socket | None = None
        self._thread_error: Exception | None = None

        self._previous_buttons = {
            "right_primary": False,
            "right_secondary": False,
            "left_primary": False,
            "left_secondary": False,
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return (
            self._thread is not None
            and self._thread.is_alive()
            and not self._stop_event.is_set()
        )

    @property
    def thread_error(self) -> Exception | None:
        return self._thread_error

    @property
    def has_received_packet(self) -> bool:
        return self._first_packet_event.is_set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self.is_running:
            raise RuntimeError(
                "VRReceiver is already running."
            )

        self._stop_event.clear()
        self._first_packet_event.clear()
        self._thread_error = None

        sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        )

        sock.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1,
        )

        # A short timeout allows stop() to terminate the thread without
        # waiting for another Quest packet.
        sock.settimeout(0.1)

        try:
            sock.bind(
                (
                    self.local_ip,
                    self.local_port,
                )
            )
        except Exception:
            sock.close()
            raise

        self._socket = sock

        if self.send_handshake:
            self._send_endpoint_handshake()

        self._thread = threading.Thread(
            target=self._receive_loop,
            name="rb-vr-udp-receiver",
            daemon=True,
        )
        self._thread.start()

        logger.info(
            "VR UDP receiver listening on %s:%d.",
            self.local_ip,
            self.local_port,
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

        if (
            thread is not None
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=1.0)

        self._thread = None

        logger.info(
            "VR UDP receiver stopped."
        )

    def _send_endpoint_handshake(self) -> None:
        if self.meta_quest_ip is None:
            raise RuntimeError(
                "meta_quest_ip is not configured."
            )

        target_info = {
            "ip": self.local_ip,
            "port": self.local_port,
        }

        message = json.dumps(
            target_info
        ).encode("utf-8")

        # Use a temporary socket, matching the original implementation.
        with socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        ) as handshake_socket:
            handshake_socket.sendto(
                message,
                (
                    self.meta_quest_ip,
                    self.meta_quest_port,
                ),
            )

        logger.info(
            "Sent PC endpoint to Meta Quest: %s",
            target_info,
        )

    # ------------------------------------------------------------------
    # Public reads
    # ------------------------------------------------------------------

    def wait_for_first_packet(
        self,
        timeout_s: float,
    ) -> bool:
        if timeout_s <= 0.0:
            raise ValueError(
                "timeout_s must be positive."
            )

        ready = self._first_packet_event.wait(
            timeout_s
        )

        self._raise_thread_error()
        return ready

    def get_state(
        self,
        *,
        require_fresh: bool = False,
    ) -> VRState:
        """Return a copied snapshot of the latest VR state."""

        self._raise_thread_error()

        with self._state_lock:
            state = _copy_state(
                self._state
            )

        if require_fresh and self.is_stale(
            state=state
        ):
            raise TimeoutError(
                "Meta Quest tracking data is stale."
            )

        return state

    def consume_button_events(
        self,
    ) -> VRButtonEvents:
        """Return and clear accumulated rising-edge button events."""

        self._raise_thread_error()

        with self._state_lock:
            events = copy.copy(
                self._events
            )
            self._events = VRButtonEvents()

        return events

    def is_stale(
        self,
        *,
        state: VRState | None = None,
    ) -> bool:
        if state is None:
            with self._state_lock:
                packet_time = (
                    self._state.packet_monotonic_time
                )
        else:
            packet_time = (
                state.packet_monotonic_time
            )

        if packet_time <= 0.0:
            return True

        return (
            time.monotonic() - packet_time
            > self.tracking_timeout_s
        )

    def tracking_age_s(self) -> float:
        with self._state_lock:
            packet_time = (
                self._state.packet_monotonic_time
            )

        if packet_time <= 0.0:
            return float("inf")

        return max(
            0.0,
            time.monotonic() - packet_time,
        )

    def _raise_thread_error(self) -> None:
        if self._thread_error is not None:
            raise RuntimeError(
                "VR UDP receiver failed."
            ) from self._thread_error

    # ------------------------------------------------------------------
    # Receive loop
    # ------------------------------------------------------------------

    def _receive_loop(self) -> None:
        sock = self._socket

        if sock is None:
            return

        try:
            while not self._stop_event.is_set():
                try:
                    payload, address = sock.recvfrom(
                        self.receive_buffer_bytes
                    )
                except socket.timeout:
                    continue
                except OSError:
                    if self._stop_event.is_set():
                        return
                    raise

                source_ip, source_port = address

                # Ignore unrelated UDP traffic when a Quest address is
                # explicitly configured.
                if (
                    self.meta_quest_ip is not None
                    and source_ip != self.meta_quest_ip
                ):
                    logger.debug(
                        "Ignored UDP packet from unexpected source %s:%d.",
                        source_ip,
                        source_port,
                    )
                    continue

                try:
                    decoded = payload.decode(
                        "utf-8"
                    )
                    packet = json.loads(
                        decoded
                    )
                    self._handle_packet(
                        packet,
                        source_ip=source_ip,
                        source_port=source_port,
                    )
                except (
                    UnicodeDecodeError,
                    json.JSONDecodeError,
                    TypeError,
                    ValueError,
                    KeyError,
                ) as exc:
                    logger.warning(
                        "Ignored invalid Meta Quest packet from %s:%d: %s",
                        source_ip,
                        source_port,
                        exc,
                    )

        except Exception as exc:
            if not self._stop_event.is_set():
                self._thread_error = exc
                logger.exception(
                    "VR UDP receiver stopped unexpectedly: %s",
                    exc,
                )
                self._first_packet_event.set()

    def _handle_packet(
        self,
        packet: Any,
        *,
        source_ip: str,
        source_port: int,
    ) -> None:
        if not isinstance(packet, dict):
            raise ValueError(
                "Quest packet root must be a JSON object."
            )

        hands = packet.get(
            "hands",
            {},
        )

        if hands is None:
            hands = {}

        if not isinstance(hands, dict):
            raise ValueError(
                "'hands' must be a JSON object."
            )

        right_raw = hands.get(
            "right"
        )
        left_raw = hands.get(
            "left"
        )
        head_raw = packet.get(
            "head"
        )

        with self._state_lock:
            state = self._state

            # ---------------- Right controller ----------------
            if right_raw is not None:
                state.right.pose_rb = _parse_pose(
                    right_raw,
                    name="hands.right",
                )
                state.right.buttons = _parse_buttons(
                    right_raw.get(
                        "buttons",
                        {},
                    )
                )
                state.right.tracked = True
            else:
                state.right.tracked = False
                state.right.buttons = VRButtons()

            # ---------------- Left controller -----------------
            if left_raw is not None:
                state.left.pose_rb = _parse_pose(
                    left_raw,
                    name="hands.left",
                )
                state.left.buttons = _parse_buttons(
                    left_raw.get(
                        "buttons",
                        {},
                    )
                )
                state.left.tracked = True
            else:
                state.left.tracked = False
                state.left.buttons = VRButtons()

            # ---------------- Head ----------------------------
            if head_raw is not None:
                state.head.pose_rb = _parse_pose(
                    head_raw,
                    name="head",
                )
                state.head.tracked = True
            else:
                state.head.tracked = False

            state.packet_monotonic_time = (
                time.monotonic()
            )
            state.source_ip = source_ip
            state.source_port = int(
                source_port
            )
            state.packet_count += 1

            self._update_button_events_locked(
                state
            )

        self._first_packet_event.set()

    def _update_button_events_locked(
        self,
        state: VRState,
    ) -> None:
        current = {
            "right_primary": (
                state.right.tracked
                and state.right.buttons.primary
            ),
            "right_secondary": (
                state.right.tracked
                and state.right.buttons.secondary
            ),
            "left_primary": (
                state.left.tracked
                and state.left.buttons.primary
            ),
            "left_secondary": (
                state.left.tracked
                and state.left.buttons.secondary
            ),
        }

        if (
            current["right_primary"]
            and not self._previous_buttons[
                "right_primary"
            ]
        ):
            self._events.right_primary = True

        if (
            current["right_secondary"]
            and not self._previous_buttons[
                "right_secondary"
            ]
        ):
            self._events.right_secondary = True

        if (
            current["left_primary"]
            and not self._previous_buttons[
                "left_primary"
            ]
        ):
            self._events.left_primary = True

        if (
            current["left_secondary"]
            and not self._previous_buttons[
                "left_secondary"
            ]
        ):
            self._events.left_secondary = True

        self._previous_buttons = current

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> VRReceiver:
        self.start()
        return self

    def __exit__(
        self,
        exc_type,
        exc,
        traceback,
    ) -> None:
        self.stop()
