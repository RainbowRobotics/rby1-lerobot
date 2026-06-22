"""Thread-safe UDP receiver for Meta Quest controller data.

The companion Meta Quest app streams JSON packets to ``local_ip:local_port``
at high rate.  This module starts a background thread that decodes those
packets, keeps the latest snapshot under a lock, and detects rising-edge
button events (A / B / X / Y) so that consumers don't miss presses between
ticks.

Packet schema::

    {
      "hands": {
        "right": {
          "position": [x, y, z],
          "rotation": [qx, qy, qz, qw],
          "buttons": {
            "trigger":       0.0-1.0,
            "grip":          0.0-1.0,
        "thumbstickAxis": [x, y],
        "primaryButton":  bool,     (A on right, X on left)
        "secondaryButton": bool     (B on right, Y on left)
      }
    },
    "left": { ... }
  },
  "head": {
    "position": [x, y, z],
    "rotation": [qx, qy, qz, qw]
  }
}
"""

from __future__ import annotations

import json
import logging
import socket
import threading
from typing import Any

logger = logging.getLogger(__name__)


class VRReceiver:
    """
    Sends a handshake UDP packet to the Meta Quest so it knows where to stream
    controller data, then listens on local_port for incoming JSON packets.

    Thread-safe: get_state() / consume_button_events() may be called from any
    thread while the background receiver thread is running.
    """

    def __init__(
        self,
        local_ip: str,
        local_port: int,
        meta_quest_ip: str,
        meta_quest_port: int,
    ) -> None:
        self.local_ip = local_ip
        self.local_port = local_port
        self.meta_quest_ip = meta_quest_ip
        self.meta_quest_port = meta_quest_port

        self._lock = threading.Lock()
        self._controller_state: dict[str, Any] = {}

        # Latched button events (set True on press, consumed once)
        self._event_right_a: bool = False
        self._event_right_b: bool = False
        self._event_left_a: bool = False
        self._event_left_b: bool = False

        self._running = False
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Send handshake to Meta Quest and start background receiver thread."""
        self._send_handshake()
        self._running = True
        self._thread = threading.Thread(target=self._udp_loop, daemon=True)
        self._thread.start()
        logger.info(
            f"VRReceiver started — listening on {self.local_ip}:{self.local_port}"
        )

    def stop(self) -> None:
        """Stop the receiver thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("VRReceiver stopped.")

    # ------------------------------------------------------------------ #
    #  Data access (thread-safe)                                           #
    # ------------------------------------------------------------------ #

    def get_state(self) -> dict[str, Any]:
        """Return a snapshot of the latest controller state dict."""
        with self._lock:
            return dict(self._controller_state)

    def consume_button_events(self) -> dict[str, bool]:
        """
        Atomically read and clear all latched button events.

        Returns
        -------
        dict with keys: right_a, right_b, left_a, left_b
        """
        with self._lock:
            events = {
                "right_a": self._event_right_a,
                "right_b": self._event_right_b,
                "left_a":  self._event_left_a,
                "left_b":  self._event_left_b,
            }
            self._event_right_a = False
            self._event_right_b = False
            self._event_left_a  = False
            self._event_left_b  = False
        return events

    # ------------------------------------------------------------------ #
    #  Internal                                                             #
    # ------------------------------------------------------------------ #

    def _send_handshake(self) -> None:
        """Tell the Meta Quest where to send controller data."""
        target_info = {"ip": self.local_ip, "port": self.local_port}
        message = json.dumps(target_info).encode("utf-8")
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.sendto(message, (self.meta_quest_ip, self.meta_quest_port))
        logger.info(
            f"Handshake sent to Meta Quest at "
            f"{self.meta_quest_ip}:{self.meta_quest_port} → {target_info}"
        )

    def _udp_loop(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.settimeout(1.0)
            srv.bind(("0.0.0.0", self.local_port))
            logger.info(
                f"UDP server bound to 0.0.0.0:{self.local_port}"
            )
            while self._running:
                try:
                    data, _ = srv.recvfrom(4096)
                except socket.timeout:
                    continue
                except Exception as exc:
                    logger.warning(f"UDP receive error: {exc}")
                    continue

                try:
                    parsed: dict = json.loads(data.decode("utf-8"))
                except json.JSONDecodeError:
                    continue

                with self._lock:
                    self._controller_state = parsed
                    self._parse_button_events(parsed)

    def _parse_button_events(self, state: dict) -> None:
        """Latch edge-triggered button presses (call under self._lock)."""
        hands = state.get("hands", {})

        left = hands.get("left", {})
        if left:
            btns = left.get("buttons", {})
            if btns.get("primaryButton", False):
                self._event_left_a = True
            if btns.get("secondaryButton", False):
                self._event_left_b = True

        right = hands.get("right", {})
        if right:
            btns = right.get("buttons", {})
            if btns.get("primaryButton", False):
                self._event_right_a = True
            if btns.get("secondaryButton", False):
                self._event_right_b = True
