"""In-process latest-frame bus for camera images.

``Rby1.get_observation()`` publishes every camera frame it reads; a
teleoperator running in the same process (e.g. ``rby1_isaac`` with the
headset camera panels) reads the latest frame per camera without opening the
device a second time. Frames are stored by reference (no copy); consumers
must treat them as read-only.
"""

from __future__ import annotations

import threading
import time
from typing import Any


class FrameBus:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frames: dict[str, tuple[Any, float, int]] = {}
        self._seq = 0

    def publish(self, name: str, frame: Any, t: float | None = None) -> None:
        with self._lock:
            self._seq += 1
            self._frames[name] = (frame, time.monotonic() if t is None else t, self._seq)

    def latest(self, name: str) -> tuple[Any, float, int] | None:
        """``(frame, timestamp, seq)`` of the newest frame of ``name`` or None."""
        with self._lock:
            return self._frames.get(name)

    def names(self) -> list[str]:
        with self._lock:
            return list(self._frames)

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()


# Process-wide bus shared by the robot and teleoperator plugins.
bus = FrameBus()
