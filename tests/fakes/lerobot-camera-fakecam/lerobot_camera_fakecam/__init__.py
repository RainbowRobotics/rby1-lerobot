"""Hardware-free LeRobot camera (``--robot.cameras='{"front": {"type": "fake", ...}}'``).

Returns a synthetic moving-gradient RGB frame; never opens a device.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import CameraConfig


@CameraConfig.register_subclass("fake")
@dataclass
class FakeCameraConfig(CameraConfig):
    fps: int | None = 30
    width: int | None = 320
    height: int | None = 240


class FakeCamera(Camera):
    def __init__(self, config: FakeCameraConfig):
        super().__init__(config)
        self.config = config
        self._connected = False
        self._i = 0

    @property
    def is_connected(self) -> bool:
        return self._connected

    @staticmethod
    def find_cameras():
        return [{"name": "fake", "type": "fake", "id": 0}]

    def connect(self, warmup: bool = True) -> None:
        self._connected = True

    def read(self, *a, **k):
        self._i += 1
        h, w = self.config.height, self.config.width
        f = np.zeros((h, w, 3), dtype=np.uint8)
        f[..., 0] = (np.arange(w)[None, :] + self._i) % 256
        f[..., 1] = (np.arange(h)[:, None] * 2) % 256
        f[..., 2] = (self._i * 3) % 256
        return f

    def async_read(self, timeout_ms: float = 200, *a, **k):
        return self.read()

    def read_latest(self, *a, **k):
        return self.read()

    def disconnect(self) -> None:
        self._connected = False
