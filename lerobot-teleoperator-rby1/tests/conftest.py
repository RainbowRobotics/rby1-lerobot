"""Test fixtures.

The pure modules under test do not need LeRobot, but the package ``__init__``
imports it. When ``lerobot`` is not installed (e.g. a plain dev box) a minimal
stub of the three symbols the package needs is installed so the tests can run
offline; on a real LeRobot environment the stub is never used.
"""

from __future__ import annotations

import abc
import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path


def _install_lerobot_stub() -> None:
    lerobot = types.ModuleType("lerobot")
    teleoperators = types.ModuleType("lerobot.teleoperators")
    teleoperator = types.ModuleType("lerobot.teleoperators.teleoperator")
    config = types.ModuleType("lerobot.teleoperators.config")
    utils = types.ModuleType("lerobot.utils")
    errors = types.ModuleType("lerobot.utils.errors")

    @dataclass(kw_only=True)
    class TeleoperatorConfig:
        id: str | None = None
        calibration_dir: Path | None = None
        _registry: dict = None  # type: ignore[assignment]

        @classmethod
        def register_subclass(cls, name):
            def deco(sub):
                TeleoperatorConfig._registry = TeleoperatorConfig._registry or {}
                TeleoperatorConfig._registry[name] = sub
                return sub

            return deco

    class Teleoperator(abc.ABC):
        def __init__(self, cfg):
            self.id = cfg.id
            self.calibration = {}

        def __str__(self):
            return f"{self.id} {self.__class__.__name__}"

    class DeviceAlreadyConnectedError(ConnectionError):
        pass

    class DeviceNotConnectedError(ConnectionError):
        pass

    config.TeleoperatorConfig = TeleoperatorConfig
    teleoperator.Teleoperator = Teleoperator
    errors.DeviceAlreadyConnectedError = DeviceAlreadyConnectedError
    errors.DeviceNotConnectedError = DeviceNotConnectedError
    lerobot.teleoperators = teleoperators
    lerobot.utils = utils
    teleoperators.teleoperator = teleoperator
    teleoperators.config = config
    utils.errors = errors
    for name, mod in {
        "lerobot": lerobot,
        "lerobot.teleoperators": teleoperators,
        "lerobot.teleoperators.teleoperator": teleoperator,
        "lerobot.teleoperators.config": config,
        "lerobot.utils": utils,
        "lerobot.utils.errors": errors,
    }.items():
        sys.modules[name] = mod


if importlib.util.find_spec("lerobot") is None:
    _install_lerobot_stub()

# Make the package importable from a source checkout without installation.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
