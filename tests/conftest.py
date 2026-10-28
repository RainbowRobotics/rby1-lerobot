"""Hardware-free test setup: shadow the real rby1_sdk with tests/fakes/rby1_sdk.

The fake must be importable before any plugin module imports ``rby1_sdk``, so
the path is prepended at collection time rather than in a fixture.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

FAKES_DIR = Path(__file__).parent / "fakes"
sys.path.insert(0, str(FAKES_DIR))
os.environ.setdefault("RBY1_FAKE_SUMMARY", "0")

import pytest  # noqa: E402
import rby1_sdk  # noqa: E402

assert getattr(rby1_sdk, "FAKE", False), f"real rby1_sdk imported from {rby1_sdk.__file__}"

ADDRESS = "stub.invalid:1"


@pytest.fixture(scope="session", autouse=True)
def _register_plugins() -> None:
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()


@pytest.fixture(scope="module")
def connected_robot():
    """A Rby1 (Model A, v1.2) connected to the fake SDK; connect() takes ~6 s."""
    from lerobot_robot_rby1 import Rby1, Rby1Config

    robot = Rby1(Rby1Config(address=ADDRESS, model="a", version="1.2"))
    robot.connect()
    yield robot
    robot.disconnect()
