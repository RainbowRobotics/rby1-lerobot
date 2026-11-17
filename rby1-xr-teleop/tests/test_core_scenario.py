"""The default session through ``XrTeleopCore`` alone: no LeRobot, no XR runtime, no robot SDK."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rby1_xr_teleop.core import XrTeleopCore
from rby1_xr_teleop.scripted import FollowingRobot
from rby1_xr_teleop.settings import XrTeleopSettings
from rby1_xr_teleop.state import RobotInfo, RobotState
from scenario import DT, ScenarioSession, compare_with_trace, run_default_scenario, trace_of

# In the repository the same scenario is recorded through the LeRobot device.
SNAPSHOT = Path(__file__).resolve().parents[2] / "tests" / "snapshots" / "isaac_scenario_trace.json"


def _se3(x: float, y: float, z: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


def initial_state() -> RobotState:
    return RobotState(
        torso=_se3(0.0, 0.0, 1.0), right_ee=_se3(0.4, -0.2, 0.9), left_ee=_se3(0.4, 0.2, 0.9),
        head_q=np.array([0.0, 0.85]), right_q=np.zeros(7), left_q=np.zeros(7), odometry=np.eye(3),
    )


class CoreSession(ScenarioSession):
    def __init__(self, **settings) -> None:
        super().__init__()
        self.cfg = XrTeleopSettings(**settings)
        self.robot = FollowingRobot(initial_state(), dt=DT)
        self.core = XrTeleopCore(self.cfg, RobotInfo(version="1.3", arm_reach={"right": 0.66, "left": 0.66}))
        self.core.reset(self.robot.state)

    def _step(self):
        command = self.core.step(self.op.xr_frame(), self.robot.state, self.clock)
        return command.as_action(), self.robot.apply(command)

    def odometry(self):
        return self.robot.state.odometry

    def resets(self):
        return self.robot.resets

    def start_poses(self):
        start = self.robot.start
        return start.torso, start.right_ee, start.left_ee


def test_default_session_through_the_core_alone() -> None:
    session = run_default_scenario(CoreSession())
    assert len(session.rows) > 500
    if not SNAPSHOT.is_file():
        pytest.skip("no stored trace next to this package (not inside the repository)")
    # The stored trace went through float32 XR samples and the action encoding of
    # the LeRobot device; the same behaviour, to that precision.
    compare_with_trace(json.loads(SNAPSHOT.read_text()), trace_of(session.rows), tolerance=2e-5)


def test_core_imports_no_framework() -> None:
    """In a fresh interpreter, importing the whole package pulls in no framework."""
    import subprocess
    import sys

    code = (
        "import sys, pkgutil, importlib, rby1_xr_teleop\n"
        "for m in pkgutil.iter_modules(rby1_xr_teleop.__path__):\n"
        "    importlib.import_module('rby1_xr_teleop.' + m.name)\n"
        "bad = [n for n in ('lerobot', 'isaaccapture', 'isaacteleop', 'rby1_sdk', 'torch') if n in sys.modules]\n"
        "print(','.join(bad))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "", f"imported by the core package: {out.stdout.strip()}"
