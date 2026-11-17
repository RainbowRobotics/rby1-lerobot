"""One teleoperation session of ``rby1_isaac`` with the DEFAULT settings, start to end.

The other Isaac tests switch most newer defaults off to look at one mechanism
at a time. This scenario (``rby1-xr-teleop/tests/scenario.py``) runs what an
operator actually uses — start on Right A with the reach calibration,
grip-gated head and base, body drive, return to start — through the LeRobot
device, against a fake robot that follows the commands, and checks on every
tick that nothing moves when it must not and that no target jumps.

The action of every tick is also compared with a stored trace
(``snapshots/isaac_scenario_trace.json``): refactors of the teleoperator must
leave it unchanged. Regenerate it with ``RBY1_UPDATE_SNAPSHOTS=1`` only for an
intended change of behaviour.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rby1-xr-teleop" / "tests"))

import isaac_fakes as fakes  # noqa: E402
from isaac_operator import Operator, ScriptedSession  # noqa: E402
from lerobot_teleoperator_rby1.isaac_teleop import teleop_rby1_xr as mod  # noqa: E402
from lerobot_teleoperator_rby1.isaac_teleop.config_isaac_teleop import Rby1XRConfig  # noqa: E402
from lerobot_teleoperator_rby1.isaac_teleop.teleop_rby1_xr import Rby1XR  # noqa: E402
from scenario import DT, ScenarioSession, compare_with_trace, run_default_scenario, trace_of  # noqa: E402

SNAPSHOT = Path(__file__).parent / "snapshots" / "isaac_scenario_trace.json"


class DeviceSession(ScenarioSession):
    """The scenario through ``Rby1XR`` (scripted XR session, fake robot link that follows)."""

    def __init__(self, monkeypatch, **cfg: Any) -> None:
        super().__init__()
        monkeypatch.setattr(mod.time, "monotonic", lambda: self.clock)
        monkeypatch.setattr(mod, "build_pipeline", lambda *, with_body: ("pipeline", True))
        monkeypatch.setattr(mod, "build_external_inputs", lambda anchor: {"base_T_anchor": anchor})
        monkeypatch.setattr(mod, "print_xr_connect_help", lambda: None)
        self.op = Operator()  # with frame(): the Isaac Capture pipeline outputs
        self.session = ScriptedSession(self.op.frame)
        readers: list[fakes.FakeStateReader] = []

        def reader_factory(address, model):
            readers.append(fakes.FakeStateReader(address, model))
            return readers[-1]

        self.cfg = Rby1XRConfig(auto_launch_cloudxr=False, **cfg)
        self.t = Rby1XR(self.cfg, session_factory=lambda pipeline: self.session, state_reader_factory=reader_factory)
        self.t.connect()
        self.core = self.t._core
        self.reader = readers[0]
        self.robot = fakes.FollowingRobot(self.reader, dt=DT)

    def _step(self):
        action = self.t.get_action()
        return action, self.robot.apply(action)

    def odometry(self):
        return self.reader.odometry

    def resets(self):
        return self.robot.resets

    def start_poses(self):
        return self.robot.start[:3]

    def close(self):
        self.t.disconnect()


def test_default_session_scenario_and_trace(monkeypatch) -> None:
    s = run_default_scenario(DeviceSession(monkeypatch))
    trace = trace_of(s.rows)
    if os.environ.get("RBY1_UPDATE_SNAPSHOTS") == "1":
        lines = ",\n".join(json.dumps(row, separators=(",", ":")) for row in trace["rows"])
        SNAPSHOT.write_text(
            json.dumps({"dt": trace["dt"], "keys": trace["keys"]})[:-1] + ',\n"rows": [\n' + lines + "\n]}\n"
        )
    compare_with_trace(json.loads(SNAPSHOT.read_text()), trace, tolerance=1e-7)
