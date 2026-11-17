"""One teleoperation session of ``rby1_isaac`` with the DEFAULT settings, start to end.

The other Isaac tests switch most newer defaults off to look at one mechanism
at a time. This scenario runs what an operator actually uses — start on Right
A with the reach calibration, grip-gated head and base, body drive, return to
start — against a fake robot that follows the commands, and checks on every
tick that nothing moves when it must not and that no target jumps.

The action of every tick is also compared with a stored trace
(``snapshots/isaac_scenario_trace.json``): refactors of the teleoperator must
leave it unchanged. Regenerate it with ``RBY1_UPDATE_SNAPSHOTS=1`` only for an
intended change of behaviour.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import isaac_fakes as fakes
from isaac_operator import HAND_OUT, HAND_REST, Operator, ScriptedSession
from lerobot_teleoperator_rby1.isaac_teleop import teleop_rby1_xr as mod
from lerobot_teleoperator_rby1.isaac_teleop.config_isaac_teleop import Rby1XRConfig
from lerobot_teleoperator_rby1.isaac_teleop.retargeters import se2_relative
from lerobot_teleoperator_rby1.isaac_teleop.teleop_rby1_xr import Rby1XR

SNAPSHOT = Path(__file__).parent / "snapshots" / "isaac_scenario_trace.json"
DT = 1.0 / 30.0
EE_PREFIXES = ("torso_ee", "right_ee", "left_ee")
POSE_KEYS = [f"{p}.{s}" for p in EE_PREFIXES for s in ("x", "y", "z", "wx", "wy", "wz")]
HEAD_KEYS = ["head_0.pos", "head_1.pos"]
BASE_KEYS = ["x.vel", "y.vel", "theta.vel"]


class Session:
    """Runs the teleoperator tick by tick and checks the per-tick invariants."""

    def __init__(self, monkeypatch, **cfg: Any) -> None:
        self.clock = [1000.0]
        monkeypatch.setattr(mod.time, "monotonic", lambda: self.clock[0])
        monkeypatch.setattr(mod, "build_pipeline", lambda *, with_body: ("pipeline", True))
        monkeypatch.setattr(mod, "build_external_inputs", lambda anchor: {"base_T_anchor": anchor})
        monkeypatch.setattr(mod, "print_xr_connect_help", lambda: None)
        self.op = Operator()
        self.session = ScriptedSession(self.op.frame)
        readers: list[fakes.FakeStateReader] = []

        def reader_factory(address, model):
            readers.append(fakes.FakeStateReader(address, model))
            return readers[-1]

        self.cfg = Rby1XRConfig(auto_launch_cloudxr=False, **cfg)
        self.t = Rby1XR(self.cfg, session_factory=lambda pipeline: self.session, state_reader_factory=reader_factory)
        self.t.connect()
        self.reader = readers[0]
        self.robot = fakes.FollowingRobot(self.reader, dt=DT)
        self.rows: list[tuple[str, dict[str, float]]] = []

    # ------------------------------------------------------------------
    def tick(self, label: str) -> dict[str, float]:
        self.clock[0] += DT
        action = self.t.get_action()
        self._check(label, action)
        self.rows.append((label, dict(action)))
        self.clock[0] += self.robot.apply(action)  # the follower blocks during a reset.cmd motion
        return action

    def hold(self, n: int, label: str) -> dict[str, float]:
        action = {}
        for _ in range(n):
            action = self.tick(label)
        return action

    def move(self, n: int, label: str, **targets: Any) -> dict[str, float]:
        """Move operator fields linearly to ``targets`` over ``n`` ticks."""
        start = {k: np.asarray(getattr(self.op, k), dtype=float) for k in targets}
        action = {}
        for i in range(1, n + 1):
            for k, goal in targets.items():
                v = start[k] + (np.asarray(goal, dtype=float) - start[k]) * (i / n)
                setattr(self.op, k, float(v) if v.ndim == 0 else tuple(v.tolist()))
            action = self.tick(label)
        return action

    # ------------------------------------------------------------------
    def _grip_held(self) -> bool:
        thr = self.cfg.clutch_threshold
        op = self.op
        return (op.right_tracked and op.right_squeeze > thr) or (op.left_tracked and op.left_squeeze > thr)

    def _check(self, label: str, a: dict[str, float]) -> None:
        t, cfg = self.t, self.cfg
        where = f"tick {len(self.rows)} ({label})"
        live = t.is_started and not t.is_stopped
        if not live or not self._grip_held():
            assert [a[k] for k in BASE_KEYS] == [0.0, 0.0, 0.0], f"{where}: base moves without start / grip"
        if not self.rows:
            return
        prev = self.rows[-1][1]
        returning = "reset.cmd" in a or "reset.cmd" in prev or t._core._returning
        if not live and not returning:
            for k in POSE_KEYS + HEAD_KEYS:
                assert a[k] == pytest.approx(prev[k], abs=1e-12), f"{where}: {k} changes while not started / stopped"
        if returning:
            return  # the follower's own ready-pose motion: targets are re-seeded
        limits = {"torso_ee": 0.03, "right_ee": cfg.ee_max_linear_vel * DT * 1.01, "left_ee": cfg.ee_max_linear_vel * DT * 1.01}
        for prefix, limit in limits.items():
            step = math.dist([a[f"{prefix}.{s}"] for s in "xyz"], [prev[f"{prefix}.{s}"] for s in "xyz"])
            assert step <= limit + 1e-9, f"{where}: {prefix} target jumps {step:.4f} m"
        for k in HEAD_KEYS:
            assert abs(a[k] - prev[k]) <= 0.12, f"{where}: {k} jumps {abs(a[k] - prev[k]):.3f} rad"


def _pose(a: dict[str, float], prefix: str) -> np.ndarray:
    return np.array([a[f"{prefix}.{s}"] for s in ("x", "y", "z", "wx", "wy", "wz")])


def run_default_scenario(monkeypatch) -> Session:
    s = Session(monkeypatch)
    t, op, reader = s.t, s.op, s.reader
    grips = dict(right_squeeze=0.9, left_squeeze=0.9)
    no_grips = dict(right_squeeze=0.0, left_squeeze=0.0)

    def set_op(**fields: Any) -> None:
        for k, v in fields.items():
            setattr(op, k, v)

    # 1. Before the first Right A nothing follows: grips, trigger, thumbstick, a turned head.
    a0 = s.tick("before-start")
    set_op(right_trigger=1.0, right_stick=(0.0, 1.0), neck_deg=25.0, left_x=True, **grips)
    a = s.hold(5, "before-start")
    assert not t.is_started
    assert _pose(a, "right_ee") == pytest.approx(_pose(a0, "right_ee")) and a["right_gripper_0.pos"] == 1.0
    set_op(right_trigger=0.0, right_stick=(0.0, 0.0), neck_deg=0.0, left_x=False, **no_grips)

    # 2. Both arms straight out, Right A: the reach is measured and teleoperation starts.
    s.move(10, "arms-out", right_hand=HAND_OUT, left_hand=(0.58, 0.0, 0.0))
    s.hold(18, "arms-out")
    op.right_a = True
    s.tick("right-a-start")
    op.right_a = False
    assert t.is_started
    assert t._core._abs["right"].calibrated_reach == pytest.approx(0.60, abs=1e-3)
    assert t._core._abs["left"].calibrated_reach == pytest.approx(0.58, abs=1e-3)

    # 3. Hands back to rest. Without a grip neither the head nor the base moves.
    a = s.move(12, "rest", right_hand=HAND_REST, left_hand=HAND_REST)
    held = dict(a)
    set_op(neck_deg=20.0, right_stick=(0.0, 1.0))
    a = s.hold(4, "no-grip")
    assert [a[k] for k in HEAD_KEYS] == pytest.approx([held[k] for k in HEAD_KEYS])
    assert _pose(a, "right_ee") == pytest.approx(_pose(held, "right_ee"))
    op.right_stick = (0.0, 0.0)

    # 4. Squeeze both grips: nothing jumps; then the arms follow the hands.
    set_op(**grips)
    a = s.tick("engage")
    assert _pose(a, "right_ee") == pytest.approx(_pose(held, "right_ee"), abs=1e-9)
    assert _pose(a, "left_ee") == pytest.approx(_pose(held, "left_ee"), abs=1e-9)
    assert t._core._clutch["right"].engaged and t._core._clutch["left"].engaged
    before = s.hold(65, "ramp")  # engage_ramp_s
    assert _pose(before, "right_ee")[:3] == pytest.approx(_pose(held, "right_ee")[:3], abs=1e-6)  # offsets latched
    s.move(20, "arm-move", right_hand=(0.35, -0.05, -0.25))
    a = s.hold(3, "arm-move")
    scale = t._core._abs["right"].robot_reach / t._core._abs["right"].calibrated_reach
    moved = _pose(a, "right_ee")[:3] - _pose(before, "right_ee")[:3]
    assert moved == pytest.approx(scale * np.array([0.10, -0.05, 0.10]), abs=2e-3)
    assert _pose(a, "left_ee")[:3] == pytest.approx(_pose(before, "left_ee")[:3], abs=1e-6)

    # The head follows the neck while a grip is held (it started from where it was at the squeeze).
    s.move(15, "look", neck_deg=40.0)
    a = s.hold(22, "look")
    assert a["head_0.pos"] == pytest.approx(held["head_0.pos"] + math.radians(20.0), abs=2e-3)
    op.right_trigger = 0.5
    assert s.tick("trigger")["right_gripper_0.pos"] == pytest.approx(0.5)

    # 5. The torso follows the chest while both arms are clutched.
    torso0 = _pose(a, "torso_ee")
    s.move(15, "lean", lean=(0.06, 0.0))
    a = s.hold(2, "lean")
    assert _pose(a, "torso_ee")[:3] == pytest.approx(torso0[:3] + [0.06, 0.0, 0.0], abs=1e-6)
    s.move(15, "lean-back", lean=(0.0, 0.0))

    # 6. The thumbstick drives the base.
    op.right_stick = (0.0, 1.0)
    a = s.hold(10, "stick")
    assert a["x.vel"] == pytest.approx(s.cfg.base_max_linear) and a["theta.vel"] == 0.0
    op.right_stick = (0.0, 0.0)
    a = s.tick("stick-released")
    assert a["x.vel"] == 0.0

    # 7. Body drive: hold Left X, walk and turn; the base follows 1:1, the torso holds,
    #    the clutched arms stay as they are relative to the torso.
    odometry0 = reader.odometry.copy()
    arms0 = (_pose(a, "right_ee"), _pose(a, "left_ee"))
    torso0, head0 = _pose(a, "torso_ee"), a["head_0.pos"]
    op.left_x = True
    s.tick("drive-enter")
    s.move(45, "drive-walk", xy=(0.3, 0.1))
    a = s.hold(40, "drive-settle")
    x, y, yaw = se2_relative(odometry0, reader.odometry)
    assert (x, y) == pytest.approx((0.3, 0.1), abs=0.012) and yaw == pytest.approx(0.0, abs=math.radians(1.2))
    s.move(30, "drive-turn", body_deg=30.0)
    a = s.hold(40, "drive-settle")
    x, y, yaw = se2_relative(odometry0, reader.odometry)
    assert yaw == pytest.approx(math.radians(30.0), abs=math.radians(1.2))
    assert _pose(a, "torso_ee") == pytest.approx(torso0, abs=1e-9)
    assert a["head_0.pos"] == pytest.approx(head0, abs=2e-3)  # the neck did not turn
    assert _pose(a, "right_ee")[:3] == pytest.approx(arms0[0][:3], abs=5e-3)
    assert _pose(a, "left_ee")[:3] == pytest.approx(arms0[1][:3], abs=5e-3)
    op.left_x = False
    a = s.tick("drive-exit")
    assert [a[k] for k in BASE_KEYS] == [0.0, 0.0, 0.0] and t._core._body_drive is None

    # 8. Right B during a body drive stops everything.
    op.left_x = True
    s.tick("drive-enter")
    a = s.move(8, "drive-walk", xy=(0.4, 0.1))
    assert a["x.vel"] > 0.0
    op.right_b = True
    a = s.tick("right-b")
    op.right_b = False
    assert t.is_stopped and [a[k] for k in BASE_KEYS] == [0.0, 0.0, 0.0]
    s.hold(5, "stopped")
    set_op(left_x=False, **no_grips)
    s.tick("stopped")

    # 9. Right A resumes and returns to the start pose (the follower's ready-pose motion).
    op.right_a = True
    a = s.tick("right-a-return")
    op.right_a = False
    assert not t.is_stopped and a["reset.cmd"] == pytest.approx(s.cfg.ready_return_duration_s)
    a = s.hold(3, "returned")
    assert s.robot.resets == 1 and "reset.cmd" not in a
    for prefix, start in zip(EE_PREFIXES, s.robot.start[:3]):
        assert _pose(a, prefix)[:3] == pytest.approx(start[:3, 3], abs=1e-9)
    assert t._core._abs["right"].calibrated_reach == pytest.approx(0.60, abs=1e-3)  # arms were not out: kept

    # 10. Squeezing again after the return does not move the arms.
    set_op(**grips)
    b = s.tick("engage")
    assert _pose(b, "right_ee") == pytest.approx(_pose(a, "right_ee"), abs=1e-9)
    s.hold(65, "ramp")

    # 11. Body tracking lost while clutched: the torso holds. The arms keep following
    #     with the shoulder estimated from the headset (shoulder_source="auto").
    op.body_tracked = False
    a = s.hold(40, "no-body")
    assert not t._core._clutch["torso"].engaged and t._core._clutch["right"].engaged
    # Headset lost as well, for longer than abs_hold_s: now the arms hold too.
    op.head_tracked = False
    s.hold(34, "no-body-no-head")
    a = s.hold(6, "no-body-no-head")
    assert not t._core._clutch["right"].engaged and not t._core._clutch["left"].engaged
    # Tracking is back: the arms ramp in again from where they held.
    set_op(body_tracked=True, head_tracked=True)
    b = s.tick("tracking-back")
    assert _pose(b, "right_ee") == pytest.approx(_pose(a, "right_ee"), abs=1e-9)
    s.hold(69, "tracking-back")
    assert t._core._clutch["right"].engaged

    # 12. The headset is lost during a body drive: the base stops until Left X is pressed again.
    op.left_x = True
    s.tick("drive-enter")
    a = s.move(6, "drive-walk", xy=(0.46, 0.1))
    assert a["x.vel"] > 0.0
    op.head_tracked = False
    a = s.tick("drive-lost")
    assert [a[k] for k in BASE_KEYS] == [0.0, 0.0, 0.0] and t._core._body_drive is None and t._core._body_drive_blocked
    op.head_tracked = True
    a = s.hold(3, "drive-blocked")
    assert [a[k] for k in BASE_KEYS] == [0.0, 0.0, 0.0]
    op.left_x = False
    s.tick("drive-exit")

    # 13. Both grips released: everything holds.
    set_op(**no_grips)
    a = s.tick("released")
    set_op(neck_deg=0.0, right_stick=(1.0, 0.0), right_hand=(0.4, 0.1, -0.1))
    b = s.hold(5, "released")
    assert _pose(b, "right_ee") == pytest.approx(_pose(a, "right_ee"), abs=1e-12)
    assert b["head_0.pos"] == pytest.approx(a["head_0.pos"], abs=1e-12) and b["y.vel"] == 0.0

    t.disconnect()
    return s


def _trace(rows: list[tuple[str, dict[str, float]]]) -> dict[str, Any]:
    keys = sorted({k for _, a in rows for k in a})
    return {
        "dt": DT,
        "keys": keys,
        "rows": [[label, *[None if k not in a else round(float(a[k]), 9) for k in keys]] for label, a in rows],
    }


def test_default_session_scenario_and_trace(monkeypatch) -> None:
    s = run_default_scenario(monkeypatch)
    trace = _trace(s.rows)
    if os.environ.get("RBY1_UPDATE_SNAPSHOTS") == "1":
        lines = ",\n".join(json.dumps(row, separators=(",", ":")) for row in trace["rows"])
        SNAPSHOT.write_text(
            json.dumps({"dt": trace["dt"], "keys": trace["keys"]})[:-1] + ',\n"rows": [\n' + lines + "\n]}\n"
        )
    stored = json.loads(SNAPSHOT.read_text())
    assert stored["dt"] == pytest.approx(DT) and stored["keys"] == trace["keys"]
    assert [row[0] for row in stored["rows"]] == [row[0] for row in trace["rows"]], "the tick sequence changed"
    for i, (old, new) in enumerate(zip(stored["rows"], trace["rows"])):
        for key, a, b in zip(trace["keys"], old[1:], new[1:]):
            assert (a is None) == (b is None), f"tick {i} ({new[0]}): {key} present/absent changed"
            if a is not None:
                assert b == pytest.approx(a, abs=1e-7), f"tick {i} ({new[0]}): {key} changed"
