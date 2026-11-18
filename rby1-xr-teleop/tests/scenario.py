"""One teleoperation session with the DEFAULT settings, start to end — the script and its checks.

``run_default_scenario`` drives a :class:`ScenarioSession`: reach calibration
on Right A, grip-gated head and base, body drive, Right B, return to start,
tracking loss. Every tick is checked: nothing moves when it must not, no
target jumps. Two sessions run it:

* ``test_core_scenario.py`` (here): the core alone, without any framework;
* ``tests/test_isaac_scenario.py`` (repository): the LeRobot device on top of it.

Both compare the action of every tick with the stored trace
(``tests/snapshots/isaac_scenario_trace.json`` in the repository).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from rby1_xr_teleop.retargeters import se2_relative
from rby1_xr_teleop.scripted import HAND_OUT, HAND_REST, Operator

DT = 1.0 / 30.0
EE_PREFIXES = ("torso_ee", "right_ee", "left_ee")
POSE_KEYS = [f"{p}.{s}" for p in EE_PREFIXES for s in ("x", "y", "z", "wx", "wy", "wz")]
HEAD_KEYS = ["head_0.pos", "head_1.pos"]
BASE_KEYS = ["x.vel", "y.vel", "theta.vel"]


class ScenarioSession:
    """Tick-by-tick driver with the per-tick invariants; subclasses provide the teleoperator.

    A subclass sets ``self.core`` (the :class:`XrTeleopCore` under test) and
    ``self.cfg`` (its settings) and implements :meth:`_step`, :meth:`odometry`,
    :meth:`resets`, :meth:`start_poses` and :meth:`close`.
    """

    def __init__(self) -> None:
        self.clock = 1000.0
        self.op = Operator()
        self.rows: list[tuple[str, dict[str, float]]] = []

    # -- provided by the subclass ---------------------------------------
    def _step(self) -> tuple[dict[str, float], float]:
        """Run one tick; return the action and the seconds the robot blocked (return motion)."""
        raise NotImplementedError

    def odometry(self) -> np.ndarray:
        raise NotImplementedError

    def resets(self) -> int:
        raise NotImplementedError

    def start_poses(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Torso, right and left end-effector pose (4x4) the robot started at."""
        raise NotImplementedError

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    def tick(self, label: str) -> dict[str, float]:
        self.clock += DT
        action, blocked = self._step()
        self._check(label, action)
        self.rows.append((label, dict(action)))
        self.clock += blocked  # the robot blocks during its return-to-ready motion
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
        core, cfg = self.core, self.cfg
        where = f"tick {len(self.rows)} ({label})"
        live = core.started and not core.stopped
        if not live or not self._grip_held():
            assert [a[k] for k in BASE_KEYS] == [0.0, 0.0, 0.0], f"{where}: base moves without start / grip"
        if not self.rows:
            return
        prev = self.rows[-1][1]
        returning = "reset.cmd" in a or "reset.cmd" in prev or core._returning
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


def run_default_scenario(s: ScenarioSession) -> ScenarioSession:
    """One session with the default settings, connect to release; asserts along the way."""
    core, op = s.core, s.op
    grips = dict(right_squeeze=0.9, left_squeeze=0.9)
    no_grips = dict(right_squeeze=0.0, left_squeeze=0.0)

    def set_op(**fields: Any) -> None:
        for k, v in fields.items():
            setattr(op, k, v)

    # 1. Before the first Right A nothing follows: grips, trigger, thumbstick, a turned head.
    a0 = s.tick("before-start")
    set_op(right_trigger=1.0, right_stick=(0.0, 1.0), neck_deg=25.0, left_x=True, **grips)
    a = s.hold(5, "before-start")
    assert not core.started
    assert _pose(a, "right_ee") == pytest.approx(_pose(a0, "right_ee")) and a["right_gripper_0.pos"] == 1.0
    set_op(right_trigger=0.0, right_stick=(0.0, 0.0), neck_deg=0.0, left_x=False, **no_grips)

    # 2. Both arms straight out, Right A: the reach is measured and teleoperation starts.
    s.move(10, "arms-out", right_hand=HAND_OUT, left_hand=(0.58, 0.0, 0.0))
    s.hold(18, "arms-out")
    op.right_a = True
    s.tick("right-a-start")
    op.right_a = False
    assert core.started
    assert core._abs["right"].calibrated_reach == pytest.approx(0.60, abs=1e-3)
    assert core._abs["left"].calibrated_reach == pytest.approx(0.58, abs=1e-3)

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
    assert core._clutch["right"].engaged and core._clutch["left"].engaged
    before = s.hold(65, "ramp")  # engage_ramp_s
    assert _pose(before, "right_ee")[:3] == pytest.approx(_pose(held, "right_ee")[:3], abs=1e-6)  # offsets latched
    s.move(20, "arm-move", right_hand=(0.35, -0.05, -0.25))
    a = s.hold(3, "arm-move")
    scale = core._abs["right"].robot_reach / core._abs["right"].calibrated_reach
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
    odometry0 = s.odometry().copy()
    arms0 = (_pose(a, "right_ee"), _pose(a, "left_ee"))
    torso0, head0 = _pose(a, "torso_ee"), a["head_0.pos"]
    op.left_x = True
    s.tick("drive-enter")
    s.move(45, "drive-walk", xy=(0.3, 0.1))
    a = s.hold(40, "drive-settle")
    x, y, yaw = se2_relative(odometry0, s.odometry())
    assert (x, y) == pytest.approx((0.3, 0.1), abs=0.012) and yaw == pytest.approx(0.0, abs=math.radians(1.2))
    s.move(30, "drive-turn", body_deg=30.0)
    a = s.hold(40, "drive-settle")
    x, y, yaw = se2_relative(odometry0, s.odometry())
    assert yaw == pytest.approx(math.radians(30.0), abs=math.radians(1.2))
    assert _pose(a, "torso_ee") == pytest.approx(torso0, abs=1e-9)
    assert a["head_0.pos"] == pytest.approx(head0, abs=2e-3)  # the neck did not turn
    assert _pose(a, "right_ee")[:3] == pytest.approx(arms0[0][:3], abs=5e-3)
    assert _pose(a, "left_ee")[:3] == pytest.approx(arms0[1][:3], abs=5e-3)
    op.left_x = False
    a = s.tick("drive-exit")
    assert [a[k] for k in BASE_KEYS] == [0.0, 0.0, 0.0] and core._body_drive is None

    # 8. Right B during a body drive stops everything.
    op.left_x = True
    s.tick("drive-enter")
    a = s.move(8, "drive-walk", xy=(0.4, 0.1))
    assert a["x.vel"] > 0.0
    op.right_b = True
    a = s.tick("right-b")
    op.right_b = False
    assert core.stopped and [a[k] for k in BASE_KEYS] == [0.0, 0.0, 0.0]
    # The grippers hold too: a trigger pulled while stopped does nothing.
    op.right_trigger = 1.0
    b = s.hold(5, "stopped")
    assert b["right_gripper_0.pos"] == pytest.approx(a["right_gripper_0.pos"]) == pytest.approx(0.5)
    set_op(left_x=False, right_trigger=0.5, **no_grips)
    s.tick("stopped")

    # 9. Right A resumes and returns to the start pose (the follower's ready-pose motion).
    op.right_a = True
    a = s.tick("right-a-return")
    op.right_a = False
    assert not core.stopped and a["reset.cmd"] == pytest.approx(s.cfg.ready_return_duration_s)
    a = s.hold(3, "returned")
    assert s.resets() == 1 and "reset.cmd" not in a
    for prefix, start in zip(EE_PREFIXES, s.start_poses()):
        assert _pose(a, prefix)[:3] == pytest.approx(start[:3, 3], abs=1e-9)
    assert core._abs["right"].calibrated_reach == pytest.approx(0.60, abs=1e-3)  # arms were not out: kept

    # 10. Squeezing again after the return does not move the arms.
    set_op(**grips)
    b = s.tick("engage")
    assert _pose(b, "right_ee") == pytest.approx(_pose(a, "right_ee"), abs=1e-9)
    s.hold(65, "ramp")

    # 11. Body tracking lost while clutched: the torso holds. The arms follow for
    #     abs_hold_s more with the shoulder where it was, then hold. The shoulder does
    #     not change over to the headset estimate, which would shift the hand targets.
    before = s.rows[-1][1]
    assert core._shoulder_locked == {"right": "body", "left": "body"}
    op.body_tracked = False
    a = s.hold(40, "no-body")
    assert not core._clutch["torso"].engaged
    assert not core._clutch["right"].engaged and not core._clutch["left"].engaged
    assert core._abs_state["right"] == core._abs_state["left"] == "hold (no body)"
    for prefix in ("right_ee", "left_ee"):
        assert _pose(a, prefix)[:3] == pytest.approx(_pose(before, prefix)[:3], abs=1e-4)
    # While they hold, a moving hand or a lost headset changes nothing.
    op.head_tracked = False
    s.move(10, "no-body-no-head", right_hand=(0.45, -0.05, -0.15))
    b = s.hold(14, "no-body-no-head")
    assert _pose(b, "right_ee") == pytest.approx(_pose(a, "right_ee"), abs=1e-12)
    s.move(10, "no-body-no-head", right_hand=(0.35, -0.05, -0.25))
    a = s.hold(6, "no-body-no-head")
    # Tracking is back: the arms ramp in again from where they held.
    set_op(body_tracked=True, head_tracked=True)
    b = s.tick("tracking-back")
    assert _pose(b, "right_ee") == pytest.approx(_pose(a, "right_ee"), abs=1e-9)
    s.hold(69, "tracking-back")
    assert core._clutch["right"].engaged and core._shoulder_locked["right"] == "body"

    # 12. The headset is lost during a body drive: the base stops until Left X is pressed again.
    op.left_x = True
    s.tick("drive-enter")
    a = s.move(6, "drive-walk", xy=(0.46, 0.1))
    assert a["x.vel"] > 0.0
    op.head_tracked = False
    a = s.tick("drive-lost")
    assert [a[k] for k in BASE_KEYS] == [0.0, 0.0, 0.0] and core._body_drive is None and core._body_drive_blocked
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

    s.close()
    return s


def trace_of(rows: list[tuple[str, dict[str, float]]]) -> dict[str, Any]:
    keys = sorted({k for _, a in rows for k in a})
    return {
        "dt": DT,
        "keys": keys,
        "rows": [[label, *[None if k not in a else round(float(a[k]), 9) for k in keys]] for label, a in rows],
    }


def compare_with_trace(stored: dict[str, Any], trace: dict[str, Any], tolerance: float) -> None:
    """The action of every tick equals the stored one within ``tolerance``."""
    assert stored["dt"] == pytest.approx(DT) and stored["keys"] == trace["keys"]
    assert [row[0] for row in stored["rows"]] == [row[0] for row in trace["rows"]], "the tick sequence changed"
    for i, (old, new) in enumerate(zip(stored["rows"], trace["rows"])):
        for key, a, b in zip(trace["keys"], old[1:], new[1:]):
            assert (a is None) == (b is None), f"tick {i} ({new[0]}): {key} present/absent changed"
            if a is not None:
                assert b == pytest.approx(a, abs=tolerance), f"tick {i} ({new[0]}): {key} changed"
