#!/usr/bin/env python
"""End-to-end check of ``rby1_isaac`` against a SIMULATED RB-Y1 core, without a headset.

Runs the real follower plugin (``lerobot_robot_rby1.Rby1``) and the real
teleoperator (``Rby1XR``) in the usual loop — observation, ``get_action``,
``send_action`` — against an RB-Y1 core that drives a simulator (the vendor
``rby1-sim-isaac`` image in SDK mode: the same core software as on the robot).
The XR input comes from a scripted operator instead of a headset, so no
``isaaccapture`` / CloudXR is needed.

What it answers without a robot:
  hold    is there a jolt right after the ready pose, when the first
          end-effector command arrives?
  arms    Right A start with the reach calibration, squeeze, the arm follows.
  loss    body tracking drops out while the arms are squeezed: they hold where
          they are (no shift) and follow again when it returns. Needs ``arms``.
  drive   Left X body drive: do the SIGNS match (a step forward drives forward,
          a left turn turns left) and does the base settle on the target?
  stop    Right B stops the base.
  return  Right A returns to the start pose; squeezing afterwards does not jump.

    python lerobot-teleoperator-rby1/scripts/isaac_sim_e2e.py --out /tmp/e2e [--scenarios hold,arms,drive]

It refuses any address that is not local: this script moves the robot it is
pointed at. On a shared machine check first that 127.0.0.1:50051 is YOUR
simulator (``ss -ltn | grep 50051``, ``docker ps``).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests"))  # isaac_operator / isaac_fakes (not the fake rby1_sdk in tests/fakes)

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
SCENARIOS = ("hold", "arms", "loss", "drive", "stop", "return")
logger = logging.getLogger("isaac_sim_e2e")


class StateLogger(threading.Thread):
    """Samples the robot's joint positions / velocities on its own read-only SDK link."""

    def __init__(self, address: str, model: str, rate_hz: float = 250.0) -> None:
        super().__init__(daemon=True, name="e2e-state-logger")
        import rby1_sdk as rby

        self._robot = rby.create_robot(address, model)
        if not self._robot.connect():
            raise ConnectionError(f"state logger could not connect to {address}")
        self.names = list(self._robot.model().robot_joint_names)
        self._period = 1.0 / rate_hz
        self._halt = threading.Event()
        self.t: list[float] = []
        self.q: list[np.ndarray] = []
        self.qd: list[np.ndarray] = []

    def run(self) -> None:
        while not self._halt.is_set():
            t0 = time.monotonic()
            state = self._robot.get_state()
            self.t.append(t0)
            self.q.append(np.asarray(state.position, dtype=float).copy())
            self.qd.append(np.asarray(state.velocity, dtype=float).copy())
            time.sleep(max(0.0, self._period - (time.monotonic() - t0)))

    def stop(self) -> None:
        self._halt.set()
        self.join(timeout=2.0)
        try:
            self._robot.disconnect()
        except Exception:  # noqa: BLE001
            pass

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = min(len(self.t), len(self.q), len(self.qd))
        return np.asarray(self.t[:n]), np.asarray(self.q[:n]), np.asarray(self.qd[:n])


def se2_xyyaw(T: np.ndarray | None) -> tuple[float, float, float]:  # noqa: N803
    if T is None:
        return (math.nan, math.nan, math.nan)
    return float(T[0, 2]), float(T[1, 2]), math.atan2(float(T[1, 0]), float(T[0, 0]))


class Run:
    """The teleoperation loop with a scripted operator; logs every tick."""

    def __init__(self, robot: Any, teleop: Any, op: Any, fps: float) -> None:
        self.robot, self.teleop, self.op = robot, teleop, op
        self.period = 1.0 / fps
        self.rows: list[dict[str, Any]] = []
        self.first_command_t: float | None = None

    def tick(self, label: str) -> dict[str, float]:
        t0 = time.monotonic()
        self.robot.get_observation()
        action = self.teleop.get_action()
        if self.first_command_t is None:
            self.first_command_t = time.monotonic()
        self.robot.send_action(action)
        snap = self.teleop._core._last_snapshot
        x, y, yaw = se2_xyyaw(getattr(snap, "odometry", None))
        row: dict[str, Any] = {"t": t0, "label": label, "odom_x": x, "odom_y": y, "odom_yaw": yaw}
        for name, T in (("torso", snap.torso), ("right_ee", snap.right_ee), ("left_ee", snap.left_ee)):
            row.update({f"meas_{name}_{a}": float(T[i, 3]) for i, a in enumerate("xyz")})
        row.update({"meas_head_0": float(snap.head_q[0]), "meas_head_1": float(snap.head_q[1])})
        row.update({f"cmd_{k}": float(v) for k, v in action.items()})
        self.rows.append(row)
        time.sleep(max(0.0, self.period - (time.monotonic() - t0)))
        return action

    def hold(self, seconds: float, label: str) -> dict[str, float]:
        action: dict[str, float] = {}
        for _ in range(max(1, round(seconds / self.period))):
            action = self.tick(label)
        return action

    def move(self, seconds: float, label: str, **targets: Any) -> dict[str, float]:
        n = max(1, round(seconds / self.period))
        start = {k: np.asarray(getattr(self.op, k), dtype=float) for k in targets}
        action: dict[str, float] = {}
        for i in range(1, n + 1):
            for k, goal in targets.items():
                v = start[k] + (np.asarray(goal, dtype=float) - start[k]) * (i / n)
                setattr(self.op, k, float(v) if v.ndim == 0 else tuple(v.tolist()))
            action = self.tick(label)
        return action

    def last(self, label: str | None = None) -> dict[str, Any]:
        rows = self.rows if label is None else [r for r in self.rows if r["label"] == label]
        return rows[-1]


def check(results: list[dict[str, Any]], name: str, ok: bool, detail: str) -> None:
    results.append({"check": name, "ok": bool(ok), "detail": detail})
    logger.info("%s  %s — %s", "PASS" if ok else "FAIL", name, detail)


def scenario_hold(run: Run, state: StateLogger, results: list, seconds: float) -> None:
    """Nothing is pressed: the robot should stay exactly where the ready pose left it."""
    run.hold(seconds, "hold")
    t, q, qd = state.arrays()
    t_first = run.first_command_t
    before = (t >= t_first - 0.3) & (t < t_first)
    after = (t >= t_first) & (t <= t_first + 1.5)
    if before.sum() < 5 or after.sum() < 20:
        check(results, "hold: state samples around the first command", False, f"{before.sum()} / {after.sum()} samples")
        return
    base = q[before].mean(axis=0)
    dev = np.abs(q[after] - base).max(axis=0)
    vel = np.abs(qd[after]).max(axis=0)
    names = np.asarray(state.names)
    body = np.array([n.startswith(("torso_", "right_arm_", "left_arm_", "head_")) for n in names])
    top = np.argsort(-(dev * body))[:4]
    detail = ", ".join(f"{names[i]} {math.degrees(dev[i]):.2f}° ({math.degrees(vel[i]):.1f}°/s)" for i in top)
    check(
        results,
        "hold: no jolt at the first end-effector command (max joint deviation < 1.0° within 1.5 s)",
        float((dev * body).max()) < math.radians(1.0),
        f"largest: {detail}",
    )


def scenario_arms(run: Run, results: list) -> None:
    from isaac_operator import HAND_OUT, HAND_REST

    op, teleop = run.op, run.teleop
    run.move(1.0, "arms-out", right_hand=HAND_OUT, left_hand=HAND_OUT)
    run.hold(1.0, "arms-out")
    op.right_a = True
    run.tick("right-a-start")
    op.right_a = False
    reach = {side: m.calibrated_reach for side, m in teleop._core._abs.items()}
    check(results, "arms: Right A starts and measures the reach", teleop.is_started and all(reach.values()),
          f"started={teleop.is_started}, reach={ {k: None if v is None else round(v, 3) for k, v in reach.items()} }")
    run.move(1.0, "rest", right_hand=HAND_REST, left_hand=HAND_REST)
    before = run.last()
    op.right_squeeze = op.left_squeeze = 0.9
    run.hold(teleop.config.engage_ramp_s + 1.0, "engage")
    held = run.last()
    drift = math.dist([held[f"meas_right_ee_{a}"] for a in "xyz"], [before[f"meas_right_ee_{a}"] for a in "xyz"])
    check(results, "arms: squeezing does not move the arm (measured right hand within 2 cm)", drift < 0.02,
          f"moved {drift * 100:.1f} cm")
    run.move(1.5, "arm-move", right_hand=(HAND_REST[0] + 0.10, HAND_REST[1], HAND_REST[2] + 0.05))
    run.hold(1.5, "arm-move")
    end = run.last()
    target = [end[f"cmd_right_ee.{a}"] for a in "xyz"]
    measured = [end[f"meas_right_ee_{a}"] for a in "xyz"]
    moved = measured[0] - held["meas_right_ee_x"]
    check(results, "arms: the right hand follows forward (moved > 5 cm, within 5 cm of the target)",
          moved > 0.05 and math.dist(target, measured) < 0.05,
          f"moved {moved * 100:.1f} cm forward, {math.dist(target, measured) * 100:.1f} cm from the target")


def scenario_loss(run: Run, results: list) -> None:
    """Body tracking drops out while both arms are squeezed, then comes back."""
    op, teleop = run.op, run.teleop
    core = teleop._core

    def shift(a: dict[str, Any], b: dict[str, Any], kind: str) -> float:
        sep = "." if kind == "cmd" else "_"
        return max(
            math.dist([a[f"{kind}_{side}_ee{sep}{x}"] for x in "xyz"], [b[f"{kind}_{side}_ee{sep}{x}"] for x in "xyz"])
            for side in ("right", "left")
        )

    op.right_squeeze = op.left_squeeze = 0.9
    run.hold(1.0, "loss-settle")
    before = run.last()
    op.body_tracked = False
    run.hold(teleop.config.abs_hold_s + 1.5, "no-body")
    lost = run.last()
    holding = not core._clutch["right"].engaged and not core._clutch["left"].engaged
    check(
        results,
        # The targets are relative to the measured torso, which settles by a millimetre or two
        # when it stops following; a change of shoulder source would be several centimetres.
        "loss: body tracking lost — the arms hold where they were (targets within 5 mm, measured hands within 1 cm)",
        holding and shift(before, lost, "cmd") < 5e-3 and shift(before, lost, "meas") < 0.01,
        f"holding={holding} ({core._abs_state['right']}), target moved {shift(before, lost, 'cmd') * 1000:.2f} mm, "
        f"hand moved {shift(before, lost, 'meas') * 100:.2f} cm",
    )
    op.body_tracked = True
    run.hold(teleop.config.engage_ramp_s + 1.0, "body-back")
    back = run.last()
    following = core._clutch["right"].engaged and core._clutch["left"].engaged
    check(
        results,
        "loss: the arms follow again when body tracking returns (measured hands within 2 cm of before)",
        following and shift(before, back, "meas") < 0.02,
        f"following={following}, hand moved {shift(before, back, 'meas') * 100:.2f} cm",
    )


def _odometry(run: Run) -> tuple[float, float, float]:
    r = run.last()
    return r["odom_x"], r["odom_y"], r["odom_yaw"]


def _relative(start: tuple[float, float, float], now: tuple[float, float, float]) -> tuple[float, float, float]:
    """Pose ``now`` in the frame of ``start`` (odometry x, y, yaw)."""
    c, s = math.cos(start[2]), math.sin(start[2])
    dx, dy = now[0] - start[0], now[1] - start[1]
    yaw = (now[2] - start[2] + math.pi) % (2.0 * math.pi) - math.pi
    return c * dx + s * dy, -s * dx + c * dy, yaw


def scenario_drive(run: Run, results: list, holonomic: bool) -> None:
    op, teleop = run.op, run.teleop
    if not teleop.is_started:
        check(results, "drive: needs a started session (run the arms scenario first)", False, "not started")
        return
    op.right_squeeze = op.left_squeeze = 0.9
    run.hold(0.5, "drive-prepare")
    start = _odometry(run)
    x0, y0 = op.xy
    op.left_x = True
    run.tick("drive-enter")
    check(results, "drive: Left X enters the body drive", teleop._core._body_drive is not None, f"active={teleop._core._body_drive is not None}")

    run.move(3.0, "drive-forward", xy=(x0 + 0.30, y0))
    run.hold(3.0, "drive-forward")
    fx, fy, fyaw = _relative(start, _odometry(run))
    alive = teleop._core._body_drive is not None
    check(results, "drive: a step forward drives the base forward (+0.30 m, within 3 cm)",
          alive and abs(fx - 0.30) < 0.03 and abs(fy) < 0.03,
          f"odometry moved x {fx:+.3f} m, y {fy:+.3f} m, yaw {math.degrees(fyaw):+.1f}°; mode alive={alive}")

    if holonomic and teleop._core._body_drive is not None:
        run.move(2.5, "drive-left", xy=(x0 + 0.30, y0 + 0.20))
        run.hold(3.0, "drive-left")
        lx, ly, _ = _relative(start, _odometry(run))
        alive = teleop._core._body_drive is not None
        check(results, "drive: a step to the left drives the base left (+0.20 m, within 3 cm)",
              alive and abs(ly - 0.20) < 0.03 and abs(lx - 0.30) < 0.03,
              f"odometry at x {lx:+.3f} m, y {ly:+.3f} m; mode alive={alive}")

    if teleop._core._body_drive is not None:
        run.move(3.0, "drive-turn", body_deg=op.body_deg + 30.0)
        run.hold(3.0, "drive-turn")
        _, _, yaw = _relative(start, _odometry(run))
        alive = teleop._core._body_drive is not None
        check(results, "drive: turning the body left turns the base left (+30°, within 3°)",
              alive and abs(math.degrees(yaw) - 30.0) < 3.0,
              f"odometry turned {math.degrees(yaw):+.1f}°; mode alive={alive}")


def scenario_stop(run: Run, results: list) -> None:
    op, teleop = run.op, run.teleop
    if teleop._core._body_drive is None:
        op.left_x = True
        run.tick("drive-enter")
    x, y = op.xy
    run.move(1.0, "stop-walk", xy=(x + 0.15 * math.cos(math.radians(op.body_deg)), y + 0.15 * math.sin(math.radians(op.body_deg))))
    op.right_b = True
    action = run.tick("right-b")
    op.right_b = False
    run.hold(1.0, "stopped")
    a = _odometry(run)
    run.hold(1.0, "stopped")
    b = _odometry(run)
    still = math.dist(a[:2], b[:2]) < 0.005 and abs(b[2] - a[2]) < math.radians(0.5)
    zero = all(action.get(k, 0.0) == 0.0 for k in ("x.vel", "y.vel", "theta.vel"))
    check(results, "stop: Right B zeroes the base command and the base comes to rest", teleop.is_stopped and zero and still,
          f"stopped={teleop.is_stopped}, command zero={zero}, drift over 1 s {math.dist(a[:2], b[:2]) * 100:.2f} cm")
    op.left_x = False
    op.right_squeeze = op.left_squeeze = 0.0
    run.tick("stopped")


def scenario_return(run: Run, results: list) -> None:
    op, teleop = run.op, run.teleop
    start = teleop._core._start_snapshot
    op.right_squeeze = op.left_squeeze = 0.0
    run.tick("pre-return")
    op.right_a = True
    action = run.tick("right-a-return")  # send_action blocks for the follower's ready-pose motion
    op.right_a = False
    run.hold(1.5, "returned")
    end = run.last()
    err = max(
        math.dist([end[f"meas_{name}_{a}"] for a in "xyz"], T[:3, 3])
        for name, T in (("torso", start.torso), ("right_ee", start.right_ee), ("left_ee", start.left_ee))
    )
    check(results, "return: Right A sends reset.cmd and the robot is back at the start pose (within 3 cm)",
          "reset.cmd" in action and err < 0.03, f"reset.cmd={'reset.cmd' in action}, largest position error {err * 100:.1f} cm")
    op.right_squeeze = op.left_squeeze = 0.9
    run.hold(teleop.config.engage_ramp_s + 1.0, "re-engage")
    after = run.last()
    drift = math.dist([after[f"meas_right_ee_{a}"] for a in "xyz"], [end[f"meas_right_ee_{a}"] for a in "xyz"])
    check(results, "return: squeezing after the return does not move the arm (within 2 cm)", drift < 0.02,
          f"moved {drift * 100:.1f} cm")
    op.right_squeeze = op.left_squeeze = 0.0
    run.tick("released")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--address", default="127.0.0.1:50051", help="SIMULATED robot core (local addresses only)")
    parser.add_argument("--out", required=True, help="directory for ticks.csv, state.npz, report.json")
    parser.add_argument("--scenarios", default=",".join(SCENARIOS), help=f"comma-separated subset of {SCENARIOS}")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--hold-seconds", type=float, default=3.0)
    parser.add_argument("--ee-auto-action-period", choices=("true", "false"), default="true")
    parser.add_argument("--ready-torso-deg", default="0,45,-90,45,0,0")
    parser.add_argument(
        "--robot-set", action="append", default=[], metavar="FIELD=JSON",
        help="override a Rby1Config field, e.g. --robot-set ee_arm_stiffness='[1075,990,580,265,190,205,300]'",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    host = args.address.rsplit(":", 1)[0]
    if host not in LOCAL_HOSTS:
        parser.error(f"{args.address} is not a local address: this script only drives a simulator on this machine")
    scenarios = [s for s in args.scenarios.split(",") if s]
    unknown = sorted(set(scenarios) - set(SCENARIOS))
    if unknown:
        parser.error(f"unknown scenarios {unknown}")
    if "loss" in scenarios and "arms" not in scenarios:
        parser.error("the loss scenario needs arms (it starts from the squeezed arms)")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from isaac_operator import Operator, ScriptedSession
    from lerobot_robot_rby1 import Rby1, Rby1Config
    from lerobot_teleoperator_rby1.isaac_teleop import teleop_rby1_xr as mod
    from lerobot_teleoperator_rby1.isaac_teleop.config_isaac_teleop import Rby1XRConfig

    # The XR session is scripted: no isaaccapture pipeline is built.
    mod.build_pipeline = lambda *, with_body: ("scripted", True)
    mod.build_external_inputs = lambda anchor: {}
    mod.print_xr_connect_help = lambda: None

    op = Operator()
    robot_config: dict[str, Any] = dict(
        address=args.address,
        action_mode="ee",
        use_torso=True,
        use_head=True,
        use_mobile_base=True,
        use_gripper=False,  # the simulator's gripper is not driven through the SDK
        ee_ready_torso_deg=[float(v) for v in args.ready_torso_deg.split(",")],
        ee_auto_action_period=args.ee_auto_action_period == "true",
    )
    for item in args.robot_set:
        key, _, value = item.partition("=")
        robot_config[key] = json.loads(value)
    robot = Rby1(Rby1Config(**robot_config))
    teleop = mod.Rby1XR(
        Rby1XRConfig(auto_launch_cloudxr=False, robot_address=args.address, use_gripper=False, status_log_period_s=2.0),
        session_factory=lambda pipeline: ScriptedSession(op.frame),
    )
    results: list[dict[str, Any]] = []
    state: StateLogger | None = None
    run: Run | None = None
    try:
        teleop.connect()
        robot.connect()  # powers on, enables control and moves to the ready pose
        model = str(getattr(teleop._reader, "model_name", "m"))
        state = StateLogger(args.address, model)
        state.start()
        time.sleep(0.5)
        run = Run(robot, teleop, op, args.fps)
        if "hold" in scenarios:
            scenario_hold(run, state, results, args.hold_seconds)
        else:
            run.hold(0.5, "warm-up")
        if "arms" in scenarios:
            scenario_arms(run, results)
        if "loss" in scenarios:
            scenario_loss(run, results)
        if "drive" in scenarios:
            scenario_drive(run, results, holonomic=model.lower() != "a")
        if "stop" in scenarios:
            scenario_stop(run, results)
        if "return" in scenarios:
            scenario_return(run, results)
    finally:
        if state is not None:
            state.stop()
        for device in (teleop, robot):
            try:
                device.disconnect()
            except Exception as e:  # noqa: BLE001
                logger.warning("disconnect failed: %r", e)

    if run is not None and run.rows:
        keys = sorted({k for row in run.rows for k in row})
        with open(out / "ticks.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(run.rows)
    if state is not None:
        t, q, qd = state.arrays()
        np.savez_compressed(out / "state.npz", t=t, q=q, qd=qd, names=np.asarray(state.names))
    summary = {
        "address": args.address,
        "scenarios": scenarios,
        "ee_auto_action_period": args.ee_auto_action_period == "true",
        "robot_set": args.robot_set,
        "ticks": 0 if run is None else len(run.rows),
        "results": results,
        "passed": bool(results) and all(r["ok"] for r in results),
    }
    (out / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
