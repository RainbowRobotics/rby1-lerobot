"""Time one ``rby1_isaac`` tick (``Rby1XR.get_action``) with a scripted operator.

Not a test (pytest does not collect it). Measures the teleoperator's own
computation with both arms clutched and the posture hints on — no XR session,
no robot I/O:

    python tests/bench_isaac_tick.py [--ticks 3000] [--profile]

``--profile`` adds the functions with the largest cumulative time.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import logging
import math
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import isaac_fakes as fakes  # noqa: E402
from isaac_operator import HAND_OUT, HAND_REST, Operator  # noqa: E402
from lerobot_teleoperator_rby1.isaac_teleop import teleop_rby1_xr as mod  # noqa: E402
from lerobot_teleoperator_rby1.isaac_teleop.config_isaac_teleop import Rby1XRConfig  # noqa: E402


def build():
    mod.build_pipeline = lambda *, with_body: ("pipeline", True)
    mod.build_external_inputs = lambda anchor: {}
    mod.print_xr_connect_help = lambda: None
    op = Operator(right_hand=HAND_OUT, left_hand=HAND_OUT)
    session = fakes.FakeSession()  # frames are built outside the timed call
    session.push(op.frame())
    readers: list[fakes.FakeStateReader] = []

    def reader_factory(address, model):
        readers.append(fakes.FakeStateReader(address, model))
        return readers[-1]

    cfg = Rby1XRConfig(auto_launch_cloudxr=False, status_log_period_s=0.0, engage_ramp_s=0.05)
    teleop = mod.Rby1XR(cfg, session_factory=lambda pipeline: session, state_reader_factory=reader_factory)
    teleop.connect()
    robot = fakes.FollowingRobot(readers[0], dt=1.0 / 60.0)

    def tick() -> None:
        session.push(op.frame())
        robot.apply(teleop.get_action())

    for _ in range(40):  # arms out long enough for the reach calibration
        tick()
        time.sleep(0.02)
    op.right_a = True
    tick()
    op.right_a = False
    op.right_hand = op.left_hand = HAND_REST
    op.right_squeeze = op.left_squeeze = 0.9
    for _ in range(10):
        tick()
        time.sleep(0.01)
    assert teleop.is_started and teleop._clutch["right"].engaged and teleop._clutch["left"].engaged
    return teleop, op, robot, session


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ticks", type=int, default=3000)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    teleop, op, robot, session = build()

    def frames():
        for i in range(args.ticks):
            w = 0.03 * math.sin(i * 0.05)
            op.right_hand = (HAND_REST[0] + w, HAND_REST[1], HAND_REST[2] + 0.5 * w)
            op.left_hand = (HAND_REST[0], HAND_REST[1] + w, HAND_REST[2])
            op.neck_deg = 10.0 * math.sin(i * 0.02)
            yield op.frame()

    def run_teleop_only() -> float:
        spent = 0.0
        for frame in frames():
            session.push(frame)
            t0 = time.perf_counter()
            action = teleop.get_action()
            spent += time.perf_counter() - t0
            robot.apply(action)
        return spent

    spent = run_teleop_only()
    print(f"get_action: {spent / args.ticks * 1e3:.3f} ms/tick over {args.ticks} ticks (both arms clutched, hints on)")
    if args.profile:
        prepared = list(frames())
        profiler = cProfile.Profile()
        for frame in prepared:
            session.push(frame)
            profiler.enable()
            action = teleop.get_action()
            profiler.disable()
            robot.apply(action)
        out = io.StringIO()
        pstats.Stats(profiler, stream=out).strip_dirs().sort_stats("cumulative").print_stats(22)
        print("\n".join(line[:118] for line in out.getvalue().splitlines()[4:32]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
