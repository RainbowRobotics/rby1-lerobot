"""Where an arm's operator shoulder comes from, and what happens when that tracking is lost.

With ``shoulder_source="auto"`` an arm takes the tracked body shoulder when it
is valid as the arm first engages, else the estimate from the headset. It then
keeps that source until the next Right A: when the source is lost the arm holds
instead of changing over to the other one, which would move the shoulder — and
the hand target with it — by several centimetres.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from rby1_xr_teleop.scripted import HAND_OUT, HAND_REST
from test_core_scenario import CoreSession

GRIPS = dict(right_squeeze=0.9, left_squeeze=0.9)
BOTH = ("right", "left")


def _xyz(action: dict[str, float], prefix: str = "right_ee") -> np.ndarray:
    return np.array([action[f"{prefix}.{axis}"] for axis in "xyz"])


def _set(session: CoreSession, **fields) -> None:
    for name, value in fields.items():
        setattr(session.op, name, value)


def _press_right_a(session: CoreSession) -> None:
    session.op.right_a = True
    session.tick("right-a")
    session.op.right_a = False


def _started(*, body_tracked: bool = True, **settings) -> CoreSession:
    """A session after the first Right A (arms out for the reach), hands back at rest, no grip."""
    s = CoreSession(**settings)
    s.op.body_tracked = body_tracked
    s.tick("connect")
    s.move(10, "arms-out", right_hand=HAND_OUT, left_hand=HAND_OUT)
    s.hold(18, "arms-out")
    _press_right_a(s)
    assert s.core.started
    s.move(12, "rest", right_hand=HAND_REST, left_hand=HAND_REST)
    return s


def _engaged(s: CoreSession) -> dict[str, float]:
    """Squeeze both grips and wait until the engage ramp is over."""
    _set(s, **GRIPS)
    action = s.hold(70, "engage")
    assert all(s.core._clutch[side].engaged and s.core._abs_state[side] == "track" for side in BOTH)
    return action


def test_arm_holds_when_its_body_shoulder_is_lost(caplog) -> None:
    s = _started()
    assert s.core._shoulder_locked == {"right": None, "left": None}  # chosen at the first engage
    a = _engaged(s)
    assert s.core._shoulder_locked == {"right": "body", "left": "body"}

    s.op.body_tracked = False
    with caplog.at_level(logging.WARNING, logger="rby1_xr_teleop"):
        # Up to abs_hold_s the arm goes on with the shoulder where it was: no shift.
        b = s.hold(25, "no-body")
        assert s.core._clutch["right"].engaged and s.core._clutch["left"].engaged
        for prefix in ("right_ee", "left_ee"):
            assert _xyz(b, prefix) == pytest.approx(_xyz(a, prefix), abs=1e-6)
        # After it the arm holds; the headset (still tracked) does not take over.
        b = s.hold(10, "no-body")
    assert not s.core._clutch["right"].engaged and not s.core._clutch["left"].engaged
    assert s.core._abs_state["right"] == s.core._abs_state["left"] == "hold (no body)"
    assert _xyz(b) == pytest.approx(_xyz(a), abs=1e-6)
    for side in BOTH:  # said once per arm, not on every tick
        assert sum(f"{side} arm holds" in record.getMessage() for record in caplog.records) == 1

    # The hand moves while the arm holds: the target stays.
    s.move(10, "no-body", right_hand=(0.40, 0.0, -0.20))
    c = s.tick("no-body")
    assert _xyz(c) == pytest.approx(_xyz(b), abs=1e-12)

    # Body tracking returns: the arm ramps to the hand, with the body shoulder as before.
    s.op.body_tracked = True
    c = s.hold(70, "body-back")
    assert s.core._clutch["right"].engaged and s.core._shoulder_locked["right"] == "body"
    mapper = s.core._abs["right"]
    moved = (mapper.robot_reach / mapper.calibrated_reach) * np.array([0.15, 0.0, 0.15])
    assert _xyz(c) - _xyz(a) == pytest.approx(moved, abs=2e-3)


def test_headset_shoulder_is_kept_when_body_tracking_appears_and_chosen_again_on_right_a() -> None:
    s = _started(body_tracked=False)
    a = _engaged(s)
    assert s.core._shoulder_locked == {"right": "headset", "left": "headset"}

    # Body tracking starts mid-session: the arms keep the headset estimate, nothing shifts.
    s.op.body_tracked = True
    b = s.hold(15, "body-appears")
    assert s.core._shoulder_locked == {"right": "headset", "left": "headset"}
    for prefix in ("right_ee", "left_ee"):
        assert _xyz(b, prefix) == pytest.approx(_xyz(a, prefix), abs=1e-9)

    # Right A (return to the start pose) starts over: now the body shoulder is taken.
    _press_right_a(s)
    s.hold(80, "after-right-a")
    assert s.core._shoulder_locked == {"right": "body", "left": "body"}
    assert s.core._clutch["right"].engaged


def test_arm_on_the_headset_shoulder_holds_when_the_headset_is_lost() -> None:
    s = _started(body_tracked=False)
    a = _engaged(s)
    _set(s, body_tracked=True, head_tracked=False)  # the body does not take over either
    b = s.hold(35, "no-headset")
    assert not s.core._clutch["right"].engaged and s.core._abs_state["right"] == "hold (no headset)"
    assert _xyz(b) == pytest.approx(_xyz(a), abs=1e-6)
    s.op.head_tracked = True
    s.hold(70, "headset-back")
    assert s.core._clutch["right"].engaged and s.core._shoulder_locked["right"] == "headset"


@pytest.mark.parametrize("source", ["body", "headset"])
def test_explicit_shoulder_source_is_used_from_the_start(source: str) -> None:
    s = _started(shoulder_source=source)
    _engaged(s)
    assert s.core._shoulder_locked == {"right": source, "left": source}
