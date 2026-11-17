"""Every one of the 32 teleoperation modes against the rules, stated independently."""

from __future__ import annotations

import itertools

import pytest

from lerobot_teleoperator_rby1.isaac_teleop.gating import Mode, compute_gates

FLAGS = ("started", "stopped", "returning", "grip_held", "body_drive")
MODES = [Mode(**dict(zip(FLAGS, bits))) for bits in itertools.product((False, True), repeat=5)]


@pytest.mark.parametrize("mode", MODES, ids=lambda m: "".join(f[0].upper() if getattr(m, f) else "-" for f in FLAGS))
@pytest.mark.parametrize("require_grip", [True, False])
def test_gates(mode: Mode, require_grip: bool) -> None:
    g = compute_gates(mode, head_base_require_grip=require_grip)
    live = mode.started and not mode.stopped and not mode.returning
    assert mode.active == live

    # Arms and posture hints: live teleoperation only.
    assert g.arms == live and g.posture_hints == live
    # The torso additionally holds during a body drive.
    assert g.torso == (live and not mode.body_drive)
    assert (g.torso_hold is None) == g.torso
    # Head and base: live, and a grip held when that is required.
    gripped = live and (mode.grip_held or not require_grip)
    assert g.head == gripped and g.base == gripped
    assert g.base_sticks == (gripped and not mode.body_drive)
    assert g.body_drive == gripped
    # The grippers only wait for the start.
    assert g.grippers == mode.started

    # Nothing but the grippers may move unless teleoperation is live.
    if not live:
        assert not (g.arms or g.posture_hints or g.torso or g.head or g.base or g.base_sticks or g.body_drive)


def test_hold_reasons_and_feature_switches() -> None:
    def gates(**flags):
        mode = Mode(**{**dict.fromkeys(FLAGS, False), **flags})
        return compute_gates(mode)

    assert gates().arm_hold == "hold (press Right A to start)"
    assert gates(started=True, stopped=True).arm_hold == "hold"
    assert gates().torso_hold == "not started (press Right A)"
    assert gates(started=True, stopped=True).torso_hold == "stopped (Right B)"
    assert gates(started=True, returning=True).torso_hold == "returning to start pose (Right A)"
    # A body drive is reported first, whatever else holds the torso.
    assert gates(started=True, grip_held=True, body_drive=True).torso_hold == "base drive (Left X)"
    assert gates(body_drive=True).torso_hold == "base drive (Left X)"

    live = Mode(started=True, stopped=False, returning=False, grip_held=True, body_drive=False)
    g = compute_gates(live, use_mobile_base=False)
    assert g.head and not (g.base or g.base_sticks or g.body_drive)
    g = compute_gates(live, body_drive_enabled=False)
    assert g.base and g.base_sticks and not g.body_drive
