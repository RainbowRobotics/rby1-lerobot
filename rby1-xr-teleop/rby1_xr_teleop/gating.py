"""What may follow the operator in a given mode — one truth table for every feature.

The teleoperator's mode is five facts (started, stopped, returning, a grip
held, body drive active). Whether the arms, the torso, the head, the grippers
or the base may move is decided here and nowhere else, so a new mode cannot be
forgotten in one of the features.

Pure functions, no state: ``tests/test_isaac_gating.py`` checks all 32 modes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Mode:
    started: bool      # Right A pressed once (always True without start_on_right_a)
    stopped: bool      # Right B: everything frozen until Right A
    returning: bool    # the follower runs its return-to-start motion
    grip_held: bool    # at least one grip squeezed past the clutch threshold
    body_drive: bool   # Left X body drive is active

    @property
    def active(self) -> bool:
        """Teleoperation is live: started, not stopped, not returning."""
        return self.started and not self.stopped and not self.returning


@dataclass(frozen=True)
class Gates:
    arms: bool             # an arm may follow its controller (its own dead-man still applies)
    posture_hints: bool    # the arm posture hints may be updated
    torso: bool            # the torso may follow the chest (torso_engage still applies)
    head: bool             # the head may follow the headset
    grippers: bool         # the grippers may follow the triggers
    base: bool             # the base may move at all
    base_sticks: bool      # … driven by the thumbsticks
    body_drive: bool       # the body-drive mode may be entered / may continue
    arm_hold: str          # status text of an arm that holds because of the mode
    torso_hold: str | None  # why the torso holds because of the mode (None: it may follow)


def compute_gates(
    mode: Mode,
    *,
    head_base_require_grip: bool = True,
    use_mobile_base: bool = True,
    body_drive_enabled: bool = True,
) -> Gates:
    active = mode.active
    head_base = active and (mode.grip_held or not head_base_require_grip)
    base = use_mobile_base and head_base
    if mode.body_drive:
        torso_hold: str | None = "base drive (Left X)"
    elif not mode.started:
        torso_hold = "not started (press Right A)"
    elif mode.stopped:
        torso_hold = "stopped (Right B)"
    elif mode.returning:
        torso_hold = "returning to start pose (Right A)"
    else:
        torso_hold = None
    return Gates(
        arms=active,
        posture_hints=active,
        torso=torso_hold is None,
        head=head_base,
        # The grippers only wait for the start: they keep following the
        # triggers while stopped (Right B) and during a return.
        grippers=mode.started,
        base=base,
        base_sticks=base and not mode.body_drive,
        body_drive=body_drive_enabled and base,
        arm_hold="hold" if mode.started else "hold (press Right A to start)",
        torso_hold=torso_hold,
    )
