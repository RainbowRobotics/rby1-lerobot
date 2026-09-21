"""Offline replay of the anchored delta control law. No robot, no headset.

Drives RbVr with a synthetic 30 Hz hand trajectory. No robot, no headset.

Phase 1 walks a 300 mm diameter circle in the frozen torso XY plane with the
hand's ROTATION held at identity, so the 1:1 position scale can be checked
and any rotation drift is a leak from translation into orientation.

Phase 2 holds the hand still and sweeps its rotation, so the scaled
orientation law can be checked against orientation_scale.

Together these are the only pre-hardware check of the full control law.

    python tests/manual/replay_delta.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_rb_vr_state_machine import (  # noqa: E402
    FakeClock,
    FakeReceiver,
    FakeRobot,
    START_Q_DEG,
    q_of,
)

from lerobot_teleoperator_rb import RbVr, RbVrConfig  # noqa: E402
from lerobot_teleoperator_rb import rb_vr as rb_vr_module  # noqa: E402
from lerobot_teleoperator_rb.frame_transforms import (  # noqa: E402
    controller_translation_delta_mm,
)
from lerobot_teleoperator_rb.rb10e_kinematics import RB10E  # noqa: E402


TICK_S = 1.0 / 30.0
RADIUS_MM = 150.0
REVOLUTION_TICKS = 300
ROTATION_SETTLE_TICKS = 400


def main() -> int:
    clock = FakeClock()
    rb_vr_module.time.monotonic = clock

    robot = FakeRobot(np.deg2rad(START_Q_DEG))
    rb_vr_module.get_active_rb_cobot = lambda: robot

    teleop = RbVr(RbVrConfig(id="replay", send_handshake=False))
    teleop._receiver = FakeReceiver()
    teleop._kinematics = RB10E()
    teleop._is_connected = True
    receiver = teleop._receiver

    # Home, then engage the clutch with the hand at the origin.
    receiver.press_primary()
    teleop.get_action()
    for _ in range(int(RbVr.HOMING_DURATION_S / TICK_S) + 5):
        clock.advance(TICK_S)
        teleop.get_action()

    receiver.set_grip(1.0)
    clock.advance(TICK_S)
    teleop.get_action()

    anchor = teleop._anchor_ee_pose.copy()
    probe = RB10E()

    print(
        f"{'tick':>5} {'state':<10} {'|delta|':>8} "
        f"{'|ee-anchor|':>12} {'rot err':>9} {'ik resid':>9}"
    )

    radii = []
    travels = []
    rotation_errors = []
    residuals = []

    # The hand's rotation is deliberately left at identity for this phase,
    # so any orientation drift below is translation leaking into rotation.
    for tick in range(REVOLUTION_TICKS + 1):
        angle = 2.0 * np.pi * tick / REVOLUTION_TICKS
        hand = np.array(
            [
                RADIUS_MM * np.sin(angle),
                RADIUS_MM * (1.0 - np.cos(angle)),
                0.0,
            ]
        )
        receiver.set_controller_position_mm(hand)

        clock.advance(TICK_S)
        action = teleop.get_action()

        probe.set_q(q_of(action))
        reached = probe.get_fk()

        travelled = float(
            np.linalg.norm(reached[:3, 3] - anchor[:3, 3])
        )
        rotation_error = float(
            np.max(np.abs(reached[:3, :3] - anchor[:3, :3]))
        )
        # The delta is expressed in the FROZEN torso frame, so the target
        # is the anchor plus the torso-rotated hand translation.
        delta_mm = controller_translation_delta_mm(
            receiver.state.right.pose_rb,
            teleop._anchor_controller_pose,
            teleop._operator_torso_pose,
        )
        target = anchor[:3, 3] + delta_mm
        residual = float(
            np.linalg.norm(reached[:3, 3] - target)
        )

        if tick % 25 == 0:
            print(
                f"{tick:>5} {teleop._control_state.value:<10} "
                f"{np.linalg.norm(hand):>8.1f} {travelled:>12.1f} "
                f"{rotation_error:>9.2e} {residual:>9.2f}"
            )

        # Skip the first quarter revolution: the joint-step clamp needs a
        # few ticks to catch up with the commanded circle.
        if tick > REVOLUTION_TICKS // 4:
            radii.append(np.linalg.norm(hand))
            travels.append(travelled)
            rotation_errors.append(rotation_error)
            residuals.append(residual)

    print()
    print(f"commanded hand radius     : {RADIUS_MM:.1f} mm")
    print(
        "orientation drift, hand not rotating: "
        f"{max(rotation_errors):.3e} (target: ~0)"
    )
    max_residual = max(residuals)
    print(
        "max tracking error        : "
        f"{max_residual:.2f} mm"
    )
    print(
        "hand vs EE travel (1:1)   : "
        f"max mismatch {max(abs(np.array(radii) - np.array(travels))):.3f} mm"
    )

    # 5e-3 on a rotation-matrix entry is about 0.3 deg, which is IK
    # convergence noise at 5 iterations per tick.
    ok = max(rotation_errors) < 5e-3 and max_residual < 5.0

    ok = replay_rotation(teleop, clock, receiver, anchor) and ok

    print()
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def replay_rotation(teleop, clock, receiver, anchor):
    """Sweep the hand's rotation with its position held at the anchor."""
    gain = teleop._config.orientation_scale
    axis = np.array([0.0, 0.0, 1.0])

    receiver.set_controller_position_mm(np.zeros(3))
    probe = RB10E()

    print()
    print(f"Phase 2: hand rotation sweep, orientation_scale = {gain:.2f}")
    print(f"{'hand deg':>9} {'expected':>9} {'achieved':>9} {'err':>8} {'':>6}")

    # Tracking is exact while the elbow has room. Past roughly 110 deg of
    # hand rotation about the worst axis, joint 2 reaches full extension and
    # the IK settles a few degrees off. That is a singular configuration,
    # not an iteration shortage -- running it harder makes it worse -- so
    # the check is split rather than blanket-loosened.
    exact_below_deg = 110.0
    worst_exact = 0.0
    worst_large = 0.0

    for hand_deg in (0.0, 15.0, 30.0, 60.0, 90.0, 150.0):
        receiver.set_controller_rotation(
            Rotation.from_rotvec(np.deg2rad(hand_deg) * axis).as_matrix()
        )

        # The joint-step clamp needs time; a third of a second is plenty.
        for _ in range(ROTATION_SETTLE_TICKS):
            clock.advance(TICK_S)
            action = teleop.get_action()

        probe.set_q(q_of(action))
        delta = probe.get_fk()[:3, :3] @ anchor[:3, :3].T
        achieved = float(
            np.rad2deg(
                np.linalg.norm(Rotation.from_matrix(delta).as_rotvec())
            )
        )
        # No clamp: gain is the only bound. The hand delta itself cannot
        # exceed 180 deg because as_rotvec always reports the short way
        # round, so the command tops out at gain * 180.
        expected = gain * min(hand_deg, 360.0 - hand_deg)
        error = abs(achieved - expected)

        if hand_deg <= exact_below_deg:
            worst_exact = max(worst_exact, error)
            note = ""
        else:
            worst_large = max(worst_large, error)
            note = "elbow"

        print(
            f"{hand_deg:>9.1f} {expected:>9.2f} {achieved:>9.2f} "
            f"{error:>8.3f} {note:>6}"
        )

    ceiling = gain * 180.0

    print()
    print(
        f"max error below {exact_below_deg:.0f} deg : "
        f"{worst_exact:.3f} deg (target < 1.0)"
    )
    print(
        "max error above           : "
        f"{worst_large:.3f} deg (target < 8.0, elbow near full extension)"
    )
    print(f"command ceiling           : {ceiling:.1f} deg (gain * 180)")

    return worst_exact < 1.0 and worst_large < 8.0


if __name__ == "__main__":
    raise SystemExit(main())
