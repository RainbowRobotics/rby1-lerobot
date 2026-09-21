import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lerobot_teleoperator_rb.constants import (
    DEFAULT_READY_POSE_RAD,
    VR_HOME_POSE_RAD,
)
from lerobot_teleoperator_rb.rb10e_kinematics import (
    RB10E,
)


def joint_limits_rad(kinematics: RB10E) -> np.ndarray:
    return np.column_stack(
        (kinematics._q_min, kinematics._q_max)
    )


def position_residual_mm(
    kinematics: RB10E,
    target: np.ndarray,
) -> float:
    return float(
        np.linalg.norm(
            target[:3, 3] - kinematics.get_fk()[:3, 3]
        )
    )


def test_initial_fk_and_jacobian_are_valid():
    kinematics = RB10E()

    assert kinematics.get_q().shape == (6,)
    assert kinematics.get_fk().shape == (4, 4)
    assert kinematics.get_jacobian().shape == (6, 6)

    assert np.all(np.isfinite(kinematics.get_fk()))
    assert np.all(np.isfinite(kinematics.get_jacobian()))

    assert np.allclose(
        kinematics.get_fk()[3],
        [0.0, 0.0, 0.0, 1.0],
    )
    assert np.isclose(
        np.linalg.det(kinematics.get_fk()[:3, :3]),
        1.0,
        atol=1e-6,
    )


def test_current_fk_target_keeps_same_joint_pose():
    """IKLM(q, FK(q), n) == q.

    This fixed-point property is what guarantees that engaging the clutch
    with a zero position delta never moves the arm. RbVr._begin_following
    depends on it, so keep this test.
    """
    kinematics = RB10E()

    q_before = kinematics.get_q()
    target = kinematics.get_fk()

    q_after = kinematics.IKLM(q_before, target, 5)

    assert np.allclose(q_after, q_before, atol=1e-10)
    assert position_residual_mm(kinematics, target) == pytest.approx(
        0.0, abs=1e-9
    )


def test_offset_target_converges():
    kinematics = RB10E()

    target = kinematics.get_fk()
    target[0, 3] += 10.0

    q_after = kinematics.IKLM(
        kinematics.get_q(),
        target,
        10,
    )

    assert np.all(np.isfinite(q_after))
    assert np.allclose(
        kinematics.get_fk()[:3, 3],
        target[:3, 3],
        atol=1e-3,
    )
    assert position_residual_mm(kinematics, target) < 1e-3


def test_warm_seed_converges_within_the_configured_iteration_budget():
    """A 100 mm step converges inside the default ik_iterations budget.

    This is the numeric justification for warm-starting each following tick
    from the previous solution instead of re-seeding from the measurement.
    """
    kinematics = RB10E()

    target = kinematics.get_fk()
    target[0, 3] += 100.0

    kinematics.IKLM(kinematics.get_q(), target, 5)

    assert position_residual_mm(kinematics, target) < 1.0


def test_solution_respects_joint_limits():
    kinematics = RB10E()

    target = kinematics.get_fk()
    target[1, 3] += 50.0

    q_after = kinematics.IKLM(
        kinematics.get_q(),
        target,
        10,
    )
    limits = joint_limits_rad(kinematics)

    assert np.all(q_after >= limits[:, 0])
    assert np.all(q_after <= limits[:, 1])


def test_seed_is_clipped_to_joint_limits():
    kinematics = RB10E()

    kinematics.set_q(
        np.full(6, 1000.0, dtype=np.float64)
    )

    limits = joint_limits_rad(kinematics)

    assert np.all(kinematics.get_q() <= limits[:, 1])
    assert np.all(kinematics.get_q() >= limits[:, 0])


def test_invalid_target_pose_is_rejected():
    kinematics = RB10E()

    with pytest.raises(ValueError):
        kinematics.IKLM(
            kinematics.get_q(),
            np.eye(3),
            5,
        )

    non_finite = np.eye(4)
    non_finite[0, 3] = np.nan

    with pytest.raises(ValueError):
        kinematics.IKLM(
            kinematics.get_q(),
            non_finite,
            5,
        )


def test_non_positive_iteration_count_is_rejected():
    kinematics = RB10E()

    with pytest.raises(ValueError):
        kinematics.IKLM(
            kinematics.get_q(),
            kinematics.get_fk(),
            0,
        )


def test_ik_seed_is_not_the_default_ready_pose():
    """The RB10E IK seed and DEFAULT_READY_POSE_DEG are different poses.

    constants.DEFAULT_READY_POSE_DEG claims in its comment to be "the
    existing startup / IK seed pose", but the solver seed at
    RB10E.__init__ is a different joint vector. Lock the distinction down
    so the two are not silently conflated.
    """
    kinematics = RB10E()

    assert np.allclose(
        np.rad2deg(kinematics.get_q()),
        [40.0, 10.0, -100.0, 50.0, -90.0, 0.0],
    )
    assert not np.allclose(
        kinematics.get_q(),
        np.asarray(DEFAULT_READY_POSE_RAD),
    )


def test_flange_normal_is_the_y_column():
    """Joint 5 spins about fk[:, 1], so that column is the flange normal.

    The whole model puts joint axes on the Y column rather than Z. Reading
    fk[:, 2] as the approach axis is off by 90 degrees, which is exactly the
    mistake that would silently pick a wrong home pose.
    """
    base = [0.0, -20.0, 110.0, 0.0, 90.0, 0.0]

    columns = {0: [], 1: [], 2: []}
    positions = []

    for roll_deg in (0.0, 30.0, 60.0, 90.0):
        q = list(base)
        q[5] = roll_deg

        kinematics = RB10E()
        kinematics.set_q(np.deg2rad(q))
        fk = kinematics.get_fk()

        for index in columns:
            columns[index].append(fk[:3, index])

        positions.append(fk[:3, 3])

    # The Y column is invariant under joint 5 ...
    assert np.allclose(
        columns[1],
        columns[1][0],
        atol=1e-12,
    )

    # ... while the other two rotate with it.
    assert not np.allclose(columns[0], columns[0][0], atol=1e-6)
    assert not np.allclose(columns[2], columns[2][0], atol=1e-6)

    # No tool offset is modelled, so joint 5 does not move the flange.
    assert np.allclose(positions, positions[0], atol=1e-9)


def test_gripper_approach_is_the_negated_y_column():
    """Wrist -> flange runs along -fk[:, 1], a fixed 259.3 mm."""
    for q_deg in (
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -20.0, 110.0, 0.0, 90.0, 0.0],
        [40.0, 10.0, -100.0, 50.0, -90.0, 0.0],
    ):
        kinematics = RB10E()
        kinematics.set_q(np.deg2rad(q_deg))

        offset = kinematics.T5[:3, 3] - kinematics.T4[:3, 3]
        distance = float(np.linalg.norm(offset))

        assert np.isclose(distance, 259.3, atol=1e-6)
        assert np.allclose(
            offset / distance,
            -kinematics.get_fk()[:3, 1],
            atol=1e-9,
        )


# ---------------------------------------------------------------------------
# Rotation targets
# ---------------------------------------------------------------------------
#
# Every convergence test above perturbs the TRANSLATION only. Once RbVr
# started commanding a scaled orientation delta, that left the orientation
# half of IKLM untested, which matters more than it looks: update_error_vector
# writes the rotation rows as 2*sin(theta)*axis, dimensionless against the
# millimetres in rows 0-2, and IKLM's LM damping mixes both into one scalar.
# The concern was that orientation would converge far more slowly than
# position. It does not -- these tests pin that.


def orientation_residual_deg(
    kinematics: RB10E,
    target: np.ndarray,
) -> float:
    error = target[:3, :3] @ kinematics.get_fk()[:3, :3].T
    return float(
        np.rad2deg(
            np.linalg.norm(Rotation.from_matrix(error).as_rotvec())
        )
    )


def rotated_target(kinematics: RB10E, axis, angle_deg: float) -> np.ndarray:
    """The anchor pose with a base-frame rotation pre-multiplied on.

    Pre-multiplication matches how RbVr builds its target and how
    update_error_vector measures the error.
    """
    target = kinematics.get_fk()
    target[:3, :3] = (
        Rotation.from_rotvec(
            np.deg2rad(angle_deg) * np.asarray(axis, dtype=float)
        ).as_matrix()
        @ target[:3, :3]
    )
    return target


@pytest.mark.parametrize("axis", [[0, 0, 1], [1, 0, 0], [0, 1, 0]])
@pytest.mark.parametrize("angle_deg", [18.0, 45.0])
def test_rotation_target_converges_from_the_vr_home_pose(axis, angle_deg):
    """Warm-started at the control rate, with the configured budget.

    Five iterations per tick is RbVrConfig.ik_iterations, so ten passes of
    the loop below is a third of a second at 30 Hz.

    WHICH AXIS IS CHEAP DEPENDS ON THE HOME ORIENTATION:

        base X   nearly free. It is the flange spin axis fk[:3, 1] at this
                 pose, so the command is one joint-5 turn: 18 deg leaves
                 1.4 percent after a single tick.
        base Y   about 31 percent left after one tick, and the axis that
                 eats joint 2's remaining travel.
        base Z   about 29 percent left, and the axis that eats wrist
                 margin.

    THE BOUND BELOW IS MEASURED, not inherited. Ten ticks from this home
    leave, worst case, 2.3e-2 deg (45 deg about base Z); base X converges
    exactly and base Y leaves 1.6e-4 deg. The bound is 1e-1 deg, i.e. under
    two tenths of a millimetre of arc at 250 mm, with roughly a factor of
    four of headroom. Re-measure it if the home pose moves again -- an
    earlier home failed this test outright at 45 deg about base Z (9.1e-2
    deg against a 1e-2 bound).
    """
    kinematics = RB10E()
    kinematics.set_q(np.array(VR_HOME_POSE_RAD))

    target = rotated_target(kinematics, axis, angle_deg)

    for _ in range(10):
        kinematics.IKLM(kinematics.get_q(), target, 5)

    assert orientation_residual_deg(kinematics, target) < 1e-1
    assert position_residual_mm(kinematics, target) < 1e-3


def test_rotation_target_makes_real_progress_in_a_single_tick():
    """One tick must not stall, or the arm would lag the hand indefinitely.

    Base Y is deliberately the EXPENSIVE axis from this home pose -- the
    free wrist spin is base X -- so this is the worst case, not the best.
    Measured: 18 deg about base Y comes down to 5.6 deg in one tick, so the
    bound below is 40 percent. The cheap axis is far tighter (base X leaves
    1.4 percent), which is why the axis here is not a parameter: this test
    exists to pin the slowest one.
    """
    kinematics = RB10E()
    kinematics.set_q(np.array(VR_HOME_POSE_RAD))

    target = rotated_target(kinematics, [0, 1, 0], 18.0)

    kinematics.IKLM(kinematics.get_q(), target, 5)

    assert orientation_residual_deg(kinematics, target) < 0.40 * 18.0


def test_rotation_target_does_not_disturb_the_position():
    """A pure rotation command must not translate the flange."""
    kinematics = RB10E()
    kinematics.set_q(np.array(VR_HOME_POSE_RAD))

    anchor_position = kinematics.get_fk()[:3, 3].copy()
    target = rotated_target(kinematics, [0, 1, 0], 45.0)

    for _ in range(10):
        kinematics.IKLM(kinematics.get_q(), target, 5)

    assert np.allclose(
        kinematics.get_fk()[:3, 3],
        anchor_position,
        atol=1e-3,
    )


def test_rotation_target_turns_the_gripper_approach_axis():
    """The approach axis is -fk[:3, 1], not the Z column.

    Every joint axis lives on its frame's Y column in this model and the
    tool transform is identity, so assuming "Z out of the flange" picks a
    pose 90 degrees off.

    The command axis is base Z on purpose. At this home pose the approach
    sits on base -X, so a base-X command would rotate it onto itself and
    the assertion below would hold for a solver that did nothing at all.
    """
    kinematics = RB10E()
    kinematics.set_q(np.array(VR_HOME_POSE_RAD))

    approach_before = -kinematics.get_fk()[:3, 1].copy()
    assert np.allclose(approach_before, [-1.0, 0.0, 0.0], atol=1e-9)

    target = rotated_target(kinematics, [0, 0, 1], 30.0)

    for _ in range(10):
        kinematics.IKLM(kinematics.get_q(), target, 5)

    approach_after = -kinematics.get_fk()[:3, 1]
    expected = (
        Rotation.from_rotvec(np.deg2rad(30.0) * np.array([0.0, 0.0, 1.0]))
        .as_matrix()
        @ approach_before
    )

    assert np.allclose(approach_after, expected, atol=1e-4)
