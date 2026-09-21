import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lerobot_teleoperator_rb.constants import VR_HOME_POSE_RAD
from lerobot_teleoperator_rb.frame_transforms import (
    T_CONV,
    T_FOR_HEAD,
    T_FOR_RB10E,
    R_OPERATOR_TO_BASE,
    apply_scale,
    build_delta_target,
    build_position_delta_target,
    compute_user_scale,
    controller_rotation_delta,
    controller_translation_delta_mm,
    controller_pose_to_rb10e_target,
    head_pose_to_torso,
    invert_se3,
    pose_to_se3,
    quest_pose_to_rb,
    scale_rotation,
)
from lerobot_teleoperator_rb.rb10e_kinematics import RB10E


def test_pose_to_se3_identity_quaternion():
    pose = pose_to_se3(
        [0.1, -0.2, 0.3],
        [0.0, 0.0, 0.0, 1.0],
    )

    assert np.allclose(pose[:3, :3], np.eye(3))
    assert np.allclose(
        pose[:3, 3],
        [100.0, -200.0, 300.0],
    )


def test_invert_se3():
    pose = pose_to_se3(
        [0.1, 0.2, 0.3],
        [0.0, 0.0, 0.0, 1.0],
    )

    inverse = invert_se3(pose)

    assert np.allclose(
        inverse @ pose,
        np.eye(4),
        atol=1e-10,
    )


def test_quest_basis_conversion_matches_legacy_formula():
    position = [0.1, 0.2, 0.3]
    quaternion = [0.0, 0.0, 0.0, 1.0]

    quest_pose = pose_to_se3(position, quaternion)
    expected = T_CONV.T @ quest_pose @ T_CONV

    actual = quest_pose_to_rb(
        position,
        quaternion,
    )

    assert np.allclose(actual, expected)


def test_head_to_torso_matches_legacy_formula():
    head_pose = np.eye(4)

    expected = head_pose @ T_FOR_HEAD
    actual = head_pose_to_torso(head_pose)

    assert np.allclose(actual, expected)


def test_compute_user_scale():
    torso = np.eye(4)
    controller = np.eye(4)
    controller[0, 3] = 700.0

    scale = compute_user_scale(
        controller,
        torso,
        reference_reach_mm=1300.0,
    )

    assert np.isclose(scale, 1300.0 / 700.0)


def test_apply_scale_with_copy_does_not_mutate_input():
    pose = np.eye(4)
    pose[0, 3] = 100.0
    original = pose.copy()

    scaled = apply_scale(pose, 2.0, copy=True)

    assert np.allclose(pose, original)

    # The original RB10E height correction is commented out in apply_scale,
    # so target_z_offset_mm is accepted but ignored. Translation is scaled,
    # rotation is untouched.
    assert np.allclose(
        scaled[:3, 3],
        [200.0, 0.0, 0.0],
    )
    assert np.allclose(scaled[:3, :3], np.eye(3))


def test_apply_scale_mutates_in_place_by_default():
    pose = np.eye(4)
    pose[0, 3] = 100.0

    scaled = apply_scale(pose, 2.0)

    assert scaled is pose
    assert np.isclose(pose[0, 3], 200.0)


def test_controller_target_matches_legacy_formula():
    head_pose = pose_to_se3(
        [0.0, 0.0, 1.6],
        [0.0, 0.0, 0.0, 1.0],
    )
    controller_pose = pose_to_se3(
        [0.4, -0.2, 1.2],
        [0.0, 0.0, 0.0, 1.0],
    )

    torso_pose = head_pose @ T_FOR_HEAD
    user_scale = 1.8

    expected = (
        invert_se3(torso_pose)
        @ controller_pose
        @ T_FOR_RB10E
    )
    expected[:3, 3] *= user_scale
    expected[0, 3] += 600.0
    expected[1, 3] += 400.0
    expected[2, 3] += 900.0

    actual = controller_pose_to_rb10e_target(
        controller_pose,
        torso_pose,
        user_scale,
        target_x_offset_mm=600.0,
        target_y_offset_mm=400.0,
        target_z_offset_mm=900.0,
    )

    assert np.allclose(actual, expected)


# ---------------------------------------------------------------------------
# Anchored position-delta helpers
# ---------------------------------------------------------------------------


def rotation_z(angle_rad):
    transform = np.eye(4)
    cos = np.cos(angle_rad)
    sin = np.sin(angle_rad)
    transform[:3, :3] = [
        [cos, -sin, 0.0],
        [sin, cos, 0.0],
        [0.0, 0.0, 1.0],
    ]
    return transform


def test_build_position_delta_target_with_zero_delta_is_the_anchor():
    anchor = rotation_z(0.3)
    anchor[:3, 3] = [100.0, -200.0, 300.0]

    target = build_position_delta_target(anchor, np.zeros(3))

    assert np.allclose(target, anchor)


def test_build_position_delta_target_keeps_the_anchor_orientation():
    anchor = rotation_z(0.7)
    anchor[:3, 3] = [10.0, 20.0, 30.0]

    target = build_position_delta_target(
        anchor,
        [50.0, -60.0, 70.0],
    )

    assert np.allclose(target[:3, :3], anchor[:3, :3])
    assert np.allclose(
        target[:3, 3],
        [60.0, -40.0, 100.0],
    )


def test_build_position_delta_target_does_not_mutate_the_anchor():
    anchor = np.eye(4)
    anchor[:3, 3] = [1.0, 2.0, 3.0]
    original = anchor.copy()

    build_position_delta_target(anchor, [10.0, 10.0, 10.0])

    assert np.allclose(anchor, original)


def test_controller_translation_delta_is_zero_at_the_anchor():
    torso = rotation_z(0.4)
    torso[:3, 3] = [11.0, 22.0, 33.0]

    controller = rotation_z(-0.2)
    controller[:3, 3] = [400.0, 500.0, 600.0]

    delta = controller_translation_delta_mm(
        controller,
        controller,
        torso,
    )

    assert np.allclose(delta, np.zeros(3))


def test_controller_translation_delta_applies_the_torso_rotation():
    torso = rotation_z(np.pi / 2.0)

    anchor = np.eye(4)
    anchor[:3, 3] = [0.0, 0.0, 0.0]

    current = np.eye(4)
    current[:3, 3] = [100.0, 0.0, 0.0]

    delta = controller_translation_delta_mm(current, anchor, torso)

    # Rz(90).T @ [100, 0, 0] == [0, -100, 0] in the operator frame, then
    # R_OPERATOR_TO_BASE (Rz(-90)) carries it to [-100, 0, 0] in base.
    assert np.allclose(delta, [-100.0, 0.0, 0.0])


def test_operator_to_base_maps_operator_axes_onto_the_home_ee_axes():
    """Operator +X -> EE -Y, +Y -> EE +X, +Z -> EE +Z at VR_HOME_POSE.

    This is the operator-facing contract for the controls, and it is a
    JOINT property of two constants: the columns on the left are what the
    identity pipeline yields for operator forward / left / up (base -Y / +X
    / +Z), and the right-hand side reads the home EE rotation out of the
    kinematics rather than restating it, so re-tuning VR_HOME_POSE_DEG
    without re-deriving R_OPERATOR_TO_BASE fails here instead of on the
    hardware.

    The home EE frame is the base frame yawed -90 deg, so the right-hand
    side is base -X / -Y / +Z and R_OPERATOR_TO_BASE is Rz(-90 deg).
    """
    kinematics = RB10E()
    kinematics.set_q(np.asarray(VR_HOME_POSE_RAD))
    home_ee_rotation = kinematics.get_fk()[:3, :3]

    identity_pipeline = np.array(
        [
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    wanted = np.column_stack(
        [
            -home_ee_rotation[:, 1],
            home_ee_rotation[:, 0],
            home_ee_rotation[:, 2],
        ]
    )

    assert np.allclose(
        R_OPERATOR_TO_BASE @ identity_pipeline,
        wanted,
        atol=1e-12,
    )


def test_operator_to_base_is_a_proper_rotation():
    """Must preserve lengths AND handedness.

    A determinant of -1 here would mirror the operator's motion, which is
    the failure mode that looks like "the VR side is left-handed".
    """
    assert np.allclose(
        R_OPERATOR_TO_BASE @ R_OPERATOR_TO_BASE.T,
        np.eye(3),
    )
    assert np.isclose(np.linalg.det(R_OPERATOR_TO_BASE), 1.0)


def test_delta_preserves_length():
    torso = rotation_z(0.83)

    anchor = np.eye(4)
    anchor[:3, 3] = [12.0, -34.0, 56.0]

    current = np.eye(4)
    current[:3, 3] = [212.0, 66.0, -44.0]

    delta = controller_translation_delta_mm(current, anchor, torso)

    assert np.isclose(
        np.linalg.norm(delta),
        np.linalg.norm(current[:3, 3] - anchor[:3, 3]),
    )


def test_controller_translation_delta_ignores_torso_translation():
    """The frozen-torso design rests on this property.

    The operator may lean or walk during a grip stroke without the robot
    moving; only the hand's translation matters.
    """
    rotation = rotation_z(0.9)

    torso_a = rotation.copy()
    torso_a[:3, 3] = [0.0, 0.0, 0.0]

    torso_b = rotation.copy()
    torso_b[:3, 3] = [1234.0, -567.0, 890.0]

    anchor = np.eye(4)
    anchor[:3, 3] = [10.0, 20.0, 30.0]

    current = np.eye(4)
    current[:3, 3] = [110.0, -80.0, 30.0]

    assert np.allclose(
        controller_translation_delta_mm(current, anchor, torso_a),
        controller_translation_delta_mm(current, anchor, torso_b),
    )


def test_delta_matches_the_legacy_absolute_law():
    """The new delta law preserves the hardware-tested axis mapping.

    With user_scale == 1 and zero offsets, the difference of two absolute
    targets is exactly the anchored delta. Whatever direction the operator
    learned from the absolute implementation still holds.
    """
    head_pose = pose_to_se3(
        [0.0, 0.0, 1.6],
        [0.0, 0.0, 0.0, 1.0],
    )
    torso_pose = head_pose_to_torso(head_pose)

    anchor = pose_to_se3(
        [0.4, -0.2, 1.2],
        [0.0, 0.0, 0.0, 1.0],
    )
    current = pose_to_se3(
        [0.55, -0.05, 1.35],
        [0.0, 0.0, 0.0, 1.0],
    )

    def absolute_translation(controller_pose):
        return controller_pose_to_rb10e_target(
            controller_pose,
            torso_pose,
            1.0,
            target_x_offset_mm=0.0,
            target_y_offset_mm=0.0,
            target_z_offset_mm=0.0,
        )[:3, 3]

    legacy_delta = absolute_translation(current) - absolute_translation(anchor)

    assert np.allclose(
        controller_translation_delta_mm(current, anchor, torso_pose),
        R_OPERATOR_TO_BASE @ legacy_delta,
        atol=1e-9,
    )


# ---------------------------------------------------------------------------
# Anchored rotation delta
# ---------------------------------------------------------------------------


def rotation_about(axis, angle_rad):
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec(
        angle_rad * np.asarray(axis, dtype=float)
    ).as_matrix()
    return transform


def angle_of(rotation):
    return float(
        np.linalg.norm(Rotation.from_matrix(rotation).as_rotvec())
    )


def axis_of(rotation):
    rotvec = Rotation.from_matrix(rotation).as_rotvec()
    return rotvec / np.linalg.norm(rotvec)


def test_controller_rotation_delta_is_identity_at_the_anchor():
    torso = rotation_about([0.3, -0.2, 0.9], 0.7)
    torso[:3, 3] = [11.0, 22.0, 1400.0]

    controller = rotation_about([1.0, 0.4, -0.2], 0.5)
    controller[:3, 3] = [400.0, 500.0, 600.0]

    delta = controller_rotation_delta(controller, controller, torso)

    assert np.allclose(delta, np.eye(3))


def test_controller_rotation_delta_preserves_the_angle():
    """The rotation analogue of test_delta_preserves_length.

    Both factors are rotations, so the hand's rotation ANGLE survives the
    frame change untouched. Scaling is scale_rotation's job, not this
    function's.
    """
    torso = rotation_about([0.1, 0.9, -0.3], 1.1)
    torso[:3, 3] = [50.0, -60.0, 1350.0]

    anchor = rotation_about([0.2, -0.1, 0.3], 0.4)
    anchor[:3, 3] = [300.0, 100.0, 1200.0]

    axis = np.array([1.0, 2.0, -3.0])
    axis = axis / np.linalg.norm(axis)
    turn = np.deg2rad(37.0)

    current = anchor.copy()
    current[:3, :3] = (
        Rotation.from_rotvec(turn * axis).as_matrix() @ anchor[:3, :3]
    )

    delta = controller_rotation_delta(current, anchor, torso)

    assert angle_of(delta) == pytest.approx(turn, abs=1e-12)


def test_controller_rotation_delta_ignores_torso_translation():
    """The operator may lean or walk mid-stroke without rotating the tool."""
    rotation = rotation_about([0.0, 0.0, 1.0], 0.4)[:3, :3]

    torso_a = np.eye(4)
    torso_a[:3, :3] = rotation

    torso_b = np.eye(4)
    torso_b[:3, :3] = rotation
    torso_b[:3, 3] = [1234.0, -567.0, 890.0]

    anchor = rotation_about([0.4, 0.5, 0.6], 0.3)
    anchor[:3, 3] = [200.0, 300.0, 400.0]

    current = anchor.copy()
    current[:3, :3] = (
        rotation_about([0.0, 1.0, 0.0], 0.25)[:3, :3] @ anchor[:3, :3]
    )

    assert np.allclose(
        controller_rotation_delta(current, anchor, torso_a),
        controller_rotation_delta(current, anchor, torso_b),
    )


def test_controller_rotation_delta_applies_the_torso_rotation_as_a_similarity():
    """The rotation counterpart of the translation axis test.

    Same Rz(90) torso, so the two can be read side by side: a translation
    along RB +X comes out as [-100, 0, 0], and a rotation about RB +X comes
    out as a rotation about [-1, 0, 0] (Rz(-90) @ [0, -1, 0]).
    """
    torso = rotation_z(np.pi / 2.0)

    anchor = np.eye(4)
    current = rotation_about([1.0, 0.0, 0.0], np.deg2rad(30.0))

    delta = controller_rotation_delta(current, anchor, torso)

    assert angle_of(delta) == pytest.approx(np.deg2rad(30.0))
    assert np.allclose(axis_of(delta), [-1.0, 0.0, 0.0], atol=1e-12)


def test_controller_rotation_delta_cancels_a_constant_hand_offset():
    """Why T_FOR_RB10E is not applied to the delta path.

    A constant hand-to-flange offset multiplies BOTH controller poses on
    the right and cancels: R_now @ T @ T.T @ R_anchor.T. Applying it here
    would be a no-op at best, and a frame error if it were ever applied to
    only one side.
    """
    torso = rotation_about([0.2, 0.7, 0.1], 0.8)
    torso[:3, 3] = [30.0, 40.0, 1300.0]

    anchor = rotation_about([0.5, -0.2, 0.1], 0.6)
    anchor[:3, 3] = [250.0, 150.0, 1100.0]

    current = anchor.copy()
    current[:3, :3] = (
        rotation_about([0.0, 0.0, 1.0], 0.9)[:3, :3] @ anchor[:3, :3]
    )
    current[:3, 3] = anchor[:3, 3] + [10.0, -20.0, 30.0]

    offset = np.eye(4)
    offset[:3, :3] = T_FOR_RB10E[:3, :3]

    assert np.allclose(
        controller_rotation_delta(current, anchor, torso),
        controller_rotation_delta(current @ offset, anchor @ offset, torso),
        atol=1e-12,
    )


def test_rotation_delta_axis_map_matches_the_translation_axis_map():
    """Licenses reusing the README axis table for rotation.

    R_OPERATOR_TO_BASE is a proper rotation, so it carries an operator-frame
    axis to the same base-frame axis whether that axis describes a
    translation direction or a rotation axis. Feed one axis through both
    laws and check they agree.
    """
    torso = rotation_z(0.6)
    torso[:3, 3] = [12.0, -34.0, 1400.0]

    operator_axis = np.array([1.0, -2.0, 0.5])
    operator_axis = operator_axis / np.linalg.norm(operator_axis)

    anchor = np.eye(4)
    anchor[:3, 3] = [100.0, 200.0, 300.0]

    moved = anchor.copy()
    moved[:3, 3] = anchor[:3, 3] + 100.0 * (torso[:3, :3] @ operator_axis)

    turned = anchor.copy()
    turned[:3, :3] = Rotation.from_rotvec(
        np.deg2rad(15.0) * (torso[:3, :3] @ operator_axis)
    ).as_matrix()

    translation_axis = controller_translation_delta_mm(moved, anchor, torso)
    translation_axis = translation_axis / np.linalg.norm(translation_axis)

    rotation_axis = axis_of(controller_rotation_delta(turned, anchor, torso))

    assert np.allclose(translation_axis, rotation_axis, atol=1e-12)


# ---------------------------------------------------------------------------
# scale_rotation
# ---------------------------------------------------------------------------


def test_scale_rotation_scales_the_angle_and_keeps_the_axis():
    axis = np.array([1.0, 2.0, 3.0])
    axis = axis / np.linalg.norm(axis)

    rotation = Rotation.from_rotvec(np.deg2rad(30.0) * axis).as_matrix()
    scaled = scale_rotation(rotation, 0.3)

    assert np.rad2deg(angle_of(scaled)) == pytest.approx(9.0, abs=1e-9)
    assert np.allclose(axis_of(scaled), axis, atol=1e-12)


def test_scale_rotation_at_zero_gain_is_exactly_the_identity():
    """Exact equality, not allclose.

    This is what makes "orientation_scale=0.0 reproduces the previous
    control law" a testable claim rather than a hope: identity @ anchor is
    bit-for-bit anchor.
    """
    rotation = Rotation.from_rotvec([0.3, -0.7, 1.2]).as_matrix()

    assert np.array_equal(scale_rotation(rotation, 0.0), np.eye(3))


def test_scale_rotation_at_unit_gain_is_the_input():
    rotation = Rotation.from_rotvec([0.3, -0.7, 1.2]).as_matrix()

    assert np.allclose(scale_rotation(rotation, 1.0), rotation, atol=1e-12)


def test_scale_rotation_rejects_a_non_finite_gain():
    with pytest.raises(ValueError, match="gain must be finite"):
        scale_rotation(np.eye(3), float("nan"))


def test_scale_rotation_rejects_a_non_rotation():
    with pytest.raises(ValueError, match="not orthonormal"):
        scale_rotation(np.diag([1.0, 1.0, 2.0]), 0.3)

    with pytest.raises(ValueError, match="shape"):
        scale_rotation(np.eye(2), 0.3)


def test_scale_rotation_rejects_a_reflection():
    with pytest.raises(ValueError, match="reflection"):
        scale_rotation(np.diag([1.0, 1.0, -1.0]), 0.3)


# ---------------------------------------------------------------------------
# build_delta_target
# ---------------------------------------------------------------------------


def test_build_delta_target_without_rotation_matches_the_position_helper():
    anchor = rotation_z(0.7)
    anchor[:3, 3] = [10.0, 20.0, 30.0]

    assert np.array_equal(
        build_delta_target(anchor, [50.0, -60.0, 70.0]),
        build_position_delta_target(anchor, [50.0, -60.0, 70.0]),
    )


def test_build_delta_target_pre_multiplies_the_rotation():
    """Pre- vs post-multiply is the decision most likely to be silently
    flipped by a later edit, so assert the difference, not just the value.
    """
    anchor = rotation_about([0.0, 0.0, 1.0], 0.9)
    anchor[:3, 3] = [10.0, 20.0, 30.0]

    delta = rotation_about([1.0, 0.0, 0.0], 0.5)[:3, :3]

    target = build_delta_target(anchor, np.zeros(3), rotation_delta=delta)

    assert np.allclose(target[:3, :3], delta @ anchor[:3, :3])
    assert not np.allclose(target[:3, :3], anchor[:3, :3] @ delta)


def test_build_delta_target_translates_independently_of_the_rotation():
    anchor = rotation_z(0.4)
    anchor[:3, 3] = [10.0, 20.0, 30.0]

    delta = rotation_about([0.0, 1.0, 0.0], 0.6)[:3, :3]

    target = build_delta_target(
        anchor,
        [50.0, -60.0, 70.0],
        rotation_delta=delta,
    )

    assert np.allclose(target[:3, 3], [60.0, -40.0, 100.0])


def test_build_delta_target_does_not_mutate_the_anchor():
    anchor = rotation_z(0.4)
    anchor[:3, 3] = [1.0, 2.0, 3.0]
    original = anchor.copy()

    build_delta_target(
        anchor,
        [10.0, 10.0, 10.0],
        rotation_delta=rotation_about([0.0, 1.0, 0.0], 0.6)[:3, :3],
    )

    assert np.allclose(anchor, original)


def test_build_delta_target_rejects_a_non_rotation_delta():
    anchor = np.eye(4)

    with pytest.raises(ValueError, match="rotation_delta is not orthonormal"):
        build_delta_target(
            anchor,
            np.zeros(3),
            rotation_delta=np.diag([1.0, 1.0, 2.0]),
        )
