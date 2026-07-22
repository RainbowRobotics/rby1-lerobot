import numpy as np

from lerobot_teleoperator_rb.frame_transforms import (
    T_CONV,
    T_FOR_HEAD,
    T_FOR_RB10E,
    compute_user_scale,
    controller_pose_to_rb10e_target,
    head_pose_to_torso_pose,
    invert_se3,
    pose_to_se3,
    quest_pose_to_rb_frame,
    scale_translation,
)


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

    actual = quest_pose_to_rb_frame(
        position,
        quaternion,
    )

    assert np.allclose(actual, expected)


def test_head_to_torso_matches_legacy_formula():
    head_pose = np.eye(4)

    expected = head_pose @ T_FOR_HEAD
    actual = head_pose_to_torso_pose(head_pose)

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


def test_scale_translation_does_not_mutate_input():
    pose = np.eye(4)
    pose[0, 3] = 100.0
    original = pose.copy()

    scaled = scale_translation(
        pose,
        scale=2.0,
        z_offset_mm=300.0,
    )

    assert np.allclose(pose, original)
    assert np.allclose(
        scaled[:3, 3],
        [200.0, 0.0, 300.0],
    )


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
    expected[2, 3] += 300.0

    actual = controller_pose_to_rb10e_target(
        controller_pose,
        torso_pose,
        user_scale=user_scale,
        position_scale=1.0,
        z_offset_mm=300.0,
    )

    assert np.allclose(actual, expected)
