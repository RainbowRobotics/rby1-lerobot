from lerobot_teleoperator_rb import RbVr, RbVrConfig


EXPECTED_JOINTS = {
    "joint_0",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
}


def test_joint_action_features():
    teleop = RbVr(
        RbVrConfig(
            id="test_rb_vr",
            send_handshake=False,
            use_gripper=False,
        )
    )

    assert set(teleop.action_features) == EXPECTED_JOINTS
    assert all(
        feature_type is float
        for feature_type in teleop.action_features.values()
    )


def test_joint_action_features_with_gripper():
    teleop = RbVr(
        RbVrConfig(
            id="test_rb_vr_gripper",
            send_handshake=False,
            use_gripper=True,
        )
    )

    assert set(teleop.action_features) == {
        *EXPECTED_JOINTS,
        "gripper_0",
    }


def test_feedback_features_are_empty():
    teleop = RbVr(
        RbVrConfig(
            id="test_rb_vr_feedback",
            send_handshake=False,
        )
    )

    assert teleop.feedback_features == {}
