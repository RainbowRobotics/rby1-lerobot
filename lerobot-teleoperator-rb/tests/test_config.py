import pytest
from lerobot.teleoperators.config import TeleoperatorConfig

import lerobot_teleoperator_rb
from lerobot_teleoperator_rb import RbVrConfig


def test_rb_vr_config_is_registered():
    config_class = TeleoperatorConfig.get_choice_class("rb_vr")

    assert config_class is RbVrConfig


def test_rb_vr_config_defaults():
    config = RbVrConfig(
        id="test_rb_vr",
        send_handshake=False,
    )

    assert config.type == "rb_vr"
    assert config.local_port == 5005
    assert config.meta_quest_port == 6000
    assert config.controller_hand == "right"
    assert config.grip_threshold == 0.5
    assert config.tracking_timeout_s == 0.25
    assert config.ik_iterations == 5
    assert config.orientation_scale == 1.0
    assert config.require_initialization_button is True


def test_rb_vr_config_custom_values():
    config = RbVrConfig(
        id="custom_rb_vr",
        local_ip="192.168.0.10",
        local_port=7000,
        meta_quest_ip="192.168.0.20",
        meta_quest_port=8000,
        controller_hand="left",
        grip_threshold=0.7,
        use_gripper=True,
        tracking_timeout_s=0.5,
        ik_iterations=8,
        orientation_scale=0.5,
    )

    assert config.local_ip == "192.168.0.10"
    assert config.local_port == 7000
    assert config.meta_quest_ip == "192.168.0.20"
    assert config.meta_quest_port == 8000
    assert config.controller_hand == "left"
    assert config.grip_threshold == 0.7
    assert config.use_gripper is True
    assert config.tracking_timeout_s == 0.5
    assert config.orientation_scale == 0.5
    assert config.ik_iterations == 8


def test_rb_vr_config_orientation_scale_accepts_zero():
    """0.0 is the "freeze the orientation" value, not an invalid one.

    This is the test that catches anyone reaching for
    _validate_finite_positive, which rejects 0.0.
    """
    config = RbVrConfig(
        id="frozen_orientation",
        send_handshake=False,
        orientation_scale=0.0,
    )

    assert config.orientation_scale == 0.0


def test_rb_vr_config_orientation_scale_accepts_one():
    config = RbVrConfig(
        id="one_to_one_orientation",
        send_handshake=False,
        orientation_scale=1.0,
    )

    assert config.orientation_scale == 1.0


@pytest.mark.parametrize(
    "value",
    [-0.1, 1.1, float("nan"), float("inf")],
)
def test_rb_vr_config_rejects_invalid_orientation_scale(value):
    with pytest.raises(ValueError, match="orientation_scale"):
        RbVrConfig(
            id="bad_orientation",
            send_handshake=False,
            orientation_scale=value,
        )
