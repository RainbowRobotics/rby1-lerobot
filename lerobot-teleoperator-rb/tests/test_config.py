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
    assert config.ik_iterations == 3
    assert config.require_initialization_button is True
    assert config.hold_last_target is True


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
        max_joint_delta_rad=0.05,
    )

    assert config.local_ip == "192.168.0.10"
    assert config.local_port == 7000
    assert config.meta_quest_ip == "192.168.0.20"
    assert config.meta_quest_port == 8000
    assert config.controller_hand == "left"
    assert config.grip_threshold == 0.7
    assert config.use_gripper is True
    assert config.tracking_timeout_s == 0.5
    assert config.ik_iterations == 8
    assert config.max_joint_delta_rad == 0.05
