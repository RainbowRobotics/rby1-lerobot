"""Damping ratios reach the SDK command builders (they used to be dropped)."""

from __future__ import annotations

import numpy as np
import pytest
import rby1_sdk as rby
from lerobot_robot_rby1 import Rby1Config
from lerobot_robot_rby1 import command_builders as cb
from lerobot_robot_rby1.constants import LEFT_ARM_NAMES, POS_SUFFIX, RIGHT_ARM_NAMES


def _joint_body(cfg: Rby1Config):
    action = {f"{n}{POS_SUFFIX}": 0.0 for n in RIGHT_ARM_NAMES + LEFT_ARM_NAMES}
    model = rby.Model_A()
    limits = np.ones(model.robot_dof)
    body, _ = cb.build_body_command(rby, cfg, action, model, limits, limits, 1.0)
    return body


@pytest.mark.parametrize("zeta", [1.0, 0.7])
def test_joint_impedance_sends_damping_ratio(zeta: float) -> None:
    cfg = Rby1Config(address="stub.invalid:1", use_joint_impedance=True, joint_impedance_damping_ratio=zeta)
    body = _joint_body(cfg)
    for arm in ("set_right_arm_command", "set_left_arm_command"):
        limb = body.get(arm)[0]
        assert type(limb).__name__ == "JointImpedanceControlCommandBuilder"
        assert limb.get("set_damping_ratio") == (zeta,)


def test_joint_position_mode_has_no_damping_ratio() -> None:
    body = _joint_body(Rby1Config(address="stub.invalid:1"))
    limb = body.get("set_right_arm_command")[0]
    assert type(limb).__name__ == "JointPositionCommandBuilder"
    assert limb.get("set_damping_ratio") is None


def test_ee_solvers_send_joint_damping_ratio() -> None:
    cfg = Rby1Config(
        address="stub.invalid:1", action_mode="ee", use_torso=True,
        ee_torso_damping_ratio=0.9, ee_arm_damping_ratio=0.8,
    )
    eye = np.eye(4)
    targets = cb.CartesianTargets(torso=eye, right_arm=eye, left_arm=eye)
    body = cb.build_ee_command(rby, cfg, targets, cb.ResetFlags(), np.zeros(7), np.zeros(7))
    for setter, zeta in (
        ("set_torso_command", 0.9),
        ("set_right_arm_command", 0.8),
        ("set_left_arm_command", 0.8),
    ):
        assert body.get(setter)[0].get("set_joint_damping_ratio") == (zeta,)


def test_whole_body_mode_is_gone() -> None:
    with pytest.raises(TypeError):
        Rby1Config(address="stub.invalid:1", ee_whole_body=True)
    assert not hasattr(cb, "build_whole_body_command")


@pytest.mark.parametrize("name", ["joint_impedance_damping_ratio", "ee_arm_damping_ratio"])
def test_non_positive_damping_ratio_is_rejected(name: str) -> None:
    with pytest.raises(ValueError, match=name):
        Rby1Config(address="stub.invalid:1", **{name: 0.0})
