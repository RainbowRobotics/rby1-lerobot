"""rby1_isaac as a lerobot plugin: registration, robot link, profile, keys shared with the robot."""

from __future__ import annotations

import json
from dataclasses import dataclass

import draccus
import isaac_fakes as fakes
import pytest
from lerobot.robots.config import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot_robot_rby1 import Rby1, Rby1Config
from lerobot_robot_rby1.constants import RESET_CMD_KEY
from lerobot_teleoperator_rby1 import Rby1XR, Rby1XRConfig
from lerobot_teleoperator_rby1.isaac_teleop.robot_state import Rby1StateReader

ADDRESS = "stub.invalid:1"


@dataclass
class _Cli:
    robot: RobotConfig
    teleop: TeleoperatorConfig


def test_parse_isaac_teleop_args() -> None:
    cfg = draccus.parse(
        config_class=_Cli,
        args=[
            "--robot.type=rby1", "--robot.action_mode=ee", "--robot.use_torso=true", "--robot.use_head=true",
            "--robot.ee_ready_torso_deg=[0,45,-90,45,0,0]",
            "--teleop.type=rby1_isaac", "--teleop.wear_mode=neck",
        ],
    )
    assert cfg.teleop.type == "rby1_isaac" and isinstance(cfg.teleop, Rby1XRConfig)
    assert cfg.teleop.robot_model == "auto"
    assert cfg.robot.ee_ready_torso_deg == [0, 45, -90, 45, 0, 0]


def test_robot_model_choices() -> None:
    with pytest.raises(ValueError, match="robot_model"):
        Rby1XRConfig(robot_model="ub")


@pytest.mark.parametrize("flags", [
    {"use_torso": True, "use_mobile_base": True, "use_head": True},
    {"use_torso": False, "use_mobile_base": False, "use_head": False},
    {"use_torso": True, "use_mobile_base": False, "use_head": True, "record_posture_hint": True},
])
def test_action_keys_match_the_follower(flags: dict) -> None:
    """With mirrored use_* flags the teleop's action keys are the robot's (EE mode)."""
    teleop = Rby1XR(Rby1XRConfig(auto_launch_cloudxr=False, **flags), session_factory=lambda pipeline: None)
    robot = Rby1(Rby1Config(address=ADDRESS, action_mode="ee", **flags))
    assert set(teleop.action_features) == set(robot.action_features)
    assert RESET_CMD_KEY not in teleop.action_features


def test_state_reader_detects_the_model() -> None:
    reader = Rby1StateReader(ADDRESS, "auto")
    reader.connect()
    try:
        assert reader.model_name in ("a", "m") and reader.version in ("1.0", "1.1", "1.2", "1.3")
        snap = reader.read()
        assert snap.torso.shape == (4, 4) and snap.head_q.shape == (2,) and snap.right_q.shape == (7,)
    finally:
        reader.close()
    assert not reader.is_connected


def test_teleop_config_is_recorded_in_the_profile(tmp_path, monkeypatch) -> None:
    from lerobot_teleoperator_rby1.isaac_teleop import teleop_rby1_xr as mod

    monkeypatch.setattr(mod, "print_xr_connect_help", lambda: None)
    path = tmp_path / "rby1_profile.json"
    cfg = Rby1XRConfig(auto_launch_cloudxr=False, session_start="first_action", profile_out=str(path))
    teleop = Rby1XR(cfg, session_factory=lambda pipeline: fakes.FakeSession(),
                    state_reader_factory=fakes.FakeStateReader)
    teleop.connect()
    try:
        data = json.loads(path.read_text())
        assert data["teleop"]["type"] == "rby1_isaac"
        assert data["teleop"]["config"]["wear_mode"] == cfg.wear_mode
    finally:
        teleop.disconnect()
