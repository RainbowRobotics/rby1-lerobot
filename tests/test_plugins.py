"""Plugin discovery and CLI-style config parsing against the installed lerobot."""

from __future__ import annotations

from dataclasses import dataclass

import draccus
import lerobot.cameras.opencv  # noqa: F401  (registers "opencv"; the lerobot CLIs import it too)
import lerobot.cameras.realsense  # noqa: F401
from lerobot.robots.config import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig


def test_plugins_are_registered() -> None:
    assert "rby1" in RobotConfig.get_known_choices()
    assert {"rby1_leader_arm", "rby1_vr"} <= set(TeleoperatorConfig.get_known_choices())


@dataclass
class _Cli:
    robot: RobotConfig
    teleop: TeleoperatorConfig


def _parse(*args: str) -> _Cli:
    return draccus.parse(config_class=_Cli, args=list(args))


def test_parse_readme_teleoperate_args() -> None:
    cfg = _parse(
        "--robot.type=rby1",
        "--robot.address=192.168.30.1:50051",
        '--robot.cameras={"front": {"type": "intelrealsense", "serial_number_or_name": "XXXXXXXXX",'
        ' "fps": 30, "width": 640, "height": 480},'
        ' "right": {"type": "opencv", "index_or_path": 0, "fps": 30, "width": 640, "height": 480}}',
        "--teleop.type=rby1_leader_arm",
    )
    assert cfg.robot.type == "rby1"
    assert cfg.robot.address == "192.168.30.1:50051"
    assert cfg.robot.cameras["front"].type == "intelrealsense"
    assert cfg.robot.cameras["front"].width == 640
    assert cfg.robot.cameras["right"].type == "opencv"
    assert cfg.teleop.type == "rby1_leader_arm"


def test_parse_vr_teleop_ee_mode() -> None:
    cfg = _parse("--robot.type=rby1", "--robot.action_mode=ee", "--teleop.type=rby1_vr")
    assert cfg.robot.action_mode == "ee"
    assert cfg.teleop.type == "rby1_vr"
