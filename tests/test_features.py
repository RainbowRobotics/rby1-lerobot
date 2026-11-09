"""Freeze the observation/action key layout.

Policies and datasets depend on the exact key order, and upstream rollout
(lerobot main) re-checks state/action order against the checkpoint. A change
here is a breaking change: regenerate with ``RBY1_UPDATE_SNAPSHOTS=1`` only on
purpose, and bump the plugin's major/minor version.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from lerobot_robot_rby1 import Rby1, Rby1Config

SNAPSHOT = Path(__file__).parent / "snapshots" / "features.json"

CONFIGS: dict[str, dict] = {
    "default": {},
    "torso": {"use_torso": True},
    "velocity_torque": {"use_velocity": True, "use_torque": True},
    "mobile_base": {"use_mobile_base": True},
    "right_arm_only": {"use_left_arm": False},
    "no_gripper": {"use_gripper": False},
    "ee": {"action_mode": "ee", "use_torso": True},
}


def _features(overrides: dict) -> dict[str, list[str]]:
    robot = Rby1(Rby1Config(address="stub.invalid:1", **overrides))
    return {
        "observation": list(robot.observation_features),
        "action": list(robot.action_features),
    }


def test_feature_snapshot() -> None:
    current = {name: _features(overrides) for name, overrides in CONFIGS.items()}
    if os.environ.get("RBY1_UPDATE_SNAPSHOTS") == "1":
        SNAPSHOT.write_text(json.dumps(current, indent=2) + "\n")
    expected = json.loads(SNAPSHOT.read_text())
    assert current == expected


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, True),
        ({"use_torso": True, "use_velocity": True}, True),
        ({"use_mobile_base": True}, False),
        ({"action_mode": "ee"}, False),
    ],
)
def test_supports_position_hold(overrides: dict, expected: bool) -> None:
    robot = Rby1(Rby1Config(address="stub.invalid:1", **overrides))
    assert robot.supports_position_hold is expected
    if expected:
        assert all(key.endswith(".pos") for key in robot.action_features)
