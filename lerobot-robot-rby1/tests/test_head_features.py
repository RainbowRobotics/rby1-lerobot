"""use_head feature / command tests (need lerobot importable; no hardware)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("lerobot")

from lerobot_robot_rby1 import command_builders as cb  # noqa: E402
from lerobot_robot_rby1.config_rby1 import Rby1Config  # noqa: E402
from lerobot_robot_rby1.constants import HEAD_Q_MAX, HEAD_Q_MIN  # noqa: E402
from lerobot_robot_rby1.rby1 import Rby1  # noqa: E402

HEAD_KEYS = ["head_0.pos", "head_1.pos"]


def _robot(**kw) -> Rby1:
    return Rby1(Rby1Config(use_gripper=False, **kw))


@pytest.mark.parametrize("mode", ["joint", "ee"])
def test_use_head_false_leaves_features_unchanged(mode):
    r = _robot(action_mode=mode)
    assert not any(k in r.action_features for k in HEAD_KEYS)
    assert not any(k in r.observation_features for k in HEAD_KEYS)


@pytest.mark.parametrize("mode", ["joint", "ee"])
def test_use_head_true_adds_keys_in_both_modes(mode):
    r = _robot(action_mode=mode, use_head=True, use_velocity=True)
    assert all(k in r.action_features for k in HEAD_KEYS)
    assert all(k in r.observation_features for k in HEAD_KEYS)
    assert "head_0.vel" in r.observation_features
    assert r._head_target_from_action({"head_0.pos": 0.1, "head_1.pos": 0.2}).tolist() == [0.1, 0.2]
    assert r._head_target_from_action({"head_0.pos": 0.1}) is None


class _Builder:
    def __init__(self):
        self.calls = {}

    def __getattr__(self, name):
        def rec(*a, **k):
            self.calls[name] = (a, k)
            return self

        return rec


class _FakeRby:
    def __init__(self):
        self.builders = []

    def JointPositionCommandBuilder(self):  # noqa: N802
        b = _Builder()
        self.builders.append(b)
        return b

    def CommandHeaderBuilder(self):  # noqa: N802
        return _Builder()


def test_build_head_command_clips_to_urdf_limits():
    rby = _FakeRby()
    cb.build_head_command(rby, np.array([9.0, -9.0]), np.ones(2), np.ones(2), minimum_time=0.1)
    (pos,), _ = rby.builders[0].calls["set_position"]
    np.testing.assert_allclose(pos, [HEAD_Q_MAX[0], HEAD_Q_MIN[1]])
    assert rby.builders[0].calls["set_minimum_time"][0] == (0.1,)
