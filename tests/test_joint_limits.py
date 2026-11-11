"""Joint position limits read from the robot over a short read-only connection."""

from __future__ import annotations

import pytest
import rby1_sdk as rby
from lerobot_robot_rby1.joint_limits import read_joint_limits


def test_reads_the_limits_of_each_group(monkeypatch) -> None:
    monkeypatch.setenv("RBY1_FAKE_MODEL", "M")
    monkeypatch.setenv("RBY1_FAKE_VERSION", "1.3")
    monkeypatch.setitem(rby.JOINT_Q_LIMITS, "left_arm_6", (-1.5, 1.5))
    n = len(rby.CALL_LOG)
    limits = read_joint_limits("stub.invalid:1")
    assert (limits.model, limits.version) == ("m", "1.3")
    assert [len(g[0]) for g in (limits.torso, limits.right_arm, limits.left_arm)] == [6, 7, 7]
    assert (limits.torso[0][2], limits.torso[1][2]) == rby.JOINT_Q_LIMITS["torso_2"]
    assert (limits.left_arm[0][6], limits.left_arm[1][6]) == (-1.5, 1.5)
    # Read-only: nothing is powered, enabled or commanded, and the handle is closed.
    names = [call[0] for call in rby.CALL_LOG[n:]]
    assert set(names) == {"create_robot", "connect", "disconnect"}
    assert names[-1] == "disconnect"


def test_unreachable_robot_is_an_error(monkeypatch) -> None:
    monkeypatch.setenv("RBY1_FAKE_CONNECT_FAIL", "1")
    with pytest.raises(RuntimeError, match="could not connect"):
        read_joint_limits("stub.invalid:1", connect_timeout_sec=0.1)
