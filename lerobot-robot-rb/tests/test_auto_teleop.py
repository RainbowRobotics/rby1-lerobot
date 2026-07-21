"""Unit tests for the software-only rb_auto teleoperator (no hardware)."""

import math

import numpy as np
import pytest

from lerobot_robot_rb import EE_NAMES, JOINT_NAMES, RbAutoTeleop, RbAutoTeleopConfig


def _make(space="joint", **kwargs):
    center = kwargs.pop("center", [0.0] * 6)
    cfg = RbAutoTeleopConfig(space=space, center=center, **kwargs)
    return RbAutoTeleop(cfg)


class TestConfigValidation:
    def test_bad_space(self):
        with pytest.raises(ValueError):
            RbAutoTeleopConfig(space="cartesian")

    def test_bad_axis(self):
        with pytest.raises(ValueError):
            RbAutoTeleopConfig(axis=6)

    def test_bad_amplitude(self):
        with pytest.raises(ValueError):
            RbAutoTeleopConfig(amplitude=0.0)

    def test_bad_center_length(self):
        with pytest.raises(ValueError):
            RbAutoTeleopConfig(center=[0.0] * 5)


class TestJointMode:
    def test_action_keys(self):
        teleop = _make()
        assert list(teleop.action_features) == JOINT_NAMES

    def test_sine_centre_and_amplitude(self):
        teleop = _make(axis=5, amplitude=0.5, period_s=1.0)
        teleop.connect()
        actions = np.array(
            [[a[name] for name in JOINT_NAMES] for a in _sample(teleop)]
        )
        # Non-driven joints stay at the centre.
        np.testing.assert_allclose(actions[:, :5], 0.0)
        # Driven joint stays within the configured amplitude (radians).
        assert np.abs(actions[:, 5]).max() <= math.radians(0.5) + 1e-9
        teleop.disconnect()


class TestEeMode:
    def test_action_keys(self):
        teleop = _make(space="ee")
        assert list(teleop.action_features) == EE_NAMES

    def test_position_axis_amplitude_in_metres(self):
        teleop = _make(space="ee", axis=2, amplitude=5.0, period_s=1.0)
        teleop.connect()
        actions = np.array(
            [[a[name] for name in EE_NAMES] for a in _sample(teleop)]
        )
        # 5 mm amplitude -> 0.005 m in dataset units.
        assert np.abs(actions[:, 2]).max() <= 0.005 + 1e-12
        assert np.abs(actions[:, 2]).max() > 0.0
        np.testing.assert_allclose(actions[:, :2], 0.0)
        teleop.disconnect()

    def test_rotation_axis_amplitude_in_radians(self):
        teleop = _make(space="ee", axis=4, amplitude=2.0, period_s=1.0)
        teleop.connect()
        actions = np.array(
            [[a[name] for name in EE_NAMES] for a in _sample(teleop)]
        )
        assert np.abs(actions[:, 4]).max() <= math.radians(2.0) + 1e-12
        teleop.disconnect()


class TestConnectionGuards:
    def test_get_action_requires_connect(self):
        teleop = _make()
        with pytest.raises(Exception):
            teleop.get_action()

    def test_connect_without_center_requires_robot(self):
        cfg = RbAutoTeleopConfig()
        teleop = RbAutoTeleop(cfg)
        with pytest.raises(Exception):
            teleop.connect()


def _sample(teleop, n=50):
    for _ in range(n):
        yield teleop.get_action()
