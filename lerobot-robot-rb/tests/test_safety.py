"""Unit tests for the pure safety/clamping helpers (no rbpodo needed)."""

import numpy as np
import pytest

from lerobot_robot_rb import clamp_ee_target, clamp_target, wrap_deg


class TestClampTarget:
    def test_passthrough_within_limits(self):
        limits = np.array([[-180.0, 180.0]] * 6)
        last = np.zeros(6)
        target = np.full(6, 1.0)
        out = clamp_target(target, last, limits, max_delta_deg=2.0)
        np.testing.assert_allclose(out, target)

    def test_per_step_delta_clamp(self):
        limits = np.array([[-180.0, 180.0]] * 6)
        last = np.zeros(6)
        target = np.full(6, 10.0)
        out = clamp_target(target, last, limits, max_delta_deg=2.0)
        np.testing.assert_allclose(out, np.full(6, 2.0))

    def test_joint_limit_clip(self):
        limits = np.array([[-5.0, 5.0]] * 6)
        last = np.full(6, 4.5)
        target = np.full(6, 100.0)
        out = clamp_target(target, last, limits, max_delta_deg=2.0)
        np.testing.assert_allclose(out, np.full(6, 5.0))


class TestWrapDeg:
    def test_wrap(self):
        np.testing.assert_allclose(
            wrap_deg(np.array([0.0, 190.0, -190.0, 360.0, 539.0])),
            np.array([0.0, -170.0, 170.0, 0.0, 179.0]),
        )


class TestClampEeTarget:
    def test_position_delta_clamp(self):
        last = np.zeros(6)
        target = np.array([100.0, -100.0, 1.0, 0.0, 0.0, 0.0])
        out = clamp_ee_target(target, last, 5.0, 2.0)
        np.testing.assert_allclose(out[:3], [5.0, -5.0, 1.0])

    def test_rotation_delta_clamp_wrap_aware(self):
        # Crossing the +/-180 deg seam must take the short way.
        last = np.array([0.0, 0.0, 0.0, 179.0, 0.0, 0.0])
        target = np.array([0.0, 0.0, 0.0, -179.0, 0.0, 0.0])
        out = clamp_ee_target(target, last, 5.0, 2.0)
        # Short-way delta is +2 deg -> 181, continuous with last.
        assert out[3] == pytest.approx(181.0)

    def test_workspace_bounds(self):
        last = np.array([995.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        target = np.array([2000.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        bounds = np.array([[-1000.0, 1000.0]] * 3)
        out = clamp_ee_target(target, last, 100.0, 2.0, bounds)
        assert out[0] == pytest.approx(1000.0)

    def test_no_motion_is_identity(self):
        last = np.array([100.0, 200.0, 300.0, 10.0, -20.0, 30.0])
        out = clamp_ee_target(last.copy(), last, 5.0, 2.0)
        np.testing.assert_allclose(out, last)
