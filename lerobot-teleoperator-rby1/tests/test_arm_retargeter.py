import math

import numpy as np
import pytest

from lerobot_teleoperator_rby1.isaac_teleop.arm_retargeter import (
    ARM_LIMITS,
    ArmPostureRetargeter,
    forward_points,
    robot_reach,
    robot_shoulder_position,
    solve_shoulder_elbow,
)


def _dirs(q, side, version="1.3"):
    S, E, W = forward_points(q, side, version)
    return E - S, W - E


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("version", ["1.2", "1.3"])
def test_round_trip_random_configurations(side, version):
    rng = np.random.default_rng(7)
    lim = ARM_LIMITS[side]
    for _ in range(200):
        q = np.array(
            [
                rng.uniform(-2.5, 2.5),
                rng.uniform(-2.5, -0.1) if side == "right" else rng.uniform(0.1, 2.5),
                rng.uniform(-2.5, 2.5),
                rng.uniform(-2.4, -0.35),  # elbow clearly bent (azimuth well defined)
            ]
        )
        assert np.all(q >= lim[:, 0] - 1e-9) and np.all(q <= lim[:, 1] + 1e-9)
        u, f = _dirs(q, side, version)
        est = solve_shoulder_elbow(u, f, side, q_prev=q, version=version)
        assert est is not None
        for i in range(4):
            assert abs(_wrap(est[i] - q[i])) < 1e-5, (side, version, i, q, est)


def test_straight_arm_hanging_down():
    # A straight human arm (upper arm and forearm collinear, pointing down)
    # maps to q0 = q1 = 0 and the robot's own "straight" elbow angle, which is
    # about -12 deg because of the x offsets of the two link vectors.
    # (q0 is a few degrees, not 0: the upper-arm rest vector is tilted +x.)
    down = np.array([0.0, 0.0, -1.0])
    for side in ("right", "left"):
        q = solve_shoulder_elbow(down, down, side, q_prev=np.zeros(4))
        assert abs(q[0]) < math.radians(8) and abs(q[1]) < math.radians(1)
        assert q[3] == pytest.approx(math.radians(-12.1), abs=0.01)
        S, E, W = forward_points(q, side)
        np.testing.assert_allclose((E - S) / np.linalg.norm(E - S), down, atol=1e-6)
        np.testing.assert_allclose((W - E) / np.linalg.norm(W - E), down, atol=1e-6)


def test_signs_of_shoulder_joints():
    # Raising the upper arm forward (about the shoulder pitch axis) -> q0 sign.
    for side in ("right", "left"):
        q_true = np.array([-0.8, -0.3 if side == "right" else 0.3, 0.0, -1.0])
        u, f = _dirs(q_true, side)
        q = solve_shoulder_elbow(u, f, side)
        assert q[0] == pytest.approx(-0.8, abs=1e-6)
        # arm_1 sign follows the URDF limit of each side
        assert (q[1] <= 0.0) if side == "right" else (q[1] >= 0.0)
        assert q[3] == pytest.approx(-1.0, abs=1e-6)


def test_elbow_angle_maps_to_q3():
    q_true = np.array([0.3, -0.4, 0.2, -math.pi / 2])
    u, f = _dirs(q_true, "right")
    q = solve_shoulder_elbow(u, f, "right", q_prev=q_true)
    assert q[3] == pytest.approx(-math.pi / 2, abs=1e-6)


def test_degenerate_inputs_return_none():
    assert solve_shoulder_elbow(np.zeros(3), np.array([0, 0, -1.0]), "right") is None


def test_retargeter_smoothing_hold_and_timeout():
    r = ArmPostureRetargeter("right", smoothing=0.5, max_vel=100.0, hold_s=0.5)
    R = np.eye(3)
    q_a = np.array([0.3, -0.4, 0.2, -1.2])
    S, E, W = forward_points(q_a, "right")
    assert r.update(S, E, W, R, t=0.0, dt=0.02) is not None
    np.testing.assert_allclose(r.hint, q_a, atol=1e-6)  # first sample taken as is
    q_b = q_a + np.array([0.2, 0.0, 0.0, 0.0])
    S, E, W = forward_points(q_b, "right")
    h = r.update(S, E, W, R, t=0.02, dt=0.02)
    assert h[0] == pytest.approx(q_a[0] + 0.1, abs=1e-6)  # EMA half-way
    # Invalid input: hold, then time out.
    assert r.update(None, E, W, R, t=0.3, dt=0.02) is not None
    assert r.update(None, E, W, R, t=0.9, dt=0.02) is None


def test_retargeter_velocity_limit():
    r = ArmPostureRetargeter("right", smoothing=1.0, max_vel=1.0, hold_s=1.0)
    R = np.eye(3)
    S, E, W = forward_points(np.array([0.0, -0.4, 0.0, -1.0]), "right")
    r.update(S, E, W, R, t=0.0, dt=0.02)
    S, E, W = forward_points(np.array([1.0, -0.4, 0.0, -1.0]), "right")
    h = r.update(S, E, W, R, t=0.02, dt=0.02)
    assert h[0] == pytest.approx(0.02, abs=1e-6)  # 1 rad/s * 0.02 s


def test_reach_and_shoulder_position():
    assert robot_reach("1.3") == pytest.approx(0.5855, abs=1e-3)
    assert robot_reach("1.2") == pytest.approx(0.532, abs=1e-3)
    T = np.eye(4)
    T[:3, 3] = [0, 0, 1.0]
    np.testing.assert_allclose(robot_shoulder_position(T, "right"), [0, -0.22, 1.080073451539], atol=1e-6)
