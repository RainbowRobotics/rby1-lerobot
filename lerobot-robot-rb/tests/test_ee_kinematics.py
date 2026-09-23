"""Hardware-free tests for the URDF-based RB end-effector kinematics."""

import builtins
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


_MODULE_PATH = Path(__file__).parents[1] / "lerobot_robot_rb" / "ee_kinematics.py"
_SPEC = importlib.util.spec_from_file_location("_rb_ee_kinematics", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
RBEEKinematics = _MODULE.RBEEKinematics


_PASSIVE_MEASURED_DEG = np.array(
    [-173.1671600341797, -2.7800848484039307, 150.32669067382812,
     50.89628601074219, 273.85675048828125, 188.9039306640625]
)
_PASSIVE_REFERENCE_DEG = np.array(
    [-173.04730224609375, -3.028733253479004, 149.30467224121094,
     50.8963508605957, 273.85687255859375, 188.9039764404297]
)


@pytest.mark.parametrize("model", ["rb10", "rb10e"])
def test_passive_telemetry_separates_tracking_from_velocity(model):
    kin = RBEEKinematics(model)
    measured = np.deg2rad(_PASSIVE_MEASURED_DEG)
    reference = np.deg2rad(_PASSIVE_REFERENCE_DEG)
    target = kin.forward(measured)
    target[:3] += [-0.026, -0.003, 0.029]
    step = 0.5 / 30
    tracking = 0.05

    command = kin.inverse_step(
        target,
        measured,
        max_joint_delta=step,
        previous_q=reference,
        max_tracking_error=tracking,
    )

    command_from_reference = kin.forward(command)[:3] - kin.forward(reference)[:3]
    assert command_from_reference[0] < 0.0
    assert command_from_reference[2] > 0.0
    assert np.max(np.abs(command - reference)) <= step + 1e-10
    assert np.max(np.abs(command - measured)) <= tracking + 1e-10


@pytest.mark.parametrize("model", ["rb10", "rb10e"])
def test_step_holds_anchor_pose_despite_measurement_bias(model):
    kin = RBEEKinematics(model)
    measured = np.deg2rad(_PASSIVE_MEASURED_DEG)
    reference = np.deg2rad(_PASSIVE_REFERENCE_DEG)

    command = kin.inverse_step(
        kin.forward(reference),
        measured,
        max_joint_delta=0.5 / 30,
        previous_q=reference,
        max_tracking_error=0.05,
    )

    np.testing.assert_allclose(command, reference, atol=1e-12)


def test_step_rejects_previous_command_beyond_explicit_tracking_bound():
    kin = RBEEKinematics()
    measured = np.deg2rad(_PASSIVE_MEASURED_DEG)
    reference = np.deg2rad(_PASSIVE_REFERENCE_DEG)
    with pytest.raises(ValueError, match="exceeds max_tracking_error"):
        kin.inverse_step(
            kin.forward(reference),
            measured,
            max_joint_delta=0.5 / 30,
            previous_q=reference,
            max_tracking_error=0.01,
        )


@pytest.mark.parametrize("value", [0.0, -0.1, math.nan, math.inf, "bad"])
def test_step_tracking_error_validation(value):
    kin = RBEEKinematics()
    seed = np.zeros(6)
    with pytest.raises(ValueError, match="max_tracking_error"):
        kin.inverse_step(
            kin.forward(seed),
            seed,
            max_joint_delta=0.01,
            max_tracking_error=value,
        )


@pytest.mark.parametrize("model", ["rb10", "rb10e"])
def test_recorded_scale_goal_over_old_bound_is_recovered(model):
    kin = RBEEKinematics(model)
    seed = np.deg2rad([-168, -14, 147, 48, 283, 182])
    expected = seed + np.array([0.2464, -0.04, 0.03, 0.05, 0.01, -0.02])
    target = kin.forward(expected)
    with pytest.raises(RuntimeError):
        kin.inverse(target, seed, max_joint_delta=0.2)
    np.testing.assert_allclose(kin.inverse(target, seed), expected, atol=1e-6)


@pytest.mark.parametrize("model", ["rb10", "rb10e"])
@pytest.mark.parametrize("offset", [
    [0.04, -0.06, 0.02, 0.05, -0.03, 0.04],
    [0.2464, -0.04, 0.03, 0.05, 0.01, -0.02],
    [-0.1, 0.08, -0.04, 0.05, -0.02, 0.1],
])
def test_servo_waypoint_follows_cartesian_line_and_rotation(model, offset):
    kin = RBEEKinematics(model)
    seed = np.deg2rad([-168, -14, 147, 48, 283, 182])
    target = kin.forward(seed + offset)
    before = kin._transform(seed)
    goal_rotation = Rotation.from_euler("xyz", target[3:]).as_matrix()
    step = 0.5 / 30
    result = kin.inverse_step(target, seed, max_joint_delta=step, previous_q=seed)
    after = kin._transform(result)
    intended = target[:3] - before[:3, 3]
    actual = after[:3, 3] - before[:3, 3]
    fraction = float(np.dot(actual, intended) / np.dot(intended, intended))
    assert 0 < fraction <= 1
    np.testing.assert_allclose(actual, fraction * intended, atol=1e-6)
    requested_rotation = Rotation.from_matrix(goal_rotation @ before[:3, :3].T).as_rotvec()
    actual_rotation = Rotation.from_matrix(after[:3, :3] @ before[:3, :3].T).as_rotvec()
    np.testing.assert_allclose(actual_rotation, fraction * requested_rotation, atol=2e-6)
    assert np.max(np.abs(result - seed)) <= step + 1e-10


def test_step_rejects_unreachable_goal_and_disjoint_tracking_bounds():
    kin = RBEEKinematics()
    seed = np.deg2rad([40, -70, -100, 160, -60, 0])
    target = kin.forward(seed)
    with pytest.raises(ValueError, match="intersection"):
        kin.inverse_step(target, seed, max_joint_delta=0.01, previous_q=seed + 0.03)
    target[2] = 10
    with pytest.raises(RuntimeError):
        kin.inverse_step(target, seed, max_joint_delta=0.01)


def test_inverse_bounds_only_narrow_existing_limits():
    kin = RBEEKinematics()
    seed = np.deg2rad([40, -70, -100, 160, -60, 0])
    target = kin.forward(seed + 0.02)
    bounds = np.column_stack((seed - 0.005, seed + 0.005))
    with pytest.raises(RuntimeError):
        kin.inverse(target, seed, joint_bounds=bounds)
    with pytest.raises(ValueError, match="joint_bounds"):
        kin.inverse(target, seed, joint_bounds=np.full((6, 2), np.nan))
    with pytest.raises(ValueError, match="interval"):
        kin.inverse(target, seed, joint_bounds=np.column_stack((seed + 0.1, seed - 0.1)))


_ORIGINS = {
    "rb10": (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.197),
        (0.0, -0.1875, 0.6127),
        (0.0, 0.1484, 0.57015),
        (0.0, -0.11715, 0.0),
        (0.0, 0.0, 0.11715),
        (0.0, -0.1153, 0.0),
    ),
    "rb10e": (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.197),
        (0.0, -0.1875, 0.6127),
        (0.0, 0.1514, 0.57015),
        (0.0, -0.11715, 0.0),
        (0.0, 0.0, 0.11715),
        (0.0, -0.1153, 0.0),
    ),
}
_AXES = (
    (0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0),
)


def _oracle_rotation(axis, angle):
    axis = np.asarray(axis)
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _oracle_transform(model, q):
    """Independent direct homogeneous-transform oracle from source constants."""
    result = np.eye(4)
    for index, axis in enumerate(_AXES):
        translation = np.eye(4)
        translation[:3, 3] = _ORIGINS[model][index]
        rotation = np.eye(4)
        rotation[:3, :3] = _oracle_rotation(axis, q[index])
        result = result @ translation @ rotation
    tcp = np.eye(4)
    tcp[:3, 3] = _ORIGINS[model][-1]
    return result @ tcp


@pytest.mark.parametrize("model", ["rb10", "rb10e"])
def test_batched_fk_matches_independent_oracle(model):
    kinematics = RBEEKinematics(model)
    samples = np.random.default_rng(42).uniform(-2.0, 2.0, size=(16, 6))
    samples[0] = np.deg2rad([-168, -14, 147, 48, 283, 182])
    expected = np.stack([_oracle_transform(model, q) for q in samples])
    np.testing.assert_allclose(kinematics._transform(samples), expected, atol=1e-12)


@pytest.mark.parametrize("model", ["rb10", "rb10e"])
def test_goal_beyond_one_servo_step_remains_exact(model):
    kinematics = RBEEKinematics(model)
    seed = np.deg2rad([-168, -14, 147, 48, 283, 182])
    goal = seed + np.array([0.04, -0.06, 0.02, 0.05, -0.03, 0.04])
    pose = kinematics.forward(goal)
    with pytest.raises(RuntimeError, match="residual|converge"):
        kinematics.inverse(pose, seed, max_joint_delta=0.5 / 30)
    actual = kinematics.inverse(pose, seed)
    np.testing.assert_allclose(kinematics._transform(actual), _oracle_transform(model, goal), atol=1e-6)
    assert np.max(np.abs(actual - seed)) <= 0.2


@pytest.mark.parametrize(
    "model, expected_y", [("rb10", -0.27155), ("rb10e", -0.26855)]
)
def test_zero_pose_known_from_urdf(model, expected_y):
    pose = RBEEKinematics(model).forward(np.zeros(6))
    np.testing.assert_allclose(
        pose, [0.0, expected_y, 1.497, 0.0, 0.0, 0.0], atol=1e-12
    )


@pytest.mark.parametrize("model", ["rb10", "rb10e"])
def test_fk_matches_independent_oracle(model):
    q = np.array([0.31, -0.42, 0.57, -0.23, 0.36, -0.18])
    actual = RBEEKinematics(model).forward(q)
    expected = _oracle_transform(model, q)
    np.testing.assert_allclose(actual[:3], expected[:3, 3], atol=1e-12)
    np.testing.assert_allclose(
        Rotation.from_euler("xyz", actual[3:]).as_matrix(), expected[:3, :3], atol=1e-12
    )


def test_models_preserve_urdf_geometry_difference():
    q = np.zeros(6)
    delta = (
        RBEEKinematics("rb10e").forward(q)[:3]
        - RBEEKinematics("rb10").forward(q)[:3]
    )
    np.testing.assert_allclose(delta, [0.0, 0.003, 0.0], atol=1e-12)


@pytest.mark.parametrize("model", ["rb10", "rb10e"])
@pytest.mark.parametrize("degrees", [
    [-172.5253, -3.3101, 141.8867, 41.4234, 277.4747, 180.0],
    [-181.023782, 22.300725, 150.901241, -172.606516, 90.897401, -0.010878],
])
def test_recorded_and_vr_home_angles_keep_their_controller_branch(model, degrees):
    kinematics = RBEEKinematics(model)
    seed = np.deg2rad(degrees)
    pose = kinematics.forward(seed)
    expected = _oracle_transform(model, seed)
    np.testing.assert_allclose(pose[:3], expected[:3, 3], atol=1e-12)
    np.testing.assert_allclose(
        Rotation.from_euler("xyz", pose[3:]).as_matrix(), expected[:3, :3], atol=1e-12
    )
    target_q = seed + np.deg2rad([0.1, -0.1, 0.1, -0.1, 0.1, 0.1])
    target = kinematics.forward(target_q)
    solved = kinematics.inverse(target, seed, max_joint_delta=0.5 / 30)
    np.testing.assert_allclose(kinematics.forward(solved)[:3], target[:3], atol=1e-6)
    assert np.max(np.abs(solved - seed)) <= 0.5 / 30 + 1e-10
    # Crossing 180 degrees must never become a command near -180 degrees.
    if seed[5] == pytest.approx(math.pi):
        assert solved[5] > math.pi


def test_rb10_envelope_matches_the_existing_servoj_model():
    path = _MODULE_PATH.with_name("models.py")
    spec = importlib.util.spec_from_file_location("_rb_models_limits_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    np.testing.assert_allclose(
        RBEEKinematics("rb10").joint_limits,
        np.deg2rad(module.MODEL_SPECS["rb10"].joint_limits_deg),
    )


@pytest.mark.parametrize("model, elbow_deg", [("rb10", 165.0), ("rb10e", 154.0)])
def test_effective_safety_joint_limits(model, elbow_deg):
    kinematics = RBEEKinematics(model)
    expected = np.deg2rad([[-360.0, 360.0]] * 6)
    if model == "rb10e":
        expected[1] = np.deg2rad([-180.0, 180.0])
    expected[2] = np.deg2rad([-elbow_deg, elbow_deg])
    np.testing.assert_allclose(kinematics.joint_limits, expected, atol=1e-12)
    np.testing.assert_allclose(kinematics.safety_joint_limits, expected, atol=1e-12)

    returned = kinematics.joint_limits
    returned[:] = 0.0
    np.testing.assert_allclose(kinematics.joint_limits, expected, atol=1e-12)


@pytest.mark.parametrize("model", ["rb10", "rb10e"])
def test_fk_and_ik_reject_outside_conservative_envelope(model):
    kinematics = RBEEKinematics(model)
    zero_pose = kinematics.forward(np.zeros(6))
    elbow_170 = np.zeros(6)
    elbow_170[2] = math.radians(170.0)
    with pytest.raises(ValueError, match="effective safety joint limits"):
        kinematics.forward(elbow_170)
    with pytest.raises(ValueError, match="effective safety joint limits"):
        kinematics.inverse(zero_pose, elbow_170)

    base_361 = np.zeros(6)
    base_361[0] = math.radians(361.0)
    with pytest.raises(ValueError, match="effective safety joint limits"):
        kinematics.forward(base_361)
    with pytest.raises(ValueError, match="effective safety joint limits"):
        kinematics.inverse(zero_pose, base_361)


@pytest.mark.parametrize("model", ["rb10", "rb10e"])
def test_random_nearby_fk_ik_roundtrips(model):
    rng = np.random.default_rng(20260922)
    kinematics = RBEEKinematics(model, max_joint_delta=0.25)
    sample_count = 12
    converged = 0
    for _ in range(sample_count):
        seed = rng.uniform(-1.0, 1.0, size=6)
        target_q = seed + rng.uniform(-0.08, 0.08, size=6)
        target_pose = kinematics.forward(target_q)
        solved = kinematics.inverse(target_pose, seed)
        solved_pose = kinematics.forward(solved)
        np.testing.assert_allclose(solved_pose[:3], target_pose[:3], atol=1e-6)
        rotation_error = Rotation.from_matrix(
            Rotation.from_euler("xyz", target_pose[3:]).as_matrix()
            @ Rotation.from_euler("xyz", solved_pose[3:]).as_matrix().T
        ).magnitude()
        assert rotation_error <= 1e-6
        assert np.max(np.abs(solved - seed)) <= 0.25 + 1e-10
        converged += 1
    assert converged == sample_count


@pytest.mark.parametrize("model", ["rb10", "rb10e"])
def test_inverse_converges_near_elbow_safety_boundary(model):
    kinematics = RBEEKinematics(model, max_joint_delta=0.15)
    elbow_upper = kinematics.joint_limits[2, 1]
    target_q = np.array([0.3, -0.5, elbow_upper - 0.02, -0.4, 0.35, 0.2])
    seed = target_q.copy()
    seed[2] -= 0.08
    target_pose = kinematics.forward(target_q)

    solved = kinematics.inverse(target_pose, seed)
    solved_pose = kinematics.forward(solved)
    np.testing.assert_allclose(solved_pose[:3], target_pose[:3], atol=1e-6)
    rotation_error = Rotation.from_matrix(
        Rotation.from_euler("xyz", target_pose[3:]).as_matrix()
        @ Rotation.from_euler("xyz", solved_pose[3:]).as_matrix().T
    ).magnitude()
    assert rotation_error <= 1e-6
    assert solved[2] <= elbow_upper


def test_singular_known_pose_is_finite_and_bounded():
    kinematics = RBEEKinematics("rb10", max_joint_delta=0.1)
    pose = kinematics.forward(np.zeros(6))
    solved = kinematics.inverse(pose, np.zeros(6))
    assert np.all(np.isfinite(solved))
    np.testing.assert_allclose(solved, np.zeros(6), atol=1e-12)


@pytest.mark.parametrize(
    "call",
    [
        lambda k: k.forward([0.0] * 5),
        lambda k: k.forward([0.0, 0.0, 0.0, 0.0, 0.0, math.nan]),
        lambda k: k.forward([2 * math.pi + 0.001, 0.0, 0.0, 0.0, 0.0, 0.0]),
        lambda k: k.inverse([0.0] * 6, [2 * math.pi + 0.001, 0.0, 0.0, 0.0, 0.0, 0.0]),
        lambda k: k.inverse([0.0, 0.0, 0.0, 0.0, math.inf, 0.0], [0.0] * 6),
    ],
)
def test_invalid_inputs_are_rejected(call):
    with pytest.raises(ValueError):
        call(RBEEKinematics())


def test_unreachable_pose_is_rejected():
    with pytest.raises(RuntimeError, match="residual exceeds tolerance"):
        RBEEKinematics(max_joint_delta=1.0).inverse(
            [5.0, 0.0, 0.0, 0.0, 0.0, 0.0], np.zeros(6)
        )


def test_seed_continuity_bound_rejects_distant_solution():
    target_solver = RBEEKinematics(max_joint_delta=1.0)
    target_pose = target_solver.forward([0.0, -0.4, 0.45, -0.3, 0.25, 0.2])
    local_solver = RBEEKinematics(max_joint_delta=0.02)
    with pytest.raises(RuntimeError, match="residual exceeds tolerance"):
        local_solver.inverse(target_pose, np.zeros(6))


def test_inverse_per_call_joint_delta_override():
    kinematics = RBEEKinematics(max_joint_delta=0.01)
    seed = np.array([0.2, -0.4, 0.6, -0.3, 0.35, -0.2])
    target_q = seed + np.array([0.04, -0.03, 0.05, -0.04, 0.03, 0.04])
    target_pose = kinematics.forward(target_q)

    with pytest.raises(RuntimeError, match="residual exceeds tolerance"):
        kinematics.inverse(target_pose, seed)

    solved = kinematics.inverse(target_pose, seed, max_joint_delta=0.1)
    solved_pose = kinematics.forward(solved)
    np.testing.assert_allclose(solved_pose[:3], target_pose[:3], atol=1e-6)
    rotation_error = Rotation.from_matrix(
        Rotation.from_euler("xyz", target_pose[3:]).as_matrix()
        @ Rotation.from_euler("xyz", solved_pose[3:]).as_matrix().T
    ).magnitude()
    assert rotation_error <= 1e-6
    assert np.max(np.abs(solved - seed)) <= 0.1 + 1e-10
    assert kinematics.max_joint_delta == pytest.approx(0.01)


@pytest.mark.parametrize("value", [0.0, -0.1, math.nan, math.inf, "bad"])
def test_inverse_per_call_joint_delta_validation(value):
    kinematics = RBEEKinematics()
    pose = kinematics.forward(np.zeros(6))
    with pytest.raises(ValueError, match="max_joint_delta"):
        kinematics.inverse(pose, np.zeros(6), max_joint_delta=value)


def test_model_and_configuration_validation():
    assert RBEEKinematics().max_joint_delta == pytest.approx(0.5)
    with pytest.raises(ValueError, match="unsupported model"):
        RBEEKinematics("rb5")
    with pytest.raises(ValueError, match="max_joint_delta"):
        RBEEKinematics(max_joint_delta=0.0)


def test_missing_scipy_has_clear_startup_error(monkeypatch):
    real_import = builtins.__import__

    def import_without_scipy(name, *args, **kwargs):
        if name.startswith("scipy"):
            raise ModuleNotFoundError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_scipy)
    spec = importlib.util.spec_from_file_location("_rb_ee_no_scipy", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        with pytest.raises(ImportError, match="requires SciPy in the client environment"):
            spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
