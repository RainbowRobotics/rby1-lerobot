"""Robot profile: write while recording, compare / apply before inference."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import rby1_sdk as rby
from lerobot_robot_rby1 import Rby1, Rby1Config
from lerobot_robot_rby1 import profile as rprofile
from lerobot_robot_rby1.profile_cli import main as profile_cli

FAST = {"move_to_ready_on_connect": False, "use_gripper": False}


def _connect(**kw) -> Rby1:
    base = {"address": "stub.invalid:1", "model": "a", "version": "1.2", **FAST}
    robot = Rby1(Rby1Config(**{**base, **kw}))
    robot.connect()
    return robot


@pytest.fixture
def recorded(tmp_path):
    """Profile written by a 'recording' session, as lerobot-record would leave it."""
    path = tmp_path / "ds" / "meta" / rprofile.PROFILE_FILENAME
    robot = _connect(profile_out=str(path))
    robot.disconnect()
    return path


def test_profile_contents(recorded) -> None:
    data = json.loads(recorded.read_text())
    assert data["format"] == rprofile.FORMAT
    assert data["robot"]["model"] == "a" and data["robot"]["version"] == "1.2"
    assert data["robot"]["urdf_sha256"] and data["robot"]["pid_gains"]["right_arm"]
    assert len(data["core_parameters"]) == len(rby.CORE_PARAMETERS)
    # Recorded after configure(): the plugin's cutoff, not the robot's stored 15 Hz.
    assert data["core_parameters"]["joint_position_command.cutoff_frequency"]["value"] == 5.0
    assert data["robot_config"]["ee_arm_stiffness"][0] == 1075.0


def test_cutoff_is_session_only_and_restored() -> None:
    robot = _connect()
    fake = robot._robot
    assert json.loads(fake.get_parameter("joint_position_command.cutoff_frequency")) == 5.0
    robot.disconnect()
    assert json.loads(fake.get_parameter("joint_position_command.cutoff_frequency")) == 15.0
    assert fake.param_writes and all(write_db is False for _, _, write_db in fake.param_writes)


def test_same_setup_passes_check(recorded, caplog) -> None:
    with caplog.at_level(logging.INFO):
        robot = _connect(profile_in=str(recorded.parent.parent))  # dataset root
        robot.disconnect()
    assert "matches the reference profile" in caplog.text


def test_feature_mismatch_fails_before_connecting(recorded) -> None:
    n = len(rby.CALL_LOG)
    with pytest.raises(ValueError, match="use_torso"):
        _connect(profile_in=str(recorded), use_torso=True)
    assert "create_robot" not in [c[0] for c in rby.CALL_LOG[n:]]


def test_firmware_version_mismatch_is_an_error(recorded) -> None:
    with pytest.raises(ValueError, match="robot.version"):
        _connect(profile_in=str(recorded), version="1.3")


def test_model_mismatch_is_only_a_warning(recorded, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        robot = _connect(profile_in=str(recorded), model="m")
        robot.disconnect()
    assert "robot.model" in caplog.text


def test_behaviour_difference_warns_in_check_mode(recorded, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        robot = _connect(profile_in=str(recorded), ee_arm_damping_ratio=0.5)
        assert robot._config.ee_arm_damping_ratio == 0.5  # check does not change it
        robot.disconnect()
    assert "ee_arm_damping_ratio" in caplog.text


def test_apply_adopts_config_and_core_parameters_for_the_session(recorded) -> None:
    data = json.loads(recorded.read_text())
    data["robot_config"]["ee_arm_stiffness"] = [70.0] * 7
    data["robot_config"]["ee_arm_target_gains"] = [1.0, 2.0, 3.0, 4.0]
    data["core_parameters"]["default.velocity_limit_scaling"]["value"] = 0.5
    recorded.write_text(json.dumps(data))

    robot = _connect(profile_in=str(recorded), profile_mode="apply", address="other.invalid:1")
    fake = robot._robot
    assert robot._config.ee_arm_stiffness == [70.0] * 7
    assert robot._config.ee_arm_target_gains == (1.0, 2.0, 3.0, 4.0)
    assert robot._config.address == "other.invalid:1"  # recorded-only, never applied
    assert json.loads(fake.get_parameter("default.velocity_limit_scaling")) == 0.5
    assert fake.params_db["default.velocity_limit_scaling"] == 1.0  # nothing persisted
    robot.disconnect()
    assert json.loads(fake.get_parameter("default.velocity_limit_scaling")) == 1.0


def test_one_profile_per_dataset(recorded) -> None:
    with pytest.raises(ValueError, match="one profile per dataset"):
        _connect(profile_out=str(recorded), ee_arm_damping_ratio=0.5)
    robot = _connect(profile_out=str(recorded), ee_arm_damping_ratio=0.5, profile_force=True)
    robot.disconnect()
    extras = [p for p in recorded.parent.glob("rby1_profile.*.json")]
    assert len(extras) == 1
    assert json.loads(recorded.read_text())["robot_config"]["ee_arm_damping_ratio"] == 1.0


def test_teleop_section_is_kept(tmp_path) -> None:
    path = tmp_path / "meta" / rprofile.PROFILE_FILENAME
    rprofile.update_profile_section(path, "teleop", {"type": "rby1_leader_arm", "config": {"init_duration": 5.0}})
    robot = _connect(profile_out=str(path))  # the robot connects after the teleop
    robot.disconnect()
    data = json.loads(path.read_text())
    assert data["teleop"]["type"] == "rby1_leader_arm" and data["robot_config"]


def test_cli_attach_and_diff(recorded, tmp_path, capsys) -> None:
    other = tmp_path / "other.json"
    data = json.loads(recorded.read_text())
    data["robot_config"]["use_torque"] = True
    other.write_text(json.dumps(data))

    assert profile_cli(["diff", str(recorded), str(recorded)]) == 0
    assert profile_cli(["diff", str(recorded), str(other)]) == 1
    assert "[error] robot_config.use_torque" in capsys.readouterr().out

    ds = tmp_path / "ds2"
    (ds / "meta").mkdir(parents=True)
    assert profile_cli(["attach", str(recorded), "--dataset", str(ds)]) == 0
    assert (ds / "meta" / rprofile.PROFILE_FILENAME).is_file()
    assert profile_cli(["attach", str(other), "--dataset", str(ds)]) == 1  # one profile per dataset
    assert "one profile per dataset" in capsys.readouterr().err
    assert profile_cli(["attach", str(other), "--dataset", str(ds), "--force"]) == 0


def test_cli_export_is_read_only(tmp_path) -> None:
    out = tmp_path / "export.json"
    n = len(rby.CALL_LOG)
    assert profile_cli(["export", "--out", str(out), "--robot.address=stub.invalid:1", "--robot.model=a",
                        "--robot.version=1.2"]) == 0
    calls = [c[0] for c in rby.CALL_LOG[n:]]
    assert "power_on" not in calls and "servo_on" not in calls and "enable_control_manager" not in calls
    data = rprofile.load_profile(out)
    assert data["core_parameters"]["joint_position_command.cutoff_frequency"]["value"] == 15.0


# ── Format 1 (plugin 0.3.0) profiles ────────────────────────────────────────
V1_PROFILE = Path(__file__).parent / "data" / "profile_v1.json"  # written by the v0.3.0 code


def test_v1_profile_is_upgraded_to_current_field_names() -> None:
    assert json.loads(V1_PROFILE.read_text())["format_version"] == 1
    data = rprofile.load_profile(V1_PROFILE)
    cfg = data["robot_config"]
    assert data["format_version"] == rprofile.FORMAT_VERSION
    # Every field is a current Rby1Config field, and none is missing.
    current = rprofile.to_jsonable(Rby1Config(address="stub.invalid:1"))
    assert set(cfg) == set(current)
    assert cfg["joint_impedance_torso_stiffness"] == [400.0] * 6
    assert cfg["joint_impedance_arm_stiffness"] == [150.0] * 7
    assert cfg["ee_arm_damping_ratio"] == 1.0 and cfg["ee_action_period"] == pytest.approx(1 / 15)
    # The fixture's left arm differed from the right in two places.
    assert set(data[rprofile.UPGRADE_NOTES_KEY]) == {"ee_arm_damping_ratio", "joint_impedance_arm_stiffness"}


def test_v1_profile_compares_and_applies(tmp_path, caplog) -> None:
    ref = rprofile.load_profile(V1_PROFILE)
    cur = rprofile.build_profile(
        Rby1Config(address="stub.invalid:1", model="a", version="1.2", use_torso=True, action_mode="ee"),
        robot={"model": "a", "version": "1.2"}, version="1.2",
    )
    diffs = [d for d in rprofile.diff_profiles(ref, cur) if d.severity != rprofile.INFO]
    # Same settings: only the left-arm values that no longer exist, the fixed
    # EE period of 0.3.0 and the impedance defaults changed in 0.3.1
    # are reported.
    assert sorted(d.key for d in diffs) == [
        "robot_config.ee_arm_damping_ratio", "robot_config.ee_arm_stiffness",
        "robot_config.ee_arm_torque_limit", "robot_config.ee_auto_action_period",
        "robot_config.joint_impedance_arm_stiffness",
        "robot_config.joint_impedance_arm_torque_limit",
        "robot_config.joint_impedance_torso_torque_limit",
    ]
    assert all(d.severity == rprofile.WARN for d in diffs)
    # The old profile recorded hard-coded arm limits; the limits now come from
    # the robot (robot.joint_limits), so the missing constant is not a difference.
    assert "arm_q_limits" in ref["plugin_constants"] and "arm_q_limits" not in cur["plugin_constants"]

    data = json.loads(V1_PROFILE.read_text())
    data["robot_config"]["right_arm_stiffness"] = [70.0] * 7
    data["robot_config"]["min_time_factor"] = 1.5
    path = tmp_path / "v1.json"
    path.write_text(json.dumps(data))
    robot = _connect(profile_in=str(path), profile_mode="apply", use_torso=True, action_mode="ee",
                     use_gripper=True)
    assert robot._config.ee_arm_stiffness == [70.0] * 7
    assert robot._config.ee_min_time_factor == 1.5
    assert robot._config.ee_auto_action_period is False  # as recorded with 0.3.0
    robot.disconnect()


def test_v1_joint_mode_torso_profile_is_rejected(tmp_path) -> None:
    """0.3.0 joint mode observed the torso without commanding it: other action keys."""
    data = json.loads(V1_PROFILE.read_text())
    data["robot_config"]["action_mode"] = "joint"
    path = tmp_path / "v1.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="torso observed but not commanded"):
        _connect(profile_in=str(path), use_torso=True, use_gripper=True)
