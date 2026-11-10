"""RB-Y1 robot profile: the settings that make a dataset and a policy consistent.

A profile is a JSON snapshot of everything that changes how the robot responds
to the same action:

* ``robot``            model / firmware version, core and SDK versions, URDF
                       hash, URDF joint limits and the joint drivers' PID gains
* ``core_parameters``  every parameter the core exposes (``get_parameter_list``)
* ``robot_config``     the :class:`~lerobot_robot_rby1.config_rby1.Rby1Config`
* ``plugin_constants`` behaviour-relevant values hard-coded in this plugin
* ``teleop``           the teleoperator's config (recorded only)

It is written next to a dataset (``<root>/meta/rby1_profile.json``, which
``LeRobotDataset`` uploads and downloads with the rest of ``meta/``) while
recording, and compared against the live robot before inference, so a policy
trained on one robot is not silently run on a differently configured one.

Severity of a difference (see :func:`diff_profiles`):

* ``error`` -- the observation/action layout or the joint semantics differ
  (feature flags, camera geometry, firmware version): the policy cannot run.
* ``warn``  -- the robot responds differently (gains, limits, core parameters,
  model A vs M, URDF, PID gains).
* ``info``  -- bookkeeping (plugin / lerobot / SDK versions).

Values that only describe the setup (address, camera serials, hotspot, battery,
...) are recorded but never compared.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

FORMAT = "rby1-lerobot-profile"
FORMAT_VERSION = 2  # 1: plugin 0.3.0 field names (upgraded on load, see _upgrade_v1)
PROFILE_FILENAME = "rby1_profile.json"
# Where profiles go when Rby1Config.profile_out is not set (override with RBY1_PROFILE_DIR).
DEFAULT_PROFILE_DIR = Path("~/.cache/rby1_lerobot/profiles")

# ── Core parameters (core/robots/rby1a/parameters.h) ───────────────────────
# Compared and, with profile_mode="apply", set for the session (write_db=False).
APPLIED_CORE_PARAMETERS = frozenset(
    {
        "cartesian_command.cutoff_frequency",
        "optimal_control_command.cutoff_frequency",
        "joint_group_position_command.cutoff_frequency",
        "default.velocity_limit_scaling",
        "default.acceleration_limit_scaling",
        "default.torque_limit_scaling",
        "default.linear_acceleration_limit",
        "default.angular_acceleration_limit",
        "default.se2_velocity_command__acceleration_limit",
        "joint_position_command.canceling_time",
        "cartesian_command.canceling_time",
        "stop_command.canceling_time",
        "joint_velocity_command.canceling_time",
        "se2_velocity_command.canceling_time",
        "joint_impedance_control_command.canceling_time",
        "cartesian_impedance_control_command.canceling_time",
        "joint_group_position_command.canceling_time",
        "friction_compensation_enabled",
        "friction_compensation_gain",
        "manipulability_threshold",
        "soft_stop_enabled",
        "real_time_control.max_timeout_count",
    }
)
# Compared but never applied: joint_position_command.cutoff_frequency is owned
# by Rby1Config.joint_position_cutoff_frequency (applied through the config);
# gravity describes how the robot is mounted.
COMPARED_CORE_PARAMETERS = APPLIED_CORE_PARAMETERS | {
    "joint_position_command.cutoff_frequency",
    "gravity",
}
# Every other core parameter (identity, battery, network, boot/power targets,
# command timeouts, jog, stop tracking errors the plugin overrides per command)
# is recorded only.

# ── Rby1Config fields ───────────────────────────────────────────────────────
# Different values change the observation/action features: the policy's input
# and output layout no longer match.
FEATURE_FIELDS = frozenset(
    {
        "use_torso",
        "use_right_arm",
        "use_left_arm",
        "use_gripper",
        "use_mobile_base",
        "use_velocity",
        "use_torque",
        "action_mode",
    }
)
CAMERA_ERROR_KEYS = frozenset({"width", "height", "rotation"})
CAMERA_WARN_KEYS = frozenset({"fps"})
# Describe the setup, not the behaviour.
RECORDED_ONLY_FIELDS = frozenset(
    {
        "id",
        "calibration_dir",
        "address",
        "connect_timeout_sec",
        "move_to_ready_on_connect",
        "model",
        "version",
        "profile_out",
        "profile_in",
        "profile_mode",
        "profile_force",
    }
)

ERROR, WARN, INFO = "error", "warn", "info"
_SEVERITY_ORDER = {ERROR: 0, WARN: 1, INFO: 2}


@dataclass
class Diff:
    key: str  # dotted path, e.g. "robot_config.ee_arm_stiffness"
    reference: Any
    current: Any
    severity: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.key}: profile={_short(self.reference)} robot={_short(self.current)}"


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------
def to_jsonable(value: Any) -> Any:
    """Convert dataclasses, enums, numpy and paths into plain JSON values."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        out = {f.name: to_jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
        type_name = getattr(value, "type", None)
        if isinstance(type_name, str):
            out["type"] = type_name
        return out
    if isinstance(value, enum.Enum):
        return to_jsonable(value.value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _short(value: Any, limit: int = 80) -> str:
    text = json.dumps(value, default=str)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _equal(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    return a == b


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------
def read_core_parameters(sdk_robot: Any) -> dict[str, dict[str, Any]]:
    """All core parameters as ``{name: {"type": code, "value": json value}}``."""
    params: dict[str, dict[str, Any]] = {}
    for name, type_code in sdk_robot.get_parameter_list():
        try:
            params[name] = {"type": int(type_code), "value": json.loads(sdk_robot.get_parameter(name))}
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"profile: cannot read core parameter {name}: {exc}")
    return params


def _pid_gains(sdk_robot: Any) -> dict[str, Any]:
    gains: dict[str, Any] = {}
    for group in ("torso", "right_arm", "left_arm", "head"):
        getter = getattr(sdk_robot, f"get_{group}_position_pid_gains", None)
        if getter is None:
            continue
        try:
            gains[group] = [[g.p_gain, g.i_gain, g.d_gain] for g in getter()]
        except Exception as exc:  # noqa: BLE001  (drivers may be unpowered)
            gains[group] = f"<unavailable: {exc.__class__.__name__}>"
    return gains


def default_semantics(model: str, version: str) -> str:
    """The semantics tag RB-Y1 datasets, server configs and rollouts share.

    lerobot's remote inference admits a client only when its semantics string
    equals the deployment's; it names the units and conventions of the state /
    action vectors (joint radians, gripper 0..1 open=1, base m/s and rad/s).
    """
    return f"rby1-{model}-v{version.replace('.', '')}-radians-gripper01-v1"


def read_robot_section(sdk_robot: Any, model: str, version: str) -> dict[str, Any]:
    """Identity and model data of the connected robot (read-only calls)."""
    import rby1_sdk as rby

    section: dict[str, Any] = {
        "model": model,
        "version": version,
        "semantics": default_semantics(model, version),
        "sdk_version": getattr(rby, "__version__", None),
    }
    try:
        info = sdk_robot.get_robot_info()
        section["core_version"] = getattr(info, "version", None)
        section["robot_version"] = getattr(info, "robot_version", None)
        section["robot_model_name"] = getattr(info, "robot_model_name", None)
    except Exception as exc:  # noqa: BLE001
        section["core_version"] = f"<unavailable: {exc}>"
    try:
        section["urdf_sha256"] = hashlib.sha256(sdk_robot.get_robot_model().encode()).hexdigest()
    except Exception as exc:  # noqa: BLE001
        section["urdf_sha256"] = f"<unavailable: {exc.__class__.__name__}>"
    try:
        dyn = sdk_robot.get_dynamics()
        names = list(sdk_robot.model().robot_joint_names)
        state = dyn.make_state([], names)
        section["joint_limits"] = {
            "names": names,
            "q_lower": to_jsonable(dyn.get_limit_q_lower(state)),
            "q_upper": to_jsonable(dyn.get_limit_q_upper(state)),
            "qdot_upper": to_jsonable(dyn.get_limit_qdot_upper(state)),
            "qddot_upper": to_jsonable(dyn.get_limit_qddot_upper(state)),
        }
    except Exception as exc:  # noqa: BLE001
        section["joint_limits"] = f"<unavailable: {exc.__class__.__name__}>"
    section["pid_gains"] = _pid_gains(sdk_robot)
    return section


def plugin_constants(version: str) -> dict[str, Any]:
    """Behaviour-relevant values hard-coded in this plugin (not in Rby1Config)."""
    from . import command_builders as cb
    from . import constants as c
    from .constants import ready_pose_for_version

    body, right, left, head = ready_pose_for_version(version)
    return to_jsonable(
        {
            "ready_pose": {"body": body, "head": head},
            "joint_command_hold_time": cb._CONTROL_HOLD_TIME,
            "gripper": {
                "homing_torque": c.GRIPPER_HOMING_TORQUE,
                "position_torque": c.GRIPPER_POSITION_TORQUE,
            },
        }
    )


def _versions() -> dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for key, dist in (("plugin_version", "lerobot_robot_rby1"), ("lerobot_version", "lerobot")):
        try:
            out[key] = version(dist)
        except PackageNotFoundError:
            out[key] = None
    return out


def build_profile(
    robot_config: Any,
    *,
    robot: dict[str, Any] | None = None,
    core_parameters: dict[str, Any] | None = None,
    version: str = "1.2",
) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **_versions(),
        "robot": robot or {},
        "core_parameters": core_parameters or {},
        "robot_config": to_jsonable(robot_config),
        "plugin_constants": plugin_constants(version),
    }
    return profile


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    else:
        out[prefix] = value


def _severity(key: str) -> str | None:
    """Severity of a difference at a dotted key, ``None`` if it is not compared."""
    section, _, rest = key.partition(".")
    if section == "robot_config":
        field = rest.split(".", 1)[0]
        if field in RECORDED_ONLY_FIELDS:
            return None
        if field == "cameras":
            parts = rest.split(".")
            if len(parts) < 3:
                return ERROR  # a camera added or removed
            if parts[2] in CAMERA_ERROR_KEYS:
                return ERROR
            if parts[2] in CAMERA_WARN_KEYS:
                return WARN
            return None  # type, serial, colour mode, ...
        if field in FEATURE_FIELDS:
            return ERROR
        return WARN
    if section == "robot":
        field = rest.split(".", 1)[0]
        if field == "version":
            return ERROR
        if field in ("sdk_version", "core_version", "robot_version", "robot_model_name", "semantics"):
            return INFO
        return WARN  # model (A vs M), urdf_sha256, joint_limits, pid_gains
    if section == "core_parameters":
        name = rest.rsplit(".", 1)[0] if rest.endswith((".value", ".type")) else rest
        return WARN if name in COMPARED_CORE_PARAMETERS else None
    if section == "plugin_constants":
        if rest.split(".", 1)[0] == "arm_q_limits":
            # Recorded by older plugins, which clipped to hard-coded limits. The
            # limits now come from the robot and are compared as robot.joint_limits.
            return None
        return WARN
    if section in ("plugin_version", "lerobot_version"):
        return INFO
    return None  # format, created_at, teleop, ...


def diff_profiles(reference: dict[str, Any], current: dict[str, Any]) -> list[Diff]:
    """Differences between a reference profile and the current one, most severe first.

    Sections missing from either side (e.g. a profile exported without a robot
    connection) are not compared.
    """
    diffs: list[Diff] = []
    for section in ("robot", "core_parameters", "robot_config", "plugin_constants"):
        ref_sec, cur_sec = reference.get(section), current.get(section)
        if not ref_sec or not cur_sec:
            continue
        # Camera dicts: compare the set of names as a whole first.
        if section == "robot_config":
            ref_cams = (ref_sec.get("cameras") or {}).keys()
            cur_cams = (cur_sec.get("cameras") or {}).keys()
            if set(ref_cams) != set(cur_cams):
                diffs.append(Diff("robot_config.cameras", sorted(ref_cams), sorted(cur_cams), ERROR))
        ref_flat: dict[str, Any] = {}
        cur_flat: dict[str, Any] = {}
        _flatten(section, ref_sec, ref_flat)
        _flatten(section, cur_sec, cur_flat)
        for key in sorted(ref_flat.keys() | cur_flat.keys()):
            severity = _severity(key)
            if severity is None:
                continue
            if section == "robot_config" and key.startswith("robot_config.cameras.") and key.count(".") < 3:
                continue
            ref_v, cur_v = ref_flat.get(key, "<missing>"), cur_flat.get(key, "<missing>")
            if not _equal(ref_v, cur_v):
                diffs.append(Diff(key, ref_v, cur_v, severity))
    # Values a format-1 profile held that the current fields cannot express.
    for side, other in ((reference, current), (current, reference)):
        for notes_key, severity in ((UPGRADE_NOTES_KEY, WARN), (UPGRADE_ERRORS_KEY, ERROR)):
            for name, note in (side.get(notes_key) or {}).items():
                if any(d.key == f"robot_config.{name}" for d in diffs):
                    continue
                if _equal(note, (other.get(notes_key) or {}).get(name)):
                    continue  # both sides are the same old profile
                shown = (other.get("robot_config") or {}).get(name, "<missing>")
                pair = (note, shown) if side is reference else (shown, note)
                diffs.append(Diff(f"robot_config.{name}", *pair, severity))
    for key in ("plugin_version", "lerobot_version"):
        if key in reference and key in current and reference[key] != current[key]:
            diffs.append(Diff(key, reference[key], current[key], INFO))
    diffs.sort(key=lambda d: (_SEVERITY_ORDER[d.severity], d.key))
    return diffs


def format_diffs(diffs: list[Diff]) -> str:
    return "\n".join(f"  {d}" for d in diffs) if diffs else "  (no differences)"


# ---------------------------------------------------------------------------
# Applying
# ---------------------------------------------------------------------------
def apply_robot_config(config: Any, reference: dict[str, Any]) -> list[str]:
    """Overwrite the behaviour fields of ``config`` (warn severity) with the profile's.

    Feature fields, camera geometry and recorded-only fields are never touched.
    Returns the names of the fields that changed.
    """
    ref = reference.get("robot_config") or {}
    changed = []
    for f in dataclasses.fields(config):
        name = f.name
        if name not in ref or name in FEATURE_FIELDS or name in RECORDED_ONLY_FIELDS or name == "cameras":
            continue
        current = getattr(config, name)
        if _equal(to_jsonable(current), ref[name]):
            continue
        setattr(config, name, _like(current, ref[name]))
        changed.append(name)
    return changed


def _like(template: Any, value: Any) -> Any:
    """Convert a JSON value back to the container types of ``template``."""
    if isinstance(template, tuple):
        return tuple(_like(t, v) for t, v in zip(template, value)) if len(template) == len(value) else tuple(value)
    if isinstance(template, dict):
        sample = next(iter(template.values()), None)
        return {k: _like(sample, v) if sample is not None else v for k, v in value.items()}
    if isinstance(template, np.ndarray):
        return np.asarray(value, dtype=template.dtype)
    if isinstance(template, float) and isinstance(value, (int, float)):
        return float(value)
    return value


def apply_core_parameters(
    sdk_robot: Any, reference: dict[str, Any], current: dict[str, Any]
) -> Callable[[], None]:
    """Set the reference's applicable core parameters for this session only.

    Uses ``set_parameter(..., write_db=False)`` so nothing is persisted on the
    robot, and returns a function that restores the previous values.
    """
    ref = reference.get("core_parameters") or {}
    restore: list[tuple[str, str]] = []
    for name in sorted(APPLIED_CORE_PARAMETERS):
        if name not in ref or name not in current:
            continue
        if _equal(ref[name]["value"], current[name]["value"]):
            continue
        new, old = json.dumps(ref[name]["value"]), json.dumps(current[name]["value"])
        if sdk_robot.set_parameter(name, new, write_db=False):
            restore.append((name, old))
            logger.info(f"profile apply: {name} {old} -> {new} (this session only)")
        else:
            logger.warning(f"profile apply: robot rejected {name}={new}")

    def _restore() -> None:
        for name, old in reversed(restore):
            try:
                sdk_robot.set_parameter(name, old, write_db=False)
            except Exception as exc:  # noqa: BLE001
                logger.error(f"profile restore: {name} failed: {exc}")

    return _restore


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------
# Format 1 (plugin 0.3.0) -> format 2 (plugin 0.3.1 and later) Rby1Config field names.
_V1_RENAMES = {
    "startup_ramp_duration": "joint_startup_ramp_duration",
    "startup_min_time": "joint_startup_min_time",
    "normal_min_time": "joint_min_time",
    "acceleration_limit_scale": "joint_acceleration_limit_scale",
    "use_impedance": "use_joint_impedance",
    "impedance_damping_ratio": "joint_impedance_damping_ratio",
    "wrist_velocity_limit_scale": "joint_impedance_wrist_velocity_limit_scale",
    "ee_dt": "ee_action_period",
    "min_time_factor": "ee_min_time_factor",
    "torso_stiffness": "ee_torso_stiffness",
    "torso_torque_limit": "ee_torso_torque_limit",
    "torso_damping_ratio": "ee_torso_damping_ratio",
    "torso_joint_limits": "ee_torso_joint_limits",
    "torso_target_gains": "ee_torso_target_gains",
    "right_arm_joint_limits": "ee_right_arm_joint_limits",
    "left_arm_joint_limits": "ee_left_arm_joint_limits",
    "arm_target_gains": "ee_arm_target_gains",
    "null_right_arm_deg": "ee_null_right_arm_deg",
    "null_left_arm_deg": "ee_null_left_arm_deg",
    "nullspace_weight": "ee_nullspace_weight",
    "nullspace_stiffness": "ee_nullspace_stiffness",
    "nullspace_damping_ratio": "ee_nullspace_damping_ratio",
}
# Format 1 had one value per arm; format 2 has one for both (the right arm's is kept).
_V1_ARM_PAIRS = {
    "ee_arm_stiffness": ("right_arm_stiffness", "left_arm_stiffness"),
    "ee_arm_torque_limit": ("right_arm_torque_limit", "left_arm_torque_limit"),
    "ee_arm_damping_ratio": ("right_arm_damping_ratio", "left_arm_damping_ratio"),
}
# Format 1 had 20-element [torso 6 | right arm 7 | left arm 7] arrays.
_V1_BODY_ARRAYS = {
    "impedance_stiffness": ("joint_impedance_torso_stiffness", "joint_impedance_arm_stiffness"),
    "impedance_torque_limit": ("joint_impedance_torso_torque_limit", "joint_impedance_arm_torque_limit"),
}
# Top-level key: {field: note} for format-1 values the current fields cannot hold.
UPGRADE_NOTES_KEY = "upgrade_notes"
# Top-level key: {field: note} for format-1 settings whose meaning changed, so
# the profile's dataset no longer matches the features of the same config.
UPGRADE_ERRORS_KEY = "upgrade_errors"


def _upgrade_v1(data: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a format-1 profile's ``robot_config`` with the current field names."""
    old = data.get("robot_config") or {}
    if not old:
        return {**data, "format_version": FORMAT_VERSION}
    cfg = {_V1_RENAMES.get(k, k): v for k, v in old.items()}
    cfg.setdefault("ee_auto_action_period", False)  # 0.3.0 used the fixed ee_dt
    notes: dict[str, str] = {}
    for new, (right, left) in _V1_ARM_PAIRS.items():
        cfg.pop(right, None)
        cfg.pop(left, None)
        if right in old:
            cfg[new] = old[right]
            if left in old and not _equal(old[right], old[left]):
                notes[new] = f"format-1 profile had a different left arm value: {_short(old[left])}"
    for name, (torso, arm) in _V1_BODY_ARRAYS.items():
        cfg.pop(name, None)
        values = old.get(name)
        if isinstance(values, list) and len(values) == 20:
            cfg[torso], cfg[arm] = values[:6], values[6:13]
            if not _equal(values[6:13], values[13:20]):
                notes[arm] = f"format-1 profile had a different left arm value: {_short(values[13:20])}"
    out = {**data, "format_version": FORMAT_VERSION, "upgraded_from_format": 1, "robot_config": cfg}
    if notes:
        out[UPGRADE_NOTES_KEY] = notes
    if cfg.get("action_mode") == "joint" and cfg.get("use_torso"):
        # Plugin 0.3.0 observed the torso in joint mode but did not command it:
        # such datasets have no torso_*.pos action keys.
        out[UPGRADE_ERRORS_KEY] = {
            "use_torso": "true, torso observed but not commanded (plugin 0.3.0 joint mode; "
            "use plugin 0.3.0, commit 13d68d4, with this dataset)"
        }
    return out


def upgrade_profile(data: dict[str, Any], source: Any = "profile") -> dict[str, Any]:
    """Bring a profile dict to the current format (older formats are converted)."""
    version = int(data.get("format_version", 0))
    if version > FORMAT_VERSION:
        raise ValueError(f"{source} uses profile format {version}; this plugin reads {FORMAT_VERSION}.")
    return _upgrade_v1(data) if version < 2 else data


def load_profile(path: str | os.PathLike) -> dict[str, Any]:
    data = json.loads(Path(path).read_text())
    if data.get("format") != FORMAT:
        raise ValueError(f"{path} is not an RB-Y1 profile (format={data.get('format')!r}).")
    return upgrade_profile(data, path)


def resolve_profile_source(source: str) -> Path:
    """Find a profile from a file, a dataset root or a Hugging Face dataset repo id."""
    path = Path(source).expanduser()
    if path.is_file():
        return path
    if path.is_dir():
        candidate = path / "meta" / PROFILE_FILENAME
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"No meta/{PROFILE_FILENAME} in dataset root {path}.")
    # Local lerobot dataset cache first, then the Hub (same file lerobot downloads with meta/).
    try:
        from lerobot.utils.constants import HF_LEROBOT_HOME

        cached = Path(HF_LEROBOT_HOME) / source / "meta" / PROFILE_FILENAME
        if cached.is_file():
            return cached
    except Exception:  # noqa: BLE001
        pass
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(source, f"meta/{PROFILE_FILENAME}", repo_type="dataset"))


def write_profile(path: str | os.PathLike, profile: dict[str, Any], force: bool = False) -> Path:
    """Write ``profile`` to ``path``, keeping one profile per dataset.

    If ``path`` already holds a robot profile that differs (error or warn), the
    write is refused unless ``force``; with ``force`` the new profile goes to a
    time-stamped sibling and the original stays. A ``teleop`` section already in
    the file (written by the teleoperator, which connects first) is kept.
    """
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        existing = upgrade_profile(json.loads(path.read_text()), path)
        merged = {**existing, **{k: v for k, v in profile.items() if k != "teleop"}}
        for key in (UPGRADE_NOTES_KEY, UPGRADE_ERRORS_KEY, "upgraded_from_format"):
            merged.pop(key, None)
        if existing.get("robot_config"):
            diffs = [d for d in diff_profiles(existing, profile) if d.severity in (ERROR, WARN)]
            if diffs:
                if not force:
                    raise ValueError(
                        f"{path} already holds a different RB-Y1 profile (one profile per dataset):\n"
                        f"{format_diffs(diffs)}\nRecord into a new dataset, or set profile_force=True "
                        "to keep this one as an extra time-stamped profile."
                    )
                extra = path.with_name(f"{path.stem}.{time.strftime('%Y%m%d-%H%M%S')}{path.suffix}")
                extra.write_text(json.dumps(profile, indent=2) + "\n")
                logger.warning(f"Profile differs from {path}; wrote {extra} instead.")
                return extra
        path.write_text(json.dumps(merged, indent=2) + "\n")
        return path
    path.write_text(json.dumps(profile, indent=2) + "\n")
    return path


def update_profile_section(path: str | os.PathLike, key: str, value: Any) -> Path:
    """Set one top-level section (e.g. ``teleop``) in a profile file, creating it if needed."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        upgrade_profile(json.loads(path.read_text()), path)
        if path.is_file()
        else {"format": FORMAT, "format_version": FORMAT_VERSION}
    )
    data[key] = to_jsonable(value)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def default_profile_path() -> Path:
    """A new, unique file under the profile directory (one per connect)."""
    directory = Path(os.environ.get("RBY1_PROFILE_DIR", DEFAULT_PROFILE_DIR)).expanduser()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    n = 0
    while True:
        path = directory / f"{stamp}-{os.getpid()}-{n}.json"
        if not path.exists():
            return path
        n += 1
