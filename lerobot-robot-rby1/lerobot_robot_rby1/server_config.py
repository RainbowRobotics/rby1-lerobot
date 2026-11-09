"""Generate a ``lerobot-policy-server`` config (server.yaml) for an RB-Y1 checkpoint.

lerobot 0.7.0's remote inference admits a rollout client only when the
deployment's feature names, shapes and semantics string match the robot's
exactly. Writing that YAML by hand is error-prone, so this derives it from the
dataset the policy was trained on (``meta/info.json`` carries the component
names and the HWC image shapes) and cross-checks it against the checkpoint's
``config.json`` and, when available, the dataset's RB-Y1 profile.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from . import profile as rprofile

logger = logging.getLogger(__name__)

DEFAULT_LISTEN = "tcp/0.0.0.0:7447"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def resolve_checkpoint(checkpoint: str) -> Path:
    """Local directory, or a Hub model repo (downloaded)."""
    path = Path(checkpoint).expanduser()
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(checkpoint, repo_type="model"))


def resolve_dataset_info(dataset: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """``(meta/info.json, meta/rby1_profile.json or None)`` from a dataset root or Hub repo id."""
    root = Path(dataset).expanduser()
    if root.is_dir():
        info = _load_json(root / "meta" / "info.json")
        prof_path = root / "meta" / rprofile.PROFILE_FILENAME
        profile = rprofile.load_profile(prof_path) if prof_path.is_file() else None
        return info, profile
    from huggingface_hub import hf_hub_download

    info = _load_json(Path(hf_hub_download(dataset, "meta/info.json", repo_type="dataset")))
    try:
        profile = rprofile.load_profile(rprofile.resolve_profile_source(dataset))
    except Exception:  # noqa: BLE001  (dataset recorded without a profile)
        profile = None
    return info, profile


def dataset_from_checkpoint(checkpoint_dir: Path) -> str | None:
    """The dataset the checkpoint was trained on (train_config.json), if recorded."""
    train_cfg = checkpoint_dir / "train_config.json"
    if not train_cfg.is_file():
        return None
    dataset = _load_json(train_cfg).get("dataset") or {}
    root = dataset.get("root")
    if root and Path(root).is_dir():
        return root
    return dataset.get("repo_id")


def _feature_entry(name: str, feature: dict[str, Any], semantics: str) -> dict[str, Any]:
    """Mirror lerobot.rollout.context._feature_spec: video/image -> rgb uint8 without names."""
    image = feature["dtype"] in {"video", "image"}
    entry: dict[str, Any] = {
        "name": name,
        "shape": list(feature["shape"]),
        "dtype": "uint8" if image else feature["dtype"],
        "kind": "rgb" if image else "tensor",
        "semantics": f"rgb-{name.rsplit('.', 1)[-1]}-v1" if image else semantics,
    }
    if not image:
        entry["names"] = list(feature.get("names") or [])
    return entry


def build_server_config(
    *,
    checkpoint: str,
    dataset: str | None = None,
    deployment: str = "rby1",
    semantics: str | None = None,
    device: str = "cuda",
    listen_endpoint: str = DEFAULT_LISTEN,
    fps: float | None = None,
    n_action_steps: int | None = None,
) -> dict[str, Any]:
    ckpt_dir = resolve_checkpoint(checkpoint)
    policy_cfg = _load_json(ckpt_dir / "config.json")
    dataset = dataset or dataset_from_checkpoint(ckpt_dir)
    if not dataset:
        raise ValueError(
            "The checkpoint's train_config.json names no dataset; pass --dataset <root or repo id>."
        )
    info, profile = resolve_dataset_info(dataset)

    if profile is not None:
        robot = profile.get("robot", {})
        semantics = semantics or robot.get("semantics") or rprofile.default_semantics(
            robot.get("model", "a"), robot.get("version", "1.2")
        )
    if not semantics:
        raise ValueError(
            "No semantics: the dataset has no RB-Y1 profile, pass --semantics "
            "(e.g. rby1-a-v12-radians-gripper01-v1)."
        )

    features = info["features"]
    action = features["action"]
    expected_action = policy_cfg["output_features"]["action"]["shape"]
    if list(action["shape"]) != list(expected_action):
        raise ValueError(
            f"Dataset action shape {action['shape']} differs from the checkpoint's {expected_action}."
        )
    obs_entries = []
    for name, spec in policy_cfg["input_features"].items():
        if name not in features:
            raise ValueError(f"Checkpoint input {name} is not a dataset feature.")
        entry = _feature_entry(name, features[name], semantics)
        ds_shape = list(features[name]["shape"])
        ck_shape = list(spec["shape"])
        # Checkpoints store images CHW; the dataset / server use HWC.
        if entry["kind"] == "rgb" and ck_shape != ds_shape and ck_shape == [ds_shape[2], *ds_shape[:2]]:
            pass
        elif ck_shape != ds_shape:
            raise ValueError(f"{name}: dataset shape {ds_shape} differs from the checkpoint's {ck_shape}.")
        obs_entries.append(entry)
    action_entry = _feature_entry("action", action, semantics)
    if profile is not None:
        cfg_flags = {k: profile["robot_config"].get(k) for k in rprofile.FEATURE_FIELDS}
        logger.info(f"Dataset profile feature flags: {cfg_flags}")

    return {
        "deployment": deployment,
        "model": {"repo_or_path": str(ckpt_dir if Path(checkpoint).expanduser().is_dir() else checkpoint),
                  "revision": None, "device": device},
        "execution": {
            "supported_modes": ["chunk"],
            "action_fps": float(fps or info["fps"]),
            **({"n_action_steps": n_action_steps} if n_action_steps else {}),
        },
        "zenoh": {"mode": "peer", "listen_endpoints": [listen_endpoint]},
        "semantics": semantics,
        "robot_type": info.get("robot_type") or "rby1",
        "features": obs_entries,
        "action_feature": action_entry,
    }


def write_server_config(config: dict[str, Any], out: str | Path) -> Path:
    import yaml

    out = Path(out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Generated by `rby1-profile server-config` for lerobot-policy-server.\n"
        "# Feature names/shapes come from the dataset meta/info.json; the semantics string\n"
        "# must be passed unchanged to lerobot-rollout --inference.semantics.\n"
    )
    out.write_text(header + yaml.safe_dump(config, sort_keys=False))
    return out
