#!/usr/bin/env python
"""``lerobot-record`` wrapper that splits ``observation.state`` into groups.

Stock ``lerobot-record`` merges every scalar observation into a single
``observation.state`` vector (``hw_to_dataset_features``; it lives in
``lerobot.utils.feature_utils`` on newer LeRobot and
``lerobot.datasets.feature_utils`` on older). For the RB-Y1 mobile-base
experiments we instead want the observation recorded as three separate state
keys so each can be included / excluded independently when training:

    observation.state.odometry        base_x.pos, base_y.pos, base_theta.pos
    observation.state.wheel_velocity  <wheel>.vel per mobility joint
    observation.state.endpoint_state  endpoint_state (0/1)

This script monkey-patches only the observation feature builder and then defers
to the normal ``lerobot-record`` entrypoint, so every other behaviour (teleop,
cameras, keyboard controls, dataset writing) is unchanged. It lets the *original*
builder run first (so camera / depth-map handling stays correct across versions)
and then reshapes only the ``observation.state`` key into the sub-groups. Any
observation names that do not match a group stay in a residual
``observation.state`` vector.

NOTE: this deliberately deviates from the LeRobot convention of a single
``observation.state``; standard policies expecting that key will need their
input-feature config adjusted. That is intentional here — the split keys let us
train on each group separately and compare.

Usage — identical flags to ``lerobot-record``::

    python record_split_state.py \
      --robot.type=rby1 --robot.address=192.168.30.1:50051 \
      --robot.use_mobile_base=true --robot.use_base_pose=true \
      --robot.use_wheel_velocity=true --robot.use_endpoint_state=true \
      --robot.use_right_arm=false --robot.use_left_arm=false \
      --teleop.type=rby1_keyboard \
      --dataset.repo_id=test --dataset.push_to_hub=false \
      --dataset.single_task="drive" --dataset.num_episodes=5 --dataset.fps=30
"""

from __future__ import annotations

import lerobot.datasets.pipeline_features as pipeline_features
from lerobot.utils.constants import OBS_STR

# Grab the original builder straight off the module that actually calls it, so
# this works regardless of which sub-module hw_to_dataset_features lives in
# across LeRobot versions (it moved between 0.5.1 and 0.5.2).
_orig_hw_to_dataset_features = pipeline_features.hw_to_dataset_features

# Odometry observation keys (from Rby1 use_base_pose).
_ODOMETRY_NAMES = ("base_x.pos", "base_y.pos", "base_theta.pos")

# Sub-group order within observation.state.* ("" = residual observation.state).
_GROUP_ORDER = ("odometry", "wheel_velocity", "endpoint_state", "")


def _group_of(name: str) -> str:
    """Return the observation.state sub-group a scalar feature name belongs to."""
    if name in _ODOMETRY_NAMES:
        return "odometry"
    if name == "endpoint_state":
        return "endpoint_state"
    if name.endswith(".vel") and "wheel" in name:
        return "wheel_velocity"
    return ""  # residual -> plain observation.state


def _split_hw_to_dataset_features(hw_features, prefix, use_video=True):
    """hw_to_dataset_features + split of observation.state into sub-keys.

    Delegates to the original builder (keeping action and camera / depth-map
    handling identical), then replaces the single ``observation.state`` feature
    with per-group ``observation.state.{odometry,wheel_velocity,endpoint_state}``
    keys (plus a residual ``observation.state`` for anything unmatched).
    """
    features = _orig_hw_to_dataset_features(hw_features, prefix, use_video)
    if prefix != OBS_STR:
        return features

    state_key = f"{OBS_STR}.state"
    state = features.pop(state_key, None)
    if state is None:  # observation had no scalar state (e.g. images only)
        return features

    grouped: dict[str, list[str]] = {g: [] for g in _GROUP_ORDER}
    for name in state["names"]:
        grouped[_group_of(name)].append(name)

    split: dict[str, dict] = {}
    for group in _GROUP_ORDER:
        names = grouped[group]
        if not names:
            continue
        key = f"{state_key}.{group}" if group else state_key
        split[key] = {"dtype": "float32", "shape": (len(names),), "names": names}

    # Split state keys first, then whatever else the original produced (images).
    return {**split, **features}


# Patch the reference used by aggregate_pipeline_dataset_features at build time.
pipeline_features.hw_to_dataset_features = _split_hw_to_dataset_features


if __name__ == "__main__":
    # Import after patching so the record entrypoint picks up the patched builder.
    from lerobot.scripts.lerobot_record import main

    main()
