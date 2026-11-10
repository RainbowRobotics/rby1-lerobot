# RB-Y1

[Rainbow Robotics RB-Y1](https://rainbow-robotics.com/en/products/rb-y1/) is a bimanual robot designed for physical AI research. This repository provides a LeRobot plugin for controlling RB-Y1 with a leader arm teleoperation setup, enabling intuitive data collection and experimentation.

## Overview
<img width="1599" height="905" alt="Image" src="https://github.com/user-attachments/assets/bbea002e-b8b5-4a8b-8aae-a5fb38f830f0" />

## Platform Requirements
- **RB-Y1**: Currently, **only robots version 1.2 or lower are supported.** Support for version 1.3 is coming soon.
- **Linux**: Tested on Ubuntu 22.04 (x86-64 and ARM64).
- **Python 3.12+** and **LeRobot 0.7.0** (see [Compatibility](#compatibility)).
- **Ethernet connection to RPC**: Ethernet connection to the RPC at `192.168.30.1` (or desired IP).
- **Teleoperation device**: Leader arm connection via 4-Pin MOCO cable to the RB-Y1 UPC(VR-based teleoperation is not currently supported; this feature is coming soon).

## ⚠️Safety Guide

Before operating RB-Y1, please read the official safety documentation provided
by Rainbow Robotics. Key points:

- **Clear workspace**: Keep the robot's full range of motion free of people and
  obstacles before powering on.
- **Secure the robot**: Secure the base to a stable surface before
  operation.
- **Payload limits**: Do not exceed specified payload(3kg) limits for the arms.
- **Emergency stop**: Know the location and operation of the hardware
  emergency-stop switch.
- **Leader arm posture**: Place the leader arm in a safe posture before running
  `connect()` — the arm will execute a smooth trajectory to the ready pose.

## Hardware Setup

### RB-Y1(Follower Robot)

1. Power on RPC, UPC and verify it boots normally.
2. Verify connectivity at UPC:
   ```bash
   ping 192.168.30.1
   ```

### Leader Arm

1. Connect the leader arm to the RB-Y1(UPC) via 4-Pin MOCO cable.
2. Verify connectivity at UPC:
   ```bash
   ls /dev/rb*
   # should show `/dev/rby1_leader_arm` or `/dev/rby1_master_arm` for the leader arm
   # This command verifies that the RS485 communication module is connected, but it does not confirm that the actual reader arm is connected.
   ```

## Install LeRobot 🤗

Follow the [LeRobot Installation Guide](https://huggingface.co/docs/lerobot/installation)
(Python 3.12 environment), then install the RB-Y1 SDK and plugins.

After completing the LeRobot installation, activate the `lerobot` conda environment:
```bash
conda activate lerobot
```

```bash
# 0. Install the LeRobot CLI dependencies
#    From PyPI:
pip install 'lerobot[core_scripts]==0.7.0'
#    or, inside a LeRobot source checkout (tag v0.7.0):
#    pip install -e ".[core_scripts]"
#    Add the `training` extra (e.g. 'lerobot[core_scripts,training]') to train
#    policies with lerobot-train on the same machine.
#    LeRobot 0.7.0 installs CUDA 13 torch wheels on Linux (NVIDIA driver >= 580.65).
#    On a machine without a GPU (e.g. the UPC) install CPU torch first, or with uv:
#    UV_TORCH_BACKEND=cpu uv pip install 'lerobot[core_scripts]==0.7.0'

# 1. Create a working directory and move into it
#    (to avoid accidentally cloning inside the LeRobot package folder)
mkdir -p ~/rby1-lerobot && cd ~

# 2. Install the RB-Y1 SDK
pip install rby1-sdk

# 3. Clone this repository
git clone https://github.com/rainbowrobotics/rby1-lerobot.git
cd rby1-lerobot

# 4. Install the RB-Y1 robot, teleoperator plugins and dependencies
pip install -e lerobot-robot-rby1
pip install -e lerobot-teleoperator-rby1

# 5. (only for Intel RealSense cameras) Install pyrealsense2
#    x86-64:
pip install 'lerobot[intelrealsense]==0.7.0'
#    RB-Y1's UPC (ARM64): the official pyrealsense2 package on PyPI does not
#    support ARM64, so a pre-built wheel (ARM64, Python 3.12) is included here.
pip install pyrealsense2-2.56.5-cp312-cp312-linux_aarch64.whl
```

The leader arm needs the SDK's leader-arm URDF. The `rby1-sdk` PyPI wheel does not
ship it; pass `--teleop.leader_arm_model_path=<rby1-sdk checkout>/models/leader_arm/model.urdf`
(or set `RBY1_LEADER_ARM_MODEL`) if `connect()` reports that it cannot find the model.

## Cameras (Optional)


> **Note**: Camera bracket accessories are available for purchase. Example photos of camera mounting configurations are below.
<img width="360" height="480" alt="Image" src="https://github.com/user-attachments/assets/e5fb529b-80c6-45f6-b950-dca57087bf3c" />
<img width="360" height="480" alt="Image" src="https://github.com/user-attachments/assets/b92c4f1f-e2ee-4ce9-b9b6-e77e49c88a99" />

Mount the Intel RealSense cameras and record their serial numbers for the configuration step.
You can verify the serial numbers using the following command:
```bash
lerobot-find-cameras realsense # or opencv
```
Alternatively, you can mount and use any camera of your choice. Please refer to the link below for instructions on how to connect cameras, including OpenCV:

[LeRobot Camera Guide](https://huggingface.co/docs/lerobot/cameras)


## Teleoperate

### Without Camera

```bash
lerobot-teleoperate \
  --robot.type=rby1 \
  --robot.address=192.168.30.1:50051 \
  --teleop.type=rby1_leader_arm
```

The follower robot mirrors the leader arm's joint positions in real time. No
calibration is required — absolute encoders ensure the arms are in sync as soon
as both units reach the ready pose.

### With Camera

Add one or more cameras by passing a `cameras` configuration:

```bash
lerobot-teleoperate \
  --robot.type=rby1 \
  --robot.address=192.168.30.1:50051 \
  --robot.cameras='{"front": {"type": "intelrealsense", "serial_number_or_name": "XXXXXXXXX", "fps": 30, "width": 640, "height": 480}}' \
  --teleop.type=rby1_leader_arm \
  --display_data=true
```

Replace `XXXXXXXXX` with your RealSense serial number. You can add `right` and
`left` cameras using the same pattern.

## Record Data and upload to HF

```bash
lerobot-record \
  --robot.type=rby1 \
  --robot.address=192.168.30.1:50051 \
  --teleop.type=rby1_leader_arm \
  --dataset.repo_id=<hf_username>/<dataset_name> \
  --dataset.single_task="<task description>" \
  --dataset.num_episodes=50 \
  --dataset.fps=30
```

## Record Data without uploading

```bash
lerobot-record \
  --robot.type=rby1 \
  --robot.address=192.168.30.1:50051 \
  --teleop.type=rby1_leader_arm \
  --dataset.repo_id=test_dataset \
  --dataset.root=./datasets \
  --dataset.push_to_hub=false \
  --dataset.single_task="<task description>" \
  --dataset.num_episodes=50 \
  --dataset.fps=30

```


## Configuration Options

### Robot (`Rby1Config`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `address` | `"192.168.30.1:50051"` | gRPC address of the robot |
| `model` | `"auto"` | Model variant: `"m"`, `"a"`, or `"auto"` to detect (`"ub"` is not supported) |
| `version` | `"auto"` | Robot version: `"1.0"`–`"1.3"`, or `"auto"` to detect |
| `use_right_arm` | `True` | Include right arm in observation / action |
| `use_left_arm` | `True` | Include left arm in observation / action |
| `use_torso` | `False` | Include torso joints |
| `use_gripper` | `True` | Enable RB-Y1 grippers |
| `use_mobile_base` | `False` | Enable base control (Model M/A) |
| `use_velocity` | `False` | Add `.vel` channels to observations |
| `use_torque` | `False` | Add `.torque` channels to observations |
| `cameras` | `{}` | Dict of camera name → `CameraConfig` |
| `use_joint_impedance` | `False` | Joint-impedance vs. joint-position control |
| `joint_impedance_damping_ratio` | `1.0` | Joint-impedance damping ratio ζ (1.0 = critical, < 1 overshoots) |

### Teleoperator (`Rby1LeaderArmConfig`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `version` | `"1.2"` | Robot version of the paired follower (sets init pose) |
| `use_right_arm` | `True` | Read right arm joints |
| `use_left_arm` | `True` | Read left arm joints |
| `use_gripper` | `True` | Read gripper state from trigger buttons |
| `control_frequency` | `100.0` | Leader arm control loop frequency (Hz) |
| `init_duration` | `5.0` | Time (s) to reach ready pose on `connect()` |

### Observation / Action Keys

Keys follow the LeRobot naming convention:

| Kind | Key | Example |
|------|-----|---------|
| Joint position | `<joint>.pos` | `right_arm_0.pos`, `torso_0.pos` |
| Gripper position | `<gripper>.pos` | `right_gripper_0.pos` (1.0 = open) |
| Mobile base velocity | `x.vel`, `y.vel`, `theta.vel` | action only |
| Joint velocity (`use_velocity`) | `<joint>.vel` | `right_arm_0.vel` |
| Joint torque (`use_torque`) | `<joint>.torque` | `right_arm_0.torque` |
| End-effector pose (`action_mode="ee"`) | `<group>_ee.{x,y,z,wx,wy,wz}` | `right_ee.x` |

> [!WARNING]
> Datasets recorded before the `.pos` suffix was introduced use bare keys
> (`right_arm_0`). Feature names must match to append episodes, so **start a new
> dataset** rather than continuing an old one. Inference with a checkpoint trained
> on an older dataset still works — the key order is unchanged.

## Rollout

Deploy a trained policy on the robot with `lerobot-rollout`:

```bash
lerobot-rollout \
  --robot.type=rby1 \
  --robot.address=192.168.30.1:50051 \
  --policy.path=<hf_username>/<policy_name>
```

Limitations:

- Requires **lerobot 0.7.0 or newer** (plugin 0.3.x). Plugin 0.2.x targets lerobot 0.6.1.
- `action_mode="ee"` is **not supported**. Rollout keeps only `.pos` / `.vel` state and
  action features, and end-effector keys (`right_ee.x`, …) match neither — this is an
  upstream limitation that applies to LeRobot's own EE robots too.
- `use_torque=True` datasets do not round-trip: rollout drops the `.torque` channels, so
  the state dimension will not match a policy trained on such a dataset. `connect()` logs
  a warning when this is set.

## Remote inference

Run the policy on a GPU machine and the robot loop on the UPC with lerobot's
remote inference (lerobot >= 0.7.0; it replaces the gRPC `lerobot-async-rby1`
package of plugin 0.2.x). The server loads a LeRobot checkpoint and admits one
client whose feature names, shapes and `semantics` string match the deployment
exactly, so generate the server config from the training dataset instead of
writing it by hand:

```bash
# 1. Inference machine (GPU)
pip install 'lerobot[core_scripts,remote]==0.7.0'
pip install -e lerobot-robot-rby1            # registers robot_type "rby1"
rby1-profile server-config --checkpoint <hf_user>/<policy> --out server.yaml
#   --dataset <root or hf_user/dataset> if the checkpoint's train_config.json does not name it
#   --semantics ... if the dataset has no RB-Y1 profile (default: the profile's semantics)
lerobot-policy-server --config_path=server.yaml      # listens on tcp/0.0.0.0:7447

# 2. Robot machine (UPC)
pip install 'lerobot[core_scripts,remote]==0.7.0'    # plus the plugins as above
lerobot-rollout \
  --robot.type=rby1 --robot.address=192.168.30.1:50051 \
  --robot.cameras='{...}' --robot.profile_in=<hf_user>/<dataset> \
  --inference.type=remote --inference.endpoint=tcp/<inference-host>:7447 \
  --inference.deployment=rby1 --inference.semantics=<semantics from server.yaml> \
  --inference.mode=chunk --fps=30 --task="<task description>"
```

`rby1-profile server-config` prints the matching `lerobot-rollout` flags. The
semantics string (e.g. `rby1-a-v12-radians-gripper01-v1`) is recorded in the
dataset's robot profile, written into `server.yaml` and checked at admission, so
a checkpoint trained on one robot configuration cannot be served to another.

Notes:

- RB-Y1 implements lerobot's command hold (`hold()`): when no fresh action is
  available (network stall, server fault) the arms keep their last target and the
  mobile base is stopped. This works in joint, EE and mobile-base configurations,
  which the generic position hold would reject.
- The 0.6.x async client's `aggregate_fn` / `chunk_size_threshold` have no
  counterpart; tune `--inference.chunk_merge={auto,append,aligned}`,
  `--inference.blend_steps/--inference.blend_weight` and `--inference.refill_seconds`
  instead. `--inference.encoding=jpeg` reduces camera bandwidth over Wi-Fi.
- Only LeRobot-native policies can be served. The old GR00T / Pi0.5 ZMQ backends
  (which spoke to the NVIDIA GR00T and openpi servers directly) were removed with
  the async package; `git show v0.2.0:lerobot-async-rby1/` has them.

## Keeping robots consistent (robot profile)

A policy behaves the same only if the robot that runs it is configured like
the robot that recorded its dataset. The plugin can write a **robot profile**
(`rby1_profile.json`) into the dataset and compare it with the robot at
inference time. The profile contains:

- the robot: model, firmware version, core / SDK versions, URDF hash and joint
  limits, joint-driver PID gains
- every core parameter (`get_parameter_list`), e.g. command cutoff frequencies,
  velocity / acceleration / torque limit scaling, canceling times, friction
  compensation
- the full `Rby1Config` and the plugin's hard-coded constants (ready pose, joint
  clip limits)
- the teleoperator config (recorded only)

The file lives in the dataset's `meta/` folder, which `LeRobotDataset` uploads
and downloads with the rest of the metadata, so it travels with the dataset.

```bash
# Record: write the profile into the dataset
lerobot-record ... --dataset.root=<root> \
  --robot.profile_out=<root>/meta/rby1_profile.json \
  --teleop.profile_out=<root>/meta/rby1_profile.json

# Inference: compare the robot with the dataset's profile (a dataset root,
# a Hugging Face dataset repo id, or a profile file)
lerobot-rollout ... --robot.profile_in=<hf_user>/<dataset> --robot.profile_mode=check
```

`profile_mode`:

| Mode | Behaviour |
|------|-----------|
| `check` (default) | Logs every difference. Fails before the robot moves if the observation/action layout (`use_*`, `action_mode`, camera names / size / rotation) or the firmware version differ. Model A vs M, gains, limits, core parameters, URDF and PID gains only warn. |
| `apply` | As `check`, then adopts the profile's behaviour settings (gains, limits, timing) and core parameters. Core parameters are set for this session only (`write_db=False`) and restored on disconnect; URDF and PID gains are never changed. |
| `off` | No comparison. |

One dataset holds one profile: recording into a dataset whose profile differs
fails unless `--robot.profile_force=true`, which keeps the original and writes a
time-stamped extra profile. Without `profile_out` the profile is written to
`~/.cache/rby1_lerobot/profiles/` (or `$RBY1_PROFILE_DIR`).

The `rby1-profile` command works without recording:

```bash
rby1-profile export --out robot.json --robot.address=192.168.30.1:50051   # read-only, no motion
rby1-profile attach robot.json --dataset <root or hf_user/dataset>
rby1-profile diff <hf_user>/<dataset> robot.json   # exit code 1 on layout / firmware mismatch
```

The joint position command cutoff (`joint_position_cutoff_frequency`, default
5 Hz) is now also applied for the session only and restored on disconnect;
earlier versions wrote 5 Hz to the robot's stored parameters.

## Compatibility

| Plugin version | LeRobot | Python | Notes |
|----------------|---------|--------|-------|
| 0.3.x | 0.7.0 | 3.12, 3.13 | Tested in CI. Remote inference via lerobot's `lerobot-policy-server` / `--inference.type=remote`. |
| 0.3.x | `main` (next release) | 3.12 | Robot/teleop tested weekly in CI. |
| 0.2.x | 0.6.1 | 3.12, 3.13 | Legacy: last version with the gRPC `lerobot-async-rby1` package. |
| 0.1.x | 0.4 – 0.6.0 | ≥ 3.10 | Bare joint keys (no `.pos` suffix), unmaintained. |

### Versioning and releases

The plugins use their own [semantic version](https://semver.org/) (all three packages
share one version), and every [GitHub Release](https://github.com/RainbowRobotics/rby1-lerobot/releases)
names the LeRobot release it was tested against, e.g. `v0.2.0 — LeRobot 0.6.1`.

- **Minor** (`0.2.0 → 0.3.0`): support for a new LeRobot release, or a breaking change in
  the plugins (config fields, observation/action keys).
- **Patch** (`0.2.0 → 0.2.1`): fixes that keep the same LeRobot requirement and keys.

To install the plugins for a given release, check out its tag:

```bash
git clone --branch v0.2.0 https://github.com/rainbowrobotics/rby1-lerobot.git
```

Releasing (maintainers): bump `version` in the three `pyproject.toml` files (and the
teleoperator's `lerobot_robot_rby1>=` requirement), update the table above, merge to
`main`, then push a `vX.Y.Z` tag on `main`. The release workflow checks that the tag
matches the package versions, runs CI and publishes the GitHub Release.

Breaking changes in 0.2.0 (besides the `.pos` keys and the lerobot/Python pins):
the `reset_*_arm_on_record` options and the EE whole-body solver (`ee_whole_body`,
`wb_joint_*`, `*_target_gains_wb`, `min_time_factor_wb`) were removed, and
`min_time_factor_pc` / `torso_target_gains_pc` / `arm_target_gains_pc` were renamed to
`min_time_factor` / `torso_target_gains` / `arm_target_gains`.

Breaking change in 0.3.0: the `lerobot-async-rby1` package (`rby1-policy-server`,
`rby1-robot-client`, `rby1-robot-replay`, gRPC + GR00T/Pi0.5 ZMQ backends) was removed;
lerobot 0.7.0 dropped the gRPC AsyncInference service it was built on. Use lerobot's remote
inference instead (see [Remote inference](#remote-inference)). The old package is in the git
history (`git show v0.2.0:lerobot-async-rby1/`).

LeRobot commands that do not apply to RB-Y1:

- `lerobot-find-joint-limits` and `lerobot-setup-motors` target LeRobot motor-bus arms
  and do not support RB-Y1; joint limits come from the RB-Y1 URDF.
- `lerobot-calibrate` is a no-op (absolute encoders), but it still connects and moves
  the robot to the ready pose.

## Repository Layout

| Path | Description |
|------|-------------|
| [`lerobot-robot-rby1/`](lerobot-robot-rby1) | RB-Y1 follower robot plugin (`--robot.type=rby1`) |
| [`lerobot-teleoperator-rby1/`](lerobot-teleoperator-rby1) | RB-Y1 teleoperator plugins (`--teleop.type=rby1_leader_arm`, `--teleop.type=rby1_vr`) |
| [`tests/`](tests) | Hardware-free tests using a fake `rby1_sdk` |

## Resources

- [Rainbow Robotics Website](https://rainbow-robotics.com/en/)
- [RB-Y1 Product Page](https://rainbow-robotics.com/en/products/rb-y1/)
- [RB-Y1 Documentation](https://rainbowrobotics.github.io/rby1-dev/)
- [LeRobot Documentation](https://huggingface.co/docs/lerobot)
- [LeRobot Installation Guide](https://huggingface.co/docs/lerobot/installation)

