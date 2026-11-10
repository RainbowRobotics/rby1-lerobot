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

Pass any field as `--robot.<field>=<value>`. The fields are grouped by the RB-Y1
controller they feed; a group that is not active is ignored. All fields and their
comments are in [`config_rby1.py`](lerobot-robot-rby1/lerobot_robot_rby1/config_rby1.py).

**Connection**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `address` | `"192.168.30.1:50051"` | gRPC address of the robot |
| `model` | `"auto"` | Model variant: `"m"`, `"a"`, or `"auto"` to detect (`"ub"` is not supported) |
| `version` | `"auto"` | Robot version: `"1.0"`–`"1.3"`, or `"auto"` to detect |
| `connect_timeout_sec` | `3.0` | Timeout of the model / version detection |
| `move_to_ready_on_connect` | `True` | Move to the ready pose on `connect()` |

**Features** — what the observation and the action contain

| Parameter | Default | Description |
|-----------|---------|-------------|
| `action_mode` | `"joint"` | `"joint"`: joint positions. `"ee"`: end-effector poses (e.g. from `rby1_vr`) |
| `use_right_arm` | `True` | Include the right arm in observation / action |
| `use_left_arm` | `True` | Include the left arm in observation / action |
| `use_torso` | `False` | Include the torso in observation / action (commanded in both action modes) |
| `use_gripper` | `True` | Enable RB-Y1 grippers |
| `use_mobile_base` | `False` | Enable base control (Model M/A) |
| `use_velocity` | `False` | Add `.vel` channels to observations |
| `use_torque` | `False` | Add `.torque` channels to observations |
| `cameras` | `{}` | Dict of camera name → `CameraConfig` |

**Joint mode** (`action_mode="joint"`) — JointPosition / JointImpedance commands

| Parameter | Default | Description |
|-----------|---------|-------------|
| `joint_min_time` | `0.07` | `minimum_time` (s) of each command. Also used for the base velocity command |
| `joint_startup_min_time` | `5.0` | `minimum_time` (s) of the first command; ramps down to `joint_min_time` |
| `joint_startup_ramp_duration` | `5.0` | Length (s) of that ramp; `0` disables it |
| `joint_acceleration_limit_scale` | `30.0` | Multiplies the URDF acceleration limits |
| `joint_position_cutoff_frequency` | `5.0` | Low-pass cutoff (Hz) of JointPosition commands, set on the robot for the session |
| `use_joint_impedance` | `False` | Send JointImpedance instead of JointPosition commands |
| `joint_impedance_arm_stiffness` | `[150]*7` | Arm stiffness (Nm/rad), used for both arms |
| `joint_impedance_arm_torque_limit` | `[40]*7` | Arm torque limit (Nm) |
| `joint_impedance_torso_stiffness` | `[400]*6` | Torso stiffness (Nm/rad), with `use_torso` |
| `joint_impedance_torso_torque_limit` | `[500]*6` | Torso torque limit (Nm) |
| `joint_impedance_damping_ratio` | `1.0` | Damping ratio ζ (1.0 = critical, < 1 overshoots) |
| `joint_impedance_wrist_velocity_limit_scale` | `10.0` | Multiplies the velocity limit of the last joint of each arm |

**EE mode** (`action_mode="ee"`) — CartesianImpedance commands, one solver each for the torso, right arm and left arm

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ee_auto_action_period` | `True` | Measure the interval between actions and use it as the action period |
| `ee_action_period` | `1/15` | Action period (s): the starting value, or the fixed value with `ee_auto_action_period=False` |
| `ee_min_time_factor` | `1.02` | `minimum_time` = action period × this factor |
| `ee_hold_time` | `1.0` | The command keeps holding this long (s) if the stream stalls |
| `ee_reset_position_threshold` / `ee_reset_rotation_threshold` | `0.1` m / `0.5` rad | A target jump beyond this re-references the solver to the robot's state |
| `ee_arm_stiffness` / `ee_arm_torque_limit` / `ee_arm_damping_ratio` | `[80,80,80,60,60,60,60]` / `[40,40,40,30,30,30,30]` / `1.0` | Arm solver joint gains, used for both arms |
| `ee_arm_target_gains` | `(3.0, 2π, 150, 80π)` | Arm target limits: linear / angular velocity, linear / angular acceleration |
| `ee_right_arm_joint_limits` / `ee_left_arm_joint_limits` | elbow `(-2.6, -0.5)` | Extra joint limits (rad) given to the arm solvers |
| `ee_null_right_arm_deg` / `ee_null_left_arm_deg` | — | Nullspace joint target of each arm (deg) |
| `ee_nullspace_weight` / `ee_nullspace_stiffness` / `ee_nullspace_damping_ratio` | `[2,2,2,2,2,2,1]` / `0.2` / `1.0` | Nullspace gains |
| `ee_torso_stiffness` / `ee_torso_torque_limit` / `ee_torso_damping_ratio` | `[1700]*6` / `[600]*6` / `1.0` | Torso solver joint gains |
| `ee_torso_target_gains` | `(1.0, π/2, 20, 40π)` | Torso target limits (same order as the arm) |
| `ee_torso_joint_limits` | — | Extra joint limits (rad) given to the torso solver |

**Mobile base** (`use_mobile_base=True`) — SE2Velocity commands

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mobile_base_hold_time` | `0.5` | The base keeps the last velocity this long (s) after the stream stops |
| `mobile_base_max_linear_velocity` | `0.6` | Clamp of the x / y speed (m/s) |
| `mobile_base_max_angular_velocity` | `1.4` | Clamp of the yaw rate (rad/s) |

**Robot profile** — `profile_out`, `profile_in`, `profile_mode`, `profile_force`:
see [Keeping robots consistent](#keeping-robots-consistent-robot-profile).

### Teleoperator (`Rby1LeaderArmConfig`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `version` | `"1.2"` | Robot version of the paired follower (sets init pose) |
| `use_right_arm` | `True` | Read right arm joints |
| `use_left_arm` | `True` | Read left arm joints |
| `use_torso` | `False` | Output a fixed torso target (set it together with `--robot.use_torso=true`) |
| `torso_target_deg` | ready pose (`[0, 0, 0, 20, 0, 0]`) | The torso target (deg, `torso_0..5`) |
| `use_gripper` | `True` | Read gripper state from trigger buttons |
| `control_frequency` | `100.0` | Leader arm control loop frequency (Hz) |
| `init_duration` | `5.0` | Time (s) to reach ready pose on `connect()` |

The leader arm has no torso joints. With `use_torso=True` on both the robot and the
leader arm, every action carries `torso_target_deg`, so the robot holds its torso there
and the dataset records the torso action a policy needs. If the torso is somewhere else
when teleoperation starts, the robot moves it to the target during the startup ramp
(`joint_startup_min_time`, 5 s by default).

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

- Requires **lerobot 0.7.0 or newer** (plugin 0.3.x and later). Plugin 0.2.x targets lerobot 0.6.1.
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
| 0.4.x | 0.7.0 | 3.12, 3.13 | Tested in CI. Config fields grouped by controller, torso command in joint mode, measured EE action period. |
| 0.4.x | `main` (next release) | 3.12 | Robot/teleop tested weekly in CI. |
| 0.3.x | 0.7.0 | 3.12, 3.13 | Remote inference via lerobot's `lerobot-policy-server` / `--inference.type=remote`. |
| 0.2.x | 0.6.1 | 3.12, 3.13 | Legacy: last version with the gRPC `lerobot-async-rby1` package. |
| 0.1.x | 0.4 – 0.6.0 | ≥ 3.10 | Bare joint keys (no `.pos` suffix), unmaintained. |

### Versioning and releases

The plugins use their own [semantic version](https://semver.org/) (both packages
share one version), and every [GitHub Release](https://github.com/RainbowRobotics/rby1-lerobot/releases)
names the LeRobot release it was tested against, e.g. `v0.2.0 — LeRobot 0.6.1`.

- **Minor** (`0.2.0 → 0.3.0`): support for a new LeRobot release, or a breaking change in
  the plugins (config fields, observation/action keys).
- **Patch** (`0.2.0 → 0.2.1`): fixes that keep the same LeRobot requirement and keys.

To install the plugins for a given release, check out its tag:

```bash
git clone --branch v0.2.0 https://github.com/rainbowrobotics/rby1-lerobot.git
```

Releasing (maintainers): bump `version` in the two `pyproject.toml` files (and the
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

Breaking changes in 0.4.0:

- **`Rby1Config` fields renamed** so the name says which controller a field belongs to.
  Update `--robot.<field>` arguments and config files:

  | 0.3.x | 0.4.0 |
  |-------|-------|
  | `startup_ramp_duration`, `startup_min_time`, `normal_min_time` | `joint_startup_ramp_duration`, `joint_startup_min_time`, `joint_min_time` |
  | `acceleration_limit_scale` | `joint_acceleration_limit_scale` |
  | `use_impedance`, `impedance_damping_ratio` | `use_joint_impedance`, `joint_impedance_damping_ratio` |
  | `impedance_stiffness` (20 values: torso, right, left) | `joint_impedance_torso_stiffness` (6) + `joint_impedance_arm_stiffness` (7, both arms) |
  | `impedance_torque_limit` (20 values) | `joint_impedance_torso_torque_limit` (6) + `joint_impedance_arm_torque_limit` (7) |
  | `wrist_velocity_limit_scale` | `joint_impedance_wrist_velocity_limit_scale` |
  | `ee_dt`, `min_time_factor` | `ee_action_period`, `ee_min_time_factor` |
  | `torso_stiffness`, `torso_torque_limit`, `torso_damping_ratio`, `torso_joint_limits`, `torso_target_gains` | the same names with an `ee_` prefix |
  | `right_arm_stiffness` + `left_arm_stiffness` (and `_torque_limit`, `_damping_ratio`) | `ee_arm_stiffness`, `ee_arm_torque_limit`, `ee_arm_damping_ratio` (one value for both arms) |
  | `right_arm_joint_limits`, `left_arm_joint_limits`, `arm_target_gains` | `ee_right_arm_joint_limits`, `ee_left_arm_joint_limits`, `ee_arm_target_gains` |
  | `null_right_arm_deg`, `null_left_arm_deg`, `nullspace_*` | the same names with an `ee_` prefix |

- **Joint mode commands the torso** when `use_torso=True`: the action starts with
  `torso_0..5.pos`. In 0.3.x the torso was only observed, so datasets and policies recorded
  in joint mode with `use_torso=True` have a different action layout; use plugin 0.3.x
  with them. The leader arm now accepts `use_torso=True` and outputs a fixed torso target.
- **EE mode measures the action period** instead of assuming 15 Hz
  (`ee_auto_action_period=False` restores the fixed period).
- Robot profiles written by 0.3.x are still read: field names are converted on load, and
  `profile_mode=apply` restores the fixed EE period they were recorded with.

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

