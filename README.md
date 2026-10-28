# RB-Y1

[Rainbow Robotics RB-Y1](https://rainbow-robotics.com/en/products/rb-y1/) is a bimanual robot designed for physical AI research. This repository provides a LeRobot plugin for controlling RB-Y1 with a leader arm teleoperation setup, enabling intuitive data collection and experimentation.

## Overview
<img width="1599" height="905" alt="Image" src="https://github.com/user-attachments/assets/bbea002e-b8b5-4a8b-8aae-a5fb38f830f0" />

## Platform Requirements
- **RB-Y1**: Currently, **only robots version 1.2 or lower are supported.** Support for version 1.3 is coming soon.
- **Linux**: Tested on Ubuntu 22.04 (x86-64 and ARM64).
- **Python 3.12+** and **LeRobot 0.6.1** (see [Compatibility](#compatibility)).
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
pip install 'lerobot[core_scripts]==0.6.1'
#    or, inside a LeRobot source checkout (tag v0.6.1):
#    pip install -e ".[core_scripts]"
#    Add the `training` extra (e.g. 'lerobot[core_scripts,training]') to train
#    policies with lerobot-train on the same machine.

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
pip install 'lerobot[intelrealsense]==0.6.1'
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
| `model` | `"auto"` | Model variant: `"m"`, `"a"`, `"ub"`, or `"auto"` to detect |
| `version` | `"auto"` | Robot version: `"1.0"`–`"1.3"`, or `"auto"` to detect |
| `use_right_arm` | `True` | Include right arm in observation / action |
| `use_left_arm` | `True` | Include left arm in observation / action |
| `use_torso` | `False` | Include torso joints |
| `use_gripper` | `True` | Enable RB-Y1 grippers |
| `use_mobile_base` | `False` | Enable base control (Model M/A) |
| `use_velocity` | `False` | Add `.vel` channels to observations |
| `use_torque` | `False` | Add `.torque` channels to observations |
| `cameras` | `{}` | Dict of camera name → `CameraConfig` |
| `use_impedance` | `False` | Joint-impedance vs. joint-position control |

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

- Requires **lerobot 0.6.1 or newer**. `lerobot-rollout` does not exist before 0.6.0,
  and 0.6.0 keeps only `.pos` features, dropping the mobile-base (`x.vel`, …) and
  `use_velocity` channels.
- `action_mode="ee"` is **not supported**. Rollout keeps only `.pos` / `.vel` state and
  action features, and end-effector keys (`right_ee.x`, …) match neither — this is an
  upstream limitation that applies to LeRobot's own EE robots too.
- `use_torque=True` datasets do not round-trip: rollout drops the `.torque` channels, so
  the state dimension will not match a policy trained on such a dataset. `connect()` logs
  a warning when this is set.

## Compatibility

| Plugin version | LeRobot | Python | Notes |
|----------------|---------|--------|-------|
| 0.2.x | 0.6.1 | 3.12, 3.13 | Tested in CI. `lerobot-async-rby1` requires `<0.6.2`. |
| 0.2.x | `main` (next release) | 3.12 | Robot/teleop tested weekly in CI. `lerobot-async-rby1` is not supported (upstream removed the AsyncInference gRPC service). |
| 0.1.x | 0.4 – 0.6.0 | ≥ 3.10 | Bare joint keys (no `.pos` suffix), unmaintained. |

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
| [`lerobot-async-rby1/`](lerobot-async-rby1) | Async inference client/server (`rby1-policy-server`, `rby1-robot-client`, `rby1-robot-replay`) with gRPC, GR00T-ZMQ and Pi0.5-ZMQ backends |
| [`tests/`](tests) | Hardware-free tests using a fake `rby1_sdk` |

## Resources

- [Rainbow Robotics Website](https://rainbow-robotics.com/en/)
- [RB-Y1 Product Page](https://rainbow-robotics.com/en/products/rb-y1/)
- [RB-Y1 Documentation](https://rainbowrobotics.github.io/rby1-dev/)
- [LeRobot Documentation](https://huggingface.co/docs/lerobot)
- [LeRobot Installation Guide](https://huggingface.co/docs/lerobot/installation)

