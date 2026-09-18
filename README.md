# RB-Y1

[Rainbow Robotics RB-Y1](https://rainbow-robotics.com/en/products/rb-y1/) is a bimanual robot designed for physical AI research. This repository provides a LeRobot plugin for controlling RB-Y1 with a leader arm teleoperation setup, enabling intuitive data collection and experimentation.

## Overview
<img width="1599" height="905" alt="Image" src="https://github.com/user-attachments/assets/bbea002e-b8b5-4a8b-8aae-a5fb38f830f0" />

## Platform Requirements
- **RB-Y1**: Currently, **only robots version 1.2 or lower are supported.** Support for version 1.3 is coming soon.
- **Linux**: Tested on Ubuntu 22.04 (x86-64 and ARM64).
- **Ethernet connection to RPC**: Ethernet connection to the RPC at `192.168.30.1` (or desired IP).
- **Teleoperation device**: Leader arm connection via 4-Pin MOCO cable to the RB-Y1 UPC, or an XR headset (Meta Quest 3 / Pico) through [NVIDIA Isaac Teleop](https://github.com/NVIDIA/IsaacTeleop) — see [Teleoperate with NVIDIA Isaac Teleop](#teleoperate-with-nvidia-isaac-teleop-xr-headset).

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

Follow the [LeRobot Installation Guide](https://huggingface.co/docs/lerobot/installation),
then install the RB-Y1 SDK and plugins.

After completing the LeRobot installation, activate the `lerobot` conda environment:
```bash
conda activate lerobot
```

```bash
# 0. Install required lerobot packages
pip install -e ".[core_scripts]"

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

# 5. (for RB-Y1's UPC only) Install pyrealsense2
#    The official pyrealsense2 package on PyPI does not support ARM64.
#    A pre-built wheel for the UPC (ARM64, Python 3.12) is included in this repository.
pip install pyrealsense2-2.56.5-cp312-cp312-linux_aarch64.whl
```

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

## Teleoperate with NVIDIA Isaac Teleop (XR headset)

`--teleop.type=rby1_isaac` drives the RB-Y1 from an XR headset through
[NVIDIA Isaac Teleop](https://github.com/NVIDIA/IsaacTeleop): the CloudXR runtime
runs on the host that runs `lerobot-teleoperate` / `lerobot-record`, and the headset
connects with its **browser** (no app to install). Both controllers move the arms
(end-effector space, executed by the robot's onboard Cartesian impedance solver), the
headset orientation drives the head joints, body tracking drives the torso and the
thumbsticks drive the mobile base.

### Requirements

- Host with an NVIDIA GPU: Ubuntu 22.04/24.04, Python ≥ 3.11. x86_64 workstations are
  the documented target; `isaacteleop` also ships aarch64 wheels and selects an
  experimental CloudXR runtime on **Jetson Orin** (the RB-Y1 UPC). On the UPC run the
  preflight first (below).
- Headset: Meta Quest 3 (controllers, head and inside-out body tracking) or Pico 4
  Ultra (+ motion trackers for body tracking). The headset and the host must be on
  the same network.

### Install

```bash
pip install -e "lerobot-teleoperator-rby1[isaac]"   # isaacteleop[cloudxr,retargeters-lite]
python -m isaacteleop.cloudxr --accept-eula          # once: downloads the CloudXR runtime
```

### Jetson Orin (UPC) preflight

The CloudXR runtime on Orin can abort the process when a Python thread is created
after the runtime started; `lerobot-record` creates its keyboard and camera threads
after `teleop.connect()`. The preflight runs each ordering in a subprocess and prints
a verdict (`GO`, `GO-DEFERRED` → add `--teleop.session_start=first_action`, or
`NO-GO` → run on an x86 host):

```bash
python lerobot-teleoperator-rby1/scripts/isaac_preflight_upc.py \
  --robot 192.168.30.1:50051 --camera-serial XXXXXXXXX --record-dryrun
```

### Run

```bash
lerobot-teleoperate \
  --robot.type=rby1 \
  --robot.address=192.168.30.1:50051 \
  --robot.action_mode=ee \
  --robot.use_torso=true \
  --robot.use_head=true \
  --robot.use_mobile_base=true \
  --teleop.type=rby1_isaac \
  --teleop.robot_address=192.168.30.1:50051
```

`lerobot-record` takes the same `--robot.*` / `--teleop.*` arguments plus the
`--dataset.*` ones from the sections below. The teleoperator's `use_torso`,
`use_right_arm`, `use_left_arm`, `use_gripper`, `use_mobile_base` and `use_head`
flags **must mirror the robot's** so the action keys match.

The process prints the host IP addresses and waits for the headset:

1. In the headset browser open `https://nvidia.github.io/IsaacTeleop/client`.
2. Enter the host IP, accept the self-signed certificate at `https://<ip>:48322/`, Connect.
3. Once the controllers are tracked the robot holds its pose until you squeeze a grip.

| Input | Effect |
|-------|--------|
| Squeeze (grip) > `clutch_threshold` | That arm follows its controller (clutch engaged); release to hold |
| Trigger | Gripper (fully pressed = closed) |
| Right thumbstick / left thumbstick | Base linear velocity / yaw rate |
| Right **B** | Stop: freeze every target, zero the base |
| Right **A** | Release every clutch and return arms, torso and head to the start pose (the pose right after the ready-pose motion) over `ready_return_duration_s`; also resumes after a stop and re-centres the head origin. The base is not moved |
| Headset orientation | `head_0` (pan) / `head_1` (tilt) relative to the pose at the first tracked frame |
| Body tracking (`torso_source=body`) or headset (`head`) | Torso pose, while **both** arms are clutched (`torso_engage=both_arms`, default; `any_arm` / `always` available) |

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
| `use_head` | `False` | Include the head pan / tilt joints (`head_0.pos`, `head_1.pos`) in observation and action, in either action mode |
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
| `reset_right_arm_on_record` | `False` | Return right arm to init pose between episodes |
| `reset_left_arm_on_record` | `False` | Return left arm to init pose between episodes |

### Teleoperator (`Rby1XRConfig`, `--teleop.type=rby1_isaac`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `robot_address`, `robot_model` | `"192.168.30.1:50051"`, `"m"` | Read-only robot link (state + forward kinematics for the clutch) |
| `use_torso`, `use_right_arm`, `use_left_arm`, `use_gripper`, `use_mobile_base`, `use_head` | `True` | Must mirror `Rby1Config` |
| `auto_launch_cloudxr` | `True` | Launch the CloudXR runtime from this process (`LEROBOT_CLOUDXR_SKIP_AUTOLAUNCH=1` also disables it) |
| `cloudxr_env_file` | `None` | KEY=value profile for CloudXR; default is the packaged `default.env` (Quest3 profile) |
| `session_start` | `"connect"` | `"first_action"` opens the XR session on the first `get_action()` (Jetson Orin mitigation) |
| `tracking_wait_timeout_s` | `0.0` | Give up waiting for the headset after this many seconds (0 = forever) |
| `clutch_threshold` | `0.5` | Squeeze value above which an arm follows |
| `latch_orientation` | `"measured"` | Home orientation on engage: measured EE pose (`"commanded"` = upstream SO-101 behaviour) |
| `thumbstick_deadzone`, `base_max_linear`, `base_max_angular` | `0.15`, `0.3`, `0.6` | Thumbstick → base velocity mapping |
| `head_yaw_sign`, `head_pitch_sign` | `1.0`, `-1.0` | Flip if the head moves the wrong way |
| `head_yaw_limit_deg`, `head_pitch_min_deg`, `head_pitch_max_deg` | `80`, `-45`, `80` | Head joint clamps |
| `head_smoothing` | `0.3` | EMA weight of the new head sample (1.0 = no filtering) |
| `torso_source` | `"body"` | `"body"` (Isaac Teleop body tracking), `"head"` or `"none"` |
| `torso_engage` | `"both_arms"` | When the torso follows: `both_arms` (both grips squeezed), `any_arm`, or `always` |
| `torso_body_joint` | `"SPINE3"` | Body joint driving the torso (XR_BD 24-joint names) |
| `ready_return_duration_s` | `4.0` | Duration of the Right-A return-to-start motion |
| `resync_position_threshold_m`, `resync_rotation_threshold_deg` | `0.03`, `10` | Targets are re-seeded from the measured pose on the first action and whenever a component that is not clutched drifted past these (record reset, manual move) |
| `status_log_period_s` | `5.0` | Log a one-line tracking / clutch status (controllers, head, body joints valid, why the torso holds); 0 = off |
| `torso_max_rot_delta_deg`, `torso_max_z_delta_m` | `35`, `0.15` | Safety clamps on the torso delta since engage |

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
| Head joints (`use_head`) | `head_0.pos`, `head_1.pos` | pan, tilt (rad) |

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

- Requires **lerobot 0.6.0 or newer** (`lerobot-rollout` does not exist in earlier releases).
- `action_mode="ee"` is **not supported**. Rollout keeps only `.pos` / `.vel` state and
  action features, and end-effector keys (`right_ee.x`, …) match neither — this is an
  upstream limitation that applies to LeRobot's own EE robots too.
- `use_torque=True` datasets do not round-trip: rollout drops the `.torque` channels, so
  the state dimension will not match a policy trained on such a dataset. `connect()` logs
  a warning when this is set.

## Repository Layout

| Path | Description |
|------|-------------|
| [`lerobot-robot-rby1/`](lerobot-robot-rby1) | RB-Y1 follower robot plugin (`--robot.type=rby1`) |
| [`lerobot-teleoperator-rby1/`](lerobot-teleoperator-rby1) | RB-Y1 teleoperator plugins (`--teleop.type=rby1_leader_arm`, `--teleop.type=rby1_vr`, `--teleop.type=rby1_isaac`) |

The LeRobot framework is installed as a Python package dependency (`pip install lerobot[dataset]`).

## Resources

- [Rainbow Robotics Website](https://rainbow-robotics.com/en/)
- [RB-Y1 Product Page](https://rainbow-robotics.com/en/products/rb-y1/)
- [RB-Y1 Documentation](https://rainbowrobotics.github.io/rby1-dev/)
- [LeRobot Documentation](https://huggingface.co/docs/lerobot)
- [LeRobot Installation Guide](https://huggingface.co/docs/lerobot/installation)
- [NVIDIA Isaac Teleop](https://github.com/NVIDIA/IsaacTeleop) · [LeRobot Isaac Teleop example](https://github.com/huggingface/lerobot/tree/main/examples/isaac_teleop_to_so101)

