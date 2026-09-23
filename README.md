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

Absolute (non-clutch) arm mapping with the elbow following your own:

```bash
lerobot-teleoperate ... --teleop.type=rby1_isaac \
  --teleop.arm_mode=ee_absolute --teleop.arm_posture_hint=true
```

#### Camera panels in the headset (Televiz)

The robot cameras can be shown in the headset as floating panels (one per camera,
rendered by Isaac Teleop's Televiz through the same CloudXR connection). The frames
come from the robot's own camera reads (`--robot.cameras`), so nothing is opened twice:

```bash
lerobot-teleoperate ... \
  --robot.cameras='{"front": {...}, "left": {...}, "right": {...}}' \
  --teleop.viz_enabled=true --teleop.viz_cameras='["front","left","right"]' \
  --teleop.viz_offsets_x='[0.0,-1.1,1.1]'
```

Panels follow your head position and yaw (`viz_lock_mode=gimbal`; `head` = full head
lock, `world` = fixed in the room). On **Jetson Orin** keep `viz_openxr_composition=false`
(default) and set the **Video Codec to H.264** in the CloudXR web client, otherwise the
panels stay black. Run the preflight `--only V_viz --viz-seconds 15` first: three colour
bars must be visible.

#### Wearing the headset around the neck (`wear_mode=neck`)

For long sessions the headset can hang from the neck instead of being worn:

```bash
lerobot-teleoperate ... --teleop.wear_mode=neck --teleop.torso_engage=any_arm \
  --teleop.arm_length_source=config
```

- The robot **head is held at the ready pose** and the **headset pose drives the torso**
  (smoothed by `neck_torso_smoothing`), so bending forward with the headset on your
  chest bends the robot.
- The operator frame (Right A) comes from the body-tracking shoulder line when
  available, otherwise from the headset → both-controllers direction; the absolute-EE
  shoulder is estimated from the headset (`neck_shoulder_offset`).
- **Disable the Quest proximity sensor first** (Meta Quest Developer Hub → Device
  Actions → *Proximity Sensor* off — it re-enables ~10 min after MQDH disconnects — or
  cover the sensor with tape). Otherwise the headset sleeps within seconds of being taken
  off, the OpenXR session loses focus and the controllers stop streaming.

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
| Squeeze (grip) > `clutch_threshold` | `arm_mode=ee_clutch` (default): that arm follows the controller *delta* from the moment of the squeeze (clutch); release to hold. `arm_mode=ee_absolute`: dead-man switch — while held, the hand position **relative to your shoulder** (body tracking) is mapped onto the robot shoulder (scaled by the reach ratio) and the controller orientation (times the offset latched on Right A) becomes the gripper orientation; (re-)engaging ramps to the target over `engage_ramp_s` |
| Trigger | Gripper (fully pressed = closed) |
| Right thumbstick / left thumbstick | Base linear velocity / yaw rate |
| Right **B** | Stop: freeze every target, zero the base |
| Right **A** | Release every clutch and return arms, torso and head to the start pose (the pose right after the ready-pose motion) over `ready_return_duration_s`; re-reference the operator frame so the direction you are facing becomes robot +X; also resumes after a stop and re-centres the head origin. The base is not moved |
| Headset orientation | `head_0` (pan) / `head_1` (tilt) relative to the pose at the first tracked frame |
| Body tracking (`torso_source=body`) or headset (`head`) | Torso pose, while **both** arms are clutched (`torso_engage=both_arms`, default; `any_arm` / `always` available) |
| Headset pose (`wear_mode=neck`) | Torso pose (the robot head stays at the ready pose) |
| Body tracking shoulder / elbow (`arm_posture_hint=true`) | The arm posture (shoulder + elbow angles) is retargeted to `arm_0..arm_3` and sent as a **nullspace hint** to the Cartesian solver: the elbow follows yours while the EE pose keeps priority. Not recorded unless `record_posture_hint` (both sides) |

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
| `posture_hint_weight` | `4.0` | Nullspace weight of joints carrying an IOBT posture hint (`<side>_arm_<i>.null` action keys from `rby1_isaac`); EE mode, per-component solvers |
| `record_posture_hint` | `False` | Expose the hint keys in `action_features` so `lerobot-record` stores them (adds 8 action dims) |
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
| `wear_mode` | `"head"` | `head` (worn) or `neck` (hanging from the neck: head joints held at the ready pose, headset pose → torso, headset-based operator frame / shoulder estimate) |
| `neck_torso_smoothing`, `neck_shoulder_offset`, `shoulder_source` | `0.3`, `[-0.05,0.20,-0.15]`, `"auto"` | Neck mode: EMA on the headset pose driving the torso; shoulder position relative to the headset (left side, mirrored for right); absolute-EE shoulder from `body` / `headset` / `auto` |
| `viz_enabled`, `viz_cameras`, `viz_offsets_x`, `viz_offset_y`, `viz_distance_m`, `viz_width_m` | `False`, `[front,left,right]`, `[0,-1.1,1.1]`, `0`, `1.5`, `1.0` | Televiz camera panels: robot camera names and their placement (m) |
| `viz_lock_mode`, `viz_openxr_composition`, `viz_wait_headset_s` | `"gimbal"`, `False`, `-1` | Panel lock mode; runtime vs Televiz compositing (keep False on Jetson Orin); wait for the headset when creating the XR session |
| `arm_mode` | `"ee_clutch"` | `ee_clutch` (delta from the squeeze moment) or `ee_absolute` (hand relative to the IOBT shoulder, scaled onto the robot shoulder; squeeze = dead-man) |
| `engage_ramp_s`, `ee_max_linear_vel`, `ee_max_angular_vel` | `2.0`, `1.0`, `3.0` | `ee_absolute`: ramp to the target on engage; rate limits (m/s, rad/s) |
| `ee_position_scale`, `ee_reach_max_ratio` | `1.0`, `0.98` | `ee_absolute`: extra multiplier on the robot/human reach ratio; clamp of the hand-to-shoulder distance |
| `arm_length_source`, `human_arm_length_m` | `"body"`, `0.62` | `ee_absolute`: human reach from body tracking (`|S-E|+|E-W|`) or from the config |
| `ee_orientation_latch_on_a`, `ee_orientation_offset_rpy_deg` | `True`, `[0,0,0]` | `ee_absolute`: controller → gripper orientation offset, latched on Right A (and on the first action) or fixed |
| `arm_posture_hint`, `record_posture_hint` | `False`, `False` | IOBT shoulder / elbow → nullspace hint (`<side>_arm_<i>.null`, i = 0..3); record them (adds 8 action dims; set `--robot.record_posture_hint=true` too) |
| `posture_hint_smoothing`, `posture_hint_max_vel`, `posture_hint_hold_s`, `hint_wrist_source` | `0.3`, `2.0`, `1.0`, `"controller"` | Hint filtering; wrist point from the controller (default) or the IOBT wrist |
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
| `torso_max_rot_delta_deg`, `torso_max_z_delta_m`, `torso_use_xy` | `90`, `0.15`, `True` | Safety clamps on the torso delta since engage (rotation, height); `torso_use_xy` also passes the x/y translation of the chest through |

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

