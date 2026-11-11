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

## Teleoperate with NVIDIA Isaac Capture (XR headset)

> [!WARNING]
> This device was developed on an earlier branch and has been moved to the current
> robot plugin and to `isaaccapture` 1.6 with offline tests only. It has **not been
> run on a robot or a headset in this form**. The arm gains in EE mode are now the
> stiff defaults (`ee_arm_stiffness` `[1075, …]`, about 13 times the 80 Nm/rad the
> device was tuned with), so start with small, slow motions.

`--teleop.type=rby1_isaac` drives the RB-Y1 from an XR headset through
[NVIDIA Isaac Capture](https://github.com/NVIDIA/IsaacCapture) (the package formerly
named Isaac Teleop): the CloudXR runtime runs on the host that runs
`lerobot-teleoperate` / `lerobot-record`, and the headset connects with its
**browser** (no app to install). Both controllers move the arms
(end-effector space, executed by the robot's onboard Cartesian impedance solver), the
headset orientation drives the head joints, body tracking drives the torso and the
thumbsticks drive the mobile base.

### Requirements

- Host with an NVIDIA GPU: Ubuntu 22.04/24.04, Python 3.12. x86_64 workstations are
  the documented target; `isaaccapture` also ships aarch64 wheels and selects an
  experimental CloudXR runtime on **Jetson Orin** (the RB-Y1 UPC). On the UPC run the
  preflight first (below).
- Headset: Meta Quest 3 (controllers, head and inside-out body tracking) or Pico 4
  Ultra (+ motion trackers for body tracking). The headset and the host must be on
  the same network.

### Install

```bash
pip install -e "lerobot-teleoperator-rby1[isaac]"   # isaaccapture[cloudxr,retargeters-lite] 1.6.x
# once: accept the NVIDIA CloudXR EULA; downloads the runtime and starts the service
python -m isaaccapture.cloudxr.service start --accept-eula
```

The CloudXR runtime is a **host service** that is separate from the teleoperation
process. `rby1_isaac` attaches to a running service, or starts one when none runs, and
leaves it running when the program ends, so the headset stays connected between runs.

```bash
python -m isaaccapture.cloudxr.service status
python -m isaaccapture.cloudxr.service stop      # e.g. before changing the device profile or env file
```

- Without the EULA acceptance `connect()` stops with the command above instead of
  asking on the terminal.
- A running service keeps the device profile and env file it was started with;
  `cloudxr_device_profile` / `cloudxr_env_file` only apply when this process starts
  the service. Stop the service first to change them.
- `--teleop.cloudxr_run_embedded=true` runs the runtime inside the teleoperation
  process and stops it with the process (e.g. in a container). It is refused while a
  service is running.

### Jetson Orin (UPC) preflight

With the in-process runtime of `isaacteleop` 1.4, the CloudXR runtime on Orin aborted
the process when a Python thread was created after the runtime started;
`lerobot-record` creates its keyboard and camera threads after `teleop.connect()`.
Whether the detached service of 1.6 removes this is not yet confirmed on the UPC. The
preflight runs each ordering in a subprocess and prints a verdict (`GO`,
`GO-DEFERRED` → add `--teleop.session_start=first_action`, or `NO-GO` → run on an x86
host). Every stage opens a session, which waits for the headset, so keep the CloudXR
web client connected while it runs:

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
  --robot.ee_ready_torso_deg='[0,45,-90,45,0,0]' \
  --teleop.type=rby1_isaac \
  --teleop.robot_address=192.168.30.1:50051
```

`--robot.ee_ready_torso_deg` is needed with `--robot.use_torso=true`: the default ready
torso (`[0, 0, 0, 20, 0, 0]`, `torso_2` = 0) lies outside the torso solver's joint limits
(`torso_2` ≤ −0.2 rad), and the teleoperator's torso box (`torso_pos_*`, `torso_rpy_*`)
was measured around the crouched pose `[0, 45, -90, 45, 0, 0]`. The robot warns at
connect when the ready torso is outside the limits.

The arms use the absolute mapping (hand relative to your body-tracked shoulder, scaled
onto the robot shoulder) with the elbow following your own (`arm_posture_hint`, default
on). To disable the elbow hint:

```bash
lerobot-teleoperate ... --teleop.type=rby1_isaac --teleop.arm_posture_hint=false
```

#### Camera panels in the headset (Televiz)

The robot cameras can be shown in the headset as floating panels (one per camera,
rendered by Isaac Capture's Televiz through the same CloudXR connection). The frames
come from the robot's own camera reads (`--robot.cameras`), so nothing is opened twice:

```bash
lerobot-teleoperate ... \
  --robot.cameras='{"front": {...}, "left": {...}, "right": {...}}' \
  --teleop.viz_enabled=true --teleop.viz_cameras='["front","left","right"]' \
  --teleop.viz_offsets_x='[0.0,-1.1,1.1]'
```

Televiz takes frames as CUDA arrays. They are uploaded with **CuPy** (`cupy-cuda12x`,
installed by `isaaccapture[cloudxr]`), as in the upstream `camera_viz` example. The
teleoperator runs a small upload check when the Televiz session is created and stops
with a hint if CuPy cannot reach the GPU:

```bash
python -c "import cupy; print(cupy.zeros(1).device)"
```

`cupy-cuda12x` has not been checked on the UPC (JetPack 6) yet.

Panels follow your head position, yaw and pitch — never roll — (`viz_lock_mode=gimbal`;
`viz_follow_pitch=false` for yaw only, `head` = full head lock including roll, `world` =
fixed in the room) and turn to face you (`viz_face_head`).

**Moving the panels in the headset.** Like the CloudXR web client's own control panel,
every camera panel has a **grab bar** above it (`viz_grab_enabled`, default on). Point a
*free* hand at the bar — free = squeeze released, arm not clutched — it brightens; **hold
the trigger** and move the controller to drag the panel (the bar turns orange), push /
pull it with that hand's thumbstick, release the trigger to drop it. While a free hand
points at or holds a bar its trigger is a UI click: that gripper keeps its last value and
the hand cannot clutch until it lets go. Moved panels keep the lock mode (gimbal: the new
offsets follow your head) and are saved to `viz_layout_file`
(`~/.cache/rby1_isaac/viz_layout.json`), restored on the next run; `--teleop.viz_layout_reset=true`
starts from the config values again. The controller rays live in the tracking session's
OpenXR space and the panels in Televiz's; the first 30 ticks estimate the transform
between the two from the head pose seen by both (`viz_frame_bridge=auto`, logged as
"frame bridge: translation … rotation …", identity when both spaces coincide). 

On **Jetson Orin** keep `viz_openxr_composition=false` (default) and set the **Video
Codec to H.264** in the CloudXR web client, otherwise the panels show a solid colour.
This is the setting NVIDIA documents for Orin and it is still required in
`isaaccapture` 1.6.152 (see "Jetson Orin, XR" in the
[Televiz documentation](https://github.com/NVIDIA/IsaacCapture)). Run the preflight
`--only V_viz --viz-seconds 15` first: three colour bars must be visible. Adding
`--viz-openxr-composition` tests whether a newer runtime has fixed the native path.

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

The process prints the client address for each network interface and then **waits
until the headset is connected** (`isaaccapture` opens the session only then; Ctrl-C
aborts, `--teleop.xr_connect_timeout_s=<s>` gives up after that time):

1. In the headset browser open `https://<host-ip>:48322/client/?serverIP=<host-ip>&port=48322`
   (served by the CloudXR service on this host) and accept the self-signed certificate.
2. Press Connect.
3. Once the controllers are tracked the robot holds its pose until you squeeze a grip.

If the host cannot serve the client, use `https://nvidia.github.io/IsaacCapture/client/`:
enter the host IP there and first accept the certificate at `https://<host-ip>:48322/`.
The old address `nvidia.github.io/IsaacTeleop/client` no longer exists.

The status line (every `status_log_period_s`) shows the headset link, e.g.
`- (disconnected)`. When the headset disconnects, the head pose is treated as lost and
the targets are held, as for any tracking loss.

| Input | Effect |
|-------|--------|
| Squeeze (grip) > `clutch_threshold` | Dead-man switch — while held, the hand position **relative to your shoulder** (body tracking) is mapped onto the robot shoulder (scaled by the reach ratio) and the controller orientation becomes the gripper orientation. The first squeeze after connect / Right A latches the position and orientation offsets so that the controller then ↦ the start-pose gripper (no motion on that squeeze); (re-)engaging ramps to the target over `engage_ramp_s` |
| Trigger | Gripper (fully pressed = closed) |
| Right thumbstick / left thumbstick | Base linear velocity / yaw rate |
| Right **B** | Stop: freeze every target, zero the base |
| Right **A** | Release every clutch and return arms, torso and head to the start pose (the pose right after the ready-pose motion) over `ready_return_duration_s`; re-reference the operator frame so the direction you are facing becomes robot +X; also resumes after a stop and re-centres the head origin. The base is not moved |
| Headset orientation | `head_0` (pan) / `head_1` (tilt) from the headset yaw / pitch in the operator frame. The target is a look direction in the **robot base frame**: the joints are re-solved for the measured torso orientation every tick, so bending or turning the torso does not move the gaze (`head_gaze_frame`). Like the EE offsets, the look direction at the moment of Right A (and of the first action) is latched onto the start-pose head joints (`head_latch_on_a`), so after a return the robot head is straight where you were looking |
| Body tracking (`torso_source=body`) or headset (`head`) | Torso pose, while **both** arms are clutched (`torso_engage=both_arms`, default; `any_arm` / `always` available) |
| Headset pose (`wear_mode=neck`) | Torso pose (the robot head stays at the ready pose) |
| Body tracking shoulder / elbow (`arm_posture_hint=true`) | The arm posture (shoulder + elbow angles) is retargeted to `arm_0..arm_3` and sent as a **nullspace hint** to the Cartesian solver: the elbow follows yours while the EE pose keeps priority. On squeeze the hint starts at the arm's measured joints and ramps (`posture_hint_max_vel`) towards your posture; it is frozen while released, kept through body-tracking dropouts and dropped on Right A. An elbow inward of the shoulder (beyond the robot's `arm_1` limit) is clipped to the nearest reachable posture. Not recorded unless `record_posture_hint` (both sides) |

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
| `use_head` | `False` | Include the head pan / tilt joints (`head_0.pos`, `head_1.pos`) in observation and action, in both action modes |
| `record_posture_hint` | `False` | Put the arm posture hint keys (`<side>_arm_<i>.null`, i = 0..3, from `rby1_isaac`) into the action features so `lerobot-record` stores them (adds 8 action dims). The hints are applied either way |
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
| `joint_impedance_arm_stiffness` | `[1075, 990, 580, 265, 190, 205, 300]` | Arm stiffness (Nm/rad, `arm_0..6`), used for both arms. Larger towards the base, following the inertia each joint moves |
| `joint_impedance_arm_torque_limit` | `[30, 30, 25, 18, 8, 8, 8]` | Arm torque limit (Nm) of the impedance torque (gravity compensation comes on top): the most a joint pushes back when held away from its target. The arm is stiff, its compliance comes from this limit |
| `joint_impedance_torso_stiffness` | `[400]*6` | Torso stiffness (Nm/rad), with `use_torso` |
| `joint_impedance_torso_torque_limit` | `[270, 270, 270, 120, 120, 120]` | Torso torque limit (Nm): the URDF effort limits |
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
| `ee_arm_stiffness` / `ee_arm_torque_limit` / `ee_arm_damping_ratio` | `[1075,990,580,265,190,205,300]` / `[30,30,25,18,8,8,8]` / `1.0` | Arm solver joint gains, used for both arms. Stiffness and torque limit are the joint impedance values |
| `ee_arm_target_gains` | `(3.0, 2π, 150, 80π)` | Arm target limits: linear / angular velocity, linear / angular acceleration |
| `ee_right_arm_joint_limits` / `ee_left_arm_joint_limits` | elbow `(-2.6, -0.5)` | Extra joint limits (rad) given to the arm solvers |
| `ee_null_right_arm_deg` / `ee_null_left_arm_deg` | — | Nullspace joint target of each arm (deg) |
| `ee_nullspace_weight` / `ee_nullspace_stiffness` / `ee_nullspace_damping_ratio` | `[2,2,2,2,2,2,1]` / `0.2` / `1.0` | Nullspace gains |
| `ee_posture_hint_weight` | `4.0` | Nullspace weight of arm joints that carry a posture hint in the action |
| `ee_ready_torso_deg` | `None` | Torso ready pose (deg, `torso_0..5`) used in EE mode instead of the default. The default (`torso_2` = 0) is outside `ee_torso_joint_limits`; use e.g. `[0, 45, -90, 45, 0, 0]` with `use_torso=True` |
| `ee_torso_stiffness` / `ee_torso_torque_limit` / `ee_torso_damping_ratio` | `[1700]*6` / `[600]*6` / `1.0` | Torso solver joint gains |
| `ee_torso_target_gains` | `(1.0, π/2, 20, 40π)` | Torso target limits (same order as the arm) |
| `ee_torso_joint_limits` | — | Extra joint limits (rad) given to the torso solver |

**Mobile base** (`use_mobile_base=True`) — SE2Velocity commands

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mobile_base_hold_time` | `0.5` | The base keeps the last velocity this long (s) after the stream stops |
| `mobile_base_max_linear_velocity` | `0.6` | Clamp of the x / y speed (m/s) |
| `mobile_base_max_angular_velocity` | `1.4` | Clamp of the yaw rate (rad/s) |

**Robot log** — `forward_robot_logs` (`True`) forwards the robot's warnings and errors
to the Python log, `forward_robot_logs_info` (`False`) adds the info lines,
`robot_log_stream_rate` (`10.0` Hz) is the polling rate.

An action may also carry `reset.cmd` (seconds, > 0): on that tick the robot runs its
ready-pose motion instead of the commands (`rby1_isaac` sends it on Right A). It is
not an action feature and is never recorded.

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
| `resync_speed_deg_s` | `60.0` | Speed (deg/s per joint) at which the output catches up with the arm after a button press, when the arm was moved while the button was released |
| `return_to_zero_on_disconnect` | `true` | On disconnect (end of `lerobot-record` / `lerobot-teleoperate`) the leader arm moves to its zero position (all joints 0°, the arms hanging down) before its control loop stops. The robot does not follow this motion and it is not recorded |
| `return_speed_deg_s` | `30.0` | Peak joint speed (deg/s) of that return motion |
| `robot_address` | `"192.168.30.1:50051"` | Address of the robot the leader arm drives: used to switch on the 12 V supply and to read the joint limits |
| `auto_power_on_12v` | `True` | Switch on the robot's 12 V supply (leader arm power) on `connect()` |
| `require_robot_joint_limits` | `True` | Fail `connect()` when the joint limits cannot be read from the robot. `False`: warn and output targets without clipping (leader arm used without a robot) |

Joint limits are not stored in the plugins. The robot plugin and the leader arm both read
them from the robot (its URDF, so they match the model and version) at `connect()` and
clip joint targets to them. The leader arm connects to `robot_address` for this, reads
only, and disconnects before the robot plugin connects. `lerobot-record` stores the
teleoperator's action, so the recorded action is already inside the robot's limits.
`torso_target_deg` outside the robot's torso limits is an error at `connect()`.

The leader arm has no torso joints. With `use_torso=True` on both the robot and the
leader arm, every action carries `torso_target_deg`, so the robot holds its torso there
and the dataset records the torso action a policy needs. If the torso is somewhere else
when teleoperation starts, the robot moves it to the target during the startup ramp
(`joint_startup_min_time`, 5 s by default).

### Teleoperator (`Rby1XRConfig`, `--teleop.type=rby1_isaac`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `robot_address`, `robot_model`, `robot_version` | `"192.168.30.1:50051"`, `"auto"`, `"auto"` | Read-only robot link (state + forward kinematics for the clutch); model `"m"` / `"a"` and version are detected with `"auto"` |
| `profile_out` | `None` | Write this teleoperator's settings into the robot profile's `teleop` section (see [Keeping robots consistent](#keeping-robots-consistent-robot-profile)) |
| `use_torso`, `use_right_arm`, `use_left_arm`, `use_gripper`, `use_mobile_base`, `use_head` | `True` | Must mirror `Rby1Config` |
| `auto_launch_cloudxr` | `True` | Attach to the CloudXR service, starting it when none runs (`LEROBOT_CLOUDXR_SKIP_AUTOLAUNCH=1` also disables it) |
| `cloudxr_run_embedded` | `False` | Run the CloudXR runtime inside this process and stop it on disconnect, instead of the detached service |
| `cloudxr_install_dir`, `cloudxr_device_profile`, `cloudxr_env_file` | `~/.cloudxr`, `"Quest3"`, `None` | Runtime directory (holds the EULA marker), device profile and KEY=value env file (default: the packaged `default.env`). Profile and env file apply only when this process starts the runtime |
| `xr_connect_timeout_s` | `0.0` | Opening the session waits for the headset; give up after this many seconds (0 = wait forever) |
| `session_start` | `"connect"` | `"first_action"` opens the XR session on the first `get_action()` (Jetson Orin mitigation) |
| `tracking_wait_timeout_s` | `0.0` | After the session is open: give up waiting for tracked controllers after this many seconds (0 = forever) |
| `wear_mode` | `"head"` | `head` (worn) or `neck` (hanging from the neck: head joints held at the ready pose, headset pose → torso, headset-based operator frame / shoulder estimate) |
| `neck_torso_smoothing`, `neck_shoulder_offset`, `shoulder_source` | `0.3`, `[-0.05,0.20,-0.15]`, `"auto"` | Neck mode: EMA on the headset pose driving the torso; shoulder position relative to the headset (left side, mirrored for right); absolute-EE shoulder from `body` / `headset` / `auto` |
| `viz_enabled`, `viz_cameras`, `viz_offsets_x`, `viz_offset_y`, `viz_distance_m`, `viz_width_m` | `False`, `[front,left,right]`, `[0,-1.1,1.1]`, `0`, `1.5`, `1.0` | Televiz camera panels: robot camera names and their placement (m) |
| `viz_lock_mode`, `viz_follow_pitch`, `viz_openxr_composition`, `viz_wait_headset_s` | `"gimbal"`, `True`, `False`, `-1` | Panel lock mode; gimbal also follows the head pitch (never roll); runtime vs Televiz compositing (keep False on Jetson Orin); wait for the headset when creating the XR session |
| `viz_grab_enabled`, `viz_grab_threshold`, `viz_grab_push_rate_mps`, `viz_face_head` | `True`, `0.7`, `0.5`, `True` | Grab bars above the panels (free hand + trigger drags, thumbstick pushes / pulls); trigger level that starts a grab (release < 0.3); panels turn to face the head |
| `viz_frame_bridge`, `viz_layout_file`, `viz_layout_reset` | `"auto"`, `~/.cache/rby1_isaac/viz_layout.json`, `False` | Televiz ← tracking-space transform from simultaneous head poses (`identity` to skip); where moved panels are saved / restored (`""` = off); ignore the saved layout once |
| `engage_ramp_s`, `ee_max_linear_vel`, `ee_max_angular_vel` | `2.0`, `1.0`, `3.0` | Ramp to the absolute target on engage; rate limits (m/s, rad/s) |
| `ee_position_scale`, `ee_reach_max_ratio` | `1.0`, `0.98` | Extra multiplier on the robot/human reach ratio; clamp of the hand-to-shoulder distance |
| `arm_length_source`, `human_arm_length_m` | `"body"`, `0.62` | Human reach from body tracking (`|S-E|+|E-W|`) or from the config |
| `ee_orientation_latch_on_a`, `ee_position_latch_on_a`, `ee_orientation_offset_rpy_deg` | `True`, `True`, `[0,0,0]` | Controller → gripper orientation / position offsets, latched at the **first squeeze** after connect and after Right A (the controller then ↦ the start-pose gripper, so squeezing without moving the hand does not move the arm; the posture used to press A is irrelevant); `[0,0,0]` fixed orientation offset when not latched |
| `reference_source` | `"shoulders"` | Operator frame (+X) from the body-tracking shoulder line at the first action and on Right A alike (arms hold until both shoulders are valid; Right A without shoulders falls back to the gaze with a warning); `gaze` = headset look direction both times; `auto` = shoulder line when valid else gaze |
| `arm_posture_hint`, `record_posture_hint` | `True`, `False` | IOBT shoulder / elbow → nullspace hint (`<side>_arm_<i>.null`, i = 0..3); record them (adds 8 action dims; set `--robot.record_posture_hint=true` too) |
| `posture_hint_smoothing`, `posture_hint_max_vel`, `posture_hint_hold_s`, `hint_wrist_source` | `0.3`, `2.0`, `1.0`, `"controller"` | Hint filtering (EMA weight, rad/s ramp from the measured joints on squeeze, re-seed after this long without body tracking); wrist point from the controller (default) or the IOBT wrist |
| `clutch_threshold` | `0.5` | Squeeze value above which an arm follows |
| `thumbstick_deadzone`, `base_max_linear`, `base_max_angular` | `0.15`, `0.3`, `0.6` | Thumbstick → base velocity mapping |
| `head_yaw_offset_deg`, `head_pitch_offset_deg` | `0`, `0` | Head joints = headset yaw / pitch in the operator frame + these offsets (only used without `head_latch_on_a`) |
| `head_gaze_frame` | `"base"` | `base`: head joints compensate the measured torso orientation so the gaze stays put in the base frame while the torso moves (also for the held head in neck mode); `torso`: joints commanded directly |
| `head_latch_on_a` | `True` | Right A / first action / re-sync: the look direction of that moment ↦ the start (or measured) head joints, like `ee_orientation_latch_on_a`; `False` = pure absolute mapping with the fixed offsets |
| `head_yaw_sign`, `head_pitch_sign` | `1.0`, `-1.0` | Flip if the head moves the wrong way |
| `head_yaw_limit_deg`, `head_pitch_min_deg`, `head_pitch_max_deg` | `80`, `-45`, `80` | Head joint clamps |
| `head_smoothing` | `0.3` | EMA weight of the new head sample (1.0 = no filtering) |
| `torso_source` | `"body"` | `"body"` (Isaac Capture body tracking), `"head"` or `"none"` |
| `torso_engage` | `"both_arms"` | When the torso follows: `both_arms` (both grips squeezed), `any_arm`, or `always` |
| `torso_body_joint` | `"SPINE3"` | Body joint driving the torso (XR_BD 24-joint names) |
| `ready_return_duration_s` | `4.0` | Minimum time of the Right-A return: one `reset.cmd` action makes the follower run its joint-position ready-pose motion (as at connect; the loop blocks for the duration, base excluded) |
| `resync_position_threshold_m`, `resync_rotation_threshold_deg` | `0.03`, `10` | Targets are re-seeded from the measured pose on the first action and whenever a component that is not clutched drifted past these (record reset, manual move) |
| `status_log_period_s` | `5.0` | Log a one-line tracking / clutch status (controllers, head, body joints valid, why the torso holds); 0 = off |
| `torso_pos_min`, `torso_pos_max` | `[-0.15,-0.20,0.80]`, `[0.45,0.20,1.20]` | Absolute box (m, base frame) for the torso target `link_torso_5` (the chest delta since engage is applied 1:1; this box is the only limit). The crouched ready pose (`--robot.ee_ready_torso_deg='[0,45,-90,45,0,0]'`) is x = y = 0, z = 1.10 |
| `torso_rpy_min_deg`, `torso_rpy_max_deg` | `[-15,-20,-45]`, `[15,50,45]` | Absolute roll / pitch / yaw box (fixed-axis XYZ, deg; pitch + = leaning forward). Both boxes come from `scripts/torso_workspace.py` (reachable range with the chest above the knees, folded postures excluded); clipped axes show in the status log as `clamped[...]` |

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
| Arm posture hint (`record_posture_hint`, EE mode) | `<side>_arm_<i>.null`, i = 0..3 | `right_arm_3.null` (action only) |

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
- the full `Rby1Config` and the plugin's hard-coded constants (ready pose)
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
| 0.3.1 and later | 0.7.0 | 3.12, 3.13 | Tested in CI. Remote inference via lerobot's `lerobot-policy-server` / `--inference.type=remote`. Config fields grouped by controller, torso command in joint mode, measured EE action period, joint limits read from the robot. |
| 0.3.1 and later | `main` (next release) | 3.12 | Robot/teleop tested weekly in CI. |
| 0.2.x | 0.6.1 | 3.12, 3.13 | Legacy: last version with the gRPC `lerobot-async-rby1` package. |
| 0.1.x | 0.4 – 0.6.0 | ≥ 3.10 | Bare joint keys (no `.pos` suffix), unmaintained. |

### Versioning and releases

The plugins use their own [semantic version](https://semver.org/) (both packages
share one version), and every [GitHub Release](https://github.com/RainbowRobotics/rby1-lerobot/releases)
names the LeRobot release it was tested against, e.g. `v0.2.0 — LeRobot 0.6.1`.

- **Minor** (`0.2.0 → 0.3.0`): support for a new LeRobot release, or a breaking change in
  the plugins (config fields, observation/action keys).
- **Patch** (`0.2.0 → 0.2.1`): fixes that keep the same LeRobot requirement and keys.
- Exception: 0.3.0 was withdrawn one day after it was published, and 0.3.1 is the first
  release of the 0.3 series. That is why 0.3.1 contains breaking changes (listed below).
  The 0.3.0 code is commit `13d68d4`.

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

Breaking changes in 0.3.1 (compared with the withdrawn 0.3.0):

- **`Rby1Config` fields renamed** so the name says which controller a field belongs to.
  Update `--robot.<field>` arguments and config files:

  | 0.3.0 | 0.3.1 |
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
  `torso_0..5.pos`. In 0.3.0 the torso was only observed, so datasets and policies recorded
  in joint mode with `use_torso=True` have a different action layout; use plugin 0.3.0
  (commit `13d68d4`) with them. The leader arm now accepts `use_torso=True` and outputs a fixed torso target.
- **EE mode measures the action period** instead of assuming 15 Hz
  (`ee_auto_action_period=False` restores the fixed period).
- **Joint limits are read from the robot** instead of constants in the plugins. The old
  constants were wrong for `torso_2` (−150°..90°, the robot has −140°..80°) and for the
  Model M v1.3 wrist (`arm_5`, `arm_6`). The leader arm now needs to reach the robot at
  `--teleop.robot_address` on `connect()`; set `--teleop.require_robot_joint_limits=false`
  to use it without a robot.
- **Arm impedance defaults changed**: the arm is now stiff and gets its compliance from
  the torque limit. `joint_impedance_arm_stiffness` is `[1075, 990, 580, 265, 190, 205, 300]`
  (was `[150]*7`; larger towards the base) and `joint_impedance_arm_torque_limit` is
  `[30, 30, 25, 18, 8, 8, 8]` (was `[40]*7`), with `use_joint_impedance=True`. EE mode uses
  the same values: `ee_arm_stiffness` was `[80,80,80,60,60,60,60]` and `ee_arm_torque_limit`
  `[40,40,40,30,30,30,30]`. Pass the old values (e.g.
  `--robot.joint_impedance_arm_stiffness='[150,150,150,150,150,150,150]'`) to keep the old
  behaviour, e.g. for a policy trained on data recorded with it.
  The joint impedance torso torque limits are now the URDF effort limits
  `[270, 270, 270, 120, 120, 120]` (was `[500]*6`).
- Robot profiles written by 0.3.0 are still read: field names are converted on load, and
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
| [`lerobot-teleoperator-rby1/`](lerobot-teleoperator-rby1) | RB-Y1 teleoperator plugins (`--teleop.type=rby1_leader_arm`, `--teleop.type=rby1_vr`, `--teleop.type=rby1_isaac`) |
| [`tests/`](tests) | Hardware-free tests using a fake `rby1_sdk` |

## Resources

- [Rainbow Robotics Website](https://rainbow-robotics.com/en/)
- [RB-Y1 Product Page](https://rainbow-robotics.com/en/products/rb-y1/)
- [RB-Y1 Documentation](https://rainbowrobotics.github.io/rby1-dev/)
- [LeRobot Documentation](https://huggingface.co/docs/lerobot)
- [LeRobot Installation Guide](https://huggingface.co/docs/lerobot/installation)
- [NVIDIA Isaac Capture](https://github.com/NVIDIA/IsaacCapture) (XR teleoperation, `rby1_isaac`)

