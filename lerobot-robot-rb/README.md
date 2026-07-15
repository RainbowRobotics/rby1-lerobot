# lerobot-robot-rb

LeRobot robot interface for the Rainbow Robotics **RB-Series** collaborative
arms (RB3 / RB5 / RB10), built on the official
[`rbpodo`](https://github.com/RainbowRobotics/rbpodo) client SDK.

The RB10 is registered as `--robot.type=rb10`. The implementation is shared
across the series (`RbCobot`); adding another model only requires a
`MODEL_SPECS` entry in `models.py` plus a ~10-line registration shim (see
`rb10.py`).

## Installation

```bash
pip install -e lerobot-robot-rb
# With the RB-Y1 Dynamixel gripper driver:
pip install -e "lerobot-robot-rb[gripper-rby1]"
```

Requires Python 3.10–3.12 (`rbpodo` ships wheels up to 3.12).

## Usage

Bring-up without motion (control-box simulator):

```bash
lerobot-record \
  --robot.type=rb10 \
  --robot.ip=10.0.2.7 \
  --robot.operation_mode=simulation \
  ...
```

Switch to `--robot.operation_mode=real` only after the on-hardware
checklist below.

## Conventions

| Interface | Convention |
|---|---|
| Joint keys | `joint_0` .. `joint_5` (base → wrist) |
| Joint units | radians at the dataset boundary (rbpodo degrees internally) |
| Gripper key | `gripper_0`, normalised, **1.0 = open** |
| Cameras | one `(H, W, 3)` image per configured camera |

## Configuration highlights

| Field | Default | Notes |
|---|---|---|
| `model` | `rb10` | `MODEL_SPECS` key (joint limits, ready pose) |
| `ip` / `command_port` / `data_port` | `10.0.2.7` / 5000 / 5001 | control-box channels |
| `operation_mode` | `simulation` | `real` moves the physical robot |
| `speed_bar` | 0.3 | global speed override (0, 1] |
| `max_joint_delta_deg` | 2.0 | per-step command clamp (2° @ 30 Hz ≈ 60 °/s) |
| `joint_limits_deg` | model default | override per joint |
| `servo_t1/t2/gain/alpha` | 0.01 / 0.1 / 1.0 / 1.0 | `move_servo_j` params (retune on hardware) |
| `move_to_ready_on_connect` | `False` | validate `ready_pose_deg` for your cell first |
| `reset_on_record` | `False` | PTP to ready pose between record episodes |
| `use_velocity` / `use_current` | `False` | extra observation channels (`.vel` is finite-differenced) |
| `gripper_type` | `none` | `rby1_dynamixel` for the RB-Y1 Dynamixel gripper |
| `gripper_port` / `gripper_ids` / `gripper_invert` | `/dev/rby1_gripper` / `[0]` / `False` | Dynamixel bus settings |

## Safety / on-hardware checklist

> **WARNING**: the joint limits in `models.py` are placeholders. Replace
> them with your variant's datasheet values (e.g. RB10-1300E) before
> running in `real` mode.

1. Keep the E-stop within reach; start with `speed_bar<=0.3` and
   `max_joint_delta_deg<=2.0`.
2. In `real` mode, compare `get_observation()` against the teach pendant
   readout (degree/radian, sign conventions).
3. Identity replay — `send_action(get_observation())` — must not move the
   robot.
4. Command a small offset on one joint; verify the delta clamp ramps a
   large requested jump instead of executing it at once.
5. `reset()` PTP, then a full `lerobot-record` episode at 30 fps; verify
   dataset units.

## Extending

* **New RB model**: add a `MODEL_SPECS` entry (datasheet joint limits, a
  safe ready pose) and copy `rb10.py` to e.g. `rb5.py` with the new
  registered name; export it from `__init__.py`.
* **New gripper**: implement `RbGripperBase` (hardware convention
  0.0 = open, 1.0 = closed) and add a branch in `gripper.make_gripper` —
  e.g. the RH-P12-RN through rbpodo's built-in `gripper_rts_rhp12rn_*` API.
