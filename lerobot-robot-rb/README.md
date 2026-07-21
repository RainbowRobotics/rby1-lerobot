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

## Motion model

`send_action()` never sends servo commands directly. It updates a target
that a background worker interpolates linearly and refreshes through
`move_servo_j` (joint mode) or `move_servo_l` (EE mode) every 5 ms, so the
control box receives a continuous reference instead of action-rate steps.
Stepped references combined with a small `servo_t1` cause stop-and-go
torque spikes that the control box misreads as **external collisions** —
if you see spurious collision stops, check `servo_t1` (should be at least
the action period) and `servo_alpha` (must be < 1; smaller is smoother)
before touching the pendant's collision sensitivity.

## Action spaces

* `action_space="joint"` (default): actions are joint positions
  (radians), streamed with `move_servo_j`.
* `action_space="ee"`: actions are TCP poses `ee_x..ee_rz`
  (metres / radians), streamed with `move_servo_l`. Inverse kinematics
  runs on the control box — no URDF required. This is the mode to pair
  with pose-based teleoperators such as LeRobot's `--teleop.type=phone`
  (via a small mapping step, see the SO-100 phone example).

  EE streaming must start from a non-singular pose: with the arm nearly
  fully stretched the control box rejects every target with
  "unsorvable" / "armstratch" errors and the robot does not move (the
  streaming worker logs these). Move to a bent-elbow pose such as the
  ready pose first.

## Record without a physical teleoperator

The package registers a software-only teleoperator named `rb_auto` that
captures the connected robot's pose and moves one axis with a small sine
wave — an end-to-end recording smoke test without extra hardware:

```bash
lerobot-record \
  --robot.type=rb10 \
  --robot.ip=<CONTROL_BOX_IP> \
  --robot.operation_mode=simulation \
  --robot.gripper_type=none \
  --teleop.type=rb_auto \
  --teleop.axis=5 \
  --teleop.amplitude=0.5 \
  --teleop.period_s=8.0 \
  --dataset.repo_id=local/rb10-auto-smoke \
  --dataset.root=/tmp/rb10-auto-smoke \
  --dataset.single_task="Verify RB10 automatic recording" \
  --dataset.num_episodes=1 \
  --dataset.episode_time_s=10 \
  --dataset.reset_time_s=0 \
  --dataset.video=false \
  --dataset.push_to_hub=false \
  --play_sounds=false
```

For the EE variant add `--robot.action_space=ee --teleop.space=ee`
(amplitude is then millimetres on axes 0-2, degrees on 3-5). In simulation
mode observations use rbpodo's `*_ref` state; physical mode uses measured
state.

## First real-robot smoke test

`scripts/rb10_real_smoke.sh` is a dry run by default and only prints the
prepared command. At the robot, after the checklist below:

```bash
RB10_REAL_TEST_CONFIRM=CELL_CLEAR_ESTOP_READY \
  ./lerobot-robot-rb/scripts/rb10_real_smoke.sh --execute
```

Arm power/servo initialization is never automatic — power on from the
pendant first. The driver aborts before streaming if the control box
reports an E-stop, collision, freedrive, disabled collision detection, or
incomplete arm activation.

## Conventions

| Interface | Convention |
|---|---|
| Joint keys | `joint_0` .. `joint_5` (base → wrist) |
| Joint units | radians at the dataset boundary (rbpodo degrees internally) |
| EE keys | `ee_x, ee_y, ee_z, ee_rx, ee_ry, ee_rz` (metres / radians; rbpodo mm / degrees internally) |
| Gripper key | `gripper_0`, normalised, **1.0 = open** |
| Cameras | one `(H, W, 3)` image per configured camera |

## Configuration highlights

| Field | Default | Notes |
|---|---|---|
| `model` | `rb10` | `MODEL_SPECS` key (joint limits, ready pose) |
| `ip` / `command_port` / `data_port` | `10.0.2.7` / 5000 / 5001 | control-box channels |
| `operation_mode` | `simulation` | `real` moves the physical robot |
| `real_mode_confirm` | `False` | mandatory acknowledgement for real mode |
| `action_space` | `joint` | `ee` streams TCP poses through control-box IK |
| `speed_bar` | 0.3 | global speed override (0, 1] |
| `servo_t1/t2/gain/alpha` | 0.05 / 0.1 / 1.0 / 0.3 | Rainbow: t1 = time-to-target, t2 = hold time, alpha = LPF gain in (0,1) |
| `servo_command_period_s` | 0.005 | worker refresh period (200 Hz) |
| `interp_min_s` / `interp_max_s` | 0.02 / 0.2 | bounds for the per-target interpolation window |
| `max_joint_delta_deg` | 2.0 | per-step command clamp (2° @ 30 Hz ≈ 60 °/s) |
| `max_ee_pos_delta_mm` / `max_ee_rot_delta_deg` | 5.0 / 2.0 | EE per-step clamps (5 mm @ 30 Hz ≈ 150 mm/s) |
| `ee_bounds_mm` | `None` | optional TCP workspace box |
| `joint_limits_deg` | model default | override per joint |
| `require_collision_detection` | `True` | refuse real streaming when disabled |
| `move_to_ready_on_connect` | `False` | validate `ready_pose_deg` for your cell first |
| `reset_on_record` | `False` | PTP to ready pose between record episodes |
| `use_velocity` / `use_current` | `False` | extra observation channels (`.vel` is finite-differenced) |
| `gripper_type` | `none` | `rby1_dynamixel` for the RB-Y1 Dynamixel gripper |
| `gripper_port` / `gripper_ids` / `gripper_invert` | `/dev/rby1_gripper` / `[0]` / `False` | Dynamixel bus settings |

## Safety / on-hardware checklist

> **WARNING**: `models.py` contains the published RB10-1300 catalogue joint
> ranges. Verify the physical arm's nameplate and override
> `joint_limits_deg` if it is a different variant. The ready pose remains
> cell-specific.

1. Keep the E-stop within reach; start with `speed_bar<=0.3` and the
   default per-step clamps.
2. Power/activate the arm from the pendant; confirm Real Robot mode, no
   alarms, collision detection enabled.
3. In `real` mode, compare `get_observation()` against the teach pendant
   readout (degree/radian, sign conventions).
4. Identity replay — `send_action(get_observation())` — must not move the
   robot.
5. Run `scripts/rb10_real_smoke.sh` (dry run, then `--execute`); watch for
   smooth motion with no collision-detection stops before increasing
   amplitude, duration, or speed.
6. For EE mode, repeat 4-5 with `SPACE=ee` and validate `ee_bounds_mm`
   for your cell.

## Extending

* **New RB model**: add a `MODEL_SPECS` entry (datasheet joint limits, a
  safe ready pose) and copy `rb10.py` to e.g. `rb5.py` with the new
  registered name; export it from `__init__.py`.
* **New gripper**: implement `RbGripperBase` (hardware convention
  0.0 = open, 1.0 = closed) and add a branch in `gripper.make_gripper` —
  e.g. the RH-P12-RN through rbpodo's built-in `gripper_rts_rhp12rn_*` API.
