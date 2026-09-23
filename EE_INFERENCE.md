# RB10 / RB10-E EE Inference

## Contract

- Robot observations: six joints in radians plus gripper (`1=open`).
- Client FK: selected URDF, `link0 -> tcp`, no extra tool offset.
- Wire state and action: `[x, y, z, rx, ry, rz, gripper]`, metres/radians,
  extrinsic XYZ Euler angles, `R = Rz @ Ry @ Rx`.
- OpenPI config: `pi05_rbc10_iros_ee`. Server encodes the first two rotation
  matrix columns as 6D, applies the checkpoint's 10D normalization, and
  decodes output to absolute XYZ/Euler/gripper. Only XYZ is delta in training.
- Client IK: seeded by the previous command (controller `jnt_ref` on activation), bounded local solution,
  followed by the existing joint-impedance ServoJ driver. No ServoL is used.
- `rb10`: `rb10_1300e.urdf`; `rb10e`: `rb10_1300e_u.urdf`.
  Both use the `rb10` driver; `kinematics_model` selects the physical geometry.
- Wrist maps to `observation/base_image`; optional front maps to
  `observation/front_image`. Missing front is omitted, not sent as a real
  black camera. The server applies its missing-camera mask.

## Server

Use the updated OpenPI `iros_ee` code and a completed **EE** checkpoint,
not a flower/joint checkpoint. Keep the checkpoint `assets` directory.

```bash
cd /data1/kgs/pi05_rb
bash deployment_scripts/serve_rb10_ee.sh /absolute/path/to/completed/checkpoint 0.0.0.0 8000
```

This uses `pi05_rbc10_iros_ee` and warms up wrist-only and wrist/front inputs
before listening. It does not connect to any robot. Run it on a free inference
GPU/machine, not the GPUs currently occupied by training. The endpoint is
unauthenticated: use a trusted private network/firewall, not public exposure.
The client rejects a server without the matching `rb10-ee-v1` descriptor.

## Client

Use the existing client Python environment with LeRobot, NumPy, SciPy,
draccus, msgpack, pyzmq and the configured camera/gripper dependencies.
No packages are installed by these scripts. The launch script selects this
checkout through `PYTHONPATH`; do not accidentally run an older checkout.

```bash
cd /path/to/IROS/rby1-lerobot

# RB10: replace the example addresses with the actual server and robot IPs.
bash scripts/inference_ee.sh rb10 SERVER_IP:8000 ROBOT_IP

# RB10-E: same policy, different client FK/IK geometry.
bash scripts/inference_ee.sh rb10e SERVER_IP:8000 ROBOT_IP
```

The script defaults to wrist and front cameras and the driver's
`rby1_dynamixel` gripper option (the RB adapter uses control-box current commands).
Confirm camera serial numbers and gripper wiring/inversion against the recording
setup. Both wrist and front use no rotation (`rotation: 0`) and 640x480 images,
per the revised camera setup. This does not establish that training images
have the same orientation; compare them separately before evaluating the policy.
Override `--robot.cameras` for a different camera selection. No additional
client-side resize/crop/flip is added.

- Real mode (`--dry_run=false`) starts with a profiled joint move to the
  dataset ready pose, then waits paused. `f`: start inference from the current
  pose; `s`: stop inference, invalidate queued and in-flight work, and return
  to the ready pose (`s` again during that move aborts it and stays paused);
  `q`: exit. Dry-run never moves.
- Ready pose (`--ready_pose_rad`, default in `DEFAULT_READY_POSE_RAD`) is the
  mean first-frame joint state of the 197 episodes of
  `rainbowrobotics/rb10_iros_double_trim` (revision `f7e8b828`):
  `[-174.79, -2.42, 143.55, 40.28, 275.53, 179.77]` deg. Per-joint spread
  across episodes is 1.7-4.9 deg (std). The move is a cosine profile over at
  least `--ready_move_duration_s` (5 s), streamed through the same ServoJ
  gate and per-tick speed/tracking limits as inference, anchored to the
  controller reference; it holds `--ready_settle_s` (0.5 s) at the target and
  opens the gripper (`--ready_gripper_open`, default true; every recorded
  episode starts open) and caps its peak joint speed at
  `--ready_max_joint_speed_rad_s` (0.15 rad/s, so a 44 deg move takes ~8 s
  instead of 5 s: at 5 s the servo lag plus impedance sag tripped the 0.15 rad
  tracking bound once on 2026-09-23). `--move_to_ready_on_start=false` /
  `--move_to_ready_on_stop=false` disable each trigger.
- The client enforces `inference_safe_start=True`: connecting defers impedance
  setup and gripper initialization. `dry_run=true` never enables actuators or
  calls `send_action`. It still connects to hardware and cameras, so it is
  **not** a hardware-free test mode. Before initialization, gripper state is
  the driver's cached assumed-open value, not measured feedback.
- For an operator-approved motion trial, explicitly add `--dry_run=false`.
  The first `f` prepares impedance control and can open the configured gripper.
  Subsequent pause/resume cycles do not reinitialize it. Keep physical emergency
  stop available; software bounds do not establish collision-free movement.
- Connection failures retry indefinitely with bounded per-request waits.
  A timeout pauses commands and discards stale responses. After recovery,
  press `f`; reconnection never automatically resumes movement.
- Chunks are selected by observation age, not blindly replayed from index 0.
  Euler angles are not linearly averaged. This client uses latest valid
  chunks, not the RBY1 `weighted_average` action aggregator.

## Limits And Validation

Default joint-speed limit is `0.5 rad/s` at 30 Hz, with an additional
per-command cap; IK failure, limit violation, stale state, expired action,
or a control-computation overrun pauses inference. These are rejection gates,
not collision avoidance or a certified safety controller. Do not increase
limits merely to suppress a fault; inspect the target/current state first.

Goal IK uses a bounded 0.5 rad local neighbourhood of the command anchor, with
full position/orientation residual checks. The previous 0.2 rad goal box
excluded valid recorded targets (dataset maximum observed gap: 0.4682 rad).
This is a planning bound, not permission to move 0.5 rad in one tick.
For a distant goal, the solver constructs a straight XYZ waypoint and shortest
SO(3) rotation interpolation, then solves IK again INSIDE the intersection of
model limits, measured joints +/- the tracking error limit, and previous command
+/- the per-tick speed limit. It never independently clips joints after IK: that can reverse
TCP coordinate directions. Up to three successively smaller waypoints are
attempted; unreachable goals, invalid intersections or residual failures still
pause. The control deadline and actual command speed limits are unchanged.
This checks setpoint FK, not physical closed-loop tracking or collision safety.

Activation anchors the trajectory to controller `jnt_ref`; each subsequent
waypoint starts at the previous command, not the lagging measured pose.
Model observations still use measured `jnt_ang`, matching the dataset contract.
`max_joint_tracking_error_rad` defaults to 0.05 rad per joint and is separate
from the unchanged 0.5 rad/s command speed. Excessive reference/measurement or
command/measurement error pauses inference and requires explicit `f` to rearm.
This bound does not establish safe contact forces or correct TCP calibration.

The controller holds a steady bias between `jnt_ang` and `jnt_ref` under
joint impedance (2026-09-22 RB10 logs: elbow +1.0 to +2.8 deg, always the same
sign, pose dependent; up to 0.068 rad against a 0.05 rad limit). Activation
records this offset (`offset_deg` in the anchor log) and every tracking
comparison and IK search box uses the offset-corrected measurement, so the
bias is not counted as tracking error and cannot collapse the search box on
one side. The limit value is unchanged. Replaying the 212 logged ticks with
this correction removed all 20 residual failures and cut worst-case IK time
from 37 ms to 19 ms without loosening any tolerance. The physical cause of
the bias (compliance, gravity feedforward, calibration) is still unconfirmed;
an offset above the limit at activation still fails closed.

The IK solver is a box-constrained Levenberg-Marquardt on the analytic
geometric Jacobian (one forward pass per evaluation, at most 100 evaluations
per solve, stagnation at an active bound ends the solve early so the smaller
waypoint retry runs instead). Replaying the 2026-09-22 logged ticks on the
client PC: median 5.3 ms, maximum 12.7 ms per `inverse_step`, versus 14.6 ms
median / 37 ms maximum with the previous SciPy `least_squares` path, whose
overhead (not the FK) dominated and overran the 33 ms budget whenever the
network thread competed for the interpreter. Acceptance still uses an
independent SciPy rotation check at the unchanged 1e-6 m / 1e-6 rad.

Live logs after the offset correction still show 2-2.8 deg of normal
tracking lag on moving joints (ServoJ `alpha=0.1`, the same value used while
recording). That lag consumes the default 0.05 rad tracking bound; the
default was left unchanged here, so pass
`--max_joint_tracking_error_rad=0.15` explicitly for motion trials if the
`tracking_error_deg` log field approaches 2.86 deg. This is an operator
decision, recorded in the log header.

`--servo_mode=cartesian` sends bounded TCP waypoints as `move_servo_l`
instead of client IK + ServoJ, for A/B comparison. The waypoint continues from
the previous TCP command (anchored to `FK(jnt_ref)` at activation) along the
straight line / shortest rotation to the model target with one shared
fraction (`--max_tcp_speed_m_s`, `--max_tcp_angular_speed_rad_s`), and the
offset-corrected measured TCP must stay within `--max_tcp_tracking_error_m` /
`--max_tcp_tracking_error_rad` of the previous command. The control box
solves IK with its own TCP setting: the passive 2026-09-22 comparison shows
controller `tcp_ref` within 0.7 mm / 0.2 deg of `FK(jnt_ref)` and the same
Z-Y'-X'' (= `Rz @ Ry @ Rx`) Euler convention, so only units are converted.
Joint envelopes can only be checked on measured joints in this mode, and the
ready move still uses ServoJ; whether the control box accepts ServoL directly
after ServoJ streaming is unverified until tried on hardware.

The one-control-period IK budget is measured from observation acquisition,
after the camera reads: each RealSense `async_read` blocks until a new frame,
so counting that wait against IK made almost every tick "obsolete".
Chunk indexing uses the same acquisition instant. An overrun logs
`observation_ms`/`ik_ms` before pausing.

Client logs once per second by default (`--action_log_interval_s=0` disables):
`current_xyz`, `target_xyz`, `target_delta`, `command_fk_delta`, and
`command_from_anchor_delta`, all in
base frame `link0`, metres, for the actual selected chunk index. Server EE
diagnostics (`--ee-action-log-interval 1`, 0 disables) show observation XYZ,
absolute output XYZ after unnormalization/delta reconstruction and
`xyz_minus_observation` at the first/middle/last chunk indices. Negative Z means
a lower target in the base frame; human "forward" must be mapped to the physical
base axes rather than assumed to equal +X. These logs distinguish model target
direction from command FK direction without modifying model outputs.
Measured, previous-command and planned-command joint angles are also logged
in degrees, with `tracking_error_deg` (previous command minus offset-corrected
measurement), `observation_ms` and `ik_ms`. Rejected actions log the same
tracking error and the activation offset. Planned commands are not controller
acknowledgements; physical tracking still needs independent measurement.

URDF supplies geometry, not the controller travel envelope: its generic
`[-3.14, 3.14]` limits exclude valid recorded poses. FK/IK use the existing
RB10 model envelope (+/-360 degrees except elbow +/-165) and the existing
RB10-E teleoperation envelope (+/-360 except shoulder +/-180 and elbow +/-154).
These are deployment envelopes, not a claim about certified physical travel.
Controller angles are never wrapped; IK stays bounded near the command anchor.
States outside the envelope are rejected with per-joint diagnostics. TCP/tool calibration and the joint zero
conventions must match the dataset conversion before physical use.

Hardware-free tests cover both URDFs, local FK/IK roundtrips, camera modes,
wire validation, socket timeout recovery, action invalidation, command gates
and timestamped latest-state delivery. A cross-repository test runs the real
OpenPI EE transforms over local ZMQ with a deterministic substitute for the
neural network. This verifies the pipeline, not learned task success.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  lerobot-async-rby1/tests/test_robot_client_ee.py \
  lerobot-async-rby1/tests/test_rb10_ee_policy.py \
  lerobot-robot-rb/tests/test_ee_kinematics.py \
  lerobot-robot-rb/tests/test_inference_safety.py \
  lerobot-robot-rb/tests/test_state_freshness.py
```

`test_openpi_ee_roundtrip.py` additionally needs the matching OpenPI source
and dependencies on `PYTHONPATH`. Socket tests bind only localhost.

### Verification Scope (2026-09-22)

- Client, both FK/IK models, mocked adapter safety, and state-receiver process
  tests: 84 passed, including spawn/fork/forkserver and timeout/reconnect races.
- Server protocol/prewarm tests plus both models and both camera modes through
  real OpenPI transforms and local ZMQ: 31 passed in an isolated CPU environment.
- CLI parsing, shell syntax, new-code Ruff checks and diff whitespace checks passed.
  Existing unrelated F541/F841 lint findings remain in cobot.py/config_rb.py.
- The server implementation, its tests and launcher were deployed to
  `/data1/kgs/pi05_rb`; SHA-256 hashes match the verified files. Training files
  and the running training process were not changed by this inference work.
- No physical robot motion, target-client timing test, or new checkpoint
  task-success evaluation was performed. These remain deployment checks.
