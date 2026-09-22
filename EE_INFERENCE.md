# RB10 / RB10-E EE Inference

## Contract

- Robot observations: six joints in radians plus gripper (`1=open`).
- Client FK: selected URDF, `link0 -> tcp`, no extra tool offset.
- Wire state and action: `[x, y, z, rx, ry, rz, gripper]`, metres/radians,
  extrinsic XYZ Euler angles, `R = Rz @ Ry @ Rx`.
- OpenPI config: `pi05_rbc10_iros_ee`. Server encodes the first two rotation
  matrix columns as 6D, applies the checkpoint's 10D normalization, and
  decodes output to absolute XYZ/Euler/gripper. Only XYZ is delta in training.
- Client IK: seeded by current measured joints, bounded local solution,
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

The script defaults to the currently configured wrist camera and the driver's
`rby1_dynamixel` gripper option (the RB adapter uses control-box current commands).
Confirm camera serial numbers and gripper wiring/inversion against the recording
setup. Override `--robot.cameras` to supply both wrist and front. Match the
recorded RGB orientation exactly: do not rotate front a second time if the
camera driver already rotates it. No client-side resize/crop/flip is added.

- Starts paused. `f`: start inference; `s`: pause and invalidate queued and
  in-flight work; `q`: exit.
- `s` does **not** move to a ready pose. No cell-validated ready trajectory
  was supplied, so there is no automatic movement to one.
- The client enforces `inference_safe_start=True`: connecting defers impedance
  setup and gripper initialization. `dry_run=true` never enables actuators or
  calls `send_action`. It still connects to hardware and cameras, so it is
  **not** a hardware-free test mode. Before initialization, gripper state is
  the driver's cached assumed-open value, not measured feedback.
- For an operator-approved motion trial, explicitly add `--dry_run=false`.
  The first `f` prepares impedance control and can open the configured gripper;
  subsequent pause/resume cycles do not reinitialize it.
  Set the initial pose using the pendant first and keep physical emergency
  stop available. No actual motion trial was performed during development.
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

FK/IK intersect the packaged URDF limits `[-3.14, 3.14]` with conservative
software elbow limits: RB10 +/-165 degrees, RB10-E +/-154 degrees. These are
deployment envelopes, not a claim about certified physical travel. States
outside them are rejected rather than silently wrapped. TCP/tool calibration and the joint zero
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
