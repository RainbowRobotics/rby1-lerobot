# rby1-xr-teleop

XR teleoperation of the Rainbow Robotics RB-Y1 as a library: the operator's
controllers, headset and tracked body go in, robot targets come out. It depends
on NumPy and SciPy only — no LeRobot, no `isaaccapture`, no `rby1_sdk` — so the
same logic can run behind the LeRobot device (`--teleop.type=rby1_isaac` in
[`lerobot-teleoperator-rby1`](../lerobot-teleoperator-rby1)), in a simulator or
in a test.

```bash
pip install -e rby1-xr-teleop        # Python >= 3.11
```

## One tick

```python
from rby1_xr_teleop.core import XrTeleopCore
from rby1_xr_teleop.settings import XrTeleopSettings
from rby1_xr_teleop.state import RobotInfo, RobotState

core = XrTeleopCore(XrTeleopSettings(), RobotInfo(version="1.3", holonomic=True))
core.reset(state)                         # targets = where the robot is now
command = core.step(frame, state, now)    # every tick
```

| The host gives | |
|---|---|
| `frame`: [`XRFrame`](rby1_xr_teleop/xr_frame.py) | Both controllers (grip pose, squeeze, trigger, thumbstick, A/B or X/Y), the headset pose and the tracked body (24 joints, the `XR_BD_body_tracking` layout). Positions in metres with the robot base's axes (x forward, y left, z up); the heading is arbitrary — the core references it on the first frame and on Right A. A device that is not tracked is `None`. |
| `state`: [`RobotState`](rby1_xr_teleop/state.py) | Measured poses of `link_torso_5`, `link_right_arm_6`, `link_left_arm_6` in the base frame, the head joints, the arm joints (for the posture hints) and the base odometry as an SE(2) matrix (needed for the body drive). |
| `now` | A monotonic time in seconds. |
| `RobotInfo` (once) | Robot version (arm geometry), whether the base is holonomic, and optionally the measured shoulder → `link_arm_6` distance of the straight arm. |

| The host gets: [`TeleopCommand`](rby1_xr_teleop/command.py) | |
|---|---|
| `torso`, `right_ee`, `left_ee` | 4×4 target poses in the base frame (Cartesian impedance on the RB-Y1). |
| `head` | `[head_0, head_1]` joint targets (rad). |
| `right_gripper`, `left_gripper` | 1.0 = open. |
| `base` | `(vx, vy, wz)` in the base frame. |
| `hints` | Per arm, joint targets for `arm_0..arm_3` meant for the solver's nullspace (the elbow follows the operator's). |
| `reset_seconds` | Set for one tick on a return to start: the robot should run its own return-to-ready motion over at least this long instead of following the targets. |

`command.as_action()` is the same command as a flat dictionary with the key names of
the RB-Y1 LeRobot plugins (`right_ee.x`, `head_0.pos`, `x.vel`, `right_arm_0.null`,
`reset.cmd` …). A field that is `None` is not commanded.

What the operator does — squeeze to move an arm, Right A to start / return, Right B to
stop, Left X to drive the base with the body — and every option is described in
[`settings.py`](rby1_xr_teleop/settings.py) and in the repository README.

## Getting an `XRFrame`

- **Isaac Capture / Isaac Teleop pipeline**: [`isaac_io.frame_from_outputs`](rby1_xr_teleop/isaac_io.py)
  turns the outputs of a pipeline that exposes the controller, head and full-body sources
  into an `XRFrame`. It does not import the NVIDIA package; the layouts it reads are the
  same in `isaacteleop` 1.4 and `isaaccapture` 1.6.
- **Script**: [`scripted.Operator`](rby1_xr_teleop/scripted.py) builds frames from a few
  numbers (stance, body and neck angle, hand positions, grips, buttons);
  `scripted.FollowingRobot` is a robot state that follows the commands. The tests and the
  simulator check of the LeRobot device use them.

## Modules

| Module | |
|---|---|
| `core` | `XrTeleopCore`: the per-tick logic |
| `settings`, `state`, `command` | `XrTeleopSettings`, `RobotState` / `RobotInfo`, `TeleopCommand` |
| `gating` | Which of arms, torso, head, grippers, base may move in each mode (one truth table) |
| `operator_frame` | Which way the operator faces: the yaw applied to every incoming pose |
| `absolute_ee`, `arm_retargeter` | Shoulder-relative arm mapping; operator arm posture → `arm_0..arm_3` |
| `retargeters`, `clutch`, `so3` | Head, torso box, base, pose helpers; engage-relative clutch; small SO(3) functions |
| `xr_frame`, `isaac_io`, `scripted` | Input data types; Isaac Capture output parsing; scripted operator |

## Using it from Isaac Lab

Not implemented here. What an adapter has to bridge:

- Isaac Lab 3.0 pins `isaacteleop` 1.4 and its `IsaacTeleopDevice` owns the XR session
  and steps it once per `advance()`. A second session in the same process is not known to
  work, so the frame has to come from Isaac Lab's own pipeline outputs
  (`isaac_io.frame_from_outputs` reads them), not from a session created by this package.
- `RobotState` comes from the articulation (body poses in the robot base frame, joint
  positions, the base pose as odometry).
- `TeleopCommand` holds task-space targets. The RB-Y1 tasks in `rby1_lab` take joint-space
  actions, so the environment needs inverse-kinematics action terms for the torso and the
  arms (and a base velocity term) before the command can be applied.

## Tests

```bash
pytest rby1-xr-teleop/tests      # needs only this package and pytest
```

`tests/scenario.py` is one whole session with the default settings; it is also run through
the LeRobot device in the repository's `tests/`.
