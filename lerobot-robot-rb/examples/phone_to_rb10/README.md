# Phone teleoperation for the RB10

Drive the RB10's TCP from an Android phone, using LeRobot's built-in `phone`
teleoperator and the robot's `action_space="ee"` mode. Inverse kinematics runs
on the control box, so **no URDF or kinematics solver is needed**.

This is position-only teleoperation: the phone's translation drives the TCP's
x/y/z in the **robot base frame**, and the TCP keeps its current orientation.

## Setup

```bash
pip install -e lerobot-robot-rb
pip install "lerobot[phone]"          # hebi-py, teleop, fastapi

# Optional: add the on-screen X/Y/Z jog buttons (see below)
bash examples/phone_to_rb10/frontend/apply.sh
```

## Run

```bash
python examples/phone_to_rb10/teleoperate.py
```

The `teleop` package prints an HTTPS URL. On the phone (same network):

1. Open the URL — you must type `https://` and accept the self-signed
   certificate warning ("Advanced" → "Proceed anyway"). WebXR requires a
   secure context.
2. Tap **Start**.
3. Hold the phone with the top edge pointing along the robot's +x and the
   screen facing up (+z), then press **Move** to capture the calibration pose.

## Controls

| Control | Effect |
|---|---|
| **Move** (hold) | Phone translation drives the TCP. Release to stop. |
| **X/Y/Z +/-** (hold) | Jog the TCP along that base-frame axis. Requires the patch. |

Both are deadman-style: motion happens only while the button is held. Each
fresh press re-latches the reference, so control always resumes from wherever
the arm currently is.

Tunables at the top of `teleoperate.py`: `POSITION_SCALE`, `JOG_SPEED_MM_S`,
`CMD_LEASH_M`, `EE_BOUNDS_MM`, `AUTO_CLEAR_KINEMATICS_EMS`.

## The jog-button patch

The phone UI is served by the third-party [`teleop`](https://pypi.org/project/teleop/)
package (Spes Robotics, Apache-2.0). Its stock page has Move / Gripper / A / B
buttons and a scale slider, but no per-axis jog buttons.

Rather than vendoring a copy of that project's frontend, `frontend/jog.patch`
is a ~40-line patch that adds six jog buttons and sends their state in the
pose message. `frontend/apply.sh` applies it to the `teleop` copy installed in
your environment (idempotent; keeps `.bak` files).

**Phone teleoperation works without the patch** — you simply get the IMU pose
control and no jog buttons. `teleoperate.py` treats missing jog data as "no
buttons pressed".

If `teleop` is upgraded and the patch stops applying, re-create it:

```bash
TELEOP_DIR=$(python -c 'import teleop, os; print(os.path.dirname(teleop.__file__))')
# edit $TELEOP_DIR/index.html and $TELEOP_DIR/assets/teleop-ui.js by hand, then:
cd "$TELEOP_DIR" && diff -u index.html.bak index.html > jog.patch
```

## Safety

* Start from a bent-elbow (non-singular) pose — near full stretch the control
  box rejects Cartesian targets ("unsorvable" / "armstratch").
* Keep the E-stop within reach; verify each axis direction with small motions
  before working at speed.
* The driver clamps per-step EE motion and can optionally bound the workspace
  (`EE_BOUNDS_MM`). A latched *kinematics* EMS (IK failure) is auto-cleared and
  the loop resumes; real safety stops (collision, soft-estop) always abort.
