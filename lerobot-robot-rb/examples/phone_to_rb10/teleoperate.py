#!/usr/bin/env python
"""Phone teleoperation for the RB10 (position-only, EE mode).

Bridges LeRobot's built-in ``phone`` teleoperator to ``Rb10`` running in
``action_space="ee"``. The phone reports a 6-DoF pose, but this script uses
**only its position**: the phone's translation drives the robot's TCP x/y/z,
and the phone's rotation is ignored entirely. Targets stream through
``move_servo_l`` — inverse kinematics runs on the control box, so no URDF or
kinematics solver is involved.

``move_servo_l`` takes a full 6-DoF pose, so an orientation must still be
sent; this script always sends the robot's *current measured* orientation,
which means orientation is never commanded to change — the TCP keeps its
attitude while you translate it.

Why a script (and not just ``--teleop.type=phone``): the phone teleoperator
emits ``phone.pos``/``phone.rot`` (raw pose), not the robot's ``ee_*`` action
keys, and the record/teleoperate CLIs pass the teleop action through
unchanged (identity processor). This loop is the missing mapping layer.

Safety
------
* Deadman: motion happens only while you hold a finger on the WebXR page
  (Android ``move`` event). Release to stop; the robot holds position.
* On each fresh press the position reference is re-latched, so control
  resumes from wherever the arm is — no jump.
* The robot's per-step EE clamp and optional ``ee_bounds_mm`` box stay in
  force.
* Start from a bent-elbow (non-singular) pose or the control box rejects
  the Cartesian targets ("unsorvable"/"armstratch").

Android setup
-------------
1. ``pip install "lerobot[phone]"`` (hebi-py, teleop, fastapi).
2. Run this script; the ``teleop`` package prints a URL and serves a WebXR
   page. Open it in the phone's browser (same network, HTTPS/localhost
   rules apply — see the teleop package docs).
3. Hold the phone: top edge forward = robot +x, screen up = robot +z.
   Touch and drag on the page to capture the calibration pose, then keep
   the finger down to drive.
"""

import time

import numpy as np

from lerobot.teleoperators.phone import Phone, PhoneConfig
from lerobot.teleoperators.phone.config_phone import PhoneOS
from lerobot.teleoperators.phone.phone_processor import MapPhoneActionToRobotAction

from lerobot_robot_rb import EE_NAMES, Rb10, Rb10Config

# ── Edit these for your cell ─────────────────────────────────────────────
ROBOT_IP = "192.168.0.210"
FPS = 30

# Phone metres -> robot metres. 1.0 is 1:1; lower it if motion feels fast.
# The robot's max_ee_pos_delta_mm still caps per-step speed regardless.
POSITION_SCALE = 0.5

# Optional TCP workspace box in the control-box global frame, mm:
# [(xmin, xmax), (ymin, ymax), (zmin, zmax)]. None disables the box — set it
# once you know a safe envelope for your cell.
EE_BOUNDS_MM = None

# Auto-clear a latched *kinematics* emergency-stop (IK failure toward a
# singularity / out of reach) and resume, instead of aborting. Real safety
# stops — collision, self-collision, soft-estop, device error — are NEVER
# auto-cleared; they still abort. Set False to abort on any stop.
AUTO_CLEAR_KINEMATICS_EMS = True
# Stop trying after this many consecutive auto-clears with no good motion
# in between (keeps a persistent singularity from looping forever).
MAX_CONSECUTIVE_CLEARS = 5

# On-screen jog buttons (X/Y/Z +/-) added to the WebXR page. While a button
# is held the TCP jogs along that base-frame axis at JOG_SPEED_MM_S; release
# stops. Jog works only when the phone "hold to move" button is NOT held.
JOG_SPEED_MM_S = 40.0
# The commanded position is kept within this distance (metres) of the arm's
# measured pose so it cannot run away if the arm stalls (e.g. at a limit).
CMD_LEASH_M = 0.08
# ─────────────────────────────────────────────────────────────────────────


def read_jog(teleop: "Phone") -> dict:
    """Best-effort read of the WebXR jog-button state.

    The jog fields are extra keys the patched frontend adds to the pose
    message; LeRobot's ``Phone`` forwards the raw message verbatim to the
    Android backend, which stores it as ``_latest_message``. We read it
    there because ``get_action()`` only surfaces the known button keys.
    """
    impl = getattr(teleop, "_phone_impl", None)
    msg = getattr(impl, "_latest_message", None)
    if not isinstance(msg, dict):
        return {}
    jog = msg.get("jog", {})
    return jog if isinstance(jog, dict) else {}


def main() -> None:
    robot = Rb10(
        Rb10Config(
            ip=ROBOT_IP,
            operation_mode="real",
            real_mode_confirm=True,
            action_space="ee",
            speed_bar=0.2,
            max_ee_pos_delta_mm=5.0,
            ee_bounds_mm=EE_BOUNDS_MM,
        )
    )
    teleop = Phone(PhoneConfig(phone_os=PhoneOS.ANDROID))
    mapper = MapPhoneActionToRobotAction(platform=PhoneOS.ANDROID)

    robot.connect()
    teleop.connect()
    print("Teleop ready. Hold 'move' and tilt the phone to drive, or hold an "
          "X/Y/Z +/- button to jog. Release to stop. Ctrl-C to quit.")

    jog_step_m = JOG_SPEED_MM_S / 1000.0 / FPS  # per-frame jog distance
    jog_axis = {
        "xPlus": (0, +1), "xMinus": (0, -1),
        "yPlus": (1, +1), "yMinus": (1, -1),
        "zPlus": (2, +1), "zMinus": (2, -1),
    }

    # Persistent commanded TCP position (metres). Both the phone pose
    # (absolute, from a latch) and the jog buttons (incremental) drive it.
    cmd_pos: np.ndarray | None = None
    ref_pos: np.ndarray | None = None  # phone latch reference
    consecutive_clears = 0

    try:
        while True:
            t0 = time.perf_counter()

            try:
                obs = robot.get_observation()
                ##### work here #####
                cur_ee = np.array([obs[k] for k in EE_NAMES], dtype=np.float64)
                if cmd_pos is None:
                    cmd_pos = cur_ee[:3].copy()

                raw = teleop.get_action()
                mapped = mapper.action(dict(raw)) if raw else None
                enabled = bool(mapped["enabled"]) if mapped else False

                if enabled:
                    # Phone pose: absolute position from the rising-edge latch.
                    if ref_pos is None:
                        ref_pos = cmd_pos.copy()
                    cmd_pos[0] = ref_pos[0] + POSITION_SCALE * float(mapped["target_x"])
                    cmd_pos[1] = ref_pos[1] + POSITION_SCALE * float(mapped["target_y"])
                    cmd_pos[2] = ref_pos[2] + POSITION_SCALE * float(mapped["target_z"])
                else:
                    ref_pos = None
                    # Jog buttons: incremental base-frame nudges.
                    jog = read_jog(teleop)
                    for key, (axis, sign) in jog_axis.items():
                        if jog.get(key):
                            cmd_pos[axis] += sign * jog_step_m

                # Leash: never let the command run away from the real arm.
                cmd_pos = np.clip(
                    cmd_pos, cur_ee[:3] - CMD_LEASH_M, cur_ee[:3] + CMD_LEASH_M
                )

                # Orientation always follows the measured attitude (position
                # control only); overwrite just the x/y/z of the current pose.
                target = cur_ee.copy()
                target[:3] = cmd_pos
                robot.send_action(
                    {k: float(v) for k, v in zip(EE_NAMES, target)}
                )
                consecutive_clears = 0
            except RuntimeError as exc:
                # Only a kinematics EMS (IK failure) is recoverable; a real
                # safety stop re-raises via try_clear_kinematics_estop
                # returning False.
                if not AUTO_CLEAR_KINEMATICS_EMS:
                    raise
                consecutive_clears += 1
                if consecutive_clears > MAX_CONSECUTIVE_CLEARS:
                    print(
                        f"\nGave up after {MAX_CONSECUTIVE_CLEARS} auto-clears "
                        "— the arm is likely stuck at a singularity. Jog it "
                        "to a bent-elbow pose from the pendant, then restart."
                    )
                    raise
                if not robot.try_clear_kinematics_estop():
                    raise  # real safety event: abort
                print(
                    "Kinematics EMS cleared. Release the button and re-engage "
                    "to resume; move back toward the centre of the workspace."
                )
                ref_pos = None
                cmd_pos = None  # re-seed from the measured pose next frame
                _ = exc  # (message already surfaced by the driver log)

            _sleep_to_rate(t0)
    except KeyboardInterrupt:
        print("\nStopping teleop.")
    finally:
        robot.disconnect()
        teleop.disconnect()


def _sleep_to_rate(t0: float) -> None:
    dt = 1.0 / FPS - (time.perf_counter() - t0)
    if dt > 0:
        time.sleep(dt)


if __name__ == "__main__":
    main()
