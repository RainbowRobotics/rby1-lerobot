"""Behaviour settings of the XR teleoperation core.

A plain dataclass: no robot-learning framework is needed to use it. The
LeRobot device ``rby1_isaac`` inherits these fields into its ``Rby1XRConfig``
so every one of them is a ``--teleop.<field>`` option there.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .xr_frame import BodyJointIndex

TORSO_SOURCE_CHOICES = ("body", "head", "none")
TORSO_ENGAGE_CHOICES = ("both_arms", "any_arm", "always")
ARM_LENGTH_SOURCE_CHOICES = ("calibrate", "body", "config")
HINT_WRIST_SOURCE_CHOICES = ("controller", "body")
WEAR_MODE_CHOICES = ("head", "neck")
HEAD_GAZE_FRAME_CHOICES = ("base", "torso")
REFERENCE_SOURCE_CHOICES = ("shoulders", "gaze", "auto")
SHOULDER_SOURCE_CHOICES = ("auto", "body", "headset")


@dataclass(kw_only=True)
class XrTeleopSettings:
    """How the operator's motion becomes robot targets (see each field)."""

    # ── Joint-group selection (mirror Rby1Config) ─────────────────────
    use_torso: bool = True
    use_right_arm: bool = True
    use_left_arm: bool = True
    use_gripper: bool = True
    use_mobile_base: bool = True
    use_head: bool = True

    # ── Wear mode ─────────────────────────────────────────────────────
    # "head": headset worn normally (head joints follow the gaze, torso from
    #         body tracking / head, operator frame from the shoulder line).
    # "neck": headset hanging from the neck. The robot head is held at the
    #         start (ready) pose, the HEADSET pose drives the torso, the
    #         operator frame is taken from the shoulder line or, when body
    #         tracking is unavailable, from the headset→controllers direction,
    #         and the absolute-EE shoulder is estimated from the headset pose.
    #         Requires the Quest proximity sensor to be disabled (MQDH Device
    #         Actions → Proximity Sensor off, or tape) so the session stays
    #         FOCUSED and the controllers keep streaming.
    wear_mode: str = "head"
    neck_torso_smoothing: float = 0.3     # EMA on the headset pose driving the torso
    neck_shoulder_offset: list[float] = field(default_factory=lambda: [-0.05, 0.20, -0.15])  # from the headset, left side (y mirrored for right)
    shoulder_source: str = "auto"         # absolute mode: "body" | "headset" | "auto" (body when valid)

    # ── Arm mapping (absolute) ────────────────────────────────────────
    # The hand position RELATIVE TO THE OPERATOR'S SHOULDER (IOBT body
    # tracking) is scaled by the robot/human reach ratio onto the robot
    # shoulder; the orientation is the controller orientation times an offset
    # latched on Right A. Squeeze is a dead-man switch (follow while held,
    # hold when released); (re-)engaging ramps to the absolute target over
    # `engage_ramp_s`. Needs body tracking.
    engage_ramp_s: float = 2.0
    ee_position_scale: float = 1.0          # extra multiplier on the reach ratio
    ee_reach_max_ratio: float = 0.98        # clamp |hand - shoulder| to this × robot reach
    # Operator reach the hand-to-shoulder vector is scaled by (robot reach / this):
    #   "calibrate": measured when Right A is pressed with both arms held straight
    #                out in front — the shoulder -> controller distance, fixed for
    #                the session (needs start_on_right_a);
    #   "body":      IOBT |S-E|+|E-W|, updated while running;
    #   "config":    human_arm_length_m.
    arm_length_source: str = "calibrate"
    # Shoulder -> controller distance (m): the fixed value for "config", and the
    # value used until a reach is measured / estimated in the other modes.
    human_arm_length_m: float = 0.62
    # "calibrate": the reach is the median of the samples of the last
    # arm_calibration_window_s before the press. A reach outside [min, max] is
    # rejected (arm not extended, tracking off). When the two arms differ by
    # more than arm_calibration_max_asymmetry (fraction), the longer one is
    # used for both: an arm that is not fully extended only measures short.
    arm_calibration_window_s: float = 0.5
    arm_calibration_min_m: float = 0.45
    arm_calibration_max_m: float = 0.85
    arm_calibration_max_asymmetry: float = 0.10
    shoulder_smoothing: float = 0.2         # EMA on the IOBT shoulder position
    ee_max_linear_vel: float = 1.0          # m/s rate limit of the absolute target
    ee_max_angular_vel: float = 3.0         # rad/s
    # Offsets between the controller and the gripper, latched at the FIRST
    # SQUEEZE after Right A (and after connect) — not at the button press, so
    # the hand posture used to press A does not leak into the mapping:
    #   orientation: controller orientation then ↦ start-pose gripper orientation
    #   position:    hand-relative-to-shoulder then ↦ start-pose gripper position
    # so the arm does not move when you squeeze without moving your hand. The
    # mapping stays absolute (constant offsets) so the IOBT elbow hint agrees.
    ee_orientation_latch_on_a: bool = True
    ee_position_latch_on_a: bool = True
    # Right A (and the first action / a re-sync): the headset look direction
    # of that moment ↦ the start-pose head joints, exactly like the EE
    # orientation offsets. False = pure absolute mapping with the fixed
    # head_*_offset_deg (the head jumps to the gaze after Right A).
    head_latch_on_a: bool = True
    ee_orientation_offset_rpy_deg: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    abs_hold_s: float = 1.0                 # keep following this long without body joints

    # ── IOBT arm posture → nullspace hint ─────────────────────────────
    # Shoulder / elbow / hand positions are retargeted to arm_0..arm_3 and sent
    # as `<side>_arm_<i>.null` keys; the follower uses them as the Cartesian
    # solver's nullspace target (soft, EE has priority). Not recorded unless
    # record_posture_hint (then the dataset action gains 8 dims).
    arm_posture_hint: bool = True
    record_posture_hint: bool = False
    posture_hint_smoothing: float = 0.3
    posture_hint_max_vel: float = 2.0       # rad/s per joint
    posture_hint_hold_s: float = 1.0
    hint_wrist_source: str = "controller"   # "controller" (grip position) | "body" (IOBT wrist)

    # ── Operator frame (which direction is robot +X) ──────────────────
    # "shoulders": the body-tracking shoulder line, at the first action and on
    #              Right A alike (the arms hold until both shoulders are valid;
    #              on Right A without valid shoulders the gaze is used with a
    #              warning). Consistent between the two reference events.
    # "gaze":      the headset look direction, both times.
    # "auto":      shoulder line when valid, otherwise gaze (legacy).
    # Neck mode: after the shoulder line, the headset→controllers direction is
    # tried before the gaze (the head is not looking along the body).
    reference_source: str = "shoulders"

    # ── Dead-man switch / session ─────────────────────────────────────
    # Squeeze value above which an arm follows its controller.
    clutch_threshold: float = 0.5
    # True: the head and the mobile base (thumbsticks and body drive) only move
    # while at least one grip is squeezed past clutch_threshold; with both
    # released they hold, like the arms. Squeezing again maps the look
    # direction of that moment onto the head joints as they are (with
    # head_latch_on_a), so the head does not jump.
    head_base_require_grip: bool = True
    # True: nothing follows the headset (arms, torso, head, grippers, base)
    # until Right A is pressed for the first time. That first press only starts
    # teleoperation (and references the operator frame); it does not run the
    # return-to-start motion later presses do. False: live as soon as the
    # controllers are tracked.
    start_on_right_a: bool = True

    # ── Thumbsticks -> mobile base (body-frame velocity) ─────────────
    thumbstick_deadzone: float = 0.15
    base_max_linear: float = 0.3   # m/s at full deflection
    base_max_angular: float = 0.6  # rad/s at full deflection

    # ── Body drive: hold Left X -> the base follows the operator 1:1 ──
    # While Left X is held the torso target is frozen and the base follows the
    # operator's own motion since the press: the headset's horizontal
    # displacement and the turn of the shoulder line, against the robot's
    # odometry (proportional position control, velocity commands). The
    # operator frame turns with the body meanwhile, so the head command is the
    # neck angle and the arms stay as they are relative to the torso. The
    # thumbsticks are ignored while it is active; releasing Left X stops the base.
    body_drive_enabled: bool = True
    body_drive_kp_linear: float = 2.0       # (m/s) per m of position error
    body_drive_kp_angular: float = 2.0      # (rad/s) per rad of heading error
    body_drive_max_linear: float = 0.4      # m/s
    body_drive_max_angular: float = 0.8     # rad/s
    body_drive_deadband_m: float = 0.01
    body_drive_deadband_deg: float = 1.0
    body_drive_smoothing: float = 0.3       # EMA weight of the new operator sample (1.0 = none)
    # Leave the mode (until Left X is pressed again) when the base is this far
    # behind: a tracking jump, or a base that cannot follow.
    body_drive_max_error_m: float = 0.6
    body_drive_max_error_deg: float = 60.0

    # ── Headset orientation -> head_0 (yaw) / head_1 (pitch) ─────────
    # The gaze follows the headset yaw / pitch measured in the operator frame
    # (Right A sets which direction is straight ahead and latches the look
    # direction onto the start head joints; without the latch, looking
    # straight ahead gives head_*_offset_deg).
    head_yaw_offset_deg: float = 0.0
    head_pitch_offset_deg: float = 0.0
    head_yaw_sign: float = 1.0      # +head_0 = look left; flip if reversed on HW
    head_pitch_sign: float = -1.0   # +head_1 = look down (READY_HEAD pitch = +49 deg)
    head_yaw_gain: float = 1.0
    head_pitch_gain: float = 1.0
    head_yaw_limit_deg: float = 80.0
    head_pitch_min_deg: float = -45.0
    head_pitch_max_deg: float = 80.0
    head_smoothing: float = 0.3     # EMA weight of the new sample (1.0 = none)
    # "base": the head target is a look direction in the robot base frame and
    # the joints are re-solved for the measured torso orientation every tick —
    # the torso bending / turning does not move the gaze (the headset did not
    # move). "torso": joints commanded directly (gaze rides with the torso).
    head_gaze_frame: str = "base"

    # ── Torso driver ──────────────────────────────────────────────────
    # "body": Isaac Capture FullBodySource (Quest 3 IOBT / Pico trackers),
    # "head": headset pose, "none": torso target frozen at its start pose.
    torso_source: str = "body"
    # When the torso follows its driver: only while BOTH enabled arms are
    # clutched (default, the operator has to "hold on" with both hands), while
    # ANY arm is clutched, or ALWAYS (from the first valid driver frame on).
    torso_engage: str = "both_arms"
    # Body joint driving the torso (name from BodyJointIndex, XR_BD layout).
    torso_body_joint: str = "SPINE3"
    # Joints that must be valid for a body frame to drive the torso.
    body_required_joints: list[str] = field(
        default_factory=lambda: ["PELVIS", "SPINE3", "NECK"]
    )
    # Absolute workspace box of the torso target (link_torso_5 in the robot
    # base frame): position [x, y, z] (m) and fixed-axis XYZ roll / pitch /
    # yaw (deg, pitch + = leaning forward). Derived from the reachable range
    # computed by scripts/torso_workspace.py (ready pose: x = y = 0,
    # z = 1.10 m, rpy = 0) with margins that keep the chest above the knees
    # and rule out folded postures. The chest pose delta since engage is
    # applied 1:1 (no extra delta clamps); this box is the only limit.
    torso_pos_min: list[float] = field(default_factory=lambda: [-0.15, -0.20, 0.80])
    torso_pos_max: list[float] = field(default_factory=lambda: [0.45, 0.20, 1.20])
    torso_rpy_min_deg: list[float] = field(default_factory=lambda: [-15.0, -20.0, -45.0])
    torso_rpy_max_deg: list[float] = field(default_factory=lambda: [15.0, 50.0, 45.0])

    # ── Right A: return to the start pose ─────────────────────────────
    # The pose measured on the first action (the follower has just reached
    # its ready pose) is remembered; Right A releases every clutch and sends
    # one `reset.cmd` action: the follower runs its joint-position ready-pose
    # motion (as at connect) with this minimum time, blocking for its
    # duration, then re-references the EE solver. Base excluded.
    ready_return_duration_s: float = 4.0

    # ── Re-synchronisation with the robot ─────────────────────────────
    # The targets are re-seeded from the measured robot pose on the first
    # get_action() (the follower moves to its ready pose AFTER teleop.connect())
    # and whenever a component that is NOT clutched is found further than these
    # thresholds from its held target (record reset, manual move): the robot
    # then stays where it is instead of being dragged back to a stale target.
    resync_position_threshold_m: float = 0.03
    resync_rotation_threshold_deg: float = 10.0
    resync_head_threshold_deg: float = 5.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise ``ValueError`` for an inconsistent setting."""
        for name, value, choices in (
            ("torso_source", self.torso_source, TORSO_SOURCE_CHOICES),
            ("torso_engage", self.torso_engage, TORSO_ENGAGE_CHOICES),
            ("wear_mode", self.wear_mode, WEAR_MODE_CHOICES),
            ("head_gaze_frame", self.head_gaze_frame, HEAD_GAZE_FRAME_CHOICES),
            ("reference_source", self.reference_source, REFERENCE_SOURCE_CHOICES),
            ("shoulder_source", self.shoulder_source, SHOULDER_SOURCE_CHOICES),
            ("arm_length_source", self.arm_length_source, ARM_LENGTH_SOURCE_CHOICES),
            ("hint_wrist_source", self.hint_wrist_source, HINT_WRIST_SOURCE_CHOICES),
        ):
            if value not in choices:
                raise ValueError(f"{name} must be one of {choices}, got {value!r}")
        if self.arm_length_source == "calibrate" and not self.start_on_right_a:
            raise ValueError(
                'arm_length_source="calibrate" measures the reach on Right A and needs '
                'start_on_right_a=True (use "body" or "config" otherwise)'
            )
        if not 0.0 < self.arm_calibration_min_m < self.arm_calibration_max_m:
            raise ValueError("arm_calibration_min_m must be positive and below arm_calibration_max_m")
        if self.arm_calibration_window_s <= 0.0 or self.arm_calibration_max_asymmetry < 0.0:
            raise ValueError("arm_calibration_window_s must be > 0 and arm_calibration_max_asymmetry >= 0")
        for name in (
            "body_drive_kp_linear", "body_drive_kp_angular", "body_drive_max_linear",
            "body_drive_max_angular", "body_drive_max_error_m", "body_drive_max_error_deg",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be > 0")
        if self.body_drive_deadband_m < 0.0 or self.body_drive_deadband_deg < 0.0:
            raise ValueError("body_drive_deadband_m / _deg must be >= 0")
        if not 0.0 < self.body_drive_smoothing <= 1.0:
            raise ValueError("body_drive_smoothing must be in (0, 1]")
        if self.record_posture_hint and not self.arm_posture_hint:
            raise ValueError("record_posture_hint requires arm_posture_hint=True")
        if len(self.ee_orientation_offset_rpy_deg) != 3:
            raise ValueError("ee_orientation_offset_rpy_deg must have 3 values")
        if len(self.neck_shoulder_offset) != 3:
            raise ValueError("neck_shoulder_offset must have 3 values")
        for lo_name, hi_name in (("torso_pos_min", "torso_pos_max"), ("torso_rpy_min_deg", "torso_rpy_max_deg")):
            lo, hi = getattr(self, lo_name), getattr(self, hi_name)
            if len(lo) != 3 or len(hi) != 3:
                raise ValueError(f"{lo_name} / {hi_name} must have 3 values")
            if any(a > b for a, b in zip(lo, hi)):
                raise ValueError(f"{lo_name} must be <= {hi_name} component-wise")
        if any(abs(v) >= 180.0 for v in self.torso_rpy_min_deg + self.torso_rpy_max_deg):
            raise ValueError("torso_rpy_*_deg must be within (-180, 180)")
        if not (0.0 < self.head_smoothing <= 1.0):
            raise ValueError("head_smoothing must be in (0, 1]")
        valid_joints = {m.name for m in BodyJointIndex}
        for name in [self.torso_body_joint, *self.body_required_joints]:
            if name not in valid_joints:
                raise ValueError(
                    f"Unknown body joint {name!r}; expected one of {sorted(valid_joints)}"
                )
