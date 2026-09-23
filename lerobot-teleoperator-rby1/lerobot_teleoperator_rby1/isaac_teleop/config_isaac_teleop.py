"""Config dataclasses for the Isaac Teleop devices.

Adapted from the upstream LeRobot example
``examples/isaac_teleop_to_so101/isaac_teleop/config_isaac_teleop.py``.

Unlike upstream, :class:`IsaacTeleopConfig` does **not** shadow the draccus
``_choice_registry``: :class:`Rby1XRConfig` registers ``rby1_isaac`` in the
global :class:`TeleoperatorConfig` registry, so the device is selectable with
``--teleop.type=rby1_isaac`` from the stock ``lerobot-teleoperate`` and
``lerobot-record`` CLIs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig

# Static rebase from the OpenXR anchor frame (X=Right, Y=Up, Z=Backward) into the
# robot base frame (X=Forward, Y=Left, Z=Up). A proper rotation (det=+1):
# controller motion forward -> robot +X, right -> robot -Y, up -> robot +Z.
DEFAULT_BASE_T_ANCHOR: list[list[float]] = [
    [0.0, 0.0, -1.0, 0.0],
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]

SESSION_START_CHOICES = ("connect", "first_action")
LATCH_ORIENTATION_CHOICES = ("measured", "commanded")
TORSO_SOURCE_CHOICES = ("body", "head", "none")
TORSO_ENGAGE_CHOICES = ("both_arms", "any_arm", "always")
ARM_MODE_CHOICES = ("ee_clutch", "ee_absolute")
ARM_LENGTH_SOURCE_CHOICES = ("body", "config")
HINT_WRIST_SOURCE_CHOICES = ("controller", "body")
ROBOT_VERSION_CHOICES = ("auto", "1.2", "1.3")
WEAR_MODE_CHOICES = ("head", "neck")
SHOULDER_SOURCE_CHOICES = ("auto", "body", "headset")
VIZ_LOCK_CHOICES = ("gimbal", "head", "world")


@dataclass(kw_only=True)
class IsaacTeleopConfig(TeleoperatorConfig):
    """Shared config for all Isaac Teleop-backed teleoperators."""

    # Session name shown by the CloudXR runtime / OpenXR.
    app_name: str = "rby1_isaac"

    # Launch the CloudXR runtime from this process. Set False (or export
    # LEROBOT_CLOUDXR_SKIP_AUTOLAUNCH=1) when the runtime is already running.
    auto_launch_cloudxr: bool = True
    # KEY=value profile passed to CloudXRLauncher(env_config=...). None uses the
    # packaged ``default.env`` (Quest3 profile).
    cloudxr_env_file: str | None = None
    # Where CloudXRLauncher installs / finds the runtime.
    cloudxr_install_dir: str = "~/.cloudxr"
    # CloudXR device profile ("Quest3" also serves Pico / WebXR clients).
    cloudxr_device_profile: str = "Quest3"

    # Static anchor -> robot-base rebase applied in-graph to every tracked pose.
    base_T_anchor: list[list[float]] = field(  # noqa: N815
        default_factory=lambda: [row.copy() for row in DEFAULT_BASE_T_ANCHOR]
    )


@TeleoperatorConfig.register_subclass("rby1_isaac")
@dataclass(kw_only=True)
class Rby1XRConfig(IsaacTeleopConfig):
    """RB-Y1 XR device: both controllers -> arm EE poses, headset -> head
    joints, body tracking -> torso EE pose.

    This teleoperator only *produces* actions; execution and all Cartesian
    impedance tuning live in the follower ``Rby1Config`` (``action_mode="ee"``).
    The ``use_*`` flags below must mirror the follower's so the action keys
    match.
    """

    # ── Read-only robot link (state + forward kinematics for the clutch) ──
    robot_address: str = "192.168.30.1:50051"
    robot_model: str = "m"  # "a" | "m" | "ub"

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

    # ── Headset camera panels (Televiz) ──────────────────────────────
    # One quad per robot camera name (as configured in --robot.cameras); the
    # frames come from the in-process frame bus. Televiz then owns the OpenXR
    # session (the tracking session attaches to it).
    viz_enabled: bool = False
    viz_cameras: list[str] = field(default_factory=lambda: ["front", "left", "right"])
    viz_offsets_x: list[float] = field(default_factory=lambda: [0.0, -1.1, 1.1])  # m, per camera
    viz_offset_y: float = 0.0
    viz_distance_m: float = 1.5
    viz_width_m: float = 1.0
    viz_lock_mode: str = "gimbal"         # "gimbal" (position + yaw) | "head" | "world"
    viz_openxr_composition: bool = False  # keep False on Jetson Orin (black quads otherwise)
    viz_wait_headset_s: int = -1          # VizSession.create waits for the headset (-1 = forever)

    # ── Arm mapping mode ──────────────────────────────────────────────
    # "ee_clutch":   squeeze latches a clutch; the arm follows the controller
    #                DELTA from that moment (re-anchorable, no calibration).
    # "ee_absolute": the hand position RELATIVE TO THE OPERATOR'S SHOULDER
    #                (IOBT body tracking) is scaled by the robot/human reach
    #                ratio onto the robot shoulder; the orientation is the
    #                controller orientation times an offset latched on Right A.
    #                Squeeze is a dead-man switch (follow while held, hold
    #                when released); (re-)engaging ramps to the absolute
    #                target over `engage_ramp_s`. Needs body tracking.
    arm_mode: str = "ee_clutch"
    engage_ramp_s: float = 2.0
    ee_position_scale: float = 1.0          # extra multiplier on the reach ratio
    ee_reach_max_ratio: float = 0.98        # clamp |hand - shoulder| to this × robot reach
    arm_length_source: str = "body"         # "body" (IOBT |S-E|+|E-W|) | "config"
    human_arm_length_m: float = 0.62        # used when arm_length_source="config"
    shoulder_smoothing: float = 0.2         # EMA on the IOBT shoulder position
    ee_max_linear_vel: float = 1.0          # m/s rate limit of the absolute target
    ee_max_angular_vel: float = 3.0         # rad/s
    ee_orientation_latch_on_a: bool = True  # Right A: controller orientation ↦ measured EE
    ee_orientation_offset_rpy_deg: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    abs_hold_s: float = 1.0                 # keep following this long without body joints
    # Robot version selects the reach constants ("auto" = probe the robot).
    robot_version: str = "auto"

    # ── IOBT arm posture → nullspace hint (both arm modes) ────────────
    # Shoulder / elbow / hand positions are retargeted to arm_0..arm_3 and sent
    # as `<side>_arm_<i>.null` keys; the follower uses them as the Cartesian
    # solver's nullspace target (soft, EE has priority). Not recorded unless
    # record_posture_hint (then the dataset action gains 8 dims).
    arm_posture_hint: bool = False
    record_posture_hint: bool = False
    posture_hint_smoothing: float = 0.3
    posture_hint_max_vel: float = 2.0       # rad/s per joint
    posture_hint_hold_s: float = 1.0
    hint_wrist_source: str = "controller"   # "controller" (grip position) | "body" (IOBT wrist)

    # ── Clutch ────────────────────────────────────────────────────────
    # Squeeze value above which an arm follows its controller (dead-man
    # switch in ee_absolute mode).
    clutch_threshold: float = 0.5
    # "measured": on engage, latch both home position AND orientation from the
    # measured EE pose (7-DOF arms track orientation, so no offset builds up).
    # "commanded": upstream SO-101 behaviour (orientation from last command).
    latch_orientation: str = "measured"
    # When to open the CloudXR/OpenXR session: at connect() (default) or lazily
    # on the first get_action(). "first_action" is a mitigation for Jetson Orin
    # hosts where creating Python threads after the CloudXR service starts can
    # abort the process: lerobot-record creates its keyboard-listener and
    # camera threads between teleop.connect() and the first get_action().
    session_start: str = "connect"
    # Seconds to wait for the headset controllers before giving up (0 = forever).
    tracking_wait_timeout_s: float = 0.0

    # ── Thumbsticks -> mobile base (body-frame velocity) ─────────────
    thumbstick_deadzone: float = 0.15
    base_max_linear: float = 0.3   # m/s at full deflection
    base_max_angular: float = 0.6  # rad/s at full deflection

    # ── Headset orientation -> head_0 (yaw) / head_1 (pitch) ─────────
    head_yaw_sign: float = 1.0      # +head_0 = look left; flip if reversed on HW
    head_pitch_sign: float = -1.0   # +head_1 = look down (READY_HEAD pitch = +49 deg)
    head_yaw_gain: float = 1.0
    head_pitch_gain: float = 1.0
    head_yaw_limit_deg: float = 80.0
    head_pitch_min_deg: float = -45.0
    head_pitch_max_deg: float = 80.0
    head_smoothing: float = 0.3     # EMA weight of the new sample (1.0 = none)

    # ── Torso driver ──────────────────────────────────────────────────
    # "body": Isaac Teleop FullBodySource (Quest 3 IOBT / Pico trackers),
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
    torso_rot_scale: float = 1.0
    torso_z_scale: float = 1.0
    torso_use_xy: bool = True
    # Safety clamps on the delta from the torso pose latched at engage.
    torso_max_rot_delta_deg: float = 90.0
    torso_max_z_delta_m: float = 0.15

    # ── Right A: return to the start pose ─────────────────────────────
    # The pose measured on the first action (the follower has just reached
    # its ready pose) is remembered; Right A releases every clutch and moves
    # arms, torso and head back to it over this many seconds (base excluded).
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

    # ── Diagnostics ───────────────────────────────────────────────────
    # Log a one-line tracking / clutch status this often (seconds; 0 = off).
    status_log_period_s: float = 5.0

    def __post_init__(self) -> None:
        parent_post_init = getattr(super(), "__post_init__", None)
        if parent_post_init is not None:
            parent_post_init()
        if self.session_start not in SESSION_START_CHOICES:
            raise ValueError(
                f"session_start must be one of {SESSION_START_CHOICES}, got {self.session_start!r}"
            )
        if self.latch_orientation not in LATCH_ORIENTATION_CHOICES:
            raise ValueError(
                f"latch_orientation must be one of {LATCH_ORIENTATION_CHOICES}, "
                f"got {self.latch_orientation!r}"
            )
        if self.torso_source not in TORSO_SOURCE_CHOICES:
            raise ValueError(
                f"torso_source must be one of {TORSO_SOURCE_CHOICES}, got {self.torso_source!r}"
            )
        if self.torso_engage not in TORSO_ENGAGE_CHOICES:
            raise ValueError(
                f"torso_engage must be one of {TORSO_ENGAGE_CHOICES}, got {self.torso_engage!r}"
            )
        for name, value, choices in (
            ("wear_mode", self.wear_mode, WEAR_MODE_CHOICES),
            ("shoulder_source", self.shoulder_source, SHOULDER_SOURCE_CHOICES),
            ("viz_lock_mode", self.viz_lock_mode, VIZ_LOCK_CHOICES),
            ("arm_mode", self.arm_mode, ARM_MODE_CHOICES),
            ("arm_length_source", self.arm_length_source, ARM_LENGTH_SOURCE_CHOICES),
            ("hint_wrist_source", self.hint_wrist_source, HINT_WRIST_SOURCE_CHOICES),
            ("robot_version", self.robot_version, ROBOT_VERSION_CHOICES),
        ):
            if value not in choices:
                raise ValueError(f"{name} must be one of {choices}, got {value!r}")
        if self.record_posture_hint and not self.arm_posture_hint:
            raise ValueError("record_posture_hint requires arm_posture_hint=True")
        if len(self.ee_orientation_offset_rpy_deg) != 3:
            raise ValueError("ee_orientation_offset_rpy_deg must have 3 values")
        if len(self.neck_shoulder_offset) != 3:
            raise ValueError("neck_shoulder_offset must have 3 values")
        if self.viz_enabled and len(self.viz_offsets_x) != len(self.viz_cameras):
            raise ValueError("viz_offsets_x must have one entry per viz_cameras entry")
        if self.robot_model.strip().lower() not in ("a", "m", "ub"):
            raise ValueError(f'robot_model must be "a", "m" or "ub", got {self.robot_model!r}')
        if not (0.0 < self.head_smoothing <= 1.0):
            raise ValueError("head_smoothing must be in (0, 1]")
        from .xr_frame import BodyJointIndex  # local import: keep this module lerobot-only

        valid_joints = {m.name for m in BodyJointIndex}
        for name in [self.torso_body_joint, *self.body_required_joints]:
            if name not in valid_joints:
                raise ValueError(
                    f"Unknown body joint {name!r}; expected one of {sorted(valid_joints)}"
                )

    @property
    def needs_body(self) -> bool:
        """Whether the FullBodySource must be in the pipeline."""
        return self.torso_source == "body" or self.arm_mode == "ee_absolute" or self.arm_posture_hint
