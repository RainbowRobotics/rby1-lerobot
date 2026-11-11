"""User-tunable configuration of the RB-Y1 follower robot.

The fields are grouped by the controller they feed:

====================  =====================================  ==========================
Section               Controller (rby1-sdk command)          Active when
====================  =====================================  ==========================
Joint mode            JointPosition / JointImpedance         ``action_mode="joint"``
Joint impedance       JointImpedance                         ``use_joint_impedance=True``
EE mode               CartesianImpedance (per group solver)  ``action_mode="ee"``
Mobile base           SE2Velocity                            ``use_mobile_base=True``
====================  =====================================  ==========================

Fields of a section that is not active are ignored.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig

from .constants import (  # noqa: F401  (ready poses are re-exported for callers)
    ARM_DOF,
    ARM_TARGET_GAINS,
    DEFAULT_LEFT_ARM_JOINT_LIMITS,
    DEFAULT_NULL_LEFT_DEG,
    DEFAULT_NULL_RIGHT_DEG,
    DEFAULT_RIGHT_ARM_JOINT_LIMITS,
    DEFAULT_TORSO_JOINT_LIMITS,
    READY_HEAD,
    READY_LEFT,
    READY_LEFT_V13,
    READY_POSE,
    READY_POSE_V13,
    READY_RIGHT,
    READY_RIGHT_V13,
    READY_TORSO,
    TORSO_DOF,
    TORSO_TARGET_GAINS,
    JointLimits,
    ready_pose_for_version,
)


@RobotConfig.register_subclass("rby1")
@dataclass
class Rby1Config(RobotConfig):
    # ════════════════════════════════════════════════════════════════════
    # Connection
    # ════════════════════════════════════════════════════════════════════

    # gRPC address of the robot (host:port)
    address: str = "192.168.30.1:50051"

    # Robot model variant: "a" (24-DOF), "m" (26-DOF mecanum), or "auto" to
    # detect it from the robot via rby1-sdk on connect(). The upper-body "ub"
    # model is not supported: the ready pose, torso keys and EE solver gains
    # assume the 6-DOF torso of Model A/M.
    model: str = "auto"

    # Firmware version: "auto" to detect via rby1-sdk, or one of "1.0"/"1.1"/
    # "1.2"/"1.3". The version selects the ready pose set (v1.3 differs from
    # earlier versions near the wrist).
    version: str = "auto"

    # Timeout (seconds) for the model/version auto-detection probe connection.
    connect_timeout_sec: float = 3.0

    # Move robot to ready pose automatically when connect() is called.
    # Set False to skip the ready pose motion (e.g. during unit tests or
    # when the robot is already in a safe position).
    move_to_ready_on_connect: bool = True

    # ════════════════════════════════════════════════════════════════════
    # Features — what the observation and the action contain
    # ════════════════════════════════════════════════════════════════════

    # "joint": actions are joint positions (radians) for the enabled groups,
    #          executed as joint position / joint impedance commands.
    # "ee":    actions are end-effector poses in the base frame
    #          (torso_ee.* / right_ee.* / left_ee.* — position in metres plus
    #          a rotation vector in radians), executed via the onboard
    #          Cartesian impedance solver. Produced e.g. by the rby1_vr
    #          teleoperator.
    action_mode: str = "joint"

    # Joint groups included in the observation and the action. A disabled
    # group is not commanded and holds its position. With use_torso=True the
    # torso is commanded in both action modes (torso_0..5.pos in joint mode,
    # torso_ee.* in EE mode), so the teleoperator or policy must provide it.
    use_torso: bool = False
    use_right_arm: bool = True
    use_left_arm: bool = True

    # Enable physical Dynamixel gripper (set False for simulation / gripper-less setups)
    use_gripper: bool = True

    # Enable the omnidirectional mobile base. When True, send_action consumes
    # the base velocity action keys (x.vel, y.vel, theta.vel) and forwards them
    # to the robot as an SE2 velocity command.
    use_mobile_base: bool = False

    # Include joint velocities / torques in observation_features
    use_velocity: bool = False
    use_torque: bool = False

    # Map of camera name -> CameraConfig. Defaults to no cameras; see the
    # README for a RealSense example.
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # ════════════════════════════════════════════════════════════════════
    # Joint mode (action_mode="joint") — JointPosition / JointImpedance
    # ════════════════════════════════════════════════════════════════════

    # Each command carries a minimum_time (s): the robot takes at least this
    # long to reach the target. At the beginning of a session it is ramped
    # linearly from joint_startup_min_time down to joint_min_time over
    # joint_startup_ramp_duration seconds, so the follower converges smoothly
    # to the first target instead of snapping to it. Set the duration to 0.0
    # to disable the ramp.
    # joint_min_time is also the minimum_time of the mobile-base velocity
    # command, in both action modes.
    joint_startup_ramp_duration: float = 5.0
    joint_startup_min_time: float = 5.0
    joint_min_time: float = 0.07

    # Velocity / acceleration limits are read from the URDF dynamics model at
    # connect() time. This factor multiplies ALL joints' URDF acceleration
    # limits: the raw values are very conservative; 30x is the SDK default
    # for real-time teleoperation responsiveness.
    joint_acceleration_limit_scale: float = 30.0

    # Low-pass cutoff (Hz) of the JointPosition controller, set on the robot
    # for the session by configure() (core parameter
    # joint_position_command.cutoff_frequency; restored on disconnect). It is
    # set in every mode, but only affects JointPosition commands: joint mode
    # without impedance, and the ready-pose motion.
    joint_position_cutoff_frequency: float = 5.0

    # ── JointImpedance (use_joint_impedance=True) ──────────────────────
    # When True, joint actions are sent as JointImpedance commands instead of
    # JointPosition, yielding compliant behaviour.
    use_joint_impedance: bool = False

    # Joint stiffness (Nm/rad) and torque limit (Nm). The arm values apply to
    # both arms; the torso values are used when use_torso=True.
    # Higher stiffness -> stiffer, lower -> more compliant.
    # The arm stiffness follows the inertia each joint moves, so it is larger
    # towards the base: arm_0..3 give a natural frequency of about 3.5 Hz,
    # arm_4..5 about 6 Hz and arm_6 about 5 Hz in the ready pose.
    # The arm is stiff and its compliance comes from the torque limit: the
    # limit bounds the impedance torque only (gravity compensation is added
    # on top), so it is the most a joint pushes back when it is held away
    # from its target. arm_4..6 need 8 Nm to follow the target at all (the
    # wrist roll does not move with 4 Nm). The torso torque limits are the
    # joint effort limits of the robot's URDF.
    joint_impedance_torso_stiffness: List[float] = field(default_factory=lambda: [400.0] * 6)
    joint_impedance_torso_torque_limit: List[float] = field(
        default_factory=lambda: [270.0, 270.0, 270.0, 120.0, 120.0, 120.0]
    )
    joint_impedance_arm_stiffness: List[float] = field(
        default_factory=lambda: [1075.0, 990.0, 580.0, 265.0, 190.0, 205.0, 300.0]
    )
    joint_impedance_arm_torque_limit: List[float] = field(
        default_factory=lambda: [30.0, 30.0, 25.0, 18.0, 8.0, 8.0, 8.0]
    )

    # Damping ratio zeta (> 0) applied to all joints. The RB-Y1 controller
    # computes D = zeta * (sqrt(M) sqrt(K) + sqrt(K) sqrt(M)) from the mass
    # matrix M and stiffness K, i.e. 2 * zeta * sqrt(m * k) per joint, so
    # 1.0 is critical damping (no overshoot); values below 1.0 respond faster
    # but overshoot (about 5 % at 0.7), values above 1.0 are overdamped.
    # The same convention applies to every *_damping_ratio below.
    joint_impedance_damping_ratio: float = 1.0

    # Multiply the wrist joint (last joint of each arm) velocity limit by
    # this factor. Only applied with use_joint_impedance=True, where the wrist
    # needs higher velocity headroom. (SDK example uses 10x.)
    joint_impedance_wrist_velocity_limit_scale: float = 10.0

    # ════════════════════════════════════════════════════════════════════
    # EE mode (action_mode="ee") — CartesianImpedance
    # The torso, right arm and left arm each run their own solver.
    # ════════════════════════════════════════════════════════════════════

    # The Cartesian commands use
    #   minimum_time = <action period> * ee_min_time_factor,
    # so the robot reaches each target just as the next one arrives.
    # With ee_auto_action_period=True the period is measured from the interval
    # between send_action calls (moving average; pauses such as episode saving
    # are ignored; kept within 0.01-0.2 s), so it follows --fps / the policy
    # rate. ee_action_period (s) is then only the starting value, used for the
    # first command and after a ready-pose motion. With False, ee_action_period
    # is used as a fixed period (the behaviour of plugin 0.3.0).
    ee_auto_action_period: bool = True
    ee_action_period: float = 1 / 15
    ee_min_time_factor: float = 1.02

    # Cartesian commands keep holding for this long (s) if the stream stalls.
    ee_hold_time: float = 1.0

    # When a new EE target jumps further than these thresholds from the
    # previous one (e.g. the teleoperator re-latched after the robot was
    # reset), set_reset_reference is sent so the solver re-references to
    # the robot's actual state instead of chasing the jump violently.
    ee_reset_position_threshold: float = 0.1   # metres
    ee_reset_rotation_threshold: float = 0.5   # radians

    # ── Torso solver ───────────────────────────────────────────────────
    ee_torso_stiffness: List[float] = field(default_factory=lambda: [1700.0] * 6)
    ee_torso_torque_limit: List[float] = field(default_factory=lambda: [600.0] * 6)
    ee_torso_damping_ratio: float = 1.0
    ee_torso_joint_limits: JointLimits = field(
        default_factory=lambda: dict(DEFAULT_TORSO_JOINT_LIMITS)
    )
    # add_target limits: (linear velocity m/s, angular velocity rad/s,
    # linear acceleration m/s^2, angular acceleration rad/s^2).
    ee_torso_target_gains: Tuple[float, float, float, float] = TORSO_TARGET_GAINS

    # ── Arm solvers (the same gains drive both arms) ───────────────────
    # Joint stiffness and torque limit: the same values as joint impedance
    # (joint_impedance_arm_stiffness / _torque_limit above).
    ee_arm_stiffness: List[float] = field(
        default_factory=lambda: [1075.0, 990.0, 580.0, 265.0, 190.0, 205.0, 300.0]
    )
    ee_arm_torque_limit: List[float] = field(
        default_factory=lambda: [30.0, 30.0, 25.0, 18.0, 8.0, 8.0, 8.0]
    )
    ee_arm_damping_ratio: float = 1.0
    ee_arm_target_gains: Tuple[float, float, float, float] = ARM_TARGET_GAINS
    ee_right_arm_joint_limits: JointLimits = field(
        default_factory=lambda: dict(DEFAULT_RIGHT_ARM_JOINT_LIMITS)
    )
    ee_left_arm_joint_limits: JointLimits = field(
        default_factory=lambda: dict(DEFAULT_LEFT_ARM_JOINT_LIMITS)
    )

    # ── Arm nullspace ──────────────────────────────────────────────────
    ee_null_right_arm_deg: List[float] = field(
        default_factory=lambda: list(DEFAULT_NULL_RIGHT_DEG)
    )
    ee_null_left_arm_deg: List[float] = field(
        default_factory=lambda: list(DEFAULT_NULL_LEFT_DEG)
    )
    ee_nullspace_weight: List[float] = field(
        default_factory=lambda: [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0]
    )
    ee_nullspace_stiffness: float = 0.2
    ee_nullspace_damping_ratio: float = 1.0

    # ════════════════════════════════════════════════════════════════════
    # Mobile base (use_mobile_base=True) — SE2Velocity
    # ════════════════════════════════════════════════════════════════════

    # The base keeps the last commanded velocity for this long (s) after the
    # action stream stops (e.g. while lerobot-record saves an episode), then
    # decelerates to a stop. Keep it short: it is how far the base coasts.
    mobile_base_hold_time: float = 0.5
    # Base velocity commands are clamped to these limits (m/s for the x/y
    # speed, rad/s for yaw). The defaults are the rby1_vr joystick maximum.
    mobile_base_max_linear_velocity: float = 0.6
    mobile_base_max_angular_velocity: float = 1.4

    # ════════════════════════════════════════════════════════════════════
    # Robot profile (dataset <-> inference consistency)
    # See lerobot_robot_rby1.profile and the README "Keeping robots consistent".
    # ════════════════════════════════════════════════════════════════════
    # profile_out : write the robot profile here on connect, e.g.
    #               <dataset_root>/meta/rby1_profile.json while recording.
    #               None -> ~/.cache/rby1_lerobot/profiles/<time>.json.
    # profile_in  : reference profile to compare against on connect: a file,
    #               a dataset root, or a Hugging Face dataset repo id.
    # profile_mode: "check" (log differences; fail on feature / firmware
    #               version mismatches), "apply" (also adopt the profile's
    #               behaviour settings and core parameters for this session),
    #               or "off".
    # profile_force: when profile_out already holds a different profile, keep
    #               it and write a time-stamped extra profile instead of failing.
    profile_out: str | None = None
    profile_in: str | None = None
    profile_mode: str = "check"
    profile_force: bool = False

    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.action_mode not in ("joint", "ee"):
            raise ValueError(
                f'Rby1Config.action_mode must be "joint" or "ee", got "{self.action_mode}".'
            )
        if self.profile_mode not in ("check", "apply", "off"):
            raise ValueError(
                f'Rby1Config.profile_mode must be "check", "apply" or "off", got "{self.profile_mode}".'
            )
        if self.model.strip().lower() not in ("a", "m", "auto"):
            raise ValueError(
                f'Rby1Config.model must be "a", "m" or "auto", got "{self.model}" '
                '(the upper-body "ub" model is not supported).'
            )
        if self.version.strip().lower() not in ("auto", "", "1.0", "1.1", "1.2", "1.3"):
            raise ValueError(
                'Rby1Config.version must be "auto" or one of "1.0"/"1.1"/"1.2"/"1.3", '
                f'got "{self.version}".'
            )
        for name in (
            # The controller rejects commands with a non-positive damping ratio.
            "joint_impedance_damping_ratio", "ee_torso_damping_ratio",
            "ee_arm_damping_ratio", "ee_nullspace_damping_ratio",
            "joint_position_cutoff_frequency", "ee_action_period", "ee_hold_time",
            "mobile_base_hold_time", "mobile_base_max_linear_velocity",
            "mobile_base_max_angular_velocity",
        ):
            if not getattr(self, name) > 0:
                raise ValueError(f"Rby1Config.{name} must be > 0, got {getattr(self, name)}.")
        for name, expected in (
            ("joint_impedance_torso_stiffness", TORSO_DOF),
            ("joint_impedance_torso_torque_limit", TORSO_DOF),
            ("joint_impedance_arm_stiffness", ARM_DOF),
            ("joint_impedance_arm_torque_limit", ARM_DOF),
            ("ee_torso_stiffness", TORSO_DOF),
            ("ee_torso_torque_limit", TORSO_DOF),
            ("ee_arm_stiffness", ARM_DOF),
            ("ee_arm_torque_limit", ARM_DOF),
            ("ee_null_right_arm_deg", ARM_DOF),
            ("ee_null_left_arm_deg", ARM_DOF),
            ("ee_nullspace_weight", ARM_DOF),
        ):
            if len(getattr(self, name)) != expected:
                raise ValueError(
                    f"Rby1Config.{name} must have length {expected}, got {len(getattr(self, name))}."
                )
