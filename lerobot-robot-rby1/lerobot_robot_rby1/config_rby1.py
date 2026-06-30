from dataclasses import dataclass, field
import math
from typing import List, Tuple

from lerobot.cameras import CameraConfig, Cv2Rotation
from lerobot.robots.config import RobotConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

from .constants import ARM_DOF, TORSO_DOF

# Joint-limit dictionary type alias: joint name -> (lower, upper) in radians.
JointLimits = dict[str, Tuple[float, float]]


# ---------------------------------------------------------------------------
# EE action mode defaults (Cartesian impedance).
# ---------------------------------------------------------------------------

# Nullspace arm targets (degrees) used by the per-component Cartesian solver.
_DEFAULT_NULL_RIGHT_DEG = [40.0, -30.0, -5.0, -135.0, -10.0, 20.0,  40.0]
_DEFAULT_NULL_LEFT_DEG  = [40.0,  30.0, -5.0, -135.0,  10.0, 20.0, -40.0]

# Joint limits enforced by the Cartesian impedance solvers (radians).
_DEFAULT_WB_JOINT_LIMITS: JointLimits = {
    "right_arm_3": (-2.6, -0.5),
    "right_arm_5": (0.2, 1.9),
    "left_arm_3":  (-2.6, -0.5),
    "left_arm_5":  (0.2, 1.9),
    "torso_1":     (-0.523598776, 1.3),
    "torso_2":     (-2.617993878, -0.2),
}

_DEFAULT_TORSO_JOINT_LIMITS: JointLimits = {
    "torso_1": (-0.523598776, 1.6),
    "torso_2": (-2.617993878, -0.2),
}

_DEFAULT_RIGHT_ARM_JOINT_LIMITS: JointLimits = {"right_arm_3": (-2.6, -0.5)}
_DEFAULT_LEFT_ARM_JOINT_LIMITS: JointLimits  = {"left_arm_3":  (-2.6, -0.5)}

# Cartesian add_target gain tuples for rby1_sdk add_target(...):
# (linear_acc_limit, linear_vel_limit, angular_acc_limit, angular_vel_limit).
_PI = math.pi
_TORSO_TARGET_GAINS_WB = (1.0, _PI * 0.5, 10.0, _PI * 20.0)
_TORSO_TARGET_GAINS_PC = (1.0, _PI * 0.5, 20.0, _PI * 40.0)
_ARM_TARGET_GAINS_WB   = (2.0, _PI * 2.0,  20.0, _PI * 80.0)
_ARM_TARGET_GAINS_PC   = (3.0, _PI * 2.0, 150.0, _PI * 80.0)


def _default_cameras() -> dict[str, CameraConfig]:
    """Default camera layout: no cameras.

    All entries below are commented out, so this returns an empty dict by
    default. The front / right / left RealSense block is kept as a template —
    uncomment it (with the correct serial numbers) or override
    ``Rby1Config.cameras`` to add cameras.
    """
    return {
        # Example RealSense D435 config for one robot. Override with actual serial numbers and desired settings.

        # "front": RealSenseCameraConfig(
        #     serial_number_or_name="427622271135", fps=30, width=640, height=480
        # ),
        # "right": RealSenseCameraConfig(
        #     serial_number_or_name="230422270977", fps=30, width=480, height=640,
        #     rotation=Cv2Rotation.ROTATE_90,
        # ),
        # "left": RealSenseCameraConfig(
        #     serial_number_or_name="335122272086", fps=30, width=480, height=640,
        #     rotation=Cv2Rotation.ROTATE_90,
        # ),
    }


@RobotConfig.register_subclass("rby1")
@dataclass
class Rby1Config(RobotConfig):
    # gRPC address of the robot (host:port)
    address: str = "192.168.30.1:50051"

    # Robot model variant: "a" (24-DOF), "m" (26-DOF mecanum), "ub" (18-DOF,
    # no base), or "auto" to detect it from the robot via rby1-sdk on connect().
    model: str = "auto"

    # Firmware version: "auto" to detect via rby1-sdk, or one of "1.0"/"1.1"/
    # "1.2"/"1.3". The version selects the ready pose set (v1.3 differs from
    # earlier versions near the wrist). Ignored for the "ub" model.
    version: str = "auto"

    # Timeout (seconds) for the model/version auto-detection probe connection.
    connect_timeout_sec: float = 3.0

    # Include joint velocities in observation_features
    use_velocity: bool = False

    # Include joint torques in observation_features
    use_torque: bool = False

    # Enable physical Dynamixel gripper (set False for simulation / gripper-less setups)
    use_gripper: bool = True

    # Map of camera name -> CameraConfig. Defaults to no cameras (the
    # RealSense layout in _default_cameras() is a commented-out template);
    # pass an explicit dict to enable cameras.
    cameras: dict[str, CameraConfig] = field(default_factory=_default_cameras)

    # ── Joint group selection ──────────────────────────────────────────
    # Select which joint groups to include in observation / action features.
    # Disabled groups are held at their current position during send_action().
    use_torso: bool = False
    use_right_arm: bool = True
    use_left_arm: bool = True

    # ── Action mode ────────────────────────────────────────────────────
    # "joint": actions are joint positions (radians) for the enabled arms,
    #          executed as joint position / impedance commands.
    # "ee":    actions are end-effector poses in the base frame
    #          (torso_ee.* / right_ee.* / left_ee.* — position in metres plus
    #          a rotation vector in radians), executed via the onboard
    #          Cartesian impedance solver. Produced e.g. by the rby1_vr
    #          teleoperator. In this mode use_torso also enables torso EE
    #          actions (the torso stays observation-only in joint mode).
    action_mode: str = "joint"

    # Enable the omnidirectional mobile base. When True, send_action consumes
    # the base velocity action keys (x.vel, y.vel, theta.vel) and forwards them
    # to the robot as an SE2 velocity command. Requires a model with a base
    # ("a" or "m"). Leave False for upper-body-only setups.
    use_mobile_base: bool = False

    # ── Record reset selection ────────────────────────────────────────
    # When lerobot-record calls robot.reset() between episodes, only the
    # enabled arms are moved back to the ready pose. Disabled arms keep
    # their current joint positions.
    reset_right_arm_on_record: bool = False
    reset_left_arm_on_record: bool = False

    # ── Impedance control ──────────────────────────────────────────────
    # When True, send_action uses JointImpedanceControlCommandBuilder
    # instead of JointPositionCommandBuilder, yielding compliant behaviour
    # similar to the teleoperation stream.
    use_impedance: bool = False

    # Joint stiffness (Nm/rad) for the 20 body joints in order:
    #   torso_0…5 (6), right_arm_0…6 (7), left_arm_0…6 (7)
    # Higher values → stiffer, lower → more compliant.
    impedance_stiffness: List[float] = field(
        default_factory=lambda: [400.0] * 6 + [150.0] * 7 + [150.0] * 7
    )

    # Torque limits (Nm) for the 20 body joints (same ordering as above).
    impedance_torque_limit: List[float] = field(
        default_factory=lambda: [500.0] * 6 + [40.0] * 7 + [40.0] * 7
    )

    # Damping ratio applied to all joints [0.0, 1.0].
    # 0.7 gives critical damping; lower values allow faster motion.
    impedance_damping_ratio: float = 0.7

    # ── Velocity / acceleration limits ─────────────────────────────────
    # Limits are read from the URDF dynamics model at connect() time.
    # These scaling factors adjust the raw URDF values.

    # Multiply the wrist joint (last joint of each arm) velocity limit
    # by this factor.  The wrist needs higher velocity headroom for
    # impedance-mode teleoperation.  (SDK example uses 10×.)
    wrist_velocity_limit_scale: float = 10.0

    # Multiply ALL joints' URDF acceleration limits by this factor.
    # The raw URDF values are very conservative; 30× is the SDK default
    # for real-time teleoperation responsiveness.
    acceleration_limit_scale: float = 30.0

    # Move robot to ready pose automatically when connect() is called.
    # Set False to skip the ready pose motion (e.g. during unit tests or
    # when the robot is already in a safe position).
    move_to_ready_on_connect: bool = True

    # ── Startup ramp ───────────────────────────────────────────────────
    # At the beginning of a teleop session the follower robot needs to
    # smoothly converge to the leader's current pose.  A large
    # `minimum_time` per command slows the robot down, giving it time to
    # catch up gradually rather than snapping to the first target.
    #
    # startup_ramp_duration : seconds over which minimum_time is linearly
    #   interpolated from startup_min_time → normal_min_time.  Set to 0.0 to
    #   disable the ramp.
    # startup_min_time      : minimum_time (seconds) used at t=0.
    # (Joint mode only; EE commands use ee_dt * min_time_factor_* instead.)
    startup_ramp_duration: float = 5.0
    startup_min_time: float = 5.0
    normal_min_time: float = 0.07

    # ════════════════════════════════════════════════════════════════════
    # EE action mode (action_mode="ee") — Cartesian impedance tuning.
    # The fields below only apply when action_mode="ee".
    # ════════════════════════════════════════════════════════════════════

    # Use a single whole-body Cartesian impedance solver for torso + arms.
    # False → separate torso / right-arm / left-arm solvers (more stable).
    ee_whole_body: bool = False

    # Expected action period (s); Cartesian command minimum_time is
    # ee_dt * min_time_factor_wb (whole-body / mobility) or
    # ee_dt * min_time_factor_pc (per-component torso/arm).
    ee_dt: float = 1 / 15
    min_time_factor_wb: float = 1.01
    min_time_factor_pc: float = 1.02

    # Cartesian commands keep holding for this long (s) if the stream stalls.
    ee_hold_time: float = 30.0

    # ── Reference reset (jump) detection ──────────────────────────────
    # When a new EE target jumps further than these thresholds from the
    # previous one (e.g. the teleoperator re-latched after the robot was
    # reset), set_reset_reference is sent so the solver re-references to
    # the robot's actual state instead of chasing the jump violently.
    ee_reset_position_threshold: float = 0.1   # metres
    ee_reset_rotation_threshold: float = 0.5   # radians

    # ── Whole-body solver tuning (ee_whole_body=True) ──────────────────
    # Stiffness / torque-limit arrays are laid out as
    # [torso (6) | right arm (7) | left arm (7)].
    wb_joint_stiffness: List[float] = field(
        default_factory=lambda: [400.0] * 6 + [60.0] * 7 + [60.0] * 7
    )
    wb_joint_torque_limit: List[float] = field(
        default_factory=lambda: [500.0] * 6 + [30.0] * 7 + [30.0] * 7
    )
    wb_joint_limits: JointLimits = field(
        default_factory=lambda: dict(_DEFAULT_WB_JOINT_LIMITS)
    )

    # ── Per-component solver tuning (ee_whole_body=False) ───────────────
    torso_stiffness: List[float] = field(default_factory=lambda: [1700.0] * 6)
    torso_torque_limit: List[float] = field(default_factory=lambda: [600.0] * 6)
    torso_damping_ratio: float = 0.7
    torso_joint_limits: JointLimits = field(
        default_factory=lambda: dict(_DEFAULT_TORSO_JOINT_LIMITS)
    )

    right_arm_stiffness: List[float] = field(
        default_factory=lambda: [90.0, 90.0, 90.0, 70.0, 70.0, 70.0, 70.0]
    )
    right_arm_torque_limit: List[float] = field(
        default_factory=lambda: [40.0, 40.0, 40.0, 30.0, 30.0, 30.0, 30.0]
    )
    right_arm_damping_ratio: float = 0.6
    right_arm_joint_limits: JointLimits = field(
        default_factory=lambda: dict(_DEFAULT_RIGHT_ARM_JOINT_LIMITS)
    )

    left_arm_stiffness: List[float] = field(
        default_factory=lambda: [60.0, 60.0, 60.0, 50.0, 50.0, 50.0, 50.0]
    )
    left_arm_torque_limit: List[float] = field(
        default_factory=lambda: [40.0, 40.0, 40.0, 30.0, 30.0, 30.0, 30.0]
    )
    left_arm_damping_ratio: float = 0.4
    left_arm_joint_limits: JointLimits = field(
        default_factory=lambda: dict(_DEFAULT_LEFT_ARM_JOINT_LIMITS)
    )

    # ── Cartesian add_target gain tuples ─────────────────────────────
    torso_target_gains_wb: Tuple[float, float, float, float] = _TORSO_TARGET_GAINS_WB
    torso_target_gains_pc: Tuple[float, float, float, float] = _TORSO_TARGET_GAINS_PC
    arm_target_gains_wb:   Tuple[float, float, float, float] = _ARM_TARGET_GAINS_WB
    arm_target_gains_pc:   Tuple[float, float, float, float] = _ARM_TARGET_GAINS_PC

    # ── Nullspace targets (per-component arm solvers) ─────────────────
    null_right_arm_deg: List[float] = field(
        default_factory=lambda: list(_DEFAULT_NULL_RIGHT_DEG)
    )
    null_left_arm_deg: List[float] = field(
        default_factory=lambda: list(_DEFAULT_NULL_LEFT_DEG)
    )
    nullspace_weight: List[float] = field(
        default_factory=lambda: [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0]
    )
    nullspace_stiffness: float = 0.2
    nullspace_damping_ratio: float = 0.3

    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.action_mode not in ("joint", "ee"):
            raise ValueError(
                f'Rby1Config.action_mode must be "joint" or "ee", got "{self.action_mode}".'
            )
        if self.model.strip().lower() not in ("a", "m", "ub", "auto"):
            raise ValueError(
                f'Rby1Config.model must be "a", "m", "ub" or "auto", got "{self.model}".'
            )
        if self.version.strip().lower() not in ("auto", "", "1.0", "1.1", "1.2", "1.3"):
            raise ValueError(
                'Rby1Config.version must be "auto" or one of "1.0"/"1.1"/"1.2"/"1.3", '
                f'got "{self.version}".'
            )
        body_dof = TORSO_DOF + 2 * ARM_DOF
        self._check_len("impedance_stiffness", self.impedance_stiffness, body_dof)
        self._check_len("impedance_torque_limit", self.impedance_torque_limit, body_dof)
        self._check_len("wb_joint_stiffness", self.wb_joint_stiffness, body_dof)
        self._check_len("wb_joint_torque_limit", self.wb_joint_torque_limit, body_dof)
        self._check_len("torso_stiffness", self.torso_stiffness, TORSO_DOF)
        self._check_len("torso_torque_limit", self.torso_torque_limit, TORSO_DOF)
        self._check_len("right_arm_stiffness", self.right_arm_stiffness, ARM_DOF)
        self._check_len("right_arm_torque_limit", self.right_arm_torque_limit, ARM_DOF)
        self._check_len("left_arm_stiffness", self.left_arm_stiffness, ARM_DOF)
        self._check_len("left_arm_torque_limit", self.left_arm_torque_limit, ARM_DOF)
        self._check_len("null_right_arm_deg", self.null_right_arm_deg, ARM_DOF)
        self._check_len("null_left_arm_deg", self.null_left_arm_deg, ARM_DOF)
        self._check_len("nullspace_weight", self.nullspace_weight, ARM_DOF)

    @staticmethod
    def _check_len(name: str, value: List[float], expected: int) -> None:
        if len(value) != expected:
            raise ValueError(
                f"Rby1Config.{name} must have length {expected}, got {len(value)}."
            )
