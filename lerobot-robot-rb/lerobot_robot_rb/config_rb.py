import math
from dataclasses import dataclass, field
from typing import List, Tuple

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig

from .models import DOF, MODEL_SPECS


@dataclass
class RbCobotConfig(RobotConfig):
    """Shared configuration for the RB-Series cobots (RB3/RB5/RB10).

    This base class is intentionally not registered with LeRobot; the
    per-model shims fix the ``model`` field and register the CLI name
    (e.g. :class:`~lerobot_robot_rb.rb10.Rb10Config` as ``rb10``).
    """

    # MODEL_SPECS key selecting the per-model constants (joint limits,
    # default ready pose). Fixed by the per-model config shims.
    model: str = "rb10"

    # IP address of the control box (shown on the teaching pendant).
    ip: str = "10.0.2.7"

    # TCP ports of the command (Cobot) and data (CobotData) channels.
    command_port: int = 5000
    data_port: int = 5001

    # "simulation" runs the control-box simulator (no physical motion) —
    # recommended for bring-up; "real" moves the physical robot.
    operation_mode: str = "simulation"

    # Real mode is rejected unless this explicit acknowledgement is set.
    real_mode_confirm: bool = False

    # Refuse real-mode streaming while the control box reports collision
    # detection disabled.
    require_collision_detection: bool = True

    # "joint": actions are joint positions (radians), streamed with
    # move_servo_j. "ee": actions are TCP poses (metres / radians, keys
    # ee_x..ee_rz), streamed with move_servo_l — inverse kinematics runs on
    # the control box, no URDF needed.
    action_space: str = "joint"

    # Global speed override applied on connect (rbpodo set_speed_bar), (0, 1].
    speed_bar: float = 0.3

    # ── Blocking PTP motion (connect / reset), rbpodo move_j ───────────
    ptp_speed: float = 60.0  # deg/s
    ptp_acc: float = 80.0    # deg/s^2

    # Move to the ready pose at the end of connect(). Ships disabled until
    # the ready pose has been validated for the actual cell.
    move_to_ready_on_connect: bool = False

    # Ready pose override (degrees, length 6); None uses the model default.
    ready_pose_deg: List[float] | None = None

    # Move back to the ready pose when lerobot-record calls reset()
    # between episodes.
    reset_on_record: bool = False

    # ── Real-time streaming (send_action) ───────────────────────────────
    # move_servo_j/l parameters. Rainbow's script reference: t1 = time to
    # arrive at the target (>= 0.002), t2 = motion hold time after arrival
    # (0.02 < t2 < 0.2), gain = velocity tracking rate (> 0), alpha =
    # low-pass-filter gain, smaller is smoother (0 < alpha < 1).
    # t1 should be >= the interpolation step so consecutive targets blend.
    servo_t1: float = 0.05
    servo_t2: float = 0.1
    servo_gain: float = 1.0
    servo_alpha: float = 0.3

    # Background worker period. The worker linearly interpolates between
    # consecutive send_action() targets and refreshes move_servo_j/l at
    # this rate (5 ms, matching rbpodo's reference example), so the robot
    # receives a continuous reference instead of 30 Hz steps.
    servo_command_period_s: float = 0.005

    # Bounds for the per-target interpolation window. The actual window
    # tracks the measured send_action() call interval, clamped to this
    # range (seconds).
    interp_min_s: float = 0.02
    interp_max_s: float = 0.2

    # ── Safety ──────────────────────────────────────────────────────────
    # Joint limits override ((lower, upper) pairs, degrees); None uses the
    # model default.
    joint_limits_deg: List[Tuple[float, float]] | None = None

    # Maximum commanded joint change per send_action() call (degrees).
    # 2 deg per step at 30 Hz caps the commanded speed at 60 deg/s.
    max_joint_delta_deg: float = 2.0

    # EE-mode per-step clamps: position (mm) and orientation (deg) change
    # per send_action() call. 5 mm at 30 Hz caps TCP speed at 150 mm/s.
    max_ee_pos_delta_mm: float = 5.0
    max_ee_rot_delta_deg: float = 2.0

    # Optional EE workspace box ((min, max) per x/y/z axis, mm, in the
    # control box's global frame); None disables the box clamp.
    ee_bounds_mm: List[Tuple[float, float]] | None = None

    # Timeout for a single data-channel poll (seconds).
    data_timeout_sec: float = 0.5

    # The command-channel ACK can arrive before the data channel reflects a
    # Real/Simulation mode change; wait up to this long for the requested
    # mode instead of trusting the first stale state packet.
    operation_mode_timeout_sec: float = 5.0

    # ── Optional observation channels ───────────────────────────────────
    # CobotData exposes no joint-velocity field; "<joint>.vel" is
    # finite-differenced from consecutive polls and therefore noisy.
    use_velocity: bool = False

    # Measured joint currents (A) as "<joint>.current".
    use_current: bool = False

    # ── Gripper ─────────────────────────────────────────────────────────
    # "none", or "rby1_dynamixel" for the RB-Y1 Dynamixel gripper attached
    # to the control PC over USB (requires the gripper-rby1 extra).
    # Additional grippers (e.g. the RH-P12-RN through the rbpodo built-in
    # gripper API) plug into the same interface — see gripper.make_gripper.
    gripper_type: str = "none"

    # Serial port of the Dynamixel gripper bus.
    gripper_port: str = "/dev/rby1_gripper"

    # Dynamixel motor IDs on the gripper bus (single motor by default).
    gripper_ids: List[int] = field(default_factory=lambda: [0])

    # Flip the open/close encoder direction discovered by the homing sweep
    # (depends on the gripper's mounting orientation).
    gripper_invert: bool = False

    # Map of camera name -> CameraConfig; {} disables cameras.
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.model not in MODEL_SPECS:
            raise ValueError(
                f"RbCobotConfig.model must be one of {sorted(MODEL_SPECS)}, "
                f'got "{self.model}".'
            )
        if self.operation_mode not in ("real", "simulation"):
            raise ValueError(
                'RbCobotConfig.operation_mode must be "real" or "simulation", '
                f'got "{self.operation_mode}".'
            )
        if self.operation_mode == "real" and not self.real_mode_confirm:
            raise ValueError("Real mode requires `real_mode_confirm=true`.")
        if self.action_space not in ("joint", "ee"):
            raise ValueError(
                'RbCobotConfig.action_space must be "joint" or "ee", '
                f'got "{self.action_space}".'
            )
        if not 0.0 < self.speed_bar <= 1.0:
            raise ValueError(
                f"RbCobotConfig.speed_bar must be in (0, 1], got {self.speed_bar}."
            )
        if self.ptp_speed <= 0 or self.ptp_acc <= 0:
            raise ValueError("RbCobotConfig.ptp_speed and ptp_acc must be > 0.")
        for name, value in (
            ("servo_t1", self.servo_t1),
            ("servo_t2", self.servo_t2),
            ("servo_gain", self.servo_gain),
            ("servo_command_period_s", self.servo_command_period_s),
            ("interp_min_s", self.interp_min_s),
            ("interp_max_s", self.interp_max_s),
            ("max_joint_delta_deg", self.max_joint_delta_deg),
            ("max_ee_pos_delta_mm", self.max_ee_pos_delta_mm),
            ("max_ee_rot_delta_deg", self.max_ee_rot_delta_deg),
            ("data_timeout_sec", self.data_timeout_sec),
            ("operation_mode_timeout_sec", self.operation_mode_timeout_sec),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"RbCobotConfig.{name} must be finite and > 0, got {value}."
                )
        if not 0.0 < self.servo_alpha < 1.0:
            raise ValueError(
                "RbCobotConfig.servo_alpha must be in (0, 1) per Rainbow's "
                f"script reference, got {self.servo_alpha}."
            )
        if self.interp_min_s >= self.interp_max_s:
            raise ValueError(
                "RbCobotConfig.interp_min_s must be < interp_max_s, got "
                f"{self.interp_min_s} >= {self.interp_max_s}."
            )
        if self.ready_pose_deg is not None:
            self._check_len("ready_pose_deg", self.ready_pose_deg, DOF)
        if self.joint_limits_deg is not None:
            self._check_len("joint_limits_deg", self.joint_limits_deg, DOF)
            self._check_pairs("joint_limits_deg", self.joint_limits_deg)
        if self.ee_bounds_mm is not None:
            self._check_len("ee_bounds_mm", self.ee_bounds_mm, 3)
            self._check_pairs("ee_bounds_mm", self.ee_bounds_mm)
        if self.gripper_type not in ("none", "rby1_dynamixel"):
            raise ValueError(
                'RbCobotConfig.gripper_type must be "none" or "rby1_dynamixel", '
                f'got "{self.gripper_type}".'
            )
        if not self.gripper_ids:
            raise ValueError("RbCobotConfig.gripper_ids must not be empty.")

    @staticmethod
    def _check_len(name: str, value: list, expected: int) -> None:
        if len(value) != expected:
            raise ValueError(
                f"RbCobotConfig.{name} must have length {expected}, got {len(value)}."
            )

    @staticmethod
    def _check_pairs(name: str, pairs: list) -> None:
        for i, pair in enumerate(pairs):
            if len(pair) != 2 or pair[0] >= pair[1]:
                raise ValueError(
                    f"RbCobotConfig.{name}[{i}] must be a (lower, upper) "
                    f"pair with lower < upper, got {pair}."
                )
