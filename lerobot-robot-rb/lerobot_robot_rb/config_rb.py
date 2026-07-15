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

    # ── Real-time streaming (send_action), rbpodo move_servo_j ─────────
    # Servo parameters (t1, t2, gain, alpha) — reference values from the
    # rbpodo examples; retune on hardware for your control rate.
    servo_t1: float = 0.01
    servo_t2: float = 0.1
    servo_gain: float = 1.0
    servo_alpha: float = 1.0

    # ── Safety ──────────────────────────────────────────────────────────
    # Joint limits override ((lower, upper) pairs, degrees); None uses the
    # model default.
    joint_limits_deg: List[Tuple[float, float]] | None = None

    # Maximum commanded joint change per send_action() call (degrees).
    # 2 deg per step at 30 Hz caps the commanded speed at 60 deg/s.
    max_joint_delta_deg: float = 2.0

    # Timeout for a single data-channel poll (seconds).
    data_timeout_sec: float = 0.5

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
        if not 0.0 < self.speed_bar <= 1.0:
            raise ValueError(
                f"RbCobotConfig.speed_bar must be in (0, 1], got {self.speed_bar}."
            )
        if self.ptp_speed <= 0 or self.ptp_acc <= 0:
            raise ValueError("RbCobotConfig.ptp_speed and ptp_acc must be > 0.")
        if self.max_joint_delta_deg <= 0:
            raise ValueError(
                f"RbCobotConfig.max_joint_delta_deg must be > 0, "
                f"got {self.max_joint_delta_deg}."
            )
        if self.ready_pose_deg is not None:
            self._check_len("ready_pose_deg", self.ready_pose_deg, DOF)
        if self.joint_limits_deg is not None:
            self._check_len("joint_limits_deg", self.joint_limits_deg, DOF)
            for i, pair in enumerate(self.joint_limits_deg):
                if len(pair) != 2 or pair[0] >= pair[1]:
                    raise ValueError(
                        f"RbCobotConfig.joint_limits_deg[{i}] must be a "
                        f"(lower, upper) pair with lower < upper, got {pair}."
                    )
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
