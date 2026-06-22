from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from lerobot.teleoperators.config import TeleoperatorConfig

from .constants import (
    ARM_DOF,
    SUPPORTED_LEADER_VERSIONS,
    leader_init_pose_for_version,
)

_LEADER_DOF = 2 * ARM_DOF  # 14


@TeleoperatorConfig.register_subclass("rby1_leader_arm")
@dataclass
class Rby1LeaderArmConfig(TeleoperatorConfig):
    """Configuration for the RB-Y1 leader arm (physical master arm) teleoperator.

    The leader arm is a 14-DOF device (7 per arm) with per-hand trigger
    and button.  Joint positions are read from the leader arm's Dynamixel
    servos and returned directly as the teleoperator action — following
    LeRobot convention where the teleoperator does NOT send commands to
    the robot itself.
    """

    # ── Leader arm device ─────────────────────────────────────────────
    # Path to the leader arm URDF model (required for gravity compensation).
    # If None, auto-detected relative to the rby1_sdk installation.
    leader_arm_model_path: str | None = None

    # RB-Y1 firmware version this leader arm is paired with. Selects the
    # version-specific init (ready) pose; v1.3 differs from v1.2 in the wrist
    # configuration. Must match the follower robot's resolved version.
    version: str = "1.2"

    # Leader arm control callback frequency in Hz.
    control_frequency: float = 100.0

    # ── Joint group selection ─────────────────────────────────────────
    # Must match the corresponding flags in Rby1Config so that
    # action_features are consistent between teleoperator and robot.
    use_torso: bool = False       # Leader arm has no torso DOFs
    use_right_arm: bool = True
    use_left_arm: bool = True

    # Enable gripper control via trigger buttons on the leader arm.
    use_gripper: bool = True

    # ── Record reset selection ────────────────────────────────────────
    # When lerobot-record calls teleop.reset() between episodes, only the
    # enabled arms are moved back to the init pose. Disabled arms keep
    # their current joint positions.
    reset_right_arm_on_record: bool = False
    reset_left_arm_on_record: bool = False

    # ── Initialisation trajectory ─────────────────────────────────────
    # Duration (seconds) for the smooth cosine interpolation from the
    # current leader arm pose to the init pose on connect().
    init_duration: float = 5.0

    # Init (ready) pose for each side in degrees.
    # When left as None, the pose is derived from the follower robot's ready
    # pose for `version` (see leader_init_pose_for_version), with the wrist
    # offset subtracted so the robot lands exactly on its ready pose.
    right_init_q_deg: List[float] | None = None
    left_init_q_deg: List[float] | None = None

    # ── Leader arm joint limits (14 DOF, degrees) ─────────────────────
    # Indices 0-6: right arm, 7-13: left arm.
    min_q_deg: List[float] = field(
        default_factory=lambda: [
            -180, -60, -15, -150, -180,  -90, -155,
            -180,  35, -105, -150, -180,  -90, -155,
        ]
    )
    max_q_deg: List[float] = field(
        default_factory=lambda: [
            180, -35, 105, 1,  180, 110, 155,
            180,  60, 15, 1,  180, 110, 155,
        ]
    )

    # ── Leader arm torque / safety parameters ─────────────────────────
    # Maximum torque applied per joint (Nm).
    torque_limit: List[float] = field(
        default_factory=lambda: [4.0, 4.0, 4.0, 4.0, 1.5, 1.5, 1.5] * 2
    )

    # Viscous damping gain per joint.
    viscous_gain: List[float] = field(
        default_factory=lambda: [0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 0.002] * 2
    )

    # Barrier gain at joint limits (Nm/rad).
    joint_limit_barrier: float = 1.5

    # Scaling factor for gravity compensation torque when the arm is in
    # free (current-control) mode (button pressed).
    gravity_comp_scale: float = 0.4

    # ── Wrist offset (leader → robot mapping) ────────────────────────
    # Mechanical zero-offset for the last joint of each arm (degrees).
    # Leader-arm reads 0 → robot receives +offset.
    right_wrist_offset_deg: float = 90.0
    left_wrist_offset_deg: float = -90.0

    # ── Gripper trigger ───────────────────────────────────────────────
    # Raw trigger range is [0, gripper_trigger_max].
    # Normalised to 0.0–1.0 for the action dict.
    gripper_trigger_max: float = 1000.0

    # ── Startup ───────────────────────────────────────────────────────
    # Seconds to wait after starting the control loop for the leader arm
    # to reach the ready pose before returning from connect().
    startup_wait_time: float = 5.0

    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        version = self.version.strip()
        if version not in SUPPORTED_LEADER_VERSIONS:
            raise ValueError(
                f"Rby1LeaderArmConfig.version must be one of "
                f"{SUPPORTED_LEADER_VERSIONS}, got {self.version!r}."
            )
        self.version = version

        default_right, default_left = leader_init_pose_for_version(
            version,
            self.right_wrist_offset_deg,
            self.left_wrist_offset_deg,
        )
        if self.right_init_q_deg is None:
            self.right_init_q_deg = default_right
        if self.left_init_q_deg is None:
            self.left_init_q_deg = default_left

        self._check_len("right_init_q_deg", self.right_init_q_deg, ARM_DOF)
        self._check_len("left_init_q_deg", self.left_init_q_deg, ARM_DOF)
        self._check_len("min_q_deg", self.min_q_deg, _LEADER_DOF)
        self._check_len("max_q_deg", self.max_q_deg, _LEADER_DOF)
        self._check_len("torque_limit", self.torque_limit, _LEADER_DOF)
        self._check_len("viscous_gain", self.viscous_gain, _LEADER_DOF)
        if self.control_frequency <= 0:
            raise ValueError("control_frequency must be > 0")
        if self.init_duration < 0:
            raise ValueError("init_duration must be >= 0")

    @staticmethod
    def _check_len(name: str, value: List[float], expected: int) -> None:
        if len(value) != expected:
            raise ValueError(
                f"Rby1LeaderArmConfig.{name} must have length {expected}, got {len(value)}."
            )
