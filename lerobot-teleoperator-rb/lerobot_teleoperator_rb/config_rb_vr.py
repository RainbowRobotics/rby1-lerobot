from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("rb_vr")
@dataclass(kw_only=True)
class RbVrConfig(TeleoperatorConfig):
    """Configuration for the Meta Quest VR teleoperator for RB cobots.

    The teleoperator receives Meta Quest controller poses over UDP,
    converts the selected controller pose into the RB10E coordinate frame,
    solves joint-space IK, and emits six joint targets in radians.

    The paired robot must use ``action_space="joint"``.
    """

    # ------------------------------------------------------------------
    # Meta Quest UDP communication
    # ------------------------------------------------------------------

    # PC interface receiving Quest packets.
    local_ip: str = "0.0.0.0"
    local_port: int = 5005

    # Quest address used to send the initial PC IP/port handshake.
    # Set to None when the Quest application sends directly without
    # requiring the handshake.
    meta_quest_ip: str | None = None
    meta_quest_port: int = 6000
    send_handshake: bool = True

    # ------------------------------------------------------------------
    # Controller mapping
    # ------------------------------------------------------------------

    controller_hand: Literal["right", "left"] = "right"

    # Grip activates/deactivates arm following.
    grip_threshold: float = 0.5

    # Trigger can later be mapped to the physical gripper.
    use_gripper: bool = False

    # If no packet arrives within this duration, stop updating the target
    # and hold the last valid joint command.
    tracking_timeout_s: float = 0.25

    # ------------------------------------------------------------------
    # VR-to-RB workspace mapping
    # ------------------------------------------------------------------

    # Existing implementation:
    #     user_scale = 1300 mm / measured arm reach.
    auto_user_scale: bool = True
    reference_reach_mm: float = 1300.0
    default_user_scale: float = 1300.0 / 700.0

    # Existing apply_scale() adds 300 mm to the target Z position.
    target_z_offset_mm: float = 300.0

    # Optional additional multiplier after user calibration.
    position_scale: float = 1.0

    # ------------------------------------------------------------------
    # RB10E inverse kinematics
    # ------------------------------------------------------------------

    ik_iterations: int = 3

    # Start the IK solver from the most recently emitted joint target.
    # On first use, initialise it from the connected RB robot state.
    initialize_ik_from_robot: bool = True

    # ------------------------------------------------------------------
    # Safety / fail-safe behavior
    # ------------------------------------------------------------------

    # Maximum change in the joint target produced by one get_action() call.
    # The RbCobot follower also performs its own final action clamp.
    # A 버튼을 누르기 전까지 현재 자세를 유지한다.
    require_initialization_button: bool = True

    # get_action() 한 번당 허용할 최대 joint target 변화량.
    max_joint_delta_rad: float = 0.10

    # IK 결과가 이 오차보다 크면 명령을 폐기한다.
    max_ik_position_error_mm: float = 50.0
    max_ik_orientation_error: float = 1.0

    # When tracking is lost or grip is released, keep emitting the last
    # valid target rather than zeros.
    hold_last_target: bool = True