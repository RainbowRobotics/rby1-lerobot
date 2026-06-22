"""Dataclass-based configuration for the VR teleoperator."""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("rby1_vr")
@dataclass
class Rby1VRConfig(TeleoperatorConfig):
    """Configuration for the Meta Quest VR teleoperator.

    This teleoperator only *produces* actions (EE poses + gripper + mobile
    base); command execution and all Cartesian-impedance tuning live in the
    follower robot's ``Rby1Config`` (``action_mode="ee"``).
    """

    # ── Robot connection (read-only: state + FK for latching) ────────
    robot_address: str = "192.168.30.1:50051"
    robot_model: str = "m"  # "a" | "m" | "ub"

    # ── Meta Quest UDP communication ─────────────────────────────────
    local_ip: str = "192.168.20.3"            # IP of this PC (Quest must reach it)
    local_port: int = 5005                     # UDP port we listen on
    meta_quest_ip: str = "192.168.20.2"       # Quest headset IP
    meta_quest_port: int = 6000                # Quest UDP port for the handshake

    # ── Joint group selection ─────────────────────────────────────────
    # Must match the corresponding flags in Rby1Config (action_mode="ee")
    # so action_features stay consistent between teleoperator and robot.
    use_torso: bool = True
    use_right_arm: bool = True
    use_left_arm: bool = True

    # ── Control options ───────────────────────────────────────────────
    use_gripper: bool = True
    use_mobile_base: bool = True

    # ── VR button thresholds ─────────────────────────────────────────
    grip_threshold: float = 0.8                 # grip > threshold → following

    # ── Mobile base integrator gains ─────────────────────────────────
    mobile_base_linear_acc_gain: float = 0.3
    mobile_base_angular_acc_gain: float = 0.7
    mobile_base_linear_damping: float = 0.5
    mobile_base_angular_damping: float = 0.5
    mobile_base_output_scale: float = 2.0
