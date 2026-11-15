"""Dataclass-based configuration for the whole-body VR teleoperator."""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("rby1_vr_wholebody")
@dataclass
class Rby1VRWholebodyConfig(TeleoperatorConfig):
    """Configuration for the Meta Quest whole-body teleoperator (body + hand tracking).

    The teleoperator only *produces* actions; the follower robot must run with
    ``action_mode="wholebody"`` (and matching ``use_head`` / ``use_gripper`` /
    ``use_mobile_base``). Tuning defaults are those of the semi-humanoid CES
    teleoperation (``rby1_ces_elbow.py``).
    """

    # ── Robot connection (read-only: state, odometry and FK for the latch) ──
    robot_address: str = "192.168.30.1:50051"
    # "a" | "m" | "auto". The arm model is RB-Y1 M v1.3; connect() checks it
    # against the robot's dynamics model (see ignore_model_check).
    robot_model: str = "auto"
    # Fail connect() when the bundled v1.3 arm model and the robot's FK differ
    # by more than model_check_tol_m (the hand mapping would be wrong).
    ignore_model_check: bool = False
    model_check_tol_m: float = 0.005

    # ── Headset stream (Meta_with_hands app: TCP server, NDJSON) ─────
    headset_ip: str = "192.168.20.2"
    headset_port: int = 5555
    # Accept body tracking frames the headset marks as low fidelity.
    body_allow_low_fidelity: bool = False

    # Record this teleoperator's config in the robot profile file (same path
    # as --robot.profile_out). Stored under "teleop", never compared.
    profile_out: str | None = None

    # ── Emitted action groups (must match the follower Rby1Config) ───
    use_head: bool = True
    use_gripper: bool = True
    use_mobile_base: bool = True
    # Swap the gripper open/close command.
    gripper_invert: bool = False

    # ── Keyboard (terminal running lerobot-teleoperate / -record) ────
    # Enter = latch (operator pose ↔ robot pose, start / resume),
    # Space = hold (base stops, arms / torso keep the last target).
    # Without a TTY the first latch happens automatically after
    # auto_latch_s seconds of tracked body (None = wait forever).
    auto_latch_s: float | None = None
    # Period (s) of the status log line; 0 disables it.
    status_log_period_s: float = 2.0

    # ── Hand mapping ──────────────────────────────────────────────────
    # "elbow": operator elbow flexion → robot elbow angle (default).
    # "palm":  robot shoulder + arm_scale · (palm − operator shoulder).
    hand_map: str = "elbow"
    # palm map only: scale factor, or None = auto (robot / operator arm length).
    arm_scale: float | None = 1.0
    # Hand orientation target rotated about the arm_6 axis (deg): right +,
    # left −, so both gripper tips turn inwards. 0 = no offset.
    arm6_offset_deg: float = 30.0
    tau_elbow: float = 0.10                  # s, operator flexion low-pass (elbow map)
    elbow_slew_deg: float = 150.0            # deg/s, flexion command rate limit (elbow map)

    # ── Input filters (s) ─────────────────────────────────────────────
    # Keep tau_hand == tau_body: the hand target is (palm − shoulder), so a
    # mismatch shifts the hand by v·Δτ while walking.
    tau_hand: float = 0.04
    tau_body: float = 0.04
    tau_heading: float = 0.12
    tau_hmd: float = 0.05

    # ── Command slew ──────────────────────────────────────────────────
    hand_slew_v: float = 2.0                 # m/s
    hand_slew_w: float = 6.0                 # rad/s
    torso_slew_w: float = 2.0                # rad/s
    # Rejoin offset bleed (start / hand re-acquired).
    bleed_lin: float = 0.10                  # m/s
    bleed_ang: float = 0.6                   # rad/s
    bleed_time: float = 0.5                  # s

    # ── Base (operator pelvis → base position, odometry closed loop) ─
    # "pelvis": heading from the pelvis rotation; "shoulder": from the shoulder line.
    heading: str = "pelvis"
    # Follow the pelvis height with the torso_5 z target.
    torso_height: bool = False
    base_vmax: float = 0.8                   # m/s (also clamped by the robot's limit)
    base_wmax: float = 1.5                   # rad/s
    base_kp: float = 2.0
    base_kp_yaw: float = 3.0
    base_kff: float = 0.8                    # operator-velocity feed-forward 0..1
    base_tau_ff: float = 0.15
    base_err_stop: float = 0.02              # m — Schmitt dead band
    base_err_go: float = 0.04
    base_yaw_stop_deg: float = 1.0
    base_yaw_go_deg: float = 2.5
    base_max_lag: float = 0.5                # m
    base_max_lag_yaw_deg: float = 60.0

    # ── Gripper (thumb-index pinch) ───────────────────────────────────
    pinch_open_m: float = 0.08
    pinch_threshold: float = 0.5
    pinch_hyst: float = 0.03

    def __post_init__(self) -> None:
        parent_post_init = getattr(super(), "__post_init__", None)
        if parent_post_init is not None:
            parent_post_init()
        if self.hand_map not in ("elbow", "palm"):
            raise ValueError(f'hand_map must be "elbow" or "palm", got "{self.hand_map}".')
        if self.heading not in ("pelvis", "shoulder"):
            raise ValueError(f'heading must be "pelvis" or "shoulder", got "{self.heading}".')
        if self.robot_model.strip().lower() not in ("a", "m", "auto"):
            raise ValueError(f'robot_model must be "a", "m" or "auto", got "{self.robot_model}".')
        if self.bleed_time <= 0 or self.pinch_open_m <= 0:
            raise ValueError("bleed_time and pinch_open_m must be > 0.")
