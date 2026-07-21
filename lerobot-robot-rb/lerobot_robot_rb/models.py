"""Model specifications for the Rainbow Robotics RB-Series cobots.

The RB-Series (RB3 / RB5 / RB10 ...) shares one control stack and one client
SDK (``rbpodo``); the models differ only in reach, payload, joint limits and
speed ratings.  Supporting a new model therefore only requires a new
:data:`MODEL_SPECS` entry plus a small registration shim module (see
``rb10.py`` for the template).
"""

from __future__ import annotations

from dataclasses import dataclass

# All RB-Series cobots are 6-DOF arms (joint_0 = base ... joint_5 = wrist).
DOF = 6

# Observation / action key names. Model-independent so datasets and policies
# stay compatible across the series.
JOINT_NAMES = [f"joint_{i}" for i in range(DOF)]
GRIPPER_NAME = "gripper_0"

# End-effector (TCP) pose keys for action_space="ee". Position in metres,
# orientation as the control box's rx/ry/rz Euler angles in radians at the
# dataset boundary (rbpodo talks mm / degrees internally).
EE_NAMES = ["ee_x", "ee_y", "ee_z", "ee_rx", "ee_ry", "ee_rz"]


@dataclass(frozen=True)
class RbModelSpec:
    """Per-model physical constants (angles in degrees, the rbpodo convention)."""

    # Joint limits as (lower, upper) pairs, degrees.
    joint_limits_deg: tuple[tuple[float, float], ...]
    # Default ready/home pose used by move_to_ready_pose() and reset(), degrees.
    default_ready_pose_deg: tuple[float, ...]
    # Conservative per-joint speed cap used for safety defaults, deg/s.
    max_joint_speed_deg_s: float


MODEL_SPECS: dict[str, RbModelSpec] = {
    # RB10-1300 joint ranges from Rainbow Robotics' RB-Series catalogue.
    # Verify the physical arm's nameplate before real mode and override
    # joint_limits_deg if it is a different RB10 variant. The ready pose
    # remains cell-specific and must be validated before use.
    "rb10": RbModelSpec(
        joint_limits_deg=(
            (-360.0, 360.0),
            (-360.0, 360.0),
            (-165.0, 165.0),
            (-360.0, 360.0),
            (-360.0, 360.0),
            (-360.0, 360.0),
        ),
        default_ready_pose_deg=(0.0, -20.0, 110.0, 0.0, 90.0, 0.0),
        max_joint_speed_deg_s=60.0,
    ),
    # Future models: add a spec here plus a registration shim module, e.g.
    # "rb5": RbModelSpec(...),
    # "rb3": RbModelSpec(...),
}
