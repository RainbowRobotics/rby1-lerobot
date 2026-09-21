import numpy as np
import pytest

from lerobot_teleoperator_rb.constants import (
    DEFAULT_READY_POSE_RAD,
    RB10E_JOINT_LIMITS_RAD,
    VR_HOME_POSE_DEG,
    VR_HOME_POSE_RAD,
)
from lerobot_teleoperator_rb.rb10e_kinematics import RB10E


def home_kinematics() -> RB10E:
    kinematics = RB10E()
    kinematics.set_q(np.asarray(VR_HOME_POSE_RAD))
    return kinematics


def test_vr_home_pose_values():
    assert VR_HOME_POSE_DEG == (
        -172.5253,
        -3.3101,
        141.8867,
        41.4234,
        277.4747,
        180.0,
    )


def test_vr_home_pose_shoulder_sum_fixes_the_orientation():
    """j1 + j2 + j3 == 180 is what keeps the flange axis-aligned.

    Joints 1-3 are a planar chain: their SUM sets the flange pitch while
    their individual values slide the wrist along that plane. Rounding the
    three independently breaks the sum and tilts the flange, which is the
    likely failure mode next time this pose is re-tuned.

    It is 180 and not 0 because the taught pose is WRIST-FLIPPED: j5 sits at
    180 deg, and the extra half turn has to be paid back somewhere.
    """
    assert sum(VR_HOME_POSE_DEG[1:4]) == pytest.approx(180.0, abs=1e-9)


def test_vr_home_pose_yaw_difference_fixes_the_orientation():
    """j0 - j4 == -450 is the other half of keeping the flange axis-aligned.

    Joint 0 turns about base +Z and joint 4 about the flange spin axis, which
    this pose parks on base +X. With the tool horizontal and the wrist
    flipped the two trade against each other freely as long as their
    DIFFERENCE holds, so only that is pinned -- which is why j0 is not a
    round -180 here. Spending that freedom is what lets the arm keep the
    taught end-effector position while the tool snaps onto the base axes.

    Watch the sign. The previous, j5 == 0 home pose pinned j0 + j4 instead.
    Carrying that rule over to this wrist-flipped pose tilts the flange.
    """
    assert (
        VR_HOME_POSE_DEG[0] - VR_HOME_POSE_DEG[4]
    ) == pytest.approx(-450.0, abs=1e-9)


def test_vr_home_pose_wrist_is_flipped():
    """j5 == 180 is the third constraint, and the reason for the other two."""
    assert VR_HOME_POSE_DEG[5] == pytest.approx(180.0, abs=1e-9)


def test_vr_home_pose_ee_position():
    """The taught end-effector position, at the working height.

    get_fk() reports the GRIPPER TIP: this model carries the gripper inside
    its last link (259.3 mm from the wrist against the bare flange's
    115.9 mm), so the control box's tcp_pos reads 143.44 mm back along the
    tool +Y column from here.
    """
    position = home_kinematics().get_fk()[:3, 3]

    assert position == pytest.approx(
        [-618.57, 110.45, 264.01],
        abs=0.01,
    )


def test_vr_home_pose_ee_xy_plane_is_level_with_the_base_xy_plane():
    """ZERO TILT. This is the constraint the pose was re-cut for.

    The hand-guided teach sat 3.21 deg out of level -- the control box read
    rx = 1.4627, ry = 2.9902 -- which tilts the plane the operator's
    horizontal hand motion is mapped into. The EE +Z column must be base +Z
    exactly, not nearly.
    """
    rotation = home_kinematics().get_fk()[:3, :3]

    assert np.allclose(rotation[:, 2], [0.0, 0.0, 1.0], atol=1e-12)
    assert np.allclose(rotation[2, :], [0.0, 0.0, 1.0], atol=1e-12)


def test_vr_home_pose_ee_orientation_is_the_base_frame_yawed_minus_90():
    """The home EE orientation must be exactly Rz(-90 deg).

    RbVr anchors here and then rotates away from this orientation by a
    scaled fraction of the hand's rotation, so this is the reference every
    stroke is measured from, not merely a starting point. A pose that is
    "nearly" aligned would tilt that reference in every recorded episode.

    The -90 deg of yaw is what frame_transforms.R_OPERATOR_TO_BASE cancels
    -- see test_operator_to_base_maps_operator_axes_onto_the_home_ee_axes.
    The two constants have to be re-derived together.
    """
    rotation = home_kinematics().get_fk()[:3, :3]

    assert np.allclose(
        rotation,
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        atol=1e-12,
    )


def test_vr_home_pose_tool_y_is_exactly_base_x():
    """The tool +Y axis must coincide with base +X with zero offset.

    Gripper approach is -fk[:, 1], not the Z column, so this pins the
    approach to base -X: horizontal, pointing away from the base column.
    See the frame convention note in RB10E.update_fk_and_jcbn.
    """
    rotation = home_kinematics().get_fk()[:3, :3]

    assert np.allclose(
        rotation[:, 1],
        [1.0, 0.0, 0.0],
        atol=1e-12,
    )

    assert np.allclose(
        -rotation[:, 1],
        [-1.0, 0.0, 0.0],
        atol=1e-12,
    )


def test_vr_home_pose_is_in_the_working_volume():
    """Well away from the base column and at roughly table height.

    Joint 0 is near -180 deg, so the arm parks along base -X rather than +X.
    Measured: reach 628 mm, z 264 mm -- the taught pose works lower and
    further out than the previous home did (580 mm, 318 mm).
    """
    position = home_kinematics().get_fk()[:3, 3]

    reach_mm = float(np.linalg.norm(position[:2]))

    assert 300.0 < reach_mm < 900.0
    assert 200.0 < position[2] < 700.0
    assert position[0] < -350.0


def test_vr_home_pose_is_not_near_a_singularity():
    """Measured: smallest singular value 0.212, condition number 9.32.

    The wrist is comfortable -- j4 == 277.47 deg is 82.5 deg from the
    j4 == 360 deg singularity -- so what this bound really guards is the
    arm folding up as the pose is re-tuned. The joint that runs out first
    is j2, and test_vr_home_pose_is_within_joint_limits covers that.
    """
    jacobian = home_kinematics().get_jacobian()

    # Rows 0:3 are mm/rad and rows 3:6 are dimensionless, so rescale the
    # linear block to metres before taking a condition number.
    scaled = jacobian.copy()
    scaled[:3, :] /= 1000.0

    singular_values = np.linalg.svd(scaled, compute_uv=False)

    assert singular_values.max() / singular_values.min() < 20.0


def test_vr_home_pose_is_not_the_default_ready_pose():
    assert not np.allclose(
        np.asarray(VR_HOME_POSE_RAD),
        np.asarray(DEFAULT_READY_POSE_RAD),
    )


def test_vr_home_pose_is_within_joint_limits():
    limits = np.asarray(RB10E_JOINT_LIMITS_RAD)

    assert np.all(np.asarray(VR_HOME_POSE_RAD) >= limits[:, 0])
    assert np.all(np.asarray(VR_HOME_POSE_RAD) <= limits[:, 1])


def test_vr_home_pose_is_within_the_rb10_model_joint_limits():
    from lerobot_robot_rb.models import MODEL_SPECS

    limits = np.asarray(MODEL_SPECS["rb10"].joint_limits_deg)

    assert np.all(np.asarray(VR_HOME_POSE_DEG) >= limits[:, 0])
    assert np.all(np.asarray(VR_HOME_POSE_DEG) <= limits[:, 1])
