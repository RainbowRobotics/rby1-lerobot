import numpy as np
from scipy.spatial.transform import Rotation

from lerobot_teleoperator_rby1.isaac_teleop.xr_frame import (
    OUT_BODY,
    OUT_CONTROLLER_LEFT,
    OUT_CONTROLLER_RIGHT,
    OUT_HEAD,
    BodyJointIndex,
    ButtonEdge,
    ControllerInputIndex,
    frame_from_outputs,
)
from tests import fakes


def test_full_frame_parses_all_sources():
    body_pos = np.zeros((24, 3), np.float32)
    body_pos[BodyJointIndex.SPINE3] = [1, 2, 3]
    outputs = {
        OUT_CONTROLLER_RIGHT: fakes.controller((0.1, 0.2, 0.3), squeeze=0.9, trigger=0.4, thumb=(0.5, -0.5), primary=True),
        OUT_CONTROLLER_LEFT: fakes.ABSENT,
        OUT_HEAD: fakes.head((0, 0, 1.6)),
        OUT_BODY: fakes.body(positions=body_pos),
    }
    f = frame_from_outputs(outputs, want_body=True)
    assert f.left is None
    np.testing.assert_allclose(f.right.position, [0.1, 0.2, 0.3])
    assert f.right.squeeze == 0.9 and f.right.trigger == 0.4 and f.right.primary and not f.right.secondary
    np.testing.assert_allclose(f.right.thumbstick, [0.5, -0.5])
    np.testing.assert_allclose(f.head.position, [0, 0, 1.6])
    assert f.head.is_tracked
    np.testing.assert_allclose(f.body.positions[BodyJointIndex.SPINE3], [1, 2, 3])
    assert f.body.valid.all()
    assert f.any_controller


def test_invalid_or_partial_controller_is_dropped():
    outputs = {OUT_CONTROLLER_RIGHT: fakes.controller(valid=False)}
    assert frame_from_outputs(outputs, want_body=False).right is None
    partial = fakes.FakeGroup({ControllerInputIndex.GRIP_IS_VALID: True, ControllerInputIndex.SQUEEZE_VALUE: 1.0})
    assert frame_from_outputs({OUT_CONTROLLER_RIGHT: partial}, want_body=False).right is None
    assert frame_from_outputs({}, want_body=True).body is None
    assert frame_from_outputs({OUT_HEAD: fakes.head(valid=False)}, want_body=False).head is None


def test_body_python_transform_matches_left_multiply():
    T = np.eye(4)
    T[:3, :3] = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)
    pos = np.zeros((24, 3), np.float32)
    pos[BodyJointIndex.PELVIS] = [0, 1, -2]  # anchor: 1 up, 2 forward (-Z)
    quat = np.tile(Rotation.from_euler("y", 90, degrees=True).as_quat().astype(np.float32), (24, 1))
    f = frame_from_outputs({OUT_BODY: fakes.body(positions=pos, orientations=quat)}, want_body=True, body_transform=T)
    np.testing.assert_allclose(f.body.positions[BodyJointIndex.PELVIS], [2, 0, 1], atol=1e-6)
    expected = (Rotation.from_matrix(T[:3, :3]) * Rotation.from_euler("y", 90, degrees=True)).as_matrix()
    np.testing.assert_allclose(f.body.joint_pose(BodyJointIndex.PELVIS)[:3, :3], expected, atol=1e-6)


def test_button_edges_fire_once_per_press():
    edges = ButtonEdge()
    held = frame_from_outputs({OUT_CONTROLLER_RIGHT: fakes.controller(primary=True)}, want_body=False)
    released = frame_from_outputs({OUT_CONTROLLER_RIGHT: fakes.controller()}, want_body=False)
    assert edges.update(held)["right_a"]
    assert not edges.update(held)["right_a"]
    assert not edges.update(released)["right_a"]
    assert edges.update(held)["right_a"]
    gone = frame_from_outputs({}, want_body=False)
    assert not any(edges.update(gone).values())


def test_head_without_valid_or_tracked_fields_is_accepted():
    from lerobot_teleoperator_rby1.isaac_teleop.xr_frame import HeadInputIndex

    three_field = fakes.FakeGroup(
        {
            HeadInputIndex.POSITION: np.array([0, 0, 1.5], np.float32),
            HeadInputIndex.ORIENTATION: np.array([0, 0, 0, 1], np.float32),
        }
    )
    f = frame_from_outputs({OUT_HEAD: three_field}, want_body=False)
    assert f.head is not None and f.head.is_tracked
