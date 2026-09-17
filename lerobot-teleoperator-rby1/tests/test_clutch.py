import numpy as np
from scipy.spatial.transform import Rotation

from lerobot_teleoperator_rby1.isaac_teleop.clutch import Clutch

IDENT = np.array([0.0, 0.0, 0.0, 1.0])


def _home(pos=(0.4, -0.2, 0.9), rot=None):
    T = np.eye(4)
    if rot is not None:
        T[:3, :3] = rot
    T[:3, 3] = pos
    return T


def test_engage_frame_outputs_home_and_translation_is_one_to_one():
    c = Clutch(_home())
    c.engage([1.0, 2.0, 3.0], IDENT, _home())
    pos, quat = c.rebase([1.0, 2.0, 3.0], IDENT)
    np.testing.assert_allclose(pos, [0.4, -0.2, 0.9])
    np.testing.assert_allclose(quat, IDENT)
    pos, _ = c.rebase([1.1, 2.0, 3.05], IDENT)
    np.testing.assert_allclose(pos, [0.5, -0.2, 0.95])


def test_rotation_delta_is_left_composed_in_base_frame():
    home_rot = Rotation.from_euler("x", 90, degrees=True).as_matrix()
    c = Clutch(_home(rot=home_rot))
    c.engage([0, 0, 0], IDENT, _home(rot=home_rot))
    yaw = Rotation.from_euler("z", 30, degrees=True)
    _, quat = c.rebase([0, 0, 0], yaw.as_quat())
    expected = (yaw * Rotation.from_matrix(home_rot)).as_matrix()
    np.testing.assert_allclose(Rotation.from_quat(quat).as_matrix(), expected, atol=1e-9)


def test_measured_vs_commanded_orientation_latch():
    c = Clutch(_home())
    sagged = _home(pos=(0.4, -0.2, 0.85), rot=Rotation.from_euler("y", 10, degrees=True).as_matrix())
    c.engage([0, 0, 0], IDENT, sagged, latch_orientation="measured")
    np.testing.assert_allclose(c.home, sagged)
    c2 = Clutch(_home())
    c2.engage([0, 0, 0], IDENT, sagged, latch_orientation="commanded")
    np.testing.assert_allclose(c2.home[:3, 3], sagged[:3, 3])  # position from measurement
    np.testing.assert_allclose(c2.home[:3, :3], np.eye(3))  # orientation from last command


def test_disengage_holds_last_commanded():
    c = Clutch(_home())
    c.engage([0, 0, 0], IDENT, _home())
    c.rebase([0.2, 0, 0], IDENT)
    c.disengage()
    assert not c.engaged
    np.testing.assert_allclose(c.last_commanded[:3, 3], [0.6, -0.2, 0.9])
    # Re-engage without a measurement latches the held command (no jump).
    c.engage([5, 5, 5], IDENT)
    np.testing.assert_allclose(c.home[:3, 3], [0.6, -0.2, 0.9])
