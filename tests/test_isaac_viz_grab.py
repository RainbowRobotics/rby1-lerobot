import json
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lerobot_teleoperator_rby1.isaac_teleop.viz_grab import (
    HandInput,
    PanelGrabber,
    QuadGeom,
    load_layouts,
    ray_quad_hit,
    save_layouts,
)
from lerobot_teleoperator_rby1.isaac_teleop.viz_panels import (
    PanelLayout,
    face_orientation,
    layout_from_center,
    xr_panel_pose,
)


def _quad(center, size=(1.0, 0.06), R=None):
    return QuadGeom(np.asarray(center, float), np.eye(3) if R is None else R, size)


def _hand(origin, target, trigger=0.0, free=True, thumb_y=0.0):
    o = np.asarray(origin, float)
    d = np.asarray(target, float) - o
    return HandInput(o, d / np.linalg.norm(d), trigger, free, thumb_y)


def test_ray_quad_hit_inside_outside_behind():
    quad = _quad((0, 2.0, -1.5))  # facing +Z (towards the viewer at the origin)
    assert ray_quad_hit(np.array([0, 1.6, 0.0]), _hand((0, 1.6, 0), (0, 2.0, -1.5)).direction, quad) == pytest.approx(
        math.hypot(0.4, 1.5)
    )
    assert ray_quad_hit(np.array([0, 1.6, 0.0]), _hand((0, 1.6, 0), (0.6, 2.0, -1.5)).direction, quad) is None  # outside (w/2 = 0.5)
    assert ray_quad_hit(np.array([0, 1.6, 0.0]), _hand((0, 1.6, 0), (0, 2.05, -1.5)).direction, quad) is None  # above the bar
    assert ray_quad_hit(np.array([0, 1.6, 0.0]), np.array([0, 0, 1.0]), quad) is None  # pointing away
    assert ray_quad_hit(np.array([0, 1.6, 0.0]), np.array([1.0, 0, 0]), quad) is None  # parallel


def test_grab_drag_release_moves_center_by_ray_point():
    g = PanelGrabber(grab_threshold=0.7, release_threshold=0.3, push_rate_mps=1.0)
    handles = {"front": _quad((0, 2.0, -1.5))}
    centers = {"front": np.array([0, 1.6, -1.5])}
    origin = (0, 1.6, 0)
    # Hover only.
    moved, rel = g.step({"right": _hand(origin, (0, 2.0, -1.5), 0.0), "left": None}, handles, centers, 0.02)
    assert moved == {} and rel == [] and g.hover["right"] == "front" and g.grabbing["right"] is None
    # Trigger rises -> grab.
    moved, rel = g.step({"right": _hand(origin, (0, 2.0, -1.5), 0.9), "left": None}, handles, centers, 0.02)
    assert g.grabbing["right"] == "front" and moved == {} or "front" in moved
    # Move the ray 0.3 m to the right (same distance): the centre follows by the same vector.
    d0 = math.hypot(0.4, 1.5)
    dir2 = np.asarray([0.3, 0.4, -1.5]) / np.linalg.norm([0.3, 0.4, -1.5])
    moved, rel = g.step({"right": HandInput(np.asarray(origin, float), dir2, 0.9, True), "left": None}, handles, centers, 0.02)
    point = np.asarray(origin) + dir2 * d0
    np.testing.assert_allclose(moved["front"], point + (centers["front"] - np.array([0, 2.0, -1.5])), atol=1e-9)
    # Push with the thumbstick: distance grows by rate * dt.
    moved, _ = g.step({"right": HandInput(np.asarray(origin, float), dir2, 0.9, True, thumb_y=1.0), "left": None}, handles, centers, 0.1)
    point2 = np.asarray(origin) + dir2 * (d0 + 0.1)
    np.testing.assert_allclose(moved["front"], point2 + (centers["front"] - np.array([0, 2.0, -1.5])), atol=1e-9)
    # Release.
    moved, rel = g.step({"right": HandInput(np.asarray(origin, float), dir2, 0.1, True), "left": None}, handles, centers, 0.02)
    assert rel == ["front"] and moved == {} and g.grabbing["right"] is None


def test_grab_requires_free_hand_and_rising_edge():
    g = PanelGrabber()
    handles = {"front": _quad((0, 2.0, -1.5))}
    centers = {"front": np.array([0, 1.6, -1.5])}
    origin = (0, 1.6, 0)
    # Clutched (not free) hand: hover but no grab.
    g.step({"right": _hand(origin, (0, 2.0, -1.5), 0.9, free=False), "left": None}, handles, centers, 0.02)
    assert g.hover["right"] == "front" and g.grabbing["right"] is None
    # Trigger already held when the ray arrives on the bar: no grab (needs a rising edge).
    g2 = PanelGrabber()
    g2.step({"right": _hand(origin, (5, 5, -1.5), 0.9), "left": None}, handles, centers, 0.02)  # miss, trigger held
    g2.step({"right": _hand(origin, (0, 2.0, -1.5), 0.9), "left": None}, handles, centers, 0.02)  # now on the bar
    assert g2.grabbing["right"] is None
    g2.step({"right": _hand(origin, (0, 2.0, -1.5), 0.1), "left": None}, handles, centers, 0.02)
    g2.step({"right": _hand(origin, (0, 2.0, -1.5), 0.9), "left": None}, handles, centers, 0.02)
    assert g2.grabbing["right"] == "front"
    # Tracking loss releases.
    _, rel = g2.step({"right": None, "left": None}, handles, centers, 0.02)
    assert rel == ["front"] and g2.grabbing["right"] is None


def test_one_hand_per_panel_and_nearest_handle():
    g = PanelGrabber()
    handles = {"front": _quad((0, 2.0, -1.5)), "near": _quad((0, 1.8, -0.75), size=(0.4, 0.06))}
    centers = {"front": np.array([0, 1.6, -1.5]), "near": np.array([0, 1.5, -0.75])}
    origin = (0, 1.6, 0)
    # The ray towards the far bar also crosses the near bar first -> near wins.
    g.step({"right": _hand(origin, (0, 1.8, -0.75), 0.9), "left": None}, handles, centers, 0.02)
    assert g.grabbing["right"] == "near"
    # The other hand cannot grab a panel that is already held (its ray misses the far bar).
    g.step(
        {"right": _hand(origin, (0, 1.8, -0.75), 0.9), "left": _hand((0.1, 1.6, 0), (0.1, 1.82, -0.75), 0.9)},
        handles, centers, 0.02,
    )
    assert g.grabbing["left"] is None and g.hover["left"] is None


def test_layout_from_center_round_trip_all_modes():
    head_p = np.array([0.2, 1.6, 0.1])
    x, y, z, w = (Rotation.from_euler("y", 40, degrees=True) * Rotation.from_euler("x", -20, degrees=True) * Rotation.from_euler("z", 15, degrees=True)).as_quat()
    head_q = np.array([w, x, y, z])
    lay = PanelLayout("front", offset_x=-0.7, offset_y=0.25, distance=1.8, width=0.9)
    for mode, pitch in (("gimbal", True), ("gimbal", False), ("head", True)):
        pos, _ = xr_panel_pose(head_p, head_q, lay, mode, pitch)
        back = layout_from_center(np.asarray(pos), head_p, head_q, PanelLayout("front"), mode, pitch)
        assert (back.offset_x, back.offset_y, back.distance) == pytest.approx((lay.offset_x, lay.offset_y, lay.distance), abs=1e-9)
        assert back.width == 1.0  # width is not part of the centre -> unchanged from the given layout
    # Clamps.
    far = layout_from_center(head_p + np.array([0, 0, -50.0]), head_p, np.array([1, 0, 0, 0]), lay, "gimbal", False)
    assert far.distance == 5.0


def test_face_orientation_turns_side_panels_towards_the_head():
    head_p = np.array([0, 1.6, 0])
    q = face_orientation(np.array([1.1, 1.6, -1.5]), head_p, np.array([1, 0, 0, 0]), follow_pitch=False)
    w, x, y, z = q
    R = Rotation.from_quat([x, y, z, w]).as_matrix()
    normal = R[:, 2]  # +Z of the quad points back at the viewer
    expect = head_p - np.array([1.1, 1.6, -1.5])
    np.testing.assert_allclose(normal, expect / np.linalg.norm(expect), atol=1e-9)
    assert abs(R[1, 0]) < 1e-9  # right vector stays horizontal (no roll)
    # Centre panel: identical to the plain gimbal orientation.
    np.testing.assert_allclose(face_orientation(np.array([0, 1.6, -1.5]), head_p, np.array([1, 0, 0, 0]), False), [1, 0, 0, 0], atol=1e-9)


def test_layout_json_round_trip(tmp_path):
    path = str(tmp_path / "layout.json")
    layouts = [PanelLayout("front", 0.1, 0.2, 1.7, 0.8), PanelLayout("left", -1.0)]
    assert save_layouts(path, layouts)
    data = json.loads((tmp_path / "layout.json").read_text())
    assert data["front"] == {"offset_x": 0.1, "offset_y": 0.2, "distance": 1.7, "width": 0.8}
    # Only matching names are applied; unknown cameras keep their defaults.
    loaded = load_layouts(path, [PanelLayout("front"), PanelLayout("right", 1.1)])
    assert loaded[0] == PanelLayout("front", 0.1, 0.2, 1.7, 0.8) and loaded[1] == PanelLayout("right", 1.1)
    assert load_layouts("", layouts) == layouts and not save_layouts("", layouts)
    (tmp_path / "layout.json").write_text("not json")
    assert load_layouts(path, layouts) == layouts
