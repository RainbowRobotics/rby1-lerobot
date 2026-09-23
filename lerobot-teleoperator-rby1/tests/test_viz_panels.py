import math
import threading
import time

import numpy as np
import pytest

from lerobot_teleoperator_rby1.isaac_teleop.viz_panels import CameraPanels, PanelLayout, xr_panel_pose


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeLayer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.submits = []
        self.placements = []

    def submit(self, frame):
        self.submits.append(frame)

    def set_placement(self, placement):
        self.placements.append(placement)


class FakeVizSession:
    def __init__(self, cfg):
        self.cfg = cfg
        self.layers = []
        self.renders = 0
        self.destroyed = False
        self.close = False
        self.head = _Obj(position=(0.0, 1.6, 0.0), orientation=(1.0, 0.0, 0.0, 0.0))

    def add_quad_layer(self, cfg):
        assert threading.current_thread().name == "televiz-render"
        layer = FakeLayer(cfg)
        self.layers.append(layer)
        return layer

    def render(self):
        self.renders += 1
        time.sleep(0.001)
        return _Obj(should_render=True, reference_space_changed=False)

    def should_close(self):
        return self.close

    def head_pose_now(self):
        return self.head

    def get_oxr_handles(self):
        return (1, 2, 3, 4)

    def get_frame_timing_stats(self):
        return _Obj(fps=72.0, stale_layers=[])

    def destroy(self):
        assert threading.current_thread().name == "televiz-render"
        self.destroyed = True


class FakeTeleviz:
    """Minimal stand-in for the `isaacteleop.viz` module."""

    class DisplayMode:
        kXr = 2

    class PixelFormat:
        kRGBA8 = 0

    class VizSessionConfig(_Obj):
        pass

    class QuadLayerConfig(_Obj):
        pass

    class Resolution:
        def __init__(self, w, h):
            self.w, self.h = w, h

    class Pose3D:
        def __init__(self, position, orientation):
            self.position, self.orientation = position, orientation

    class QuadLayerPlacement:
        def __init__(self, pose, size_meters):
            self.pose, self.size_meters = pose, size_meters

    class VizSession:
        last = None

        @staticmethod
        def create(cfg):
            FakeTeleviz.VizSession.last = FakeVizSession(cfg)
            return FakeTeleviz.VizSession.last


def test_xr_panel_pose_gimbal_and_head():
    lay = PanelLayout("front", offset_x=0.5, offset_y=0.1, distance=1.5)
    pos, q = xr_panel_pose(np.array([0, 1.6, 0]), np.array([1, 0, 0, 0]), lay, "gimbal")
    np.testing.assert_allclose(pos, [0.5, 1.7, -1.5], atol=1e-9)
    np.testing.assert_allclose(q, [1, 0, 0, 0], atol=1e-9)
    # Head yawed 90 deg to the left (facing -X): the panel sits at -X, offset to the "right" = -Z.
    yaw = math.pi / 2
    q_head = np.array([math.cos(yaw / 2), 0, math.sin(yaw / 2), 0])
    pos, q = xr_panel_pose(np.array([0, 1.6, 0]), q_head, lay, "gimbal")
    np.testing.assert_allclose(pos, [-1.5, 1.7, -0.5], atol=1e-9)
    np.testing.assert_allclose(q, q_head, atol=1e-9)
    # Full head lock uses the head orientation for the offset too.
    pos_h, q_h = xr_panel_pose(np.array([0, 1.6, 0]), q_head, lay, "head")
    np.testing.assert_allclose(pos_h, [-1.5, 1.7, -0.5], atol=1e-9)
    np.testing.assert_allclose(q_h, q_head, atol=1e-9)


def test_camera_panels_thread_lifecycle_and_submits():
    frames = {}
    seq = {"n": 0}

    def source(name):
        return frames.get(name)

    def push(name, h, w):
        seq["n"] += 1
        frames[name] = (np.zeros((h, w, 3), np.uint8), time.monotonic(), seq["n"])

    panels = CameraPanels(
        [PanelLayout("front", 0.0), PanelLayout("left", -1.1), PanelLayout("right", 1.1)],
        source,
        lock_mode="gimbal",
        uploader=lambda f: ("uploaded", f.shape),
        viz_module=FakeTeleviz,
        daemon=True,
    )
    session = panels.create_session("test", ["XR_EXT_x"], -1)
    assert session.cfg.required_extensions == ["XR_EXT_x"] and session.cfg.xr_system_wait_seconds == -1
    assert panels.oxr_handles() == (1, 2, 3, 4)
    panels.start_render_thread()
    time.sleep(0.05)
    assert session.renders == 0  # nothing before begin_rendering()
    panels.begin_rendering()
    push("front", 480, 640)
    push("left", 480, 640)
    deadline = time.time() + 2.0
    while time.time() < deadline and len(session.layers) < 2:
        time.sleep(0.01)
    assert [l.cfg.name for l in session.layers] == ["front", "left"]  # lazy, in layout order; 'right' has no frame
    assert session.layers[0].cfg.resolution.w == 640 and session.layers[0].cfg.resolution.h == 480
    assert session.layers[0].cfg.openxr_composition is False
    time.sleep(0.05)
    n_front = len(session.layers[0].submits)
    assert n_front == 1  # one submit per new frame (latest-wins), not per render
    push("front", 480, 640)
    time.sleep(0.05)
    assert len(session.layers[0].submits) == 2
    assert session.layers[0].submits[-1] == ("uploaded", (480, 640, 3))
    # Placement follows the head (gimbal): front centred 1.5 m ahead at head height.
    pl = session.layers[0].placements[-1]
    np.testing.assert_allclose(pl.pose.position, [0.0, 1.6, -1.5], atol=1e-9)
    np.testing.assert_allclose(pl.size_meters, [1.0, 0.75], atol=1e-9)
    assert session.renders > 0
    assert panels.stop_rendering(timeout=2.0)
    assert not session.destroyed
    assert panels.destroy(timeout=2.0)
    assert session.destroyed and panels.session is None


def test_camera_panels_detects_session_loss():
    panels = CameraPanels([PanelLayout("front")], lambda n: None, viz_module=FakeTeleviz, daemon=True)
    session = panels.create_session("t", [], 0)
    panels.start_render_thread()
    panels.begin_rendering()
    session.close = True
    deadline = time.time() + 2.0
    while time.time() < deadline and not panels.lost:
        time.sleep(0.01)
    assert panels.lost
    assert panels.destroy(timeout=2.0)


def test_status_string():
    panels = CameraPanels([PanelLayout("front")], lambda n: None, viz_module=FakeTeleviz)
    assert panels.status().startswith("viz: 0/1 panels")
    with pytest.raises(RuntimeError):
        panels.start_render_thread()
