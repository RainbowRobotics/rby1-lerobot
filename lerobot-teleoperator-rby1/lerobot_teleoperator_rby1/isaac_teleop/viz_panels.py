"""Robot camera panels in the headset via Isaac Teleop's Televiz (`isaacteleop.viz`).

One ``QuadLayer`` per camera, fed from the in-process
:mod:`lerobot_robot_rby1.frame_bus` (published by ``Rby1.get_observation()``).
Televiz owns the graphics OpenXR session; the tracking ``TeleopSession``
attaches to it through ``oxr_handles`` (see ``base.py``).

Threading (per the Televiz docs): layers are added and ``render()`` is
called on a dedicated render thread; ``submit()`` also happens there so each
layer has exactly one producer. ``destroy()`` runs on that same thread and
is skipped when the thread cannot be joined (use-after-free rule).
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np
from scipy.spatial.transform import Rotation

from .viz_grab import HANDS, HandInput, PanelGrabber, QuadGeom

logger = logging.getLogger(__name__)

# Grab handle bar above each panel (see viz_grab.py): texture and size.
HANDLE_RES = (256, 24)          # (w, h) px
HANDLE_HEIGHT_M = 0.06
HANDLE_GAP_M = 0.03
HANDLE_COLORS = {"idle": (90, 90, 90), "hover": (200, 200, 200), "grab": (255, 180, 0)}

FrameSource = Callable[[str], "tuple[Any, float, int] | None"]
Uploader = Callable[[np.ndarray], Any]


@dataclass(frozen=True)
class PanelLayout:
    name: str
    offset_x: float = 0.0   # m, right (+) of the view centre
    offset_y: float = 0.0   # m, up (+)
    distance: float = 1.5   # m in front of the viewer
    width: float = 1.0      # m; height follows the frame aspect ratio


CUDA_UPLOAD_HINT = (
    "Televiz needs the camera frames as CUDA arrays and no working upload backend was "
    "found. On Jetson (JetPack 6.x = CUDA 12.6) the PyPI torch wheels are built for CUDA "
    "12.8+ and NVIDIA's JetPack torch wheels exist only for Python 3.10, so install the "
    "driver-API backend instead: `pip install \"cuda-bindings>=12.6,<13\"` (12.x driver API, "
    "cp312 aarch64 wheels; only libcuda is needed), or `pip install cupy-cuda12x`."
)


def _rgba_host(frame: np.ndarray) -> np.ndarray:
    frame = np.ascontiguousarray(frame)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return frame
    rgba = np.empty((*frame.shape[:2], 4), np.uint8)
    rgba[..., :3] = frame
    rgba[..., 3] = 255
    return rgba


def _upload_torch(frame: np.ndarray, _cache: dict = {}) -> Any:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    t = torch.from_numpy(np.ascontiguousarray(frame)).to("cuda", non_blocking=False)
    if t.shape[-1] == 4:
        return t.contiguous()
    key = tuple(t.shape[:2])
    alpha = _cache.get(key)
    if alpha is None:
        alpha = torch.full((*key, 1), 255, dtype=torch.uint8, device="cuda")
        _cache[key] = alpha
    return torch.cat([t, alpha], dim=-1).contiguous()


def _upload_cupy(frame: np.ndarray) -> Any:
    import cupy

    return cupy.asarray(_rgba_host(frame))


class DriverDeviceArray:
    """A CUDA device buffer allocated through the driver API (``cuda-python``).

    Exposes ``__cuda_array_interface__`` so Televiz (or CuPy / torch) can read
    it. Only ``libcuda`` (the GPU driver) is needed — no CUDA toolkit runtime,
    which is what makes it work in a Python 3.12 environment on JetPack 6.x.
    """

    _ctx: Any = None
    _drv: Any = None

    def __init__(self, shape: tuple[int, ...]) -> None:
        drv = self._driver()
        self.shape = tuple(int(x) for x in shape)
        self.nbytes = int(np.prod(self.shape))
        err, ptr = drv.cuMemAlloc(self.nbytes)
        self._check(err, "cuMemAlloc")
        self._ptr = ptr

    @classmethod
    def _driver(cls) -> Any:
        if cls._drv is None:
            try:
                from cuda.bindings import driver as drv  # cuda-python >= 12.6
            except ImportError:
                from cuda import cuda as drv  # older cuda-python layout
            cls._check_static(drv, drv.cuInit(0)[0], "cuInit")
            err, dev = drv.cuDeviceGet(0)
            cls._check_static(drv, err, "cuDeviceGet")
            err, ctx = drv.cuDevicePrimaryCtxRetain(dev)
            cls._check_static(drv, err, "cuDevicePrimaryCtxRetain")
            cls._drv, cls._ctx = drv, ctx
        # The primary context must be current on the calling thread (render thread).
        cls._check_static(cls._drv, cls._drv.cuCtxSetCurrent(cls._ctx)[0], "cuCtxSetCurrent")
        return cls._drv

    @staticmethod
    def _check_static(drv: Any, err: Any, what: str) -> None:
        if err != drv.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"{what} failed: {err}")

    def _check(self, err: Any, what: str) -> None:
        self._check_static(self._drv, err, what)

    def upload(self, host: np.ndarray) -> "DriverDeviceArray":
        drv = self._driver()
        host = np.ascontiguousarray(host)
        if host.nbytes != self.nbytes:
            raise ValueError(f"frame has {host.nbytes} bytes, buffer {self.nbytes}")
        self._check(drv.cuMemcpyHtoD(self._ptr, host.ctypes.data, self.nbytes)[0], "cuMemcpyHtoD")
        return self

    @property
    def __cuda_array_interface__(self) -> dict:
        return {"shape": self.shape, "typestr": "|u1", "data": (int(self._ptr), False), "version": 3, "strides": None}

    def __del__(self) -> None:
        try:
            if self._drv is not None and getattr(self, "_ptr", None) is not None:
                self._drv.cuMemFree(self._ptr)
        except Exception:  # noqa: BLE001
            pass


_driver_buffers: dict[tuple[int, ...], DriverDeviceArray] = {}


def _upload_cuda_driver(frame: np.ndarray) -> Any:
    rgba = _rgba_host(frame)
    buf = _driver_buffers.get(rgba.shape)
    if buf is None:
        buf = DriverDeviceArray(rgba.shape)
        _driver_buffers[rgba.shape] = buf
    return buf.upload(rgba)


_uploader_impl: Callable[[np.ndarray], Any] | None = None


def rgb_to_rgba_cuda(frame: np.ndarray) -> Any:
    """Upload an (H, W, 3) uint8 RGB host frame as a contiguous (H, W, 4) CUDA array.

    Tries PyTorch first, then CuPy; the working backend is cached. Raises a
    RuntimeError with install guidance when neither can reach the GPU.
    """
    global _uploader_impl
    if _uploader_impl is not None:
        return _uploader_impl(frame)
    errors = []
    for name, fn in (("torch", _upload_torch), ("cuda-python", _upload_cuda_driver), ("cupy", _upload_cupy)):
        try:
            out = fn(frame)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {e.__class__.__name__}: {str(e).splitlines()[0][:160]}")
            continue
        _uploader_impl = fn
        logger.info("Televiz CUDA upload backend: %s", name)
        return out
    raise RuntimeError(CUDA_UPLOAD_HINT + " Tried: " + " | ".join(errors))


def check_cuda_upload(uploader: Callable[[np.ndarray], Any]) -> None:
    """Run one tiny upload so a broken CUDA stack fails at connect time, not per frame."""
    uploader(np.zeros((2, 2, 3), np.uint8))


def _quat_wxyz_from_matrix(R: np.ndarray) -> tuple[float, float, float, float]:  # noqa: N803
    qx, qy, qz, qw = Rotation.from_matrix(R).as_quat()
    if qw < 0.0:
        qx, qy, qz, qw = -qx, -qy, -qz, -qw
    return float(qw), float(qx), float(qy), float(qz)


def _levelled_basis(fwd: np.ndarray, R_head: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # noqa: N803
    """Right / up / forward with roll removed (right stays horizontal)."""
    up_world = np.array([0.0, 1.0, 0.0])
    n = float(np.linalg.norm(fwd))
    fwd = fwd / n if n > 1e-9 else np.array([0.0, 0.0, -1.0])
    right = np.cross(fwd, up_world)
    if np.linalg.norm(right) < 0.1:  # looking (almost) straight up / down
        right = R_head @ np.array([1.0, 0.0, 0.0])
        right[1] = 0.0
    right = right / max(float(np.linalg.norm(right)), 1e-9)
    up = np.cross(right, fwd)
    return right, up, fwd


def panel_basis(
    head_position: np.ndarray,
    head_orientation_wxyz: np.ndarray,
    lock_mode: str,
    follow_pitch: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(origin, right, up, forward)`` of the frame the panel offsets live in.

    ``head``: the full head frame. ``gimbal``: head position, yaw (and pitch
    when ``follow_pitch``), roll never. OpenXR space (Y up, forward -Z).
    """
    w, x, y, z = [float(v) for v in head_orientation_wxyz]
    R = Rotation.from_quat([x, y, z, w]).as_matrix()  # scipy = xyzw
    p = np.asarray(head_position, dtype=float)
    if lock_mode == "head":
        return p, R[:, 0].copy(), R[:, 1].copy(), -R[:, 2]
    fwd = R @ np.array([0.0, 0.0, -1.0])
    if not follow_pitch:
        fwd[1] = 0.0
    right, up, fwd = _levelled_basis(fwd, R)
    return p, right, up, fwd


def xr_panel_pose(
    head_position: np.ndarray,
    head_orientation_wxyz: np.ndarray,
    layout: PanelLayout,
    lock_mode: str,
    follow_pitch: bool = False,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Panel pose (position, orientation wxyz) in OpenXR space (Y up, forward -Z).

    ``gimbal``: follow the head position and yaw (plus pitch when
    ``follow_pitch``; roll is never followed). ``head``: full head lock.
    (``world`` callers compute this once and keep it.)
    """
    p, right, up, fwd = panel_basis(head_position, head_orientation_wxyz, lock_mode, follow_pitch)
    pos = p + fwd * layout.distance + right * layout.offset_x + up * layout.offset_y
    q = _quat_wxyz_from_matrix(np.column_stack([right, up, -fwd]))
    return tuple(pos.tolist()), q


def layout_from_center(
    center: np.ndarray,
    head_position: np.ndarray,
    head_orientation_wxyz: np.ndarray,
    layout: PanelLayout,
    lock_mode: str,
    follow_pitch: bool = False,
    distance_range: tuple[float, float] = (0.3, 5.0),
    offset_limit: float = 3.0,
) -> PanelLayout:
    """Inverse of :func:`xr_panel_pose`: the layout that puts the panel at ``center``."""
    p, right, up, fwd = panel_basis(head_position, head_orientation_wxyz, lock_mode, follow_pitch)
    d = np.asarray(center, dtype=float) - p
    return replace(
        layout,
        offset_x=float(np.clip(np.dot(d, right), -offset_limit, offset_limit)),
        offset_y=float(np.clip(np.dot(d, up), -offset_limit, offset_limit)),
        distance=float(np.clip(np.dot(d, fwd), distance_range[0], distance_range[1])),
    )


def face_orientation(
    center: np.ndarray, head_position: np.ndarray, head_orientation_wxyz: np.ndarray, follow_pitch: bool
) -> tuple[float, float, float, float]:
    """Orientation (wxyz) of a panel at ``center`` turned towards the head (roll-free)."""
    w, x, y, z = [float(v) for v in head_orientation_wxyz]
    R = Rotation.from_quat([x, y, z, w]).as_matrix()
    fwd = np.asarray(center, dtype=float) - np.asarray(head_position, dtype=float)
    if not follow_pitch:
        fwd[1] = 0.0
    right, up, fwd = _levelled_basis(fwd, R)
    return _quat_wxyz_from_matrix(np.column_stack([right, up, -fwd]))


def _se3(position: Any, orientation_wxyz: Any) -> np.ndarray:
    w, x, y, z = [float(v) for v in orientation_wxyz]
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    T[:3, 3] = np.asarray(position, dtype=float)
    return T


def _handle_image(state: str) -> np.ndarray:
    w, h = HANDLE_RES
    img = np.empty((h, w, 4), np.uint8)
    img[..., :3] = HANDLE_COLORS[state]
    img[..., 3] = 255
    return img


class CameraPanels:
    """Render thread + layers for the camera panels. Inject fakes for tests.

    Besides the camera quads, an optional grab handle bar is placed above each
    panel (``grab_enabled``): controller aim rays hovering / squeezing the
    trigger on it drag the panel (see :mod:`viz_grab`). The teleoperator feeds
    the hands with :meth:`set_hands` every tick; the interaction itself runs
    on the render thread at the display rate.
    """

    def __init__(
        self,
        layouts: list[PanelLayout],
        frame_source: FrameSource,
        *,
        lock_mode: str = "gimbal",
        follow_pitch: bool = True,
        face_head: bool = True,
        openxr_composition: bool = False,
        uploader: Uploader | None = None,
        viz_module: Any | None = None,
        poll_period_s: float = 0.002,
        daemon: bool = False,
        grab_enabled: bool = False,
        grab_threshold: float = 0.7,
        grab_push_rate_mps: float = 0.5,
        frame_bridge: str = "auto",
        on_layout_changed: Callable[[list[PanelLayout]], None] | None = None,
    ) -> None:
        self._layouts_lock = threading.Lock()
        self._layouts: dict[str, PanelLayout] = {lay.name: lay for lay in layouts}
        self._source = frame_source
        self.lock_mode = lock_mode
        self.follow_pitch = follow_pitch
        self.face_head = face_head
        self.openxr_composition = openxr_composition
        self._upload = uploader or rgb_to_rgba_cuda
        self._viz = viz_module
        self._poll_period = poll_period_s
        self._daemon = daemon
        self.grab_enabled = grab_enabled
        self._grabber = (
            PanelGrabber(grab_threshold=grab_threshold, push_rate_mps=grab_push_rate_mps) if grab_enabled else None
        )
        self.frame_bridge = frame_bridge
        self._on_layout_changed = on_layout_changed

        self.session: Any = None
        self._layers: dict[str, Any] = {}
        self._handles: dict[str, Any] = {}
        self._handle_state: dict[str, str] = {}
        self._last_seq: dict[str, int] = {}
        self._world_pose: dict[str, tuple] = {}
        self._geoms: dict[str, QuadGeom] = {}       # handle quads (Televiz frame)
        self._centers: dict[str, np.ndarray] = {}   # panel centres (Televiz frame)
        self._last_head: tuple[np.ndarray, np.ndarray] | None = None
        self._hands_lock = threading.Lock()
        self._hands: dict[str, HandInput | None] = {h: None for h in HANDS}
        self._hands_head: np.ndarray | None = None
        self._hands_seq = 0
        self._hands_seen = 0
        self._bridge = np.eye(4)
        self._bridge_samples: list[np.ndarray] = []
        self._bridge_done = frame_bridge != "auto"
        self._last_grab_t: float | None = None
        self._thread: threading.Thread | None = None
        self._begin = threading.Event()
        self._stop = threading.Event()
        self._destroy = threading.Event()
        self._loop_exited = threading.Event()
        self.lost = False
        self.render_count = 0
        self.last_stats: Any = None
        self._warned_missing: set[str] = set()

    # ------------------------------------------------------------------
    # Layouts (thread-safe)
    # ------------------------------------------------------------------
    @property
    def layouts(self) -> list[PanelLayout]:
        return self.layouts_snapshot()

    def layouts_snapshot(self) -> list[PanelLayout]:
        with self._layouts_lock:
            return list(self._layouts.values())

    def layout(self, name: str) -> PanelLayout:
        with self._layouts_lock:
            return self._layouts[name]

    def set_layout(self, name: str, layout: PanelLayout) -> None:
        with self._layouts_lock:
            if name not in self._layouts:
                raise KeyError(name)
            self._layouts[name] = replace(layout, name=name)
        self._world_pose.pop(name, None)

    # ------------------------------------------------------------------
    # Hands (from the teleoperator tick)
    # ------------------------------------------------------------------
    def set_hands(self, hands: dict[str, HandInput | None], head_pose: np.ndarray | None = None) -> None:
        """Latest controller aim rays (XR anchor frame) and the head pose of the same tick."""
        with self._hands_lock:
            self._hands = {h: hands.get(h) for h in HANDS}
            self._hands_head = None if head_pose is None else np.asarray(head_pose, dtype=float).copy()
            self._hands_seq += 1

    def grabbing(self) -> dict[str, str | None]:
        """Panel name grabbed by each hand (None = not grabbing)."""
        if self._grabber is None:
            return {h: None for h in HANDS}
        return self._grabber.grabbing

    def hover(self) -> dict[str, str | None]:
        if self._grabber is None:
            return {h: None for h in HANDS}
        return dict(self._grabber.hover)

    def ui_active(self) -> dict[str, bool]:
        """Hands pointing at / holding a handle bar (their trigger means UI, not gripper)."""
        if self._grabber is None:
            return {h: False for h in HANDS}
        g, hv = self._grabber.grabbing, self._grabber.hover
        return {h: (g.get(h) is not None or hv.get(h) is not None) for h in HANDS}

    # ------------------------------------------------------------------
    def _televiz(self) -> Any:
        if self._viz is None:
            import isaacteleop.viz as televiz

            self._viz = televiz
        return self._viz

    def create_session(self, app_name: str, required_extensions: list[str], wait_headset_s: int) -> Any:
        """Create the XR VizSession (blocks until the headset connects when wait < 0)."""
        tv = self._televiz()
        cfg = tv.VizSessionConfig()
        cfg.mode = tv.DisplayMode.kXr
        cfg.app_name = app_name
        cfg.required_extensions = list(required_extensions)
        cfg.xr_system_wait_seconds = int(wait_headset_s)
        self.session = tv.VizSession.create(cfg)
        logger.info(
            "Televiz XR session created (%d camera panels, lock=%s, grab=%s).",
            len(self._layouts), self.lock_mode, "on" if self.grab_enabled else "off",
        )
        try:
            check_cuda_upload(self._upload)
        except Exception as e:  # noqa: BLE001
            self.session.destroy()
            self.session = None
            raise RuntimeError(f"Camera panel upload check failed: {e}") from e
        return self.session

    def oxr_handles(self) -> Any:
        return None if self.session is None else self.session.get_oxr_handles()

    # ------------------------------------------------------------------
    def start_render_thread(self) -> None:
        if self.session is None:
            raise RuntimeError("create_session() first")
        self._thread = threading.Thread(target=self._run, name="televiz-render", daemon=self._daemon)
        self._thread.start()

    def begin_rendering(self) -> None:
        """Release the frame loop (call after the TeleopSession is attached)."""
        self._begin.set()

    def stop_rendering(self, timeout: float = 5.0) -> bool:
        """Leave the frame loop (before the trackers detach). True when the loop exited."""
        self._stop.set()
        self._begin.set()
        return self._loop_exited.wait(timeout)

    def destroy(self, timeout: float = 5.0) -> bool:
        """Destroy the session on the render thread and join. False = thread stuck (not destroyed)."""
        self._stop.set()
        self._begin.set()
        self._destroy.set()
        if self._thread is None:
            if self.session is not None:
                self.session.destroy()
                self.session = None
            return True
        self._thread.join(timeout)
        if self._thread.is_alive():
            logger.error(
                "televiz render thread did not join within %.1fs; the VizSession is NOT "
                "destroyed (only that thread may destroy it).", timeout,
            )
            return False
        return True

    # ------------------------------------------------------------------
    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                if not self._begin.wait(0.05):
                    continue
                if self._stop.is_set():
                    break
                self._ensure_layers()
                self._submit_new_frames()
                self._apply_placements()
                self._grab_step()
                info = self.session.render()
                self.render_count += 1
                if getattr(info, "reference_space_changed", False):
                    self._world_pose.clear()
                    self._bridge_samples.clear()
                    self._bridge_done = self.frame_bridge != "auto"
                if self.render_count % 300 == 0:
                    try:
                        self.last_stats = self.session.get_frame_timing_stats()
                    except Exception:  # noqa: BLE001
                        pass
                if self.session.should_close():
                    self.lost = True
                    logger.warning("Televiz: the runtime asked the session to close (headset gone?).")
                    break
        except Exception:  # noqa: BLE001
            logger.exception("Televiz render loop failed")
            self.lost = True
        finally:
            self._loop_exited.set()
            self._destroy.wait()
            try:
                if self.session is not None:
                    self.session.destroy()
            except Exception:  # noqa: BLE001
                logger.exception("Televiz destroy failed")
            self.session = None

    def _ensure_layers(self) -> None:
        tv = self._televiz()
        for layout in self.layouts_snapshot():
            if layout.name in self._layers:
                continue
            item = self._source(layout.name)
            if item is None:
                continue
            frame = item[0]
            h, w = int(frame.shape[0]), int(frame.shape[1])
            cfg = tv.QuadLayerConfig()
            cfg.name = layout.name
            cfg.resolution = tv.Resolution(w, h)
            cfg.format = tv.PixelFormat.kRGBA8
            cfg.openxr_composition = self.openxr_composition
            cfg.placement = tv.QuadLayerPlacement(
                tv.Pose3D(position=(layout.offset_x, layout.offset_y, -layout.distance), orientation=(1.0, 0.0, 0.0, 0.0)),
                size_meters=(layout.width, layout.width * h / w),
            )
            self._layers[layout.name] = self.session.add_quad_layer(cfg)
            logger.info("Televiz panel '%s': %dx%d, %.2fx%.2f m at %.1f m.", layout.name, w, h, layout.width, layout.width * h / w, layout.distance)
            if self.grab_enabled:
                hcfg = tv.QuadLayerConfig()
                hcfg.name = f"{layout.name}.handle"
                hcfg.resolution = tv.Resolution(HANDLE_RES[0], HANDLE_RES[1])
                hcfg.format = tv.PixelFormat.kRGBA8
                hcfg.openxr_composition = self.openxr_composition
                hcfg.placement = tv.QuadLayerPlacement(
                    tv.Pose3D(
                        position=(layout.offset_x, layout.offset_y + layout.width * h / w / 2.0 + HANDLE_GAP_M + HANDLE_HEIGHT_M / 2.0, -layout.distance),
                        orientation=(1.0, 0.0, 0.0, 0.0),
                    ),
                    size_meters=(layout.width, HANDLE_HEIGHT_M),
                )
                self._handles[layout.name] = self.session.add_quad_layer(hcfg)
                self._set_handle_state(layout.name, "idle", force=True)

    def _set_handle_state(self, name: str, state: str, force: bool = False) -> None:
        if not force and self._handle_state.get(name) == state:
            return
        layer = self._handles.get(name)
        if layer is None:
            return
        try:
            layer.submit(self._upload(_handle_image(state)))
            self._handle_state[name] = state
        except Exception as e:  # noqa: BLE001
            key = f"{name}.handle"
            if key not in self._warned_missing:
                logger.error("Televiz submit failed for '%s': %s", key, e)
                self._warned_missing.add(key)

    def _submit_new_frames(self) -> None:
        for name, layer in self._layers.items():
            item = self._source(name)
            if item is None:
                continue
            frame, _t, seq = item
            if self._last_seq.get(name) == seq:
                continue
            self._last_seq[name] = seq
            try:
                layer.submit(self._upload(frame))
            except Exception as e:  # noqa: BLE001
                if name not in self._warned_missing:
                    logger.error("Televiz submit failed for '%s': %s", name, e)
                    self._warned_missing.add(name)

    def _apply_placements(self) -> None:
        if not self._layers:
            return
        tv = self._televiz()
        head = self.session.head_pose_now()
        if head is None:
            return
        hp = np.asarray(head.position, dtype=float)
        hq = np.asarray(head.orientation, dtype=float)
        self._last_head = (hp, hq)
        for layout in self.layouts_snapshot():
            layer = self._layers.get(layout.name)
            if layer is None:
                continue
            if self.lock_mode == "world":
                if layout.name not in self._world_pose:
                    self._world_pose[layout.name] = xr_panel_pose(hp, hq, layout, "gimbal", self.follow_pitch)
                pos, q = self._world_pose[layout.name]
            else:
                pos, q = xr_panel_pose(hp, hq, layout, self.lock_mode, self.follow_pitch)
            if self.face_head and self.lock_mode != "head":
                q = face_orientation(np.asarray(pos), hp, hq, self.follow_pitch)
            h, w = self._frame_hw(layout.name)
            size = (layout.width, layout.width * h / w)
            layer.set_placement(tv.QuadLayerPlacement(tv.Pose3D(position=pos, orientation=q), size_meters=size))
            center = np.asarray(pos, dtype=float)
            self._centers[layout.name] = center
            handle = self._handles.get(layout.name)
            if handle is not None:
                R = _se3((0, 0, 0), q)[:3, :3]
                hcenter = center + R[:, 1] * (size[1] / 2.0 + HANDLE_GAP_M + HANDLE_HEIGHT_M / 2.0)
                hsize = (layout.width, HANDLE_HEIGHT_M)
                handle.set_placement(tv.QuadLayerPlacement(tv.Pose3D(position=tuple(hcenter.tolist()), orientation=q), size_meters=hsize))
                self._geoms[layout.name] = QuadGeom(hcenter, R, hsize)

    # ------------------------------------------------------------------
    # Grab interaction (render thread)
    # ------------------------------------------------------------------
    def _update_bridge(self, head_teleop: np.ndarray | None) -> None:
        """Estimate Televiz-frame ← teleop-frame from simultaneous head poses (once)."""
        if self._bridge_done or head_teleop is None or self._last_head is None:
            return
        hp, hq = self._last_head
        T_viz = _se3(hp, hq)
        self._bridge_samples.append(T_viz @ np.linalg.inv(head_teleop))
        if len(self._bridge_samples) < 30:
            return
        Ts = np.stack(self._bridge_samples)
        R_mean = Rotation.from_matrix(Ts[:, :3, :3]).mean().as_matrix()
        t_mean = Ts[:, :3, 3].mean(axis=0)
        self._bridge = np.eye(4)
        self._bridge[:3, :3] = R_mean
        self._bridge[:3, 3] = t_mean
        self._bridge_done = True
        ang = math.degrees(float(np.linalg.norm(Rotation.from_matrix(R_mean).as_rotvec())))
        logger.info(
            "Televiz/teleop frame bridge: translation %.3f m, rotation %.1f deg (identity = same OpenXR space).",
            float(np.linalg.norm(t_mean)), ang,
        )

    def _grab_step(self) -> None:
        if self._grabber is None or not self._geoms or self._last_head is None:
            return
        with self._hands_lock:
            hands = dict(self._hands)
            head_teleop = self._hands_head
            seq = self._hands_seq
        now = time.monotonic()
        dt = 0.0 if self._last_grab_t is None else min(now - self._last_grab_t, 0.1)
        self._last_grab_t = now
        if seq != self._hands_seen:
            self._hands_seen = seq
            self._update_bridge(head_teleop)
        Rb, tb = self._bridge[:3, :3], self._bridge[:3, 3]
        hands_viz: dict[str, HandInput | None] = {}
        for hand, inp in hands.items():
            if inp is None:
                hands_viz[hand] = None
            else:
                hands_viz[hand] = replace(inp, origin=Rb @ inp.origin + tb, direction=Rb @ inp.direction)
        moved, released = self._grabber.step(hands_viz, self._geoms, self._centers, dt)
        hp, hq = self._last_head
        for name, center in moved.items():
            lay = self.layout(name)
            new = layout_from_center(center, hp, hq, lay, "gimbal" if self.lock_mode == "world" else self.lock_mode, self.follow_pitch)
            with self._layouts_lock:
                self._layouts[name] = new
            if self.lock_mode == "world":
                q = self._world_pose.get(name, (None, (1.0, 0.0, 0.0, 0.0)))[1]
                self._world_pose[name] = (tuple(np.asarray(center, dtype=float).tolist()), q)
        if released:
            logger.info("Camera panel(s) released: %s.", released)
            if self._on_layout_changed is not None:
                try:
                    self._on_layout_changed(self.layouts_snapshot())
                except Exception:  # noqa: BLE001
                    logger.exception("on_layout_changed failed")
        grabbed = self._grabber.grabbed_panels()
        hovered = {n for n in self._grabber.hover.values() if n is not None}
        for name in self._handles:
            state = "grab" if name in grabbed else ("hover" if name in hovered else "idle")
            self._set_handle_state(name, state)

    def _frame_hw(self, name: str) -> tuple[int, int]:
        item = self._source(name)
        if item is None:
            return 9, 16
        return int(item[0].shape[0]), int(item[0].shape[1])

    def status(self) -> str:
        st = self.last_stats
        fps = getattr(st, "fps", None) if st is not None else None
        stale = getattr(st, "stale_layers", None) if st is not None else None
        text = f"viz: {len(self._layers)}/{len(self._layouts)} panels, renders={self.render_count}" + (
            f", fps={fps:.0f}" if isinstance(fps, (int, float)) else ""
        ) + (f", stale={list(stale)}" if stale else "") + (" LOST" if self.lost else "")
        if self._grabber is not None:
            hov = [f"{h[0].upper()}:{n}" for h, n in self._grabber.hover.items() if n is not None]
            grb = [f"{h[0].upper()}:{n}" for h, n in self._grabber.grabbing.items() if n is not None]
            if hov:
                text += f", hover={hov}"
            if grb:
                text += f", grab={grb}"
        return text


_bus_import_failed = False


def bus_frame_source(name: str) -> tuple[Any, float, int] | None:
    """Latest frame of ``name`` from the robot plugin's in-process frame bus."""
    global _bus_import_failed
    if _bus_import_failed:
        return None
    try:
        from lerobot_robot_rby1.frame_bus import bus
    except Exception as e:  # noqa: BLE001  (robot plugin not importable, e.g. tests)
        _bus_import_failed = True
        logger.warning("Camera frame bus unavailable (%s); the headset panels stay empty.", e)
        return None
    return bus.latest(name)


__all__ = [
    "CameraPanels",
    "PanelLayout",
    "bus_frame_source",
    "face_orientation",
    "layout_from_center",
    "panel_basis",
    "rgb_to_rgba_cuda",
    "xr_panel_pose",
]
