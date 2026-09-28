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
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

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
    "driver-API backend instead: `pip install \"cuda-python==12.6.*\"` (matches the "
    "JetPack driver), or `pip install cupy-cuda12x`."
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


def xr_panel_pose(
    head_position: np.ndarray,
    head_orientation_wxyz: np.ndarray,
    layout: PanelLayout,
    lock_mode: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Panel pose (position, orientation wxyz) in OpenXR space (Y up, forward -Z).

    ``gimbal``: follow the head position and yaw only (no pitch / roll).
    ``head``: full head lock. (``world`` callers compute this once and keep it.)
    """
    from scipy.spatial.transform import Rotation

    w, x, y, z = [float(v) for v in head_orientation_wxyz]
    R = Rotation.from_quat([x, y, z, w]).as_matrix()  # scipy = xyzw
    p = np.asarray(head_position, dtype=float)
    if lock_mode == "head":
        local = np.array([layout.offset_x, layout.offset_y, -layout.distance])
        pos = p + R @ local
        return tuple(pos.tolist()), (w, x, y, z)
    fwd = R @ np.array([0.0, 0.0, -1.0])
    fwd[1] = 0.0
    n = float(np.linalg.norm(fwd))
    fwd = fwd / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])
    right = np.array([-fwd[2], 0.0, fwd[0]])
    up = np.array([0.0, 1.0, 0.0])
    pos = p + fwd * layout.distance + right * layout.offset_x + up * layout.offset_y
    yaw = math.atan2(-fwd[0], -fwd[2])  # R_y(yaw) maps -Z onto fwd
    q = (math.cos(yaw / 2.0), 0.0, math.sin(yaw / 2.0), 0.0)
    return tuple(pos.tolist()), q


class CameraPanels:
    """Render thread + layers for the camera panels. Inject fakes for tests."""

    def __init__(
        self,
        layouts: list[PanelLayout],
        frame_source: FrameSource,
        *,
        lock_mode: str = "gimbal",
        openxr_composition: bool = False,
        uploader: Uploader | None = None,
        viz_module: Any | None = None,
        poll_period_s: float = 0.002,
        daemon: bool = False,
    ) -> None:
        self.layouts = list(layouts)
        self._source = frame_source
        self.lock_mode = lock_mode
        self.openxr_composition = openxr_composition
        self._upload = uploader or rgb_to_rgba_cuda
        self._viz = viz_module
        self._poll_period = poll_period_s
        self._daemon = daemon

        self.session: Any = None
        self._layers: dict[str, Any] = {}
        self._last_seq: dict[str, int] = {}
        self._world_pose: dict[str, tuple] = {}
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
        logger.info("Televiz XR session created (%d camera panels, lock=%s).", len(self.layouts), self.lock_mode)
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
                info = self.session.render()
                self.render_count += 1
                if getattr(info, "reference_space_changed", False):
                    self._world_pose.clear()
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
        for layout in self.layouts:
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
        for layout in self.layouts:
            layer = self._layers.get(layout.name)
            if layer is None:
                continue
            if self.lock_mode == "world":
                if layout.name not in self._world_pose:
                    self._world_pose[layout.name] = xr_panel_pose(hp, hq, layout, "gimbal")
                pos, q = self._world_pose[layout.name]
            else:
                pos, q = xr_panel_pose(hp, hq, layout, self.lock_mode)
            h, w = self._frame_hw(layout.name)
            layer.set_placement(tv.QuadLayerPlacement(tv.Pose3D(position=pos, orientation=q), size_meters=(layout.width, layout.width * h / w)))

    def _frame_hw(self, name: str) -> tuple[int, int]:
        item = self._source(name)
        if item is None:
            return 9, 16
        return int(item[0].shape[0]), int(item[0].shape[1])

    def status(self) -> str:
        st = self.last_stats
        fps = getattr(st, "fps", None) if st is not None else None
        stale = getattr(st, "stale_layers", None) if st is not None else None
        return f"viz: {len(self._layers)}/{len(self.layouts)} panels, renders={self.render_count}" + (
            f", fps={fps:.0f}" if isinstance(fps, (int, float)) else ""
        ) + (f", stale={list(stale)}" if stale else "") + (" LOST" if self.lost else "")


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


__all__ = ["CameraPanels", "PanelLayout", "bus_frame_source", "rgb_to_rgba_cuda", "xr_panel_pose"]
