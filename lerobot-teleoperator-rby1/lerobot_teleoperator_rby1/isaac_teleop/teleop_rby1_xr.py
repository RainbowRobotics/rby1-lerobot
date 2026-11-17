"""RB-Y1 XR device for NVIDIA Isaac Capture, exposed to LeRobot as ``rby1_isaac``.

Control flow (one ``TeleopSession.step()`` per :meth:`get_action`)::

    controllers / head / body  ──►  XRFrame (robot base frame)
        controller + IOBT shoulder ──► AbsoluteEeMapper ──► right_ee.* / left_ee.*
        IOBT shoulder / elbow    ──► ArmPostureRetargeter ──► <side>_arm_<i>.null
        body chest (or head)     ──► Clutch + clamp ──► torso_ee.*
        headset look direction   ──► HeadRetargeter ──► head_0.pos / head_1.pos
        triggers                 ──► *_gripper_0.pos   (1.0 = open)
        thumbsticks              ──► x.vel / y.vel / theta.vel

Following the LeRobot convention this teleoperator only *produces* actions;
the paired ``lerobot_robot_rby1.Rby1`` follower (``action_mode="ee"``,
``use_head=True`` …) executes them. The robot link held here is read-only
(state + forward kinematics so the clutch latches on the *measured* pose).

Button mapping (Meta Quest naming)
----------------------------------
    Squeeze (grip) > threshold   Dead-man switch: while held the hand position
                                 relative to the operator's IOBT shoulder is
                                 mapped onto the robot shoulder; release = hold
    Trigger                      Gripper (1 = closed)
    Right thumbstick             Base linear velocity; left thumbstick: yaw
    Right B                      Stop: freeze every target, zero the base
    Right A                      Release every clutch and return arms, torso and
                                 head to the start pose (measured right after the
                                 follower's ready-pose motion); re-reference the
                                 operator frame (current facing direction = robot
                                 +X); also resumes after a stop

Operator frame
--------------
OpenXR reports poses in a reference space whose forward is wherever the
headset looked when the session started. All poses are therefore rotated
about the robot z axis by ``-yaw(head)`` taken at the first tracked head
frame and again on every Right A, so "push the controller forward" always
means robot +X for the operator as they currently stand.
    Torso                        Follows the chest (body tracking) or the head
                                 while BOTH arms are clutched
"""

from __future__ import annotations

import logging
import socket
import subprocess
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..constants import (
    BASE_VEL_NAMES,
    HEAD_NAMES,
    LEFT_EE_NAMES,
    NULL_SUFFIX,
    POS_SUFFIX,
    RIGHT_EE_NAMES,
    TORSO_EE_NAMES,
)
from rby1_xr_teleop.core import XrTeleopCore
from rby1_xr_teleop.state import RobotInfo
from rby1_xr_teleop.xr_frame import XRFrame

from .base import IsaacTeleopTeleoperator
from .config_isaac_teleop import Rby1XRConfig
from .robot_state import Rby1StateReader
from .viz_grab import HandInput, hand_input, load_layouts, save_layouts
from .viz_panels import CameraPanels, PanelLayout, bus_frame_source
from .xr_frame import build_external_inputs, build_pipeline, frame_from_outputs

logger = logging.getLogger(__name__)

# The CloudXR service serves the web client itself (isaaccapture >= 1.6,
# host_client); the hosted page is the fallback when that is unreachable.
CLOUDXR_WEB_CLIENT_URL = "https://nvidia.github.io/IsaacCapture/client/"
CLOUDXR_WSS_PORT = 48322
HEADSET_DEVICE_ID = "openxr/headset"
_XR_CONNECT_REMINDER_S = 15.0
_SKIP_IFACE_PREFIXES = ("lo", "docker", "br-", "veth", "virbr", "l4tbr")


class Rby1XR(IsaacTeleopTeleoperator):
    """XR-controller / headset / body-tracking teleoperator for the RB-Y1.

    The class name must be ``Rby1XRConfig`` minus ``Config``: LeRobot's
    ``make_teleoperator_from_config`` resolves third-party device classes by
    that rule, importing them from the config's parent package.
    """

    config_class = Rby1XRConfig
    name = "rby1_isaac"

    def __init__(
        self,
        config: Rby1XRConfig,
        *,
        session_factory: Any | None = None,
        launcher_factory: Any | None = None,
        state_reader_factory: Callable[[str, str], Any] | None = None,
        viz_module: Any | None = None,
        viz_uploader: Any | None = None,
    ) -> None:
        super().__init__(config, session_factory=session_factory, launcher_factory=launcher_factory)
        self._viz_module_override = viz_module
        self._viz_uploader_override = viz_uploader
        self.config: Rby1XRConfig = config
        self._reader_factory = state_reader_factory or Rby1StateReader
        self._reader: Any = None
        self._connected = False
        # The teleoperation logic; created in connect() once the robot is known.
        self._core: XrTeleopCore | None = None
        self._tracking = False
        self._last_status_log = 0.0

        self._external_inputs: dict[str, Any] | None = None
        self._body_transform: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------

    @property
    def action_features(self) -> dict[str, type]:
        cfg = self.config
        names: list[str] = []
        if cfg.use_torso:
            names += TORSO_EE_NAMES
        if cfg.use_right_arm:
            names += RIGHT_EE_NAMES
        if cfg.use_left_arm:
            names += LEFT_EE_NAMES
        features: dict[str, type] = {n: float for n in names}
        if cfg.use_gripper:
            if cfg.use_right_arm:
                features[f"right_gripper_0{POS_SUFFIX}"] = float
            if cfg.use_left_arm:
                features[f"left_gripper_0{POS_SUFFIX}"] = float
        if cfg.use_mobile_base:
            for n in BASE_VEL_NAMES:
                features[n] = float
        if cfg.use_head:
            for n in HEAD_NAMES:
                features[f"{n}{POS_SUFFIX}"] = float
        if cfg.arm_posture_hint and cfg.record_posture_hint:
            for side, enabled in (("right", cfg.use_right_arm), ("left", cfg.use_left_arm)):
                if enabled:
                    for i in range(4):
                        features[f"{side}_arm_{i}{NULL_SUFFIX}"] = float
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_tracking(self) -> bool:
        """Whether the last frame carried at least one tracked controller."""
        return self._tracking

    @property
    def is_stopped(self) -> bool:
        return self._core is not None and self._core.stopped

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002
        if self._connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected.")
        cfg = self.config
        logger.info("rby1_isaac: opening read-only robot link at %s …", cfg.robot_address)
        self._reader = self._reader_factory(cfg.robot_address, cfg.robot_model)
        logger.info("rby1_isaac: wear_mode=%s", cfg.wear_mode)
        if cfg.wear_mode == "neck":
            logger.warning(
                "wear_mode=neck: the headset must NOT go to sleep while hanging from the neck — "
                "disable the Quest proximity sensor (MQDH → Device Actions → Proximity Sensor off, "
                "or cover the sensor). The robot head is held at the start pose and the headset "
                "pose drives the torso."
            )
        if cfg.viz_enabled:
            layouts = [
                PanelLayout(name, offset_x=ox, offset_y=cfg.viz_offset_y, distance=cfg.viz_distance_m, width=cfg.viz_width_m)
                for name, ox in zip(cfg.viz_cameras, cfg.viz_offsets_x)
            ]
            if cfg.viz_layout_file and not cfg.viz_layout_reset:
                layouts = load_layouts(cfg.viz_layout_file, layouts)
            self._viz = CameraPanels(
                layouts,
                bus_frame_source,
                lock_mode=cfg.viz_lock_mode,
                follow_pitch=cfg.viz_follow_pitch,
                face_head=cfg.viz_face_head,
                openxr_composition=cfg.viz_openxr_composition,
                viz_module=self._viz_module_override,
                uploader=self._viz_uploader_override,
                daemon=self._viz_module_override is not None,  # test doubles must not pin the process
                grab_enabled=cfg.viz_grab_enabled,
                grab_threshold=cfg.viz_grab_threshold,
                grab_push_rate_mps=cfg.viz_grab_push_rate_mps,
                frame_bridge=cfg.viz_frame_bridge,
                on_layout_changed=self._save_panel_layout if cfg.viz_layout_file else None,
            )
            if cfg.viz_grab_enabled:
                logger.info(
                    "Camera panels can be moved: point a free hand (squeeze released) at the bar above a "
                    "panel and hold the trigger; thumbstick pushes / pulls. Layout file: %s",
                    cfg.viz_layout_file or "(not persisted)",
                )
        try:
            self._reader.connect()
            self._core = self._make_core()
            if cfg.session_start == "connect":
                self._open_session()
            else:
                logger.info(
                    "rby1_isaac: session_start=first_action — the CloudXR/OpenXR session "
                    "opens on the first get_action() call."
                )
        except Exception:
            self._close_reader()
            raise
        self._connected = True
        if cfg.profile_out:
            from lerobot_robot_rby1.profile import update_profile_section

            path = update_profile_section(cfg.profile_out, "teleop", {"type": self.name, "config": cfg})
            logger.info("%s config recorded in %s.", self, path)
        logger.info("%s connected.", self)

    def disconnect(self) -> None:
        if not self._connected:
            return
        try:
            super().disconnect()
        finally:
            self._close_reader()
            self._external_inputs = None
            self._connected = False
            self._tracking = False
            logger.info("%s disconnected.", self)

    def _close_reader(self) -> None:
        if self._reader is not None:
            try:
                self._reader.close()
            finally:
                self._reader = None

    def _open_session(self) -> None:
        """Launch CloudXR + TeleopSession, wait for the headset, latch the start pose."""
        # Both VizSession.create() and TeleopSession entry block until the
        # headset connects: show the connection instructions before that.
        print_xr_connect_help()
        super().connect()
        try:
            self._external_inputs = build_external_inputs(self.config.base_T_anchor)
            self._wait_for_tracking()
            self._core.reset(self._reader.read())
        except Exception:
            # No half-state: a live session behind a raised connect would leak
            # the CloudXR runtime.
            super().disconnect()
            raise

    def _make_core(self) -> XrTeleopCore:
        """The teleoperation logic for the robot the state reader is connected to."""
        cfg = self.config
        reader = self._reader
        version = cfg.robot_version if cfg.robot_version != "auto" else getattr(reader, "version", "1.3")
        arm_reach = getattr(reader, "arm_reach", None)
        reach = None
        if arm_reach is not None and cfg.arm_length_source == "calibrate":
            # The calibrated operator reach is shoulder -> controller, so the
            # robot side is shoulder -> the commanded link (link_<side>_arm_6).
            reach = {
                side: float(arm_reach(side))
                for side, enabled in (("right", cfg.use_right_arm), ("left", cfg.use_left_arm))
                if enabled
            }
        info = RobotInfo(
            version=version,
            holonomic=str(getattr(reader, "model_name", "m")).lower() != "a",
            arm_reach=reach,
        )
        return XrTeleopCore(cfg, info, on_arm_calibrated=self._record_arm_calibration)

    def _build_pipeline(self) -> Any:
        pipeline, body_in_graph = build_pipeline(with_body=self.config.needs_body)
        self._body_transform = (
            None if body_in_graph else np.asarray(self.config.base_T_anchor, dtype=float)
        )
        return pipeline

    # ------------------------------------------------------------------
    # Frame acquisition
    # ------------------------------------------------------------------

    def _read_raw_frame(self) -> XRFrame:
        """Step the session; the frame is in the XR anchor frame (operator frame not applied yet)."""
        outputs = self._step(
            execution_events=self._running_events(), external_inputs=self._external_inputs
        )
        raw = frame_from_outputs(
            outputs,
            want_body=self.config.needs_body,
            body_transform=self._body_transform,
        )
        self._tracking = raw.any_controller
        return raw

    def _wait_for_tracking(self) -> None:
        """Block until a controller is tracked (user-paced; Ctrl-C aborts)."""
        logger.info("Waiting for the headset controllers to start streaming …")
        t0 = time.monotonic()
        last_reminder = t0
        timeout = self.config.tracking_wait_timeout_s
        while True:
            frame = self._read_raw_frame()
            if frame.any_controller:
                logger.info("Headset connected — controllers are streaming.")
                return
            now = time.monotonic()
            if now - last_reminder >= _XR_CONNECT_REMINDER_S:
                print_xr_connect_help()
                logger.info("… still waiting for the headset to connect (Ctrl-C to abort).")
                last_reminder = now
            if timeout > 0.0 and now - t0 > timeout:
                raise TimeoutError(
                    f"No XR controller tracked within {timeout:.0f}s "
                    f"(tracking_wait_timeout_s)."
                )
            time.sleep(0.01)

    # ------------------------------------------------------------------
    # get_action — main per-tick entry point
    # ------------------------------------------------------------------

    def get_action(self) -> dict[str, Any]:
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if self._session is None:  # session_start == "first_action"
            self._open_session()

        raw = self._read_raw_frame()
        # One robot read per tick: engage edges latch from it and disengaged
        # components are re-synchronised to it when the robot moved on its own.
        snap = self._reader.read()
        now = time.monotonic()
        # Camera-panel grab: feed the aim rays; a grabbing hand neither clutches
        # nor drives its gripper / thumbstick while it holds a panel.
        grabbing = self._feed_panel_hands(raw)
        ui_active = self._viz.ui_active() if self._viz is not None else None
        command = self._core.step(raw, snap, now, grabbing=grabbing, ui_active=ui_active)
        self._log_status(now)
        return command.as_action()

    @property
    def is_started(self) -> bool:
        """False while waiting for the first Right A (``start_on_right_a``)."""
        return self._core is not None and self._core.started

    def _record_arm_calibration(self, calibration: dict[str, dict[str, float]]) -> None:
        """The operator reach was measured: keep it with the teleoperator config in the profile."""
        cfg = self.config
        if not cfg.profile_out:
            return
        from lerobot_robot_rby1.profile import update_profile_section

        update_profile_section(
            cfg.profile_out, "teleop", {"type": self.name, "config": cfg, "arm_calibration": calibration}
        )

    def _log_status(self, now: float) -> None:
        """Periodic one-line status: what is tracked, what is clutched, why the torso holds."""
        period = self.config.status_log_period_s
        if period <= 0.0 or now - self._last_status_log < period:
            return
        self._last_status_log = now
        logger.info(
            "%s",
            self._core.status_line(
                headset=self._headset_status(self._core.last_frame), extra=self._viz.status() if self._viz is not None else ""
            ),
        )

    # ------------------------------------------------------------------
    # Camera-panel grab (Televiz)
    # ------------------------------------------------------------------

    def _feed_panel_hands(self, raw: XRFrame | None) -> dict[str, str | None]:
        """Send the controller aim rays (XR anchor frame) to the panels; return who is grabbing."""
        viz = self._viz
        none = {"right": None, "left": None}
        if viz is None or not self.config.viz_grab_enabled:
            return none
        if raw is None:
            viz.set_hands({}, None)
            return viz.grabbing()
        # raw = base_T_anchor · anchor pose (no yaw correction): undo the anchor transform.
        anchor_T_base = np.linalg.inv(np.asarray(self.config.base_T_anchor, dtype=float))
        hands: dict[str, HandInput | None] = {}
        for side, ctrl in (("right", raw.right), ("left", raw.left)):
            if ctrl is None:
                hands[side] = None
                continue
            hands[side] = hand_input(
                ctrl.aim_pose, anchor_T_base, ctrl.trigger,
                free=not self._core.arm_engaged(side), thumb_y=ctrl.thumbstick[1],
            )
        head = None if raw.head is None else anchor_T_base @ raw.head.pose
        viz.set_hands(hands, head)
        return viz.grabbing()

    def _save_panel_layout(self, layouts: list[PanelLayout]) -> None:
        if save_layouts(self.config.viz_layout_file, layouts):
            logger.info("Camera panel layout saved to %s.", self.config.viz_layout_file)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def _headset_status(self, frame: XRFrame | None) -> str:
        """Headset link state for the status line.

        ``TeleopSession.get_device_status`` reports ``unknown`` when the
        session runs on external OpenXR handles (camera panels) and does not
        exist on test doubles; head tracking is the fallback then.
        """
        tracked = "ok" if frame is not None and frame.head is not None else "-"
        getter = getattr(self._session, "get_device_status", None)
        if getter is None:
            return tracked
        try:
            status = getter(HEADSET_DEVICE_ID)
        except Exception:  # noqa: BLE001 - status is informational only
            return tracked
        state = getattr(getattr(status, "status", None), "value", None)
        if state is None or state == "unknown":
            return tracked
        return tracked if state == "connected" else f"{tracked} ({state})"


# ---------------------------------------------------------------------------
# Headset connection help
# ---------------------------------------------------------------------------


def _primary_ipv4() -> str | None:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
        finally:
            s.close()
    except OSError:
        return None


def candidate_ipv4s() -> list[tuple[str, str]]:
    """``[(interface, ipv4), ...]`` the headset might reach this host at."""
    primary = _primary_ipv4()
    found: list[tuple[str, str]] = []
    try:
        out = subprocess.run(
            ["ip", "-4", "-o", "addr", "show"], capture_output=True, text=True, timeout=2.0
        ).stdout
        for line in out.splitlines():
            parts = line.split()
            if len(parts) < 4:
                continue
            iface, ip = parts[1], parts[3].split("/")[0]
            if iface.startswith(_SKIP_IFACE_PREFIXES) or ip.startswith(("127.", "169.254.")):
                continue
            found.append((iface, ip))
    except Exception:
        pass
    if not found and primary:
        found.append(("default", primary))
    found.sort(key=lambda t: t[1] != primary)
    return found


def hosted_client_url(host: str, port: int = CLOUDXR_WSS_PORT) -> str:
    """Web client the CloudXR service hosts, pre-filled with the server address."""
    return f"https://{host}:{port}/client/?serverIP={host}&port={port}"


def print_xr_connect_help() -> None:
    ips = candidate_ipv4s()
    lines = [
        "=" * 76,
        "Connect your XR headset to this host over NVIDIA CloudXR:",
        "  1. In the headset browser, open the client served by this host",
        "     (accept the self-signed certificate warning):",
    ]
    if ips:
        lines += [f"        {hosted_client_url(ip)}   ({iface})" for iface, ip in ips]
        if len(ips) > 1:
            lines.append("     (use the address on the same network as your headset)")
    else:
        lines.append(f"        {hosted_client_url('<host-ip>')}   (check `hostname -I` / `ip addr`)")
    lines += [
        "  2. Press Connect. This program waits until the headset is connected.",
        f"  Fallback client: {CLOUDXR_WEB_CLIENT_URL} — enter the host IP there and first accept",
        f"  the certificate at https://<host-ip>:{CLOUDXR_WSS_PORT}/ .",
        "=" * 76,
    ]
    print("\n".join(lines), flush=True)


# Backward-compatible alias (pre-rename).
Rby1XRTeleop = Rby1XR
