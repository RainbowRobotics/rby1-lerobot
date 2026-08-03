"""LeRobot lifecycle adapter for the interpolating MQ3 receiver."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np

from .DoubleBuffer import DoubleBuffer
from .MQ3_Config import MQ3Config
from .MQ3_utils import MQ3, VR_DTYPE


def rotate_pose_locally(
    pose: np.ndarray,
    angle_deg: float,
) -> np.ndarray:
    """Rotate a torso-relative pose about the torso frame's Z axis.

    Left multiplication rotates both the controller position about the torso
    origin and the controller orientation. Positive angles follow the
    right-hand rule about +Z.
    """

    angle_rad = np.deg2rad(angle_deg)
    cos_angle = float(np.cos(angle_rad))
    sin_angle = float(np.sin(angle_rad))
    rotation = np.array(
        [
            [cos_angle, -sin_angle, 0.0, 0.0],
            [sin_angle, cos_angle, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return np.asarray(pose, dtype=np.float64) @ rotation


def transform_controller_pose(
    pose: np.ndarray,
    *,
    z_rotation_deg: float,
    xyz_offset_mm: tuple[float, float, float],
) -> np.ndarray:
    """Apply torso-Z rotation followed by a torso-frame XYZ offset."""

    transformed = rotate_pose_locally(pose, z_rotation_deg)
    transformed[:3, 3] += np.asarray(xyz_offset_mm, dtype=np.float64)
    return transformed


@dataclass
class VRButtons:
    primary: bool = False
    secondary: bool = False
    trigger: float = 0.0
    grip: float = 0.0


@dataclass
class VRControllerState:
    # MQ3 publishes a torso-relative, scaled robot target in this field.
    pose_rb: np.ndarray = field(default_factory=lambda: np.eye(4))
    buttons: VRButtons = field(default_factory=VRButtons)
    tracked: bool = False


@dataclass
class VRHeadState:
    pose_rb: np.ndarray = field(default_factory=lambda: np.eye(4))
    tracked: bool = False


@dataclass
class VRState:
    right: VRControllerState = field(default_factory=VRControllerState)
    left: VRControllerState = field(default_factory=VRControllerState)
    head: VRHeadState = field(default_factory=VRHeadState)
    user_scale: float = 1.0
    packet_monotonic_time: float = 0.0
    packet_count: int = 0


@dataclass
class VRButtonEvents:
    right_primary: bool = False
    right_secondary: bool = False
    left_primary: bool = False
    left_secondary: bool = False


class VRReceiver:
    """Wrap :class:`MQ3` without duplicating its UDP/interpolation logic."""

    def __init__(
        self,
        *,
        local_ip: str,
        local_port: int,
        meta_quest_ip: str | None,
        meta_quest_port: int = 6000,
        send_handshake: bool = True,
        tracking_timeout_s: float = 0.25,
        receive_buffer_bytes: int = 65535,
        grip_threshold: float = 0.5,
        initial_user_scale: float = 1000.0 / 700.0,
        user_scale_reference_mm: float = 1300.0,
        controller_z_rotation_deg: float = 0.0,
        controller_x_offset_mm: float = 0.0,
        controller_y_offset_mm: float = 0.0,
        controller_z_offset_mm: float = 0.0,
    ) -> None:
        if send_handshake and not meta_quest_ip:
            raise ValueError("meta_quest_ip is required by MQ3")
        self.tracking_timeout_s = float(tracking_timeout_s)
        self.controller_z_rotation_deg = float(
            controller_z_rotation_deg
        )
        self.controller_xyz_offset_mm = (
            float(controller_x_offset_mm),
            float(controller_y_offset_mm),
            float(controller_z_offset_mm),
        )
        config = MQ3Config(
            recv_buffer_bytes=receive_buffer_bytes,
            stale_warning_s=tracking_timeout_s,
            send_handshake=send_handshake,
            grip_threshold=grip_threshold,
            initial_user_scale=initial_user_scale,
            user_scale_reference_mm=user_scale_reference_mm,
        )
        self._mq3 = MQ3(
            local_ip, local_port, meta_quest_ip or "", meta_quest_port, config
        )
        # Attach a dedicated reader mapping by name.  Do not read through
        # ``self._mq3.shm``: that object owns the producer mapping and is also
        # inherited by the MQ3 child process.  A separate attachment keeps
        # reader lifetime/ownership independent of the process start method.
        self._reader = DoubleBuffer(
            VR_DTYPE,
            buffer_count=config.shm_buffer_count,
            create=False,
            shm_name=self._mq3.shm.name,
        )
        self._last_sequence = 0
        self._last_buffer_index = int(
            self._reader.memory["header"]["current_index"]
        )
        self._last_update_time = 0.0
        self._previous_buttons = (False, False, False, False)
        self._events = VRButtonEvents()

    @property
    def is_running(self) -> bool:
        return self._mq3.process.is_alive()

    @property
    def has_received_packet(self) -> bool:
        return self._last_sequence > 0

    @property
    def packet_count(self) -> int:
        return self._last_sequence

    def start(self) -> None:
        if self.is_running:
            raise RuntimeError("VRReceiver is already running")
        self._mq3.start()

    def stop(self) -> None:
        process = self._mq3.process
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
        self._reader.close()
        self._mq3.shm.close()

    def _read_raw(self) -> np.ndarray:
        raw = self._reader.read()
        buffer_index = int(
            self._reader.memory["header"]["current_index"]
        )
        if buffer_index != self._last_buffer_index:
            self._last_buffer_index = buffer_index
            self._last_sequence += 1
            self._last_update_time = time.monotonic()
        return raw

    def get_state(self, *, require_fresh: bool = False) -> VRState:
        raw = self._read_raw()
        right = VRControllerState(
            pose_rb=transform_controller_pose(
                raw["right_controller_current_pose"],
                z_rotation_deg=self.controller_z_rotation_deg,
                xyz_offset_mm=self.controller_xyz_offset_mm,
            ),
            buttons=VRButtons(
                bool(raw["event_right_a_pressed"]),
                bool(raw["event_right_b_pressed"]),
                float(np.clip(raw["event_right_trigger_value"], 0.0, 1.0)),
                float(np.clip(raw["event_right_grip_value"], 0.0, 1.0)),
            ),
            tracked=bool(raw["is_right_following"]),
        )
        left = VRControllerState(
            pose_rb=transform_controller_pose(
                raw["left_controller_current_pose"],
                z_rotation_deg=self.controller_z_rotation_deg,
                xyz_offset_mm=self.controller_xyz_offset_mm,
            ),
            buttons=VRButtons(
                bool(raw["event_left_a_pressed"]),
                bool(raw["event_left_b_pressed"]),
                float(np.clip(raw["event_left_trigger_value"], 0.0, 1.0)),
                float(np.clip(raw["event_left_grip_value"], 0.0, 1.0)),
            ),
            tracked=bool(raw["is_left_following"]),
        )
        current = (
            right.buttons.primary, right.buttons.secondary,
            left.buttons.primary, left.buttons.secondary,
        )
        rising = tuple(now and not old for now, old in zip(current, self._previous_buttons))
        self._previous_buttons = current
        self._events.right_primary |= rising[0]
        self._events.right_secondary |= rising[1]
        self._events.left_primary |= rising[2]
        self._events.left_secondary |= rising[3]
        state = VRState(
            right=right,
            left=left,
            head=VRHeadState(raw["head_controller_current_pose"].copy(), self._last_sequence > 0),
            user_scale=float(raw["user_scale"]),
            packet_monotonic_time=self._last_update_time,
            packet_count=self._last_sequence,
        )
        if require_fresh and self.is_stale(state=state):
            raise TimeoutError("Meta Quest tracking data is stale")
        return state

    def consume_button_events(self) -> VRButtonEvents:
        # Refresh first so button edges are derived even when callers consume
        # events before requesting the state.
        self.get_state()
        events = copy.copy(self._events)
        self._events = VRButtonEvents()
        return events

    def is_stale(self, *, state: VRState | None = None) -> bool:
        packet_time = state.packet_monotonic_time if state else self._last_update_time
        return packet_time <= 0.0 or time.monotonic() - packet_time > self.tracking_timeout_s

    def wait_for_first_packet(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self._read_raw()
            if self.has_received_packet:
                return True
            time.sleep(0.01)
        return False
