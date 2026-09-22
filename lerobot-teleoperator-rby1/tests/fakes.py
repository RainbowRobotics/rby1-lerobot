"""Fake isaacteleop objects for offline tests."""

from __future__ import annotations

from typing import Any

import numpy as np

from lerobot_teleoperator_rby1.isaac_teleop.xr_frame import (
    NUM_BODY_JOINTS,
    ControllerInputIndex,
    FullBodyInputIndex,
    HeadInputIndex,
)

IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0])


class FakeGroup:
    """Minimal stand-in for an ``OptionalTensorGroup``."""

    def __init__(self, values: dict[int, Any] | None = None, *, is_none: bool = False):
        self._values = values or {}
        self.is_none = is_none

    def __getitem__(self, index: int) -> Any:
        if self.is_none:
            raise ValueError("absent group")
        return self._values[int(index)]


def controller(
    pos=(0.0, 0.0, 0.0),
    quat=IDENTITY_QUAT,
    *,
    squeeze=0.0,
    trigger=0.0,
    thumb=(0.0, 0.0),
    primary=False,
    secondary=False,
    valid=True,
) -> FakeGroup:
    return FakeGroup(
        {
            ControllerInputIndex.GRIP_POSITION: np.asarray(pos, dtype=np.float32),
            ControllerInputIndex.GRIP_ORIENTATION: np.asarray(quat, dtype=np.float32),
            ControllerInputIndex.GRIP_IS_VALID: valid,
            ControllerInputIndex.AIM_POSITION: np.zeros(3, np.float32),
            ControllerInputIndex.AIM_ORIENTATION: IDENTITY_QUAT.astype(np.float32),
            ControllerInputIndex.AIM_IS_VALID: valid,
            ControllerInputIndex.PRIMARY_CLICK: 1.0 if primary else 0.0,
            ControllerInputIndex.SECONDARY_CLICK: 1.0 if secondary else 0.0,
            ControllerInputIndex.THUMBSTICK_X: float(thumb[0]),
            ControllerInputIndex.THUMBSTICK_Y: float(thumb[1]),
            ControllerInputIndex.THUMBSTICK_CLICK: 0.0,
            ControllerInputIndex.MENU_CLICK: 0.0,
            ControllerInputIndex.SQUEEZE_VALUE: float(squeeze),
            ControllerInputIndex.TRIGGER_VALUE: float(trigger),
        }
    )


def head(pos=(0.0, 0.0, 0.0), quat=IDENTITY_QUAT, *, valid=True, tracked=True) -> FakeGroup:
    return FakeGroup(
        {
            HeadInputIndex.POSITION: np.asarray(pos, dtype=np.float32),
            HeadInputIndex.ORIENTATION: np.asarray(quat, dtype=np.float32),
            HeadInputIndex.IS_VALID: valid,
            HeadInputIndex.IS_TRACKED: tracked,
        }
    )


def body(positions=None, orientations=None, valid=None) -> FakeGroup:
    if positions is None:
        positions = np.zeros((NUM_BODY_JOINTS, 3), np.float32)
    if orientations is None:
        orientations = np.tile(IDENTITY_QUAT.astype(np.float32), (NUM_BODY_JOINTS, 1))
    if valid is None:
        valid = np.ones(NUM_BODY_JOINTS, np.uint8)
    return FakeGroup(
        {
            FullBodyInputIndex.JOINT_POSITIONS: positions,
            FullBodyInputIndex.JOINT_ORIENTATIONS: orientations,
            FullBodyInputIndex.JOINT_VALID: valid,
        }
    )


ABSENT = FakeGroup(is_none=True)


class FakeStepInfo:
    def __init__(self, worker_exception=None, frame_deadline_miss=False):
        self.worker_exception = worker_exception
        self.frame_deadline_miss = frame_deadline_miss
        self.returned_age_frames = 0


class FakeSession:
    """Scripted ``TeleopSession``: each ``step()`` pops the next outputs dict."""

    def __init__(self):
        self.frames: list[dict[str, Any]] = []  # queue of frames not yet returned
        self._current: dict[str, Any] | None = None  # held while the queue is empty
        self.last_step_info = None
        self.entered = False
        self.exited = False
        self.steps = 0

    def push(self, outputs: dict[str, Any]) -> None:
        self.frames.append(outputs)

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc):
        self.exited = True

    def step(self, *, execution_events=None, external_inputs=None):
        self.steps += 1
        if self.frames:
            self._current = self.frames.pop(0)
        if self._current is None:
            raise RuntimeError("FakeSession: no frame pushed")
        return self._current  # hold the last frame while the queue is empty


def se3(pos=(0.0, 0.0, 0.0), rot=None) -> np.ndarray:
    T = np.eye(4)
    if rot is not None:
        T[:3, :3] = rot
    T[:3, 3] = pos
    return T


class FakeSnapshot:
    def __init__(self, torso, right_ee, left_ee, head_q, right_q=None, left_q=None):
        self.torso = torso
        self.right_ee = right_ee
        self.left_ee = left_ee
        self.head_q = head_q
        self.right_q = np.zeros(7) if right_q is None else right_q
        self.left_q = np.zeros(7) if left_q is None else left_q


class FakeStateReader:
    def __init__(self, address: str, model: str, version: str = "auto"):
        self.address = address
        self.model = model
        self.version = "1.3" if version == "auto" else version
        self.connected = False
        self.reads = 0
        self.torso = se3((0.0, 0.0, 1.0))
        self.right_ee = se3((0.4, -0.2, 0.9))
        self.left_ee = se3((0.4, 0.2, 0.9))
        self.head_q = np.array([0.0, 0.85])

    def connect(self):
        self.connected = True

    def read(self):
        self.reads += 1
        return FakeSnapshot(
            self.torso.copy(), self.right_ee.copy(), self.left_ee.copy(), self.head_q.copy()
        )

    def close(self):
        self.connected = False
