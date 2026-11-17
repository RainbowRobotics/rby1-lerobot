"""The scripted operator as Isaac Capture pipeline outputs, for tests of the LeRobot device.

``rby1_xr_teleop.scripted.Operator`` describes the operator; here its tick is
turned into the dict a ``TeleopSession.step()`` returns, so the tests go
through the same parsing as the real session.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import isaac_fakes as fakes
from lerobot_teleoperator_rby1.isaac_teleop.xr_frame import (
    OUT_BODY,
    OUT_CONTROLLER_LEFT,
    OUT_CONTROLLER_RIGHT,
    OUT_HEAD,
)
from rby1_xr_teleop.scripted import HAND_OUT, HAND_REST, head_quat  # noqa: F401  (re-exported)
from rby1_xr_teleop.scripted import Operator as _Operator


class Operator(_Operator):
    def frame(self) -> dict[str, Any]:
        """Pipeline outputs for this tick (the dict ``TeleopSession.step()`` returns)."""
        quat = self.body_quat()
        positions, valid = self.body_joints()

        def controller(side: str) -> Any:
            if not (self.right_tracked if side == "right" else self.left_tracked):
                return fakes.ABSENT
            return fakes.controller(tuple(self.hand(side).tolist()), quat, **self.controller_inputs(side))

        head = (
            fakes.head(tuple(self.head_position().tolist()), self.head_orientation())
            if self.head_tracked
            else fakes.ABSENT
        )
        return {
            OUT_CONTROLLER_RIGHT: controller("right"),
            OUT_CONTROLLER_LEFT: controller("left"),
            OUT_HEAD: head,
            OUT_BODY: fakes.body(
                positions=positions.astype(np.float32),
                orientations=np.tile(quat.astype(np.float32), (24, 1)),
                valid=valid.astype(np.uint8),
            ),
        }


class ScriptedSession:
    """``TeleopSession`` stand-in that asks ``frame_fn`` for the outputs of every step."""

    def __init__(self, frame_fn) -> None:
        self._frame_fn = frame_fn
        self.last_step_info = None
        self.entered = False
        self.exited = False
        self.steps = 0

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc):
        self.exited = True

    def step(self, *, execution_events=None, external_inputs=None):
        self.steps += 1
        return self._frame_fn()
