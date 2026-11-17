"""Scripted XR operator for offline scenarios and the simulator end-to-end check.

Builds the per-tick outputs of the device pipeline (the dict a
``TeleopSession.step()`` returns) for an operator described by a few numbers:
where they stand, where the body and the head face, where the hands are
relative to the shoulders, and which inputs are pressed.

Positions are in the pipeline's output frame: robot-base axes (x forward,
y left, z up, metres), before the operator-frame yaw reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

import isaac_fakes as fakes
from lerobot_teleoperator_rby1.isaac_teleop.xr_frame import (
    OUT_BODY,
    OUT_CONTROLLER_LEFT,
    OUT_CONTROLLER_RIGHT,
    OUT_HEAD,
    BodyJointIndex,
)

# Robot-frame head rotation looking straight ahead (anchor -Z -> robot +X).
R_HEAD_FWD = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)

SHOULDER_Z = 1.40
SHOULDER_HALF_WIDTH = 0.20
HEAD_Z = 1.60
NECK_Z = 1.50
SPINE3_Z = 1.25
PELVIS_Z = 0.95
# Upper arm / forearm length used to place the elbow (m) and the distance of
# the controller's grip centre beyond the wrist joint.
SEGMENT = 0.31
GRIP_BEYOND_WRIST = 0.07

# Hand positions relative to the own shoulder, body frame.
HAND_REST = (0.25, 0.0, -0.35)   # forearm raised in front of the body
HAND_OUT = (0.60, 0.0, 0.0)      # arm straight out (reach calibration pose)


def head_quat(yaw_deg: float = 0.0, pitch_deg: float = 0.0) -> np.ndarray:
    """xyzw quaternion of a headset looking ``yaw_deg`` to the left and ``pitch_deg`` up."""
    R = (
        R_HEAD_FWD
        @ Rotation.from_euler("y", yaw_deg, degrees=True).as_matrix()
        @ Rotation.from_euler("x", pitch_deg, degrees=True).as_matrix()
    )
    return Rotation.from_matrix(R).as_quat()


@dataclass
class Operator:
    """The operator's pose and inputs for one tick; mutate the fields between ticks."""

    xy: tuple[float, float] = (0.0, 0.0)   # where the operator stands (m)
    body_deg: float = 0.0                  # where the body (shoulder line) faces
    neck_deg: float = 0.0                  # head turn relative to the body (left +)
    neck_pitch_deg: float = 0.0            # head tilt (up +)
    lean: tuple[float, float] = (0.0, 0.0)  # chest (SPINE3 / NECK / shoulders / head) offset, body frame (m)
    right_hand: tuple[float, float, float] = HAND_REST
    left_hand: tuple[float, float, float] = HAND_REST
    right_squeeze: float = 0.0
    left_squeeze: float = 0.0
    right_trigger: float = 0.0
    left_trigger: float = 0.0
    right_stick: tuple[float, float] = (0.0, 0.0)
    left_stick: tuple[float, float] = (0.0, 0.0)
    right_a: bool = False
    right_b: bool = False
    left_x: bool = False
    left_y: bool = False
    body_tracked: bool = True
    head_tracked: bool = True
    right_tracked: bool = True
    left_tracked: bool = True
    extra_invalid_joints: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    def _Rz(self) -> np.ndarray:  # noqa: N802
        return Rotation.from_euler("z", self.body_deg, degrees=True).as_matrix()

    def _origin(self) -> np.ndarray:
        return np.array([self.xy[0], self.xy[1], 0.0])

    def _chest(self) -> np.ndarray:
        return self._origin() + self._Rz() @ np.array([self.lean[0], self.lean[1], 0.0])

    def shoulder(self, side: str) -> np.ndarray:
        y = -SHOULDER_HALF_WIDTH if side == "right" else SHOULDER_HALF_WIDTH
        return self._chest() + self._Rz() @ np.array([0.0, y, SHOULDER_Z])

    def hand(self, side: str) -> np.ndarray:
        rel = np.asarray(self.right_hand if side == "right" else self.left_hand, dtype=float)
        return self.shoulder(side) + self._Rz() @ rel

    def head_position(self) -> np.ndarray:
        return self._chest() + np.array([0.0, 0.0, HEAD_Z])

    def _arm_joints(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        """Elbow and wrist joints of a two-segment arm whose grip centre is at the hand."""
        S, H = self.shoulder(side), self.hand(side)
        d = H - S
        n = float(np.linalg.norm(d))
        u = d / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
        wrist = H - GRIP_BEYOND_WRIST * u
        half = 0.5 * float(np.linalg.norm(wrist - S))
        drop = float(np.sqrt(max(SEGMENT**2 - half**2, 0.0)))
        elbow = 0.5 * (S + wrist) + np.array([0.0, 0.0, -drop])
        return elbow, wrist

    # ------------------------------------------------------------------
    def frame(self) -> dict[str, Any]:
        """Pipeline outputs for this tick (the dict ``TeleopSession.step()`` returns)."""
        Rz = self._Rz()
        body_quat = Rotation.from_matrix(Rz).as_quat()
        J = BodyJointIndex
        pos = np.zeros((24, 3), np.float32)
        valid = np.zeros(24, np.uint8)
        chest = self._chest()
        joints = {
            J.PELVIS: self._origin() + [0.0, 0.0, PELVIS_Z],
            J.SPINE3: chest + [0.0, 0.0, SPINE3_Z],
            J.NECK: chest + [0.0, 0.0, NECK_Z],
            J.HEAD: self.head_position(),
            J.RIGHT_SHOULDER: self.shoulder("right"),
            J.LEFT_SHOULDER: self.shoulder("left"),
        }
        for side, (e, w) in (("right", (J.RIGHT_ELBOW, J.RIGHT_WRIST)), ("left", (J.LEFT_ELBOW, J.LEFT_WRIST))):
            joints[e], joints[w] = self._arm_joints(side)
        if self.body_tracked:
            for joint, p in joints.items():
                pos[int(joint)] = p
                valid[int(joint)] = 1
            for name in self.extra_invalid_joints:
                valid[int(J[name])] = 0
        orientations = np.tile(body_quat.astype(np.float32), (24, 1))

        def controller(side: str) -> Any:
            right = side == "right"
            if not (self.right_tracked if right else self.left_tracked):
                return fakes.ABSENT
            return fakes.controller(
                tuple(self.hand(side).tolist()),
                body_quat,
                squeeze=self.right_squeeze if right else self.left_squeeze,
                trigger=self.right_trigger if right else self.left_trigger,
                thumb=self.right_stick if right else self.left_stick,
                primary=self.right_a if right else self.left_x,
                secondary=self.right_b if right else self.left_y,
            )

        head = (
            fakes.head(
                tuple(self.head_position().tolist()),
                head_quat(self.body_deg + self.neck_deg, self.neck_pitch_deg),
            )
            if self.head_tracked
            else fakes.ABSENT
        )
        return {
            OUT_CONTROLLER_RIGHT: controller("right"),
            OUT_CONTROLLER_LEFT: controller("left"),
            OUT_HEAD: head,
            OUT_BODY: fakes.body(positions=pos, orientations=orientations, valid=valid),
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
