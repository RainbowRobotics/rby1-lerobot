"""One tick of XR input as plain NumPy dataclasses.

:class:`XRFrame` is what the teleoperation core consumes: the two controllers,
the headset and (optionally) the tracked body, as positions / quaternions in
a frame with the robot base's axes (x forward, y left, z up). How a frame is
obtained — an Isaac Capture pipeline (:mod:`rby1_xr_teleop.isaac_io`), a
simulator, a script — is up to the host.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


class BodyJointIndex(IntEnum):
    """XR_BD_body_tracking layout (24 joints)."""

    PELVIS = 0
    LEFT_HIP = 1
    RIGHT_HIP = 2
    SPINE1 = 3
    LEFT_KNEE = 4
    RIGHT_KNEE = 5
    SPINE2 = 6
    LEFT_ANKLE = 7
    RIGHT_ANKLE = 8
    SPINE3 = 9
    LEFT_FOOT = 10
    RIGHT_FOOT = 11
    NECK = 12
    LEFT_COLLAR = 13
    RIGHT_COLLAR = 14
    HEAD = 15
    LEFT_SHOULDER = 16
    RIGHT_SHOULDER = 17
    LEFT_ELBOW = 18
    RIGHT_ELBOW = 19
    LEFT_WRIST = 20
    RIGHT_WRIST = 21
    LEFT_HAND = 22
    RIGHT_HAND = 23


NUM_BODY_JOINTS = 24


def pose_to_se3(position: Any, quat_xyzw: Any) -> np.ndarray:
    """4x4 SE3 from a position and an ``[qx, qy, qz, qw]`` quaternion."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(np.asarray(quat_xyzw, dtype=float)).as_matrix()
    T[:3, 3] = np.asarray(position, dtype=float)
    return T


@dataclass(frozen=True)
class ControllerState:
    position: np.ndarray      # (3,) m, robot base frame
    orientation: np.ndarray   # (4,) xyzw
    squeeze: float            # [0, 1]
    trigger: float            # [0, 1]
    thumbstick: np.ndarray    # (2,) x (right +), y (forward +)
    primary: bool             # A / X
    secondary: bool           # B / Y
    aim_position: np.ndarray | None = None     # (3,) aim (pointer) pose, same frame
    aim_orientation: np.ndarray | None = None  # (4,) xyzw; None = not provided / invalid

    @property
    def pose(self) -> np.ndarray:
        return pose_to_se3(self.position, self.orientation)

    @property
    def aim_pose(self) -> np.ndarray:
        """Aim pose when valid, else the grip pose."""
        if self.aim_position is None or self.aim_orientation is None:
            return self.pose
        return pose_to_se3(self.aim_position, self.aim_orientation)


@dataclass(frozen=True)
class HeadState:
    position: np.ndarray
    orientation: np.ndarray
    is_tracked: bool

    @property
    def pose(self) -> np.ndarray:
        return pose_to_se3(self.position, self.orientation)


@dataclass(frozen=True)
class BodyState:
    positions: np.ndarray     # (24, 3)
    orientations: np.ndarray  # (24, 4) xyzw
    valid: np.ndarray         # (24,) bool

    def joint_pose(self, index: int) -> np.ndarray:
        return pose_to_se3(self.positions[index], self.orientations[index])


@dataclass(frozen=True)
class XRFrame:
    right: ControllerState | None = None
    left: ControllerState | None = None
    head: HeadState | None = None
    body: BodyState | None = None

    @property
    def any_controller(self) -> bool:
        return self.right is not None or self.left is not None


def rotate_frame_about_z(frame: XRFrame, yaw: float) -> XRFrame:
    """Rotate every pose of ``frame`` by ``yaw`` (rad) about the robot z axis.

    Used to re-reference the operator's facing direction: after the rotation,
    the direction the headset looked at when the reference was taken is robot
    +X. Positions are rotated about the anchor origin, which is irrelevant for
    the (delta-based) clutches.
    """
    if abs(yaw) < 1e-12:
        return frame
    rz = Rotation.from_euler("z", yaw)
    R = rz.as_matrix()

    def ctrl(c: ControllerState | None) -> ControllerState | None:
        if c is None:
            return None
        return ControllerState(
            R @ c.position,
            (rz * Rotation.from_quat(c.orientation)).as_quat(),
            c.squeeze,
            c.trigger,
            c.thumbstick,
            c.primary,
            c.secondary,
            None if c.aim_position is None else R @ c.aim_position,
            None if c.aim_orientation is None else (rz * Rotation.from_quat(c.aim_orientation)).as_quat(),
        )

    head = None
    if frame.head is not None:
        head = HeadState(
            R @ frame.head.position,
            (rz * Rotation.from_quat(frame.head.orientation)).as_quat(),
            frame.head.is_tracked,
        )
    body = None
    if frame.body is not None:
        quat = frame.body.orientations.copy()
        valid = frame.body.valid
        if valid.any():
            quat[valid] = (rz * Rotation.from_quat(quat[valid])).as_quat()
        body = BodyState(frame.body.positions @ R.T, quat, valid)
    return XRFrame(right=ctrl(frame.right), left=ctrl(frame.left), head=head, body=body)


class ButtonEdge:
    """Rising-edge detector for the A/B (X/Y) buttons of both controllers."""

    KEYS = ("right_a", "right_b", "left_a", "left_b")

    def __init__(self) -> None:
        self._prev = {k: False for k in self.KEYS}

    def update(self, frame: XRFrame) -> dict[str, bool]:
        cur = {
            "right_a": frame.right is not None and frame.right.primary,
            "right_b": frame.right is not None and frame.right.secondary,
            "left_a": frame.left is not None and frame.left.primary,
            "left_b": frame.left is not None and frame.left.secondary,
        }
        events = {k: (cur[k] and not self._prev[k]) for k in self.KEYS}
        self._prev = cur
        return events
