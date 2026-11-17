"""A scripted operator and a robot that follows — for tests, benchmarks and simulators.

:class:`Operator` describes an operator by a few numbers (where they stand,
where the body and the head face, where the hands are relative to the
shoulders, which inputs are pressed) and turns that into an
:class:`~rby1_xr_teleop.xr_frame.XRFrame`. :class:`FollowingRobot` keeps a
:class:`~rby1_xr_teleop.state.RobotState` that follows the commands of the
teleoperation core the way the robot would.

Positions are in the frame the core expects as input: robot-base axes
(x forward, y left, z up, metres), before the operator-frame yaw reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .command import TeleopCommand
from .retargeters import interpolate_pose
from .so3 import quat_from_matrix, rot_x, rot_y, rot_z
from .state import RobotState
from .xr_frame import BodyJointIndex, BodyState, ControllerState, HeadState, XRFrame

# Robot-frame head rotation looking straight ahead (XR view -Z -> robot +X).
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
    return quat_from_matrix(R_HEAD_FWD @ rot_y(np.radians(yaw_deg)) @ rot_x(np.radians(pitch_deg)))


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
    def body_rotation(self) -> np.ndarray:
        return rot_z(np.radians(self.body_deg))

    def body_quat(self) -> np.ndarray:
        return quat_from_matrix(self.body_rotation())

    def _origin(self) -> np.ndarray:
        return np.array([self.xy[0], self.xy[1], 0.0])

    def _chest(self) -> np.ndarray:
        return self._origin() + self.body_rotation() @ np.array([self.lean[0], self.lean[1], 0.0])

    def shoulder(self, side: str) -> np.ndarray:
        y = -SHOULDER_HALF_WIDTH if side == "right" else SHOULDER_HALF_WIDTH
        return self._chest() + self.body_rotation() @ np.array([0.0, y, SHOULDER_Z])

    def hand(self, side: str) -> np.ndarray:
        rel = np.asarray(self.right_hand if side == "right" else self.left_hand, dtype=float)
        return self.shoulder(side) + self.body_rotation() @ rel

    def head_position(self) -> np.ndarray:
        return self._chest() + np.array([0.0, 0.0, HEAD_Z])

    def head_orientation(self) -> np.ndarray:
        return head_quat(self.body_deg + self.neck_deg, self.neck_pitch_deg)

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

    def body_joints(self) -> tuple[np.ndarray, np.ndarray]:
        """``(positions (24, 3), valid (24,) bool)`` of the tracked body (all invalid when not tracked)."""
        J = BodyJointIndex
        positions = np.zeros((24, 3))
        valid = np.zeros(24, dtype=bool)
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
                positions[int(joint)] = p
                valid[int(joint)] = True
            for name in self.extra_invalid_joints:
                valid[int(J[name])] = False
        return positions, valid

    def controller_inputs(self, side: str) -> dict:
        """Analog and button inputs of one controller."""
        right = side == "right"
        return {
            "squeeze": self.right_squeeze if right else self.left_squeeze,
            "trigger": self.right_trigger if right else self.left_trigger,
            "thumb": self.right_stick if right else self.left_stick,
            "primary": self.right_a if right else self.left_x,
            "secondary": self.right_b if right else self.left_y,
        }

    # ------------------------------------------------------------------
    def xr_frame(self) -> XRFrame:
        """This tick as the :class:`XRFrame` the teleoperation core consumes."""
        quat = self.body_quat()

        def controller(side: str) -> ControllerState | None:
            if not (self.right_tracked if side == "right" else self.left_tracked):
                return None
            inputs = self.controller_inputs(side)
            position = self.hand(side)
            return ControllerState(
                position, quat.copy(), float(inputs["squeeze"]), float(inputs["trigger"]),
                np.asarray(inputs["thumb"], dtype=float), bool(inputs["primary"]), bool(inputs["secondary"]),
                position.copy(), quat.copy(),
            )

        head = HeadState(self.head_position(), self.head_orientation(), True) if self.head_tracked else None
        positions, valid = self.body_joints()
        body = BodyState(positions, np.tile(quat, (24, 1)), valid)
        return XRFrame(right=controller("right"), left=controller("left"), head=head, body=body)


class FollowingRobot:
    """A robot state that follows the teleoperation core's commands.

    ``apply(command)`` after every ``step()``: pose and head targets become
    the measured state, the base velocity is integrated into the odometry,
    and ``reset_seconds`` puts arms, torso and head back where they were
    when this object was created (the robot's own return-to-ready motion).
    ``follow`` in (0, 1] is the fraction of the remaining distance covered
    per tick (1 = exact tracking).
    """

    def __init__(self, state: RobotState, *, dt: float, follow: float = 1.0) -> None:
        self.state = state
        self.start = state
        self.dt = dt
        self.follow = follow
        self.resets = 0

    def apply(self, command: TeleopCommand) -> float:
        """Apply one command; returns the seconds the robot would block for (``reset_seconds``)."""
        s = self.state
        reset_s = float(command.reset_seconds or 0.0)
        changes: dict = {}
        if reset_s > 0.0:
            changes.update(torso=self.start.torso, right_ee=self.start.right_ee, left_ee=self.start.left_ee,
                           head_q=self.start.head_q)
            self.resets += 1
        else:
            for name in ("torso", "right_ee", "left_ee"):
                target = getattr(command, name)
                if target is not None:
                    changes[name] = interpolate_pose(getattr(s, name), target, self.follow)
            if command.head is not None:
                changes["head_q"] = s.head_q + self.follow * (np.asarray(command.head, dtype=float) - s.head_q)
        if s.odometry is not None and command.base is not None:
            vx, vy, wz = (float(v) * self.dt for v in command.base)
            c, si = np.cos(wz), np.sin(wz)
            changes["odometry"] = s.odometry @ np.array([[c, -si, vx], [si, c, vy], [0.0, 0.0, 1.0]])
        self.state = replace(s, **changes)
        return reset_s
