"""Shared scaffolding of the ``Rby1XR`` tests: fake frames, the stubbed pipeline, a teleoperator factory."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

import isaac_fakes as fakes
from lerobot_teleoperator_rby1.isaac_teleop.config_isaac_teleop import Rby1XRConfig
from lerobot_teleoperator_rby1.isaac_teleop.teleop_rby1_xr import Rby1XR
from lerobot_teleoperator_rby1.isaac_teleop.xr_frame import (
    OUT_BODY,
    OUT_CONTROLLER_LEFT,
    OUT_CONTROLLER_RIGHT,
    OUT_HEAD,
    BodyJointIndex,
)
from rby1_xr_teleop.arm_retargeter import robot_reach

# Robot-frame head rotation looking straight ahead (anchor -Z -> robot +X).
R_HEAD_FWD = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)


def _head_quat(yaw_deg=0.0, pitch_deg=0.0):
    R = R_HEAD_FWD @ Rotation.from_euler("y", yaw_deg, degrees=True).as_matrix() @ Rotation.from_euler("x", pitch_deg, degrees=True).as_matrix()
    return Rotation.from_matrix(R).as_quat()


def _frame(right=None, left=None, head=None, body=None):
    return {
        OUT_CONTROLLER_RIGHT: right if right is not None else fakes.ABSENT,
        OUT_CONTROLLER_LEFT: left if left is not None else fakes.ABSENT,
        OUT_HEAD: head if head is not None else fakes.ABSENT,
        OUT_BODY: body if body is not None else fakes.ABSENT,
    }


def make_teleop(session, reader_holder, **cfg_overrides):
    # Most scenarios pre-date the shoulder-line reference and the engage-time
    # position offset; they opt back into the legacy behaviour (dedicated
    # tests cover the defaults).
    cfg_overrides.setdefault("reference_source", "auto")
    cfg_overrides.setdefault("ee_position_latch_on_a", False)
    # … and the first-Right-A start gate (covered by its own test).
    cfg_overrides.setdefault("start_on_right_a", False)
    cfg_overrides.setdefault("arm_length_source", "body")
    cfg_overrides.setdefault("head_base_require_grip", False)
    cfg = Rby1XRConfig(auto_launch_cloudxr=False, head_smoothing=1.0, **cfg_overrides)

    def reader_factory(address, model):
        reader_holder.append(fakes.FakeStateReader(address, model))
        return reader_holder[-1]

    return Rby1XR(
        cfg, session_factory=lambda pipeline: session, state_reader_factory=reader_factory
    )


def _body_with_arm(side, shoulder, elbow, wrist):
    pos = np.zeros((24, 3), np.float32)
    idx = {
        "right": (BodyJointIndex.RIGHT_SHOULDER, BodyJointIndex.RIGHT_ELBOW, BodyJointIndex.RIGHT_WRIST),
        "left": (BodyJointIndex.LEFT_SHOULDER, BodyJointIndex.LEFT_ELBOW, BodyJointIndex.LEFT_WRIST),
    }[side]
    for i, p in zip(idx, (shoulder, elbow, wrist)):
        pos[int(i)] = p
    return fakes.body(positions=pos)


# Absolute arm mapping fixtures: human shoulder at (0, -0.3, 1.5), reach 0.6 m
# (config), robot shoulder = torso + [0, -0.22, 0.08]; hand offsets scale by
# robot_reach / 0.6 onto the robot shoulder.
ABS_SHOULDER = np.array([0.0, -0.3, 1.5])
ABS_KW = dict(engage_ramp_s=0.05, ee_max_linear_vel=100.0, ee_max_angular_vel=1000.0, shoulder_smoothing=1.0,
              arm_length_source="config", human_arm_length_m=0.6)


def _abs_body(side="right"):
    S = ABS_SHOULDER * (np.array([1, -1, 1]) if side == "left" else 1)
    return _body_with_arm(side, S, S + [0, 0, -0.3], S + [0, 0, -0.6])


def _abs_hand(dx=0.0, dy=0.0, dz=0.0):
    return tuple((ABS_SHOULDER + [dx, dy, dz]).tolist())


def _abs_goal(reader, dx=0.0, dy=0.0, dz=0.0, side="right"):
    from lerobot_teleoperator_rby1.isaac_teleop.arm_retargeter import robot_shoulder_position

    k = robot_reach("1.3") / 0.6
    return robot_shoulder_position(reader.torso, side) + np.array([dx, dy, dz]) * k
