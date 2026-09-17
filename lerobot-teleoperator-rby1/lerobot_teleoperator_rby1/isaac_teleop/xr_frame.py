"""Raw Isaac Teleop outputs -> a plain :class:`XRFrame`.

The device pipeline (see :func:`build_pipeline`) exposes the OpenXR
controllers, head and (optionally) body-tracking sources verbatim, each
statically rebased into the robot base frame. :func:`frame_from_outputs`
turns one ``TeleopSession.step()`` result into numpy-only dataclasses so the
rest of the device (clutch, retargeters) never touches ``isaacteleop`` types
and is unit-testable without it.

The index enums below mirror the ``isaacteleop`` tensor layouts (verified
against 1.4.142); :func:`verify_index_layout` asserts they still match the
installed package before a session is built.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tensor layouts (mirrors of isaacteleop.retargeting_engine.tensor_types.indices)
# ---------------------------------------------------------------------------


class ControllerInputIndex(IntEnum):
    GRIP_POSITION = 0
    GRIP_ORIENTATION = 1
    GRIP_IS_VALID = 2
    AIM_POSITION = 3
    AIM_ORIENTATION = 4
    AIM_IS_VALID = 5
    PRIMARY_CLICK = 6
    SECONDARY_CLICK = 7
    THUMBSTICK_X = 8
    THUMBSTICK_Y = 9
    THUMBSTICK_CLICK = 10
    MENU_CLICK = 11
    SQUEEZE_VALUE = 12
    TRIGGER_VALUE = 13


class HeadInputIndex(IntEnum):
    POSITION = 0
    ORIENTATION = 1
    IS_VALID = 2
    IS_TRACKED = 3


class FullBodyInputIndex(IntEnum):
    JOINT_POSITIONS = 0
    JOINT_ORIENTATIONS = 1
    JOINT_VALID = 2


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

# Output keys of the device pipeline (OutputCombiner) — the only contract
# between build_pipeline() and frame_from_outputs().
OUT_CONTROLLER_LEFT = "controller_left"
OUT_CONTROLLER_RIGHT = "controller_right"
OUT_HEAD = "head"
OUT_BODY = "full_body"

# Leaf-node name of the static base_T_anchor rebase input fed via
# ``TeleopSession.step(external_inputs=...)`` each frame.
BASE_T_ANCHOR_INPUT = "base_T_anchor"


# ---------------------------------------------------------------------------
# Frame dataclasses
# ---------------------------------------------------------------------------


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

    @property
    def pose(self) -> np.ndarray:
        return pose_to_se3(self.position, self.orientation)


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


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _is_none(group: Any) -> bool:
    return group is None or bool(getattr(group, "is_none", False))


def _parse_controller(group: Any) -> ControllerState | None:
    """Read one controller group; None when absent, invalid or partially populated.

    All fields are read into locals before committing any of them (as upstream
    does): a failure on a partial frame must not mix live values with defaults
    (a live squeeze with a defaulted trigger would keep the clutch engaged
    while opening the gripper).
    """
    if _is_none(group):
        return None
    try:
        if not bool(group[ControllerInputIndex.GRIP_IS_VALID]):
            return None
        pos = np.asarray(group[ControllerInputIndex.GRIP_POSITION], dtype=float).reshape(3)
        quat = np.asarray(group[ControllerInputIndex.GRIP_ORIENTATION], dtype=float).reshape(4)
        squeeze = float(group[ControllerInputIndex.SQUEEZE_VALUE])
        trigger = float(group[ControllerInputIndex.TRIGGER_VALUE])
        thumb = np.array(
            [
                float(group[ControllerInputIndex.THUMBSTICK_X]),
                float(group[ControllerInputIndex.THUMBSTICK_Y]),
            ]
        )
        primary = float(group[ControllerInputIndex.PRIMARY_CLICK]) > 0.5
        secondary = float(group[ControllerInputIndex.SECONDARY_CLICK]) > 0.5
    except (IndexError, KeyError, TypeError, ValueError):
        return None
    if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(quat))):
        return None
    return ControllerState(pos, quat, squeeze, trigger, thumb, primary, secondary)


def _parse_head(group: Any) -> HeadState | None:
    if _is_none(group):
        return None
    try:
        pos = np.asarray(group[HeadInputIndex.POSITION], dtype=float).reshape(3)
        quat = np.asarray(group[HeadInputIndex.ORIENTATION], dtype=float).reshape(4)
    except (IndexError, KeyError, TypeError, ValueError):
        return None
    # is_valid / is_tracked exist on newer isaacteleop only; older layouts
    # (3-field HeadPose) have neither, so treat a present sample as valid.
    try:
        if not bool(group[HeadInputIndex.IS_VALID]):
            return None
    except (IndexError, KeyError, TypeError, ValueError):
        pass
    try:
        tracked = bool(group[HeadInputIndex.IS_TRACKED])
    except (IndexError, KeyError, TypeError, ValueError):
        tracked = True
    if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(quat))):
        return None
    return HeadState(pos, quat, tracked)


def _parse_body(group: Any, transform: np.ndarray | None) -> BodyState | None:
    if _is_none(group):
        return None
    try:
        pos = np.asarray(group[FullBodyInputIndex.JOINT_POSITIONS], dtype=float).reshape(
            NUM_BODY_JOINTS, 3
        )
        quat = np.asarray(group[FullBodyInputIndex.JOINT_ORIENTATIONS], dtype=float).reshape(
            NUM_BODY_JOINTS, 4
        )
        valid = np.asarray(group[FullBodyInputIndex.JOINT_VALID]).reshape(NUM_BODY_JOINTS) != 0
    except (IndexError, KeyError, TypeError, ValueError):
        return None
    if transform is not None:
        # Same left-multiplied rebase the in-graph transform applies:
        # base_T_joint = base_T_anchor @ anchor_T_joint.
        R_ba = transform[:3, :3]
        pos = pos @ R_ba.T + transform[:3, 3]
        quat = (Rotation.from_matrix(R_ba) * Rotation.from_quat(quat)).as_quat()
    return BodyState(pos, quat, valid)


def frame_from_outputs(
    outputs: Mapping[str, Any],
    *,
    want_body: bool,
    body_transform: np.ndarray | None = None,
) -> XRFrame:
    """Convert one ``TeleopSession.step()`` result into an :class:`XRFrame`.

    ``body_transform`` is applied in Python only when the body source could
    not be rebased in-graph (see :func:`build_pipeline`).
    """
    return XRFrame(
        right=_parse_controller(outputs.get(OUT_CONTROLLER_RIGHT)),
        left=_parse_controller(outputs.get(OUT_CONTROLLER_LEFT)),
        head=_parse_head(outputs.get(OUT_HEAD)),
        body=_parse_body(outputs.get(OUT_BODY), body_transform) if want_body else None,
    )


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


# ---------------------------------------------------------------------------
# Pipeline construction (isaacteleop imported lazily)
# ---------------------------------------------------------------------------


# Candidate names of each layout enum across isaacteleop releases.
_REMOTE_ENUM_NAMES: dict[type[IntEnum], tuple[str, ...]] = {
    ControllerInputIndex: ("ControllerInputIndex",),
    HeadInputIndex: ("HeadInputIndex", "HeadPoseIndex"),
    FullBodyInputIndex: ("FullBodyInputIndex", "BodyInputIndex"),
    BodyJointIndex: ("BodyJointIndex",),
}


def verify_index_layout() -> dict[str, str]:
    """Check the local index enums against the installed ``isaacteleop``.

    Members present on both sides must agree (a mismatch raises, since it
    would silently corrupt every parsed pose). Enums or members missing from
    the installed package only log a warning — e.g. 1.4.x has no
    ``HeadInputIndex``/``IS_TRACKED`` — and the parsers tolerate them.
    Returns ``{local_enum_name: "verified" | "missing:<names>" | "absent"}``.
    """
    from isaacteleop.retargeting_engine.tensor_types import indices as idx

    report: dict[str, str] = {}
    for local, names in _REMOTE_ENUM_NAMES.items():
        remote = next((getattr(idx, n) for n in names if hasattr(idx, n)), None)
        if remote is None:
            logger.warning(
                "isaacteleop has none of %s; %s layout could not be verified.",
                names,
                local.__name__,
            )
            report[local.__name__] = "absent"
            continue
        missing: list[str] = []
        for member in local:
            if not hasattr(remote, member.name):
                missing.append(member.name)
                continue
            remote_value = int(getattr(remote, member.name))
            if remote_value != int(member):
                raise RuntimeError(
                    f"isaacteleop layout mismatch: {local.__name__}.{member.name} is "
                    f"{remote_value} in the installed package, {int(member)} here."
                )
        if missing:
            logger.warning(
                "isaacteleop %s lacks %s (older layout); those fields are treated as absent.",
                remote.__name__,
                missing,
            )
            report[local.__name__] = "missing:" + ",".join(missing)
        else:
            report[local.__name__] = "verified"
    return report


def build_pipeline(*, with_body: bool) -> tuple[Any, bool]:
    """Build the device pipeline.

    Returns ``(pipeline, body_rebased_in_graph)``. Controllers and head are
    rebased in-graph by the ``base_T_anchor`` external input; the body source
    is rebased in-graph when the installed ``isaacteleop`` offers
    ``FullBodySource.transformed`` and in Python otherwise.
    """
    from isaacteleop.retargeting_engine.deviceio_source_nodes import (
        ControllersSource,
        HeadSource,
    )
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    verify_index_layout()

    xform = ValueInput(BASE_T_ANCHOR_INPUT, TransformMatrix())
    xform_out = xform.output("value")

    controllers = ControllersSource(name="controllers").transformed(xform_out)
    head = HeadSource(name="head").transformed(xform_out)
    outputs = {
        OUT_CONTROLLER_LEFT: controllers.output(OUT_CONTROLLER_LEFT),
        OUT_CONTROLLER_RIGHT: controllers.output(OUT_CONTROLLER_RIGHT),
        OUT_HEAD: head.output(OUT_HEAD),
    }

    body_in_graph = False
    if with_body:
        from isaacteleop.retargeting_engine.deviceio_source_nodes import FullBodySource

        body_source = FullBodySource(name="full_body")
        if hasattr(body_source, "transformed"):
            outputs[OUT_BODY] = body_source.transformed(xform_out).output(OUT_BODY)
            body_in_graph = True
        else:
            outputs[OUT_BODY] = body_source.output(OUT_BODY)

    return OutputCombiner(outputs), body_in_graph


def build_external_inputs(base_T_anchor: Any) -> dict[str, Any]:  # noqa: N803
    """Materialise the constant ``base_T_anchor`` external input (once, in connect)."""
    from isaacteleop.retargeting_engine.interface import TensorGroup
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    tg = TensorGroup(TransformMatrix())
    tg[0] = np.asarray(base_T_anchor, dtype=np.float32)
    return {BASE_T_ANCHOR_INPUT: {"value": tg}}
