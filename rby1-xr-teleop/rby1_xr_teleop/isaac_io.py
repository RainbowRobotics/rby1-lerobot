"""Isaac Capture (formerly Isaac Teleop) pipeline outputs -> :class:`~rby1_xr_teleop.xr_frame.XRFrame`.

The device pipeline exposes the OpenXR controllers, head and (optionally)
body-tracking sources verbatim, each statically rebased into the robot base
frame. :func:`frame_from_outputs` turns one ``TeleopSession.step()`` result
into the plain dataclasses of :mod:`rby1_xr_teleop.xr_frame`.

Nothing here imports ``isaaccapture``: the outputs are only indexed with the
enums below, which mirror the package's tensor layouts (the same in
``isaacteleop`` 1.4 and ``isaaccapture`` 1.6). A host that builds its session
with the real package should check them against it once (the LeRobot plugin
does, in ``verify_index_layout``).
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import IntEnum
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .xr_frame import NUM_BODY_JOINTS, BodyState, ControllerState, HeadState, XRFrame

# ---------------------------------------------------------------------------
# Tensor layouts (mirrors of isaaccapture.retargeting_engine.tensor_types.indices)
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
# Parsing
# ---------------------------------------------------------------------------


def _is_none(group: Any) -> bool:
    return group is None or bool(getattr(group, "is_none", False))


_QUAT_MIN_NORM = 1e-6


def _pose_ok(pos: np.ndarray, quat: np.ndarray) -> bool:
    """Finite position and a usable (non-zero) quaternion.

    Untracked samples arrive as all-zero poses (e.g. the head before the
    headset is tracked on layouts without an is_valid flag); feeding a zero
    quaternion to scipy raises, so they are treated as absent instead.
    """
    return bool(
        np.all(np.isfinite(pos))
        and np.all(np.isfinite(quat))
        and float(np.linalg.norm(quat)) > _QUAT_MIN_NORM
    )


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
    if not _pose_ok(pos, quat):
        return None
    aim_pos = aim_quat = None
    try:
        if bool(group[ControllerInputIndex.AIM_IS_VALID]):
            ap = np.asarray(group[ControllerInputIndex.AIM_POSITION], dtype=float).reshape(3)
            aq = np.asarray(group[ControllerInputIndex.AIM_ORIENTATION], dtype=float).reshape(4)
            if _pose_ok(ap, aq):
                aim_pos, aim_quat = ap, aq
    except (IndexError, KeyError, TypeError, ValueError):
        pass  # layouts without an aim pose: callers fall back to the grip pose
    return ControllerState(pos, quat, squeeze, trigger, thumb, primary, secondary, aim_pos, aim_quat)


def _parse_head(group: Any) -> HeadState | None:
    if _is_none(group):
        return None
    try:
        pos = np.asarray(group[HeadInputIndex.POSITION], dtype=float).reshape(3)
        quat = np.asarray(group[HeadInputIndex.ORIENTATION], dtype=float).reshape(4)
    except (IndexError, KeyError, TypeError, ValueError):
        return None
    # is_valid / is_tracked are missing on old 3-field HeadPose layouts; there
    # a present sample counts as valid and tracked.
    try:
        if not bool(group[HeadInputIndex.IS_VALID]):
            return None
    except (IndexError, KeyError, TypeError, ValueError):
        pass
    try:
        tracked = bool(group[HeadInputIndex.IS_TRACKED])
    except (IndexError, KeyError, TypeError, ValueError):
        tracked = True
    # isaaccapture >= 1.6 keeps reporting the last head pose with
    # is_valid=True, is_tracked=False after the headset disconnects or goes to
    # sleep. That pose is stale: report the head as absent so the callers'
    # input-loss handling (hold / clutch release) applies.
    if not tracked:
        return None
    if not _pose_ok(pos, quat):
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
    # Invalid joints carry zero quaternions: never treat them as valid.
    finite = np.isfinite(pos).all(axis=1) & np.isfinite(quat).all(axis=1)
    valid = valid & finite & (np.linalg.norm(quat, axis=1) > _QUAT_MIN_NORM)
    if transform is not None and valid.any():
        # Same left-multiplied rebase the in-graph transform applies:
        # base_T_joint = base_T_anchor @ anchor_T_joint. Only valid joints are
        # rotated (zero quaternions would raise in scipy).
        R_ba = transform[:3, :3]
        pos = pos @ R_ba.T + transform[:3, 3]
        quat = quat.copy()
        quat[valid] = (Rotation.from_matrix(R_ba) * Rotation.from_quat(quat[valid])).as_quat()
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
