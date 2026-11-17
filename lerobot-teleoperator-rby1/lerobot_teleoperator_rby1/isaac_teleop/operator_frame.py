"""The operator frame: which way the operator faces, as a yaw about the robot's z axis.

Poses from the XR device arrive in a frame whose axes point like the robot's
base but whose heading is wherever the headset session started. The operator
frame turns them so that the direction the operator faces is robot +X. Its
single state is the yaw correction applied to every incoming pose:

* it is *referenced* on the first usable frame and on Right A (shoulder line,
  head gaze, or headset -> controllers when the headset hangs from the neck);
* during a body drive it *follows* the body's turn.

Pure NumPy, no robot or XR-runtime dependency.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from .retargeters import head_yaw_pitch, wrap_pi
from .xr_frame import BodyJointIndex, XRFrame, rotate_frame_about_z

logger = logging.getLogger(__name__)


def shoulder_line_yaw(frame: XRFrame) -> float | None:
    """Yaw (rad) the operator's body faces, from the body-tracking shoulder line; None when invalid."""
    if frame.body is None:
        return None
    ls, rs = int(BodyJointIndex.LEFT_SHOULDER), int(BodyJointIndex.RIGHT_SHOULDER)
    if not (bool(frame.body.valid[ls]) and bool(frame.body.valid[rs])):
        return None
    across = frame.body.positions[ls] - frame.body.positions[rs]  # right -> left
    fwd = np.cross(across, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(fwd[:2]) <= 1e-3:
        return None
    return math.atan2(fwd[1], fwd[0])


class OperatorFrame:
    """Yaw correction (rad about robot z) applied to every incoming XR pose."""

    def __init__(self, *, reference_source: str = "shoulders", wear_mode: str = "head") -> None:
        self.yaw = 0.0
        self.needs_reference = True
        self._reference_source = reference_source
        self._wear_mode = wear_mode
        self._warned_waiting_shoulders = False

    def apply(self, raw: XRFrame) -> XRFrame:
        """``raw`` with every pose turned into the operator frame."""
        return rotate_frame_about_z(raw, self.yaw)

    def reference(self, raw: XRFrame, *, on_button: bool = False) -> bool:
        """Make the operator's current facing direction robot +X.

        The facing direction is taken from the body-tracking shoulder line
        when both shoulders are valid (robust to where the operator looks),
        otherwise from the head gaze. Returns True when everything derived
        from the operator frame (smoothed shoulders, posture hints, the
        controller -> gripper offsets) must start over: the yaw changed, or
        the reference was asked for with the button.
        """
        yaw: float | None = None
        source = ""
        src = self._reference_source
        if src != "gaze":
            yaw = shoulder_line_yaw(raw)
            if yaw is not None:
                source = "shoulder line"
        if yaw is None and src == "shoulders" and self._wear_mode != "neck":
            if not on_button:
                # First reference: wait for the shoulders (the arms hold meanwhile).
                if not self._warned_waiting_shoulders:
                    logger.info("Waiting for body tracking (both shoulders) to reference the operator frame …")
                    self._warned_waiting_shoulders = True
                return False
            logger.warning("Right A without valid shoulders: operator frame referenced from the head gaze instead.")
        if yaw is None and self._wear_mode == "neck" and raw.head is not None:
            # Both controllers are needed: a single hand sits ~0.2 m off-centre
            # and would bias the facing direction by tens of degrees.
            if raw.right is not None and raw.left is not None:
                mid = 0.5 * (raw.right.position + raw.left.position)
                fwd = mid - raw.head.position
                if np.linalg.norm(fwd[:2]) > 0.05:
                    yaw = math.atan2(fwd[1], fwd[0])
                    source = "headset→controllers direction"
        if yaw is None:
            if raw.head is None:
                return False
            yaw, _ = head_yaw_pitch(raw.head.pose)
            source = "head gaze"
        old = self.yaw
        self.yaw = -yaw
        self.needs_reference = False
        logger.info(
            "Operator frame re-referenced from the %s: yaw %.1f deg in the XR anchor frame is now robot +X.",
            source,
            math.degrees(yaw),
        )
        return abs(wrap_pi(self.yaw - old)) > 1e-6 or on_button
