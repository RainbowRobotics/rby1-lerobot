"""Engage-relative clutch for the XR -> RB-Y1 teleop loop.

Adapted from the upstream LeRobot example
``examples/isaac_teleop_to_so101/isaac_teleop/clutch.py``. Pure numpy/scipy
(no ``isaacteleop``), so it is unit-testable without the XR runtime.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


class Clutch:
    """Engage-relative clutch for both position AND orientation.

    Latch an origin on engage, then track the base-frame delta from it, applied
    independently to position and orientation. State:

    - ``_last_commanded_pos`` / ``_last_commanded_rot``: last commanded pose; held
      while disengaged so the component freezes where it was left.
    - ``_home_pos`` / ``_home_rot``: latched on engage — the pose the delta applies to.
    - ``_origin_pos`` / ``_origin_rot``: latched on engage — the tracked (controller /
      chest) pose the delta is measured against.

    Each engaged frame :meth:`rebase` returns::

        pos = home_pos + (grip_pos - origin_pos)  # 1:1 tracked -> target translation
        rot = (R_ctrl @ R_origin ^ -1) @ R_home  # base-frame delta, left-composed

    On the engage edge the output is exactly the home pose (no teleport). The
    orientation delta is left-composed (base frame), so hand rotation about
    base Z maps to target rotation about base Z. A re-clutch latches a fresh
    home/origin.

    ``latch_orientation`` selects where the home ORIENTATION comes from on
    engage when a measured pose is supplied: ``"measured"`` (RB-Y1 default —
    its 7-DOF arms track orientation, so latching the measurement prevents a
    snap-back after the arm was moved while disengaged) or ``"commanded"``
    (the upstream SO-101 behaviour, which avoids re-injecting the persistent
    tracking offset of a 5-DOF arm on every re-clutch).
    """

    def __init__(self, home_base_T_ee: np.ndarray):  # noqa: N803
        home = np.asarray(home_base_T_ee, dtype=float)
        self._last_commanded_pos = home[:3, 3].copy()
        self._last_commanded_rot = Rotation.from_matrix(home[:3, :3])
        self._home_pos = self._last_commanded_pos.copy()
        self._home_rot = self._last_commanded_rot
        self._origin_pos = np.zeros(3, dtype=float)
        self._origin_rot = Rotation.from_quat(np.array([0.0, 0.0, 0.0, 1.0]))
        self._engaged = False

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def engaged(self) -> bool:
        return self._engaged

    @property
    def home(self) -> np.ndarray:
        """4x4 pose latched at the last engage (the delta reference)."""
        return _se3(self._home_pos, self._home_rot)

    @property
    def last_commanded(self) -> np.ndarray:
        """4x4 pose of the last :meth:`rebase` output (held while disengaged)."""
        return _se3(self._last_commanded_pos, self._last_commanded_rot)

    # ------------------------------------------------------------------
    # Engage / rebase / disengage
    # ------------------------------------------------------------------

    def engage(
        self,
        grip_pos: np.ndarray,
        grip_quat: np.ndarray,
        measured_base_T_ee: np.ndarray | None = None,  # noqa: N803
        *,
        latch_orientation: str = "measured",
    ) -> None:
        """Latch the engage home (where the component is now) and tracked origin.

        Pass ``measured_base_T_ee`` (FK of the measured joints) so the home is
        where the component physically is — if it moved while disengaged
        (gravity sag, external contact, a record reset), latching the stale
        last-commanded pose would make the first engaged frame command a
        full-speed jump back to it.
        """
        if measured_base_T_ee is not None:
            measured = np.asarray(measured_base_T_ee, dtype=float)
            self._home_pos = measured[:3, 3].copy()
            if latch_orientation == "measured":
                self._home_rot = Rotation.from_matrix(measured[:3, :3])
            else:
                self._home_rot = self._last_commanded_rot
        else:
            self._home_pos = self._last_commanded_pos.copy()
            self._home_rot = self._last_commanded_rot
        self._origin_pos = np.asarray(grip_pos, dtype=float).copy()
        self._origin_rot = Rotation.from_quat(np.asarray(grip_quat, dtype=float))
        # The engage frame commands exactly the home pose.
        self._last_commanded_pos = self._home_pos.copy()
        self._last_commanded_rot = self._home_rot
        self._engaged = True

    def rebase(self, grip_pos: np.ndarray, grip_quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the absolute base-frame target ``(pos [m], quat [xyzw])`` for this frame."""
        pos = self._home_pos + (np.asarray(grip_pos, dtype=float) - self._origin_pos)
        rot_ctrl = Rotation.from_quat(np.asarray(grip_quat, dtype=float))
        rot = (rot_ctrl * self._origin_rot.inv()) * self._home_rot
        self._last_commanded_pos = pos.copy()
        self._last_commanded_rot = rot
        return pos, rot.as_quat()

    def disengage(self) -> None:
        """Stop following; :attr:`last_commanded` is held until the next engage."""
        self._engaged = False

    def set_commanded(self, base_T_ee: np.ndarray) -> None:  # noqa: N803
        """Overwrite the commanded pose without touching the engaged flag.

        Used by the absolute (non-clutch) arm mode, which computes the target
        itself and only uses this object as the per-arm target holder.
        """
        T = np.asarray(base_T_ee, dtype=float)
        self._last_commanded_pos = T[:3, 3].copy()
        self._last_commanded_rot = Rotation.from_matrix(T[:3, :3])

    def hold_at(self, base_T_ee: np.ndarray) -> None:  # noqa: N803
        """Re-seed the held pose (and home) to ``base_T_ee`` while disengaged.

        Used to re-synchronise with the robot after it moved on its own (ready
        pose motion after connect, record reset), so the next action does not
        drag it back to a stale target.
        """
        T = np.asarray(base_T_ee, dtype=float)
        self._last_commanded_pos = T[:3, 3].copy()
        self._last_commanded_rot = Rotation.from_matrix(T[:3, :3])
        self._home_pos = self._last_commanded_pos.copy()
        self._home_rot = self._last_commanded_rot
        self._engaged = False


def _se3(pos: np.ndarray, rot: Rotation) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()
    T[:3, 3] = pos
    return T
