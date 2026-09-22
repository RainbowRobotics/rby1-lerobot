"""Read-only RB-Y1 state access for the XR device (state + forward kinematics).

The clutch latches its home pose from the *measured* end-effector pose on every
engage, so the teleoperator needs its own read-only ``rby1_sdk`` link. Power,
servos, the control manager and every command stream are owned by the follower
robot (``lerobot_robot_rby1.Rby1``); this class never sends commands.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Link list passed to make_state; the indices below follow its order.
FK_LINKS = ["base", "link_torso_5", "link_right_arm_6", "link_left_arm_6"]
_IDX_BASE, _IDX_TORSO_5, _IDX_RIGHT_ARM_6, _IDX_LEFT_ARM_6 = range(4)


@dataclass(frozen=True)
class RobotSnapshot:
    """Measured poses (4x4, base frame) and head joints (rad) at one instant."""

    torso: np.ndarray
    right_ee: np.ndarray
    left_ee: np.ndarray
    head_q: np.ndarray
    right_q: np.ndarray | None = None  # (7,) rad
    left_q: np.ndarray | None = None


class Rby1StateReader:
    """Read-only ``rby1_sdk`` link: joint state + FK of torso / both hands."""

    def __init__(self, address: str, model: str, version: str = "auto") -> None:
        self.address = address
        self.model_name = model
        self.version = version  # "auto" is resolved in connect() (robot probe)
        self._robot: Any = None
        self._model: Any = None
        self._dyn: Any = None
        self._state: Any = None

    @property
    def is_connected(self) -> bool:
        return self._robot is not None

    def connect(self) -> None:
        try:
            import rby1_sdk as rby
        except ImportError as e:  # pragma: no cover
            raise ImportError("rby1_sdk is required for the rby1_isaac teleoperator.") from e
        robot = rby.create_robot(self.address, self.model_name)
        if not robot.connect():
            raise ConnectionError(f"Failed to connect to RB-Y1 at {self.address} (read-only).")
        self._robot = robot
        self._model = robot.model()
        self._dyn = robot.get_dynamics()
        self._state = self._dyn.make_state(FK_LINKS, self._model.robot_joint_names)
        if self.version == "auto":
            self.version = self._probe_version()
        logger.info("Read-only RB-Y1 state link open at %s (version %s)", self.address, self.version)

    def _probe_version(self) -> str:
        try:
            from lerobot_robot_rby1.model_probe import resolve_model_version

            _, version = resolve_model_version(self.model_name, "auto", self.address)
            return version or "1.3"
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not probe the robot version (%s); assuming 1.3.", e)
            return "1.3"

    def read(self) -> RobotSnapshot:
        if self._robot is None:
            raise RuntimeError("Rby1StateReader is not connected.")
        state = self._robot.get_state()
        q = np.asarray(state.position, dtype=np.float64).copy()
        self._state.set_q(q)
        self._dyn.compute_forward_kinematics(self._state)
        torso = np.asarray(
            self._dyn.compute_transformation(self._state, _IDX_BASE, _IDX_TORSO_5), dtype=float
        )
        right = np.asarray(
            self._dyn.compute_transformation(self._state, _IDX_BASE, _IDX_RIGHT_ARM_6), dtype=float
        )
        left = np.asarray(
            self._dyn.compute_transformation(self._state, _IDX_BASE, _IDX_LEFT_ARM_6), dtype=float
        )
        head_q = np.asarray(q[self._model.head_idx], dtype=np.float64).copy()
        right_q = np.asarray(q[self._model.right_arm_idx], dtype=np.float64).copy()
        left_q = np.asarray(q[self._model.left_arm_idx], dtype=np.float64).copy()
        return RobotSnapshot(torso, right, left, head_q, right_q, left_q)

    def close(self) -> None:
        if self._robot is not None:
            try:
                self._robot.disconnect()
            finally:
                self._robot = None
                self._model = None
                self._dyn = None
                self._state = None
