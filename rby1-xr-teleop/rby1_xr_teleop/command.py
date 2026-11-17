"""What the teleoperation core asks of the robot for one tick."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .retargeters import se3_to_ee_action

# Flat action keys (the convention of the RB-Y1 LeRobot plugins).
RESET_KEY = "reset.cmd"
POS_SUFFIX = ".pos"
NULL_SUFFIX = ".null"
HEAD_JOINTS = ("head_0", "head_1")
BASE_KEYS = ("x.vel", "y.vel", "theta.vel")
HINT_JOINTS = 4


@dataclass
class TeleopCommand:
    """Targets for one tick; a field that is None is not commanded.

    Poses are 4x4 targets in the robot base frame for ``link_torso_5``,
    ``link_right_arm_6`` and ``link_left_arm_6`` (Cartesian impedance on the
    RB-Y1). ``hints`` are arm posture hints: joint targets (rad) for
    ``arm_0..arm_3`` of that side, meant for the solver's nullspace.
    ``reset_seconds`` asks the robot to run its own return-to-ready motion
    over at least that many seconds instead of following the targets.
    """

    torso: np.ndarray | None = None
    right_ee: np.ndarray | None = None
    left_ee: np.ndarray | None = None
    right_gripper: float | None = None   # 1.0 = open
    left_gripper: float | None = None
    base: tuple[float, float, float] | None = None  # (vx, vy, wz), body frame
    head: np.ndarray | None = None       # [head_0, head_1] rad
    hints: dict[str, np.ndarray] = field(default_factory=dict)
    reset_seconds: float | None = None

    def as_action(self) -> dict[str, float]:
        """The command as a flat action dictionary."""
        action: dict[str, float] = {}
        if self.reset_seconds is not None:
            action[RESET_KEY] = float(self.reset_seconds)
        for prefix, pose in (("torso_ee", self.torso), ("right_ee", self.right_ee), ("left_ee", self.left_ee)):
            if pose is not None:
                action.update(se3_to_ee_action(pose, prefix))
        if self.right_gripper is not None:
            action[f"right_gripper_0{POS_SUFFIX}"] = self.right_gripper
        if self.left_gripper is not None:
            action[f"left_gripper_0{POS_SUFFIX}"] = self.left_gripper
        if self.base is not None:
            for key, value in zip(BASE_KEYS, self.base):
                action[key] = value
        if self.head is not None:
            for name, value in zip(HEAD_JOINTS, self.head):
                action[f"{name}{POS_SUFFIX}"] = float(value)
        for side, hint in self.hints.items():
            for i in range(HINT_JOINTS):
                action[f"{side}_arm_{i}{NULL_SUFFIX}"] = float(hint[i])
        return action
