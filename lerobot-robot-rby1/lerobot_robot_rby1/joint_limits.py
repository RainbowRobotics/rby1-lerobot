"""Read the joint position limits of a robot over a short, read-only connection.

The limits come from the dynamics model the robot serves (its URDF), so they
match the connected model and version. :class:`~lerobot_robot_rby1.rby1.Rby1`
reads them on its own connection; this helper is for code that has no robot
handle, such as the leader-arm teleoperator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model_probe import resolve_model_version

Limits = tuple[np.ndarray, np.ndarray]  # (q_min, q_max) in radians


@dataclass(frozen=True)
class RobotJointLimits:
    """Position limits (radians) of the body groups, plus the robot they were read from."""

    model: str
    version: str
    torso: Limits
    right_arm: Limits
    left_arm: Limits


def read_joint_limits(address: str, connect_timeout_sec: float = 3.0) -> RobotJointLimits:
    """Connect to ``address``, read the joint position limits and disconnect.

    Nothing is powered, enabled or commanded. Raises ``RuntimeError`` when the
    robot cannot be reached or the limits cannot be read.
    """
    try:
        import rby1_sdk as rby
    except ImportError as exc:
        raise RuntimeError("rby1_sdk is not installed; cannot read the robot's joint limits.") from exc

    model, version = resolve_model_version("auto", "auto", address, connect_timeout_sec)
    timeout_ms = max(1, int(round(connect_timeout_sec * 1000.0)))
    try:
        robot = rby.create_robot(address, model)
    except Exception as exc:
        raise RuntimeError(f"Failed to create an RB-Y1 SDK handle for '{address}': {exc}") from exc
    try:
        try:
            connected = robot.connect(max_retries=1, timeout_ms=timeout_ms)
        except Exception as exc:
            raise RuntimeError(f"Failed to connect to RB-Y1 at '{address}': {exc}") from exc
        if not connected:
            raise RuntimeError(f"Could not connect to RB-Y1 at '{address}' within {connect_timeout_sec:.2f}s.")
        try:
            robot_model = robot.model()
            dyn = robot.get_dynamics()
            state = dyn.make_state([], robot_model.robot_joint_names)
            q_lower = np.asarray(dyn.get_limit_q_lower(state), dtype=np.float64)
            q_upper = np.asarray(dyn.get_limit_q_upper(state), dtype=np.float64)

            def group(name: str) -> Limits:
                idx = list(getattr(robot_model, f"{name}_idx"))
                return q_lower[idx], q_upper[idx]

            return RobotJointLimits(
                model=model,
                version=version,
                torso=group("torso"),
                right_arm=group("right_arm"),
                left_arm=group("left_arm"),
            )
        except Exception as exc:
            raise RuntimeError(f"Could not read the joint limits from RB-Y1 at '{address}': {exc}") from exc
    finally:
        try:
            robot.disconnect()
        except Exception:
            pass
