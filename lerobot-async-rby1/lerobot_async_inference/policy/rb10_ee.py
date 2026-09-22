"""RB EE wire contract. FK/IK stay on the client; rotation6D stays on the server."""

from typing import Any

import numpy as np
import zmq

from .zmq_wire import MsgSerializer

JOINT_KEYS = tuple(f"joint_{i}" for i in range(6))
GRIPPER_KEY = "gripper_0"
EE_PROTOCOL = {
    "protocol": "rb10-ee-v1",
    "state_dim": 7,
    "action_dim": 7,
    "position_unit": "m",
    "angle_unit": "rad",
    "euler_convention": "RzRyRx",
    "base_frame": "link0",
    "target_frame": "tcp",
    "gripper_open": 1.0,
}


def validate_descriptor(descriptor: Any) -> None:
    if not isinstance(descriptor, dict):
        raise ValueError("Server did not return an EE protocol descriptor")
    for key, expected in EE_PROTOCOL.items():
        if descriptor.get(key) != expected:
            raise ValueError(
                f"EE server contract mismatch: {key}={descriptor.get(key)!r}, expected {expected!r}"
            )


def _vector(value, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite ({size},) vector")
    return result


def _image(value, name: str) -> np.ndarray:
    result = np.asarray(value)
    if (
        result.ndim != 3
        or result.shape[-1] != 3
        or min(result.shape[:2]) <= 0
        or result.dtype != np.uint8
    ):
        raise ValueError(
            f"{name} must be uint8 RGB HWC; got {result.shape}, {result.dtype}"
        )
    # The receiver owns this snapshot even if the camera reuses its buffer.
    return np.array(result, copy=True, order="C")


def parse_ee_actions(response: Any) -> np.ndarray:
    if not isinstance(response, dict) or "actions" not in response:
        raise ValueError("EE response must contain actions")
    actions = np.asarray(response["actions"], dtype=np.float64)
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim != 2 or actions.shape[1] != 7 or not 0 < actions.shape[0] <= 40:
        raise ValueError(
            f"Expected EE actions (T,7), 1 <= T <= 40; got {actions.shape}"
        )
    if not np.all(np.isfinite(actions)):
        raise ValueError("EE actions contain non-finite values")
    # Model predictions can overshoot the normalized gripper range slightly.
    actions = np.array(actions, copy=True, order="C")
    actions[:, 6] = np.clip(actions[:, 6], 0.0, 1.0)
    return actions


class EEPolicyAdapter:
    def __init__(
        self,
        model: str,
        wrist_camera_key: str = "wrist",
        front_camera_key: str = "front",
    ):
        from lerobot_robot_rb.ee_kinematics import RBEEKinematics

        if (
            not wrist_camera_key
            or not front_camera_key
            or wrist_camera_key == front_camera_key
        ):
            raise ValueError("Wrist/front camera keys must be distinct and nonempty")
        self.kinematics = RBEEKinematics(model=model)
        self.wrist_camera_key = wrist_camera_key
        self.front_camera_key = front_camera_key

    def validate_robot(self, robot) -> None:
        expected = {*JOINT_KEYS, GRIPPER_KEY}
        if set(robot.action_features) != expected:
            raise ValueError(
                f"EE client requires six radian joint actions plus gripper_0, got {robot.action_features}"
            )
        required = expected | {self.wrist_camera_key}
        missing = required - set(robot.observation_features)
        if missing:
            raise ValueError(f"EE client missing robot observations: {sorted(missing)}")

    def build_observation(self, raw: dict, task: str) -> dict:
        q = _vector([raw[key] for key in JOINT_KEYS], 6, "joint state")
        gripper = float(raw[GRIPPER_KEY])
        if not np.isfinite(gripper) or not 0.0 <= gripper <= 1.0:
            raise ValueError("Observed gripper must be finite in [0,1], with 1=open")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("Task must be a nonempty string")
        pose = _vector(self.kinematics.forward(q), 6, "FK pose")
        result = {
            "observation/state": np.concatenate([pose, [gripper]]).astype(np.float32),
            "observation/base_image": _image(
                raw[self.wrist_camera_key], self.wrist_camera_key
            ),
            "prompt": task,
        }
        if self.front_camera_key in raw:
            result["observation/front_image"] = _image(
                raw[self.front_camera_key], self.front_camera_key
            )
        return result

    def joint_action(
        self, action, current_q, *, max_joint_delta: float
    ) -> dict[str, float]:
        target = _vector(action, 7, "EE action")
        seed = _vector(current_q, 6, "IK seed")
        if not np.isfinite(max_joint_delta) or max_joint_delta <= 0:
            raise ValueError("max_joint_delta must be finite and positive")
        q = _vector(
            self.kinematics.inverse(target[:6], seed, max_joint_delta=max_joint_delta),
            6,
            "IK result",
        )
        if np.any(np.abs(q - seed) > max_joint_delta + 1e-9):
            raise ValueError("IK result exceeds the per-tick joint displacement limit")
        return {
            **dict(zip(JOINT_KEYS, q.tolist(), strict=True)),
            GRIPPER_KEY: float(np.clip(target[6], 0, 1)),
        }


class ReconnectingEEClient:
    """Single-thread-owned REQ socket; failed requests are never replayed.

    The caller retries with a fresh observation. A timeout discards the old
    socket so a late response cannot be mistaken for the next request.
    """

    def __init__(self, server_address: str, timeout_ms: int = 1000):
        if not isinstance(timeout_ms, int) or timeout_ms <= 0:
            raise ValueError("timeout_ms must be a positive integer")
        self.address = (
            server_address
            if server_address.startswith("tcp://")
            else f"tcp://{server_address}"
        )
        if not self.address.removeprefix("tcp://").rpartition(":")[2].isdigit():
            raise ValueError("Expected server_address host:port")
        self.timeout_ms = timeout_ms
        self.ctx = zmq.Context()
        self.sock = None
        self._new_socket()

    def _new_socket(self):
        if self.sock is not None:
            self.sock.close(linger=0)
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.IMMEDIATE, 1)
        self.sock.connect(self.address)

    def _call(self, endpoint: str, data: dict | None = None):
        if self.ctx is None:
            raise RuntimeError("EE client is closed")
        request = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        try:
            self.sock.send(MsgSerializer.to_bytes(request))
            raw = self.sock.recv()
        except zmq.Again as exc:
            self._new_socket()
            raise TimeoutError(
                f"EE server timeout: {endpoint}; stale request discarded"
            ) from exc
        except zmq.ZMQError:
            self._new_socket()
            raise
        result = MsgSerializer.from_bytes(raw)
        if isinstance(result, dict) and "error" in result:
            raise ValueError(f"EE server rejected request: {result['error']}")
        return result

    def describe(self) -> dict:
        result = self._call("describe")
        validate_descriptor(result)
        return result

    def get_action(self, observation: dict) -> dict:
        return self._call("get_action", {"observation": observation})

    def close(self):
        if self.sock is not None:
            self.sock.close(linger=0)
            self.sock = None
        if self.ctx is not None:
            self.ctx.term()
            self.ctx = None
