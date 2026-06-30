import io
from typing import Any

import numpy as np
import torch

from ..helpers import TimedAction


ARM_DOF = 7
GRIPPER_DOF = 1
ACTION_DOF = 16

RIGHT_ARM_KEYS = [f"right_arm_{i}" for i in range(ARM_DOF)]
LEFT_ARM_KEYS = [f"left_arm_{i}" for i in range(ARM_DOF)]
GRIPPER_KEYS = ["right_gripper_0", "left_gripper_0"]

PI05_ACTION_KEYS = [*RIGHT_ARM_KEYS, *LEFT_ARM_KEYS, *GRIPPER_KEYS]
PI05_STATE_KEYS = PI05_ACTION_KEYS


def _import_zmq_dependencies():
    try:
        import msgpack
        import zmq
    except ImportError as exc:
        raise ImportError(
            "The 'pi05_zmq' backend requires both 'msgpack' and 'pyzmq' to be installed."
        ) from exc

    return msgpack, zmq


class MsgSerializer:
    @staticmethod
    def to_bytes(data: Any) -> bytes:
        msgpack, _ = _import_zmq_dependencies()
        return msgpack.packb(data, default=MsgSerializer._encode)

    @staticmethod
    def from_bytes(data: bytes):
        msgpack, _ = _import_zmq_dependencies()
        return msgpack.unpackb(
            data,
            object_hook=MsgSerializer._decode,
            strict_map_key=False,
        )

    @staticmethod
    def _encode(obj):
        if isinstance(obj, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
        raise TypeError(f"Non-serializable type: {type(obj)}")

    @staticmethod
    def _decode(obj):
        if not isinstance(obj, dict):
            return obj
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj


class Pi05ZMQClient:
    def __init__(self, server_address: str, timeout_ms: int):
        _, zmq = _import_zmq_dependencies()
        self._zmq = zmq
        self.address = normalize_zmq_server_address(server_address)
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(self.address)

    def _call(self, endpoint: str, data: dict[str, Any] | None = None):
        request: dict[str, Any] = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data

        try:
            self.sock.send(MsgSerializer.to_bytes(request))
            raw = self.sock.recv()
        except self._zmq.Again as exc:
            raise TimeoutError(f"Server response timeout (endpoint={endpoint})") from exc
        except self._zmq.ZMQError as exc:
            raise RuntimeError(f"ZMQ error: {exc}") from exc

        response = MsgSerializer.from_bytes(raw)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def ping(self) -> bool:
        try:
            return bool(self._call("ping"))
        except Exception:
            return False

    def get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = self._call("get_action", {"observation": observation, "options": options})

        if isinstance(response, (list, tuple)) and len(response) >= 1:
            return dict(tuple(response)[0])
        if isinstance(response, dict):
            return response

        raise RuntimeError(f"Unexpected pi05 response type: {type(response)}")

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._call("reset", {"options": options})

    def close(self) -> None:
        self.sock.close()
        self.ctx.term()


def normalize_zmq_server_address(server_address: str) -> str:
    if server_address.startswith("tcp://"):
        return server_address

    host, sep, port = server_address.rpartition(":")
    if not sep or not host or not port:
        raise ValueError(
            f"pi05 backend expects server_address in 'host:port' format, got {server_address!r}"
        )

    return f"tcp://{host}:{port}"


def validate_pi05_robot_compatibility(
    robot: Any,
    *,
    front_camera_key: str,
    left_wrist_camera_key: str,
    right_wrist_camera_key: str,
) -> None:
    action_keys = list(robot.action_features)
    if action_keys != PI05_ACTION_KEYS:
        raise ValueError(
            "The 'pi05_zmq' backend currently supports only "
            "right_arm_0..6 + left_arm_0..6 + right_gripper_0 + left_gripper_0 actions. "
            f"Received action features: {action_keys}"
        )

    required_keys = [
        *PI05_STATE_KEYS,
        front_camera_key,
        left_wrist_camera_key,
        right_wrist_camera_key,
    ]
    missing_keys = [key for key in required_keys if key not in robot.observation_features]

    if missing_keys:
        raise ValueError(
            "Robot observation features are missing keys required by the pi05 backend: "
            f"{missing_keys}"
        )


def ensure_uint8_hwc_image(image: Any) -> np.ndarray:
    image_arr = np.asarray(image)

    if image_arr.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got {image_arr.shape}")

    if image_arr.shape[0] == 3 and image_arr.shape[-1] != 3:
        image_arr = np.transpose(image_arr, (1, 2, 0))

    if image_arr.shape[-1] != 3:
        raise ValueError(f"Expected image shape (H, W, 3), got {image_arr.shape}")

    if image_arr.dtype == np.uint8:
        return np.ascontiguousarray(image_arr)

    if np.issubdtype(image_arr.dtype, np.floating):
        max_val = image_arr.max(initial=0.0)
        image_arr = np.clip(image_arr, 0.0, 1.0 if max_val <= 1.0 else 255.0)

        if max_val <= 1.0:
            image_arr = image_arr * 255.0

    return np.ascontiguousarray(image_arr.astype(np.uint8))


def build_pi05_observation(
    raw_observation: dict[str, Any],
    *,
    front_camera_key: str,
    left_wrist_camera_key: str,
    right_wrist_camera_key: str,
) -> dict[str, Any]:
    required_keys = [
        *PI05_STATE_KEYS,
        front_camera_key,
        left_wrist_camera_key,
        right_wrist_camera_key,
    ]
    missing_keys = [key for key in required_keys if key not in raw_observation]

    if missing_keys:
        raise KeyError(f"Raw observation missing keys required by pi05 backend: {missing_keys}")

    state = np.asarray([raw_observation[key] for key in PI05_STATE_KEYS], dtype=np.float32)
    task = str(raw_observation.get("task", ""))

    return {
        "observation/front_image": ensure_uint8_hwc_image(raw_observation[front_camera_key]),
        "observation/left_image": ensure_uint8_hwc_image(raw_observation[left_wrist_camera_key]),
        "observation/right_image": ensure_uint8_hwc_image(raw_observation[right_wrist_camera_key]),
        "observation/state": state,
        "prompt": task,
    }


def parse_pi05_actions(action_dict: dict[str, Any]) -> np.ndarray:
    if "actions" in action_dict:
        actions = squeeze_optional_batch(np.asarray(action_dict["actions"], dtype=np.float32))

        if actions.ndim != 2 or actions.shape[1] < ACTION_DOF:
            raise ValueError(
                f"Expected pi05 actions shape (T, >=16) or (1, T, >=16), got {actions.shape}"
            )

        return np.ascontiguousarray(actions[:, :ACTION_DOF])

    right_arm = get_pi05_action_part(action_dict, "right_arm", ARM_DOF)
    left_arm = get_pi05_action_part(action_dict, "left_arm", ARM_DOF)
    right_gripper = get_pi05_action_part(action_dict, "right_gripper", GRIPPER_DOF)
    left_gripper = get_pi05_action_part(action_dict, "left_gripper", GRIPPER_DOF)

    lengths = [
        right_arm.shape[0],
        left_arm.shape[0],
        right_gripper.shape[0],
        left_gripper.shape[0],
    ]
    if len(set(lengths)) != 1:
        raise ValueError(f"pi05 action chunk lengths do not match: {lengths}")

    actions = np.concatenate(
        [right_arm, left_arm, right_gripper, left_gripper],
        axis=-1,
    )
    return np.ascontiguousarray(actions.astype(np.float32))


def get_pi05_action_part(
    action_dict: dict[str, Any],
    key: str,
    dim: int,
) -> np.ndarray:
    if key not in action_dict:
        raise KeyError(f"pi05 action response missing key: {key}")

    action = squeeze_optional_batch(np.asarray(action_dict[key], dtype=np.float32))

    if action.ndim != 2 or action.shape[1] != dim:
        raise ValueError(
            f"Expected {key} shape (T, {dim}) or (1, T, {dim}), got {action.shape}"
        )

    return action


def squeeze_optional_batch(array: np.ndarray) -> np.ndarray:
    if array.ndim >= 1 and array.shape[0] == 1:
        return array[0]
    return array


def actions_to_timed_actions(
    actions: np.ndarray,
    *,
    timestamp: float,
    timestep: int,
    environment_dt: float,
    client_device: str = "cpu",
) -> list[TimedAction]:
    if actions.ndim != 2 or actions.shape[1] != ACTION_DOF:
        raise ValueError(f"Expected actions shape (T, {ACTION_DOF}), got {actions.shape}")

    device = torch.device(client_device)
    timed_actions = []

    for i, action in enumerate(actions):
        action_tensor = torch.from_numpy(np.ascontiguousarray(action)).to(
            device=device,
            dtype=torch.float32,
        )
        timed_actions.append(
            TimedAction(
                timestamp=timestamp + i * environment_dt,
                timestep=timestep + i,
                action=action_tensor,
            )
        )

    return timed_actions


def pi05_action_dict_to_timed_actions(
    action_dict: dict[str, Any],
    *,
    timestamp: float,
    timestep: int,
    environment_dt: float,
    client_device: str = "cpu",
) -> list[TimedAction]:
    actions = parse_pi05_actions(action_dict)
    return actions_to_timed_actions(
        actions,
        timestamp=timestamp,
        timestep=timestep,
        environment_dt=environment_dt,
        client_device=client_device,
    )