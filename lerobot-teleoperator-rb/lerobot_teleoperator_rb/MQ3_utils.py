from __future__ import annotations

import bisect
import json
import logging
import multiprocessing as mp
import socket
import time
from dataclasses import dataclass

import numpy as np

from .DoubleBuffer import DoubleBuffer
from .MQ3_Config import DEFAULT_MQ3_CONFIG, MQ3Config
from .frame_transforms import apply_scale, invert_se3, pose_to_se3
from .frame_transforms import T_conv, T_for_head, T_for_RB10E as T_for_y2c


VR_DTYPE = np.dtype([
    ("right_controller_current_pose", np.float64, (4, 4)),
    ("left_controller_current_pose", np.float64, (4, 4)),
    ("head_controller_current_pose", np.float64, (4, 4)),
    ("torso_current_pose", np.float64, (4, 4)),
    ("user_scale", np.float64),

    ("is_right_following", np.bool_),
    ("is_left_following", np.bool_),

    ("event_right_grip_value", np.float64),
    ("event_left_grip_value", np.float64),
    ("event_right_servoing", np.bool_),
    ("event_left_servoing", np.bool_),
    ("event_right_servoing_prev", np.bool_),
    ("event_left_servoing_prev", np.bool_),
    ("event_right_servoing_rising_edge", np.bool_),
    ("event_left_servoing_rising_edge", np.bool_),
    ("event_right_servoing_falling_edge", np.bool_),
    ("event_left_servoing_falling_edge", np.bool_),

    ("event_right_trigger_value", np.float64),
    ("event_left_trigger_value", np.float64),

    ("event_right_a_pressed", np.bool_),
    ("event_right_b_pressed", np.bool_),
    ("event_left_a_pressed", np.bool_),
    ("event_left_b_pressed", np.bool_),
])


@dataclass
class PoseSample:
    sender_timestamp: float
    right_pose: np.ndarray
    left_pose: np.ndarray
    head_pose: np.ndarray
    discrete_state: np.ndarray


# ---------------------------------------------------------------------------
# SE(3) helpers
# ---------------------------------------------------------------------------

def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=np.float64)
    return np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ], dtype=np.float64)


def _so3_exp(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=np.float64)
    theta = float(np.linalg.norm(phi))
    Phi = _skew(phi)
    Phi2 = Phi @ Phi

    if theta < 1e-8:
        return np.eye(3) + Phi + 0.5 * Phi2

    a = np.sin(theta) / theta
    b = (1.0 - np.cos(theta)) / (theta * theta)
    return np.eye(3) + a * Phi + b * Phi2


def _so3_log(R: np.ndarray) -> np.ndarray:
    cos_theta = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))

    if theta < 1e-8:
        return 0.5 * np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ])

    # The usual formula becomes numerically fragile near pi.
    if np.pi - theta < 1e-5:
        A = (R + np.eye(3)) * 0.5
        axis = np.sqrt(np.maximum(np.diag(A), 0.0))
        if R[2, 1] - R[1, 2] < 0:
            axis[0] *= -1
        if R[0, 2] - R[2, 0] < 0:
            axis[1] *= -1
        if R[1, 0] - R[0, 1] < 0:
            axis[2] *= -1
        norm = np.linalg.norm(axis)
        if norm < 1e-8:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis /= norm
        return theta * axis

    scale = theta / (2.0 * np.sin(theta))
    return scale * np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ])


def _transform_inverse(T: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    R = T[:3, :3]
    p = T[:3, 3]
    result[:3, :3] = R.T
    result[:3, 3] = -(R.T @ p)
    return result


def _se3_log(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    p = T[:3, 3]
    phi = _so3_log(R)
    theta = float(np.linalg.norm(phi))
    Phi = _skew(phi)
    Phi2 = Phi @ Phi

    if theta < 1e-8:
        V = np.eye(3) + 0.5 * Phi + (1.0 / 6.0) * Phi2
    else:
        theta2 = theta * theta
        theta3 = theta2 * theta
        V = (
            np.eye(3)
            + ((1.0 - np.cos(theta)) / theta2) * Phi
            + ((theta - np.sin(theta)) / theta3) * Phi2
        )

    rho = np.linalg.solve(V, p)
    return np.concatenate((rho, phi))


def _se3_exp(xi: np.ndarray) -> np.ndarray:
    xi = np.asarray(xi, dtype=np.float64)
    rho = xi[:3]
    phi = xi[3:]
    theta = float(np.linalg.norm(phi))
    Phi = _skew(phi)
    Phi2 = Phi @ Phi

    if theta < 1e-8:
        V = np.eye(3) + 0.5 * Phi + (1.0 / 6.0) * Phi2
    else:
        theta2 = theta * theta
        theta3 = theta2 * theta
        V = (
            np.eye(3)
            + ((1.0 - np.cos(theta)) / theta2) * Phi
            + ((theta - np.sin(theta)) / theta3) * Phi2
        )

    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = _so3_exp(phi)
    result[:3, 3] = V @ rho
    return result


def interpolate_se3(T0: np.ndarray, T1: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    relative = _transform_inverse(T0) @ T1
    return T0 @ _se3_exp(alpha * _se3_log(relative))


# ---------------------------------------------------------------------------
# Timestamp-ordered input buffer
# ---------------------------------------------------------------------------

class PoseJitterBuffer:
    def __init__(self, max_samples: int):
        self.max_samples = max_samples
        self.timestamps: list[float] = []
        self.samples: list[PoseSample] = []

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def first_timestamp(self) -> float:
        return self.timestamps[0]

    @property
    def last_timestamp(self) -> float:
        return self.timestamps[-1]

    def insert(self, sample: PoseSample) -> None:
        index = bisect.bisect_left(self.timestamps, sample.sender_timestamp)

        if index < len(self.timestamps) and abs(
            self.timestamps[index] - sample.sender_timestamp
        ) < 1e-12:
            self.samples[index] = sample
            return

        self.timestamps.insert(index, sample.sender_timestamp)
        self.samples.insert(index, sample)

        overflow = len(self.samples) - self.max_samples
        if overflow > 0:
            del self.timestamps[:overflow]
            del self.samples[:overflow]

    def sample(self, target_timestamp: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        if len(self.samples) < 2:
            return None

        right_index = bisect.bisect_right(self.timestamps, target_timestamp)
        if right_index == 0 or right_index >= len(self.samples):
            return None

        left_index = right_index - 1
        s0 = self.samples[left_index]
        s1 = self.samples[right_index]
        dt = s1.sender_timestamp - s0.sender_timestamp
        if dt <= 1e-12:
            alpha = 1.0
        else:
            alpha = (target_timestamp - s0.sender_timestamp) / dt

        right_pose = interpolate_se3(s0.right_pose, s1.right_pose, alpha)
        left_pose = interpolate_se3(s0.left_pose, s1.left_pose, alpha)
        head_pose = interpolate_se3(s0.head_pose, s1.head_pose, alpha)

        # Buttons/tracking flags are zero-order-held from the most recent
        # sample at or before target_timestamp.
        discrete_state = s0.discrete_state.copy()
        return right_pose, left_pose, head_pose, discrete_state

    def prune_before(self, timestamp: float) -> None:
        if len(self.samples) <= 2:
            return
        index = bisect.bisect_left(self.timestamps, timestamp)
        delete_count = max(0, index - 1)
        if delete_count:
            del self.timestamps[:delete_count]
            del self.samples[:delete_count]


class MQ3:
    def __init__(
        self,
        local_ip: str,
        local_port: int,
        meta_quest_ip: str,
        meta_quest_port: int,
        config: MQ3Config = DEFAULT_MQ3_CONFIG,
    ):
        self.local_ip = local_ip
        self.local_port = local_port
        self.meta_quest_ip = meta_quest_ip
        self.meta_quest_port = meta_quest_port
        self.config = config

        self.vr_state = np.zeros((), dtype=VR_DTYPE)
        self.vr_state["user_scale"] = config.initial_user_scale
        self._initialize_poses(self.vr_state)

        self.shm = DoubleBuffer(
            VR_DTYPE,
            buffer_count=config.shm_buffer_count,
            create=True,
            shm_name=config.shm_name,
        )

        self.process = mp.Process(
            target=self.setup_meta_quest_udp_communication,
            args=(local_ip, local_port, meta_quest_ip, meta_quest_port),
            daemon=True,
        )

    @staticmethod
    def _initialize_poses(state: np.ndarray) -> None:
        identity = np.eye(4, dtype=np.float64)
        state["right_controller_current_pose"] = identity
        state["left_controller_current_pose"] = identity
        state["head_controller_current_pose"] = identity
        state["torso_current_pose"] = identity

    def __del__(self):
        try:
            self.shm.close()
        except Exception:
            pass

    def _send_receiver_address(
        self,
        local_ip: str,
        local_port: int,
        meta_quest_ip: str,
        meta_quest_port: int,
    ) -> None:
        target_info = {"ip": local_ip, "port": local_port}
        message = json.dumps(target_info).encode("utf-8")

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.sendto(message, (meta_quest_ip, meta_quest_port))

        logging.info("Sent local PC info to Meta Quest: %s", target_info)

    def _parse_packet(
        self,
        state: dict,
        previous_sample: PoseSample | None,
        T_conv_T: np.ndarray,
    ) -> PoseSample | None:
        timestamp = state.get("timestamp")
        if timestamp is None:
            logging.warning("Dropped VR packet without sender timestamp")
            return None

        # Carry the previous pose/state through temporary tracking loss.
        if previous_sample is None:
            right_pose = np.eye(4, dtype=np.float64)
            left_pose = np.eye(4, dtype=np.float64)
            head_pose = np.eye(4, dtype=np.float64)
            discrete = np.zeros((), dtype=VR_DTYPE)
            discrete["user_scale"] = self.config.initial_user_scale
            self._initialize_poses(discrete)
        else:
            right_pose = previous_sample.right_pose.copy()
            left_pose = previous_sample.left_pose.copy()
            head_pose = previous_sample.head_pose.copy()
            discrete = previous_sample.discrete_state.copy()

        hands = state.get("hands") or {}

        right = hands.get("right")
        if right is not None:
            buttons = right.get("buttons", {})
            discrete["is_right_following"] = True
            discrete["event_right_a_pressed"] = bool(buttons.get("primaryButton", False))
            discrete["event_right_b_pressed"] = bool(buttons.get("secondaryButton", False))
            discrete["event_right_trigger_value"] = float(buttons.get("trigger", 0.0))
            discrete["event_right_grip_value"] = float(buttons.get("grip", 0.0))
            T_r = pose_to_se3(right["position"], right["rotation"])
            right_pose = T_conv_T @ T_r @ T_conv
        else:
            discrete["is_right_following"] = False

        left = hands.get("left")
        if left is not None:
            buttons = left.get("buttons", {})
            discrete["is_left_following"] = True
            discrete["event_left_a_pressed"] = bool(buttons.get("primaryButton", False))
            discrete["event_left_b_pressed"] = bool(buttons.get("secondaryButton", False))
            discrete["event_left_trigger_value"] = float(buttons.get("trigger", 0.0))
            discrete["event_left_grip_value"] = float(buttons.get("grip", 0.0))
            T_l = pose_to_se3(left["position"], left["rotation"])
            left_pose = T_conv_T @ T_l @ T_conv
        else:
            discrete["is_left_following"] = False

        head = state.get("head")
        if head is not None:
            T_h = pose_to_se3(head["position"], head["rotation"])
            head_pose = T_conv_T @ T_h @ T_conv

        return PoseSample(
            sender_timestamp=float(timestamp),
            right_pose=right_pose,
            left_pose=left_pose,
            head_pose=head_pose,
            discrete_state=discrete,
        )

    def _build_output_state(
        self,
        right_world: np.ndarray,
        left_world: np.ndarray,
        head_world: np.ndarray,
        discrete_state: np.ndarray,
    ) -> None:
        # Preserve the current scale, since it is receiver-side state rather
        # than a value that should be zero-order-held from old packets.
        current_scale = float(self.vr_state["user_scale"])
        prev_right = bool(self.vr_state["event_right_servoing_prev"])
        prev_left = bool(self.vr_state["event_left_servoing_prev"])

        self.vr_state[...] = discrete_state
        self.vr_state["user_scale"] = current_scale

        self.vr_state["head_controller_current_pose"] = head_world
        torso_world = head_world @ T_for_head
        self.vr_state["torso_current_pose"] = torso_world
        T_inv = invert_se3(torso_world)

        if self.vr_state["event_right_a_pressed"]:
            right_from_torso = T_inv @ right_world
            distance = float(np.linalg.norm(right_from_torso[:3, 3]))
            # if distance > 1e-9:
            #     self.vr_state["user_scale"] = (
            #         self.config.user_scale_reference_mm / distance
            #     )

        right_relative = T_inv @ right_world @ T_for_y2c
        left_relative = T_inv @ left_world @ T_for_y2c

        self.vr_state["right_controller_current_pose"] = apply_scale(
            right_relative,
            self.vr_state["user_scale"],
        )
        self.vr_state["left_controller_current_pose"] = apply_scale(
            left_relative,
            self.vr_state["user_scale"],
        )

        threshold = self.config.grip_threshold
        right_servoing = bool(self.vr_state["event_right_grip_value"] > threshold)
        left_servoing = bool(self.vr_state["event_left_grip_value"] > threshold)

        self.vr_state["event_right_servoing"] = right_servoing
        self.vr_state["event_left_servoing"] = left_servoing
        self.vr_state["event_right_servoing_rising_edge"] = right_servoing and not prev_right
        self.vr_state["event_left_servoing_rising_edge"] = left_servoing and not prev_left
        self.vr_state["event_right_servoing_falling_edge"] = not right_servoing and prev_right
        self.vr_state["event_left_servoing_falling_edge"] = not left_servoing and prev_left

        if self.vr_state["event_right_servoing_rising_edge"]:
            logging.info("Right hand servoing started.")
        if self.vr_state["event_left_servoing_rising_edge"]:
            logging.info("Left hand servoing started.")
        if self.vr_state["event_right_servoing_falling_edge"]:
            logging.info("Right hand servoing stopped.")
        if self.vr_state["event_left_servoing_falling_edge"]:
            logging.info("Left hand servoing stopped.")
            
        self.vr_state["event_right_servoing_prev"] = right_servoing
        self.vr_state["event_left_servoing_prev"] = left_servoing


    def setup_meta_quest_udp_communication(
        self,
        local_ip: str,
        local_port: int,
        meta_quest_ip: str,
        meta_quest_port: int,
    ) -> None:
        if self.config.send_handshake:
            self._send_receiver_address(
                local_ip,
                local_port,
                meta_quest_ip,
                meta_quest_port,
            )

        config = self.config
        output_period = 1.0 / config.output_hz
        T_conv_T = T_conv.T
        pose_buffer = PoseJitterBuffer(config.max_pose_samples)

        previous_input_sample: PoseSample | None = None
        playback_timestamp: float | None = None
        last_packet_monotonic: float | None = None
        stale_warning_emitted = False
        next_output_deadline = time.perf_counter()

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server_sock:
            server_sock.bind((local_ip, local_port))
            server_sock.setblocking(False)
            logging.info("UDP server running... %s:%d", local_ip, local_port)

            while True:
                # Drain every packet currently waiting in the kernel socket
                # buffer. Burst arrivals are retained in sender-time order.
                while True:
                    try:
                        data, _ = server_sock.recvfrom(config.recv_buffer_bytes)
                    except BlockingIOError:
                        break

                    try:
                        packet = json.loads(data)
                        sample = self._parse_packet(
                            packet,
                            previous_input_sample,
                            T_conv_T,
                        )
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                        logging.exception("Invalid VR packet")
                        continue

                    if sample is not None:
                        pose_buffer.insert(sample)
                        previous_input_sample = sample
                        last_packet_monotonic = time.monotonic()
                        stale_warning_emitted = False

                now = time.perf_counter()
                if now < next_output_deadline:
                    time.sleep(min(
                        config.socket_poll_timeout_s,
                        next_output_deadline - now,
                    ))
                    continue

                # Avoid accumulating scheduler delay after a long preemption.
                if now - next_output_deadline > output_period:
                    next_output_deadline = now
                next_output_deadline += output_period

                if playback_timestamp is None:
                    if len(pose_buffer) >= 2:
                        buffered_duration = (
                            pose_buffer.last_timestamp
                            - pose_buffer.first_timestamp
                        )
                        if buffered_duration >= config.initial_buffer_s:
                            playback_timestamp = pose_buffer.first_timestamp
                    continue

                available_ahead = pose_buffer.last_timestamp - playback_timestamp
                buffer_error = available_ahead - config.target_buffer_s
                playback_rate = float(np.clip(
                    1.0 + config.playback_catchup_gain * buffer_error,
                    config.min_playback_rate,
                    config.max_playback_rate,
                ))
                candidate_timestamp = playback_timestamp + output_period * playback_rate

                sampled = pose_buffer.sample(candidate_timestamp)
                if sampled is not None:
                    playback_timestamp = candidate_timestamp
                    right_pose, left_pose, head_pose, discrete_state = sampled
                    self._build_output_state(
                        right_pose,
                        left_pose,
                        head_pose,
                        discrete_state,
                    )
                    self.shm.write(self.vr_state)
                    pose_buffer.prune_before(playback_timestamp)
                else:
                    # No future sender-time sample exists yet. Keep the most
                    # recent shared-memory output instead of jumping ahead.
                    if (
                        config.stale_warning_s > 0.0
                        and last_packet_monotonic is not None
                        and not stale_warning_emitted
                        and time.monotonic() - last_packet_monotonic
                        >= config.stale_warning_s
                    ):
                        logging.warning(
                            "VR pose stream is stale for at least %.3f s",
                            config.stale_warning_s,
                        )
                        stale_warning_emitted = True

    def start(self) -> mp.Process:
        self.process.start()
        return self.process
