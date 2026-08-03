from dataclasses import dataclass


@dataclass(frozen=True)
class MQ3Config:
    # Shared memory
    shm_name: str = "mq3_shm"
    shm_buffer_count: int = 2

    # UDP
    recv_buffer_bytes: int = 4096
    socket_poll_timeout_s: float = 0.001
    send_handshake: bool = True

    # Resampling / jitter buffer
    output_hz: float = 100.0
    initial_buffer_s: float = 0.100
    target_buffer_s: float = 0.100
    max_pose_samples: int = 512

    # Slowly adjusts sender-time playback speed so the jitter buffer
    # converges toward target_buffer_s without discontinuous jumps.
    playback_catchup_gain: float = 0.5
    min_playback_rate: float = 0.98
    max_playback_rate: float = 1.05

    # If no future sample exists, keep the last output pose.
    # A value <= 0 disables stale warnings.
    stale_warning_s: float = 0.250

    # Input / control settings
    grip_threshold: float = 0.5
    initial_user_scale: float = 1000.0 / 700.0
    user_scale_reference_mm: float = 1300.0


DEFAULT_MQ3_CONFIG = MQ3Config()
