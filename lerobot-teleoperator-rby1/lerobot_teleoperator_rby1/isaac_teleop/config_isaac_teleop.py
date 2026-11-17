"""Config dataclasses for the Isaac Capture devices.

Adapted from the upstream LeRobot example
``examples/isaac_teleop_to_so101/isaac_teleop/config_isaac_teleop.py``.

Unlike upstream, :class:`IsaacTeleopConfig` does **not** shadow the draccus
``_choice_registry``: :class:`Rby1XRConfig` registers ``rby1_isaac`` in the
global :class:`TeleoperatorConfig` registry, so the device is selectable with
``--teleop.type=rby1_isaac`` from the stock ``lerobot-teleoperate`` and
``lerobot-record`` CLIs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig

from rby1_xr_teleop.settings import XrTeleopSettings

# Static rebase from the OpenXR anchor frame (X=Right, Y=Up, Z=Backward) into the
# robot base frame (X=Forward, Y=Left, Z=Up). A proper rotation (det=+1):
# controller motion forward -> robot +X, right -> robot -Y, up -> robot +Z.
DEFAULT_BASE_T_ANCHOR: list[list[float]] = [
    [0.0, 0.0, -1.0, 0.0],
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]

SESSION_START_CHOICES = ("connect", "first_action")
ROBOT_VERSION_CHOICES = ("auto", "1.2", "1.3")
VIZ_LOCK_CHOICES = ("gimbal", "head", "world")
VIZ_BRIDGE_CHOICES = ("auto", "identity")


@dataclass(kw_only=True)
class IsaacTeleopConfig(TeleoperatorConfig):
    """Shared config for all Isaac Capture-backed teleoperators."""

    # Session name shown by the CloudXR runtime / OpenXR.
    app_name: str = "rby1_isaac"

    # Attach to the CloudXR service, starting it when none is running. The
    # service is detached: it keeps running after this process exits (stop it
    # with `python -m isaaccapture.cloudxr.service stop`). Set False (or export
    # LEROBOT_CLOUDXR_SKIP_AUTOLAUNCH=1) to skip the launcher entirely.
    auto_launch_cloudxr: bool = True
    # Run the CloudXR runtime inside this process instead (stopped on
    # disconnect; e.g. containers). Refused while a service is already running.
    cloudxr_run_embedded: bool = False
    # Opening the session blocks until a headset connects. > 0 bounds that wait
    # (seconds, TimeoutError afterwards); 0 waits forever (Ctrl-C to abort).
    xr_connect_timeout_s: float = 0.0
    # KEY=value profile passed to CloudXRLauncher(env_config=...). None uses the
    # packaged ``default.env`` (Quest3 profile).
    cloudxr_env_file: str | None = None
    # Where CloudXRLauncher installs / finds the runtime.
    cloudxr_install_dir: str = "~/.cloudxr"
    # CloudXR device profile ("Quest3" also serves Pico / WebXR clients).
    cloudxr_device_profile: str = "Quest3"

    # Static anchor -> robot-base rebase applied in-graph to every tracked pose.
    base_T_anchor: list[list[float]] = field(  # noqa: N815
        default_factory=lambda: [row.copy() for row in DEFAULT_BASE_T_ANCHOR]
    )


@TeleoperatorConfig.register_subclass("rby1_isaac")
@dataclass(kw_only=True)
class Rby1XRConfig(IsaacTeleopConfig, XrTeleopSettings):
    """RB-Y1 XR device: both controllers -> arm EE poses, headset -> head
    joints, body tracking -> torso EE pose.

    This teleoperator only *produces* actions; execution and all Cartesian
    impedance tuning live in the follower ``Rby1Config`` (``action_mode="ee"``).
    The ``use_*`` flags must mirror the follower's so the action keys match.

    How the operator's motion becomes targets — the arm mapping, torso, head,
    base, body drive, start and grip gates — is configured by the fields
    inherited from :class:`rby1_xr_teleop.settings.XrTeleopSettings`. The
    fields below are what the LeRobot device adds: the robot link, the XR
    session, the headset camera panels and diagnostics.
    """

    # ── Read-only robot link (state + forward kinematics for the clutch) ──
    robot_address: str = "192.168.30.1:50051"
    # "a" | "m" | "auto" (detect from the robot, like the follower does)
    robot_model: str = "auto"

    # Record this teleoperator's config in the robot profile file (use the
    # same path as --robot.profile_out, e.g. <dataset_root>/meta/rby1_profile.json).
    # It is stored under "teleop" for reproducibility and never compared.
    profile_out: str | None = None

    # Robot version selects the reach constants ("auto" = probe the robot).
    robot_version: str = "auto"

    # ── Headset camera panels (Televiz) ──────────────────────────────
    # One quad per robot camera name (as configured in --robot.cameras); the
    # frames come from the in-process frame bus. Televiz then owns the OpenXR
    # session (the tracking session attaches to it).
    viz_enabled: bool = False
    viz_cameras: list[str] = field(default_factory=lambda: ["front", "left", "right"])
    viz_offsets_x: list[float] = field(default_factory=lambda: [0.0, -1.1, 1.1])  # m, per camera
    viz_offset_y: float = 0.0
    viz_distance_m: float = 1.5
    viz_width_m: float = 1.0
    viz_lock_mode: str = "gimbal"         # "gimbal" (position + yaw [+ pitch]) | "head" | "world"
    viz_follow_pitch: bool = True         # gimbal: panels also follow the head pitch (never roll)
    # Native OpenXR quad layers. Keep False on Jetson Orin: there the native
    # path shows a solid colour instead of the image (still so in isaaccapture
    # 1.6.152, see the Televiz docs "Jetson Orin, XR"); False composites the
    # panels into the projection layer instead.
    viz_openxr_composition: bool = False
    viz_wait_headset_s: int = -1          # VizSession.create waits for the headset (-1 = forever)
    # Grab handle above each panel (like the CloudXR client's control panel):
    # point a FREE hand (squeeze released) at the bar and hold the trigger to
    # drag it; the thumbstick of that hand pushes / pulls. That hand's gripper
    # and thumbstick are held while grabbing.
    viz_grab_enabled: bool = True
    viz_grab_threshold: float = 0.7       # trigger value that starts a grab (release below 0.3)
    viz_grab_push_rate_mps: float = 0.5   # thumbstick-y push / pull speed while grabbing
    viz_face_head: bool = True            # panels turn to face the head (gimbal / world)
    viz_frame_bridge: str = "auto"        # "auto": estimate Televiz<-teleop frame from head poses | "identity"
    viz_layout_file: str = "~/.cache/rby1_isaac/viz_layout.json"  # "" = do not persist moved panels
    viz_layout_reset: bool = False        # ignore the saved layout once (start from the config values)

    # ── Session ─────────────────────────────────────────────────────
    # When to open the CloudXR/OpenXR session: at connect() (default) or lazily
    # on the first get_action(). "first_action" is a mitigation for Jetson Orin
    # hosts where creating Python threads after the CloudXR service starts can
    # abort the process: lerobot-record creates its keyboard-listener and
    # camera threads between teleop.connect() and the first get_action().
    session_start: str = "connect"
    # Seconds to wait for the headset controllers before giving up (0 = forever).
    tracking_wait_timeout_s: float = 0.0

    # ── Diagnostics ───────────────────────────────────────────────────
    # Log a one-line tracking / clutch status this often (seconds; 0 = off).
    status_log_period_s: float = 5.0

    def __post_init__(self) -> None:
        parent_post_init = getattr(super(), "__post_init__", None)
        if parent_post_init is not None:
            parent_post_init()
        self.validate()  # the inherited behaviour settings
        for name, value, choices in (
            ("session_start", self.session_start, SESSION_START_CHOICES),
            ("viz_lock_mode", self.viz_lock_mode, VIZ_LOCK_CHOICES),
            ("viz_frame_bridge", self.viz_frame_bridge, VIZ_BRIDGE_CHOICES),
            ("robot_version", self.robot_version, ROBOT_VERSION_CHOICES),
        ):
            if value not in choices:
                raise ValueError(f"{name} must be one of {choices}, got {value!r}")
        if self.viz_enabled and len(self.viz_offsets_x) != len(self.viz_cameras):
            raise ValueError("viz_offsets_x must have one entry per viz_cameras entry")
        if not (0.0 < self.viz_grab_threshold <= 1.0):
            raise ValueError("viz_grab_threshold must be in (0, 1]")
        if self.robot_model.strip().lower() not in ("a", "m", "auto"):
            raise ValueError(f'robot_model must be "a", "m" or "auto", got {self.robot_model!r}')

    @property
    def needs_body(self) -> bool:
        """Whether the FullBodySource must be in the pipeline."""
        return True  # the absolute arm mapping always needs the IOBT shoulder
