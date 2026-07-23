"""Configuration for the RB-Series Meta Quest VR teleoperator.

This configuration contains only values used by the current ``RbVr``
implementation:

- Meta Quest UDP communication
- Controller selection and buttons
- User reach scaling
- RB10E target translation
- IK iteration count
- Optional gripper output

Robot TCP communication and ServoJ parameters belong to
``lerobot_robot_rb.config_rb.RbCobotConfig``.
"""

from __future__ import annotations

import ipaddress
import math
from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("rb_vr")
@dataclass
class RbVrConfig(TeleoperatorConfig):
    """Meta Quest VR teleoperator configuration.

    The default configuration requires the selected controller's primary
    button—Right A by default—to initialize VR control.

    Control sequence
    ----------------
    1. Receive valid controller and headset tracking.
    2. Press the primary button to initialize user scale.
    3. Hold Grip to enable robot following.
    4. Release Grip to stop ServoJ transmission.
    5. Press the secondary button to disarm control completely.
    """

    # ------------------------------------------------------------------
    # Meta Quest UDP communication
    # ------------------------------------------------------------------

    # Concrete PC IP is required when send_handshake=True.
    #
    # Use "0.0.0.0" only when:
    #     send_handshake=False
    local_ip: str = "0.0.0.0"
    local_port: int = 5005

    # Quest IP may be omitted only when the handshake is disabled.
    meta_quest_ip: str | None = None
    meta_quest_port: int = 6000

    # Sends:
    #     {"ip": local_ip, "port": local_port}
    #
    # to Meta Quest when the receiver starts.
    send_handshake: bool = False

    # Tracking packets older than this are considered stale.
    tracking_timeout_s: float = 0.25

    # ------------------------------------------------------------------
    # Controller and safety controls
    # ------------------------------------------------------------------

    controller_hand: str = "right"

    # Grip values greater than this enable following.
    grip_threshold: float = 0.5

    # True:
    #     Primary/A button must be pressed before Grip can move the arm.
    #
    # False:
    #     VR control is armed immediately after connection.
    require_initialization_button: bool = True

    # ------------------------------------------------------------------
    # User reach and Cartesian target
    # ------------------------------------------------------------------

    # Pressing A computes:
    #
    #     user_scale =
    #         reference_reach_mm
    #         / measured_controller_reach_mm
    auto_user_scale: bool = True

    reference_reach_mm: float = 1300.0

    # Used when auto_user_scale=False.
    default_user_scale: float = 1300.0 / 700.0

    # Additional multiplier applied after user calibration.
    position_scale: float = 1.0

    # Added to the scaled RB10E target Z translation.
    target_x_offset_mm: float = 300.0
    target_y_offset_mm: float = 200.0
    target_z_offset_mm: float = 600.0

    # ------------------------------------------------------------------
    # Inverse kinematics
    # ------------------------------------------------------------------

    # Original RB10E VR implementation uses three IKLM iterations per tick.
    ik_iterations: int = 3

    # ------------------------------------------------------------------
    # Optional gripper
    # ------------------------------------------------------------------

    # When enabled, the selected controller trigger produces:
    #
    #     gripper_0 = 1 - trigger
    #
    # LeRobot convention:
    #     1.0 = open
    #     0.0 = closed
    use_gripper: bool = False

    def __post_init__(self) -> None:
        parent_post_init = getattr(
            super(),
            "__post_init__",
            None,
        )
        if parent_post_init is not None:
            parent_post_init()

        self._validate_ip(
            self.local_ip,
            field_name="local_ip",
            allow_none=False,
        )

        self._validate_port(
            self.local_port,
            field_name="local_port",
        )

        self._validate_ip(
            self.meta_quest_ip,
            field_name="meta_quest_ip",
            allow_none=True,
        )

        self._validate_port(
            self.meta_quest_port,
            field_name="meta_quest_port",
        )

        if self.send_handshake:
            if self.meta_quest_ip is None:
                raise ValueError(
                    "meta_quest_ip is required when "
                    "send_handshake=True."
                )

            if self.local_ip == "0.0.0.0":
                raise ValueError(
                    "local_ip must be a concrete reachable PC IP "
                    "when send_handshake=True."
                )

        if self.controller_hand not in (
            "right",
            "left",
        ):
            raise ValueError(
                "controller_hand must be 'right' or 'left', "
                f"got {self.controller_hand!r}."
            )

        self._validate_finite_positive(
            self.tracking_timeout_s,
            field_name="tracking_timeout_s",
        )

        if (
            not math.isfinite(self.grip_threshold)
            or not 0.0 <= self.grip_threshold <= 1.0
        ):
            raise ValueError(
                "grip_threshold must be finite and in [0, 1], "
                f"got {self.grip_threshold}."
            )

        self._validate_finite_positive(
            self.reference_reach_mm,
            field_name="reference_reach_mm",
        )

        self._validate_finite_positive(
            self.default_user_scale,
            field_name="default_user_scale",
        )

        self._validate_finite_positive(
            self.position_scale,
            field_name="position_scale",
        )

        for field_name in (
            "target_x_offset_mm",
            "target_y_offset_mm",
            "target_z_offset_mm",
        ):
            value = getattr(self, field_name)

            if not math.isfinite(value):
                raise ValueError(
                    f"{field_name} must be finite, got {value}."
                )

        if not isinstance(
            self.ik_iterations,
            int,
        ):
            raise TypeError(
                "ik_iterations must be an integer, "
                f"got {type(self.ik_iterations).__name__}."
            )

        if self.ik_iterations <= 0:
            raise ValueError(
                "ik_iterations must be > 0, "
                f"got {self.ik_iterations}."
            )

    @staticmethod
    def _validate_ip(
        value: str | None,
        *,
        field_name: str,
        allow_none: bool,
    ) -> None:
        if value is None:
            if allow_none:
                return

            raise ValueError(
                f"{field_name} must not be None."
            )

        if not isinstance(value, str):
            raise TypeError(
                f"{field_name} must be a string, "
                f"got {type(value).__name__}."
            )

        try:
            ipaddress.IPv4Address(value)
        except ipaddress.AddressValueError as exc:
            raise ValueError(
                f"{field_name} must be a valid IPv4 address, "
                f"got {value!r}."
            ) from exc

    @staticmethod
    def _validate_port(
        value: int,
        *,
        field_name: str,
    ) -> None:
        if not isinstance(value, int):
            raise TypeError(
                f"{field_name} must be an integer, "
                f"got {type(value).__name__}."
            )

        if not 0 < value <= 65535:
            raise ValueError(
                f"{field_name} must be in [1, 65535], "
                f"got {value}."
            )

    @staticmethod
    def _validate_finite_positive(
        value: float,
        *,
        field_name: str,
    ) -> None:
        if (
            not math.isfinite(value)
            or value <= 0.0
        ):
            raise ValueError(
                f"{field_name} must be finite and > 0, "
                f"got {value}."
            )
