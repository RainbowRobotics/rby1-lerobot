"""Configuration for the RB-Series Meta Quest VR teleoperator.

This configuration contains only values used by the current ``RbVr``
implementation:

- Meta Quest UDP communication
- Controller selection and buttons
- User reach scaling
- RB10E target translation
- End-effector orientation scaling
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
    2. Press the primary button to arm control and move the arm to the
       fixed home pose (constants.VR_HOME_POSE_DEG) over five seconds.
       Grip is ignored until the arm arrives.
    3. Press Grip to anchor, then move the hand: the arm follows the
       controller's translation 1:1, and applies ``orientation_scale`` of
       the controller's rotation to the end-effector orientation.
    4. Release Grip to stop ServoJ transmission and drop the anchor. The
       next press re-anchors, so the arm never jumps.
    5. Press the secondary button to disarm control completely.

    Because the anchor is taken on the Grip rising edge, an operator who
    holds Grip through the homing ramp must release and press again.
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
    # End-effector orientation
    # ------------------------------------------------------------------

    # Fraction of the controller's rotation since the clutch anchor that is
    # applied to the end-effector orientation, about the same axis.
    #
    #   0.0  the orientation is frozen at the anchor for the whole stroke.
    #        Bit-for-bit the behaviour of every build before this knob.
    #   0.3  a 90 deg wrist flick asks for 27 deg of tool rotation.
    #   1.0  1:1, matching the position mapping.
    #
    # Turn this down first if a stroke stops tracking. A wrist flick is
    # cheap for the operator and expensive for the arm, and at this home
    # pose the binding joint is the ELBOW, not the wrist: j2 sits 12.11 deg
    # from its +-154 deg IK limit and a 30 deg tool rotation about base Y
    # spends 6.8 deg of that. IKLM clips at the limit, so the arm saturates
    # instead of following. The wrist has more room -- j4 == 277.47 deg is
    # 82.5 deg from the j4 == 360 deg singularity (it is singular at 0, 180
    # and 360), and base Z rotation is what spends it, -30 deg taking j4 to
    # within 33.9 deg. Scaling buys fine orientation control and keeps both
    # margins; the default stays 1:1 because the operator asked for
    # orientation to match the 1:1 position mapping.
    #
    # Position stays 1:1 and has no knob -- see the DEPRECATED section.
    orientation_scale: float = 1.0

    # ------------------------------------------------------------------
    # DEPRECATED: user reach and absolute Cartesian target
    # ------------------------------------------------------------------
    #
    # Every field in this section belonged to the old ABSOLUTE mapping, in
    # which the controller's position in the torso frame was scaled and
    # offset into an RB10E workspace target.
    #
    # RbVr now uses an anchored 1:1 position delta and reads NONE of them.
    # There is deliberately no scale knob: the controller-to-end-effector
    # mapping is 1:1 because nothing can make it otherwise.
    #
    # They are kept, with their validators, only so that existing shell
    # commands and saved YAML configs carrying e.g.
    # --teleop.target_z_offset_mm do not fail with a TypeError. Remove them
    # once no caller passes them.

    auto_user_scale: bool = True

    reference_reach_mm: float = 1300.0

    default_user_scale: float = 1300.0 / 700.0

    position_scale: float = 1.0

    target_x_offset_mm: float = 600.0
    target_y_offset_mm: float = 400.0
    target_z_offset_mm: float = 900.0

    # ------------------------------------------------------------------
    # Inverse kinematics
    # ------------------------------------------------------------------

    # Number of IKLM iterations per control tick. The original RB10E VR
    # implementation used three; five converges a 100 mm step to well under
    # a millimetre from a warm seed.
    ik_iterations: int = 5

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

        # NOT _validate_finite_positive: 0.0 is the meaningful "freeze the
        # orientation" value, and that validator rejects it.
        if (
            not math.isfinite(self.orientation_scale)
            or not 0.0 <= self.orientation_scale <= 1.0
        ):
            raise ValueError(
                "orientation_scale must be finite and in [0, 1], "
                f"got {self.orientation_scale}."
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
