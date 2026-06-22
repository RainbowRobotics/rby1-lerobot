"""Probe an RB-Y1 robot for its model name and version via the rby1-sdk.

This mirrors the auto-detection used by RainbowRobotics/rby1_ros2 (ros2_control
branch): connect to the robot, read ``RobotInfo.robot_model_name`` /
``robot_model_version``, and normalise them to the model / version strings the
rest of the package understands.

The probe uses ``rby1_sdk.create_robot_a`` because the gRPC backend is the same
for every model, so an A-template handle is sufficient to read ``RobotInfo``.
The handle is disconnected before returning.
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Firmware versions supported per model. A probed version outside this set
# triggers a warning and a fallback to DEFAULT_VERSION.
SUPPORTED_VERSIONS: dict[str, tuple[str, ...]] = {
    "a": ("1.0", "1.1", "1.2"),
    "m": ("1.0", "1.1", "1.2", "1.3"),
    "ub": (),  # upper-body model: no version-specific ready pose
}
DEFAULT_VERSION = "1.2"

# Accepted explicit config values.
VALID_MODELS = ("a", "m", "ub", "auto")
VALID_VERSIONS = ("auto", "", "1.0", "1.1", "1.2", "1.3")


def detect_model_and_version(
    address: str, connect_timeout_sec: float = 3.0
) -> Tuple[str, Optional[str]]:
    """Connect to ``address`` and return ``(model, version)``.

    ``model`` is normalised to ``"a"``, ``"m"`` or ``"ub"``. ``version`` is the
    first ``MAJOR.MINOR`` token found in ``RobotInfo.robot_model_version``, or
    ``None`` if no such token can be extracted (the caller should then fall back
    to its declared default).

    Raises ``RuntimeError`` if the SDK is missing, the connection fails, or the
    reported model name is not recognised.
    """
    try:
        import rby1_sdk as rby
    except ImportError as exc:
        raise RuntimeError(
            "rby1_sdk is not installed; cannot auto-detect the robot model. "
            "Install the RB-Y1 SDK, or set model=a|m|ub (and version) explicitly."
        ) from exc

    timeout_ms = max(1, int(round(connect_timeout_sec * 1000.0)))

    try:
        robot = rby.create_robot_a(address)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to create an RB-Y1 SDK handle for '{address}': {exc}"
        ) from exc

    try:
        try:
            connected = robot.connect(max_retries=1, timeout_ms=timeout_ms)
        except Exception as exc:
            raise RuntimeError(
                f"RB-Y1 model probe failed to connect to '{address}': {exc}"
            ) from exc
        if not connected:
            raise RuntimeError(
                f"RB-Y1 model probe could not connect to '{address}' "
                f"within {connect_timeout_sec:.2f}s."
            )

        try:
            info = robot.get_robot_info()
        except Exception as exc:
            raise RuntimeError(
                f"RB-Y1 model probe could not read robot info from '{address}': {exc}"
            ) from exc

        raw_name = (getattr(info, "robot_model_name", "") or "").strip().lower()
        raw_version = (getattr(info, "robot_model_version", "") or "").strip()

        if raw_name.startswith("a"):
            model = "a"
        elif raw_name.startswith("m"):
            model = "m"
        elif raw_name.startswith("ub"):
            model = "ub"
        else:
            raise RuntimeError(
                f"RB-Y1 model probe returned an unsupported model name "
                f"'{raw_name}' from '{address}'."
            )

        match = re.search(r"(\d+\.\d+)", raw_version)
        version = match.group(1) if match else None

        logger.info(
            "Detected RB-Y1 model='%s' version='%s' (raw_name='%s', raw_version='%s')",
            model,
            version,
            raw_name,
            raw_version,
        )
        return model, version
    finally:
        try:
            robot.disconnect()
        except Exception:
            pass


def resolve_model_version(
    model_arg: str,
    version_arg: str,
    address: str,
    connect_timeout_sec: float = 3.0,
) -> Tuple[str, str]:
    """Resolve the effective ``(model, version)`` from config + an optional probe.

    ``model_arg`` / ``version_arg`` are the configured values (``"auto"`` to
    detect via the SDK). The robot is probed only when either is ``"auto"``.
    A probed version outside :data:`SUPPORTED_VERSIONS` for the model logs a
    warning and falls back to :data:`DEFAULT_VERSION`. The ``"ub"`` model has no
    version-specific behaviour, so its version resolves to ``""``.
    """
    model_arg = (model_arg or "").strip().lower()
    version_arg = (version_arg or "").strip().lower()

    if model_arg not in VALID_MODELS:
        raise ValueError(f"model must be one of {VALID_MODELS}, got '{model_arg}'.")

    probed_model: Optional[str] = None
    probed_version: Optional[str] = None
    if model_arg == "auto" or version_arg == "auto":
        probed_model, probed_version = detect_model_and_version(
            address, connect_timeout_sec
        )

    model = probed_model if model_arg == "auto" else model_arg

    if model == "ub":
        version = ""  # no version-specific ready pose for the upper-body model
    elif version_arg == "auto":
        supported = SUPPORTED_VERSIONS.get(model, ())
        if probed_version is not None and probed_version in supported:
            version = probed_version
        else:
            if probed_version is not None:
                logger.warning(
                    "Probed version '%s' is not in the supported set %s for "
                    "model '%s'; falling back to default '%s'.",
                    probed_version,
                    supported,
                    model,
                    DEFAULT_VERSION,
                )
            version = DEFAULT_VERSION
    else:
        version = version_arg

    return model, version
