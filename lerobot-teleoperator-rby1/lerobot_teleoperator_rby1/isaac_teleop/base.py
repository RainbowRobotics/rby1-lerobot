"""Shared base for NVIDIA Isaac Teleop-backed LeRobot teleoperators.

Adapted from the upstream LeRobot example
``examples/isaac_teleop_to_so101/isaac_teleop/base.py`` (Apache-2.0).

:class:`IsaacTeleopTeleoperator` owns what Isaac Teleop devices share — the
CloudXR runtime and ``TeleopSession`` lifecycle, the per-step worker-health
guard, and the no-op calibration tracking devices need. A concrete device
implements :meth:`_build_pipeline` (its retargeting graph) and
:meth:`get_action` (usually via :meth:`_step`).

``isaacteleop`` is an optional dependency (``pip install -e
"lerobot-teleoperator-rby1[isaac]"``); its imports are guarded behind an
availability check so this module imports without it and constructing a
device fails fast with install instructions.
"""

from __future__ import annotations

import abc
import importlib.util
import logging
import os
from collections.abc import Mapping
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lerobot.teleoperators.teleoperator import Teleoperator

from .config_isaac_teleop import IsaacTeleopConfig

_isaacteleop_available = importlib.util.find_spec("isaacteleop") is not None

if TYPE_CHECKING or _isaacteleop_available:
    from isaacteleop.cloudxr import CloudXRLauncher
    from isaacteleop.retargeting_engine.interface import (
        ExecutionEvents,
        ExecutionState,
        GraphExecutable,
        RetargeterIO,
    )
    from isaacteleop.teleop_session_manager import TeleopSession, TeleopSessionConfig
else:  # pragma: no cover - exercised only without isaacteleop
    CloudXRLauncher = None
    ExecutionEvents = None
    ExecutionState = None
    GraphExecutable = None
    RetargeterIO = None
    TeleopSession = None
    TeleopSessionConfig = None

logger = logging.getLogger(__name__)

INSTALL_HINT = (
    "The 'isaacteleop' package is required for Isaac Teleop devices but is not "
    "installed. Install it with: pip install -e \"lerobot-teleoperator-rby1[isaac]\" "
    "and accept the CloudXR EULA once with: python -m isaacteleop.cloudxr --accept-eula"
)


def default_cloudxr_env_file() -> str:
    """Path of the packaged ``default.env`` CloudXR profile."""
    return str(files(__package__) / "default.env")


class IsaacTeleopTeleoperator(Teleoperator):
    """Abstract base for teleoperators backed by an Isaac Teleop ``TeleopSession``.

    Owns the session lifecycle and the per-step health guard; subclasses supply
    :meth:`_build_pipeline` and :meth:`get_action`.

    ``session_factory`` / ``launcher_factory`` exist for tests: they replace the
    ``TeleopSession`` and ``CloudXRLauncher`` constructors so the device can be
    driven without the XR runtime (and without ``isaacteleop`` installed).
    """

    config_class = IsaacTeleopConfig

    def __init__(
        self,
        config: IsaacTeleopConfig,
        *,
        session_factory: Any | None = None,
        launcher_factory: Any | None = None,
    ):
        if session_factory is None and not _isaacteleop_available:
            raise ImportError(INSTALL_HINT)
        super().__init__(config)
        self.config: IsaacTeleopConfig = config
        self._session: Any = None
        self._cloudxr_launcher: Any = None
        self._session_factory = session_factory
        self._launcher_factory = launcher_factory
        # Optional Televiz camera panels: when set before connect(), Televiz
        # owns the (graphics) OpenXR session and the TeleopSession attaches to it.
        self._viz: Any = None

    # ------------------------------------------------------------------
    # Pipeline construction (device override point)
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _build_pipeline(self) -> GraphExecutable:
        """Build this device's retargeting pipeline (``TeleopSessionConfig.pipeline``).

        Called once in :meth:`connect`; its output keys must match what
        :meth:`get_action` unpacks.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Lifecycle (shared)
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._session is not None

    @property
    def is_calibrated(self) -> bool:
        return True  # Tracking devices are self-calibrating.

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002
        """Auto-launch the CloudXR runtime (unless opted out) and open the session.

        The CloudXR launch blocks ~30 s and, on the first run, prompts on stdin
        for the EULA (accept once via ``python -m isaacteleop.cloudxr
        --accept-eula``). Opt out when CloudXR runs externally via
        ``config.auto_launch_cloudxr=False`` or
        ``LEROBOT_CLOUDXR_SKIP_AUTOLAUNCH=1`` (env var wins).
        """
        if self._session is not None:
            raise RuntimeError("Already connected. Call disconnect() first.")

        self._ensure_cloudxr_runtime()

        try:
            pipeline = self._build_pipeline()
            oxr_handles = None
            if self._viz is not None:
                # Televiz creates the graphics OpenXR session (blocking until the
                # headset connects); the trackers' extensions must be on that
                # XrInstance, so they are aggregated from the pipeline first.
                extensions = self._required_extensions(pipeline)
                self._viz.create_session(
                    self.config.app_name, extensions, getattr(self.config, "viz_wait_headset_s", -1)
                )
                handles = self._viz.oxr_handles()
                if handles is not None:
                    oxr_handles = self._make_oxr_handles(handles)
                self._viz.start_render_thread()
            self._session = self._make_session(pipeline, oxr_handles)
            self._session.__enter__()
            if self._viz is not None:
                self._viz.begin_rendering()
        except Exception:
            self._session = None
            if self._viz is not None:
                try:
                    self._viz.destroy()
                except Exception:
                    logger.exception("Failed to destroy the Televiz session during connect() rollback")
            try:
                self._stop_cloudxr_runtime()
            except Exception:
                logger.exception("Failed to stop CloudXR runtime during connect() rollback")
            raise
        logger.info("Isaac Teleop session started: %s", self.config.app_name)

    def _required_extensions(self, pipeline: Any) -> list[str]:
        if self._session_factory is not None:
            return []
        from isaacteleop.teleop_session_manager import get_required_oxr_extensions_from_pipeline

        return list(get_required_oxr_extensions_from_pipeline(pipeline))

    @staticmethod
    def _make_oxr_handles(handles: Any) -> Any:
        try:
            from isaacteleop.oxr import OpenXRSessionHandles
        except ImportError:  # test doubles
            return handles
        return OpenXRSessionHandles(*handles)

    def _make_session(self, pipeline: Any, oxr_handles: Any = None) -> Any:
        if self._session_factory is not None:
            return self._session_factory(pipeline)
        if oxr_handles is not None:
            session_config = TeleopSessionConfig(
                app_name=self.config.app_name, pipeline=pipeline, oxr_handles=oxr_handles
            )
        else:
            session_config = TeleopSessionConfig(app_name=self.config.app_name, pipeline=pipeline)
        return TeleopSession(session_config)

    def disconnect(self) -> None:
        try:
            # Order with Televiz: stop the frame loop -> detach the trackers
            # (session exit) -> destroy the viz session -> stop CloudXR.
            if self._viz is not None:
                self._viz.stop_rendering()
            if self._session is not None:
                # Null the handle BEFORE __exit__: even a failed session teardown
                # must not wedge the device as is_connected.
                session = self._session
                self._session = None
                session.__exit__(None, None, None)
                logger.info("Isaac Teleop session ended")
        finally:
            if self._viz is not None:
                try:
                    self._viz.destroy()
                except Exception:
                    logger.exception("Failed to destroy the Televiz session")
                self._viz = None
            # Reap the CloudXR runtime even if session teardown raised; a no-op
            # when we never launched CloudXR (opt-out / externally-owned runtime).
            self._stop_cloudxr_runtime()

    # ------------------------------------------------------------------
    # CloudXR runtime (shared)
    # ------------------------------------------------------------------

    def _ensure_cloudxr_runtime(self) -> None:
        """Auto-launch the CloudXR runtime once, unless opted out.

        Idempotent (no-op once the launcher is up). ``LEROBOT_CLOUDXR_SKIP_AUTOLAUNCH``
        is checked first and wins over ``config.auto_launch_cloudxr``. Constructing
        ``CloudXRLauncher`` mutates the process env (``XR_RUNTIME_JSON`` etc.) and
        blocks until the runtime is ready or raises ``RuntimeError``.
        """
        if self._cloudxr_launcher is not None:
            return

        if os.environ.get("LEROBOT_CLOUDXR_SKIP_AUTOLAUNCH", "").strip() == "1":
            logger.info(
                "LEROBOT_CLOUDXR_SKIP_AUTOLAUNCH=1 set; skipping CloudXR auto-launch "
                "(assuming CloudXR is already running externally)"
            )
            return

        if not self.config.auto_launch_cloudxr:
            logger.info(
                "config.auto_launch_cloudxr is False; skipping CloudXR auto-launch "
                "(assuming CloudXR is already running externally)"
            )
            return

        env_file = self.config.cloudxr_env_file or default_cloudxr_env_file()
        install_dir = str(Path(self.config.cloudxr_install_dir).expanduser())
        logger.info(
            "Launching CloudXR runtime (first run may prompt for EULA and take ~30s; "
            "install_dir=%s, env=%s, profile=%s)...",
            install_dir,
            env_file,
            self.config.cloudxr_device_profile,
        )
        factory = self._launcher_factory or CloudXRLauncher
        self._cloudxr_launcher = factory(
            install_dir=install_dir,
            env_config=env_file,
            device_profile=self.config.cloudxr_device_profile,
            accept_eula=False,
        )

    def _stop_cloudxr_runtime(self) -> None:
        """Stop the auto-launched CloudXR runtime, if any.

        Clean stop nulls the handle. On ``RuntimeError`` the handle is RETAINED so
        the launcher's ``atexit`` hook owns the retry.
        """
        if self._cloudxr_launcher is None:
            return
        try:
            self._cloudxr_launcher.stop()
        except RuntimeError:
            logger.warning(
                "CloudXR runtime could not be terminated; handle retained for atexit cleanup"
            )
        else:
            self._cloudxr_launcher = None
            logger.info("CloudXR runtime stopped")

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass  # Haptic feedback not implemented.

    # ------------------------------------------------------------------
    # Stepping (shared)
    # ------------------------------------------------------------------

    def _running_events(self) -> Any:
        """Constant ``RUNNING`` ``ExecutionEvents`` (no in-graph clutch lifecycle)."""
        if ExecutionEvents is None:  # test doubles: the fake session ignores events
            return None
        return ExecutionEvents(execution_state=ExecutionState.RUNNING, reset=False)

    def _step(
        self,
        *,
        execution_events: Any | None = None,
        external_inputs: Mapping[str, Any] | None = None,
    ) -> RetargeterIO:
        """Step the session once and return the raw pipeline outputs.

        Re-raises a retargeting-worker exception and warns on a stale frame.

        Raises:
            RuntimeError: If not connected, or if the retargeting worker raised.
        """
        if self._session is None:
            raise RuntimeError("Not connected. Call connect() first.")

        result = self._session.step(
            execution_events=execution_events,
            external_inputs=external_inputs,
        )

        info = getattr(self._session, "last_step_info", None)
        if info is not None:
            if getattr(info, "worker_exception", None) is not None:
                raise RuntimeError(
                    "Isaac Teleop retargeting worker raised an exception"
                ) from info.worker_exception
            if getattr(info, "frame_deadline_miss", False):
                logger.warning(
                    "Isaac Teleop frame deadline miss (returned_age_frames=%s)",
                    getattr(info, "returned_age_frames", "?"),
                )
        return result
