"""Shared base for NVIDIA Isaac Capture-backed LeRobot teleoperators.

Adapted from the upstream LeRobot example
``examples/isaac_teleop_to_so101/isaac_teleop/base.py`` (Apache-2.0).

:class:`IsaacTeleopTeleoperator` owns what Isaac Capture devices share — the
CloudXR runtime and ``TeleopSession`` lifecycle, the per-step worker-health
guard, and the no-op calibration tracking devices need. A concrete device
implements :meth:`_build_pipeline` (its retargeting graph) and
:meth:`get_action` (usually via :meth:`_step`).

``isaaccapture`` (NVIDIA Isaac Capture, formerly Isaac Teleop) is an optional
dependency (``pip install -e "lerobot-teleoperator-rby1[isaac]"``); its
imports are guarded behind an availability check so this module imports
without it and constructing a device fails fast with install instructions.

CloudXR runtime ownership (isaaccapture >= 1.6): the runtime is a detached
host service. ``CloudXRLauncher`` attaches to a running one or starts one, and
the service keeps running after this process exits (stop it with ``python -m
isaaccapture.cloudxr.service stop``). Only ``cloudxr_run_embedded=True`` makes
this process own the runtime and stop it on disconnect.
"""

from __future__ import annotations

import abc
import importlib.util
import logging
import os
import threading
from collections.abc import Mapping
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lerobot.teleoperators.teleoperator import Teleoperator

from .config_isaac_teleop import IsaacTeleopConfig

_isaaccapture_available = importlib.util.find_spec("isaaccapture") is not None

if TYPE_CHECKING or _isaaccapture_available:
    from isaaccapture.cloudxr import CloudXRLauncher
    from isaaccapture.retargeting_engine.interface import (
        ExecutionEvents,
        ExecutionState,
        GraphExecutable,
        RetargeterIO,
    )
    from isaaccapture.teleop_session_manager import TeleopSession, TeleopSessionConfig
else:  # pragma: no cover - exercised only without isaaccapture
    CloudXRLauncher = None
    ExecutionEvents = None
    ExecutionState = None
    GraphExecutable = None
    RetargeterIO = None
    TeleopSession = None
    TeleopSessionConfig = None

logger = logging.getLogger(__name__)

CLOUDXR_SERVICE_CLI = "python -m isaaccapture.cloudxr.service"
EULA_ACCEPT_COMMAND = f"{CLOUDXR_SERVICE_CLI} start --accept-eula"

INSTALL_HINT = (
    "The 'isaaccapture' package is required for Isaac teleop devices but is not "
    "installed. Install it with: pip install -e \"lerobot-teleoperator-rby1[isaac]\" "
    f"and accept the CloudXR EULA once with: {EULA_ACCEPT_COMMAND}"
)

if _isaaccapture_available:
    # ``import isaaccapture`` installs its own console handler on the
    # ``isaaccapture`` logger and still propagates to the root logger, so with
    # lerobot's root handlers every line would be printed twice.
    try:
        from isaaccapture.logging_config import set_propagate_to_root

        set_propagate_to_root(False)
    except Exception:  # pragma: no cover - logging layout changed upstream
        logger.debug("isaaccapture.logging_config.set_propagate_to_root unavailable", exc_info=True)


def cloudxr_eula_marker(install_dir: str) -> Path:
    """Marker file the CloudXR launcher writes once the EULA is accepted."""
    return Path(install_dir).expanduser() / "run" / "eula_accepted"


def default_cloudxr_env_file() -> str:
    """Path of the packaged ``default.env`` CloudXR profile."""
    return str(files(__package__) / "default.env")


class IsaacTeleopTeleoperator(Teleoperator):
    """Abstract base for teleoperators backed by an Isaac Capture ``TeleopSession``.

    Owns the session lifecycle and the per-step health guard; subclasses supply
    :meth:`_build_pipeline` and :meth:`get_action`.

    ``session_factory`` / ``launcher_factory`` exist for tests: they replace the
    ``TeleopSession`` and ``CloudXRLauncher`` constructors so the device can be
    driven without the XR runtime (and without ``isaaccapture`` installed).
    """

    config_class = IsaacTeleopConfig

    def __init__(
        self,
        config: IsaacTeleopConfig,
        *,
        session_factory: Any | None = None,
        launcher_factory: Any | None = None,
    ):
        if session_factory is None and not _isaaccapture_available:
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
        """Attach to (or start) the CloudXR runtime, unless opted out, and open the session.

        Starting the runtime blocks ~30 s. The EULA must have been accepted
        once (``python -m isaaccapture.cloudxr.service start --accept-eula``);
        without it this raises instead of prompting on stdin. Opt out of the
        launcher via ``config.auto_launch_cloudxr=False`` or
        ``LEROBOT_CLOUDXR_SKIP_AUTOLAUNCH=1`` (env var wins).

        Opening the session blocks until a headset connects (isaaccapture
        >= 1.6); ``config.xr_connect_timeout_s`` bounds that wait.
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
            self._enter_session(self._session)
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
        logger.info("Isaac Capture session started: %s", self.config.app_name)

    def _enter_session(self, session: Any) -> None:
        """Enter the ``TeleopSession``; it blocks until a headset is connected.

        With ``xr_connect_timeout_s > 0`` the entry runs on a worker thread so
        the wait can be bounded; a timed-out entry cannot be cancelled, so the
        (daemon) thread is left behind and the caller must give up on this
        process' session.
        """
        timeout = float(getattr(self.config, "xr_connect_timeout_s", 0.0) or 0.0)
        if timeout <= 0.0:
            logger.info("Waiting for the headset to connect (no timeout; Ctrl-C to abort) …")
            session.__enter__()
            return

        logger.info("Waiting up to %.0f s for the headset to connect …", timeout)
        error: list[BaseException] = []
        done = threading.Event()

        def enter() -> None:
            try:
                session.__enter__()
            except BaseException as e:  # noqa: BLE001 - re-raised on the caller thread
                error.append(e)
            finally:
                done.set()

        threading.Thread(target=enter, name="xr-session-enter", daemon=True).start()
        if not done.wait(timeout):
            raise TimeoutError(
                f"No headset connected within {timeout:.0f} s (xr_connect_timeout_s). "
                "Open the CloudXR web client in the headset and press Connect, then run again."
            )
        if error:
            raise error[0]

    def _required_extensions(self, pipeline: Any) -> list[str]:
        if self._session_factory is not None:
            return []
        from isaaccapture.teleop_session_manager import get_required_oxr_extensions_from_pipeline

        return list(get_required_oxr_extensions_from_pipeline(pipeline))

    @staticmethod
    def _make_oxr_handles(handles: Any) -> Any:
        try:
            from isaaccapture.oxr import OpenXRSessionHandles
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
                logger.info("Isaac Capture session ended")
        finally:
            if self._viz is not None:
                try:
                    self._viz.destroy()
                except Exception:
                    logger.exception("Failed to destroy the Televiz session")
                self._viz = None
            # Release the launcher even if session teardown raised. Only an
            # embedded runtime is stopped; the detached service keeps running.
            self._stop_cloudxr_runtime()

    # ------------------------------------------------------------------
    # CloudXR runtime (shared)
    # ------------------------------------------------------------------

    def _ensure_cloudxr_runtime(self) -> None:
        """Attach to the CloudXR runtime (starting it when none runs), unless opted out.

        Idempotent (no-op once the launcher is up). ``LEROBOT_CLOUDXR_SKIP_AUTOLAUNCH``
        is checked first and wins over ``config.auto_launch_cloudxr``. Constructing
        ``CloudXRLauncher`` mutates the process env (``XR_RUNTIME_JSON`` etc.) and
        blocks until the runtime is ready or raises ``RuntimeError``.

        By default the runtime is the detached host service: it is started when
        missing and is left running afterwards. ``config.cloudxr_run_embedded``
        runs it inside this process instead (refused while a service is running).
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
        embedded = bool(getattr(self.config, "cloudxr_run_embedded", False))
        self._check_cloudxr_eula(install_dir)
        logger.info(
            "CloudXR runtime: %s (install_dir=%s, env=%s, profile=%s). Starting it takes ~30 s; "
            "a running service keeps its own env/profile (to change them: `%s stop`, then run again).",
            "embedded in this process" if embedded else "detached service (attach, or start when none runs)",
            install_dir,
            env_file,
            self.config.cloudxr_device_profile,
            CLOUDXR_SERVICE_CLI,
        )
        factory = self._launcher_factory or CloudXRLauncher
        self._cloudxr_launcher = factory(
            install_dir=install_dir,
            env_config=env_file,
            device_profile=self.config.cloudxr_device_profile,
            accept_eula=False,
            run_embedded=embedded,
        )

    @staticmethod
    def _check_cloudxr_eula(install_dir: str) -> None:
        """Fail with instructions instead of letting the launcher prompt on stdin."""
        marker = cloudxr_eula_marker(install_dir)
        if marker.is_file():
            return
        default_dir = install_dir == str(Path("~/.cloudxr").expanduser())
        raise RuntimeError(
            f"The NVIDIA CloudXR EULA has not been accepted on this host ({marker} is missing). "
            f"Accept it once with: {EULA_ACCEPT_COMMAND}"
            + ("" if default_dir else f" --cloudxr-install-dir {install_dir}")
            + f"  (this also starts the CloudXR service; stop it with `{CLOUDXR_SERVICE_CLI} stop`)."
        )

    def _stop_cloudxr_runtime(self) -> None:
        """Release the launcher; stops the runtime only when it is embedded.

        ``CloudXRLauncher.stop()`` is a no-op when attached to the detached
        service, which outlives this process by design.
        """
        if self._cloudxr_launcher is None:
            return
        launcher = self._cloudxr_launcher
        self._cloudxr_launcher = None
        embedded = bool(getattr(self.config, "cloudxr_run_embedded", False))
        try:
            launcher.stop()
        except RuntimeError:
            logger.warning("CloudXR runtime could not be stopped cleanly", exc_info=True)
            return
        if embedded:
            logger.info("Embedded CloudXR runtime stopped")
        else:
            logger.info(
                "The CloudXR service keeps running for the next session. Stop it with: %s stop",
                CLOUDXR_SERVICE_CLI,
            )

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
                    "Isaac Capture retargeting worker raised an exception"
                ) from info.worker_exception
            if getattr(info, "frame_deadline_miss", False):
                logger.warning(
                    "Isaac Capture frame deadline miss (returned_age_frames=%s)",
                    getattr(info, "returned_age_frames", "?"),
                )
        return result
