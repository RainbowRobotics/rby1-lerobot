"""Contract with the real ``isaaccapture`` package (skipped where it is not installed, e.g. CI).

Checks that the names, signatures and index layouts the Isaac teleop code
relies on exist in the installed release. No CloudXR runtime, EULA, GPU or
headset is needed.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

pytest.importorskip("isaaccapture")


def test_used_names_import():
    from isaaccapture.cloudxr import CloudXRLauncher  # noqa: F401
    from isaaccapture.cloudxr import runtime
    from isaaccapture.logging_config import set_propagate_to_root  # noqa: F401
    from isaaccapture.oxr import OpenXRSessionHandles  # noqa: F401
    from isaaccapture.retargeting_engine.interface.execution_events import (  # noqa: F401
        ExecutionEvents,
        ExecutionState,
    )
    from isaaccapture.teleop_session_manager import (  # noqa: F401
        TeleopSession,
        TeleopSessionConfig,
        get_required_oxr_extensions_from_pipeline,
    )

    # base.cloudxr_eula_marker() mirrors this path to check before the launcher prompts.
    from lerobot_teleoperator_rby1.isaac_teleop import base

    assert Path(runtime.eula_marker("/x/run")) == base.cloudxr_eula_marker("/x")


def test_base_module_binds_the_real_package():
    from lerobot_teleoperator_rby1.isaac_teleop import base

    assert base._isaaccapture_available
    assert base.CloudXRLauncher is not None and base.TeleopSession is not None
    import logging

    assert logging.getLogger("isaaccapture").propagate is False  # no duplicated console lines


def test_launcher_and_session_signatures():
    from isaaccapture.cloudxr import CloudXRLauncher
    from isaaccapture.teleop_session_manager import TeleopSession, TeleopSessionConfig

    launcher = inspect.signature(CloudXRLauncher.__init__).parameters
    for name in ("install_dir", "env_config", "device_profile", "accept_eula", "run_embedded"):
        assert name in launcher, name
    assert launcher["run_embedded"].default is False
    assert callable(CloudXRLauncher.stop)

    session_config = inspect.signature(TeleopSessionConfig).parameters
    for name in ("app_name", "pipeline", "oxr_handles"):
        assert name in session_config, name
    assert callable(TeleopSession.get_device_status)
    step = inspect.signature(TeleopSession.step).parameters
    assert "execution_events" in step and "external_inputs" in step


def test_device_states():
    from isaaccapture.teleop_session_manager import DeviceState, TeleopSession

    assert "DeviceStatus" in str(inspect.signature(TeleopSession.get_device_status))
    # The status line treats exactly these states specially.
    assert {"connected", "unknown", "disconnected"} <= {s.value for s in DeviceState}


def test_index_layout_and_pipeline():
    from lerobot_teleoperator_rby1.isaac_teleop.xr_frame import build_pipeline, verify_index_layout

    layout = verify_index_layout()
    assert layout and set(layout.values()) == {"verified"}, layout
    pipeline, has_body = build_pipeline(with_body=False)
    assert pipeline is not None and has_body is False


def test_head_index_has_is_tracked():
    from lerobot_teleoperator_rby1.isaac_teleop import xr_frame

    assert xr_frame.HeadInputIndex.IS_VALID != xr_frame.HeadInputIndex.IS_TRACKED


def test_viz_api():
    import isaaccapture.viz as televiz

    assert "render_fps" in dir(televiz.FrameTimingStats)
    assert "openxr_composition" in dir(televiz.QuadLayerConfig)
    assert "xr_system_wait_seconds" in dir(televiz.VizSessionConfig)
    assert hasattr(televiz.QuadLayer, "submit") and hasattr(televiz.DisplayMode, "kXr")
