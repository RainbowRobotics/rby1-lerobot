"""isaaccapture 1.6 behaviour: detached CloudXR service, EULA check, headset wait and status."""

from __future__ import annotations

import logging
import threading
from types import SimpleNamespace

import pytest

import isaac_fakes as fakes
from lerobot_teleoperator_rby1.isaac_teleop import base
from lerobot_teleoperator_rby1.isaac_teleop import teleop_rby1_xr as mod
from lerobot_teleoperator_rby1.isaac_teleop.config_isaac_teleop import Rby1XRConfig
from lerobot_teleoperator_rby1.isaac_teleop.teleop_rby1_xr import Rby1XR, hosted_client_url
from lerobot_teleoperator_rby1.isaac_teleop.xr_frame import XRFrame, frame_from_outputs, OUT_HEAD


class FakeLauncher:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.stopped = 0

    def stop(self):
        self.stopped += 1


def _accept_eula(install_dir):
    marker = base.cloudxr_eula_marker(str(install_dir))
    marker.parent.mkdir(parents=True)
    marker.write_text("accepted")
    return marker


def _teleop(tmp_path, launchers, session=None, **cfg):
    def factory(**kwargs):
        launchers.append(FakeLauncher(**kwargs))
        return launchers[-1]

    config = Rby1XRConfig(cloudxr_install_dir=str(tmp_path), **cfg)
    return Rby1XR(config, session_factory=lambda pipeline: session, launcher_factory=factory)


@pytest.fixture(autouse=True)
def _no_skip_env(monkeypatch):
    monkeypatch.delenv("LEROBOT_CLOUDXR_SKIP_AUTOLAUNCH", raising=False)


def test_eula_marker_path(tmp_path):
    assert base.cloudxr_eula_marker(str(tmp_path)) == tmp_path / "run" / "eula_accepted"


def test_launcher_attaches_to_detached_service_by_default(tmp_path, caplog):
    _accept_eula(tmp_path)
    launchers: list[FakeLauncher] = []
    t = _teleop(tmp_path, launchers)
    t._ensure_cloudxr_runtime()
    t._ensure_cloudxr_runtime()  # idempotent
    assert len(launchers) == 1
    kwargs = launchers[0].kwargs
    assert kwargs["run_embedded"] is False and kwargs["accept_eula"] is False
    assert kwargs["install_dir"] == str(tmp_path) and kwargs["device_profile"] == "Quest3"
    assert kwargs["env_config"] == base.default_cloudxr_env_file()

    with caplog.at_level(logging.INFO, logger=base.logger.name):
        t._stop_cloudxr_runtime()
        t._stop_cloudxr_runtime()  # idempotent
    assert launchers[0].stopped == 1
    assert "keeps running" in caplog.text and "isaaccapture.cloudxr.service stop" in caplog.text


def test_launcher_embedded_runtime_is_stopped_with_the_process(tmp_path, caplog):
    _accept_eula(tmp_path)
    launchers: list[FakeLauncher] = []
    t = _teleop(tmp_path, launchers, cloudxr_run_embedded=True)
    t._ensure_cloudxr_runtime()
    assert launchers[0].kwargs["run_embedded"] is True
    with caplog.at_level(logging.INFO, logger=base.logger.name):
        t._stop_cloudxr_runtime()
    assert launchers[0].stopped == 1
    assert "Embedded CloudXR runtime stopped" in caplog.text and "keeps running" not in caplog.text


def test_missing_eula_raises_instead_of_prompting(tmp_path):
    launchers: list[FakeLauncher] = []
    t = _teleop(tmp_path, launchers)
    with pytest.raises(RuntimeError) as err:
        t._ensure_cloudxr_runtime()
    message = str(err.value)
    assert "python -m isaaccapture.cloudxr.service start --accept-eula" in message
    assert f"--cloudxr-install-dir {tmp_path}" in message and "eula_accepted" in message
    assert launchers == []  # the launcher (which would prompt on stdin) was never built


def test_skip_env_and_config_opt_out_need_no_eula(tmp_path, monkeypatch):
    launchers: list[FakeLauncher] = []
    _teleop(tmp_path, launchers, auto_launch_cloudxr=False)._ensure_cloudxr_runtime()
    monkeypatch.setenv("LEROBOT_CLOUDXR_SKIP_AUTOLAUNCH", "1")
    _teleop(tmp_path, launchers)._ensure_cloudxr_runtime()
    assert launchers == []


class BlockingSession:
    """``TeleopSession`` stand-in whose entry waits for a headset that never comes."""

    def __init__(self):
        self.release = threading.Event()
        self.thread_name = None

    def __enter__(self):
        self.thread_name = threading.current_thread().name
        self.release.wait(5.0)
        return self

    def __exit__(self, *exc):
        pass


def test_headset_wait_timeout(tmp_path):
    session = BlockingSession()
    t = _teleop(tmp_path, [], auto_launch_cloudxr=False, xr_connect_timeout_s=0.05)
    try:
        with pytest.raises(TimeoutError, match="xr_connect_timeout_s"):
            t._enter_session(session)
        assert session.thread_name == "xr-session-enter"
    finally:
        session.release.set()


def test_headset_wait_without_timeout_enters_on_the_caller_thread(tmp_path):
    session = BlockingSession()
    session.release.set()
    t = _teleop(tmp_path, [], auto_launch_cloudxr=False)
    assert t.config.xr_connect_timeout_s == 0.0
    t._enter_session(session)
    assert session.thread_name == threading.current_thread().name


def test_headset_wait_reraises_session_errors(tmp_path):
    class Broken:
        def __enter__(self):
            raise RuntimeError("xrCreateInstance failed")

    t = _teleop(tmp_path, [], auto_launch_cloudxr=False, xr_connect_timeout_s=2.0)
    with pytest.raises(RuntimeError, match="xrCreateInstance"):
        t._enter_session(Broken())


def _status(state):
    return SimpleNamespace(status=SimpleNamespace(value=state))


@pytest.mark.parametrize(
    ("device_status", "tracked", "expected"),
    [
        (_status("connected"), True, "ok"),
        (_status("disconnected"), False, "- (disconnected)"),
        (_status("degraded"), True, "ok (degraded)"),
        (_status("unknown"), True, "ok"),  # external OpenXR handles (camera panels)
        (_status("unknown"), False, "-"),
        (None, False, "-"),
    ],
)
def test_headset_status_line(tmp_path, device_status, tracked, expected):
    t = _teleop(tmp_path, [], auto_launch_cloudxr=False)
    asked = []
    t._session = SimpleNamespace(get_device_status=lambda device: asked.append(device) or device_status)
    frame = frame_from_outputs({OUT_HEAD: fakes.head()} if tracked else {}, want_body=False)
    assert t._headset_status(frame) == expected
    assert asked == ["openxr/headset"]


def test_headset_status_falls_back_to_tracking(tmp_path):
    t = _teleop(tmp_path, [], auto_launch_cloudxr=False)
    frame = frame_from_outputs({OUT_HEAD: fakes.head()}, want_body=False)
    t._session = fakes.FakeSession()  # no get_device_status
    assert t._headset_status(frame) == "ok"

    def boom(device):
        raise RuntimeError("session closed")

    t._session = SimpleNamespace(get_device_status=boom)
    assert t._headset_status(frame) == "ok"


def test_connect_help_lists_the_client_this_host_serves(monkeypatch, capsys):
    assert hosted_client_url("10.0.0.5") == "https://10.0.0.5:48322/client/?serverIP=10.0.0.5&port=48322"
    monkeypatch.setattr(mod, "candidate_ipv4s", lambda: [("wlan0", "10.0.0.5")])
    mod.print_xr_connect_help()
    out = capsys.readouterr().out
    assert "https://10.0.0.5:48322/client/?serverIP=10.0.0.5&port=48322" in out and "wlan0" in out
    assert "https://nvidia.github.io/IsaacCapture/client/" in out
    assert "IsaacTeleop" not in out
