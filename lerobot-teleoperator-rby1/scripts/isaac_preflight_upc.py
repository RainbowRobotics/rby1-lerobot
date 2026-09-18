#!/usr/bin/env python3
"""Phase-0 preflight for running Isaac Teleop in-process on the RB-Y1 UPC.

Why: the RB-Y1 UPC is a Jetson AGX Orin (JetPack 6.0). ``isaacteleop`` selects
its experimental CloudXR runtime on Orin (T234) and works around a known
issue where *creating a Python thread after the CloudXR service started*
aborts the process (``_PyGILState_NoteThreadState``). ``lerobot-record``
creates its keyboard-listener thread after ``teleop.connect()`` and the camera
read threads lazily on the first ``get_observation()``, so we must know which
orderings survive on this host before choosing ``session_start``.

Every risky stage runs in a **subprocess** (the failure mode is SIGABRT, exit
code -6), and the parent prints a verdict table:

    GO           all stages pass            -> --teleop.session_start=connect
    GO-DEFERRED  only B / C fail            -> --teleop.session_start=first_action
                 (record: keyboard + camera threads then exist before the
                 session opens; teleoperate: run without cameras)
    NO-GO        S / A / E / F / G fail     -> run lerobot-record on an x86 GPU
                                              host (or add a UDP bridge)

Usage (on the UPC, inside the lerobot env with the [isaac] extra installed):

    python scripts/isaac_preflight_upc.py                  # all offline stages
    python scripts/isaac_preflight_upc.py --robot 192.168.30.1:50051
    python scripts/isaac_preflight_upc.py --camera-serial 4276xxxxx
    python scripts/isaac_preflight_upc.py --record-dryrun  # stage G (needs robot)
    python scripts/isaac_preflight_upc.py --wait-headset 120   # stage S_headset

Accept the CloudXR EULA once beforehand: ``python -m isaacteleop.cloudxr --accept-eula``.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

STAGES = {
    "S_session": "CloudXR + TeleopSession: 200 steps, latency, body-source .transformed()",
    "A_threads_before": "threads created BEFORE the session survive stepping",
    "B_thread_after": "a threading.Thread created AFTER the session starts",
    "C_camera_after": "a lerobot camera (lazy read thread) started AFTER the session",
    "E_rby1_after": "rby1_sdk connect + get_state() AFTER the session",
    "F_stop_then_thread": "thread creation after session exit + launcher.stop()",
    "G_record_dryrun": "real lerobot-record, 1 short episode (needs robot)",
    "S_headset": "wait for the headset; print controller / head / body samples",
}
CRITICAL = ("S_session", "A_threads_before", "E_rby1_after", "F_stop_then_thread", "G_record_dryrun")
DEFERRABLE = ("B_thread_after", "C_camera_after")


# ---------------------------------------------------------------------------
# helpers used inside stage subprocesses
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    print(f"[preflight] {msg}", flush=True)


@contextlib.contextmanager
def _session_ctx(with_body: bool):
    """Launch CloudXR + TeleopSession; ALWAYS tear both down, even on error.

    Exiting with a live OpenXR session (or a runtime killed by atexit before the
    session is destroyed) ends in Monado IPC errors and a segfault on Orin, which
    would mask the real failure of a stage.
    """
    from isaacteleop.cloudxr import CloudXRLauncher
    from isaacteleop.teleop_session_manager import TeleopSession, TeleopSessionConfig

    from lerobot_teleoperator_rby1.isaac_teleop.base import default_cloudxr_env_file
    from lerobot_teleoperator_rby1.isaac_teleop.xr_frame import build_pipeline, verify_index_layout

    launcher = None
    session = None
    try:
        if os.environ.get("LEROBOT_CLOUDXR_SKIP_AUTOLAUNCH", "") != "1":
            _log("launching CloudXR runtime …")
            launcher = CloudXRLauncher(
                install_dir=str(Path.home() / ".cloudxr"),
                env_config=default_cloudxr_env_file(),
                accept_eula=False,
            )
        _log(f"index layout check: {verify_index_layout()}")
        pipeline, body_in_graph = build_pipeline(with_body=with_body)
        _log(f"pipeline built (body rebased in-graph: {body_in_graph})")
        session = TeleopSession(TeleopSessionConfig(app_name="rby1_isaac_preflight", pipeline=pipeline))
        session.__enter__()
        _log("TeleopSession entered")
        yield session, _external_inputs(), body_in_graph
    finally:
        if session is not None:
            try:
                session.__exit__(None, None, None)
            except Exception as e:  # noqa: BLE001
                _log(f"session exit failed: {e!r}")
        if launcher is not None:
            try:
                launcher.stop()
            except Exception as e:  # noqa: BLE001
                _log(f"launcher stop failed: {e!r}")


def _external_inputs():
    from lerobot_teleoperator_rby1.isaac_teleop.config_isaac_teleop import DEFAULT_BASE_T_ANCHOR
    from lerobot_teleoperator_rby1.isaac_teleop.xr_frame import build_external_inputs

    return build_external_inputs(DEFAULT_BASE_T_ANCHOR)


def _step_n(session, n: int, ext) -> dict:
    lat = []
    misses = 0
    for _ in range(n):
        t0 = time.perf_counter()
        session.step(external_inputs=ext)
        lat.append(time.perf_counter() - t0)
        info = session.last_step_info
        if info is not None and info.worker_exception is not None:
            raise RuntimeError("worker exception") from info.worker_exception
        if info is not None and info.frame_deadline_miss:
            misses += 1
    return {
        "steps": n,
        "mean_ms": 1000 * sum(lat) / len(lat),
        "max_ms": 1000 * max(lat),
        "deadline_misses": misses,
        "step_blocks": (sum(lat) / len(lat)) > 0.004,
    }


def _heartbeat(stop: threading.Event, counter: list) -> None:
    while not stop.is_set():
        counter[0] += 1
        time.sleep(0.05)



# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------


def stage_S_session(args) -> dict:
    with _session_ctx(with_body=True) as (session, ext, body_in_graph):
        stats = _step_n(session, 200, ext)
        stats["body_in_graph"] = body_in_graph
    return stats


def stage_A_threads_before(args) -> dict:
    stop = threading.Event()
    beats = [0]
    threading.Thread(target=_heartbeat, args=(stop, beats), daemon=True).start()
    listener = None
    try:
        from pynput import keyboard  # type: ignore

        listener = keyboard.Listener(on_press=lambda k: None)
        listener.start()
    except Exception as e:  # noqa: BLE001
        _log(f"pynput listener not started ({e}); heartbeat thread only")
    try:
        with _session_ctx(with_body=False) as (session, ext, _):
            b0 = beats[0]
            stats = _step_n(session, 100, ext)
            time.sleep(1.0)
            stats["heartbeat_alive"] = beats[0] > b0
    finally:
        stop.set()
        if listener is not None:
            listener.stop()
    return stats


def stage_B_thread_after(args) -> dict:
    stop = threading.Event()
    beats = [0]
    try:
        with _session_ctx(with_body=False) as (session, ext, _):
            _step_n(session, 20, ext)
            _log("creating a Python thread AFTER the session started …")
            threading.Thread(target=_heartbeat, args=(stop, beats), daemon=True).start()
            stats = _step_n(session, 100, ext)
            time.sleep(0.5)
            stats["thread_after_alive"] = beats[0] > 0
    finally:
        stop.set()
    return stats


def _open_camera(args):
    """Return a connected lerobot camera, or None (with a reason) if none is usable."""
    if args.camera_serial:
        from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig

        try:
            cam = RealSenseCamera(
                RealSenseCameraConfig(serial_number_or_name=args.camera_serial, fps=30, width=640, height=480)
            )
            cam.connect()
            return cam, None
        except Exception as e:  # noqa: BLE001
            return None, (
                f"RealSense {args.camera_serial} unusable ({e.__class__.__name__}: {e}); "
                "check `lerobot-find-cameras realsense`"
            )
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

    # Do not force a resolution: a webcam that cannot do 640x480 must not fail
    # the *thread* test. Native resolution is fine here.
    try:
        cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=args.camera_index))
        cam.connect()
        return cam, None
    except Exception as e:  # noqa: BLE001
        return None, f"OpenCV camera {args.camera_index} unusable ({e.__class__.__name__}: {e}); pass --camera-serial"


def stage_C_camera_after(args) -> dict:
    with _session_ctx(with_body=False) as (session, ext, _):
        _step_n(session, 20, ext)
        cam, reason = _open_camera(args)
        if cam is None:
            return {"skipped": reason}
        try:
            _log("camera connected; first async_read() starts its thread …")
            frame = cam.async_read()
            stats = _step_n(session, 100, ext)
            stats["camera_frame_shape"] = list(getattr(frame, "shape", []))
        finally:
            cam.disconnect()
    return stats


def stage_E_rby1_after(args) -> dict:
    if not args.robot:
        return {"skipped": "no --robot address"}
    import rby1_sdk as rby

    with _session_ctx(with_body=False) as (session, ext, _):
        _step_n(session, 20, ext)
        robot = rby.create_robot(args.robot, args.robot_model)
        if not robot.connect():
            raise ConnectionError(f"cannot connect to {args.robot}")
        try:
            for _ in range(100):
                robot.get_state()
            stats = _step_n(session, 100, ext)
        finally:
            robot.disconnect()
    return stats


def stage_F_stop_then_thread(args) -> dict:
    with _session_ctx(with_body=False) as (session, ext, _):
        _step_n(session, 20, ext)
    stop = threading.Event()
    beats = [0]
    threading.Thread(target=_heartbeat, args=(stop, beats), daemon=True).start()
    time.sleep(0.5)
    stop.set()
    return {"thread_after_stop_alive": beats[0] > 0}


def stage_G_record_dryrun(args) -> dict:
    if not args.robot:
        return {"skipped": "no --robot address"}
    out = Path(args.record_root).expanduser()
    cmd = [
        "lerobot-record",
        "--robot.type=rby1",
        f"--robot.address={args.robot}",
        "--robot.action_mode=ee",
        "--robot.use_torso=true",
        "--robot.use_head=true",
        "--robot.use_mobile_base=false",
        "--teleop.type=rby1_isaac",
        f"--teleop.robot_address={args.robot}",
        "--teleop.use_mobile_base=false",
        f"--teleop.session_start={args.session_start}",
        "--dataset.repo_id=preflight/rby1_isaac",
        f"--dataset.root={out}",
        "--dataset.push_to_hub=false",
        "--dataset.single_task=preflight",
        "--dataset.num_episodes=1",
        "--dataset.episode_time_s=5",
        "--dataset.reset_time_s=1",
        "--dataset.fps=15",
    ]
    if args.camera_serial:
        cams = {"front": {"type": "intelrealsense", "serial_number_or_name": args.camera_serial, "fps": 30, "width": 640, "height": 480}}
        cmd.append(f"--robot.cameras={json.dumps(cams)}")
    _log("running: " + " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise RuntimeError(f"lerobot-record exited with {rc}")
    return {"dataset_root": str(out)}


def stage_S_headset(args) -> dict:
    """Wait for the headset and print, once per second, what each source delivers.

    Use this to check IOBT / body tracking: ``body`` must become present and the
    PELVIS / SPINE3 / NECK joints valid, and SPINE3 must move when you bend.
    """
    from lerobot_teleoperator_rby1.isaac_teleop.config_isaac_teleop import DEFAULT_BASE_T_ANCHOR
    from lerobot_teleoperator_rby1.isaac_teleop.teleop_rby1_xr import print_xr_connect_help
    from lerobot_teleoperator_rby1.isaac_teleop.xr_frame import BodyJointIndex, frame_from_outputs

    required = [BodyJointIndex.PELVIS, BodyJointIndex.SPINE3, BodyJointIndex.NECK]
    seen = {"right": False, "left": False, "head": False, "body": False, "body_required_valid": False}
    steps = 0
    body_valid_max = 0
    with _session_ctx(with_body=True) as (session, ext, body_in_graph):
        body_transform = None if body_in_graph else __import__("numpy").asarray(DEFAULT_BASE_T_ANCHOR, dtype=float)
        print_xr_connect_help()
        t0 = time.monotonic()
        last_print = 0.0
        while time.monotonic() - t0 < args.wait_headset:
            out = session.step(external_inputs=ext)
            steps += 1
            f = frame_from_outputs(out, want_body=True, body_transform=body_transform)
            seen["right"] |= f.right is not None
            seen["left"] |= f.left is not None
            seen["head"] |= f.head is not None
            seen["body"] |= f.body is not None
            if f.body is not None:
                body_valid_max = max(body_valid_max, int(f.body.valid.sum()))
                seen["body_required_valid"] |= all(bool(f.body.valid[i]) for i in required)
            now = time.monotonic()
            if now - last_print >= 1.0:
                last_print = now
                raw_body = out.get("full_body")
                raw_state = "absent" if raw_body is None else ("is_none" if getattr(raw_body, "is_none", False) else "present")
                r, h, b = f.right, f.head, f.body
                _log(
                    f"t={now - t0:5.1f}s right={'-' if r is None else f'pos={r.position.round(2)} sq={r.squeeze:.2f}'} "
                    f"left={'-' if f.left is None else f'sq={f.left.squeeze:.2f}'} "
                    f"head={'-' if h is None else h.position.round(2)} "
                    f"body[raw={raw_state}]={'-' if b is None else f'valid={int(b.valid.sum())}/24 req_ok={all(bool(b.valid[i]) for i in required)} spine3={b.positions[BodyJointIndex.SPINE3].round(2)}'}"
                )
            time.sleep(0.01)
    return {"seen": seen, "steps": steps, "body_valid_max": body_valid_max, "body_in_graph": body_in_graph}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def environment_report() -> dict:
    rep = {
        "python": sys.version.split()[0],
        "machine": platform.machine(),
        "device_tree_model": _read("/proc/device-tree/model"),
        "nv_tegra_release": (_read("/etc/nv_tegra_release") or "").splitlines()[:1],
        "cloudxr_install_dir": str(Path.home() / ".cloudxr"),
        "cloudxr_installed": (Path.home() / ".cloudxr").exists(),
        "port_48322_free": _port_free(48322),
        "lerobot-record": shutil.which("lerobot-record"),
    }
    for mod in ("isaacteleop", "lerobot", "rby1_sdk", "lerobot_teleoperator_rby1", "lerobot_robot_rby1"):
        try:
            m = importlib.import_module(mod)
            rep[mod] = getattr(m, "__version__", "ok")
        except Exception as e:  # noqa: BLE001
            rep[mod] = f"MISSING ({e.__class__.__name__})"
    try:
        from lerobot_teleoperator_rby1.isaac_teleop.teleop_rby1_xr import candidate_ipv4s

        rep["candidate_ipv4s"] = candidate_ipv4s()
    except Exception:  # noqa: BLE001
        pass
    return rep


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text(errors="replace").strip("\x00\n")
    except OSError:
        return None


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


LOG_DIR = Path.home() / "preflight_logs"


def _first_error_summary(text: str) -> str:
    """First exception line(s) of the log — the root cause, not the shutdown noise."""
    lines = text.splitlines()
    summary: list[str] = []
    for i, line in enumerate(lines):
        if line.startswith("Traceback (most recent call last)"):
            for j in range(i + 1, min(i + 80, len(lines))):
                if lines[j] and not lines[j].startswith((" ", "\t", "Traceback")):
                    summary.append(lines[j])
                    break
            if len(summary) >= 3:
                break
    return "\n".join(f"  ! {x}" for x in summary)


def run_stage_subprocess(name: str, argv: list[str]) -> tuple[str, dict | str]:
    cmd = [sys.executable, __file__, "--stage", name, *argv]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{name}.log"
    log_path.write_text(f"$ {' '.join(cmd)}\n\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")
    if proc.returncode == 0:
        for line in reversed(proc.stdout.splitlines()):
            if line.startswith("RESULT "):
                return "PASS", json.loads(line[len("RESULT "):])
        return "PASS", {}
    kind = "ABORT(SIGABRT)" if proc.returncode == -6 else f"FAIL(rc={proc.returncode})"
    combined = proc.stdout + proc.stderr
    tail = "\n".join(combined.splitlines()[-15:])
    detail = f"full log: {log_path}\n{_first_error_summary(combined)}\n--- tail ---\n{tail}"
    return kind, detail


def verdict(results: dict[str, tuple[str, object]]) -> str:
    def failed(n):
        return n in results and results[n][0] != "PASS" and not (
            isinstance(results[n][1], dict) and "skipped" in results[n][1]
        )

    if any(failed(n) for n in CRITICAL):
        return "NO-GO — run lerobot-record on an x86 NVIDIA host (or add a UDP bridge)."
    if any(failed(n) for n in DEFERRABLE):
        return "GO-DEFERRED — use --teleop.session_start=first_action (record OK; teleoperate without cameras)."
    return "GO — use --teleop.session_start=connect (default)."


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=STAGES, help="(internal) run one stage in this process")
    p.add_argument("--only", nargs="*", choices=STAGES, help="run only these stages")
    p.add_argument("--robot", default="", help="RB-Y1 gRPC address for stages E and G")
    p.add_argument("--robot-model", default="m")
    p.add_argument("--camera-serial", default="", help="RealSense serial for stages C and G")
    p.add_argument("--camera-index", type=int, default=0, help="OpenCV index for stage C when no serial")
    p.add_argument("--record-dryrun", action="store_true", help="include stage G")
    p.add_argument("--record-root", default="~/preflight_datasets")
    p.add_argument("--session-start", default="connect", choices=("connect", "first_action"))
    p.add_argument("--wait-headset", type=float, default=0.0, help="seconds; >0 includes stage S_headset")
    args = p.parse_args()

    if args.stage:
        result = globals()[f"stage_{args.stage}"](args)
        print("RESULT " + json.dumps(result), flush=True)
        return 0

    print("== Environment ==")
    for k, v in environment_report().items():
        print(f"  {k:<28} {v}")

    stages = list(args.only) if args.only else ["S_session", "A_threads_before", "B_thread_after", "C_camera_after", "E_rby1_after", "F_stop_then_thread"]
    if args.record_dryrun and "G_record_dryrun" not in stages:
        stages.append("G_record_dryrun")
    if args.wait_headset > 0 and "S_headset" not in stages:
        stages.append("S_headset")
    if not args.camera_serial and "C_camera_after" in stages and not args.only:
        print("  (stage C uses OpenCV index 0; pass --camera-serial for RealSense)")

    passthrough = [
        f"--robot={args.robot}", f"--robot-model={args.robot_model}",
        f"--camera-serial={args.camera_serial}", f"--camera-index={args.camera_index}",
        f"--record-root={args.record_root}", f"--session-start={args.session_start}",
        f"--wait-headset={args.wait_headset}",
    ]
    results: dict[str, tuple[str, object]] = {}
    for name in stages:
        print(f"\n== {name}: {STAGES[name]} ==")
        if name != "S_session" and results.get("S_session", ("PASS",))[0] != "PASS":
            print("  -> SKIPPED (S_session failed; fix the session first, logs in ~/preflight_logs)")
            results[name] = ("SKIPPED", {"skipped": "S_session failed"})
            continue
        status, detail = run_stage_subprocess(name, passthrough)
        results[name] = (status, detail)
        print(f"  -> {status}")
        print("     " + (json.dumps(detail) if isinstance(detail, dict) else detail.replace("\n", "\n     ")))

    print("\n== Verdict ==")
    for name, (status, _) in results.items():
        print(f"  {name:<20} {status}")
    print("  " + verdict(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
