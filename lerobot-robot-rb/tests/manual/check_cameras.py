"""Open every configured camera at once and measure the observation cost.

No robot and no LeRobot recording involved: this exercises exactly the
cameras that RbCobot would open, so it isolates USB bandwidth and mode
negotiation problems before a recording session depends on them.

    python tests/manual/check_cameras.py [seconds]

Checks:
  * the mode each camera actually negotiated matches the config, because
    RbCobot._cameras_ft declares the dataset shape from the CONFIG and
    RealSense may silently fall back to the nearest supported mode
  * async_read() never times out over the run
  * reads PIPELINE rather than serialize: because every camera streams in
    its own thread, waiting for one camera's next frame means the others'
    have already arrived, so N cameras should cost about one frame period,
    not N of them
"""

import sys
import time

import numpy as np

from lerobot.cameras import make_cameras_from_configs

from lerobot_robot_rb.config_rb import _default_cameras


BUDGET_MS = 1000.0 / 30.0


def main() -> int:
    duration_s = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0

    configs = _default_cameras()
    cameras = make_cameras_from_configs(configs)

    print(f"설정된 카메라 {len(cameras)}대: {', '.join(cameras)}")

    serials = [c.serial_number_or_name for c in configs.values()]
    if len(set(serials)) != len(serials):
        print("!! 같은 시리얼이 두 번 쓰였습니다. 두 번째 카메라는 열리지 않습니다.")
        return 1

    for name, camera in cameras.items():
        print(f"  여는 중: {name} ...", flush=True)
        camera.connect()

    print()
    ok = True

    print("협상된 모드 (config 와 달라지면 데이터셋 shape 가 어긋납니다):")
    for name, camera in cameras.items():
        want = configs[name]
        got = (camera.width, camera.height, camera.fps)
        expect = (want.width, want.height, want.fps)
        match = got == expect
        ok &= match
        print(
            f"  {name:<8} config={expect}  실제={got}  "
            f"{'OK' if match else '<-- 불일치!'}"
        )

    print()
    print(f"{duration_s:.0f}초 동안 전 카메라 동시 읽기 ...")

    durations = []
    failures = {name: 0 for name in cameras}
    deadline = time.perf_counter() + duration_s

    while time.perf_counter() < deadline:
        start = time.perf_counter()

        for name, camera in cameras.items():
            try:
                frame = camera.async_read()
                assert isinstance(frame, np.ndarray)
            except Exception as exc:
                failures[name] += 1
                if failures[name] == 1:
                    print(f"  {name}: {type(exc).__name__}: {exc}")

        durations.append((time.perf_counter() - start) * 1e3)

    for camera in cameras.values():
        camera.disconnect()

    times = np.asarray(durations)
    over = int(np.count_nonzero(times > BUDGET_MS))

    print()
    print(f"관측 {len(times)}회")
    print(
        f"  1회 소요: 중앙값 {np.median(times):6.1f} ms | "
        f"p95 {np.percentile(times, 95):6.1f} ms | 최대 {times.max():6.1f} ms"
    )
    print(
        f"  30 Hz 예산({BUDGET_MS:.1f} ms) 초과: {over}회 "
        f"({100.0 * over / len(times):.1f}%)"
    )
    print(f"  실효 속도: {1000.0 / np.median(times):.1f} Hz")
    print(
        f"  카메라 주기 대비: {np.median(times) / BUDGET_MS:.2f}x "
        f"(1.0 = 읽기가 파이프라인됨, {len(cameras)}.0 = 순차로 합산됨)"
    )

    for name, count in failures.items():
        print(f"  {name} 읽기 실패: {count}회")
        ok &= count == 0

    # async_read blocks until a NEW frame arrives, so one round always costs
    # about one frame period even with a single camera. What matters is that
    # adding cameras does not multiply that cost.
    pipelined = np.median(times) <= 1.5 * BUDGET_MS
    if not pipelined:
        print(
            "  !! 읽기가 파이프라인되지 않고 합산되고 있습니다. "
            "USB 대역폭이나 카메라 FPS 설정을 확인하세요."
        )
    ok &= bool(pipelined)

    print()
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
