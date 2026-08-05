#!/usr/bin/env python3
"""
RB-Y1 Gripper Connection & Operation Test Script.

Tests gripper connectivity, ping, homing, read, and write step by step.
If a problem occurs, you can identify exactly which step fails.

Usage:
    python test_gripper.py
    python test_gripper.py --skip-homing        # Skip homing, read-only test
    python test_gripper.py --motor-id 0         # Test specific motor only (0=right, 1=left)
    python test_gripper.py --open-close-test     # Run open/close cycle test
"""

import argparse
import sys
import time

import numpy as np


# ── Constants (standalone test values; constants.py uses different torques) ──
GRIPPER_BAUD_RATE = 2_000_000
GRIPPER_IDS = [0, 1]           # 0 = right, 1 = left
HOMING_TORQUE = 0.5            # Nm
HOMING_STEPS = 30              # 0.1s × 30 = 3s per direction
POSITION_TORQUE = 5.0          # Nm (position mode)


def step(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def fail(msg: str) -> None:
    print(f"  ✗ {msg}")


def info(msg: str) -> None:
    print(f"    {msg}")


# ── 1. Import test ───────────────────────────────────────────────────
def test_import():
    step("1. rby1_sdk import test")
    try:
        import rby1_sdk as rby
        ok(f"rby1_sdk loaded successfully (version attr: {hasattr(rby, '__version__')})")
    except ImportError as e:
        fail(f"rby1_sdk import failed: {e}")
        print("\n  Make sure rby1_sdk is installed.")
        print("  pip install -e /home/nvidia/rby1-sdk")
        sys.exit(1)

    try:
        import rby1_sdk.upc as upc
        ok("rby1_sdk.upc loaded successfully")
        ok(f"GripperDeviceName = {upc.GripperDeviceName!r}")
    except ImportError as e:
        fail(f"rby1_sdk.upc import failed: {e}")
        sys.exit(1)

    return rby, upc


# ── 2. Port open test ────────────────────────────────────────────────
def test_open_port(rby, upc):
    step("2. Gripper port open test")

    device_name = upc.GripperDeviceName
    info(f"Device name: {device_name}")

    # Check device file existence
    import os
    dev_path = "/dev/rby1_gripper"
    if os.path.exists(dev_path):
        ok(f"{dev_path} exists")
        # Check permissions
        stat = os.stat(dev_path)
        info(f"  permissions: {oct(stat.st_mode)}")
        info(f"  owner uid: {stat.st_uid}, gid: {stat.st_gid}")
        # Check current user access
        if os.access(dev_path, os.R_OK | os.W_OK):
            ok("Read/write permission confirmed")
        else:
            fail("No read/write permission — check sudo or dialout group")
            info("  sudo usermod -aG dialout $USER")
    else:
        fail(f"{dev_path} not found")
        info("  Check USB connection and udev rules.")
        # List related devices
        info("  Current /dev/ttyUSB* devices:")
        for f in sorted(os.listdir("/dev")):
            if "ttyUSB" in f or "ttyACM" in f:
                info(f"    /dev/{f}")
        return None

    bus = rby.DynamixelBus(device_name)
    info("DynamixelBus object created")

    if bus.open_port():
        ok("Port opened successfully")
    else:
        fail("Failed to open port")
        info("  Another process may be holding the port.")
        info("  Check with: lsof /dev/rby1_gripper")
        return None

    # Set latency_timer
    try:
        upc.initialize_device(device_name)
        ok("USB latency_timer set to 1ms")
    except Exception as e:
        fail(f"initialize_device failed: {e}")
        info("  Continuing (performance may be degraded).")

    # Set baud rate
    if bus.set_baud_rate(GRIPPER_BAUD_RATE):
        ok(f"Baud rate set to {GRIPPER_BAUD_RATE:,}")
    else:
        fail(f"Failed to set baud rate: {GRIPPER_BAUD_RATE:,}")
        return None

    # Set torque constant
    bus.set_torque_constant([1.0, 1.0])
    ok("Torque constant set")

    return bus


# ── 3. Ping test ─────────────────────────────────────────────────────
def test_ping(bus, motor_ids):
    step("3. Motor ping test")
    results = {}
    for dev_id in motor_ids:
        side = "right" if dev_id == 0 else "left"
        if bus.ping(dev_id):
            ok(f"Motor ID={dev_id} ({side}) ping OK")
            results[dev_id] = True
        else:
            fail(f"Motor ID={dev_id} ({side}) no ping response")
            info("  Check motor power, cable, and ID setting.")
            results[dev_id] = False
    return results


# ── 4. Encoder read test ─────────────────────────────────────────────
def test_read_encoder(bus, motor_ids):
    step("4. Encoder read test")
    result = bus.group_fast_sync_read_encoder(motor_ids)
    if result is None:
        fail("Encoder read failed (returned None)")
        return None

    enc_map = {}
    for dev_id, enc in result:
        side = "right" if dev_id == 0 else "left"
        ok(f"Motor ID={dev_id} ({side}): encoder = {enc:.4f} rad ({np.degrees(enc):.2f} deg)")
        enc_map[dev_id] = enc
    return enc_map


# ── 5. Homing test ───────────────────────────────────────────────────
def test_homing(bus, rby, motor_ids):
    step("5. Homing test (~6 seconds)")

    def set_mode(mode):
        bus.group_sync_write_torque_enable(
            [(dev_id, rby.DynamixelBus.TorqueDisable) for dev_id in motor_ids]
        )
        bus.group_sync_write_operating_mode(
            [(dev_id, mode) for dev_id in motor_ids]
        )
        bus.group_sync_write_torque_enable(
            [(dev_id, rby.DynamixelBus.TorqueEnable) for dev_id in motor_ids]
        )

    info("Switching to CurrentControlMode...")
    set_mode(rby.DynamixelBus.CurrentControlMode)
    ok("CurrentControlMode set")

    q = np.zeros(2, dtype=np.float64)
    min_q = np.full(2, np.inf)
    max_q = np.full(2, -np.inf)

    for direction in range(2):
        dir_label = "+torque (open direction)" if direction == 0 else "-torque (close direction)"
        info(f"Direction {direction}: {dir_label}")
        torque = HOMING_TORQUE if direction == 0 else -HOMING_TORQUE
        for step_i in range(HOMING_STEPS):
            bus.group_sync_write_send_torque(
                [(dev_id, torque) for dev_id in motor_ids]
            )
            rv = bus.group_fast_sync_read_encoder(motor_ids)
            if rv is not None:
                for dev_id, enc in rv:
                    q[dev_id] = enc
            min_q = np.minimum(min_q, q)
            max_q = np.maximum(max_q, q)
            # Progress indicator
            if (step_i + 1) % 10 == 0:
                info(f"  step {step_i+1}/{HOMING_STEPS} | q={q.round(4)}")
            time.sleep(0.1)

    # Stop torque
    bus.group_sync_write_send_torque(
        [(dev_id, 0.0) for dev_id in motor_ids]
    )
    ok("Homing torque sequence complete")

    # Switch to position mode
    info("Switching to CurrentBasedPositionControlMode...")
    set_mode(rby.DynamixelBus.CurrentBasedPositionControlMode)
    bus.group_sync_write_send_torque(
        [(dev_id, POSITION_TORQUE) for dev_id in motor_ids]
    )
    ok("Position mode set")

    # Range calculation (same logic as rby1.py)
    # Right (ID=0): high encoder = CLOSED, low = OPEN
    # Left  (ID=1): low encoder  = CLOSED, high = OPEN
    open_rad = np.array([min_q[0], max_q[1]])
    close_rad = np.array([max_q[0], min_q[1]])

    info("")
    info("── Homing results ──")
    info(f"  min_q = {min_q.round(4)}")
    info(f"  max_q = {max_q.round(4)}")
    info("")
    for i, side in enumerate(("right", "left")):
        if i not in motor_ids:
            continue
        span = abs(close_rad[i] - open_rad[i])
        info(f"  {side} (ID={i}):")
        info(f"    open_rad  = {open_rad[i]:.4f} rad ({np.degrees(open_rad[i]):.2f} deg)")
        info(f"    close_rad = {close_rad[i]:.4f} rad ({np.degrees(close_rad[i]):.2f} deg)")
        info(f"    span      = {span:.4f} rad ({np.degrees(span):.2f} deg)")
        if span < 0.01:
            fail(f"  {side} range too small — homing likely failed")
        else:
            ok(f"  {side} range OK")

    return open_rad, close_rad


# ── 6. Open/close position test ──────────────────────────────────────
def test_open_close(bus, open_rad, close_rad, motor_ids, cycles=2):
    step(f"6. Open/close cycle test ({cycles} cycles)")

    for cycle in range(cycles):
        info(f"\n--- Cycle {cycle+1}/{cycles} ---")

        # Open
        info("  -> Open (normalized=0.0)")
        target = open_rad.copy()
        bus.group_sync_write_send_position(
            [(dev_id, float(target[dev_id])) for dev_id in motor_ids]
        )
        time.sleep(1.5)
        enc = bus.group_fast_sync_read_encoder(motor_ids)
        if enc:
            for dev_id, val in enc:
                side = "right" if dev_id == 0 else "left"
                info(f"    {side}: encoder={val:.4f} (target={target[dev_id]:.4f})")

        # Close
        info("  -> Close (normalized=1.0)")
        target = close_rad.copy()
        bus.group_sync_write_send_position(
            [(dev_id, float(target[dev_id])) for dev_id in motor_ids]
        )
        time.sleep(1.5)
        enc = bus.group_fast_sync_read_encoder(motor_ids)
        if enc:
            for dev_id, val in enc:
                side = "right" if dev_id == 0 else "left"
                info(f"    {side}: encoder={val:.4f} (target={target[dev_id]:.4f})")

        # Half open
        info("  -> Half open (normalized=0.5)")
        target = open_rad + 0.5 * (close_rad - open_rad)
        bus.group_sync_write_send_position(
            [(dev_id, float(target[dev_id])) for dev_id in motor_ids]
        )
        time.sleep(1.5)
        enc = bus.group_fast_sync_read_encoder(motor_ids)
        if enc:
            for dev_id, val in enc:
                side = "right" if dev_id == 0 else "left"
                info(f"    {side}: encoder={val:.4f} (target={target[dev_id]:.4f})")

    ok("Open/close test complete")


# ── 7. Continuous read test ───────────────────────────────────────────
def test_continuous_read(bus, motor_ids, duration=3.0):
    step(f"7. Continuous encoder read test ({duration}s)")
    start = time.time()
    count = 0
    errors = 0
    while time.time() - start < duration:
        result = bus.group_fast_sync_read_encoder(motor_ids)
        if result is None:
            errors += 1
        else:
            count += 1
        time.sleep(0.01)  # ~100 Hz

    elapsed = time.time() - start
    rate = count / elapsed if elapsed > 0 else 0
    ok(f"Success: {count}, Failures: {errors}, Average: {rate:.1f} Hz")
    if errors > 0:
        fail(f"{errors} read error(s) occurred.")


# ── Cleanup ──────────────────────────────────────────────────────────
def cleanup(bus, motor_ids):
    step("Cleanup: disabling torque")
    try:
        bus.group_sync_write_torque_enable(
            [(dev_id, 0) for dev_id in motor_ids]
        )
        ok("Torque disabled")
    except Exception as e:
        fail(f"Failed to disable torque: {e}")


# ── Main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="RB-Y1 Gripper Test")
    parser.add_argument("--skip-homing", action="store_true",
                        help="Skip homing, read-only test")
    parser.add_argument("--motor-id", type=int, choices=[0, 1], default=None,
                        help="Test specific motor only (0=right, 1=left)")
    parser.add_argument("--open-close-test", action="store_true",
                        help="Run open/close cycle test")
    parser.add_argument("--cycles", type=int, default=2,
                        help="Number of open/close cycles (default: 2)")
    parser.add_argument("--continuous-read", action="store_true",
                        help="Run continuous encoder read test")
    args = parser.parse_args()

    motor_ids = [args.motor_id] if args.motor_id is not None else GRIPPER_IDS

    print("\n" + "=" * 60)
    print("  RB-Y1 Gripper Connection & Operation Test")
    print(f"  Motors under test: {['right(ID=0)' if i==0 else 'left(ID=1)' for i in motor_ids]}")
    print("=" * 60)

    # 1) Import
    rby, upc = test_import()

    # 2) Port
    bus = test_open_port(rby, upc)
    if bus is None:
        print("\nFailed to open port — aborting")
        sys.exit(1)

    try:
        # 3) Ping
        ping_results = test_ping(bus, motor_ids)
        if not all(ping_results.get(i, False) for i in motor_ids):
            print("\nSome motors failed ping — aborting")
            sys.exit(1)

        # 4) Encoder read
        test_read_encoder(bus, motor_ids)

        if args.skip_homing:
            info("\n(--skip-homing: homing skipped)")
        else:
            # 5) Homing
            result = test_homing(bus, rby, motor_ids)
            if result is None:
                print("\nHoming failed — aborting")
                sys.exit(1)
            open_rad, close_rad = result

            # 6) Open/close
            if args.open_close_test:
                test_open_close(bus, open_rad, close_rad, motor_ids, cycles=args.cycles)

        # 7) Continuous read
        if args.continuous_read:
            test_continuous_read(bus, motor_ids)

        # Final encoder read
        step("Final encoder state")
        test_read_encoder(bus, motor_ids)

    finally:
        cleanup(bus, motor_ids)

    print("\n" + "=" * 60)
    print("  Test complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
