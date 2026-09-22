"""Hardware-free tests for bounded, fresh RB state delivery."""

from __future__ import annotations

import importlib.util
import importlib
import multiprocessing as mp
import queue
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).parents[1] / "lerobot_robot_rb" / "cobot.py"
)
SPEC = importlib.util.spec_from_file_location(
    "_state_freshness_cobot",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
cobot_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cobot_module
SPEC.loader.exec_module(cobot_module)

Cobot = cobot_module.Cobot
_ProducerError = cobot_module._ProducerError
_ReceivedState = cobot_module._ReceivedState
systemSTAT = cobot_module.systemSTAT
_STATE_PACKET = bytes((0x24, 0x40, 0x02, 0x03)) + struct.pack(
    cobot_module._STATE_STRUCT_FORMAT,
    *(
        0.0 if code == "f" else 0
        for code in cobot_module._STATE_STRUCT_FORMAT
    ),
)


class _SocketPlaceholder:
    pass


class _FailingDataSocket:
    def sendall(self, data: bytes) -> None:
        raise OSError("fake data socket failure")


class _OnePacketDataSocket:
    def __init__(self, stop_event: mp.synchronize.Event) -> None:
        self._stop_event = stop_event

    def sendall(self, data: bytes) -> None:
        assert data == b"reqdata"

    def recv(self, size: int) -> bytes:
        self._stop_event.set()
        return _STATE_PACKET[:size]


def _connected_cobot() -> Cobot:
    cobot = Cobot("127.0.0.1", socket_timeout_s=0.1)
    cobot.cmd_connect = 0
    cobot.data_connect = 0
    cobot.CMDSock = _SocketPlaceholder()
    cobot.DATASock = _SocketPlaceholder()
    return cobot


def _received(marker: float, age_s: float = 0.0) -> _ReceivedState:
    return _ReceivedState(
        systemSTAT(time=marker),
        time.monotonic() - age_s,
    )


def test_put_latest_replaces_full_slot_without_blocking() -> None:
    state_queue: queue.Queue[_ReceivedState] = queue.Queue(maxsize=1)

    Cobot._put_latest(state_queue, _received(1.0))
    Cobot._put_latest(state_queue, _received(2.0))

    assert state_queue.get_nowait().state.time == 2.0


def test_get_latest_state_returns_newest_available_packet() -> None:
    cobot = _connected_cobot()
    cobot.reqdata_queue = queue.Queue(maxsize=1)
    cobot.reqdata_queue.put_nowait(_received(1.0))
    Cobot._put_latest(cobot.reqdata_queue, _received(2.0))

    assert cobot.GetLatestState(timeout_s=0.1).time == 2.0


def test_get_latest_state_has_a_real_finite_timeout() -> None:
    cobot = _connected_cobot()
    cobot.reqdata_queue = queue.Queue(maxsize=1)
    started = time.monotonic()

    with pytest.raises(TimeoutError, match="No fresh RB state"):
        cobot.GetLatestState(timeout_s=0.05)

    elapsed = time.monotonic() - started
    assert 0.04 <= elapsed < 0.25


def test_none_timeout_uses_configured_finite_socket_timeout() -> None:
    cobot = _connected_cobot()
    cobot.reqdata_queue = queue.Queue(maxsize=1)
    started = time.monotonic()

    with pytest.raises(TimeoutError, match="No fresh RB state"):
        cobot.GetLatestState(timeout_s=None)

    elapsed = time.monotonic() - started
    assert 0.08 <= elapsed < 0.3


def test_cached_sample_age_is_not_reset_by_repeated_reads() -> None:
    cobot = _connected_cobot()
    cobot.reqdata_queue = queue.Queue(maxsize=1)
    cobot.reqdata_queue.put_nowait(_received(7.0))

    first = cobot.GetLatestState(timeout_s=0.05)
    second = cobot.GetLatestState(timeout_s=0.05)
    assert first is second

    time.sleep(0.06)
    with pytest.raises(TimeoutError, match="Latest RB state is stale"):
        cobot.GetLatestState(timeout_s=0.05)


def test_wait_for_first_state_waits_and_preserves_sample_for_reader() -> None:
    cobot = _connected_cobot()
    cobot.reqdata_queue = queue.Queue(maxsize=1)

    def publish() -> None:
        time.sleep(0.03)
        cobot.reqdata_queue.put_nowait(_received(3.0))

    publisher = threading.Thread(target=publish)
    publisher.start()
    assert cobot.wait_for_first_state(timeout_s=0.2)
    publisher.join()

    assert cobot.GetLatestState(timeout_s=0.2).time == 3.0


def test_wait_for_first_state_returns_false_after_timeout() -> None:
    cobot = _connected_cobot()
    cobot.reqdata_queue = queue.Queue(maxsize=1)
    started = time.monotonic()

    assert not cobot.wait_for_first_state(timeout_s=0.05)

    elapsed = time.monotonic() - started
    assert 0.04 <= elapsed < 0.25


def test_busy_producer_cannot_cause_an_unbounded_drain() -> None:
    cobot = _connected_cobot()
    cobot.reqdata_queue = queue.Queue(maxsize=1)
    stop = threading.Event()
    published = threading.Event()

    def publish_continuously() -> None:
        marker = 0.0
        while not stop.is_set():
            marker += 1.0
            Cobot._put_latest(
                cobot.reqdata_queue,
                _received(marker),
            )
            published.set()

    publisher = threading.Thread(target=publish_continuously)
    publisher.start()
    assert published.wait(timeout=0.2)
    started = time.monotonic()

    try:
        state = cobot.GetLatestState(timeout_s=0.1)
    finally:
        stop.set()
        publisher.join(timeout=1.0)

    assert state.time > 0.0
    assert time.monotonic() - started < 0.2


def test_producer_error_is_propagated_across_process() -> None:
    if mp.get_start_method() != "fork":
        pytest.skip("fake inherited socket test requires fork")

    cobot = _connected_cobot()
    cobot.DATASock = _FailingDataSocket()
    cobot._stop_event.clear()
    process = mp.Process(
        target=cobot._read_data_loop,
        args=(cobot.reqdata_queue, cobot._data_error_queue),
    )
    cobot._data_process = process
    process.start()

    try:
        with pytest.raises(RuntimeError, match="data receiver failed") as exc:
            cobot.GetLatestState(timeout_s=0.5)
        assert isinstance(exc.value.__cause__, RuntimeError)
        assert "fake data socket failure" in str(exc.value.__cause__)
    finally:
        process.join(timeout=1.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)


def test_valid_packet_is_timestamped_and_delivered_across_process() -> None:
    if mp.get_start_method() != "fork":
        pytest.skip("fake inherited socket test requires fork")

    cobot = _connected_cobot()
    cobot._stop_event.clear()
    cobot.DATASock = _OnePacketDataSocket(cobot._stop_event)
    process = mp.Process(
        target=cobot._read_data_loop,
        args=(cobot.reqdata_queue, cobot._data_error_queue),
    )
    cobot._data_process = process
    process.start()

    try:
        state = cobot.GetLatestState(timeout_s=0.5)
        assert state.time == 0.0
        assert cobot._latest_received_state is not None
        assert (
            time.monotonic()
            - cobot._latest_received_state.received_monotonic_s
            < 0.5
        )
    finally:
        process.join(timeout=1.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)


def test_serialized_producer_error_is_exposed_by_property() -> None:
    cobot = _connected_cobot()
    cobot._data_error_queue = queue.Queue(maxsize=1)
    cobot._data_error_queue.put_nowait(
        _ProducerError("ValueError", "bad fake packet")
    )

    assert "ValueError: bad fake packet" in str(cobot.data_error)


@pytest.mark.parametrize("method", ["spawn", "fork", "forkserver"])
def test_receiver_supports_process_start_methods(monkeypatch, method):
    if method not in mp.get_all_start_methods():
        pytest.skip(f"{method} is unavailable")
    monkeypatch.syspath_prepend(str(MODULE_PATH.parent))
    # A normally importable module is required for spawn unpickling.
    module = importlib.import_module("cobot")
    context = mp.get_context(method)
    parent, child = socket.socketpair()
    parent.settimeout(3)
    child.settimeout(0.2)
    stop = context.Event()
    states, errors = context.Queue(maxsize=1), context.Queue(maxsize=1)
    process = context.Process(target=module.Cobot._receive_data,
                              args=(child, stop, 0.01, states, errors))
    process.start()
    child.close()
    try:
        assert parent.recv(7) == b"reqdata"
        parent.sendall(_STATE_PACKET)
        received = states.get(timeout=3)
        assert received.state.time == 0.0
        assert time.monotonic() - received.received_monotonic_s < 1
    finally:
        stop.set()
        parent.close()
        process.join(timeout=3)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1)
        states.close()
        errors.close()
    assert process.exitcode == 0
