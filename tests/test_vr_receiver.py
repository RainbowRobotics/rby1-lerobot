"""VRReceiver: stale data, sender filtering, rising-edge buttons (local UDP only)."""

from __future__ import annotations

import json
import socket
import time

import pytest
from lerobot_teleoperator_rby1.vr_receiver import VRReceiver


def _packet(right_a: bool = False) -> bytes:
    hand = {"position": [0, 0, 0], "rotation": [0, 0, 0, 1],
            "buttons": {"trigger": 0.0, "grip": 0.0, "thumbstickAxis": [0.0, 1.0],
                        "primaryButton": right_a, "secondaryButton": False}}
    return json.dumps({"hands": {"right": hand}}).encode()


@pytest.fixture
def receiver(monkeypatch):
    monkeypatch.setattr(VRReceiver, "_send_handshake", lambda self: None)
    with socket.socket() as probe:  # pick a free port
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    rx = VRReceiver("127.0.0.1", port, "127.0.0.1", 0, stale_timeout_s=0.2)
    rx.start()
    time.sleep(0.2)
    yield rx
    rx.stop()


def _send(rx: VRReceiver, data: bytes, src: str = "127.0.0.1") -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind((src, 0))
        s.sendto(data, ("127.0.0.1", rx.local_port))
    time.sleep(0.05)


def test_state_goes_stale(receiver) -> None:
    _send(receiver, _packet())
    assert receiver.get_state()["hands"]["right"]
    time.sleep(0.3)
    assert receiver.get_state() == {}


def test_ignores_other_senders(receiver) -> None:
    _send(receiver, _packet(), src="127.0.0.2")
    assert receiver.get_state() == {}


def test_held_button_is_one_event(receiver) -> None:
    for _ in range(3):
        _send(receiver, _packet(right_a=True))
    assert receiver.consume_button_events()["right_a"] is True
    _send(receiver, _packet(right_a=True))
    assert receiver.consume_button_events()["right_a"] is False
    _send(receiver, _packet(right_a=False))
    _send(receiver, _packet(right_a=True))
    assert receiver.consume_button_events()["right_a"] is True
