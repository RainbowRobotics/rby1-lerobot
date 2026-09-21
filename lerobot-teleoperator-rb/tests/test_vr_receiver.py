import json
import socket
import time

import numpy as np
import pytest

from lerobot_teleoperator_rb.frame_transforms import (
    quest_pose_to_rb,
)
from lerobot_teleoperator_rb.vr_receiver import (
    VRReceiver,
)


def make_payload(
    *,
    primary=False,
    secondary=False,
    trigger=0.25,
    grip=0.75,
):
    return {
        "timestamp": 123.4,
        "head": {
            "position": [0.0, 0.0, 1.6],
            "rotation": [0.0, 0.0, 0.0, 1.0],
        },
        "hands": {
            "right": {
                "position": [0.4, -0.2, 1.2],
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "buttons": {
                    "primaryButton": primary,
                    "secondaryButton": secondary,
                    "trigger": trigger,
                    "grip": grip,
                },
            }
        },
    }


def free_udp_port():
    with socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM,
    ) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def send_payload(port, payload):
    with socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM,
    ) as sock:
        sock.sendto(
            json.dumps(payload).encode("utf-8"),
            ("127.0.0.1", port),
        )


def wait_for_packet_count(receiver, count, timeout_s=2.0):
    deadline = time.monotonic() + timeout_s

    while time.monotonic() < deadline:
        if receiver.get_state().packet_count >= count:
            return

        time.sleep(0.005)

    raise AssertionError(
        "Expected "
        f"{count} packets, got "
        f"{receiver.get_state().packet_count}."
    )


@pytest.fixture
def receiver():
    instance = VRReceiver(
        local_ip="127.0.0.1",
        local_port=free_udp_port(),
        meta_quest_ip=None,
        send_handshake=False,
        tracking_timeout_s=0.25,
    )
    instance.start()

    try:
        yield instance
    finally:
        instance.stop()


def test_packet_is_parsed_into_the_rb_frame(receiver):
    send_payload(
        receiver.local_port,
        make_payload(),
    )
    wait_for_packet_count(receiver, 1)

    state = receiver.get_state()

    expected_pose = quest_pose_to_rb(
        [0.4, -0.2, 1.2],
        [0.0, 0.0, 0.0, 1.0],
    )

    assert state.head.tracked
    assert state.right.tracked
    assert not state.left.tracked
    assert np.allclose(
        state.right.pose_rb,
        expected_pose,
    )
    assert state.right.buttons.trigger == 0.25
    assert state.right.buttons.grip == 0.75
    assert state.source_ip == "127.0.0.1"


def test_analog_values_are_clamped(receiver):
    send_payload(
        receiver.local_port,
        make_payload(trigger=2.0, grip=-1.0),
    )
    wait_for_packet_count(receiver, 1)

    state = receiver.get_state()

    assert state.right.buttons.trigger == 1.0
    assert state.right.buttons.grip == 0.0


def test_receiver_keeps_latest_state(receiver):
    send_payload(
        receiver.local_port,
        make_payload(trigger=0.1),
    )
    wait_for_packet_count(receiver, 1)

    send_payload(
        receiver.local_port,
        make_payload(trigger=0.9),
    )
    wait_for_packet_count(receiver, 2)

    state = receiver.get_state()

    assert state.right.buttons.trigger == 0.9
    assert state.packet_count == 2


def test_button_events_are_rising_edge_only(receiver):
    port = receiver.local_port

    # Initial unpressed state.
    send_payload(port, make_payload(primary=False))
    wait_for_packet_count(receiver, 1)

    assert not receiver.consume_button_events().right_primary

    # Rising edge.
    send_payload(port, make_payload(primary=True))
    wait_for_packet_count(receiver, 2)

    assert receiver.consume_button_events().right_primary

    # Still held: no second event.
    send_payload(port, make_payload(primary=True))
    wait_for_packet_count(receiver, 3)

    assert not receiver.consume_button_events().right_primary

    # Release and press again.
    send_payload(port, make_payload(primary=False))
    wait_for_packet_count(receiver, 4)

    send_payload(port, make_payload(primary=True))
    wait_for_packet_count(receiver, 5)

    assert receiver.consume_button_events().right_primary


def test_grip_has_no_rising_edge_event(receiver):
    """Grip is a level, not an event.

    RbVr therefore has to derive the grip rising edge itself; see
    RbVr._update_grip_state.
    """
    port = receiver.local_port

    send_payload(port, make_payload(grip=0.0))
    wait_for_packet_count(receiver, 1)
    receiver.consume_button_events()

    send_payload(port, make_payload(grip=1.0))
    wait_for_packet_count(receiver, 2)

    events = receiver.consume_button_events()

    assert not hasattr(events, "right_grip")
    assert receiver.get_state().right.buttons.grip == 1.0


def test_receiver_stale_detection():
    instance = VRReceiver(
        local_ip="127.0.0.1",
        local_port=free_udp_port(),
        meta_quest_ip=None,
        send_handshake=False,
        tracking_timeout_s=0.05,
    )
    instance.start()

    try:
        send_payload(
            instance.local_port,
            make_payload(),
        )
        assert instance.wait_for_first_packet(timeout_s=2.0)

        assert not instance.is_stale()

        time.sleep(0.1)
        assert instance.is_stale()
    finally:
        instance.stop()


def test_state_before_any_packet_is_stale_and_untracked():
    instance = VRReceiver(
        local_ip="127.0.0.1",
        local_port=free_udp_port(),
        meta_quest_ip=None,
        send_handshake=False,
    )

    state = instance.get_state()

    assert instance.is_stale()
    assert not state.right.tracked
    assert not state.head.tracked
    assert state.packet_count == 0
