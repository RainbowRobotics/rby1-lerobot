import json
import socket
import time

import numpy as np

from lerobot_teleoperator_rb.frame_transforms import (
    quest_pose_to_rb_frame,
)
from lerobot_teleoperator_rb.vr_receiver import (
    VRReceiver,
    parse_vr_payload,
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


def send_payload(port, payload):
    with socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM,
    ) as sock:
        sock.sendto(
            json.dumps(payload).encode("utf-8"),
            ("127.0.0.1", port),
        )


def wait_for_packet_count(receiver, count, timeout_s=1.0):
    deadline = time.monotonic() + timeout_s

    while time.monotonic() < deadline:
        if receiver.packet_count >= count:
            return
        time.sleep(0.01)

    raise AssertionError(
        f"Expected {count} packets, got {receiver.packet_count}."
    )


def test_parse_vr_payload():
    payload = make_payload()

    snapshot = parse_vr_payload(
        payload,
        received_at=10.0,
        sender=("127.0.0.1", 5000),
    )

    expected_pose = quest_pose_to_rb_frame(
        [0.4, -0.2, 1.2],
        [0.0, 0.0, 0.0, 1.0],
    )

    assert snapshot.received_at == 10.0
    assert snapshot.source_timestamp == 123.4
    assert snapshot.head_tracked
    assert snapshot.right.tracked
    assert not snapshot.left.tracked
    assert np.allclose(
        snapshot.right.pose_rb,
        expected_pose,
    )
    assert snapshot.right.trigger == 0.25
    assert snapshot.right.grip == 0.75


def test_analog_values_are_clamped():
    payload = make_payload(
        trigger=2.0,
        grip=-1.0,
    )

    snapshot = parse_vr_payload(payload)

    assert snapshot.right.trigger == 1.0
    assert snapshot.right.grip == 0.0


def test_receiver_keeps_latest_state():
    receiver = VRReceiver(
        local_ip="127.0.0.1",
        local_port=0,
        send_handshake=False,
    )
    receiver.start()

    try:
        assert receiver.bound_port is not None

        send_payload(
            receiver.bound_port,
            make_payload(trigger=0.1),
        )
        send_payload(
            receiver.bound_port,
            make_payload(trigger=0.9),
        )

        wait_for_packet_count(receiver, 2)

        state = receiver.get_state()

        assert state is not None
        assert state.right.trigger == 0.9
        assert receiver.packet_count == 2
    finally:
        receiver.stop()


def test_button_events_are_rising_edge_only():
    receiver = VRReceiver(
        local_ip="127.0.0.1",
        local_port=0,
        send_handshake=False,
    )
    receiver.start()

    try:
        port = receiver.bound_port
        assert port is not None

        # Initial unpressed state.
        send_payload(
            port,
            make_payload(primary=False),
        )
        wait_for_packet_count(receiver, 1)

        assert not receiver.consume_button_events()[
            "right_primary"
        ]

        # Rising edge.
        send_payload(
            port,
            make_payload(primary=True),
        )
        wait_for_packet_count(receiver, 2)

        assert receiver.consume_button_events()[
            "right_primary"
        ]

        # Still held: no second event.
        send_payload(
            port,
            make_payload(primary=True),
        )
        wait_for_packet_count(receiver, 3)

        assert not receiver.consume_button_events()[
            "right_primary"
        ]

        # Release and press again.
        send_payload(
            port,
            make_payload(primary=False),
        )
        wait_for_packet_count(receiver, 4)

        send_payload(
            port,
            make_payload(primary=True),
        )
        wait_for_packet_count(receiver, 5)

        assert receiver.consume_button_events()[
            "right_primary"
        ]
    finally:
        receiver.stop()


def test_receiver_stale_detection():
    receiver = VRReceiver(
        local_ip="127.0.0.1",
        local_port=0,
        send_handshake=False,
    )
    receiver.start()

    try:
        port = receiver.bound_port
        assert port is not None

        send_payload(port, make_payload())
        assert receiver.wait_for_first_packet(timeout_s=1.0)

        assert not receiver.is_stale(0.5)

        time.sleep(0.06)
        assert receiver.is_stale(0.05)
    finally:
        receiver.stop()
