import importlib.util
import pathlib

import numpy as np

# Import the module file directly: the package __init__ needs lerobot.
_spec = importlib.util.spec_from_file_location(
    "frame_bus", pathlib.Path(__file__).resolve().parents[1] / "lerobot_robot_rby1" / "frame_bus.py"
)
frame_bus = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(frame_bus)


def test_publish_latest_and_sequence():
    bus = frame_bus.FrameBus()
    assert bus.latest("front") is None and bus.names() == []
    f1 = np.zeros((2, 2, 3), np.uint8)
    bus.publish("front", f1, t=1.0)
    frame, t, seq = bus.latest("front")
    assert frame is f1 and t == 1.0 and seq == 1
    bus.publish("front", np.ones((2, 2, 3), np.uint8))
    assert bus.latest("front")[2] == 2
    bus.publish("left", f1)
    assert sorted(bus.names()) == ["front", "left"]
    bus.clear()
    assert bus.names() == []
