"""Real local ZMQ transport and pure FK/IK adapter tests; no robot connection."""

import importlib.util
from pathlib import Path
import queue
import sys
import threading
import time
import types

import numpy as np
import pytest
import zmq


@pytest.fixture
def protocol(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "lerobot_async_inference" / "policy"
    package = types.ModuleType("_ee_test_policy")
    package.__path__ = [str(root)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    for name in ("zmq_wire", "rb10_ee"):
        spec = importlib.util.spec_from_file_location(
            f"{package.__name__}.{name}", root / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("shape", [(1, 7), (40, 7), (1, 1, 7), (1, 40, 7)])
def test_valid_actions_and_one_step_not_squeezed(protocol, shape):
    assert protocol.parse_ee_actions({"actions": np.zeros(shape)}).shape[-1] == 7


@pytest.mark.parametrize(
    "value",
    [
        np.zeros((40, 10)),
        np.zeros((40, 32)),
        np.zeros(7),
        np.zeros((0, 7)),
        np.zeros((41, 7)),
        np.full((40, 7), np.nan),
    ],
)
def test_bad_actions_rejected(protocol, value):
    with pytest.raises(ValueError):
        protocol.parse_ee_actions({"actions": value})


def test_descriptor_rejects_joint_or_wrong_units(protocol):
    protocol.validate_descriptor(protocol.EE_PROTOCOL)
    for key in protocol.EE_PROTOCOL:
        invalid = dict(protocol.EE_PROTOCOL)
        invalid[key] = None
        with pytest.raises(ValueError, match="mismatch"):
            protocol.validate_descriptor(invalid)


def test_wire_preserves_numpy_without_pickle(protocol):
    value = {
        "state": np.arange(7, dtype=np.float32),
        "prompt": "task",
        "image": np.zeros((5, 6, 3), np.uint8),
    }
    actual = protocol.MsgSerializer.from_bytes(protocol.MsgSerializer.to_bytes(value))
    np.testing.assert_array_equal(actual["state"], value["state"])
    assert actual["state"].dtype == np.float32
    assert actual["image"].dtype == np.uint8
    assert actual["prompt"] == "task"


def test_real_req_socket_recovers_and_discards_late_reply(protocol):
    address = queue.Queue()
    errors = queue.Queue()
    received = threading.Event()

    def serve():
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REP)
        sock.setsockopt(zmq.RCVTIMEO, 3000)
        sock.setsockopt(zmq.LINGER, 0)
        port = sock.bind_to_random_port("tcp://127.0.0.1")
        address.put(f"127.0.0.1:{port}")
        try:
            for i in range(3):
                request = protocol.MsgSerializer.from_bytes(sock.recv())
                if i == 0:
                    time.sleep(0.15)
                    result = {"stale": True}
                elif request["endpoint"] == "describe":
                    result = protocol.EE_PROTOCOL
                else:
                    result = {"actions": np.ones((1, 7), np.float32)}
                sock.send(protocol.MsgSerializer.to_bytes(result))
        except Exception as exc:
            errors.put(exc)
        finally:
            received.wait(timeout=2)
            sock.close()
            ctx.term()

    worker = threading.Thread(target=serve)
    worker.start()
    client = protocol.ReconnectingEEClient(address.get(timeout=2), timeout_ms=100)
    try:
        with pytest.raises(TimeoutError):
            client.get_action({})
        time.sleep(0.2)
        assert client.describe() == protocol.EE_PROTOCOL
        np.testing.assert_array_equal(
            protocol.parse_ee_actions(client.get_action({})), np.ones((1, 7))
        )
    finally:
        received.set()
        client.close()
        worker.join(timeout=4)
    assert not worker.is_alive()
    assert errors.empty()


@pytest.fixture
def adapter(protocol):
    instance = protocol.EEPolicyAdapter.__new__(protocol.EEPolicyAdapter)
    instance.wrist_camera_key = "wrist"
    instance.front_camera_key = "front"
    instance.kinematics = types.SimpleNamespace(
        forward=lambda q: np.arange(6) * 0.1,
        inverse_step=lambda target, seed, **kwargs: seed.copy(),
        max_joint_delta=0.5,
        joint_limits=np.tile([-2 * np.pi, 2 * np.pi], (6, 1)),
    )
    return instance


def test_camera_mapping_missing_front_and_buffer_snapshot(adapter):
    raw = {
        **{f"joint_{i}": 0.0 for i in range(6)},
        "gripper_0": 0.7,
        "wrist": np.zeros((3, 4, 3), np.uint8),
    }
    wire = adapter.build_observation(raw, "instruction")
    assert "observation/front_image" not in wire
    assert wire["observation/state"].shape == (7,)
    np.testing.assert_allclose(wire["observation/state"][:6], np.arange(6) * 0.1)
    raw["wrist"][:] = 255
    assert wire["observation/base_image"].max() == 0
    raw["front"] = np.full((3, 4, 3), 10, np.uint8)
    assert (
        adapter.build_observation(raw, "instruction")["observation/front_image"].min()
        == 10
    )


def test_ik_failure_and_out_of_limit_never_become_commands(adapter):
    adapter.kinematics.inverse_step = lambda target, seed, **kwargs: seed + 1
    with pytest.raises(ValueError, match="displacement"):
        adapter.joint_action(np.zeros(7), np.zeros(6), max_joint_delta=0.01)
    with pytest.raises(ValueError, match="finite"):
        adapter.joint_action(np.full(7, np.nan), np.zeros(6), max_joint_delta=0.01)


def test_adapter_preserves_radians_and_gripper_open_convention(adapter):
    q = np.linspace(-0.5, 0.5, 6)
    action = adapter.joint_action(np.r_[np.zeros(6), 1.0], q, max_joint_delta=0.1)
    np.testing.assert_array_equal([action[f"joint_{i}"] for i in range(6)], q)
    assert action["gripper_0"] == 1.0


def test_adapter_uses_cartesian_step_solver_with_servo_limit(adapter):
    def solve(target, seed, *, max_joint_delta, previous_q, max_tracking_error):
        assert max_joint_delta == 0.5 / 30
        assert previous_q is None
        assert max_tracking_error is None
        return seed + max_joint_delta

    adapter.kinematics.inverse_step = solve
    command = adapter.joint_action(np.zeros(7), np.zeros(6), max_joint_delta=0.5 / 30)
    np.testing.assert_allclose([command[f"joint_{i}"] for i in range(6)], 0.5 / 30)


def test_step_respects_measured_and_previous_command_bounds(adapter):
    adapter.kinematics.inverse_step = lambda target, seed, **kwargs: seed.copy()
    command = adapter.joint_action(
        np.zeros(7), np.zeros(6), max_joint_delta=0.01, previous_q=np.full(6, 0.01)
    )
    np.testing.assert_allclose([command[f"joint_{i}"] for i in range(6)], 0.0)
    with pytest.raises(ValueError, match="no safe step intersection"):
        adapter.joint_action(
            np.zeros(7), np.zeros(6), max_joint_delta=0.01, previous_q=np.full(6, 0.03)
        )


def test_unreachable_goal_is_not_hidden_by_step_limiter(adapter):
    def fail(*args, **kwargs):
        raise RuntimeError("inverse kinematics residual exceeds tolerance")

    adapter.kinematics.inverse_step = fail
    with pytest.raises(RuntimeError, match="residual"):
        adapter.joint_action(np.zeros(7), np.zeros(6), max_joint_delta=0.01)


def test_explicit_tracking_preserves_command_anchor_without_relaxing_speed(adapter):
    reference = np.full(6, 0.02)

    def solve(target, seed, *, max_joint_delta, previous_q, max_tracking_error):
        assert max_joint_delta == 0.005
        assert max_tracking_error == 0.05
        np.testing.assert_array_equal(previous_q, reference)
        return previous_q.copy()

    adapter.kinematics.inverse_step = solve
    command = adapter.joint_action(
        np.zeros(7), np.zeros(6), max_joint_delta=0.005,
        previous_q=reference, max_tracking_error=0.05,
    )
    np.testing.assert_array_equal([command[f"joint_{i}"] for i in range(6)], reference)
    adapter.kinematics.inverse_step = lambda *args, **kwargs: reference + 0.006
    with pytest.raises(ValueError, match="displacement"):
        adapter.joint_action(
            np.zeros(7), np.zeros(6), max_joint_delta=0.005,
            previous_q=reference, max_tracking_error=0.05,
        )


def test_explicit_tracking_rejects_excessive_existing_error(adapter):
    with pytest.raises(ValueError, match="tracking error"):
        adapter.joint_action(
            np.zeros(7), np.zeros(6), max_joint_delta=0.005,
            previous_q=np.full(6, 0.06), max_tracking_error=0.05,
        )


def test_goal_outside_model_limits_is_rejected_before_limiting(adapter):
    adapter.kinematics.inverse_step = lambda target, seed, **kwargs: seed + 0.08
    adapter.kinematics.joint_limits = np.tile([-0.05, 0.05], (6, 1))
    with pytest.raises(ValueError, match="safety joint limits"):
        adapter.joint_action(np.zeros(7), np.zeros(6), max_joint_delta=0.01)
