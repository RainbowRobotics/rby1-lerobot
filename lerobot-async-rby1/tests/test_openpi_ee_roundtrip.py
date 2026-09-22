"""Cross-repository contract test with real transforms and local ZMQ, no model/GPU.

Run with OpenPI's environment and source tree on PYTHONPATH. The deterministic
policy substitutes only the neural network, not the EE or wire transforms.
"""

import importlib.util
from pathlib import Path
import queue
import sys
import threading
import types

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

transforms = pytest.importorskip("openpi.transforms")
rbc_ee_policy = pytest.importorskip("openpi.policies.rbc_ee_policy")
Pi05ZMQPolicyServer = pytest.importorskip("openpi.serving.zmq_policy_server").Pi05ZMQPolicyServer
config = pytest.importorskip("openpi.training.config")


def _module(monkeypatch, name, path, package=False):
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(path.parent)] if package else None
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("model", ["rb10", "rb10e"])
@pytest.mark.parametrize("front", [False, True])
def test_full_ee_wire_roundtrip(monkeypatch, model, front):
    repo = Path(__file__).resolve().parents[2]
    robot_package = types.ModuleType("lerobot_robot_rb")
    robot_package.__path__ = [str(repo / "lerobot-robot-rb/lerobot_robot_rb")]
    monkeypatch.setitem(sys.modules, "lerobot_robot_rb", robot_package)
    _module(
        monkeypatch,
        "lerobot_robot_rb.ee_kinematics",
        repo / "lerobot-robot-rb/lerobot_robot_rb/ee_kinematics.py",
    )
    policy_root = repo / "lerobot-async-rby1/lerobot_async_inference/policy"
    package = types.ModuleType("_ee_roundtrip_policy")
    package.__path__ = [str(policy_root)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    _module(monkeypatch, f"{package.__name__}.zmq_wire", policy_root / "zmq_wire.py")
    protocol = _module(
        monkeypatch, f"{package.__name__}.rb10_ee", policy_root / "rb10_ee.py"
    )
    adapter = protocol.EEPolicyAdapter(model)
    q = np.array([0.2, -0.6, 1.1, -0.8, 0.5, 0.3])
    target_q = q + np.array([0.001, -0.002, 0.001, 0.002, -0.001, 0.002])
    target = np.r_[adapter.kinematics.forward(target_q), 0.8]
    expected_actions = np.tile(target, (40, 1))

    class TransformPolicy:
        def infer(self, observation):
            encoded = rbc_ee_policy.RBCEEInputs()(
                {**observation, "actions": expected_actions}
            )
            assert bool(encoded["image_mask"]["base_0_rgb"]) == front
            mask = (True,) * 3 + (False,) * 7
            delta = transforms.DeltaActions(mask)(encoded)
            np.testing.assert_allclose(
                delta["actions"][:, 3:],
                rbc_ee_policy.encode_pose(expected_actions)[:, 3:],
            )
            # Exercise the padded model boundary without running a network.
            model_action = np.pad(delta["actions"], ((0, 0), (0, 22)))
            decoded = transforms.AbsoluteActions(mask)(
                {"actions": model_action, "state": encoded["state"]}
            )
            return rbc_ee_policy.RBCEEOutputs()(decoded)

    addresses, errors = queue.Queue(), queue.Queue()
    received = threading.Event()

    def serve():
        server = Pi05ZMQPolicyServer(
            TransformPolicy(),
            "127.0.0.1",
            0,
            3000,
            train_config=types.SimpleNamespace(data=config.LeRobotRBCEEDataConfig()),
        )
        addresses.put(server.sock.getsockopt_string(protocol.zmq.LAST_ENDPOINT))
        try:
            for _ in range(2):
                server.sock.send(
                    protocol.MsgSerializer.to_bytes(
                        server.process_request(server.sock.recv())
                    )
                )
        except Exception as exc:
            errors.put(exc)
        finally:
            received.wait(timeout=3)
            server.close()

    worker = threading.Thread(target=serve)
    worker.start()
    client = protocol.ReconnectingEEClient(addresses.get(timeout=3), timeout_ms=2000)
    raw = {
        **dict(zip(protocol.JOINT_KEYS, q.tolist())),
        "gripper_0": 0.8,
        "wrist": np.full((224, 224, 3), 64, np.uint8),
    }
    if front:
        raw["front"] = np.full((224, 224, 3), 128, np.uint8)
    try:
        client.describe()
        actions = protocol.parse_ee_actions(
            client.get_action(adapter.build_observation(raw, "test"))
        )
        command = adapter.joint_action(actions[0], q, max_joint_delta=0.02)
        result_pose = adapter.kinematics.forward(
            [command[key] for key in protocol.JOINT_KEYS]
        )
        np.testing.assert_allclose(result_pose[:3], target[:3], atol=1e-6)
        np.testing.assert_allclose(
            Rotation.from_euler("xyz", result_pose[3:]).as_matrix(),
            Rotation.from_euler("xyz", target[3:6]).as_matrix(),
            atol=1e-6,
        )
        assert command["gripper_0"] == pytest.approx(0.8)
    finally:
        received.set()
        client.close()
        worker.join(timeout=4)
    assert not worker.is_alive()
    assert errors.empty()
