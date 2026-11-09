"""lerobot 0.7.0 remote inference against RB-Y1 over a loopback Zenoh server.

Mirrors upstream tests/remote_inference/test_library_client.py: a real
PolicyServer in a thread on 127.0.0.1, the RB-Y1 plugin on the fake SDK as the
client robot, and the documented library loop (RemoteClient -> admit ->
RemoteInferenceEngine -> dispatch_tick with the robot's hold()).
"""

from __future__ import annotations

import json
import socket
import time
from threading import Event, Thread

import numpy as np
import pytest

pytest.importorskip("zenoh")
pytest.importorskip("msgpack")

from lerobot.inference import dispatch_tick  # noqa: E402
from lerobot.remote_inference import RemoteClient, RemoteInferenceEngine  # noqa: E402
from lerobot.remote_inference.server import PolicyServer, SessionWorker  # noqa: E402
from lerobot.transport.zenoh import ZenohConfig, ZenohTransport  # noqa: E402
from lerobot.utils.action_interpolator import ActionInterpolator  # noqa: E402
from lerobot_robot_rby1 import Rby1, Rby1Config  # noqa: E402
from lerobot_robot_rby1 import profile as rprofile  # noqa: E402
from lerobot_robot_rby1.server_config import build_server_config  # noqa: E402

SEMANTICS = rprofile.default_semantics("a", "1.2")
CAMERAS = {"front": {"type": "fake", "fps": 30, "width": 64, "height": 48}}
STATE_NAMES = [f"right_arm_{i}.pos" for i in range(7)] + [f"left_arm_{i}.pos" for i in range(7)] + [
    "right_gripper_0.pos",
    "left_gripper_0.pos",
]


def _camera_configs():
    import draccus
    from lerobot.cameras import CameraConfig

    return {name: draccus.decode(CameraConfig, cfg) for name, cfg in CAMERAS.items()}


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def tiny_dataset_and_checkpoint(tmp_path_factory):
    """A 16-dim RB-Y1 dataset meta + a randomly initialised ACT checkpoint matching it."""
    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    root = tmp_path_factory.mktemp("remote")
    ds = root / "ds"
    (ds / "meta").mkdir(parents=True)
    info = {
        "codebase_version": "v3.0",
        "robot_type": "rby1",
        "fps": 30,
        "features": {
            "action": {"dtype": "float32", "shape": [16], "names": STATE_NAMES},
            "observation.state": {"dtype": "float32", "shape": [16], "names": STATE_NAMES},
            "observation.images.front": {"dtype": "video", "shape": [48, 64, 3], "names": ["height", "width", "channels"]},
        },
    }
    (ds / "meta" / "info.json").write_text(json.dumps(info))
    robot_cfg = Rby1Config(address="stub.invalid:1", model="a", version="1.2", cameras=_camera_configs())
    profile = rprofile.build_profile(
        robot_cfg, robot={"model": "a", "version": "1.2", "semantics": SEMANTICS}, version="1.2"
    )
    rprofile.write_profile(ds / "meta" / rprofile.PROFILE_FILENAME, profile)

    cfg = ACTConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(16,)),
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 48, 64)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(16,))},
        chunk_size=10,
        n_action_steps=10,
        dim_model=32,
        n_heads=2,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        vision_backbone="resnet18",
        pretrained_backbone_weights=None,
        use_vae=False,
        device="cpu",
    )
    policy = ACTPolicy(cfg)
    ckpt = root / "ckpt"
    stats = {
        "observation.state": {"mean": torch.zeros(16), "std": torch.ones(16)},
        "action": {"mean": torch.zeros(16), "std": torch.ones(16)},
        "observation.images.front": {"mean": torch.zeros(3, 1, 1), "std": torch.ones(3, 1, 1)},
    }
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats)
    policy.save_pretrained(ckpt)
    pre.save_pretrained(ckpt)
    post.save_pretrained(ckpt)
    return ds, ckpt


@pytest.fixture
def remote_server(tiny_dataset_and_checkpoint):
    from lerobot.inference import RemoteInferenceConfig
    from lerobot.remote_inference import load_deployment
    from lerobot.remote_inference.configs import ServerConfig
    import draccus

    ds, ckpt = tiny_dataset_and_checkpoint
    endpoint = f"tcp/127.0.0.1:{_free_port()}"
    raw = build_server_config(checkpoint=str(ckpt), dataset=str(ds), device="cpu", listen_endpoint=endpoint)
    server_cfg = draccus.decode(ServerConfig, raw)
    runner, identity = load_deployment(server_cfg)
    worker = SessionWorker(runner, deployment="rby1", artifact_identity=identity, semantics=SEMANTICS)
    transport = ZenohTransport(ZenohConfig(listen_endpoints=[endpoint]))
    server = PolicyServer(worker, transport)
    ready = Event()
    original = transport.declare_token

    def declare_token(key):
        token = original(key)
        ready.set()
        return token

    transport.declare_token = declare_token
    thread = Thread(target=server.serve, daemon=True)
    try:
        thread.start()
        assert ready.wait(20), "policy server did not become ready"
        yield RemoteInferenceConfig(
            endpoint=endpoint,
            deployment="rby1",
            semantics=SEMANTICS,
            chunk_merge="aligned",
            handshake_timeout_s=5,
            action_timeout_s=5,
        )
    finally:
        server.stop()
        thread.join(10)
        transport.close()
        worker.close()


def test_server_config_matches_robot_features(tiny_dataset_and_checkpoint) -> None:
    ds, ckpt = tiny_dataset_and_checkpoint
    cfg = build_server_config(checkpoint=str(ckpt), dataset=str(ds), device="cpu")
    robot = Rby1(Rby1Config(address="stub.invalid:1", model="a", version="1.2", cameras=_camera_configs()))
    assert cfg["action_feature"]["names"] == list(robot.action_features)
    assert [f["name"] for f in cfg["features"]] == ["observation.state", "observation.images.front"]
    assert cfg["features"][1]["shape"] == [48, 64, 3] and cfg["features"][1]["kind"] == "rgb"
    assert cfg["semantics"] == SEMANTICS and cfg["robot_type"] == "rby1"


def test_remote_engine_drives_and_holds_rby1(remote_server) -> None:
    config = remote_server
    client = RemoteClient.connect(config)
    caps = client.capabilities
    client.admit(
        features=caps.features,
        action_feature=caps.action_feature,
        semantics=config.semantics,
        action_interval=caps.action_interval,
        mode=config.mode,
    )
    names = list(caps.action_feature.names)
    robot = Rby1(
        Rby1Config(address="stub.invalid:1", model="a", version="1.2", move_to_ready_on_connect=False,
                   cameras=_camera_configs())
    )
    robot.connect()
    holds = {"n": 0}

    def send(target):
        robot.send_action({k: float(v) for k, v in zip(names, target.tolist())})

    def hold():
        holds["n"] += 1
        robot.hold()

    engine = RemoteInferenceEngine(client, config, task="smoke task")
    interpolator = ActionInterpolator(multiplier=1)
    engine.start()
    engine.resume()
    try:
        before = robot._robot.n_stream_commands
        deadline = time.monotonic() + 15
        dispatched = 0
        while dispatched < 5 and time.monotonic() < deadline:
            obs = robot.get_observation()
            engine.submit_observation(
                {
                    "observation.state": np.array([obs[n] for n in names], dtype=np.float32),
                    "observation.images.front": np.asarray(obs["front"], dtype=np.uint8),
                },
                capture_time=time.monotonic(),
            )
            if dispatch_tick(engine, interpolator, send=send, hold=hold) is not None:
                dispatched += 1
            time.sleep(1 / 30)
        assert dispatched >= 5, "no actions reached the robot through the remote engine"
        assert robot._robot.n_stream_commands > before
        # Fault the engine: the next tick must hold (our hold() re-sends the last target).
        sent_before = robot._robot.n_stream_commands
        engine._fault("test")
        assert dispatch_tick(engine, interpolator, send=send, hold=hold) is None
        assert holds["n"] >= 1 and robot._robot.n_stream_commands == sent_before + 1
    finally:
        engine.stop()
        robot.disconnect()
        client.close()
