import pickle
import time

import grpc
import torch

from lerobot_async_inference.helpers import (
    RemotePolicyConfig,
    TimedObservation,
)
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks


SERVER_ADDRESS = "127.0.0.1:8080"


def main():
    channel = grpc.insecure_channel(
        SERVER_ADDRESS,
        grpc_channel_options(initial_backoff="0.1s"),
    )
    stub = services_pb2_grpc.AsyncInferenceStub(channel)

    print("1. Ready test")
    stub.Ready(services_pb2.Empty(), timeout=5.0)
    print("Ready done")

    print("2. SendPolicyInstructions test")
    fake_policy_config = RemotePolicyConfig(
        policy_type="act",
        pretrained_name_or_path="/tmp/fake",
        lerobot_features={
            "action": {
                "dtype": "float32",
                "shape": (16,),
                "names": [f"action_{i}" for i in range(16)],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (16,),
                "names": [f"state_{i}" for i in range(16)],
            },
        },
        actions_per_chunk=40,
        device="cpu",
    )

    policy_setup = services_pb2.PolicySetup(
        data=pickle.dumps(fake_policy_config),
    )
    stub.SendPolicyInstructions(policy_setup, timeout=5.0)
    print("SendPolicyInstructions done")

    print("3. SendObservations test")
    raw_observation = {
        **{f"right_arm_{i}": 0.0 for i in range(7)},
        **{f"left_arm_{i}": 0.0 for i in range(7)},
        "right_gripper_0": 0.0,
        "left_gripper_0": 0.0,
        "task": "dummy",
    }

    timed_obs = TimedObservation(
        timestamp=time.time(),
        observation=raw_observation,
        timestep=0,
    )
    timed_obs.must_go = True

    obs_bytes = pickle.dumps(timed_obs)
    obs_iterator = send_bytes_in_chunks(
        obs_bytes,
        services_pb2.Observation,
        log_prefix="[CLIENT] Observation",
        silent=True,
    )

    stub.SendObservations(obs_iterator, timeout=5.0)
    print("SendObservations done")

    print("4. GetActions test")
    actions_chunk = stub.GetActions(services_pb2.Empty(), timeout=5.0)

    if len(actions_chunk.data) == 0:
        raise RuntimeError("GetActions returned Empty data")

    timed_actions = pickle.loads(actions_chunk.data)

    print("GetActions done")
    print("length of timed actions:", len(timed_actions))
    print("shape of first action:", timed_actions[0].action.shape)
    print("first action:", timed_actions[0].action)

    assert len(timed_actions) == 40
    assert isinstance(timed_actions[0].action, torch.Tensor)
    assert timed_actions[0].action.shape == torch.Size([16])

    print("gRPC full dummy pipeline OK")

    channel.close()


if __name__ == "__main__":
    main()