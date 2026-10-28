"""lerobot-async-rby1 imports (lerobot 0.6.x only)."""

from __future__ import annotations

import pytest
from lerobot.transport import services_pb2_grpc

pytestmark = pytest.mark.skipif(
    not hasattr(services_pb2_grpc, "AsyncInferenceServicer"),
    reason="installed lerobot has no AsyncInference gRPC service (upstream main)",
)


def test_policy_server_imports() -> None:
    from lerobot_async_inference.policy_server import PolicyServer  # noqa: F401


def test_robot_client_imports() -> None:
    from lerobot_async_inference.robot_client import RobotClient  # noqa: F401


def test_prepare_image_only_rescales_uint8() -> None:
    import torch
    from lerobot_async_inference.helpers import prepare_image

    assert prepare_image(torch.full((3, 2, 2), 255, dtype=torch.uint8)).max() == 1.0
    assert prepare_image(torch.full((3, 2, 2), 0.5)).max() == 0.5
