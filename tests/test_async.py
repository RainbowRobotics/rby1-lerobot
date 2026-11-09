"""lerobot-async-rby1 imports (lerobot 0.6.x only)."""

from __future__ import annotations

import pytest

# The async package is only installed (and grpc only present) for lerobot 0.6.x.
pytest.importorskip("grpc", reason="grpc not installed (lerobot-async-rby1 not installed)")
pytest.importorskip("lerobot_async_inference.helpers", reason="lerobot-async-rby1 not importable", exc_type=ImportError)
from lerobot.transport import services_pb2_grpc  # noqa: E402

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
