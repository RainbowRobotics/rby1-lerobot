# LeRobot Async Inference (RB-Y1)

A standalone package that extracts LeRobot's `async_inference` module for RB-Y1.
It supports the GR00T-ZMQ / Pi0.5-ZMQ backends alongside the gRPC backend.

## Installation

```bash
pip install -e .   # this package
```

## Structure

| Module | Description |
|--------|-------------|
| `policy_server.py` | gRPC policy server |
| `robot_client.py` | Robot client (gRPC / groot_zmq / pi05_zmq) |
| `robot_replay.py` | Replay recorded actions (JSONL) |
| `policy/groot_zmq.py` | GR00T N1.6 ZMQ client |
| `policy/pi05_zmq.py` | Pi0.5 ZMQ client |

## CLI

```bash
# Policy server
lerobot-policy-server --host=0.0.0.0 --port=8080

# Robot client (gRPC)
lerobot-robot-client --server_address=127.0.0.1:8080 --robot.type=rby1

# Robot client (GR00T ZMQ)
lerobot-robot-client --backend=groot_zmq --robot.type=rby1 --server_address=127.0.0.1:5555

# Action replay
lerobot-robot-replay --robot.type=rby1 --actions-file=actions.jsonl --fps=30
```

Running via `python -m lerobot_async_inference.robot_client` works the same way.
