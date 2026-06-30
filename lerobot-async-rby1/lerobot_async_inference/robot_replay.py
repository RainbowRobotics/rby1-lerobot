import json
import time
from dataclasses import dataclass
from pathlib import Path
import sys

import draccus

from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.utils.import_utils import register_third_party_plugins

"""

python -m lerobot.async_inference.robot_replay \
  --robot.type=rby1 \
  --actions-file /home/nvidia/052_lerobot/lerobot-rby1/final_actions_20260624_224751.jsonl \
  --fps=30 \
  --start-index=100 \
  --max-steps=300


"""


@dataclass
class ReplayConfig:
    robot: RobotConfig
    actions_file: str = "/home/nvidia/052_lerobot/lerobot-rby1/final_actions_20260624_224751.jsonl"
    fps: float = 30.0
    start_index: int = 100
    max_steps: int | None = None 

def load_actions(path: Path, start_index: int = 0, max_steps: int | None = None):
    actions = []

    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            if line_idx < start_index:
                continue

            record = json.loads(line)
            actions.append(record["action"])

            if max_steps is not None and len(actions) >= max_steps:
                break

    return actions


@draccus.wrap()
def main(cfg: ReplayConfig):
    actions_path = Path(cfg.actions_file)

    if not actions_path.exists():
        raise FileNotFoundError(f"actions_file not found: {actions_path}")

    actions = load_actions(
        actions_path,
        start_index=cfg.start_index,
        max_steps=cfg.max_steps,
    )

    if len(actions) == 0:
        raise RuntimeError("No actions loaded.")

    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    dt = 1.0 / cfg.fps

    print(f"[INFO] Loaded actions: {len(actions)}")
    print(f"[INFO] Replay fps: {cfg.fps}")
    print(f"[INFO] Replay dt: {dt:.6f}s")
    print(f"[INFO] File: {actions_path}")

    try:
        for i, action in enumerate(actions):
            loop_start = time.perf_counter()

            robot.send_action(action)

            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0.0, dt - elapsed)
            time.sleep(sleep_time)

            if i % 30 == 0:
                print(f"[REPLAY] {i}/{len(actions)}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        robot.disconnect()
        print("[INFO] Robot disconnected.")


if __name__ == "__main__":
    register_third_party_plugins()
    sys.argv.append("--robot.type=rby1")
    main()