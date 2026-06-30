import numpy as np

from lerobot_async_inference.policy.pi05_zmq import (
    Pi05ZMQClient,
    build_pi05_observation,
    pi05_action_dict_to_timed_actions,
)

from pathlib import Path
import pandas as pd
import imageio.v3 as iio

############################################################################## 
# fake obs 준비 

# raw_observation = {
#     **{f"right_arm_{i}": 0.0 for i in range(7)},
#     **{f"left_arm_{i}": 0.0 for i in range(7)},
#     "right_gripper_0": 0.0,
#     "left_gripper_0": 0.0,

#     "front": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
#     "left_wrist": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
#     "right_wrist": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),

#     "task": "pick up the can",
# }


# 학습에 썼던 데이터 준비


FAKE_DATA_ROOT = "your_path_to_data"

parquet_path = FAKE_DATA_ROOT / "data" / "chunk-000" / "episode_000000.parquet"

front_video_path = FAKE_DATA_ROOT / "videos" / "chunk-000" / "observation.images.front" / "episode_000000.mp4"
left_video_path = FAKE_DATA_ROOT / "videos" / "chunk-000" / "observation.images.left" / "episode_000000.mp4"
right_video_path = FAKE_DATA_ROOT / "videos" / "chunk-000" / "observation.images.right" / "episode_000000.mp4"

df = pd.read_parquet(parquet_path)

idx = 0
row = df.iloc[idx]

front_img = iio.imread(front_video_path, index=idx)
left_img = iio.imread(left_video_path, index=idx)
right_img = iio.imread(right_video_path, index=idx)

state = row["observation.state"]

raw_observation = {
    **{f"right_arm_{i}": float(state[i]) for i in range(7)},
    **{f"left_arm_{i}": float(state[7 + i]) for i in range(7)},
    "right_gripper_0": float(state[14]),
    "left_gripper_0": float(state[15]),

    "front": front_img,
    "left_wrist": left_img,
    "right_wrist": right_img,

    "task": "pick up the can",
}









obs = build_pi05_observation(
    raw_observation,
    front_camera_key="front",
    left_wrist_camera_key="left_wrist",
    right_wrist_camera_key="right_wrist",
)


# 연결 서버정보 config 

client = Pi05ZMQClient(
    server_address="127.0.0.1:5555",
    timeout_ms=300_000,
)

############################################################################## 


print("1. ping test")
client.ping()
print("ping done")

print("2. action test")
action_dict = client.get_action(obs)

print(action_dict.keys())

timed_actions = pi05_action_dict_to_timed_actions(
    action_dict,
    timestamp=0.0,
    timestep=0,
    environment_dt=0.1,
)

print("length of timed actions :",len(timed_actions))
print("Shape of timed action", timed_actions[0].action.shape)

print(action_dict["actions"].shape)
print(timed_actions[0].action)

print("min:", action_dict["actions"].min())
print("max:", action_dict["actions"].max())
print("mean:", action_dict["actions"].mean())
print("first:", action_dict["actions"][0])
print("last:", action_dict["actions"][-1])

print("state shape:", np.asarray(state).shape)
print("state:", np.asarray(state))
print("front:", front_img.shape, front_img.dtype, front_img.min(), front_img.max())
print("left:", left_img.shape, left_img.dtype, left_img.min(), left_img.max())
print("right:", right_img.shape, right_img.dtype, right_img.min(), right_img.max())