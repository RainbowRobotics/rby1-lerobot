import numpy as np

from lerobot_async_inference.policy.groot_zmq import (
    GR00TZMQClient,
    build_groot_n16_observation,
    groot_n16_action_dict_to_timed_actions,
)


##############################################################################
# fake obs 준비

raw_observation = {
    **{f"right_arm_{i}": 0.0 for i in range(7)},
    **{f"left_arm_{i}": 0.0 for i in range(7)},
    "right_gripper_0": 0.0,
    "left_gripper_0": 0.0,

    "front": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
    "left_wrist": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
    "right_wrist": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),

    "task": "pick up the can",
}

obs = build_groot_n16_observation(
    raw_observation,
    front_camera_key="front",
    left_wrist_camera_key="left_wrist",
    right_wrist_camera_key="right_wrist",
    image_size=(224, 224),
)


# 연결 서버정보 config

client = GR00TZMQClient(
    server_address="127.0.0.1:5555",
    timeout_ms=300_000,
)

##############################################################################


print("1. ping test")
assert client.ping()
print("ping done")

print("2. action test")
action_dict = client.get_action(obs)

print(action_dict.keys())

timed_actions = groot_n16_action_dict_to_timed_actions(
    action_dict,
    timestamp=0.0,
    timestep=0,
    environment_dt=0.1,
)

print("length of timed actions :", len(timed_actions))
print("Shape of timed action", timed_actions[0].action.shape)

print("left_arm shape:", action_dict["left_arm"].shape)
print("right_arm shape:", action_dict["right_arm"].shape)
print("left_gripper shape:", action_dict["left_gripper"].shape)
print("right_gripper shape:", action_dict["right_gripper"].shape)

print(timed_actions[0].action)

for key in ["right_arm", "left_arm", "right_gripper", "left_gripper"]:
    arr = np.asarray(action_dict[key])
    print(f"{key} min:", arr.min())
    print(f"{key} max:", arr.max())
    print(f"{key} mean:", arr.mean())
    print(f"{key} first:", arr[:, 0])
    print(f"{key} last:", arr[:, -1])

client.close()