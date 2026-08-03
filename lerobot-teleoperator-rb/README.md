# RB10 Meta Quest Teleoperation

Meta Quest 컨트롤러로 RB10을 조작하고 LeRobot 데이터셋을 수집하기 위한 플러그인입니다.

## Network (예시)

| Device     | IP              | Port               |
| ---------- | --------------- | ------------------ |
| RB10       | `192.168.0.210` | TCP `5000`, `5001` |
| Control PC | `192.168.0.245` | UDP `5005`         |
| Meta Quest | `192.168.0.206` | UDP `6000`         |

## Controls

* **A 버튼**: VR 제어 초기화
* **Grip 유지**: 로봇 조작
* **Grip 해제**: 로봇 정지
* **B 버튼**: VR 제어 종료
* **Trigger**: 그리퍼 제어

실행 후 **A 버튼을 누르고 Grip을 유지**해야 로봇이 움직입니다.

## Teleoperation

```bash
lerobot-teleoperate \
  --robot.type=rb10 \
  --robot.ip=192.168.0.210 \
  --teleop.type=rb_vr \
  --teleop.local_ip=192.168.0.245 \
  --teleop.meta_quest_ip=192.168.0.206 \
  --teleop.send_handshake=true \
  --fps=35
```

팔과 + 그리퍼 사용:

```bash
lerobot-teleoperate \
  --robot.type=rb10 \
  --robot.ip=192.168.0.210 \
  --robot.gripper_type=rby1_dynamixel \
  --teleop.type=rb_vr \
  --teleop.local_ip=192.168.0.245 \
  --teleop.meta_quest_ip=192.168.0.206 \
  --teleop.send_handshake=true \
  --teleop.use_gripper=true \
  --fps=35
```



## Record Dataset

```bash
lerobot-record \
  --robot.type=rb10 \
  --robot.ip=192.168.50.100 \
  --robot.gripper_type=rby1_dynamixel \
  --teleop.type=rb_vr \
  --teleop.local_ip=192.168.50.78 \
  --teleop.meta_quest_ip=192.168.50.112 \
  --teleop.send_handshake=true \
  --teleop.use_gripper=true \
  --dataset.repo_id=rainbowrobotics/rb10_vr_demo \
  --dataset.num_episodes=20 \
  --dataset.single_task="Control the RB10 arm and gripper using Meta Quest VR." \
  --dataset.push_to_hub=true \
  --dataset.fps=30
```


`dataset.repo_id`, `dataset.num_episodes`, `dataset.single_task`는 수집 작업에 맞게 변경합니다.
