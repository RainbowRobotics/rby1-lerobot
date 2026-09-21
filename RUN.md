# RB10 + Meta Quest VR 데이터 수집 실행 명령어

`lerobot-record`로 RB10을 VR 텔레오퍼레이션하며 데이터셋을 기록한다.

## 1. 실행 전 확인

```bash
# 이 PC의 IP (--teleop.local_ip 에 넣을 값)
ip -4 -o addr show wlp0s20f3 | awk '{split($4,a,"/"); print a[1]}'

# 로봇 / Quest 도달 확인
ping -c 2 192.168.1.210     # RB10 제어박스
ping -c 2 192.168.0.16      # Meta Quest

# RealSense 인식 확인 (시리얼 2개가 config_rb.py 의 _default_cameras() 와 일치해야 함)
python -c "import pyrealsense2 as rs; print([(d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number)) for d in rs.context().devices])"
```

기대값:

| 항목 | 값 |
|---|---|
| PC (`local_ip`) | `192.168.0.124` |
| RB10 (`robot.ip`) | `192.168.1.210` — TCP 5000/5001 |
| Meta Quest (`meta_quest_ip`) | `192.168.0.16` — UDP 6000 |
| UDP 수신 포트 (`local_port`) | `5005` |
| RealSense `front` | `D405 / 315122271025` — 작업공간 고정 뷰 |
| RealSense `wrist` | `D405 / 262622274852` — 그리퍼 근접 뷰 |

## 2. 실행

```bash
lerobot-record \
  --robot.type=rb10 \
  --robot.ip=192.168.1.210 \
  --robot.gripper_type=rby1_dynamixel \
  --teleop.type=rb_vr \
  --teleop.local_ip=192.168.0.124 \
  --teleop.meta_quest_ip=192.168.0.16 \
  --teleop.send_handshake=true \
  --teleop.use_gripper=true \
  --dataset.repo_id=rainbowrobotics/rb10_vr_demo \
  --dataset.num_episodes=20 \
  --dataset.episode_time_s=1000 \
  --dataset.single_task="Control the RB10 arm and gripper using Meta Quest VR." \
  --dataset.push_to_hub=false \
  --dataset.fps=30
```

IP가 DHCP로 바뀌므로 `local_ip`를 자동으로 채우려면:

```bash
--teleop.local_ip=$(ip -4 -o addr show wlp0s20f3 | awk '{split($4,a,"/"); print a[1]}')
```

> 셸 히스토리(↑)로 올려 쓰지 말 것. 히스토리에 `--robot.address=`(없는 필드),
> `--robot.ip=192.168.0.210`, `--teleop.local_ip=192.168.1.157` 등 옛날 네트워크
> 기준의 조합이 섞여 있다.

## 3. VR 조작 순서

`RbVrConfig` 상태 머신 기준:

1. 컨트롤러 + 헤드셋 트래킹이 수신될 때까지 대기
2. **A 버튼**(오른손 primary) — arm + **초기 자세로 5초간 이동** +
   **오퍼레이터 좌표계를 현재 HMD 자세로 재설정**
   (`(-172.5253, -3.3101, 141.8867, 41.4234, 277.4747, 180)` deg). 도착할
   때까지 Grip은 무시된다. 초기 자세는 EE(그리퍼 끝) 위치
   `(-618.6, 110.5, 264.0)` mm, **EE `x,y` 평면이 base `x,y` 평면과 정확히
   평행**(틀어짐 0)하고 **그리퍼는 수평으로 base `-X`** 를 향한다.
3. **Grip 누름** — 그 순간을 앵커로 잡고 추종 시작 (ServoJ 송신).
   누른 순간에는 로봇이 움직이지 않는다.
4. **Grip 유지** — 앵커 이후의 손 이동량을 1:1로 EE 위치에 반영.
   손 회전량은 같은 축으로 `orientation_scale` 배(기본 `0.3`)만 반영되고,
   별도의 각도 제한은 없다 (`orientation_scale` 기본 `1.0` 이면 1:1).
   손 회전 델타 자체가 최대 180° 이므로 명령 회전은 `orientation_scale × 180°` 를
   넘을 수 없다. 손 회전 **110° 까지는 정확히 추종**하고, 그 이상 최악 축으로 돌리면
   팔꿈치가 펴지면서 몇 도 오차가 남는다 (위치는 정확, 발산은 없음).
5. **Grip 해제** — 추종 정지 + 앵커 폐기 (다시 누르면 재앵커)
6. **B 버튼**(secondary) — 완전 disarm

> **자세 추종 첫 실기 시험 (녹화 전 1회 필수).**
> 회전 방향은 아직 실물로 검증된 적이 없다 (축 대응표는 **이동**만 측정한 것이다).
> `--teleop.orientation_scale=0.1` 로 실행 → A → Grip → 손을 앞뒤 축 둘레로 90°
> 굴려 플랜지가 **눈으로 보기에 같은 방향으로** 도는지 확인한다.
> 거울처럼 반대로 돌면 중단하고 `lerobot-teleoperator-rb/README.md` 의
> "회전 방향" 항목을 볼 것 — 고칠 곳은 부호 하나다.

> **A 이동 중 Grip을 쥐고 있었다면 놓았다가 다시 눌러야 한다.**
> 앵커는 Grip rising edge에서 잡히므로, 도착 즉시 로봇이 움직이지 않는다.

> 손 이동량은 **A 를 누른 시점의 HMD 자세**로 정해지는 오퍼레이터 좌표계 기준으로
> 측정되고, 다음 A 까지 고정된다. 조작 중은 물론 **스트로크 사이에** 고개를 돌리거나
> 몸 방향을 바꿔도 축 대응이 달라지지 않는다. 방향이 어긋났으면 **A 를 다시 누른다**.

> **작업자 ↔ 로봇 축 대응** (작업자 오른손 좌표계: 정면 `+X`, 왼쪽 `+Y`, 위 `+Z`):
>
> | 손 이동 | base | 홈 자세 EE |
> |---|---|---|
> | `+X` 정면 | `-X` | `-Y` (그리퍼가 향하는 방향) |
> | `+Y` 왼쪽 | `-Y` | `+X` |
> | `+Z` 위 | `+Z` | `+Z` |
>
> 보정 상수 `R_OPERATOR_TO_BASE` 는 `Rz(-90°)` 다. 초기 자세 EE 가 base 를
> `-90°` 요(yaw)한 자세라, 이 회전이 그것을 상쇄해 "손 정면 = 그리퍼가
> 향하는 방향"이 된다. 초기 자세의 orientation 을 바꾸면 이 상수도 반드시
> 같이 다시 잡아야 한다. 스테이션 배치가 바뀌어도 다시 잡되 행렬식 +1 인
> 정상 회전이어야 한다.
>
> **이 표는 회전축에도 그대로 적용된다.** 손을 `+X`(정면) 축 둘레로 돌리면 툴은
> base `-X` 축 둘레로 돈다. 적용 각도는 손 회전각 × `orientation_scale`.

> 이전의 절대 매핑(손 위치 = 로봇 위치)과 달리 스케일이 **1:1** 이다.
> 예전에는 `user_scale` 이 기본 약 1.86배로 손 움직임을 증폭했으므로,
> 같은 손동작에 로봇이 약 절반만 움직이는 것이 정상이다.

## 4. 변형

카메라 없이:

```bash
  --robot.cameras='{}'
```

카메라만 따로 점검 (로봇·Quest 없이 — **녹화 전 필수**):

```bash
cd lerobot-robot-rb && python tests/manual/check_cameras.py 20
```

설정된 카메라를 전부 동시에 열어 협상된 모드·읽기 실패·관측 소요 시간을 출력한다.
실측 기준값: 두 대 동시에 **30.0 Hz, 읽기 실패 0회, 카메라 주기 대비 1.00배**
(카메라마다 독립 스레드라 읽기가 파이프라인되어, 2대여도 1대와 비용이 같다).

그리퍼 없이:

```bash
  --robot.gripper_type=none \
  --teleop.use_gripper=false
```

자세 추종 비율 조절 (`orientation_scale`, 기본 `0.3`):

```bash
  --teleop.orientation_scale=0.0   # EE 자세 완전 고정 (이 기능 도입 전과 동일)
  --teleop.orientation_scale=0.1   # 실기 첫 시험용
  --teleop.orientation_scale=0.3   # 손 90° -> 툴 27°
```

기본값은 `1.0` (1:1) 이다.

제어 주기를 `--dataset.fps` 와 맞추려면 (`control_rate_hz` 기본값은 35):

```bash
  --robot.control_rate_hz=30
```

## 5. 자주 나는 에러

| 증상 | 원인 | 조치 |
|---|---|---|
| `DecodingError \`robot\`: The fields \`ip_address\` are not valid for Rb10Config` | 필드명은 `ip` (`config_rb.py:77`). `ip_address` / `address` 는 없음 | `--robot.ip=` 사용 |
| `OSError: [Errno 99] Cannot assign requested address` (`vr_receiver.py` `sock.bind`) | `--teleop.local_ip` 이 이 PC에 할당된 주소가 아님. 같은 서브넷이어도 내 주소가 아니면 bind 불가 | `ip -4 -o addr show` 로 실제 주소 확인 후 지정 |
| `DeviceNotConnectedError: No connected RB robot is registered.` | `RbVr._resolve_robot()` 이 `get_active_rb_cobot()` 으로 로봇을 못 찾음 (`rb_vr.py:798`) | `robot.connect()` 가 먼저 끝났는지 로그 확인 |
| `meta/info.json` 만 남고 `data/`, `videos/` 없음 | 데이터셋 생성(`lerobot_record.py:439`) 이후, `robot.connect()`(456) / `teleop.connect()`(458) 단계에서 죽음 | 위 두 항목부터 확인 |
| 에러 없이 멈춤 | Quest가 UDP 패킷을 안 보냄 | Quest 앱 실행 여부, `send_handshake=true`, 방화벽 확인 |
| 손을 크게 돌렸는데 툴 자세가 명령보다 몇 도 모자람 | 손 회전 110° 이상에서 팔꿈치(j2)가 완전히 펴진 특이 자세 | 정상 범위. Grip을 놓았다 다시 눌러 재앵커하거나 `orientation_scale` 을 낮춘다 |
| 손을 돌렸는데 툴이 반대로 돔 | Quest 쿼터니언 규약이 가정과 켤레만큼 다름 | `lerobot-teleoperator-rb/README.md` 의 "회전 방향" 항목 |

기록된 데이터셋 위치:

```
~/.cache/huggingface/lerobot/rainbowrobotics/
```
