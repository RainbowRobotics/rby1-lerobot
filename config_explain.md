# RB10 + Meta Quest VR — 실행 파라미터 설명

`lerobot-record` 로 RB10을 VR 텔레오퍼레이션하며 데이터셋을 기록할 때 쓰는
파라미터들의 의미와 기본값을 정리한 문서다.

실행 절차와 트러블슈팅은 [`RUN.md`](RUN.md), VR 조작 방법은
[`lerobot-teleoperator-rb/README.md`](lerobot-teleoperator-rb/README.md) 를 참고한다.
이 문서는 **각 값이 무슨 의미인지**에만 집중한다.

모든 기본값은 아래 소스 기준이다.

| 설정 클래스 | 파일 |
|---|---|
| `Rb10Config` / `RbCobotConfig` | `lerobot-robot-rb/lerobot_robot_rb/config_rb.py` |
| `RbVrConfig` | `lerobot-teleoperator-rb/lerobot_teleoperator_rb/config_rb_vr.py` |
| `DatasetRecordConfig` / `RecordConfig` | `lerobot/src/lerobot/configs/dataset.py`, `lerobot/src/lerobot/scripts/lerobot_record.py` |
| 안전 상수 (CLI 불가) | `lerobot-teleoperator-rb/lerobot_teleoperator_rb/rb_vr.py` |

---

## 1. 기준 명령어

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
  --dataset.single_task="Control the RB10 arm and gripper using Meta Quest VR." \
  --dataset.push_to_hub=false \
  --dataset.fps=30
```

`lerobot-record` 는 `lerobot/src/lerobot/scripts/lerobot_record.py` 의 엔트리포인트다.
로봇과 텔레오퍼레이터를 연결하고, 에피소드 단위로 관측/액션을 LeRobotDataset 에 기록한다.

---

## 2. 로봇 (`--robot.*` → `Rb10Config`)

### 명령어에 쓴 것

| 인자 | 의미 |
|---|---|
| `--robot.type=rb10` | `rb10.py:18` 의 `@RobotConfig.register_subclass("rb10")` 로 등록된 RB10 어댑터 선택. `MODEL_SPECS["rb10"]` 의 관절 리밋(j2만 ±165°, 나머지 ±360°)을 사용한다 |
| `--robot.ip=192.168.1.210` | RB 제어박스 IPv4. 명령 포트 5000 / 데이터 포트 5001 로 TCP 접속. **필드명은 `ip` 다** — `address` 나 `ip_address` 는 존재하지 않는다 |
| `--robot.gripper_type=rby1_dynamixel` | 그리퍼 종류. 허용값은 `"none"` / `"rby1_dynamixel"` 둘뿐. 제어 PC에 붙은 Dynamixel 그리퍼를 `gripper_port` 기본값 `/dev/rby1_gripper`, `gripper_ids=[0]` 으로 연다. 데이터셋에 `gripper_0` 피처가 추가된다 |

### 명시하지 않았지만 적용되는 기본값

| 파라미터 | 기본값 | 의미 |
|---|---|---|
| `--robot.cameras` | D405 2대 | 비워두면 `_default_cameras()` 가 적용된다. `front`(시리얼 `315122271025`, 480x640 + ROTATE_90), `wrist`(시리얼 `262622274852`, 640x480 무회전). 데이터셋에는 `observation.images.front` / `observation.images.wrist` 로 저장된다. `--robot.cameras='{}'` 로 끌 수 있다 |
| `--robot.control_rate_hz` | `35.0` | ServoJ 파라미터 계산 기준. 아래 `--dataset.fps` 와 **반드시 맞춰야 한다** |
| `--robot.command_port` / `--robot.data_port` | `5000` / `5001` | 제어박스 TCP 포트 |
| `--robot.set_speed_bar_on_connect` | `False` | 접속해도 티칭 펜던트의 속도바를 건드리지 않는다 |
| `--robot.gripper_invert` | `False` | 그리퍼 열림/닫힘 방향 반전 |

> **카메라 2대가 암묵적으로 열린다.** 명령어에 `--robot.cameras` 가 없으면 기본값 2대를
> 열려고 시도하며, 시리얼이 실제 장치와 다르면 여기서 실패한다.
> `RUN.md` 의 pyrealsense2 열거 명령으로 먼저 확인할 것.

---

## 3. 텔레오퍼레이터 (`--teleop.*` → `RbVrConfig`)

### 명령어에 쓴 것

| 인자 | 의미 |
|---|---|
| `--teleop.type=rb_vr` | Meta Quest VR 텔레오퍼레이터 선택 |
| `--teleop.local_ip=192.168.0.124` | **이 PC의 UDP 수신 주소.** `sock.bind()` 에 그대로 쓰이므로 반드시 이 PC에 실제 할당된 주소여야 한다 (같은 서브넷의 남의 주소는 `Errno 99`). 동시에 핸드셰이크로 Quest 에 알려줄 주소이기도 하다 |
| `--teleop.meta_quest_ip=192.168.0.16` | 핸드셰이크를 보낼 Quest 주소 |
| `--teleop.send_handshake=true` | 수신기 시작 시 Quest 로 `{"ip": local_ip, "port": local_port}` 를 전송해 "여기로 트래킹 패킷을 보내라"고 알린다. `true` 면 `meta_quest_ip` 가 **필수**이고 `local_ip` 가 `0.0.0.0` 이면 안 된다 (`__post_init__` 에서 검증) |
| `--teleop.use_gripper=true` | 선택된 컨트롤러의 트리거를 `gripper_0 = 1 - trigger` 로 출력. LeRobot 규약상 **1.0 = 열림, 0.0 = 닫힘** 이라 트리거를 당길수록 닫힌다 |

### 명시하지 않았지만 적용되는 기본값

| 파라미터 | 기본값 | 의미 |
|---|---|---|
| `--teleop.local_port` | `5005` | UDP 수신 포트 |
| `--teleop.meta_quest_port` | `6000` | 핸드셰이크 전송 포트 |
| `--teleop.controller_hand` | `"right"` | 사용할 컨트롤러 |
| `--teleop.require_initialization_button` | `True` | **A 버튼을 먼저 눌러야** Grip 이 팔을 움직인다. `False` 면 접속 즉시 arm 된다 |
| `--teleop.orientation_scale` | `1.0` | 손 회전을 EE 자세에 반영하는 비율 |
| `--teleop.grip_threshold` | `0.5` | 이 값을 넘으면 추종 시작 |
| `--teleop.tracking_timeout_s` | `0.25` | 이보다 오래된 패킷은 stale |
| `--teleop.ik_iterations` | `5` | 틱당 IKLM 반복 횟수 |

---

## 4. 데이터셋 (`--dataset.*` → `DatasetRecordConfig`)

### 명령어에 쓴 것

| 인자 | 의미 |
|---|---|
| `--dataset.repo_id=rainbowrobotics/rb10_vr_demo` | `{org}/{name}` 형식의 데이터셋 식별자 |
| `--dataset.num_episodes=20` | 기록할 에피소드 수. 키보드로 조기 중단 가능 |
| `--dataset.single_task="..."` | 모든 프레임에 붙는 태스크 설명 문자열. 언어 조건부 정책 학습에 쓰인다 |
| `--dataset.push_to_hub=false` | Hugging Face Hub 업로드 안 함. **기본값이 `True` 이므로 명시적으로 꺼야 한다** |
| `--dataset.fps=30` | 기록 주기. 데이터셋 메타데이터에 박히고 제어 루프 속도도 이 값으로 맞춘다 |

> **`repo_id` 에 타임스탬프가 자동으로 붙는다.** 새로 만들 때 `stamp_repo_id()` 가
> 호출되어(`lerobot_record.py:438`) `rb10_vr_demo_20260910_143022` 처럼 바뀐다.
> 실제 저장 경로는 `~/.cache/huggingface/lerobot/rainbowrobotics/rb10_vr_demo_<타임스탬프>` 다.
> `--resume=true` 로 이어서 기록할 때는 붙이지 않는다.

---

## 5. 시간 (`*_time_s`, `*_timeout_s`)

| 파라미터 | 기본값 | 의미 |
|---|---|---|
| `--dataset.episode_time_s` | `60` | 에피소드 하나의 녹화 길이(초). 타이머가 끝나면 자동으로 다음 에피소드로 넘어간다 |
| `--dataset.reset_time_s` | `60` | 에피소드 사이 환경 리셋 시간. 이 구간도 텔레오퍼레이션은 살아있지만 **데이터셋에 기록되지 않는다** (`record_loop` 에 `dataset` 인자를 넘기지 않음). 마지막 에피소드 뒤에는 건너뛴다 |
| `--teleop.tracking_timeout_s` | `0.25` | 이보다 오래된 Quest 패킷은 stale 로 간주 → 추종 중단 + 앵커 폐기. Wi-Fi 가 불안정해 자꾸 끊기면 올릴 수 있지만, 그만큼 오래된 손 위치를 따라간다는 뜻이라 위험하다 |
| `--robot.socket_timeout_s` | `1.0` | RB 제어박스 TCP 소켓 타임아웃 |
| `--robot.first_state_timeout_s` | `2.0` | `connect()` 시 첫 580바이트 상태 패킷을 기다리는 최대 시간 |

**두 시간 파라미터가 총 소요시간을 지배한다.** 기본값 그대로 20 에피소드면
`20×60 + 19×60 ≈ 39분` 이다. 짧은 pick-and-place 라면 다음이 현실적이다.

```bash
  --dataset.episode_time_s=20 \
  --dataset.reset_time_s=10
```

**CLI 로 바꿀 수 없는 시간 상수** (`rb_vr.py` 클래스 상수):

- `HOMING_DURATION_S = 5.0` — A 버튼 후 초기 자세까지의 보간 시간.
  2배(10초)를 넘겨도 도착하지 못하면 에러를 내고 정지한다.

---

## 6. 임계값 (threshold / limit)

### CLI 로 조절 가능

| 파라미터 | 기본값 | 의미 |
|---|---|---|
| `--teleop.grip_threshold` | `0.5` | Grip 아날로그 값이 이보다 크면 추종. 손이 큰 사람은 낮추면 편하다 |
| `--teleop.orientation_scale` | `1.0` | 손 회전 대비 툴 회전 비율. `0.0` 이면 자세 완전 고정. 검증 범위는 `[0, 1]` |
| `--teleop.ik_iterations` | `5` | 틱당 IKLM 반복 횟수. **늘린다고 항상 좋아지지 않는다** — 특이 자세에서는 오히려 나빠진다 |
| `--robot.speed_bar` | `0.3` | 전역 속도바. 단 `--robot.set_speed_bar_on_connect=true` 일 때만 적용 |
| `--robot.gripper_invert` | `False` | 그리퍼 열림/닫힘 방향 반전 |

`orientation_scale` 참고값:

| 값 | 동작 |
|---|---|
| `0.0` | EE 자세를 앵커 자세로 완전히 고정 (이 기능 도입 전과 동일) |
| `0.1` | 실기 첫 시험용 — 회전 방향 확인 |
| `0.3` | 손을 90° 돌리면 툴은 27° |
| `1.0` | **기본값.** 1:1, 위치 매핑과 동일 |

### CLI 로 바꿀 수 없는 안전 임계값

모두 `rb_vr.py` 의 클래스 상수다. 안전 한계라 CLI 로 열어두지 않았다.
바꾸려면 소스를 고쳐야 한다.

| 상수 | 값 | 역할 |
|---|---|---|
| `GRIP_RELEASE_HYSTERESIS` | `0.1` | 누름은 `0.5` 초과, 뗌은 `0.4` 이하. 손이 경계에 걸쳐 매 틱 재앵커되는 채터링을 막는다 |
| `MAX_POSITION_DELTA_MM` | `700.0` | 앵커 이후 손 이동량 상한. 축별이 아니라 **norm 을 스케일**해서 명령 방향을 보존한다 |
| `IK_RESIDUAL_LIMIT_MM` | `50.0` | IK 위치 잔차가 이보다 크면 미수렴으로 보고 직전 명령을 유지한다 (작업공간 밖). **위치 행만** 본다 |
| `MAX_JOINT_STEP_RAD` | `0.05` | 틱당 관절 변화량 상한 (≈2.9°/tick, 30Hz 에서 ≈86°/s). 체인 전체에서 **유일한** rate limit 이라 가장 중요하다 |
| `FIRST_COMMAND_TOLERANCE_RAD` | `1°` | 상태 전이 직후 명령이 튀지 않았는지 검사하는 허용 오차. zero-delta IK self-check 도 이 값을 쓴다 |

---

## 7. Chunk / 파일 분할

**결론부터: `lerobot-record` CLI 로는 바꿀 수 없다.**
`DatasetRecordConfig` 에 해당 필드가 없고, `lerobot_record.py:438` 의
`LeRobotDataset.create()` 호출도 이 인자들을 넘기지 않는다.
`lerobot/src/lerobot/datasets/utils.py:78-80` 의 기본값이 그대로 박힌다.

| 상수 | 기본값 | 의미 |
|---|---|---|
| `DEFAULT_CHUNK_SIZE` | `1000` | 청크 디렉터리(`chunk-000/`) 하나에 들어갈 **파일 수** 상한. 한 디렉터리에 수만 개 파일이 쌓이는 것을 피하기 위한 값 |
| `DEFAULT_DATA_FILE_SIZE_IN_MB` | `100` | parquet 데이터 파일 하나의 크기 상한. 넘으면 다음 파일 인덱스로 넘어가고, 파일 인덱스가 1000 을 넘으면 다음 청크로 넘어간다 |
| `DEFAULT_VIDEO_FILE_SIZE_IN_MB` | `200` | MP4 비디오 파일 하나의 크기 상한 |

즉 **에피소드 하나 = 파일 하나가 아니다.** 여러 에피소드가 크기가 찰 때까지 한 파일에
이어 붙는다. 20 에피소드 정도면 청크는 하나로 끝난다.

바꾸려면 `LeRobotDataset.create(..., data_files_size_in_mb=..., video_files_size_in_mb=...)`
를 직접 호출하는 스크립트를 쓰거나, 기록 후 `dataset_metadata.py:679` 의 메서드로
재분할해야 한다.

CLI 로 조절 가능한 유일한 인접 항목:

| 파라미터 | 기본값 | 의미 |
|---|---|---|
| `--dataset.video_encoding_batch_size` | `1` | 몇 에피소드를 모아서 비디오 인코딩할지. `1` 은 즉시 인코딩이라 에피소드마다 멈칫하고, 올리면 저장은 빨라지지만 중간에 죽으면 그만큼 날린다 |

---

## 8. 제어 주기 / 속도

| 파라미터 | 기본값 | 의미 |
|---|---|---|
| `--dataset.fps` | `30` | 기록 주기이자 제어 루프 주기 |
| `--robot.control_rate_hz` | `35.0` | ServoJ 파라미터 계산 기준. `servo_t1 = 1/35`, `servo_t2 = 3×t1` |
| `--robot.data_request_hz` | `500.0` | 백그라운드 상태 수신 주기. 최신 상태만 유지하므로 큐가 쌓이지 않는다 |
| `--robot.servo_hold_multiplier` | `3.0` | `t2 = 이 값 × t1`. 다음 명령이 늦어도 팔이 버티는 시간 |
| `--robot.servo_gain` | `1.0` | ServoJ 게인 |
| `--robot.servo_alpha` | `0.1` | ServoJ 스무딩 계수. `(0, 1)` 범위 검증 |

> **`fps` 와 `control_rate_hz` 는 반드시 맞춘다.**
> `--robot.control_rate_hz=30` 을 함께 주지 않으면 ServoJ 가 35Hz 기준으로 계산된 채
> 30Hz 로 돌아간다.

---

## 9. 저장 / 인코딩 성능

| 파라미터 | 기본값 | 의미 |
|---|---|---|
| `--dataset.streaming_encoding` | `False` | `True` 면 PNG 를 거치지 않고 실시간 인코딩 → `save_episode()` 가 거의 즉시 끝난다. 실행할 때마다 켜라는 로그가 뜬다 |
| `--dataset.encoder_threads` | `None` | 인코더당 스레드 수. 스트리밍 인코딩을 켤 때 `2` 정도 권장 |
| `--dataset.encoder_queue_maxsize` | `30` | 스트리밍 인코딩 시 카메라당 버퍼 프레임 수 (30fps 에서 약 1초) |
| `--dataset.num_image_writer_threads_per_camera` | `4` | 카메라당 PNG 기록 스레드. 너무 많으면 메인 스레드를 막아 fps 가 불안정해진다 |
| `--dataset.num_image_writer_processes` | `0` | `0` 이면 스레드만 사용. fps 가 계속 불안정하면 `1` 이상 |
| `--dataset.video` | `True` | `False` 면 MP4 대신 PNG 로 저장 |
| `--dataset.rgb_encoder.vcodec` | `libsvtav1` | 코덱. `"auto"` 면 하드웨어 코덱을 찾고 없으면 libsvtav1 |
| `--dataset.rgb_encoder.crf` | `30` | 화질. 낮을수록 고화질/대용량 |
| `--dataset.rgb_encoder.g` | `2` | GOP(키프레임 간격). 랜덤 액세스 학습을 위해 매우 짧게 잡혀 있다 |

카메라 2대 + 30fps 에서 fps 가 흔들리면 이 순서로 건드리는 것이 효과적이다.

```bash
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2
```

그래도 불안정하면 `--dataset.num_image_writer_threads_per_camera` 를 조정한다.

---

## 10. 실행 편의 (`RecordConfig` 최상위)

`--robot.*` / `--teleop.*` / `--dataset.*` 접두사가 없는 최상위 인자다.

| 파라미터 | 기본값 | 의미 |
|---|---|---|
| `--display_data` | `False` | 카메라 화면 실시간 표시. **디버깅에 가장 유용하다** |
| `--display_mode` | `"rerun"` | `"rerun"` 또는 `"foxglove"` |
| `--display_ip` / `--display_port` | `None` | rerun 은 원격 서버 주소, foxglove 는 WebSocket 바인드 주소/포트 |
| `--display_compressed_images` | `False` | 원본 대신 JPEG 로 표시 (원격 접속 시 대역폭 절감) |
| `--play_sounds` | `True` | "Recording episode 3" 등을 음성으로 읽어준다. VR 헤드셋을 쓰면 화면을 볼 수 없으므로 **켜두는 것이 실용적이다** |
| `--resume` | `False` | 기존 데이터셋에 이어서 기록. `True` 면 `repo_id` 에 타임스탬프를 새로 붙이지 않고 기존 것을 쓴다 |
| `--dataset.root` | `None` | 저장 경로. 기본은 `$HF_LEROBOT_HOME/repo_id` |
| `--dataset.private` | `None` | Hub 업로드 시 비공개 여부 |
| `--dataset.tags` | `None` | Hub 데이터셋 태그 |

---

## 11. 키보드 조작 (녹화 중)

파라미터는 아니지만 실제로 가장 자주 쓴다 (`lerobot/src/lerobot/utils/keyboard_input.py:429`).

| 키 | 동작 |
|---|---|
| `→` 또는 `n` | 현재 에피소드를 타이머보다 일찍 종료하고 다음으로 |
| `←` 또는 `r` | 현재 에피소드를 버리고 재녹화 |
| `Esc` 또는 `q` | 전체 녹화 종료 (지금까지 기록한 것은 저장된다) |

문자 키(`n` / `r` / `q`)가 함께 있는 이유는 SSH/VNC 환경에서 화살표 이스케이프 시퀀스가
잘리거나 지연될 수 있기 때문이다.

> Wayland 나 헤드리스 SSH 에서는 `pynput` 이 캡처하지 못해 TTY 백엔드로 넘어가고,
> 그마저 안 되면 **키 입력 없이 타이머와 Ctrl+C 에만 의존**하게 된다.
> 이 경우 `episode_time_s` 를 현실적으로 잡아두는 것이 중요하다.

---

## 12. 자주 쓰는 변형 조합

카메라 없이 (로봇 + Quest 만 점검):

```bash
  --robot.cameras='{}'
```

그리퍼 없이:

```bash
  --robot.gripper_type=none \
  --teleop.use_gripper=false
```

제어 주기를 `fps` 와 맞추기:

```bash
  --robot.control_rate_hz=30
```

자세 추종 첫 실기 시험:

```bash
  --teleop.orientation_scale=0.1
```

짧은 태스크용 시간 단축:

```bash
  --dataset.episode_time_s=20 \
  --dataset.reset_time_s=10
```

저장 성능 개선:

```bash
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2
```

화면 보면서 디버깅:

```bash
  --display_data=true
```
