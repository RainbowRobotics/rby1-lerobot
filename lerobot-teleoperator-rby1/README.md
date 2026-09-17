# lerobot_teleoperator_rby1

LeRobot teleoperator plugins for the Rainbow Robotics RB-Y1:

| `--teleop.type` | Class | Input device |
|---|---|---|
| `rby1_leader_arm` | `Rby1LeaderArm` | RB-Y1 leader (master) arm, joint-space actions |
| `rby1_vr` | `Rby1VR` | Meta Quest app streaming controller JSON over UDP, EE-space actions |
| `rby1_isaac` | `Rby1XRTeleop` | NVIDIA Isaac Teleop (CloudXR + headset browser client): controllers → arm EE poses, headset → head joints, body tracking → torso |

```bash
pip install -e lerobot-teleoperator-rby1            # leader arm / Quest UDP
pip install -e "lerobot-teleoperator-rby1[isaac]"   # + NVIDIA Isaac Teleop device
```

See the repository root `README.md` for hardware setup, the full CLI examples and
the configuration tables. The Isaac Teleop device lives in
`lerobot_teleoperator_rby1/isaac_teleop/` and mirrors the layout of the upstream
LeRobot example `examples/isaac_teleop_to_so101`.

Tests (no hardware, no `isaacteleop` needed):

```bash
pip install -e "lerobot-teleoperator-rby1[test]"
pytest lerobot-teleoperator-rby1/tests
```
