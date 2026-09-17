# lerobot_robot_rby1

LeRobot robot plugin for the Rainbow Robotics RB-Y1 (`--robot.type=rby1`).

- `action_mode="joint"`: joint-position / impedance actions for the enabled arms.
- `action_mode="ee"`: end-effector pose actions (`<group>_ee.{x,y,z,wx,wy,wz}`)
  executed by the onboard Cartesian impedance solver (used by the `rby1_vr` and
  `rby1_isaac` teleoperators).
- `use_head=True`: head pan / tilt (`head_0.pos`, `head_1.pos`) in observation and
  action, in either mode.

See the repository root `README.md` for installation, CLI examples and the full
configuration table.

```bash
pip install -e lerobot-robot-rby1
pytest lerobot-robot-rby1/tests   # needs lerobot importable; no hardware
```
