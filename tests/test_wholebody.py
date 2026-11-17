"""Whole-body mode: Rby1 action_mode="wholebody" and the rby1_vr_wholebody teleoperator."""

from __future__ import annotations

import math

import numpy as np
import pytest
import rby1_sdk as rby

ADDRESS = "stub.invalid:1"


# ── Config validation ───────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "overrides",
    [
        {"action_mode": "wholebody"},                         # use_torso defaults to False
        {"action_mode": "wholebody", "use_torso": True, "use_left_arm": False},
        {"action_mode": "ee", "use_torso": True, "use_head": True},
        {"action_mode": "wholebody", "use_torso": True, "ready_body_deg": [0.0] * 6},
    ],
)
def test_invalid_config(overrides: dict) -> None:
    from lerobot_robot_rby1 import Rby1Config

    with pytest.raises(ValueError):
        Rby1Config(address=ADDRESS, **overrides)


def _wb_config(**overrides):
    from lerobot_robot_rby1 import Rby1Config

    base = dict(address=ADDRESS, model="m", version="1.3", action_mode="wholebody", use_torso=True,
                use_head=True, use_mobile_base=True, use_gripper=False, move_to_ready_on_connect=False)
    base.update(overrides)
    return Rby1Config(**base)


def test_wholebody_features() -> None:
    from lerobot_robot_rby1 import Rby1

    robot = Rby1(_wb_config())
    action = list(robot.action_features)
    assert action[:18] == [f"{g}_ee.{s}" for g in ("torso", "right", "left") for s in ("x", "y", "z", "wx", "wy", "wz")]
    assert action[18:32] == [f"{side}_arm_{i}.null" for side in ("right", "left") for i in range(7)]
    assert action[32:] == ["head_0.pos", "head_1.pos", "x.vel", "y.vel", "theta.vel"]
    obs = list(robot.observation_features)
    assert "head_0.pos" in obs and "torso_ee.wz" in obs and "right_arm_6.pos" in obs
    assert robot.supports_position_hold is False


# ── Robot: one Cartesian impedance command over the whole body ──────────────
@pytest.fixture()
def wb_robot():
    from lerobot_robot_rby1 import Rby1

    robot = Rby1(_wb_config(mobile_base_linear_acceleration_limit=2.0,
                            mobile_base_angular_acceleration_limit=4.0))
    robot.connect()
    yield robot
    robot.disconnect()


def _cmd(robot):
    return robot._robot.last_stream_command.get("set_command")[0]


def _action_from_obs(robot) -> dict[str, float]:
    obs = robot.get_observation()
    action = {k: float(obs[k]) for k in robot.action_features if k in obs}
    for side in ("right", "left"):
        for i in range(7):
            action[f"{side}_arm_{i}.null"] = float(obs[f"{side}_arm_{i}.pos"]) + 0.01 * i
    action.update({"x.vel": 0.1, "y.vel": 0.0, "theta.vel": 0.2})
    return action


def test_wholebody_send_action(wb_robot) -> None:
    action = _action_from_obs(wb_robot)
    action["head_1.pos"] = 5.0  # beyond the head limit → clipped
    sent = wb_robot.send_action(action)
    assert set(sent) == set(wb_robot.action_features)
    assert sent["head_1.pos"] == rby.JOINT_Q_LIMITS["head_1"][1]

    cbc = _cmd(wb_robot)
    body = cbc.get("set_body_command")[0]
    assert type(body).__name__ == "CartesianImpedanceControlCommandBuilder"
    links = [args[1] for name, args in body._calls if name == "add_target"]
    assert links == ["link_torso_5", "link_right_arm_6", "link_left_arm_6"]
    target, weight, kp, kd = body.get("set_nullspace_joint_target")
    assert list(weight[:6]) == [0.0] * 6 and list(weight[6:13]) == [2.0, 2.0, 2.0, 2.0, 0.5, 0.5, 0.5]
    assert np.allclose(target[6:13], [action[f"right_arm_{i}.null"] for i in range(7)])
    assert (kp, kd) == (0.2, 0.2)
    assert body.get("set_reset_reference") == (True,)  # first command re-references

    head = cbc.get("set_head_command")[0]
    assert list(head.get("set_position")[0]) == [action["head_0.pos"], rby.JOINT_Q_LIMITS["head_1"][1]]
    mobility = cbc.get("set_mobility_command")[0]
    lin, ang = mobility.get("set_acceleration_limit")
    assert list(lin) == [2.0, 2.0] and ang == 4.0

    wb_robot.send_action(action)
    assert _cmd(wb_robot).get("set_body_command")[0].get("set_reset_reference") == (False,)


def test_wholebody_hold_before_first_action(wb_robot) -> None:
    wb_robot.hold()  # obs has no .null keys: they come from the measured arm joints
    body = _cmd(wb_robot).get("set_body_command")[0]
    assert body.get("set_nullspace_joint_target") is not None


def test_ready_pose_override() -> None:
    from lerobot_robot_rby1 import Rby1

    torso = [0.0, 45.0, -80.0, 45.0, 0.0, 0.0]
    right = [0.0, -15.0, 0.0, -120.0, 0.0, 30.0, 0.0]
    left = [0.0, 15.0, 0.0, -120.0, 0.0, 30.0, 0.0]
    robot = Rby1(_wb_config(ready_body_deg=torso + right + left, ready_head_deg=[0.0, 30.0],
                            move_to_ready_on_connect=True, use_mobile_base=False))
    robot.connect()
    try:
        obs = robot.get_observation()
        assert math.isclose(obs["torso_2.pos"], math.radians(-80.0))
        assert math.isclose(obs["left_arm_3.pos"], math.radians(-120.0))
        assert math.isclose(obs["head_1.pos"], math.radians(30.0))
    finally:
        robot.disconnect()


# ── Retargeter on synthetic operator frames ─────────────────────────────────
def _frame(dx: float = 0.0, pinch: float = 0.06, t: float = 0.0) -> dict:
    """One NDJSON frame: operator standing upright facing +x, shifted by dx."""
    def joint(i, p, rot=None):
        j = {"i": i, "valid": True, "position": [p[0] + dx, p[1], p[2]]}
        if rot is not None:
            j["rotation"] = rot
        return j

    body = [joint(1, (0.0, 0.0, 1.0), [0.0, 0.0, 0.0, 1.0])]
    for side, y in (("left", 0.2), ("right", -0.2)):
        s, e, w = (10, 11, 12) if side == "left" else (15, 16, 17)
        body += [joint(s, (0.0, y, 1.45)), joint(e, (0.05, y, 1.20)), joint(w, (0.30, y, 1.15))]
    hands = []
    for side, y in (("left", 0.2), ("right", -0.2)):
        lm = {0: (0.30, y, 1.15), 5: (0.38, y + 0.02, 1.12), 9: (0.39, y, 1.10),
              17: (0.37, y - 0.03, 1.11), 4: (0.36, y + 0.04, 1.12)}
        lm[8] = (lm[4][0] + pinch, lm[4][1], lm[4][2])
        hands.append({
            "side": side, "tracked": True,
            "palm": {"valid": True, "position": [0.35 + dx, y, 1.12]},
            "landmarks": [{"i": i, "valid": True, "position": [p[0] + dx, p[1], p[2]]} for i, p in lm.items()],
        })
    return {"timestamp": t, "body": {"tracked": True, "fidelity": "high", "confidence": 0.9, "joints": body},
            "hands": hands,
            "hmd": {"tracked": True, "position": [dx, 0.0, 1.65], "rotation": [0.0, 0.0, 0.0, 1.0]}}


def _measured(odom=np.eye(3)):
    from lerobot_teleoperator_rby1.wb_kinematics import Rby1Arms
    from lerobot_teleoperator_rby1.wb_retarget import Measured, se3

    arms = Rby1Arms()
    model = rby.Model_M()
    q = np.zeros(model.robot_dof)
    q_arm = {"right": np.radians([0, -15, 0, -120, 0, 30, 0]), "left": np.radians([0, 15, 0, -120, 0, 30, 0])}
    q[model.right_arm_idx] = q_arm["right"]
    q[model.left_arm_idx] = q_arm["left"]
    q[model.head_idx] = [0.0, math.radians(30)]
    T_t5 = se3(np.eye(3), np.array([0.0, 0.0, 1.2]))
    T_t3 = se3(np.eye(3), np.array([0.0, 0.0, 0.9]))
    T_arm6 = {}
    for side in ("right", "left"):
        f = arms.arms[side].fk(q_arm[side])
        T_arm6[side] = T_t5 @ se3(f["R_ee"], f["W"])
    idx = {"right": np.array(model.right_arm_idx), "left": np.array(model.left_arm_idx)}
    return arms, idx, np.array(model.head_idx), Measured(q, T_t5, T_t3, T_arm6, odom, odom is not None)


def test_retargeter_latch_hold_walk_pinch() -> None:
    from lerobot_teleoperator_rby1.wb_retarget import Retargeter, Tuning
    from lerobot_teleoperator_rby1.wb_stream import FilterTaus, StreamReceiver

    arms, arm_idx, head_idx, meas = _measured()
    rx = StreamReceiver("unused", 0, FilterTaus())
    rx.apply(_frame(), now=10.0)
    rt = Retargeter(Tuning(head_mode="hmd"), arms, arm_idx, head_idx)
    rt.latch(rx.snapshot, meas, now=10.0)

    # Standing still right after the latch: robot torso and hands stay put.
    tg = rt.step(rx.snapshot, meas, now=10.0, hold=False, dt=0.02)
    assert np.allclose(tg.T_t5, meas.T_t5, atol=1e-6)
    for side in ("right", "left"):
        assert np.allclose(tg.T_hand[side][:3, 3], meas.T_arm6[side][:3, 3], atol=0.01)
        # The elbow swivel follows the operator's elbow at once, so the joints
        # change — but they must reproduce the commanded hand pose.
        f = arms.arms[side].fk(tg.q_ns[side])
        T_from_q = tg.T_t5[:3, :3] @ f["W"] + tg.T_t5[:3, 3]
        assert np.allclose(T_from_q, tg.T_hand[side][:3, 3], atol=1e-6)
    assert np.allclose(tg.base_v, 0.0) and tg.base_w == 0.0
    assert not tg.closed["right"]

    # The operator walks 20 cm forward and pinches: base drives +x, grippers close.
    t, now = 0.0, 10.0
    for k in range(10):
        t += 0.02
        now += 0.02
        rx.apply(_frame(dx=0.02 * (k + 1), pinch=0.005, t=t), now=now)
        tg = rt.step(rx.snapshot, meas, now=now, hold=False, dt=0.02)
    assert tg.base_v[0] > 0.1 and abs(tg.base_v[1]) < 0.05
    assert tg.closed["right"] and tg.closed["left"]

    # Hold: the base stops, the arms keep their target.
    tg_hold = rt.step(rx.snapshot, meas, now=now, hold=True, dt=0.02)
    assert np.allclose(tg_hold.base_v, 0.0) and tg_hold.base_w == 0.0


def test_retargeter_base_scale() -> None:
    """The base target is base_scale × the operator's pelvis displacement."""
    from lerobot_teleoperator_rby1.wb_retarget import Retargeter, Tuning
    from lerobot_teleoperator_rby1.wb_stream import FilterTaus, StreamReceiver

    arms, arm_idx, head_idx, meas = _measured()
    for scale in (0.6, 1.0):
        rx = StreamReceiver("unused", 0, FilterTaus())
        rx.apply(_frame(), now=10.0)
        rt = Retargeter(Tuning(base_scale=scale), arms, arm_idx, head_idx)
        rt.latch(rx.snapshot, meas, now=10.0)
        t, now = 0.0, 10.0
        for k in range(30):  # walk 0.2 m in 10 ticks, then stand so the input filter settles
            t += 0.02
            now += 0.02
            rx.apply(_frame(dx=0.02 * min(k + 1, 10), t=t), now=now)
            rt.step(rx.snapshot, meas, now=now, hold=False, dt=0.02)
        assert rt.p_t_last[0] == pytest.approx(scale * 0.2, abs=1e-3)
        assert abs(rt.p_t_last[1]) < 1e-6


def test_retargeter_head_gaze_follows_hand_and_toggles() -> None:
    from lerobot_teleoperator_rby1.wb_retarget import Retargeter, Tuning
    from lerobot_teleoperator_rby1.wb_stream import FilterTaus, StreamReceiver

    arms, arm_idx, head_idx, meas = _measured()
    rx = StreamReceiver("unused", 0, FilterTaus())
    rx.apply(_frame(), now=10.0)
    rt = Retargeter(Tuning(head_mode="gaze"), arms, arm_idx, head_idx)
    rt.latch(rx.snapshot, meas, now=10.0)
    assert rt.gaze_side == "left"
    for _ in range(100):  # 2 s, enough for the 2 rad/s slew
        tg = rt.step(rx.snapshot, meas, now=10.0, hold=False, dt=0.02)
    assert tg.info["gaze"] == "left" and tg.head[0] > 0.1  # left hand: +y → positive yaw
    assert tg.head[1] > 0.1                                 # hands below the head: look down
    assert rt.toggle_gaze() == "right"
    for _ in range(100):
        tg = rt.step(rx.snapshot, meas, now=10.0, hold=True, dt=0.02)  # keeps tracking during hold
    assert tg.head[0] < -0.1
    rt.latch(rx.snapshot, meas, now=10.0)
    assert rt.gaze_side == "left"  # every latch starts on the left hand


# ── Teleoperator → robot round trip on the fake SDK ─────────────────────────
def test_teleop_action_drives_wholebody_robot(monkeypatch) -> None:
    from lerobot_robot_rby1 import Rby1
    from lerobot_teleoperator_rby1 import Rby1VRWholebody, Rby1VRWholebodyConfig
    from lerobot_teleoperator_rby1 import rby1_vr_wholebody as mod

    monkeypatch.setattr(mod.StreamReceiver, "start", lambda self: None)  # no headset socket
    teleop = Rby1VRWholebody(Rby1VRWholebodyConfig(
        robot_address=ADDRESS, robot_model="m", ignore_model_check=True,  # the fake FK is not the URDF
        use_mobile_base=False, use_gripper=False, status_log_period_s=0.0,
    ))
    robot = Rby1(_wb_config(use_mobile_base=False))
    teleop.connect()
    robot.connect()
    try:
        teleop._rx.apply(_frame(), now=__import__("time").monotonic())
        teleop._try_latch(teleop._rx.snapshot, teleop._measure(), __import__("time").monotonic())
        assert teleop._state == "RUN"
        action = teleop.get_action()
        assert list(action) == list(teleop.action_features)
        assert set(action) == set(robot.action_features)
        sent = robot.send_action(action)
        assert set(sent) == set(robot.action_features)
    finally:
        robot.disconnect()
        teleop.disconnect()
