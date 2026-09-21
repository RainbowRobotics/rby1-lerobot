"""State-machine and control-law tests for RbVr.

Everything here runs without a robot and without a headset: the RB follower
and the Quest receiver are replaced by fakes, and time.monotonic is driven
by the test.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lerobot_teleoperator_rb import rb_vr as rb_vr_module
from lerobot_teleoperator_rb import RbVr, RbVrConfig
from lerobot_teleoperator_rb.constants import VR_HOME_POSE_RAD
from lerobot_teleoperator_rb.frame_transforms import (
    R_OPERATOR_TO_BASE,
    head_pose_to_torso,
)
from lerobot_teleoperator_rb.rb10e_kinematics import RB10E
from lerobot_teleoperator_rb.vr_receiver import (
    VRButtonEvents,
    VRButtons,
    VRControllerState,
    VRHeadState,
    VRState,
)

_ControlState = rb_vr_module._ControlState

START_Q_DEG = (30.0, 5.0, -95.0, 45.0, -85.0, 10.0)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeRobot:
    def __init__(self, q_rad):
        self.measured_q = np.asarray(q_rad, dtype=np.float64)
        self.is_connected = True
        self.servo_enabled = False
        self.enable_calls = 0
        self.disable_calls = 0

    def get_joint_positions(self, *, measured=True):
        return self.measured_q.copy()

    def enable_servo_commands(self):
        self.servo_enabled = True
        self.enable_calls += 1

    def disable_servo_commands(self):
        self.servo_enabled = False
        self.disable_calls += 1


class FakeReceiver:
    def __init__(self):
        self.state = VRState(
            right=VRControllerState(
                pose_rb=np.eye(4),
                buttons=VRButtons(),
                tracked=True,
            ),
            left=VRControllerState(),
            head=VRHeadState(pose_rb=np.eye(4), tracked=True),
        )
        self.state.packet_monotonic_time = 1.0
        self.events = VRButtonEvents()
        self.stale = False

    # -- API used by RbVr ---------------------------------------------
    def get_state(self, *, require_fresh=False):
        return self.state

    def consume_button_events(self):
        events = self.events
        self.events = VRButtonEvents()
        return events

    def is_stale(self, *, state=None):
        return self.stale

    def stop(self):
        pass

    # -- test helpers --------------------------------------------------
    def set_controller_position_mm(self, position):
        pose = np.eye(4)
        pose[:3, :3] = self.state.right.pose_rb[:3, :3]
        pose[:3, 3] = position
        self.state.right.pose_rb = pose

    def set_controller_rotation(self, rotation):
        # Mirror image of set_controller_position_mm: preserve the other
        # half of the pose, and assign a FRESH array rather than mutating
        # in place. _begin_following copies pose_rb when it anchors, but an
        # in-place setter would turn a future dropped .copy() into a silent
        # aliasing bug that no test would catch.
        pose = np.eye(4)
        pose[:3, :3] = np.asarray(rotation, dtype=np.float64)
        pose[:3, 3] = self.state.right.pose_rb[:3, 3]
        self.state.right.pose_rb = pose

    def set_grip(self, level):
        self.state.right.buttons.grip = float(level)

    def press_primary(self):
        self.events.right_primary = True

    def press_secondary(self):
        self.events.right_secondary = True


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(rb_vr_module.time, "monotonic", fake)
    return fake


@pytest.fixture
def robot():
    return FakeRobot(np.deg2rad(START_Q_DEG))


@pytest.fixture
def teleop(monkeypatch, robot, clock):
    monkeypatch.setattr(
        rb_vr_module,
        "get_active_rb_cobot",
        lambda: robot,
    )

    config = RbVrConfig(
        id="test_rb_vr",
        send_handshake=False,
    )
    instance = RbVr(config)

    receiver = FakeReceiver()
    instance._receiver = receiver
    instance._kinematics = RB10E()
    instance._robot = None
    instance._is_connected = True

    instance.receiver = receiver
    return instance


def q_of(action):
    return np.array(
        [action[f"joint_{index}"] for index in range(6)],
        dtype=np.float64,
    )


TICK_S = 1.0 / 30.0


def run_ticks(teleop, clock, count, *, tick_s=TICK_S):
    """Advance the loop at the real control rate.

    The per-tick joint-step clamp is deliberately tight, so a test that
    jumps seconds or hundreds of millimetres in one call measures the
    clamp rather than the control law.
    """
    action = None
    for _ in range(count):
        clock.advance(tick_s)
        action = teleop.get_action()
    return action


def home_the_arm(teleop, clock):
    """Press A and let the ramp finish. Leaves the state machine in IDLE."""
    teleop.receiver.press_primary()
    teleop.get_action()

    run_ticks(teleop, clock, int(RbVr.HOMING_DURATION_S / TICK_S) + 2)

    assert teleop._control_state is _ControlState.IDLE


# ---------------------------------------------------------------------------
# Homing
# ---------------------------------------------------------------------------


def test_primary_button_starts_homing_without_jumping(teleop, robot, clock):
    teleop.receiver.press_primary()

    action = teleop.get_action()

    assert teleop._control_state is _ControlState.HOMING
    assert robot.servo_enabled
    assert np.allclose(
        q_of(action),
        robot.measured_q,
        atol=1e-12,
    )


def test_homing_ramp_reaches_the_fixed_home_pose(teleop, robot, clock):
    teleop.receiver.press_primary()
    teleop.get_action()

    q_start = robot.measured_q.copy()
    q_end = np.asarray(VR_HOME_POSE_RAD)

    half = int((RbVr.HOMING_DURATION_S / 2.0) / TICK_S)
    midpoint = q_of(run_ticks(teleop, clock, half))

    assert np.allclose(
        midpoint,
        q_start + (q_end - q_start) * 0.5,
        atol=1e-2,
    )

    # Capture the last command emitted while still homing. The IDLE ticks
    # that follow report the robot's measured pose, and FakeRobot does not
    # move in response to commands.
    final = None
    for _ in range(half + 10):
        if teleop._control_state is not _ControlState.HOMING:
            break
        clock.advance(TICK_S)
        final = q_of(teleop.get_action())

    assert final is not None
    assert np.allclose(final, q_end, atol=1e-9)
    assert teleop._control_state is _ControlState.IDLE
    assert not robot.servo_enabled


def test_grip_held_through_homing_does_not_start_following(
    teleop, robot, clock
):
    teleop.receiver.set_grip(1.0)
    teleop.receiver.press_primary()
    teleop.get_action()

    for _ in range(5):
        clock.advance(RbVr.HOMING_DURATION_S / 4.0)
        teleop.get_action()

    assert teleop._control_state is _ControlState.IDLE
    assert teleop._anchor_ee_pose is None
    assert not robot.servo_enabled

    # Releasing and pressing again does engage following.
    teleop.receiver.set_grip(0.0)
    teleop.get_action()

    teleop.receiver.set_grip(1.0)
    teleop.get_action()

    assert teleop._control_state is _ControlState.FOLLOWING


def test_primary_button_is_ignored_while_homing(teleop, clock):
    teleop.receiver.press_primary()
    teleop.get_action()

    started_at = teleop._homing_time

    clock.advance(1.0)
    teleop.receiver.press_primary()
    teleop.get_action()

    assert teleop._control_state is _ControlState.HOMING
    assert teleop._homing_time == started_at


def test_secondary_button_aborts_homing(teleop, robot, clock):
    teleop.receiver.press_primary()
    teleop.get_action()

    teleop.receiver.press_secondary()
    teleop.get_action()

    assert teleop._control_state is _ControlState.IDLE
    assert teleop._initialized is False
    assert not robot.servo_enabled


# ---------------------------------------------------------------------------
# Clutch anchor and delta following
# ---------------------------------------------------------------------------


def test_grip_press_anchors_without_moving(teleop, robot, clock):
    home_the_arm(teleop, clock)

    teleop.receiver.set_grip(1.0)
    action = teleop.get_action()

    assert teleop._control_state is _ControlState.FOLLOWING
    assert robot.servo_enabled
    assert np.allclose(
        q_of(action),
        robot.measured_q,
        atol=RbVr.FIRST_COMMAND_TOLERANCE_RAD,
    )


def test_following_applies_a_one_to_one_position_delta(teleop, clock):
    home_the_arm(teleop, clock)

    teleop.receiver.set_grip(1.0)
    teleop.get_action()

    anchor_ee = teleop._anchor_ee_pose.copy()
    torso = head_pose_to_torso(teleop.receiver.state.head.pose_rb)

    delta_controller = np.array([100.0, 0.0, 0.0])
    teleop.receiver.set_controller_position_mm(delta_controller)

    action = run_ticks(teleop, clock, 200)

    kinematics = RB10E()
    kinematics.set_q(q_of(action))
    reached = kinematics.get_fk()

    expected_delta = R_OPERATOR_TO_BASE @ (torso[:3, :3].T @ delta_controller)

    assert np.allclose(
        reached[:3, 3],
        anchor_ee[:3, 3] + expected_delta,
        atol=1.0,
    )

    # 1:1 means the travelled distance equals the hand's travelled distance.
    assert np.isclose(
        np.linalg.norm(reached[:3, 3] - anchor_ee[:3, 3]),
        np.linalg.norm(delta_controller),
        atol=1.0,
    )


def test_pure_translation_keeps_the_anchor_orientation(teleop, clock):
    """Holds at ANY orientation_scale, including the 0.3 default.

    set_controller_position_mm never touches the controller rotation, so
    the rotation delta is the identity throughout and the orientation is
    held by the control law rather than by the gain being zero. That makes
    this a live check that translation does not leak into orientation.
    """
    home_the_arm(teleop, clock)

    teleop.receiver.set_grip(1.0)
    teleop.get_action()

    anchor_ee = teleop._anchor_ee_pose.copy()

    teleop.receiver.set_controller_position_mm([120.0, -80.0, 60.0])
    action = run_ticks(teleop, clock, 200)

    kinematics = RB10E()
    kinematics.set_q(q_of(action))

    assert np.allclose(
        kinematics.get_fk()[:3, :3],
        anchor_ee[:3, :3],
        atol=1e-3,
    )


def test_position_scale_config_has_no_effect(monkeypatch, robot, clock):
    """1:1 is enforced by the absence of a knob, not by a default of 1.0."""

    def build(position_scale):
        monkeypatch.setattr(
            rb_vr_module,
            "get_active_rb_cobot",
            lambda: robot,
        )
        instance = RbVr(
            RbVrConfig(
                id="scaled",
                send_handshake=False,
                position_scale=position_scale,
            )
        )
        instance._receiver = FakeReceiver()
        instance._kinematics = RB10E()
        instance._is_connected = True
        instance.receiver = instance._receiver

        home_the_arm(instance, clock)

        instance.receiver.set_grip(1.0)
        instance.get_action()

        instance.receiver.set_controller_position_mm([150.0, 0.0, 0.0])
        return q_of(run_ticks(instance, clock, 200))

    baseline = build(1.0)

    clock.advance(1.0)
    scaled = build(3.0)

    assert np.allclose(baseline, scaled, atol=1e-12)


def test_release_and_repress_re_anchors_without_jumping(teleop, robot, clock):
    home_the_arm(teleop, clock)

    teleop.receiver.set_grip(1.0)
    teleop.get_action()

    teleop.receiver.set_controller_position_mm([200.0, 0.0, 0.0])
    moved = q_of(run_ticks(teleop, clock, 200))

    assert not np.allclose(moved, robot.measured_q, atol=1e-6)

    # Release. The follower reports the physical arm again.
    teleop.receiver.set_grip(0.0)
    teleop.get_action()

    assert teleop._control_state is _ControlState.IDLE
    assert teleop._anchor_ee_pose is None

    # Move the hand a long way while the clutch is out.
    teleop.receiver.set_controller_position_mm([700.0, 0.0, 0.0])
    teleop.get_action()

    # Re-press: the new anchor is the current hand and arm, so no motion.
    teleop.receiver.set_grip(1.0)
    action = teleop.get_action()

    assert np.allclose(
        q_of(action),
        robot.measured_q,
        atol=RbVr.FIRST_COMMAND_TOLERANCE_RAD,
    )


def test_head_rotation_during_a_stroke_does_not_move_the_arm(teleop, clock):
    """The torso frame is frozen at grip press."""
    home_the_arm(teleop, clock)

    teleop.receiver.set_controller_position_mm([50.0, 0.0, 0.0])
    teleop.receiver.set_grip(1.0)
    teleop.get_action()

    before = q_of(run_ticks(teleop, clock, 50))

    # Yaw the head 90 degrees with a perfectly still hand.
    yaw = np.eye(4)
    yaw[:3, :3] = [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    teleop.receiver.state.head.pose_rb = yaw

    after = q_of(run_ticks(teleop, clock, 50))

    assert np.allclose(before, after, atol=1e-9)


# ---------------------------------------------------------------------------
# Grip level handling
# ---------------------------------------------------------------------------


def test_grip_hysteresis(teleop, clock):
    home_the_arm(teleop, clock)

    teleop.receiver.set_grip(0.55)
    teleop.get_action()
    assert teleop._grip_pressed

    teleop.receiver.set_grip(0.45)
    teleop.get_action()
    assert teleop._grip_pressed

    teleop.receiver.set_grip(0.35)
    teleop.get_action()
    assert not teleop._grip_pressed


# ---------------------------------------------------------------------------
# Tracking loss
# ---------------------------------------------------------------------------


def test_tracking_loss_drops_the_anchor_and_blocks_resume(
    teleop, robot, clock
):
    home_the_arm(teleop, clock)

    teleop.receiver.set_grip(1.0)
    teleop.get_action()

    assert teleop._anchor_ee_pose is not None

    teleop.receiver.stale = True
    teleop.get_action()

    assert teleop._control_state is _ControlState.IDLE
    assert teleop._anchor_ee_pose is None
    assert not robot.servo_enabled

    # Tracking returns with grip still held: no rising edge, no resume.
    teleop.receiver.stale = False
    teleop.get_action()

    assert teleop._control_state is _ControlState.IDLE
    assert teleop._anchor_ee_pose is None


# ---------------------------------------------------------------------------
# Safety limiters
# ---------------------------------------------------------------------------


def test_joint_step_is_rate_limited(teleop, clock):
    home_the_arm(teleop, clock)

    teleop.receiver.set_grip(1.0)
    previous = q_of(teleop.get_action())

    # Teleport the controller two metres in a single tick.
    teleop.receiver.set_controller_position_mm([2000.0, 0.0, 0.0])
    current = q_of(teleop.get_action())

    assert np.max(np.abs(current - previous)) <= (
        RbVr.MAX_JOINT_STEP_RAD + 1e-12
    )


def test_position_delta_is_clamped_to_the_limit(teleop, clock):
    home_the_arm(teleop, clock)

    over_limit = np.array([3000.0, 0.0, 0.0])
    clamped = teleop._limit_position_delta(over_limit)

    assert np.isclose(
        np.linalg.norm(clamped),
        RbVr.MAX_POSITION_DELTA_MM,
    )
    # Direction preserved.
    assert np.allclose(
        clamped / np.linalg.norm(clamped),
        over_limit / np.linalg.norm(over_limit),
    )


def test_following_without_an_anchor_raises(teleop, clock):
    home_the_arm(teleop, clock)

    teleop._control_state = _ControlState.FOLLOWING
    teleop._clear_anchor()

    with pytest.raises(RuntimeError):
        teleop._following_action(teleop.receiver.state.right)


# ---------------------------------------------------------------------------
# Operator reference frame (captured on A, not on Grip)
# ---------------------------------------------------------------------------


def yaw_pose(angle_rad):
    pose = np.eye(4)
    cos = np.cos(angle_rad)
    sin = np.sin(angle_rad)
    pose[:3, :3] = [
        [cos, -sin, 0.0],
        [sin, cos, 0.0],
        [0.0, 0.0, 1.0],
    ]
    return pose


def test_primary_button_captures_the_operator_frame(teleop, clock):
    assert teleop._operator_torso_pose is None

    teleop.receiver.state.head.pose_rb = yaw_pose(0.7)
    teleop.receiver.press_primary()
    teleop.get_action()

    assert np.allclose(
        teleop._operator_torso_pose,
        head_pose_to_torso(yaw_pose(0.7)),
    )


def test_operator_frame_survives_grip_release(teleop, clock):
    home_the_arm(teleop, clock)

    captured = teleop._operator_torso_pose.copy()

    teleop.receiver.set_grip(1.0)
    teleop.get_action()
    teleop.receiver.set_grip(0.0)
    teleop.get_action()

    assert teleop._anchor_ee_pose is None
    assert np.allclose(teleop._operator_torso_pose, captured)


def test_head_rotation_between_strokes_does_not_change_the_mapping(
    teleop, clock
):
    """The axis mapping is fixed by A, not by where the head was at Grip.

    Capturing per stroke would silently rotate the mapping whenever the
    operator turned between strokes.
    """
    home_the_arm(teleop, clock)

    def stroke():
        teleop.receiver.set_controller_position_mm([0.0, 0.0, 0.0])
        teleop.receiver.set_grip(1.0)
        teleop.get_action()

        teleop.receiver.set_controller_position_mm([120.0, 0.0, 0.0])
        q_out = q_of(run_ticks(teleop, clock, 200))

        teleop.receiver.set_grip(0.0)
        teleop.get_action()
        return q_out

    first = stroke()

    # Operator turns 90 degrees between strokes.
    teleop.receiver.state.head.pose_rb = yaw_pose(np.pi / 2.0)

    second = stroke()

    assert np.allclose(first, second, atol=1e-6)


def test_primary_button_rezeroes_the_operator_frame(teleop, clock):
    home_the_arm(teleop, clock)

    before = teleop._operator_torso_pose.copy()

    teleop.receiver.state.head.pose_rb = yaw_pose(np.pi / 2.0)
    teleop.receiver.press_primary()
    teleop.get_action()

    assert not np.allclose(teleop._operator_torso_pose, before)
    assert np.allclose(
        teleop._operator_torso_pose,
        head_pose_to_torso(yaw_pose(np.pi / 2.0)),
    )


def test_secondary_button_drops_the_operator_frame(teleop, clock):
    home_the_arm(teleop, clock)

    assert teleop._operator_torso_pose is not None

    teleop.receiver.press_secondary()
    teleop.get_action()

    assert teleop._operator_torso_pose is None


def test_homing_without_tracking_keeps_the_previous_operator_frame(
    teleop, clock
):
    home_the_arm(teleop, clock)

    captured = teleop._operator_torso_pose.copy()

    teleop.receiver.stale = True
    teleop.receiver.state.head.pose_rb = yaw_pose(np.pi / 2.0)
    teleop.receiver.press_primary()
    teleop.get_action()

    assert teleop._control_state is _ControlState.HOMING
    assert np.allclose(teleop._operator_torso_pose, captured)


# ---------------------------------------------------------------------------
# Scaled orientation following
# ---------------------------------------------------------------------------


def rotation_matrix(axis, angle_rad):
    return Rotation.from_rotvec(
        angle_rad * np.asarray(axis, dtype=np.float64)
    ).as_matrix()


def build_teleop(monkeypatch, robot, clock, **config_kwargs):
    """An RbVr with a non-default config, homed and ready to grip."""
    monkeypatch.setattr(
        rb_vr_module,
        "get_active_rb_cobot",
        lambda: robot,
    )
    instance = RbVr(
        RbVrConfig(
            id="oriented",
            send_handshake=False,
            **config_kwargs,
        )
    )
    instance._receiver = FakeReceiver()
    instance._kinematics = RB10E()
    instance._is_connected = True
    instance.receiver = instance._receiver

    home_the_arm(instance, clock)
    return instance


def achieved_rotation_delta(action, anchor_ee):
    kinematics = RB10E()
    kinematics.set_q(q_of(action))
    return kinematics.get_fk()[:3, :3] @ anchor_ee[:3, :3].T


def expected_base_axis(teleop, operator_axis):
    """Carry an operator-frame axis into the base frame.

    Same shape as the position tests' expected_delta: the frozen torso
    rotation, then R_OPERATOR_TO_BASE.
    """
    torso = teleop._operator_torso_pose
    return R_OPERATOR_TO_BASE @ (torso[:3, :3].T @ np.asarray(operator_axis))


def test_orientation_scale_zero_freezes_the_orientation(
    monkeypatch, robot, clock
):
    """The exact regression guard for the previous control law.

    Not covered by test_pure_translation_keeps_the_anchor_orientation: that
    test never rotates the controller, so it passes at any gain. This one
    rotates the hand 90 degrees and requires the tool not to move at all.
    """
    teleop = build_teleop(monkeypatch, robot, clock, orientation_scale=0.0)

    teleop.receiver.set_grip(1.0)
    teleop.get_action()

    anchor_ee = teleop._anchor_ee_pose.copy()

    teleop.receiver.set_controller_rotation(
        rotation_matrix([0.0, 0.0, 1.0], np.pi / 2.0)
    )
    action = run_ticks(teleop, clock, 200)

    kinematics = RB10E()
    kinematics.set_q(q_of(action))

    assert np.allclose(
        kinematics.get_fk()[:3, :3],
        anchor_ee[:3, :3],
        atol=1e-9,
    )


def test_following_applies_a_scaled_orientation_delta(teleop, clock):
    """A 30 degree hand rotation asks for 9 degrees of tool rotation."""
    home_the_arm(teleop, clock)

    teleop.receiver.set_grip(1.0)
    teleop.get_action()

    anchor_ee = teleop._anchor_ee_pose.copy()

    operator_axis = np.array([1.0, 0.0, 0.0])
    hand_turn = np.deg2rad(30.0)

    teleop.receiver.set_controller_rotation(
        rotation_matrix(operator_axis, hand_turn)
    )
    action = run_ticks(teleop, clock, 200)

    delta = achieved_rotation_delta(action, anchor_ee)
    rotvec = Rotation.from_matrix(delta).as_rotvec()

    gain = teleop._config.orientation_scale
    assert np.rad2deg(np.linalg.norm(rotvec)) == pytest.approx(
        np.rad2deg(gain * hand_turn), abs=0.05
    )

    expected_axis = expected_base_axis(teleop, operator_axis)
    assert np.allclose(
        rotvec / np.linalg.norm(rotvec),
        expected_axis / np.linalg.norm(expected_axis),
        atol=1e-3,
    )


def test_orientation_and_position_deltas_compose(teleop, clock):
    """Rotating must not disturb the 1:1 position law, or vice versa.

    This is the case that catches a pre/post-multiply error leaking
    translation into rotation. It needs more ticks than the pure-position
    tests: the orientation motion competes for the same per-tick joint-step
    budget, so the two together slew for longer.
    """
    home_the_arm(teleop, clock)

    teleop.receiver.set_grip(1.0)
    teleop.get_action()

    anchor_ee = teleop._anchor_ee_pose.copy()

    delta_controller = np.array([120.0, -80.0, 60.0])
    operator_axis = np.array([0.0, 0.0, 1.0])
    hand_turn = np.deg2rad(60.0)

    teleop.receiver.set_controller_position_mm(delta_controller)
    teleop.receiver.set_controller_rotation(
        rotation_matrix(operator_axis, hand_turn)
    )
    action = run_ticks(teleop, clock, 400)

    kinematics = RB10E()
    kinematics.set_q(q_of(action))
    reached = kinematics.get_fk()

    torso = teleop._operator_torso_pose
    expected_position = anchor_ee[:3, 3] + R_OPERATOR_TO_BASE @ (
        torso[:3, :3].T @ delta_controller
    )

    assert np.allclose(reached[:3, 3], expected_position, atol=1.0)

    gain = teleop._config.orientation_scale
    delta = reached[:3, :3] @ anchor_ee[:3, :3].T
    assert np.rad2deg(
        np.linalg.norm(Rotation.from_matrix(delta).as_rotvec())
    ) == pytest.approx(np.rad2deg(gain * hand_turn), abs=0.05)


def test_large_hand_rotations_stay_bounded(teleop, clock):
    """Without a clamp, gain is the only bound -- and that is enough.

    as_rotvec always reports the short way round, so the hand delta itself
    can never exceed 180 deg however far the operator keeps turning. The
    command therefore tops out at gain * 180 and cannot run away.

    Tracking is NOT exact out there. Past roughly 110 deg about the worst
    axis the elbow reaches full extension (joint 2 -> 0) and the IK settles
    a few degrees off the commanded orientation; running it harder makes it
    worse, not better, which is the signature of a singular configuration
    rather than an iteration shortage. Position stays converged throughout,
    so IK_RESIDUAL_LIMIT_MM cannot see it. That is the accepted cost of
    removing the orientation clamp, and this test pins the bound rather
    than pretending the tracking is exact.
    """
    home_the_arm(teleop, clock)

    teleop.receiver.set_grip(1.0)
    teleop.get_action()

    anchor_ee = teleop._anchor_ee_pose.copy()
    gain = teleop._config.orientation_scale

    for degrees in (150.0, 179.0, 181.0, 240.0):
        teleop.receiver.set_controller_rotation(
            rotation_matrix([0.0, 0.0, 1.0], np.deg2rad(degrees))
        )
        action = run_ticks(teleop, clock, 600)

        kinematics = RB10E()
        kinematics.set_q(q_of(action))
        reached = kinematics.get_fk()

        delta = reached[:3, :3] @ anchor_ee[:3, :3].T
        angle = np.rad2deg(
            np.linalg.norm(Rotation.from_matrix(delta).as_rotvec())
        )

        # The hand delta wraps to the short way round: 181 and 240 deg of
        # hand turn are 179 and 120 deg of delta.
        hand_delta = min(degrees, 360.0 - degrees)

        assert angle <= gain * 180.0 + 0.5
        assert angle == pytest.approx(gain * hand_delta, abs=6.0)

        # Whatever the orientation does, position must not drift.
        assert np.allclose(reached[:3, 3], anchor_ee[:3, 3], atol=0.1)


def test_orientation_tracks_exactly_below_the_elbow_limit(teleop, clock):
    """Up to ~110 deg the arm reaches the commanded orientation exactly.

    This is the range the operator actually works in, and it is where the
    removal of the clamp buys something: at gain 1.0 a 110 deg wrist turn
    now produces a 110 deg tool turn instead of being cut off at 45.
    """
    home_the_arm(teleop, clock)

    teleop.receiver.set_grip(1.0)
    teleop.get_action()

    anchor_ee = teleop._anchor_ee_pose.copy()
    gain = teleop._config.orientation_scale

    for degrees in (30.0, 60.0, 90.0, 110.0):
        teleop.receiver.set_controller_rotation(
            rotation_matrix([0.0, 0.0, 1.0], np.deg2rad(degrees))
        )
        action = run_ticks(teleop, clock, 600)

        delta = achieved_rotation_delta(action, anchor_ee)
        angle = np.rad2deg(
            np.linalg.norm(Rotation.from_matrix(delta).as_rotvec())
        )

        assert angle == pytest.approx(gain * degrees, abs=0.05)
