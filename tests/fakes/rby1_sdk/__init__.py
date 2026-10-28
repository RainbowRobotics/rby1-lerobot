"""Fake ``rby1_sdk`` for hardware-free testing of the RB-Y1 LeRobot plugins.

Prepend the directory that contains this package to PYTHONPATH. Nothing here
opens a socket or a serial port: ``create_robot*`` return an in-process fake
whose joint state follows the last joint-position command it received.

Mirrors the subset of the real rby1-sdk 0.10.0 API that lerobot_robot_rby1 /
lerobot_teleoperator_rby1 use (names checked against the real bindings).
"""

from __future__ import annotations

import enum
import logging
import os
import threading
import time

import numpy as np

__version__ = "0.0.0+fake"
FAKE = True
_log = logging.getLogger("rby1_sdk.fake")

# Every call into the fake robot is recorded here (name, args) so tests can
# assert on what the plugin sent.
CALL_LOG: list[tuple[str, tuple, dict]] = []
_ROBOTS: list = []


def _summary():
    import sys
    for r in _ROBOTS:
        if r.n_commands or r._model_name:
            m = r._model
            print(
                f"[fake rby1_sdk] robot model={r._model_name} commands={r.n_commands} "
                f"stream_commands={r.n_stream_commands} "
                f"right_arm_q={np.round(r._q[m.right_arm_idx], 3).tolist()}",
                file=sys.stderr,
            )


if os.environ.get("RBY1_FAKE_SUMMARY", "1") == "1":
    import atexit

    atexit.register(_summary)


def _rec(name, *args, **kwargs):
    CALL_LOG.append((name, args, kwargs))
    if os.environ.get("RBY1_FAKE_VERBOSE"):
        _log.warning("fake rby1_sdk: %s", name)


# --------------------------------------------------------------------------- #
# Models (joint layouts copied from the real Model_A / Model_M / Model_UB)
# --------------------------------------------------------------------------- #
_TORSO = [f"torso_{i}" for i in range(6)]
_RA = [f"right_arm_{i}" for i in range(7)]
_LA = [f"left_arm_{i}" for i in range(7)]
_HEAD = ["head_0", "head_1"]


class _Model:
    def __init__(self, wheels):
        self.robot_joint_names = list(wheels) + _TORSO + _RA + _LA + _HEAD
        self.robot_dof = len(self.robot_joint_names)
        n = len(wheels)
        self.mobility_idx = list(range(n))
        self.torso_idx = list(range(n, n + 6))
        self.right_arm_idx = list(range(n + 6, n + 13))
        self.left_arm_idx = list(range(n + 13, n + 20))
        self.body_idx = list(range(n, n + 20))
        self.head_idx = [n + 20, n + 21]
        self.velocity_estimation_required_idx = list(self.body_idx)


def Model_A():
    return _Model(["right_wheel", "left_wheel"])


def Model_M():
    return _Model(["wheel_fr", "wheel_fl", "wheel_rr", "wheel_rl"])


def Model_UB():
    return _Model([])


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class ControlManagerState:
    class State(enum.Enum):
        Unknown = 0
        Idle = 1
        Enabled = 2
        MinorFault = 3
        MajorFault = 4

    class ControlState(enum.Enum):
        Unknown = 0
        Idle = 1
        Executing = 2
        Switching = 3

    def __init__(self, state, control_state):
        self.state = state
        self.control_state = control_state


# --------------------------------------------------------------------------- #
# Command builders: record every setter and return self (chainable)
# --------------------------------------------------------------------------- #
class _Builder:
    def __init__(self, *args, **kwargs):
        self._calls: list[tuple[str, tuple]] = []
        self._ctor = args

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if not (name.startswith("set_") or name.startswith("add_")):
            raise AttributeError(f"{type(self).__name__} has no attribute {name!r} (fake)")

        def _setter(*args, **kwargs):
            self._calls.append((name, args))
            return self

        return _setter

    def get(self, name):
        for n, a in reversed(self._calls):
            if n == name:
                return a
        return None


def _mk(name):
    return type(name, (_Builder,), {})


for _n in [
    "ArmCommandBuilder", "BodyCommandBuilder", "BodyComponentBasedCommandBuilder",
    "CartesianCommandBuilder", "CartesianImpedanceControlCommandBuilder",
    "CommandHeaderBuilder", "ComponentBasedCommandBuilder",
    "GravityCompensationCommandBuilder", "HeadCommandBuilder",
    "ImpedanceControlCommandBuilder", "JogCommandBuilder",
    "JointGroupPositionCommandBuilder", "JointImpedanceControlCommandBuilder",
    "JointPositionCommandBuilder", "JointVelocityCommandBuilder",
    "MobilityCommandBuilder", "OptimalControlCommandBuilder", "RobotCommandBuilder",
    "SE2VelocityCommandBuilder", "StopCommandBuilder", "TorsoCommandBuilder",
    "WholeBodyCommandBuilder",
]:
    globals()[_n] = _mk(_n)


# --------------------------------------------------------------------------- #
# Dynamics
# --------------------------------------------------------------------------- #
class _DynState:
    def __init__(self, links, joints):
        self.links = list(links)
        self.joints = list(joints)
        self.q = np.zeros(len(joints))

    def set_q(self, q):
        self.q = np.asarray(q, dtype=np.float64).copy()

    def get_q(self):
        return self.q.copy()


class _Dynamics:
    def __init__(self, model):
        self._model = model

    def make_state(self, links, joints):
        return _DynState(links, joints)

    def get_limit_qdot_upper(self, state):
        return np.full(len(state.joints), 3.0)

    def get_limit_qddot_upper(self, state):
        return np.full(len(state.joints), 1.0)

    def get_limit_q_lower(self, state):
        return np.full(len(state.joints), -np.pi)

    def get_limit_q_upper(self, state):
        return np.full(len(state.joints), np.pi)

    def compute_forward_kinematics(self, state):
        pass

    def compute_transformation(self, state, i, j):
        T = np.eye(4)
        T[:3, 3] = [0.3, -0.2 if j == 2 else 0.2, 1.0 + 0.01 * j]
        return T


# --------------------------------------------------------------------------- #
# Robot
# --------------------------------------------------------------------------- #
class _Future:
    def __init__(self, value=None):
        self._v = value

    def get(self):
        return self._v

    def wait(self, *a, **k):
        return True


class _RobotInfo:
    def __init__(self, name, version):
        self.robot_model_name = name
        self.robot_model_version = version
        self.sdk_version = __version__


class _RobotState:
    def __init__(self, n):
        self.timestamp = time.time()
        self.position = np.zeros(n)
        self.velocity = np.zeros(n)
        self.torque = np.zeros(n)
        self.current = np.zeros(n)
        self.target_position = np.zeros(n)


class _CommandStream:
    """Like the real stream, it expires once a streamed base velocity command
    has outlived its control hold time (the core then finishes the mobility
    component and rejects further requests on this stream)."""

    def __init__(self, robot, priority):
        self._robot = robot
        self.priority = priority
        self._done = False
        self._mobility_deadline: float | None = None

    def send_command(self, cmd, timeout_ms=1000):
        now = time.monotonic()
        if self._done or (self._mobility_deadline is not None and now > self._mobility_deadline):
            self._done = True
            raise RuntimeError("Command stream expired (fake)")
        self._robot.n_stream_commands += 1
        self._robot._apply(cmd)
        hold = _mobility_hold_time(cmd)
        self._mobility_deadline = None if hold is None else now + hold
        return _Future()

    def is_done(self):
        return self._done

    def cancel(self):
        self._done = True


def _mobility_hold_time(cmd):
    cbc = cmd.get("set_command")
    mobility = cbc[0].get("set_mobility_command") if cbc else None
    if not mobility:
        return None
    header = mobility[0].get("set_command_header")
    hold = header[0].get("set_control_hold_time") if header else None
    return float(hold[0]) if hold else None


class FakeRobot:
    """In-process fake RB-Y1. Joint positions jump to the last commanded target."""

    def __init__(self, address, model_name):
        _rec("create_robot", address, model_name)
        self.address = address
        self._model_name = model_name
        self._model = {"a": Model_A, "m": Model_M, "ub": Model_UB}[model_name]()
        n = self._model.robot_dof
        self._q = np.zeros(n)
        self._qd = np.zeros(n)
        self._connected = False
        self._power = os.environ.get("RBY1_FAKE_POWER", "0") == "1"
        self._servo = False
        self._cm = ControlManagerState.State.Idle
        self._ctrl = ControlManagerState.ControlState.Idle
        self.n_commands = 0
        self.n_stream_commands = 0
        _ROBOTS.append(self)
        self.params: dict[str, str] = {}

    # connection --------------------------------------------------------
    def connect(self, max_retries=5, timeout_ms=1000, *a, **k):
        _rec("connect", max_retries=max_retries, timeout_ms=timeout_ms)
        if os.environ.get("RBY1_FAKE_CONNECT_FAIL") == "1":
            return False
        self._connected = True
        return True

    def disconnect(self):
        _rec("disconnect")
        self._connected = False

    def is_connected(self):
        return self._connected

    def get_robot_info(self):
        return _RobotInfo(
            os.environ.get("RBY1_FAKE_MODEL", self._model_name.upper()),
            os.environ.get("RBY1_FAKE_VERSION", "1.2"),
        )

    def model(self):
        return self._model

    def get_dynamics(self, urdf_path=None):
        return _Dynamics(self._model)

    # power / servo / control manager ----------------------------------
    def is_power_on(self, pattern):
        return self._power

    def power_on(self, pattern):
        _rec("power_on", pattern)
        self._power = True
        return True

    def power_off(self, pattern):
        self._power = False
        return True

    def is_servo_on(self, pattern):
        return self._servo

    def servo_on(self, pattern):
        _rec("servo_on", pattern)
        self._servo = True
        return True

    def get_control_manager_state(self):
        return ControlManagerState(self._cm, self._ctrl)

    def reset_fault_control_manager(self):
        self._cm = ControlManagerState.State.Idle
        return True

    def enable_control_manager(self, unlimited_mode_enabled=False):
        _rec("enable_control_manager", unlimited_mode_enabled=unlimited_mode_enabled)
        self._cm = ControlManagerState.State.Enabled
        return True

    def disable_control_manager(self):
        _rec("disable_control_manager")
        self._cm = ControlManagerState.State.Idle
        return True

    def cancel_control(self):
        self._ctrl = ControlManagerState.ControlState.Idle
        return True

    def wait_for_control_ready(self, timeout_ms):
        return True

    def set_parameter(self, name, value, write_db=True):
        self.params[name] = value
        return True

    def get_parameter(self, name):
        return self.params.get(name, "")

    def set_tool_flange_output_voltage(self, side, voltage):
        _rec("set_tool_flange_output_voltage", side, voltage)
        return True

    # commands / state -------------------------------------------------
    def create_command_stream(self, priority=1):
        return _CommandStream(self, priority)

    def send_command(self, cmd, priority=1):
        self._apply(cmd)
        return _Future()

    def get_state(self):
        s = _RobotState(self._model.robot_dof)
        s.position = self._q.copy()
        s.velocity = self._qd.copy()
        s.target_position = self._q.copy()
        return s

    def _apply(self, cmd):
        self.n_commands += 1
        cbc = cmd.get("set_command")
        if not cbc:
            return
        cbc = cbc[0]
        m = self._model
        body = cbc.get("set_body_command")
        if body:
            body = body[0]
            pos = body.get("set_position")
            if pos is not None:  # 20-DOF joint-position body command
                self._q[m.body_idx] = np.asarray(pos[0], dtype=np.float64)
            for setter, idx in (
                ("set_right_arm_command", m.right_arm_idx),
                ("set_left_arm_command", m.left_arm_idx),
                ("set_torso_command", m.torso_idx),
            ):
                sub = body.get(setter)
                if sub and sub[0].get("set_position") is not None:
                    self._q[idx] = np.asarray(sub[0].get("set_position")[0], dtype=np.float64)
        head = cbc.get("set_head_command")
        if head and head[0].get("set_position") is not None:
            self._q[m.head_idx] = np.asarray(head[0].get("set_position")[0], dtype=np.float64)


def create_robot(address, model):
    return FakeRobot(address, str(model).lower())


def create_robot_a(address):
    return FakeRobot(address, "a")


def create_robot_m(address):
    return FakeRobot(address, "m")


def create_robot_ub(address):
    return FakeRobot(address, "ub")


# --------------------------------------------------------------------------- #
# Dynamixel bus (gripper)
# --------------------------------------------------------------------------- #
class DynamixelBus:
    ProtocolVersion = 2.0
    DefaultBaudrate = 2_000_000
    TorqueEnable = 1
    TorqueDisable = 0
    CurrentControlMode = 0
    CurrentBasedPositionControlMode = 5

    def __init__(self, dev_name):
        _rec("DynamixelBus", dev_name)
        self.dev_name = dev_name
        self._enc = {0: 0.0, 1: 0.0}

    def open_port(self):
        return True

    def set_baud_rate(self, baud):
        return True

    def set_torque_constant(self, tc):
        pass

    def ping(self, dev_id):
        return True

    def group_sync_write_torque_enable(self, pairs):
        pass

    def group_sync_write_operating_mode(self, pairs):
        pass

    def group_sync_write_send_torque(self, pairs):
        # Homing: positive torque drives the jaw to "closed" (1.2 rad), negative to open.
        for dev_id, tq in pairs:
            if tq > 0:
                self._enc[dev_id] = min(self._enc[dev_id] + 0.1, 1.2)
            elif tq < 0:
                self._enc[dev_id] = max(self._enc[dev_id] - 0.1, 0.0)

    def group_sync_write_send_position(self, pairs):
        for dev_id, pos in pairs:
            self._enc[dev_id] = float(pos)

    def group_fast_sync_read_encoder(self, ids):
        return [(i, self._enc[i]) for i in ids]


from . import upc  # noqa: E402,F401
