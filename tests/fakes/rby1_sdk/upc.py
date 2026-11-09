"""Fake ``rby1_sdk.upc``: leader arm + device names, no serial I/O."""

from __future__ import annotations

import threading
import time

import numpy as np

from . import _rec

import os as _os

GripperDeviceName = _os.environ.get("RBY1_FAKE_GRIPPER_DEV", "/dev/rby1_gripper")
LeaderArmDeviceName = "/dev/rby1_leader_arm"
LegacyLeaderArmDeviceName = "/dev/rby1_master_arm"
MasterArmDeviceName = LegacyLeaderArmDeviceName


def initialize_device(name):
    _rec("upc.initialize_device", name)
    return True


def resolve_leader_arm_device_name():
    return LeaderArmDeviceName


resolve_master_arm_device_name = resolve_leader_arm_device_name


class _Button:
    def __init__(self):
        self.button = 0
        self.trigger = 0


class LeaderArm:
    DOF = 14
    DeviceCount = 16  # same as the real SDK 0.10.0 (14 joints + 2 tool boards)
    RightToolId = 0x80
    LeftToolId = 0x81
    MaximumTorque = 4.0
    TorqueScaling = 0.5

    class ControlInput:
        def __init__(self):
            self.target_operating_mode = np.zeros(14, dtype=np.int32)
            self.target_position = np.zeros(14)
            self.target_torque = np.zeros(14)

    class State:
        def __init__(self, q):
            self.q_joint = np.asarray(q, dtype=np.float64).copy()
            self.qvel_joint = np.zeros(14)
            self.torque_joint = np.zeros(14)
            self.gravity_term = np.zeros(14)
            self.operating_mode = np.zeros(14, dtype=np.int32)
            self.target_position = self.q_joint.copy()
            self.button_right = _Button()
            self.button_left = _Button()
            self.T_right = np.eye(4)
            self.T_left = np.eye(4)

    def __init__(self, dev_name=LeaderArmDeviceName):
        _rec("upc.LeaderArm", dev_name)
        self._period = 0.01
        self._q = np.zeros(14)
        self._thread = None
        self._stop = threading.Event()
        self._t0 = time.time()
        self.n_ticks = 0

    def set_model_path(self, p):
        self.model_path = p

    def set_control_period(self, p):
        self._period = float(p)

    def set_torque_constant(self, tc):
        pass

    def initialize(self, verbose=False):
        return list(range(14)) + [self.RightToolId, self.LeftToolId]

    def enable_torque(self, *a):
        return True

    def disable_torque(self, *a):
        return True

    def start_control(self, callback):
        def _loop():
            while not self._stop.is_set():
                st = LeaderArm.State(self._q)
                # Slowly squeeze the triggers so gripper actions are not constant.
                trig = 500 + 400 * np.sin(time.time() - self._t0)
                st.button_right.trigger = trig
                st.button_left.trigger = 1000 - trig
                ci = callback(st)
                if ci is not None:
                    pos_mode = ci.target_operating_mode != 0
                    self._q = np.where(pos_mode, ci.target_position, self._q)
                self.n_ticks += 1
                time.sleep(self._period)

        self._thread = threading.Thread(target=_loop, daemon=True, name="fake-leader-arm")
        self._thread.start()
        return True

    def stop_control(self, torque_disable=False):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return True


MasterArm = LeaderArm
