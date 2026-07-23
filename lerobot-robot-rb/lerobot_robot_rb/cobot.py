"""Low-level TCP client for Rainbow Robotics RB-Series control boxes.

This module preserves the communication protocol used by the original,
hardware-tested RB10E VR teleoperation code:

- Command socket: TCP port 5000
- Data socket: TCP port 5001
- Data request: ``reqdata``
- State packet: 580 bytes
- Joint command: ``move_servo_j(...)``

The original command syntax and state-packet layout are preserved. The
background data receiver uses a thread rather than multiprocessing so the
latest state and socket lifecycle remain owned by the same Python process.
"""

from __future__ import annotations

import ipaddress
import logging
import queue
import socket
import struct
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


logger = logging.getLogger(__name__)


# Packet format used by the original working cobot.py.
_STATE_STRUCT_FORMAT = (
    "fffffffffffffffffffffffffffffffffffffff"
    "iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii"
    "ffffff"
    "iiii"
    "f"
    "i"
    "f"
    "i"
    "i"
    "ffffff"
    "iiiiiiiiiii"
    "ff"
    "iiii"
    "f"
    "iiiiiiiiiii"
    "ffffff"
    "i"
    "ffffffff"
    "i"
    "ffffff"
    "i"
)

_STATE_PACKET_BYTES = 580
_STATE_PAYLOAD_BYTES = 576


@dataclass
class systemSTAT:
    """Decoded RB control-box state.

    The field order intentionally matches the original 580-byte packet
    parser. Angles and TCP values use the control box's native units:
    degrees and millimetres.
    """

    time: float = 0.0

    jnt_ref: tuple = ()
    jnt_ang: tuple = ()
    cur: tuple = ()

    tcp_ref: tuple = ()
    tcp_pos: tuple = ()

    analog_in: tuple = ()
    analog_out: tuple = ()
    digital_in: tuple = ()
    digital_out: tuple = ()

    temperature_mc: tuple = ()

    task_pc: int = 0
    task_repeat: int = 0
    task_run_id: int = 0
    task_run_num: int = 0
    task_run_time: float = 0.0
    task_state: int = 0

    default_speed: float = 0.0
    robot_state: int = 0
    power_state: int = 0

    tcp_target: tuple = ()
    jnt_info: tuple = ()

    collision_detect_onoff: int = 0
    is_freedrive_mode: int = 0
    program_mode: int = 0
    init_state_info: int = 0
    init_error: int = 0

    tfb_analog_in: tuple = ()
    tfb_digital_in: tuple = ()
    tfb_digital_out: tuple = ()
    tfb_voltage_out: float = 0.0

    op_stat_collision_occur: int = 0
    op_stat_sos_flag: int = 0
    op_stat_self_collision: int = 0
    op_stat_soft_estop_occur: int = 0
    op_stat_ems_flag: int = 0

    digital_in_config: tuple = ()
    inbox_trap_flag: tuple = ()
    inbox_check_mode: tuple = ()

    eft_fx: float = 0.0
    eft_fy: float = 0.0
    eft_fz: float = 0.0
    eft_mx: float = 0.0
    eft_my: float = 0.0
    eft_mz: float = 0.0

    information_chunk_4: int = 0

    extend_io1_analog_in: tuple = ()
    extend_io1_analog_out: tuple = ()
    extend_io1_digital_info: int = 0

    aa_joint_ref: tuple = ()
    safety_board_stat_info: int = 0


# PEP 8 alias for new code. Keep systemSTAT for compatibility with the
# original scripts.
SystemStat = systemSTAT


@dataclass
class Joint:
    j0: float = 0.0
    j1: float = 0.0
    j2: float = 0.0
    j3: float = 0.0
    j4: float = 0.0
    j5: float = 0.0


@dataclass
class Point:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0


class COBOT_STATUS(Enum):
    IDLE = 0
    PAUSED = 1
    RUNNING = 2
    UNKNOWN = 3


class CMD_TYPE(Enum):
    MOVE = 0
    NONMOVE = 1


class PG_MODE(Enum):
    SIMULATION = 0
    REAL = 1


class CIRCLE_TYPE(Enum):
    INTENDED = 0
    CONSTANT = 1
    RADIAL = 2
    SMOOTH = 3


class CIRCLE_AXIS(Enum):
    X = 0
    Y = 1
    Z = 2


class BLEND_OPTION(Enum):
    RATIO = 0
    DISTANCE = 1


class BLEND_RTYPE(Enum):
    INTENDED = 0
    CONSTANT = 1


class ITPL_RTYPE(Enum):
    INTENDED = 0
    CONSTANT = 1
    RESERVED1 = 2
    SMOOTH = 3
    RESERVED2 = 4
    CA_INTENDED = 5
    CA_CONSTANT = 6
    RESERVED3 = 7
    CA_SMOOTH = 8


class DOUT_SET(Enum):
    LOW = 0
    HIGH = 1
    BYPASS = 2


def _decode_state_packet(packet: bytes) -> systemSTAT | None:
    """Decode one complete 580-byte control-box packet.

    Returns ``None`` for packets that are not normal state packets.
    """

    if len(packet) != _STATE_PACKET_BYTES:
        raise ValueError(
            f"Expected {_STATE_PACKET_BYTES} bytes, got {len(packet)}."
        )

    if packet[0] != 0x24:
        raise ValueError(
            f"Invalid state packet header: 0x{packet[0]:02x}."
        )

    payload_size = int((packet[2] << 8) | packet[1])
    if payload_size > len(packet) - 4:
        raise ValueError(
            f"Invalid payload size {payload_size}; "
            f"packet contains {len(packet) - 4} payload bytes."
        )

    packet_type = packet[3]

    # 3 = normal system state, 4 = configuration, 10 = popup.
    if packet_type != 3:
        return None

    payload = packet[4 : 4 + _STATE_PAYLOAD_BYTES]
    result = struct.unpack(_STATE_STRUCT_FORMAT, payload)

    return systemSTAT(
        result[0],
        result[1:7],
        result[7:13],
        result[13:19],
        result[19:25],
        result[25:31],
        result[31:35],
        result[35:39],
        result[39:55],
        result[55:71],
        result[71:77],
        result[77],
        result[78],
        result[79],
        result[80],
        result[81],
        result[82],
        result[83],
        result[84],
        result[85],
        result[86:92],
        result[92:98],
        result[98],
        result[99],
        result[100],
        result[101],
        result[102],
        result[103:105],
        result[105:107],
        result[107:109],
        result[109],
        result[110],
        result[111],
        result[112],
        result[113],
        result[114],
        result[115:117],
        result[117:119],
        result[119:121],
        result[121],
        result[122],
        result[123],
        result[124],
        result[125],
        result[126],
        result[127],
        result[128:132],
        result[132:136],
        result[136],
        result[137:143],
        result[143],
    )


class Cobot:
    """Low-level Rainbow Robotics RB-Series TCP client."""

    def __init__(
        self,
        ip: str,
        command_port: int = 5000,
        data_port: int = 5001,
        *,
        socket_timeout_s: float = 1.0,
        data_request_hz: float = 500.0,
    ) -> None:
        if not self.is_valid_ip(ip):
            raise ValueError(f"Invalid IPv4 address: {ip!r}")

        if not 0 < int(command_port) <= 65535:
            raise ValueError(
                f"Invalid command_port: {command_port}"
            )

        if not 0 < int(data_port) <= 65535:
            raise ValueError(f"Invalid data_port: {data_port}")

        if socket_timeout_s <= 0.0:
            raise ValueError("socket_timeout_s must be positive.")

        if data_request_hz <= 0.0:
            raise ValueError("data_request_hz must be positive.")

        self.ip = ip
        self.CMD_PORT = int(command_port)
        self.DATA_PORT = int(data_port)

        self._socket_timeout_s = float(socket_timeout_s)
        self._data_request_period_s = 1.0 / float(
            data_request_hz
        )

        self.systemstat_global = systemSTAT()

        # Preserve the original public flags:
        # 0 = connected, non-zero = disconnected/error.
        self.cmd_connect = 1
        self.data_connect = 1

        self.bReadCmd = False
        self.moveCmdFlag = False
        self.moveCmdCnt = 0
        self.cmd_send_flag = 0

        self.__RB_VERSION__ = "oes-0.2-lerobot"

        self.CMDSock: socket.socket | None = None
        self.DATASock: socket.socket | None = None

        self._command_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._state_ready = threading.Event()
        self._stop_event = threading.Event()

        self._data_thread: threading.Thread | None = None
        self._data_error: Exception | None = None

        # Compatibility field. New code should use GetLatestState().
        self.reqdata_queue: queue.Queue[systemSTAT] = (
            queue.Queue(maxsize=1)
        )

        logger.info("RB TCP client version %s", self.__RB_VERSION__)

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    @staticmethod
    def is_valid_ip(ip: str) -> bool:
        try:
            ipaddress.IPv4Address(ip)
        except ipaddress.AddressValueError:
            return False
        return True

    # Original method name compatibility.
    def isValidIP(self) -> bool:
        return self.is_valid_ip(self.ip)

    @property
    def is_connected(self) -> bool:
        return (
            self.cmd_connect == 0
            and self.data_connect == 0
            and self.CMDSock is not None
            and self.DATASock is not None
        )

    @property
    def data_error(self) -> Exception | None:
        return self._data_error

    def ConnectToCB(self) -> bool:
        """Connect command/data sockets and start state reception."""

        if self.is_connected:
            return True

        self.DisConnectToCB()

        cmd_sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        )
        data_sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        )

        cmd_sock.settimeout(self._socket_timeout_s)
        data_sock.settimeout(self._socket_timeout_s)

        try:
            cmd_result = cmd_sock.connect_ex(
                (self.ip, self.CMD_PORT)
            )
            data_result = data_sock.connect_ex(
                (self.ip, self.DATA_PORT)
            )
        except Exception:
            cmd_sock.close()
            data_sock.close()
            raise

        self.cmd_connect = cmd_result
        self.data_connect = data_result

        if cmd_result != 0 or data_result != 0:
            cmd_sock.close()
            data_sock.close()

            self.CMDSock = None
            self.DATASock = None

            logger.error(
                "Failed to connect RB control box %s "
                "(command=%d, data=%d).",
                self.ip,
                cmd_result,
                data_result,
            )
            return False

        self.CMDSock = cmd_sock
        self.DATASock = data_sock

        self._stop_event.clear()
        self._state_ready.clear()
        self._data_error = None

        self._data_thread = threading.Thread(
            target=self._read_data_loop,
            name="rb-cobot-data",
            daemon=True,
        )
        self._data_thread.start()

        logger.info(
            "RB command/data ports connected: %s:%d, %s:%d",
            self.ip,
            self.CMD_PORT,
            self.ip,
            self.DATA_PORT,
        )
        return True

    def DisConnectToCB(self) -> bool:
        """Stop state reception and close both TCP sockets."""

        self._stop_event.set()

        for sock in (self.DATASock, self.CMDSock):
            if sock is None:
                continue

            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

            try:
                sock.close()
            except OSError:
                pass

        thread = self._data_thread
        if (
            thread is not None
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=1.0)

        self._data_thread = None
        self.CMDSock = None
        self.DATASock = None

        self.cmd_connect = 1
        self.data_connect = 1

        return True

    def __Version(self) -> None:
        print(f"RB-API : {self.__RB_VERSION__}")

    # ------------------------------------------------------------------
    # State reception
    # ------------------------------------------------------------------

    @staticmethod
    def _recv_exact(
        sock: socket.socket,
        size: int,
    ) -> bytes:
        chunks: list[bytes] = []
        received = 0

        while received < size:
            chunk = sock.recv(size - received)

            if not chunk:
                raise ConnectionError(
                    "RB data socket closed by peer."
                )

            chunks.append(chunk)
            received += len(chunk)

        return b"".join(chunks)

    def _publish_state(self, state: systemSTAT) -> None:
        with self._state_lock:
            self.systemstat_global = state

        # Keep only the latest queued state for compatibility.
        try:
            self.reqdata_queue.put_nowait(state)
        except queue.Full:
            try:
                self.reqdata_queue.get_nowait()
            except queue.Empty:
                pass

            try:
                self.reqdata_queue.put_nowait(state)
            except queue.Full:
                pass

        self._state_ready.set()

    def _read_data_loop(self) -> None:
        sock = self.DATASock

        if sock is None:
            return

        request = b"reqdata"
        deadline = time.perf_counter()

        try:
            while not self._stop_event.is_set():
                sock.sendall(request)

                packet = self._recv_exact(
                    sock,
                    _STATE_PACKET_BYTES,
                )
                state = _decode_state_packet(packet)

                if state is not None:
                    self._publish_state(state)

                deadline += self._data_request_period_s
                remaining = deadline - time.perf_counter()

                if remaining > 0.0:
                    self._stop_event.wait(remaining)
                else:
                    # Avoid burst requests after scheduling delays.
                    deadline = time.perf_counter()

        except (OSError, ConnectionError, ValueError) as exc:
            if not self._stop_event.is_set():
                self._data_error = exc
                logger.exception(
                    "RB data receiver stopped: %s",
                    exc,
                )
                self._state_ready.set()

    def wait_for_first_state(
        self,
        timeout_s: float = 2.0,
    ) -> bool:
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive.")

        ready = self._state_ready.wait(timeout_s)

        if self._data_error is not None:
            raise RuntimeError(
                "RB data receiver failed."
            ) from self._data_error

        return ready

    def GetLatestState(
        self,
        timeout_s: float | None = 1.0,
    ) -> systemSTAT:
        """Return the most recently decoded control-box state."""

        if not self.is_connected:
            raise ConnectionError(
                "RB control box is not connected."
            )

        if timeout_s is not None:
            if timeout_s <= 0.0:
                raise ValueError(
                    "timeout_s must be positive or None."
                )

            if not self._state_ready.wait(timeout_s):
                raise TimeoutError(
                    "Timed out waiting for RB state data."
                )

        if self._data_error is not None:
            raise RuntimeError(
                "RB data receiver failed."
            ) from self._data_error

        with self._state_lock:
            return self.systemstat_global

    # Original compatibility methods. The thread loop replaces the old
    # multiprocessing implementations.
    def ReqDataStart(self, sock: socket.socket) -> None:
        request = b"reqdata"

        while not self._stop_event.is_set():
            sock.sendall(request)
            self._stop_event.wait(0.01)

    def ReadDATA(
        self,
        sock: socket.socket,
        output_queue=None,
    ) -> None:
        request = b"reqdata"
        deadline = time.perf_counter()

        while not self._stop_event.is_set():
            sock.sendall(request)

            packet = self._recv_exact(
                sock,
                _STATE_PACKET_BYTES,
            )
            state = _decode_state_packet(packet)

            if state is not None:
                self._publish_state(state)

                if output_queue is not None:
                    output_queue.put(state)

            deadline += self._data_request_period_s
            remaining = deadline - time.perf_counter()

            if remaining > 0.0:
                self._stop_event.wait(remaining)
            else:
                deadline = time.perf_counter()

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def GetCurrentJoint(self) -> Joint:
        state = self.GetLatestState()
        values = state.jnt_ref

        if len(values) < 6:
            raise RuntimeError(
                "RB state does not contain six joint references."
            )

        return Joint(*map(float, values[:6]))

    def GetCurrentSplitedJoint(self) -> tuple[float, ...]:
        state = self.GetLatestState()
        values = state.jnt_ref

        if len(values) < 6:
            raise RuntimeError(
                "RB state does not contain six joint references."
            )

        return tuple(float(value) for value in values[:6])

    # ------------------------------------------------------------------
    # Command transport
    # ------------------------------------------------------------------

    def _send_command_text(self, command: str) -> bool:
        if not command:
            raise ValueError("command must not be empty.")

        sock = self.CMDSock

        if sock is None or not self.is_connected:
            raise ConnectionError(
                "RB command socket is not connected."
            )

        wire = f"{command} ".encode("utf-8")

        with self._command_lock:
            sock.sendall(wire)

        self.cmd_send_flag = 1
        return True

    def SendCOMMAND(
        self,
        command: str,
        cmd_type: CMD_TYPE,
    ) -> bool:
        """Send one control-box script command.

        ServoJ uses NONMOVE, matching the original hardware-tested code.
        MOVE retains the original wait-until-idle behavior for MoveJ.
        """

        if cmd_type == CMD_TYPE.NONMOVE:
            return self._send_command_text(command)

        deadline = time.monotonic() + 30.0

        while time.monotonic() < deadline:
            if self.IsPause():
                return False

            if self.IsIdle() and not self.bReadCmd:
                self.moveCmdFlag = True
                self.systemstat_global.robot_state = 3
                return self._send_command_text(command)

            time.sleep(0.03)

        raise TimeoutError(
            "Timed out waiting for the RB robot to become idle."
        )

    # ------------------------------------------------------------------
    # RB commands
    # ------------------------------------------------------------------

    def CobotInit(self) -> bool:
        return self.SendCOMMAND(
            "mc jall init",
            CMD_TYPE.NONMOVE,
        )

    def SetProgramMode(
        self,
        mode: PG_MODE = PG_MODE.SIMULATION,
    ) -> bool:
        if mode == PG_MODE.SIMULATION:
            command = "pgmode simulation"
        elif mode == PG_MODE.REAL:
            command = "pgmode real"
        else:
            raise ValueError(
                f"Unsupported program mode: {mode!r}"
            )

        return self.SendCOMMAND(
            command,
            CMD_TYPE.NONMOVE,
        )

    def MoveJ(
        self,
        j0: float,
        j1: float,
        j2: float,
        j3: float,
        j4: float,
        j5: float,
        spd: float,
        acc: float,
    ) -> bool:
        command = (
            "move_j(jnt["
            f"{j0},{j1},{j2},{j3},{j4},{j5}"
            f"], {spd},{acc})"
        )

        logger.info("RB MoveJ: %s", command)

        return self.SendCOMMAND(
            command,
            CMD_TYPE.MOVE,
        )

    def gripper_dxl_xm_initialization(
        self,
        device_id: int = 0,
    ):
        """Initialize the control-box Dynamixel XM gripper."""

        return self.SendCOMMAND(
            "gripper_dxl_xm_initialization"
            f"({int(device_id)})"
        )

    def gripper_dxl_xm_set_target_current(
        self,
        current_mA: int,
    ):
        """Set the Dynamixel XM gripper target current in mA."""

        return self.SendCOMMAND(
            "gripper_dxl_xm_set_target_current"
            f"({int(current_mA)})"
        )

    def ServoJ(
        self,
        joints_deg: Sequence[float],
        t1: float,
        t2: float,
        gain: float,
        alpha: float,
    ) -> bool:
        """Send the original RB10E move_servo_j command.

        Parameters use the original control-box units:
        ``joints_deg`` in degrees and timing values in seconds.
        """

        if len(joints_deg) != 6:
            raise ValueError(
                "ServoJ requires exactly six joint values."
            )

        values = [float(value) for value in joints_deg]

        command = (
            "move_servo_j(jnt["
            f"{values[0]:.3f},"
            f"{values[1]:.3f},"
            f"{values[2]:.3f},"
            f"{values[3]:.3f},"
            f"{values[4]:.3f},"
            f"{values[5]:.3f}], "
            f"{float(t1):.3f}, "
            f"{float(t2):.3f}, "
            f"{float(gain):.3f}, "
            f"{float(alpha):.3f})"
        )

        return self.SendCOMMAND(
            command,
            CMD_TYPE.NONMOVE,
        )

    def InitDxlCurrentMode(self) -> bool:
        return self.SendCOMMAND(
            "gripper_macro 37,0,0,0,0,0,0,0,0,0",
            CMD_TYPE.NONMOVE,
        )

    def MoveDxlCurrentMode(
        self,
        target_mA: float,
    ) -> bool:
        return self.SendCOMMAND(
            "gripper_macro "
            f"37,0,2,0,{float(target_mA):.3f},0,0,0,0,0",
            CMD_TYPE.NONMOVE,
        )

    def InitDxlPositionMode(self) -> bool:
        return self.SendCOMMAND(
            "gripper_macro 37,0,1,0,0,0,0,0,0,0",
            CMD_TYPE.NONMOVE,
        )

    def MoveDxlPositionMode(
        self,
        target_tick: int,
    ) -> bool:
        return self.SendCOMMAND(
            "gripper_macro "
            f"37,0,3,0,{int(target_tick)},0,0,0,0,0",
            CMD_TYPE.NONMOVE,
        )

    def SetBaseSpeed(self, spd: float) -> bool:
        speed = max(0.0, min(1.0, float(spd)))

        return self.SendCOMMAND(
            f"set_speed_bar({speed})",
            CMD_TYPE.NONMOVE,
        )

    # ------------------------------------------------------------------
    # State predicates
    # ------------------------------------------------------------------

    def IsIdle(self) -> bool:
        return self.systemstat_global.robot_state == 1

    def IsPause(self) -> bool:
        return (
            self.systemstat_global.op_stat_soft_estop_occur
            == 1
        )

    def IsInitialized(self) -> bool:
        return self.systemstat_global.init_state_info == 6

    def IsRobotReal(self) -> bool:
        return self.systemstat_global.program_mode == 0

    def IsCommandSockConnect(self) -> bool:
        return self.cmd_connect == 0

    def IsDataSockConnect(self) -> bool:
        return self.data_connect == 0

    def GetCurrentCobotStatus(self) -> COBOT_STATUS:
        state = self.systemstat_global

        if state.op_stat_soft_estop_occur == 1:
            return COBOT_STATUS.PAUSED

        if state.robot_state == 1:
            return COBOT_STATUS.IDLE

        if state.robot_state == 3:
            return COBOT_STATUS.RUNNING

        return COBOT_STATUS.UNKNOWN

    # Context manager support for standalone tests.
    def __enter__(self) -> Cobot:
        if not self.ConnectToCB():
            raise ConnectionError(
                f"Could not connect to RB control box at {self.ip}."
            )
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.DisConnectToCB()
