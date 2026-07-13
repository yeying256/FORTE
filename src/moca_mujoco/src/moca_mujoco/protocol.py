import ctypes
import socket
import threading
import time
from enum import Enum
from typing import Optional, Tuple

import numpy as np


class CommandType(Enum):
    HEARTBEAT = 1
    STATE_UPDATE = 2
    CONTROL_CMD = 3
    ERROR_REPORT = 4
    RESET_TO_HOME = 5


class MobileCommandInterface(Enum):
    NONE = 0
    VELOCITY = 1
    WRENCH = 2
    POSITION_WORLD = 3


class ArmCommandInterface(Enum):
    NONE = 0
    POSITION = 1
    VELOCITY = 2
    TORQUE = 3


def encode_command_interfaces(
    arm_interface: ArmCommandInterface,
    mobile_interface: MobileCommandInterface,
) -> int:
    return ((mobile_interface.value & 0xFF) << 8) | (arm_interface.value & 0xFF)


def decode_command_interfaces(
    encoded: int,
) -> Tuple[ArmCommandInterface, MobileCommandInterface]:
    if encoded == 0:
        return ArmCommandInterface.VELOCITY, MobileCommandInterface.VELOCITY
    arm_interface = ArmCommandInterface(encoded & 0xFF)
    mobile_interface = MobileCommandInterface((encoded >> 8) & 0xFF)
    return arm_interface, mobile_interface


class RobotStateStruct(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("msg_type", ctypes.c_int),
        ("timestamp", ctypes.c_double),
        ("reserved", ctypes.c_int),
        ("mobile_positions", ctypes.c_double * 3),
        ("arm_positions", ctypes.c_double * 7),
        ("mobile_velocities", ctypes.c_double * 3),
        ("arm_velocities", ctypes.c_double * 7),
        ("figger_position", ctypes.c_double),
    ]


class RobotCommandStruct(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("Command_type", ctypes.c_int),
        ("timestamp", ctypes.c_double),
        ("reserved", ctypes.c_int),
        ("mobile_command", ctypes.c_double * 3),
        ("arm_command", ctypes.c_double * 7),
        ("figger_position", ctypes.c_double),
    ]


class TcpBridge:
    def __init__(self, ip: str, port: int, target_frequency: float):
        self.ip = ip
        self.port = int(port)
        self.target_frequency = float(target_frequency)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.ip, self.port))
        self.sock.listen(1)
        self.sock.settimeout(0.1)

        self.conn: Optional[socket.socket] = None
        self.addr = None
        self.is_connected = False
        self.has_received_command = False

        self._stop_event = threading.Event()
        self._recv_thread: Optional[threading.Thread] = None
        self._cmd_lock = threading.Lock()
        self._state_lock = threading.Lock()

        self.robot_state_msg = RobotStateStruct()
        self.robot_command_msg = RobotCommandStruct()
        self.robot_command_msg.Command_type = CommandType.CONTROL_CMD.value
        self.robot_command_msg.reserved = encode_command_interfaces(
            ArmCommandInterface.VELOCITY,
            MobileCommandInterface.VELOCITY,
        )

        print(
            f"MuJoCo TCP server listening on {self.ip}:{self.port}... "
            "(waiting for controller client)"
        )

    def start(self):
        if self._recv_thread is not None:
            return
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()

    def stop(self):
        self._stop_event.set()
        try:
            if self.conn is not None:
                self.conn.close()
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass
        if self._recv_thread is not None:
            self._recv_thread.join(timeout=1.0)

    def poll_connection(self):
        if self.conn is not None:
            return
        try:
            self.conn, self.addr = self.sock.accept()
            self.is_connected = True
            self.has_received_command = False
            print(f"TCP client connected from {self.addr}")
        except socket.timeout:
            return
        except OSError:
            return

    def fill_state(
        self,
        mobile_positions: np.ndarray,
        arm_positions: np.ndarray,
        mobile_velocities: np.ndarray,
        arm_velocities: np.ndarray,
        finger_position: float,
    ):
        mobile_positions = np.asarray(mobile_positions, dtype=np.float64).reshape(3)
        arm_positions = np.asarray(arm_positions, dtype=np.float64).reshape(7)
        mobile_velocities = np.asarray(mobile_velocities, dtype=np.float64).reshape(3)
        arm_velocities = np.asarray(arm_velocities, dtype=np.float64).reshape(7)

        with self._state_lock:
            self.robot_state_msg.msg_type = CommandType.STATE_UPDATE.value
            self.robot_state_msg.timestamp = time.time()
            self.robot_state_msg.reserved = 0

            for i in range(3):
                self.robot_state_msg.mobile_positions[i] = mobile_positions[i]
                self.robot_state_msg.mobile_velocities[i] = mobile_velocities[i]

            for i in range(7):
                self.robot_state_msg.arm_positions[i] = arm_positions[i]
                self.robot_state_msg.arm_velocities[i] = arm_velocities[i]

            self.robot_state_msg.figger_position = float(finger_position)

    def send_state(self):
        if self.conn is None:
            return
        try:
            with self._state_lock:
                self.conn.sendall(self.robot_state_msg)
        except OSError:
            self._drop_connection("send failed")

    def get_command(
        self,
    ) -> Tuple[
        CommandType,
        np.ndarray,
        np.ndarray,
        float,
        MobileCommandInterface,
        ArmCommandInterface,
    ]:
        with self._cmd_lock:
            cmd = self.robot_command_msg
            mobile_cmd = np.array([cmd.mobile_command[i] for i in range(3)], dtype=np.float64)
            arm_cmd = np.array([cmd.arm_command[i] for i in range(7)], dtype=np.float64)
            finger = float(cmd.figger_position)
            arm_interface, mobile_interface = decode_command_interfaces(cmd.reserved)
            try:
                command_type = CommandType(cmd.Command_type)
            except ValueError:
                command_type = CommandType.CONTROL_CMD

        return command_type, mobile_cmd, arm_cmd, finger, mobile_interface, arm_interface

    def _recv_loop(self):
        msg_size = ctypes.sizeof(RobotCommandStruct)
        while not self._stop_event.is_set():
            if self.conn is None:
                time.sleep(0.01)
                continue

            try:
                data = self.conn.recv(msg_size)
                if not data:
                    self._drop_connection("client disconnected")
                    continue

                while len(data) < msg_size:
                    more = self.conn.recv(msg_size - len(data))
                    if not more:
                        break
                    data += more

                if len(data) == msg_size:
                    with self._cmd_lock:
                        self.robot_command_msg = RobotCommandStruct.from_buffer_copy(data)
                    self.has_received_command = True
            except socket.timeout:
                continue
            except OSError:
                self._drop_connection("recv failed")

    def _drop_connection(self, reason: str):
        if self.conn is not None:
            try:
                self.conn.close()
            except OSError:
                pass
        self.conn = None
        self.addr = None
        self.is_connected = False
        self.has_received_command = False
        print(f"TCP client connection closed: {reason}")
