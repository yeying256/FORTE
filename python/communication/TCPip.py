import socket
from dataclasses import dataclass, asdict
from typing import List, Optional
import ctypes
import time
import math
from isaacsim.core.prims import Articulation

import numpy as np


from enum import Enum

# 看看是不是在ide中打字
from typing import TYPE_CHECKING

# 多线程
import threading


class Command_type(Enum):
    HEARTBEAT = 1       # 心跳包
    STATE_UPDATE = 2    # 状态更新
    CONTROL_CMD = 3     # 控制指令
    ERROR_REPORT = 4    # 错误报告
    RESET_TO_HOME = 5   # 机器人回到初始状态


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


def decode_command_interfaces(encoded: int) -> tuple[ArmCommandInterface, MobileCommandInterface]:
    if encoded == 0:
        return ArmCommandInterface.VELOCITY, MobileCommandInterface.VELOCITY
    arm_interface = ArmCommandInterface(encoded & 0xFF)
    mobile_interface = MobileCommandInterface((encoded >> 8) & 0xFF)
    return arm_interface, mobile_interface


if TYPE_CHECKING:
    # 这里定义的类型不会被执行，但 IDE 会读取它们来提供补全
    class RobotStateStruct(ctypes.Structure):
        msg_type: int
        timestamp: float
        reserved: int
        # 移动操作机器人位置
        mobile_positions: "ctypes.Array[float]"
        # 机械臂位置
        arm_positions: "ctypes.Array[float]" # 或者直接用 list[float] 示意
        mobile_velocities: "ctypes.Array[float]"
        arm_velocities: "ctypes.Array[float]"
        figger_position: float
        # 注意：这里不需要写 _fields_，因为运行时用下面的真实类
else:
    # 这是真正运行的代码
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

if TYPE_CHECKING:
    # 这里定义的类型不会被执行，但 IDE 会读取它们来提供补全
    class RobotCommandStruct(ctypes.Structure):
        Command_type: int
        timestamp: float
        reserved: int
        # 移动操作机器人指令
        mobile_command: "ctypes.Array[float]"
        # 机械臂指令
        arm_command: "ctypes.Array[float]" # 或者直接用 list[float] 示意
        figger_position: float
        # 注意：这里不需要写 _fields_，因为运行时用下面的真实类
else:
    # 这是真正运行的代码
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


class TCPip:

    def __init__(self,config,moca:Articulation):

        
        self.ip = config["communication"]["isaac"]["ip"]
        self.port = config["communication"]["isaac"]["port"]
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.ip, self.port))
        self.sock.listen(1)
        # 设置超时时间
        self.sock.settimeout(1.0) 
        # 设置非阻塞 
        # self.sock.setblocking(False)
        # self.conn = None
        self.addr = None  # <--- 修复报错的关键：先定义为 None
        print(f"TCP Server listening on {self.ip}:{self.port}... (Waiting for client)")
        self.command_rec_ = []

        # 2. 实例化 (直接在堆上分配内存)

        self.robot_state_msg_ = RobotStateStruct()
        self.robot_command_msg_=RobotCommandStruct()
        self.robot_command_msg_.Command_type = Command_type.CONTROL_CMD.value
        self.robot_command_msg_.reserved = encode_command_interfaces(
            ArmCommandInterface.VELOCITY,
            MobileCommandInterface.VELOCITY,
        )
        self.target_frequency_ = config["communication"]["isaac"]["target_frequency"]
        self.default_joint_positions_ = np.asarray(
            config["isaac"]["Moca"]["init_joint_positions"], dtype=np.float64
        ).reshape(-1)
        
        self.moca_ = moca

        self._stop_event = threading.Event()
        self._thread = None
        

        # flag
        self.is_connected = False 
        self.has_received_command = False
        self.conn = None

        # mutex
        self.lock = threading.Lock()
        self.lock_cmd = threading.Lock()


    

    def update(self):
        # target_frequency = 100.0  # 100 Hz
        period = 1.0 / self.target_frequency_ # 0.01 秒
        if not self.conn:
            try:
                self.conn, self.addr = self.sock.accept()
                self.is_connected = True
                self.has_received_command = False
                print('Connected by', self.addr)
            except Exception:
                pass

        next_frame_time = time.perf_counter() # 使用高精度计时器

        while not self._stop_event.is_set():
            #  如果没有连接
            if self.conn is None:
                try:
                    # 查看是否有人连接
                    self.conn, addr = self.sock.accept()
                    self.addr = addr
                    self.is_connected = True
                    self.has_received_command = False
                    print(f"Connected to {addr}")
                except socket.timeout:
                    pass
                except OSError as e:
                    # 只有在这里捕获到的才是真的错误（如端口被占用等）
                    if self._stop_event.is_set(): break
                    print(f"⚠️ Real Socket Error: {e}")
                    time.sleep(1.0)
            # 如果连接上了
            else:
                start_time = time.perf_counter()
                # 填充数据

                # self._fill_struct()
                try:
                    # 发送数据，发送的时候加锁
                    with self.lock:
                        # print("发送开始")
                        self.conn.send(self.robot_state_msg_)
                        # print("发送结束")
                    # self.conn.send(...) 
                except:
                    self.conn = None
                    self.is_connected = False
                    self.has_received_command = False
                elapsed = time.perf_counter() - start_time
                sleep_time = period - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    # 警告：你的代码跑太慢了！超过了
                    # 此时无法维持 100Hz，只能尽力而为
                    print(f"⚠️ 警告：单帧耗时 {elapsed:.4f}s，超过周期 {period:.4f}s，频率已下降！")
                    pass
            next_frame_time += period 

    def _fill_struct(self):

        """核心逻辑：分别获取基座位姿和关节数据"""
        # --- 1. 获取基座 (Mobile) 位置和速度 ---
        # get_world_pose 返回 (position=[x,y,z], orientation=[qw,qx,qy,qz])
        base_positions, base_orientations = self.moca_.get_world_poses()
        # 转换为 numpy 确保类型一致
        # 提取第 0 个机器人的数据
        pos_xyz = base_positions[0] # [x, y, z]
        quat_wxyz = base_orientations[0] # [w, x, y, z] (注意文档说是 scalar-first)
        x = float(pos_xyz[0])
        y = float(pos_xyz[1])
        # 提取四元数分量
        w, qx, qy, qz = float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])
        # 计算 Yaw
        yaw = self._quat_to_yaw(w, qx, qy, qz)

        try:
            lin_vel = self.moca_.get_linear_velocities()[0] # (M, 3) -> [vx, vy, vz]
            ang_vel = self.moca_.get_angular_velocities()[0] # (M, 3) -> [wx, wy, wz]
            vx = float(lin_vel[0])
            vy = float(lin_vel[1])
            wz = float(ang_vel[2]) # 取绕 Z 轴的角速度
        except Exception:
            vx, vy, wz = 0.0, 0.0, 0.0
        # --- 3. 获取机械臂关节数据 ---

        joint_positions = self.moca_.get_joint_positions()
        joint_velocities = self.moca_.get_joint_velocities()

        if joint_positions is None:
            joints_pos = self.default_joint_positions_.copy()
        else:
            joints_pos = np.asarray(joint_positions, dtype=np.float64)
            if joints_pos.ndim == 0:
                joints_pos = self.default_joint_positions_.copy()
            elif joints_pos.ndim > 1:
                joints_pos = joints_pos[0]

        if joint_velocities is None:
            joints_vel = np.zeros_like(joints_pos, dtype=np.float64)
        else:
            joints_vel = np.asarray(joint_velocities, dtype=np.float64)
            if joints_vel.ndim == 0:
                joints_vel = np.zeros_like(joints_pos, dtype=np.float64)
            elif joints_vel.ndim > 1:
                joints_vel = joints_vel[0]

        joints_pos = np.atleast_1d(joints_pos).astype(np.float64, copy=False)
        joints_vel = np.atleast_1d(joints_vel).astype(np.float64, copy=False)

        total_dofs = len(joints_pos)
        # print("total_dofs = ",total_dofs)

        with self.lock:
            # --- 3. 填充 Struct ---
            self.robot_state_msg_.msg_type = Command_type.STATE_UPDATE.value
            self.robot_state_msg_.timestamp = time.time()
            self.robot_state_msg_.reserved = 0

            # A. Mobile: [x, y, yaw]
            self.robot_state_msg_.mobile_positions[0] = x
            self.robot_state_msg_.mobile_positions[1] = y
            self.robot_state_msg_.mobile_positions[2] = yaw

            # B. Mobile Vel: [vx, vy, wz]
            self.robot_state_msg_.mobile_velocities[0] = vx
            self.robot_state_msg_.mobile_velocities[1] = vy
            self.robot_state_msg_.mobile_velocities[2] = wz

            # C. Arm: 前 7 个关节 (索引 0-6)
            # 注意：请根据你的 URDF/USD 确认前 7 个是否确实是机械臂
            for i in range(7):
                if i < total_dofs:
                    self.robot_state_msg_.arm_positions[i] = float(joints_pos[i])
                    self.robot_state_msg_.arm_velocities[i] = float(joints_vel[i])
                else:
                    self.robot_state_msg_.arm_positions[i] = 0.0
                    self.robot_state_msg_.arm_velocities[i] = 0.0

            # D. Finger: 第 8 个关节 (索引 7)
            # 如果你的模型有两个手指关节 (7 和 8)，通常取第一个或平均值
            if total_dofs > 7:
                self.robot_state_msg_.figger_position = joints_pos[7]
            else:
                self.robot_state_msg_.figger_position = 0.0

        
    def _quat_to_yaw(self, w, x, y, z):

        """
        将四元数 (w, x, y, z) 转换为 Yaw (绕 Z 轴旋转角，单位：弧度)
        公式基于 Z-Y-X 欧拉角提取
        """
        # sinp_cosp = 2 * (w * z + x * y)
        # cosp_cosp = 1 - 2 * (y * y + z * z)
        # yaw = math.atan2(sinp_cosp, cosp_cosp)
        # 更通用的计算方式 (对应 scipy R.as_euler('zxy') 或 'zyx' 的 yaw 分量)

        # 对于世界坐标系下的旋转，通常使用 atan2(2*w*z + 2*x*y, 1 - 2*y^2 - 2*z^2)

        t0 = 2.0 * (w * z + x * y)
        t1 = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t0, t1)
        return yaw
    

    def recv_loop(self):
        msg_size = ctypes.sizeof(RobotCommandStruct)
        while not self._stop_event.is_set():
            if self.conn is None:
                time.sleep(0.01)
                continue
            try:
                data = self.conn.recv(msg_size)

                # 客户端断开
                if not data:
                    print("Client disconnected")
                    self.conn = None
                    self.is_connected = False
                    self.has_received_command = False
                    continue

                # 如果数据不完整（TCP可能分包）
                while len(data) < msg_size:
                    more = self.conn.recv(msg_size - len(data))
                    if not more:
                        break
                    data += more

                if len(data) == msg_size:
                    cmd = RobotCommandStruct.from_buffer_copy(data)
                    # 写入共享变量（加锁）

                    with self.lock_cmd:
                        self.robot_command_msg_ = cmd
                    self.has_received_command = True

            except socket.timeout:
                pass

            except Exception as e:
                print(f"Recv error: {e}")
                self.conn = None
                self.is_connected = False
                self.has_received_command = False
    def get_command(self):
        """
        从共享内存读取控制指令并转换为 numpy
        返回:
            command_type
            mobile_cmd (3,)
            arm_cmd (7,)
            finger
            mobile_interface
            arm_interface
        """
        with self.lock_cmd:
            cmd = self.robot_command_msg_

            mobile_cmd = np.array([
                cmd.mobile_command[0],
                cmd.mobile_command[1],
                cmd.mobile_command[2],
            ], dtype=np.float64)

            arm_cmd = np.array([
                cmd.arm_command[i] for i in range(7)
            ], dtype=np.float64)

            finger = float(cmd.figger_position)
            arm_interface, mobile_interface = decode_command_interfaces(cmd.reserved)
            try:
                command_type = Command_type(cmd.Command_type)
            except ValueError:
                command_type = Command_type.CONTROL_CMD

        return command_type, mobile_cmd, arm_cmd, finger, mobile_interface, arm_interface

    def send(self, data):
        self.conn.sendall(data)
        
    def recv(self):
        data = self.conn.recv(1024)
        return data
    
    def send_callback(self,data):
        pass


    def start_thread(self):
        # <--- 3. 创建线程
        self._thread = threading.Thread(target=self.update, daemon=True)
        self.recv_thread = threading.Thread(target=self.recv_loop, daemon=True)

        self._thread.start()
        self.recv_thread.start()

    def stop(self):
        self._stop_event.set() # 触发事件
        if self._thread:
            self._thread.join()
