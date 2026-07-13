from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
from isaacsim.core.api import World
# from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.core.prims import Articulation
import omni.timeline
import omni.usd
from pxr import Usd

import yaml
import time
from pathlib import Path
from common_tools import file as f_py
from typing import Tuple

from communication.TCPip import TCPip, MobileCommandInterface, ArmCommandInterface, Command_type





class Moca:


    def __init__(self,config):
        # self.simulation_app_ = SimulationApp({"headless": False})
        self.config_ = f_py.load_config(config)
        self.world_ = World()
# isaac:
#     Moca:
#         pwd: "/media/wangxiao/iit_work_disk/IIT_ws/Moca_polytope/Moca_Isaac/Moca_fix_wheels.usd"
#         prim_path: "/World/MOCA"

#         # 机器人初始位置
#         init_world_poses: [
#             [0, 0, 0, 1, 0, 0],
#         ]
        
        # 设置prim路径
        self.prim_paths_expr_ = self.config_["isaac"]["Moca"]["prim_path"]

        # 找到机器人路径
        self.asset_path_ = self.config_["isaac"]["Moca"]["pwd"]

        add_reference_to_stage(
            usd_path=self.asset_path_,
            prim_path=self.prim_paths_expr_
        )

        self.world_.scene.add_default_ground_plane()
        self.moca_ = Articulation(prim_paths_expr=self.prim_paths_expr_)
        self.init_world_pose_ = self.config_["isaac"]["Moca"]["init_world_poses"]
        self.init_joint_position_ = self.config_["isaac"]["Moca"]["init_joint_positions"]
        self.zero_arm_joint_drive_and_friction_ = self.config_["isaac"]["Moca"].get(
            "zero_arm_joint_drive_and_friction", True
        )
        self.arm_joint_drive_stiffness_scale_ = float(
            self.config_["isaac"]["Moca"].get("arm_joint_drive_stiffness_scale", 1.0)
        )
        self.arm_joint_drive_damping_scale_ = float(
            self.config_["isaac"]["Moca"].get("arm_joint_drive_damping_scale", 1.0)
        )
        self.arm_joint_friction_scale_ = float(
            self.config_["isaac"]["Moca"].get("arm_joint_friction_scale", 1.0)
        )

        self.control_frequency_ = self.config_["communication"]["isaac"]["target_frequency"]
        self.dt_ = 1.0 / self.control_frequency_

        # 与外界的接口
        self.tcpip = TCPip(self.config_,moca=self.moca_)
        # 单独一个线程
        self.tcpip.start_thread()
        # 单独一个线程 
        self.world_.reset()
        self.timeline_ = omni.timeline.get_timeline_interface()
        self.timeline_paused_for_tcp_ = False

        # 一个不起眼的计数器
        self.debug_count_ = 0
        self.base_velocity_cmd_ = np.zeros(3, dtype=np.float64)
        self.mobile_admittance_mass_ = np.array([105.0, 105.0, 20.0], dtype=np.float64)
        self.mobile_admittance_damping_ = np.array([180.0, 180.0, 30.0], dtype=np.float64)
        self.mobile_velocity_limits_ = np.array([0.5, 0.5, 1.0], dtype=np.float64)
        self.arm_effort_joint_indices_ = np.arange(7, dtype=np.int32)
        self.arm_effort_mode_configured_ = False
        self.arm_asset_stiffness_gains_ = None
        self.arm_asset_damping_gains_ = None
        self.arm_joint_drive_scaling_applied_ = False
        self.waiting_for_connection_ = True
        self.waiting_for_first_command_ = True
        self.joint_physics_dumped_ = False
        self.arm_joint_physics_zeroed_ = False




    def _yaw_to_quaternion_wxyz(self, yaw: float) -> np.ndarray:
        half_yaw = 0.5 * float(yaw)
        return np.array(
            [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)],
            dtype=np.float64,
        )

    def _parse_home_world_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        raw_pose = np.asarray(self.init_world_pose_, dtype=np.float64).reshape(-1)

        if raw_pose.size == 3:
            # Backward-compatible format: [x, y, z]
            position = raw_pose
            orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        elif raw_pose.size == 4:
            # Extended format: [x, y, z, yaw]
            position = raw_pose[:3]
            orientation = self._yaw_to_quaternion_wxyz(raw_pose[3])
        elif raw_pose.size == 7:
            # Full pose format: [x, y, z, qw, qx, qy, qz]
            position = raw_pose[:3]
            orientation = raw_pose[3:]
        else:
            raise ValueError(
                "init_world_poses must contain 3 values [x, y, z], "
                "4 values [x, y, z, yaw], or 7 values [x, y, z, qw, qx, qy, qz]."
            )

        orientation_norm = np.linalg.norm(orientation)
        if orientation_norm < 1e-12:
            orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        else:
            orientation = orientation / orientation_norm

        return position, orientation

    def _apply_home_state(self):
        home_position, home_orientation = self._parse_home_world_pose()
        joint_positions = np.asarray(self.init_joint_position_, dtype=np.float64)
        joint_velocities = np.zeros_like(joint_positions, dtype=np.float64)
        joint_efforts = np.zeros_like(joint_positions, dtype=np.float64)

        self.base_velocity_cmd_.fill(0.0)
        self.moca_.set_linear_velocities(np.array([[0.0, 0.0, 0.0]], dtype=np.float64))
        self.moca_.set_angular_velocities(np.array([[0.0, 0.0, 0.0]], dtype=np.float64))
        self.moca_.set_world_poses(
            positions=np.array([home_position], dtype=np.float64) / get_stage_units(),
            orientations=np.array([home_orientation], dtype=np.float64),
        )
        self.moca_.set_joint_velocities(np.array([joint_velocities], dtype=np.float64))
        self.moca_.set_joint_efforts(np.array([joint_efforts], dtype=np.float64))
        self.moca_.set_joint_positions(np.array([joint_positions], dtype=np.float64))
        self._configure_arm_pure_effort_mode()
        self._apply_arm_joint_drive_and_friction_scaling()
        self._zero_arm_joint_drive_and_friction()

    def _configure_arm_pure_effort_mode(self):
        """Configure arm torque mode.

        When zero_arm_joint_drive_and_friction_ is enabled, we additionally zero the
        runtime gains to approximate pure effort control. Otherwise we keep the USD
        asset's original drive gains as a baseline and optionally scale them.
        """
        self.moca_.set_effort_modes("force", joint_indices=self.arm_effort_joint_indices_)

        try:
            stiffnesses, dampings = self.moca_.get_gains(
                joint_indices=self.arm_effort_joint_indices_
            )
            stiffnesses = np.asarray(stiffnesses, dtype=np.float32).reshape(1, -1)
            dampings = np.asarray(dampings, dtype=np.float32).reshape(1, -1)
        except Exception:
            stiffnesses = None
            dampings = None

        if stiffnesses is not None and dampings is not None:
            if self.arm_asset_stiffness_gains_ is None:
                self.arm_asset_stiffness_gains_ = stiffnesses.copy()
            if self.arm_asset_damping_gains_ is None:
                self.arm_asset_damping_gains_ = dampings.copy()

        if self.zero_arm_joint_drive_and_friction_:
            target_stiffness = np.zeros((1, len(self.arm_effort_joint_indices_)), dtype=np.float32)
            target_damping = np.zeros((1, len(self.arm_effort_joint_indices_)), dtype=np.float32)
            self.moca_.set_gains(
                kps=target_stiffness,
                kds=target_damping,
                joint_indices=self.arm_effort_joint_indices_,
            )
        elif self.arm_asset_stiffness_gains_ is not None and self.arm_asset_damping_gains_ is not None:
            target_stiffness = (
                self.arm_asset_stiffness_gains_ * self.arm_joint_drive_stiffness_scale_
            ).astype(np.float32, copy=False)
            target_damping = (
                self.arm_asset_damping_gains_ * self.arm_joint_drive_damping_scale_
            ).astype(np.float32, copy=False)
            self.moca_.set_gains(
                kps=target_stiffness,
                kds=target_damping,
                joint_indices=self.arm_effort_joint_indices_,
            )

        if not self.arm_effort_mode_configured_:
            try:
                stiffnesses, dampings = self.moca_.get_gains(
                    joint_indices=self.arm_effort_joint_indices_
                )
                if self.zero_arm_joint_drive_and_friction_:
                    print(
                        "Configured arm joints for pure effort control:",
                        f"stiffness={np.asarray(stiffnesses).reshape(-1)}",
                        f"damping={np.asarray(dampings).reshape(-1)}",
                    )
                else:
                    print(
                        "Configured scaled arm joint drive gains:",
                        f"stiffness={np.asarray(stiffnesses).reshape(-1)}",
                        f"damping={np.asarray(dampings).reshape(-1)}",
                        f"(scale_kp={self.arm_joint_drive_stiffness_scale_}, "
                        f"scale_kd={self.arm_joint_drive_damping_scale_})",
                    )
            except Exception:
                if self.zero_arm_joint_drive_and_friction_:
                    print("Configured arm joints for pure effort control.")
                else:
                    print(
                        "Configured scaled arm joint drive gains from Isaac asset "
                        f"(scale_kp={self.arm_joint_drive_stiffness_scale_}, "
                        f"scale_kd={self.arm_joint_drive_damping_scale_})."
                    )
            self.arm_effort_mode_configured_ = True

    def _zero_arm_joint_drive_and_friction(self):
        if not self.zero_arm_joint_drive_and_friction_:
            return

        try:
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print("Arm joint physics override skipped: stage is not ready.")
                return

            joint_names = [
                "moca_franka_joint1",
                "moca_franka_joint2",
                "moca_franka_joint3",
                "moca_franka_joint4",
                "moca_franka_joint5",
                "moca_franka_joint6",
                "moca_franka_joint7",
            ]
            attr_values = {
                "drive:angular:physics:stiffness": 0.0,
                "drive:angular:physics:damping": 0.0,
                "drive:angular:physics:targetPosition": 0.0,
                "drive:angular:physics:targetVelocity": 0.0,
                "physxJoint:jointFriction": 0.0,
            }

            overrides_applied = []
            for joint_name in joint_names:
                joint_path = f"{self.prim_paths_expr_}/joints/{joint_name}"
                joint_prim = stage.GetPrimAtPath(joint_path)
                if joint_prim is None or not joint_prim.IsValid():
                    print(f"Arm joint physics override skipped: {joint_path} is not valid.")
                    continue

                joint_overrides = {}
                for attr_name, attr_value in attr_values.items():
                    attr = joint_prim.GetAttribute(attr_name)
                    if not attr.IsValid():
                        continue
                    previous_value = attr.Get()
                    attr.Set(attr_value)
                    joint_overrides[attr_name] = {
                        "before": previous_value,
                        "after": attr_value,
                    }

                if joint_overrides:
                    overrides_applied.append((joint_path, joint_overrides))

            if overrides_applied:
                print("\n========== Arm Joint Physics Overrides ==========")
                for joint_path, joint_overrides in overrides_applied:
                    print(joint_path)
                    for attr_name, attr_report in joint_overrides.items():
                        print(
                            f"  {attr_name}: {attr_report['before']} -> {attr_report['after']}"
                        )
                print("================================================\n")
                self.arm_joint_physics_zeroed_ = True
            elif not self.arm_joint_physics_zeroed_:
                print("Arm joint physics override found no writable drive/friction attributes.")
        except Exception as exc:
            print(f"Failed to override arm joint physics properties: {exc}")

    def _apply_arm_joint_drive_and_friction_scaling(self):
        if self.zero_arm_joint_drive_and_friction_:
            return

        try:
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print("Arm joint drive scaling skipped: stage is not ready.")
                return

            joint_names = [
                "moca_franka_joint1",
                "moca_franka_joint2",
                "moca_franka_joint3",
                "moca_franka_joint4",
                "moca_franka_joint5",
                "moca_franka_joint6",
                "moca_franka_joint7",
            ]

            overrides_applied = []
            for joint_name in joint_names:
                joint_path = f"{self.prim_paths_expr_}/joints/{joint_name}"
                joint_prim = stage.GetPrimAtPath(joint_path)
                if joint_prim is None or not joint_prim.IsValid():
                    continue

                joint_report = {}
                scaling_rules = {
                    "drive:angular:physics:stiffness": self.arm_joint_drive_stiffness_scale_,
                    "drive:angular:physics:damping": self.arm_joint_drive_damping_scale_,
                    "physxJoint:jointFriction": self.arm_joint_friction_scale_,
                }

                for attr_name, scale in scaling_rules.items():
                    attr = joint_prim.GetAttribute(attr_name)
                    if not attr.IsValid():
                        continue
                    previous_value = attr.Get()
                    if previous_value is None:
                        continue
                    try:
                        new_value = float(previous_value) * float(scale)
                    except Exception:
                        continue
                    attr.Set(new_value)
                    joint_report[attr_name] = {
                        "before": previous_value,
                        "after": new_value,
                    }

                if joint_report:
                    overrides_applied.append((joint_path, joint_report))

            if overrides_applied and not self.arm_joint_drive_scaling_applied_:
                print("\n========== Arm Joint Drive/Friction Scaling ==========")
                for joint_path, joint_report in overrides_applied:
                    print(joint_path)
                    for attr_name, attr_values in joint_report.items():
                        print(
                            f"  {attr_name}: {attr_values['before']} -> {attr_values['after']}"
                        )
                print(
                    "  scale_kp="
                    f"{self.arm_joint_drive_stiffness_scale_}, "
                    "scale_kd="
                    f"{self.arm_joint_drive_damping_scale_}, "
                    "scale_friction="
                    f"{self.arm_joint_friction_scale_}"
                )
                print("======================================================\n")
                self.arm_joint_drive_scaling_applied_ = True
        except Exception as exc:
            print(f"Failed to scale arm joint drive/friction properties: {exc}")

    def _dump_joint_physics_properties(self):
        if self.joint_physics_dumped_:
            return

        try:
            stage = omni.usd.get_context().get_stage()
            root_prim = stage.GetPrimAtPath(self.prim_paths_expr_) if stage is not None else None
            if stage is None or root_prim is None or not root_prim.IsValid():
                print("Joint physics dump skipped: stage or robot prim is not ready.")
                return

            joint_name_hints = {
                "moca_franka_joint1",
                "moca_franka_joint2",
                "moca_franka_joint3",
                "moca_franka_joint4",
                "moca_franka_joint5",
                "moca_franka_joint6",
                "moca_franka_joint7",
                "moca_franka_franka_gripper_finger_joint1",
                "moca_franka_franka_gripper_finger_joint2",
            }
            keywords = (
                "drive",
                "stiff",
                "damp",
                "friction",
                "limit",
                "maxforce",
                "target",
                "gear",
                "mimic",
                "tendon",
                "frequency",
                "ratio",
            )

            print("\n========== Joint Physics Properties ==========")
            for prim in Usd.PrimRange(root_prim):
                prim_name = prim.GetName()
                type_name = prim.GetTypeName()
                applied_schemas = []
                try:
                    applied_schemas = list(prim.GetAppliedSchemas())
                except Exception:
                    pass

                is_joint_like = (
                    "joint" in prim_name.lower()
                    or "joint" in type_name.lower()
                    or prim_name in joint_name_hints
                    or any("joint" in schema.lower() for schema in applied_schemas)
                )
                if not is_joint_like:
                    continue

                interesting_attrs = []
                for attr in prim.GetAttributes():
                    attr_name = attr.GetName()
                    if any(keyword in attr_name.lower() for keyword in keywords):
                        try:
                            attr_value = attr.Get()
                        except Exception as exc:
                            attr_value = f"<error reading attribute: {exc}>"
                        interesting_attrs.append((attr_name, attr_value))

                if not interesting_attrs and not applied_schemas:
                    continue

                print(f"\n{prim.GetPath()}  type={type_name}")
                if applied_schemas:
                    print(f"  appliedSchemas = {applied_schemas}")
                for attr_name, attr_value in interesting_attrs:
                    print(f"  {attr_name} = {attr_value}")

            print("==============================================\n")
            self.joint_physics_dumped_ = True
        except Exception as exc:
            print(f"Failed to dump joint physics properties: {exc}")

    def starting(self):
        self._apply_home_state()
        self._dump_joint_physics_properties()
        self._pause_timeline_for_tcp_wait("initial startup")

    def _pause_timeline_for_tcp_wait(self, reason: str):
        if self.timeline_paused_for_tcp_:
            return
        self.timeline_.pause()
        self.timeline_paused_for_tcp_ = True
        print(f"Isaac timeline paused while waiting for TCP/IP ({reason}).")

    def _resume_timeline_for_control(self):
        if not self.timeline_paused_for_tcp_:
            return
        self.timeline_.play()
        self.timeline_paused_for_tcp_ = False
        print("Isaac timeline resumed after TCP/IP control became active.")


    def update(self):

        while simulation_app.is_running():
            if not self.tcpip.is_connected:
                if self.waiting_for_connection_:
                    print("Waiting for TCP/IP client before stepping Isaac world.")
                    self.waiting_for_connection_ = False
                self._pause_timeline_for_tcp_wait("no client connected")
                self.waiting_for_first_command_ = True
                self.base_velocity_cmd_.fill(0.0)
                self.tcpip._fill_struct()
                simulation_app.update()
                time.sleep(0.01)
                continue

            if not self.waiting_for_connection_:
                print("TCP/IP client connected. Isaac state refresh enabled.")
                self.waiting_for_connection_ = True

            if not self.tcpip.has_received_command:
                if self.waiting_for_first_command_:
                    print("TCP/IP client connected. Waiting for first control command before stepping Isaac world.")
                    self.waiting_for_first_command_ = False
                    self._apply_home_state()
                self._pause_timeline_for_tcp_wait("connected but no command yet")
                self.base_velocity_cmd_.fill(0.0)
                self.tcpip._fill_struct()
                simulation_app.update()
                time.sleep(0.01)
                continue

            if not self.waiting_for_first_command_:
                print("First control command received. Isaac world stepping enabled.")
                self.waiting_for_first_command_ = True

            self._resume_timeline_for_control()
            # Apply the latest command before stepping so the current physics
            # step already uses the requested base/arm action.
            command_type, mobile_cmd, arm_cmd, finger, mobile_interface, arm_interface = self.tcpip.get_command()
            self.apply_command(
                command_type,
                mobile_cmd,
                arm_cmd,
                finger,
                mobile_interface,
                arm_interface,
            )
            self.world_.step(render=True)
            self.tcpip._fill_struct()


    def end(self):
        simulation_app.close()

    def apply_command(self, command_type, mobile_cmd, arm_cmd, finger, mobile_interface, arm_interface):
        if command_type == Command_type.RESET_TO_HOME:
            self.reset_to_home()
            return
        self.apply_arm_command(arm_cmd, finger, arm_interface)
        self.apply_mobile_command(mobile_cmd, mobile_interface)

    def reset_to_home(self):
        self._apply_home_state()

    def apply_arm_command(self, arm_cmd, finger, arm_interface):
        """
        根据接口类型执行机械臂命令。
        """
        dt = self.dt_

        self.debug_count_ += 1
        if self.debug_count_ % 100 == 0:
            print(f"arm_cmd = {arm_cmd}, finger = {finger}, arm_interface = {arm_interface}")

        try:
            current_joints = self.moca_.get_joint_positions()
            if len(np.shape(current_joints)) > 1:
                current_joints = current_joints[0]

            current_joints = np.array(current_joints, dtype=np.float64)
            if arm_interface == ArmCommandInterface.POSITION:
                joint_cmd = current_joints.copy()
                joint_cmd[:7] = arm_cmd
                joint_cmd = self._write_finger_targets(joint_cmd, finger)
                self.moca_.set_joint_positions([joint_cmd])
            elif arm_interface == ArmCommandInterface.VELOCITY:
                joint_cmd = current_joints.copy()
                joint_cmd[:7] += arm_cmd * dt
                joint_cmd = self._write_finger_targets(joint_cmd, finger)
                self.moca_.set_joint_positions([joint_cmd])
            elif arm_interface == ArmCommandInterface.TORQUE:
                joint_effort = np.zeros_like(current_joints, dtype=np.float64)
                joint_effort[:7] = arm_cmd
                self.moca_.set_joint_efforts(joint_effort)
                self._apply_finger_targets(finger)
            else:
                pass

        except Exception as e:
            print("Joint command error:", e)

    def apply_mobile_command(self, mobile_cmd, mobile_interface):
        if self.debug_count_ % 100 == 0:
            print(f"mobile_cmd = {mobile_cmd}, mobile_interface = {mobile_interface}")

        try:
            if mobile_interface == MobileCommandInterface.VELOCITY:
                self.base_velocity_cmd_ = np.array(mobile_cmd, dtype=np.float64)
            elif mobile_interface == MobileCommandInterface.WRENCH:
                wrench = np.array(mobile_cmd, dtype=np.float64)
                cmd_acc = (wrench - self.mobile_admittance_damping_ * self.base_velocity_cmd_) / self.mobile_admittance_mass_
                self.base_velocity_cmd_ += cmd_acc * self.dt_
                self.base_velocity_cmd_ = np.clip(
                    self.base_velocity_cmd_,
                    -self.mobile_velocity_limits_,
                    self.mobile_velocity_limits_,
                )
            elif mobile_interface == MobileCommandInterface.POSITION_WORLD:
                target_pose = np.array(mobile_cmd, dtype=np.float64)
                target_position = np.array(
                    [[target_pose[0], target_pose[1], 0.0]], dtype=np.float64
                )
                target_orientation = np.array(
                    [[
                        np.cos(target_pose[2] * 0.5),
                        0.0,
                        0.0,
                        np.sin(target_pose[2] * 0.5),
                    ]],
                    dtype=np.float64,
                )
                self.base_velocity_cmd_.fill(0.0)
                self.moca_.set_linear_velocities(
                    np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
                )
                self.moca_.set_angular_velocities(
                    np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
                )
                self.moca_.set_world_poses(
                    positions=target_position / get_stage_units(),
                    orientations=target_orientation,
                )
                return
            else:
                self.base_velocity_cmd_.fill(0.0)

            self._apply_base_velocity(self.base_velocity_cmd_)
        except Exception as e:
            print("Mobile command error:", e)

    def _apply_base_velocity(self, mobile_cmd):
        _, ori = self.moca_.get_world_poses()

        qw, qx, qy, qz = ori[0]
        yaw = np.arctan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz)
        )

        vx_body = float(mobile_cmd[0])
        vy_body = float(mobile_cmd[1])
        wz = float(mobile_cmd[2])

        vx_world = np.cos(yaw) * vx_body - np.sin(yaw) * vy_body
        vy_world = np.sin(yaw) * vx_body + np.cos(yaw) * vy_body

        self.moca_.set_linear_velocities(
            np.array([[vx_world, vy_world, 0.0]], dtype=np.float64)
        )
        self.moca_.set_angular_velocities(
            np.array([[0.0, 0.0, wz]], dtype=np.float64)
        )

    def _write_finger_targets(self, joint_cmd, finger):
        if len(joint_cmd) > 7:
            joint_cmd[7] = finger
        if len(joint_cmd) > 8:
            joint_cmd[8] = finger
        return joint_cmd

    def _apply_finger_targets(self, finger):
        try:
            self.moca_.set_joint_positions(
                np.array([finger, finger], dtype=np.float64),
                joint_indices=np.array([7, 8], dtype=np.int32),
            )
        except Exception:
            pass





# 初始化
# world.reset()

# moca.set_world_poses(positions=np.array([[0.0, 1.0, 0.0]]) / get_stage_units())
# moca.set_joint_positions([[-1.5, 0.0, 0.0, -1.5, 0.0, 1.5, 0.5, 0.04, 0.04]])

# # 仿真循环
# while simulation_app.is_running():
#     world.step(render=True)
