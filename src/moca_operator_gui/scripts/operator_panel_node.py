#!/usr/bin/env python3

import os
import queue
import re
import shlex
import signal
import subprocess
import threading
import time
import tkinter as tk
import json
import math
import yaml
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from tkinter import ttk
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import rospy

from hrii_robot_msgs.msg import RobotState
from moca_trajectory_generator.msg import ForceCapabilitySettings
from moca_trajectory_generator.srv import StartPlanning
from moca_vlm.msg import MassEstimateRequest, MassEstimateResult
from nav_msgs.msg import Odometry
from PIL import Image as PILImage
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import Empty, Float64, String
from std_srvs.srv import Trigger


@dataclass
class LaunchConfig:
    package: str
    launch_file: str
    args: List[str] = field(default_factory=list)


class OperatorPanel:
    def __init__(self) -> None:
        rospy.init_node("moca_operator_gui", anonymous=False, disable_signals=True)

        self.workspace_root = Path(__file__).resolve().parents[3]
        default_setup_script = self.workspace_root / "env_iit_overlay.sh"
        self.window_title = rospy.get_param("~window_title", "MOCA Operator GUI")
        self.launch_setup_script = str(
            rospy.get_param("~launch_setup_script", str(default_setup_script)))
        self.planner_start_service = rospy.get_param(
            "~planner_start_service", "/target_pose_generator/start_planning")
        self.planner_service_timeout_sec = float(
            rospy.get_param("~planner_service_timeout_sec", 20.0))
        self.reset_to_home_topic = rospy.get_param(
            "~reset_to_home_topic", "/reset_to_home")
        self.simulation_force_shutdown_process_names = [
            str(item)
            for item in rospy.get_param(
                "~simulation_force_shutdown_process_names",
                ["gzserver", "gzclient"],
            )
        ]
        self.force_shutdown_process_names = [
            str(item)
            for item in rospy.get_param(
                "~force_shutdown_process_names",
                [
                    "gzserver",
                    "gzclient",
                    "Moca_controller_node",
                    "moca_joint_impedance_controller_node",
                    "moca_joint_impe",
                    "tcpip_test",
                ],
            )
        ]
        self.simulation_reset_relaunch_delay_sec = float(
            rospy.get_param("~simulation_reset_relaunch_delay_sec", 1.5))
        self.default_vlm_config_path = str(
            rospy.get_param(
                "~default_vlm_config_path",
                str(self.workspace_root / "src/moca_vlm/config/mass_estimator_params.yaml"),
            )
        )
        self.default_planner_config_path = str(
            rospy.get_param(
                "~default_planner_config_path",
                str(self.workspace_root / "src/moca_trajectory_generator/config/target_pose_generator_params.yaml"),
            )
        )
        self.default_force_capability_settings_path = str(
            rospy.get_param(
                "~default_force_capability_settings_path",
                str(self.workspace_root / "src/moca_trajectory_generator/config/force_capability_settings.yaml"),
            )
        )
        self.runtime_vlm_config_path = str(
            Path("/tmp") / "moca_operator_gui_mass_estimator_params.yaml")
        self.runtime_planner_config_path = str(
            Path("/tmp") / "moca_operator_gui_target_pose_generator_params.yaml")
        self.force_capability_settings_topic = str(
            rospy.get_param("~force_capability_settings_topic", "/force_capability_settings"))
        self.teaching_current_sample_topic = str(
            rospy.get_param("~teaching_current_sample_topic", "/moca_teaching/current_sample"))
        self.teaching_directional_force_capacity_topic = str(
            rospy.get_param(
                "~teaching_directional_force_capacity_topic",
                "/moca_teaching/upward_force_capacity"))
        self.teaching_manipulability_topic = str(
            rospy.get_param("~teaching_manipulability_topic", "/moca_teaching/manipulability"))
        self.teaching_record_service = str(
            rospy.get_param("~teaching_record_service", "/moca_teaching_state_monitor/record"))
        self.home_sample_topic = str(
            rospy.get_param("~home_sample_topic", "/moca_home_return/home_sample"))
        self.home_return_status_topic = str(
            rospy.get_param("~home_return_status_topic", "/moca_home_return/status"))
        self.home_return_start_service = str(
            rospy.get_param("~home_return_start_service", "/moca_home_return/start"))
        self.home_return_stop_service = str(
            rospy.get_param("~home_return_stop_service", "/moca_home_return/stop"))
        self.home_return_set_current_service = str(
            rospy.get_param(
                "~home_return_set_current_service",
                "/moca_home_return/set_home_from_current"))
        self.plot_data_recorder_start_service = str(
            rospy.get_param(
                "~plot_data_recorder_start_service",
                "/moca_plot_data_recorder/start"))
        self.plot_data_recorder_wait_start_service = str(
            rospy.get_param(
                "~plot_data_recorder_wait_start_service",
                "/moca_plot_data_recorder/wait_start"))
        self.plot_data_recorder_stop_service = str(
            rospy.get_param(
                "~plot_data_recorder_stop_service",
                "/moca_plot_data_recorder/stop"))
        self.plot_data_recorder_record_point_service = str(
            rospy.get_param(
                "~plot_data_recorder_record_point_service",
                "/moca_plot_data_recorder/record_point"))
        self.plot_data_recorder_undo_point_service = str(
            rospy.get_param(
                "~plot_data_recorder_undo_point_service",
                "/moca_plot_data_recorder/undo_point"))
        self.plot_data_recorder_finish_points_service = str(
            rospy.get_param(
                "~plot_data_recorder_finish_points_service",
                "/moca_plot_data_recorder/finish_points"))
        self.plot_data_recorder_outputs_topic = str(
            rospy.get_param(
                "~plot_data_recorder_outputs_topic",
                "/moca_plot_data_recorder/latest_outputs"))

        self.launch_configs: Dict[str, LaunchConfig] = {
            "simulation": self._load_launch_config("simulation_launch"),
            "joint_simulation": self._load_launch_config("joint_simulation_launch"),
            "hardware": self._load_launch_config("hardware_launch"),
            "joint_hardware": self._load_launch_config("joint_hardware_launch"),
            "rviz": self._load_launch_config("rviz_launch"),
            "vlm": self._load_launch_config("vlm_launch"),
            "realsense": self._load_launch_config("realsense_launch"),
            "pose_marker": self._load_launch_config("pose_marker_launch"),
            "planner": self._load_launch_config("planner_launch"),
            "teaching": self._load_launch_config("teaching_launch"),
            "home_return": self._load_launch_config("home_return_launch"),
            "plot_data_recorder": self._load_launch_config("plot_data_recorder_launch"),
            "franka_state_bridge": self._load_launch_config("franka_state_bridge_launch"),
        }

        self.root = tk.Tk()
        self.root.title(self.window_title)
        self.root.geometry("1080x680")
        self.root.minsize(920, 560)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.processes: Dict[str, subprocess.Popen] = {}
        self.active_robot_id = self._robot_id_from_args(
            self.launch_configs["simulation"].args)
        self.process_labels = {
            "simulation": "仿真控制程序",
            "hardware": "实体机器人链路",
            "rviz": "RViz",
            "vlm": "VLM",
            "realsense": "RealSense相机",
            "pose_marker": "姿态球",
            "planner": "规划器",
            "teaching": "示教采样",
            "home_return": "位置回Home",
            "plot_data_recorder": "绘图数据记录",
            "franka_state_bridge": "Franka只读状态桥",
        }
        self.shutdown_requested = False
        self.status_vars = {
            "simulation": tk.StringVar(value="未启动"),
            "hardware": tk.StringVar(value="未启动"),
            "rviz": tk.StringVar(value="未启动"),
            "vlm": tk.StringVar(value="未启动"),
            "realsense": tk.StringVar(value="未启动"),
            "pose_marker": tk.StringVar(value="未启动"),
            "planner": tk.StringVar(value="未启动"),
            "teaching": tk.StringVar(value="未启动"),
            "home_return": tk.StringVar(value="未启动"),
            "plot_data_recorder": tk.StringVar(value="未启动"),
            "franka_state_bridge": tk.StringVar(value="未启动"),
        }
        self.trajectory_selection = tk.StringVar(value="lift_experiment")
        self.planning_algorithm_mode = tk.StringVar(value="ours")
        self.controller_mode = tk.StringVar(
            value=str(rospy.get_param("~default_controller_mode", "joint")))
        self._init_vlm_state()
        self._init_experiment_config_state()
        self._init_force_capability_state()
        self._init_teaching_home_state()
        self._init_plot_data_state()
        self._init_metric_merge_state()
        self._init_payload_identification_state()
        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.reset_to_home_pub = rospy.Publisher(
            self.reset_to_home_topic,
            Empty,
            queue_size=1)
        self.vlm_request_pub = rospy.Publisher(
            self.vlm_request_topic,
            MassEstimateRequest,
            queue_size=1,
        )
        self.realsense_image_sub = None
        self.realsense_image_topic = ""
        self.realsense_image_lock = threading.Lock()
        self.latest_realsense_image = None
        self.latest_realsense_stamp = rospy.Time(0)
        self.latest_realsense_wall_time_sec = 0.0
        self.realsense_photo_image = None
        self.last_realsense_preview_update_sec = 0.0
        self.force_capability_settings_pub = rospy.Publisher(
            self.force_capability_settings_topic,
            ForceCapabilitySettings,
            queue_size=1,
            latch=True,
        )
        self.home_sample_pub = rospy.Publisher(
            self.home_sample_topic,
            String,
            queue_size=1,
            latch=True,
        )
        self.vlm_result_sub = rospy.Subscriber(
            self.vlm_result_topic,
            MassEstimateResult,
            self._vlm_result_callback,
            queue_size=1,
        )
        self.teaching_current_sample_sub = rospy.Subscriber(
            self.teaching_current_sample_topic,
            String,
            self._teaching_current_sample_callback,
            queue_size=1,
        )
        self.home_return_status_sub = rospy.Subscriber(
            self.home_return_status_topic,
            String,
            self._home_return_status_callback,
            queue_size=1,
        )
        self.teaching_directional_force_capacity_sub = rospy.Subscriber(
            self.teaching_directional_force_capacity_topic,
            Float64,
            self._teaching_directional_force_capacity_callback,
            queue_size=1,
        )
        self.teaching_manipulability_sub = rospy.Subscriber(
            self.teaching_manipulability_topic,
            Float64,
            self._teaching_manipulability_callback,
            queue_size=1,
        )
        self.plot_data_outputs_sub = rospy.Subscriber(
            self.plot_data_recorder_outputs_topic,
            String,
            self._plot_data_outputs_callback,
            queue_size=1,
        )
        self.teaching_robot_state_sub = None
        self._subscribe_teaching_robot_state()
        self.teaching_odom_sub = None
        self._subscribe_teaching_odom()
        self.payload_robot_state_sub = None
        self._subscribe_payload_robot_state()
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        self._build_ui()
        self._log("上位机已启动。当前可以先启动仿真或连接实体机器人。")
        self.root.after(100, self._drain_log_queue)
        self.root.after(500, self._refresh_process_status)
        self.root.after(500, self._watch_ros_shutdown)

    def _load_launch_config(self, param_name: str) -> LaunchConfig:
        config = rospy.get_param("~" + param_name, {})
        args = [
            self._resolve_env_substitutions(str(item))
            for item in config.get("args", [])
        ]
        if param_name in (
            "simulation_launch",
            "joint_simulation_launch",
            "hardware_launch",
            "joint_hardware_launch",
            "rviz_launch",
        ):
            args = self._with_dynamic_rviz_config(args)
        return LaunchConfig(
            package=str(config.get("package", "")),
            launch_file=str(config.get("file", "")),
            args=args,
        )

    def _resolve_env_substitutions(self, value: str) -> str:
        def replace_match(match) -> str:
            env_name = match.group(1)
            default_value = match.group(2).strip()
            return os.environ.get(env_name, default_value)

        return re.sub(
            r"\$\(optenv\s+([A-Za-z_][A-Za-z0-9_]*)\s*([^)]*)\)",
            replace_match,
            value,
        )

    def _with_dynamic_rviz_config(self, args: List[str]) -> List[str]:
        robot_id = self._robot_id_from_args(args)
        rviz_config_path = self._render_robot_id_rviz_config(robot_id)
        rviz_arg = "rviz_config_file:={}".format(rviz_config_path)

        updated_args = []
        replaced = False
        for arg in args:
            if arg.startswith("rviz_config_file:="):
                updated_args.append(rviz_arg)
                replaced = True
            else:
                updated_args.append(arg)

        if not replaced:
            updated_args.append(rviz_arg)
        return updated_args

    def _robot_id_from_args(self, args: List[str]) -> str:
        for arg in args:
            if arg.startswith("robot_id:="):
                robot_id = arg.split(":=", 1)[1].strip()
                if robot_id:
                    return robot_id
        return os.environ.get("ROBOT_ID", "moca_white")

    def _render_robot_id_rviz_config(self, robot_id: str) -> str:
        template_path = self.workspace_root / "src/Moca_controller/rviz/moca_tf_tree.rviz"
        output_path = Path("/tmp") / "moca_operator_gui_{}.rviz".format(robot_id)
        try:
            template_text = template_path.read_text()
            rendered_text = re.sub(
                r"Robot Description: /[^/\s]+/robot_description",
                "Robot Description: /{}/robot_description".format(robot_id),
                template_text,
            )
            rendered_text = re.sub(
                r"Fixed Frame: [A-Za-z0-9_]+_odom",
                "Fixed Frame: {}_odom".format(robot_id),
                rendered_text,
            )
            output_path.write_text(rendered_text)
        except Exception as exc:
            self._log("生成 RViz 配置失败，将使用原始配置：{}".format(exc))
            return str(template_path)
        return str(output_path)

    def _load_yaml_file(self, path: str) -> Dict:
        resolved_path = Path(path).expanduser()
        if not resolved_path.exists():
            return {}
        with open(resolved_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            return {}
        return data

    def _write_yaml_file(self, path: str, data: Dict) -> None:
        resolved_path = Path(path).expanduser()
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        with open(resolved_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)

    def _load_default_gpt_api_key(self) -> str:
        key_path = self.workspace_root / "src/moca_vlm/config/gpt_key.txt"
        try:
            if not key_path.exists():
                return ""
            return key_path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    def _init_vlm_state(self) -> None:
        vlm_config = self._load_yaml_file(self.default_vlm_config_path)
        planner_config = self._load_yaml_file(self.default_planner_config_path)

        self.vlm_request_topic = str(vlm_config.get("request_topic", "/vlm_mass_request"))
        self.vlm_result_topic = str(vlm_config.get("result_topic", "/vlm_mass_result"))

        self.vlm_mode_var = tk.StringVar(
            value="online_api" if bool(vlm_config.get("use_api", False)) else "offline_result")
        self.vlm_model_var = tk.StringVar(value=str(vlm_config.get("model", "")))
        self.vlm_local_base_url_var = tk.StringVar(
            value=str(vlm_config.get("local_base_url", "")))
        self.vlm_api_key_var = tk.StringVar(value=self._load_default_gpt_api_key())
        self.vlm_gemini_api_key_var = tk.StringVar(
            value=str(vlm_config.get("gemini_api_key", "")))
        self.vlm_default_image_path_var = tk.StringVar(
            value=str(vlm_config.get("default_image_path", "")))
        self.vlm_image_source_var = tk.StringVar(
            value=str(vlm_config.get("image_source", "realsense")))
        self.vlm_realsense_image_topic_var = tk.StringVar(
            value=str(vlm_config.get("realsense_image_topic", "/color/image_raw")))
        self.vlm_realsense_capture_service = str(
            vlm_config.get("realsense_capture_service", "/moca_realsense/capture_image"))
        self.vlm_realsense_snapshot_path_var = tk.StringVar(
            value=str(
                vlm_config.get(
                    "realsense_snapshot_path",
                    "/tmp/moca_vlm_realsense_latest.png")))
        self.vlm_result_json_path_var = tk.StringVar(
            value=str(vlm_config.get("result_json_path", "")))
        self.vlm_auto_save_var = tk.BooleanVar(
            value=bool(vlm_config.get("auto_save_api_result", True)))
        self.vlm_reuse_last_result_var = tk.BooleanVar(
            value=bool(vlm_config.get("reuse_last_result", True)))
        self.vlm_offline_mass_var = tk.StringVar(value="0.0")
        self.vlm_offline_range_min_var = tk.StringVar(value="0.0")
        self.vlm_offline_range_max_var = tk.StringVar(value="0.0")
        self.vlm_offline_material_var = tk.StringVar(value="unknown")
        self.vlm_offline_description_var = tk.StringVar(value="")
        self.vlm_offline_confidence_var = tk.StringVar(value="medium")
        self.vlm_offline_reasoning_value = ""

        self._load_offline_result_editor_state(self.vlm_result_json_path_var.get())

        self.planner_request_vlm_var = tk.BooleanVar(
            value=bool(planner_config.get("enable_vlm_mass_request", True)))
        self.planner_vlm_image_path_var = tk.StringVar(
            value=str(planner_config.get("vlm_request_image_path", "")))
        self.planner_vlm_timeout_var = tk.StringVar(
            value=str(planner_config.get("vlm_mass_request_timeout_sec", 90.0)))
        self.planner_vlm_force_refresh_var = tk.BooleanVar(
            value=bool(planner_config.get("vlm_mass_request_force_refresh", False)))
        self.vlm_estimate_policy_var = tk.StringVar(
            value="reestimate"
            if bool(planner_config.get("vlm_mass_request_force_refresh", False))
            else "reuse_last")
        self.planner_default_payload_mass_var = tk.StringVar(
            value=str(planner_config.get("default_payload_mass_kg", 0.0)))

        self.vlm_result_status_var = tk.StringVar(value="尚无结果")
        self.vlm_result_source_var = tk.StringVar(value="-")
        self.vlm_result_model_var = tk.StringVar(value="-")
        self.vlm_result_mass_var = tk.StringVar(value="-")
        self.vlm_result_range_var = tk.StringVar(value="-")
        self.vlm_result_material_var = tk.StringVar(value="-")
        self.vlm_result_confidence_var = tk.StringVar(value="-")
        self.vlm_result_image_var = tk.StringVar(value="-")
        self.vlm_result_json_var = tk.StringVar(value="-")
        self.vlm_result_reasoning_text = ""
        self.vlm_realsense_status_var = tk.StringVar(value="RealSense 未启动")

    def _init_experiment_config_state(self) -> None:
        planner_config = self._load_yaml_file(self.default_planner_config_path)
        self.home_teaching_point_path_var = tk.StringVar(
            value=str(
                rospy.get_param(
                    "~home_teaching_point_path",
                    planner_config.get(
                        "home_teaching_point_path",
                        "$(find moca_trajectory_generator)/teaching_point/home.json"))))
        self.lift_waypoints_path_var = tk.StringVar(
            value=str(
                rospy.get_param(
                    "~teaching_waypoints_path",
                    planner_config.get(
                        "teaching_waypoints_path",
                        "$(find moca_trajectory_generator)/teaching_point/lift/way_points.json"))))
        self.move_home_teaching_point_path_var = tk.StringVar(
            value=str(
                rospy.get_param(
                    "~move_home_teaching_point_path",
                    planner_config.get(
                        "move_home_teaching_point_path",
                        planner_config.get(
                            "home_teaching_point_path",
                            "$(find moca_trajectory_generator)/teaching_point/home.json")))))
        self.move_waypoints_path_var = tk.StringVar(
            value=str(
                rospy.get_param(
                    "~move_teaching_waypoints_path",
                    planner_config.get(
                        "move_teaching_waypoints_path",
                        "$(find moca_trajectory_generator)/teaching_point/move/way_points.json"))))
        self.home_return_duration_var = tk.StringVar(
            value=str(
                rospy.get_param(
                    "~home_return_duration_sec",
                    planner_config.get("home_return_duration_sec", 8.0))))
        self.lift_waypoint_segment_duration_var = tk.StringVar(
            value=str(
                rospy.get_param(
                    "~teaching_waypoint_segment_duration_sec",
                    planner_config.get("teaching_waypoint_segment_duration_sec", 5.0))))
        self.move_home_return_duration_var = tk.StringVar(
            value=str(
                rospy.get_param(
                    "~move_home_return_duration_sec",
                    planner_config.get(
                        "move_home_return_duration_sec",
                        planner_config.get("home_return_duration_sec", 8.0)))))
        self.move_waypoint_segment_duration_var = tk.StringVar(
            value=str(
                rospy.get_param(
                    "~move_teaching_waypoint_segment_duration_sec",
                    planner_config.get(
                        "move_teaching_waypoint_segment_duration_sec",
                        planner_config.get("teaching_waypoint_segment_duration_sec", 5.0)))))
        self.single_point_home_teaching_point_path_var = tk.StringVar(
            value=str(
                rospy.get_param(
                    "~single_point_home_teaching_point_path",
                    planner_config.get(
                        "single_point_home_teaching_point_path",
                        planner_config.get(
                            "home_teaching_point_path",
                            "$(find moca_trajectory_generator)/teaching_point/home.json")))))
        self.single_point_waypoints_path_var = tk.StringVar(
            value=str(
                rospy.get_param(
                    "~single_point_waypoints_path",
                    planner_config.get(
                        "single_point_waypoints_path",
                        "$(find moca_trajectory_generator)/teaching_point/point/way_points.json"))))
        self.single_point_home_return_duration_var = tk.StringVar(
            value=str(
                rospy.get_param(
                    "~single_point_home_return_duration_sec",
                    planner_config.get(
                        "single_point_home_return_duration_sec",
                        planner_config.get("home_return_duration_sec", 8.0)))))
        self.single_point_waypoint_segment_duration_var = tk.StringVar(
            value=str(
                rospy.get_param(
                    "~single_point_teaching_waypoint_segment_duration_sec",
                    planner_config.get(
                        "single_point_teaching_waypoint_segment_duration_sec",
                        planner_config.get("teaching_waypoint_segment_duration_sec", 5.0)))))
        single_point_position = list(
            planner_config.get("single_point_target_position", [0.55, 0.0, 1.10]))
        single_point_orientation = list(
            planner_config.get("single_point_target_orientation_xyzw", [0.0, 0.0, 0.0, 1.0]))
        while len(single_point_position) < 3:
            single_point_position.append(0.0)
        while len(single_point_orientation) < 4:
            single_point_orientation.append(0.0)
        if abs(float(single_point_orientation[3])) < 1e-12 and \
                sum(float(value) * float(value) for value in single_point_orientation) < 1e-12:
            single_point_orientation[3] = 1.0
        self.single_point_x_var = tk.StringVar(value=str(single_point_position[0]))
        self.single_point_y_var = tk.StringVar(value=str(single_point_position[1]))
        self.single_point_z_var = tk.StringVar(value=str(single_point_position[2]))
        self.single_point_qx_var = tk.StringVar(value=str(single_point_orientation[0]))
        self.single_point_qy_var = tk.StringVar(value=str(single_point_orientation[1]))
        self.single_point_qz_var = tk.StringVar(value=str(single_point_orientation[2]))
        self.single_point_qw_var = tk.StringVar(value=str(single_point_orientation[3]))
        self.single_point_move_duration_var = tk.StringVar(
            value=str(planner_config.get("single_point_move_duration_sec", 5.0)))
        self.single_point_selected_index = 0
        self.single_point_selected_point: Optional[Dict[str, Any]] = None
        self.single_point_single_waypoint_path = (
            Path("/tmp") / "moca_operator_gui_single_point_selected_waypoint.json")
        self.single_point_status_var = tk.StringVar(value="尚未选择点 / No point selected.")
        self.move_next_waypoint_index = 0
        self.move_single_waypoint_path = Path("/tmp") / "moca_operator_gui_move_next_waypoint.json"
        self.move_planning_lock = threading.Lock()
        self.move_planning_active = False
        self.experiment_config_status_var = tk.StringVar(
            value="请选择第一页的实验类型 / Select an experiment on page 1.")

    def _resolve_workspace_ros_path(self, value: str) -> Path:
        resolved_text = value.strip()
        if resolved_text.startswith("$(find "):
            close_index = resolved_text.find(")")
            if close_index != -1:
                package_name = resolved_text[len("$(find "):close_index].strip()
                suffix = resolved_text[close_index + 1:].lstrip("/")
                return self.workspace_root / "src" / package_name / suffix
        if resolved_text.startswith("package://"):
            resource = resolved_text[len("package://"):]
            package_name, _, suffix = resource.partition("/")
            return self.workspace_root / "src" / package_name / suffix
        return Path(resolved_text).expanduser()

    def _load_default_force_capability_settings(self) -> Dict[str, Any]:
        path = self._resolve_workspace_ros_path(self.default_force_capability_settings_path)
        config = self._load_yaml_file(str(path))
        settings = config.get("force_capability_settings", config)
        if not isinstance(settings, dict):
            raise ValueError("Invalid force capability settings file: {}".format(path))
        return settings

    def _init_force_capability_state(self) -> None:
        default_settings = self._load_default_force_capability_settings()
        default_joint_names = list(
            default_settings.get(
                "joint_names",
                [
                    "moca_franka_joint1",
                    "moca_franka_joint2",
                    "moca_franka_joint3",
                    "moca_franka_joint4",
                    "moca_franka_joint5",
                    "moca_franka_joint6",
                    "moca_franka_joint7",
                ],
            )
        )
        self.force_capability_joint_names = [str(name) for name in default_joint_names]
        default_limits = list(default_settings.get("arm_torque_limits", []))
        if len(default_limits) != len(self.force_capability_joint_names):
            raise ValueError(
                "force_capability_settings arm_torque_limits size does not match joint_names.")
        configured_limits = rospy.get_param(
            "/force_capability_settings/arm_torque_limits", default_limits)
        if not isinstance(configured_limits, list) or \
                len(configured_limits) != len(self.force_capability_joint_names):
            configured_limits = default_limits
            rospy.set_param("/force_capability_settings/arm_torque_limits", default_limits)
        rospy.set_param("/force_capability_settings/joint_names", self.force_capability_joint_names)

        self.force_capability_default_torque_limits = [
            float(value) for value in default_limits
        ]
        self.force_capability_torque_limit_vars = [
            tk.StringVar(value="{:.6g}".format(float(value)))
            for value in configured_limits
        ]
        self.force_capability_status_var = tk.StringVar(
            value="当前值来自统一 force_capability_settings 默认值。")

    def _init_teaching_home_state(self) -> None:
        self.latest_teaching_sample_json = ""
        self.latest_teaching_sample: Optional[Dict[str, Any]] = None
        self.latest_recorded_teaching_sample_json = ""
        self.latest_recorded_teaching_sample: Optional[Dict[str, Any]] = None
        self.teaching_recorded_points: List[Dict[str, Any]] = []
        self.latest_teaching_robot_state_sample: Optional[Dict[str, Any]] = None
        self.latest_teaching_odom: Optional[Dict[str, Any]] = None
        self.latest_teaching_directional_force_capacity: Optional[float] = None
        self.latest_teaching_manipulability: Optional[float] = None
        self.teaching_robot_state_topic_var = tk.StringVar(
            value=str(
                rospy.get_param(
                    "~teaching_robot_state_topic",
                    "/{}/franka_state_bridge/robot_state".format(self.active_robot_id))))
        self.teaching_odom_topic_var = tk.StringVar(
            value=str(
                rospy.get_param(
                    "~teaching_odom_topic",
                    "/{}/robotnik_base_control/odom".format(self.active_robot_id))))
        self.teaching_points_output_path_var = tk.StringVar(
            value=str(
                rospy.get_param(
                    "~teaching_points_output_path",
                    "output/polytope_ros/teaching_points/teaching_points.json")))
        self.teaching_current_summary_var = tk.StringVar(
            value="等待示教采样节点数据 / Waiting for teaching state monitor.")
        self.teaching_directional_force_capacity_var = tk.StringVar(
            value="竖直向上力输出能力 / Upward force capacity: -")
        self.teaching_manipulability_var = tk.StringVar(
            value="可操作度 / Manipulability: -")
        self.teaching_status_var = tk.StringVar(value="尚未记录 / No record yet.")
        self.home_return_status_var = tk.StringVar(
            value="Home Return 未启动 / Not running. 不影响记录 / Recording is independent.")

    def _init_plot_data_state(self) -> None:
        self.plot_save_force_capacity_var = tk.BooleanVar(value=True)
        self.plot_save_manipulability_var = tk.BooleanVar(value=True)
        self.plot_save_manipulability_ellipsoid_var = tk.BooleanVar(value=True)
        self.plot_save_residual_polytope_var = tk.BooleanVar(value=True)
        self.plot_auto_stop_on_comparison_complete_var = tk.BooleanVar(value=True)
        self.plot_end_time_var = tk.StringVar(value="5.0")
        self.plot_data_output_dir_var = tk.StringVar(value="output/polytope_ros/data")
        self.plot_fig_output_dir_var = tk.StringVar(value="output/polytope_ros/fig")
        self.plot_status_var = tk.StringVar(
            value="尚未开始记录 / Recording has not started.")
        self.last_opened_plot_manifest = ""

    def _init_metric_merge_state(self) -> None:
        data_dir = "output/polytope_ros/data"
        fig_dir = "output/polytope_ros/fig"
        self.merge_force_capacity_input_a_var = tk.StringVar(
            value="{}/force_capacity_20260521_175857.csv".format(data_dir))
        self.merge_force_capacity_input_b_var = tk.StringVar(
            value="{}/force_capacity_20260521_175619.csv".format(data_dir))
        self.merge_force_capacity_output_var = tk.StringVar(
            value="{}/force_capacity_compare_20260521_175857_175619.png".format(fig_dir))
        self.merge_force_capacity_label_a_var = tk.StringVar(value="Run A")
        self.merge_force_capacity_label_b_var = tk.StringVar(value="Run B")
        self.merge_manipulability_input_a_var = tk.StringVar(
            value="{}/manipulability_20260521_175857.csv".format(data_dir))
        self.merge_manipulability_input_b_var = tk.StringVar(
            value="{}/manipulability_20260521_175619.csv".format(data_dir))
        self.merge_manipulability_output_var = tk.StringVar(
            value="{}/manipulability_compare_20260521_175857_175619.png".format(fig_dir))
        self.merge_manipulability_label_a_var = tk.StringVar(value="Run A")
        self.merge_manipulability_label_b_var = tk.StringVar(value="Run B")
        self.metric_merge_status_var = tk.StringVar(
            value="请选择两个 CSV 和输出图片路径 / Select two CSV files and an output image path.")

    def _default_payload_robot_state_topic(self) -> str:
        controller_name = (
            "wb_joint_imp_controller"
            if self._controller_mode_key() == "joint"
            else "wb_cart_imp_controller"
        )
        return "/{}/{}/robot_state".format(self.active_robot_id, controller_name)

    def _init_payload_identification_state(self) -> None:
        self.payload_robot_state_topic_var = tk.StringVar(
            value=str(
                rospy.get_param(
                    "~payload_robot_state_topic",
                    self._default_payload_robot_state_topic())))
        self.payload_wrench_frame_var = tk.StringVar(
            value=str(rospy.get_param("~payload_wrench_frame", "stiffness")))
        self.payload_wrench_data_mode_var = tk.StringVar(
            value=str(rospy.get_param("~payload_wrench_data_mode", "raw")))
        self.payload_gravity_var = tk.StringVar(
            value=str(rospy.get_param("~payload_gravity_acceleration_mps2", 9.81)))
        self.payload_samples: List[Dict[str, Any]] = []
        self.latest_payload_robot_state: Optional[Dict[str, Any]] = None
        self.latest_payload_estimate: Optional[Dict[str, Any]] = None
        self.payload_wrench_bias = np.zeros(6, dtype=float)
        self.payload_latest_summary_var = tk.StringVar(
            value="等待 robot_state 数据 / Waiting for robot_state.")
        self.payload_status_var = tk.StringVar(
            value="尚未记录 / No payload samples yet.")
        self.payload_result_mass_var = tk.StringVar(value="-")
        self.payload_result_com_var = tk.StringVar(value="-")
        self.payload_result_quality_var = tk.StringVar(value="-")
        self.payload_output_dir_var = tk.StringVar(
            value=str(
                rospy.get_param(
                    "~payload_identification_output_dir",
                    "output/polytope_ros/payload_identification")))

    def _load_offline_result_editor_state(self, result_json_path: str) -> None:
        payload = self._read_saved_result_payload(result_json_path)
        if payload is None:
            return

        mass_range = payload.get("mass_kg_range", [])
        self.vlm_offline_mass_var.set(str(payload.get("mass_kg", 0.0)))
        self.vlm_offline_range_min_var.set(
            str(mass_range[0] if len(mass_range) >= 1 else payload.get("mass_kg", 0.0)))
        self.vlm_offline_range_max_var.set(
            str(mass_range[1] if len(mass_range) >= 2 else payload.get("mass_kg", 0.0)))
        self.vlm_offline_material_var.set(str(payload.get("material_guess", "unknown")))
        self.vlm_offline_description_var.set(str(payload.get("object_description", "")))
        self.vlm_offline_confidence_var.set(str(payload.get("confidence", "medium")))
        self.vlm_offline_reasoning_value = str(payload.get("reasoning", ""))

    def _read_saved_result_payload(self, result_json_path: str) -> Optional[Dict[str, Any]]:
        path_text = result_json_path.strip()
        if not path_text:
            return None

        try:
            result_path = self._resolve_workspace_ros_path(path_text)
            if not result_path.exists():
                return None
            with open(result_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            return None

        try:
            if isinstance(data, dict) and "result" in data and isinstance(data["result"], dict):
                return self._coerce_offline_result_dict(data["result"])
            if isinstance(data, dict) and "mass_kg" in data:
                return self._coerce_offline_result_dict(data)
        except Exception:
            return None
        return None

    def _coerce_offline_result_dict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        mass_value = float(payload.get("mass_kg", 0.0))
        raw_range = payload.get("mass_kg_range", [])
        if isinstance(raw_range, (list, tuple)) and len(raw_range) >= 2:
            mass_range = [float(raw_range[0]), float(raw_range[1])]
        else:
            mass_range = [mass_value, mass_value]

        return {
            "mass_kg": mass_value,
            "mass_kg_range": mass_range,
            "material_guess": str(payload.get("material_guess", "unknown")),
            "object_description": str(payload.get("object_description", "")),
            "confidence": str(payload.get("confidence", "medium")),
            "reasoning": str(payload.get("reasoning", "")),
        }

    def _build_offline_result_payload(self) -> Dict[str, Any]:
        mass_value = float(self.vlm_offline_mass_var.get().strip() or "0.0")
        range_min = float(self.vlm_offline_range_min_var.get().strip() or str(mass_value))
        range_max = float(self.vlm_offline_range_max_var.get().strip() or str(mass_value))
        if range_min > range_max:
            raise ValueError("离线质量范围的最小值不能大于最大值。")

        return {
            "mass_kg": mass_value,
            "mass_kg_range": [range_min, range_max],
            "material_guess": self.vlm_offline_material_var.get().strip() or "unknown",
            "object_description": self.vlm_offline_description_var.get().strip(),
            "confidence": self.vlm_offline_confidence_var.get().strip() or "medium",
            "reasoning": self._get_offline_reasoning_text().strip(),
        }

    def _get_offline_reasoning_text(self) -> str:
        if hasattr(self, "vlm_offline_reasoning_editor"):
            return self.vlm_offline_reasoning_editor.get("1.0", tk.END).strip()
        return self.vlm_offline_reasoning_value.strip()

    def _set_offline_reasoning_text(self, text: str) -> None:
        self.vlm_offline_reasoning_value = text
        if hasattr(self, "vlm_offline_reasoning_editor"):
            self.vlm_offline_reasoning_editor.delete("1.0", tk.END)
            self.vlm_offline_reasoning_editor.insert(tk.END, text)

    def _vlm_uses_realsense(self) -> bool:
        return self.vlm_image_source_var.get() == "realsense"

    def _realsense_snapshot_path(self) -> str:
        path_text = self.vlm_realsense_snapshot_path_var.get().strip()
        return path_text if path_text else "/tmp/moca_vlm_realsense_latest.png"

    def _realsense_snapshot_path_for_request(self, request_label: str) -> Path:
        base_path = self._resolve_workspace_ros_path(self._realsense_snapshot_path())
        suffix = base_path.suffix if base_path.suffix else ".png"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", request_label.strip())
        safe_label = safe_label.strip("._-")
        stem_parts = [base_path.stem, timestamp]
        if safe_label:
            stem_parts.append(safe_label)
        return base_path.with_name("{}{}".format("_".join(stem_parts), suffix))

    def _fixed_vlm_request_image_path(self) -> str:
        planner_image_path = self.planner_vlm_image_path_var.get().strip()
        return planner_image_path if planner_image_path else self.vlm_default_image_path_var.get().strip()

    def _selected_vlm_request_image_path(
        self,
        wait_for_realsense: bool = False,
        request_label: str = "",
    ) -> str:
        if not self._vlm_uses_realsense():
            return self._fixed_vlm_request_image_path()
        return ""

    def _clear_latest_realsense_image(self) -> None:
        with self.realsense_image_lock:
            self.latest_realsense_image = None
            self.latest_realsense_stamp = rospy.Time(0)
            self.latest_realsense_wall_time_sec = 0.0

    def _ensure_realsense_preview_subscription(self) -> None:
        topic = self.vlm_realsense_image_topic_var.get().strip() or "/color/image_raw"
        if self.realsense_image_sub is not None and self.realsense_image_topic == topic:
            return

        if self.realsense_image_sub is not None:
            self.realsense_image_sub.unregister()
            self.realsense_image_sub = None

        self.realsense_image_topic = topic
        self.realsense_image_sub = rospy.Subscriber(
            topic,
            RosImage,
            self._realsense_image_callback,
            queue_size=1,
        )
        self._set_tk_var(
            self.vlm_realsense_status_var,
            "等待 RealSense 图像 / Waiting for camera image: {}".format(topic))

    def _ros_image_to_pil(self, msg: RosImage):
        encoding = msg.encoding.lower()
        channel_count = 3
        raw_mode = "RGB"
        image_mode = "RGB"

        if encoding == "rgb8":
            pass
        elif encoding == "bgr8":
            raw_mode = "BGR"
        elif encoding == "rgba8":
            channel_count = 4
            raw_mode = "RGBA"
            image_mode = "RGBA"
        elif encoding == "bgra8":
            channel_count = 4
            raw_mode = "BGRA"
            image_mode = "RGBA"
        elif encoding == "mono8":
            channel_count = 1
            raw_mode = "L"
            image_mode = "L"
        else:
            raise ValueError("暂不支持的 RealSense 图像编码：{}".format(msg.encoding))

        expected_row_bytes = int(msg.width) * channel_count
        raw_data = bytes(msg.data)
        if int(msg.step) != expected_row_bytes:
            raw_data = b"".join(
                raw_data[row * int(msg.step):row * int(msg.step) + expected_row_bytes]
                for row in range(int(msg.height)))

        image = PILImage.frombytes(
            image_mode,
            (int(msg.width), int(msg.height)),
            raw_data,
            "raw",
            raw_mode,
        )
        return image.convert("RGB")

    def _realsense_image_callback(self, msg: RosImage) -> None:
        try:
            image = self._ros_image_to_pil(msg)
        except Exception as exc:
            self.root.after(
                0,
                lambda: self.vlm_realsense_status_var.set(
                    "RealSense 图像转换失败：{}".format(exc)))
            return

        with self.realsense_image_lock:
            self.latest_realsense_image = image.copy()
            self.latest_realsense_stamp = msg.header.stamp
            self.latest_realsense_wall_time_sec = time.time()

        now = time.time()
        if now - self.last_realsense_preview_update_sec < 0.20:
            return
        self.last_realsense_preview_update_sec = now
        self.root.after(0, lambda image=image.copy(): self._update_realsense_preview(image))

    def _update_realsense_preview(self, image) -> None:
        if not hasattr(self, "vlm_realsense_preview_label"):
            return

        preview = image.copy()
        preview.thumbnail((360, 270), PILImage.LANCZOS)
        header = "P6 {} {} 255\n".format(preview.width, preview.height).encode("ascii")
        self.realsense_photo_image = tk.PhotoImage(
            data=header + preview.convert("RGB").tobytes(),
            format="PPM")
        self.vlm_realsense_preview_label.configure(
            image=self.realsense_photo_image,
            text="")
        self.vlm_realsense_status_var.set(
            "RealSense 图像已更新 / Camera image updated: {}x{}".format(
                image.width,
                image.height))

    def _save_latest_realsense_snapshot(
        self,
        wait_timeout_sec: float = 0.0,
        log_missing: bool = True,
        request_label: str = "",
        min_wall_time_sec: float = 0.0,
    ) -> str:
        deadline = time.time() + max(0.0, wait_timeout_sec)
        image = None
        while True:
            with self.realsense_image_lock:
                if (
                    self.latest_realsense_image is not None
                    and self.latest_realsense_wall_time_sec >= min_wall_time_sec
                ):
                    image = self.latest_realsense_image.copy()
                    break
            if time.time() >= deadline:
                break
            time.sleep(0.05)

        if image is None:
            if log_missing:
                self._log("尚未收到 RealSense 图像，VLM 将使用已有快照路径。")
            return ""

        latest_path = self._resolve_workspace_ros_path(self._realsense_snapshot_path())
        snapshot_path = (
            self._realsense_snapshot_path_for_request(request_label)
            if request_label
            else latest_path)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(str(snapshot_path))

        if request_label and snapshot_path != latest_path:
            latest_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(str(latest_path))

        self._set_tk_var(
            self.vlm_realsense_status_var,
            "已保存 RealSense 快照 / Snapshot saved: {}".format(snapshot_path))
        self._log("已保存本次 RealSense 请求图片：{}".format(snapshot_path))
        return str(snapshot_path)

    def _start_realsense_for_vlm(self) -> bool:
        if self._launch_process("realsense"):
            time.sleep(0.5)
            process = self.processes.get("realsense")
            if process is not None and process.poll() is not None:
                self._set_tk_var(
                    self.status_vars["realsense"],
                    "已退出({})".format(process.returncode))
                self.processes.pop("realsense", None)
                self._log("RealSense launch 已退出，未能启动相机。")
                return False
            self._log("RealSense 相机节点已启动或已在运行，图像窗口由相机节点本地显示。")
            return True
        return False

    def _planner_param_namespace(self) -> str:
        service_name = self.planner_start_service.rstrip("/")
        namespace, _, _service = service_name.rpartition("/")
        return namespace if namespace else "/target_pose_generator"

    def _vlm_force_refresh_enabled(self) -> bool:
        if hasattr(self, "vlm_estimate_policy_var"):
            force_refresh = self.vlm_estimate_policy_var.get() == "reestimate"
            if hasattr(self, "planner_vlm_force_refresh_var"):
                self.planner_vlm_force_refresh_var.set(force_refresh)
            if hasattr(self, "vlm_reuse_last_result_var"):
                self.vlm_reuse_last_result_var.set(not force_refresh)
            return force_refresh
        return bool(self.planner_vlm_force_refresh_var.get())

    def _publish_planner_vlm_runtime_params(self, image_path: str) -> None:
        namespace = self._planner_param_namespace()
        rospy.set_param(namespace + "/vlm_request_image_path", image_path)
        rospy.set_param(namespace + "/vlm_offline_mode", self.vlm_mode_var.get() != "online_api")
        rospy.set_param(
            namespace + "/vlm_mass_request_force_refresh",
            self._vlm_force_refresh_enabled())
        rospy.set_param(
            namespace + "/vlm_mass_request_timeout_sec",
            float(self.planner_vlm_timeout_var.get().strip() or "30.0"))
        self._publish_vlm_node_runtime_params()

    def _publish_vlm_node_runtime_params(self) -> None:
        rospy.set_param(
            "/moca_vlm_mass_estimator/reuse_last_result",
            not self._vlm_force_refresh_enabled())

    def _publish_planner_vlm_mode_runtime_param(self) -> None:
        namespace = self._planner_param_namespace()
        rospy.set_param(namespace + "/vlm_offline_mode", self.vlm_mode_var.get() != "online_api")

    def _publish_planner_algorithm_runtime_param(self, algorithm_name: Optional[str] = None) -> None:
        namespace = self._planner_param_namespace()
        rospy.set_param(
            namespace + "/optimizer_comparison_algorithm",
            algorithm_name or self.planning_algorithm_mode.get())

    def _experiment_kind_from_selection(self) -> str:
        if self.trajectory_selection.get() == "move_experiment":
            return "move"
        if self.trajectory_selection.get() == "single_point_optimization":
            return "point"
        return "lift"

    def _selected_experiment_values(self, experiment_kind: Optional[str] = None):
        kind = experiment_kind or self._experiment_kind_from_selection()
        if kind == "move":
            return (
                self.move_home_teaching_point_path_var.get().strip(),
                self.move_waypoints_path_var.get().strip(),
                self.move_waypoint_segment_duration_var.get().strip() or "5.0",
                self.move_home_return_duration_var.get().strip() or "8.0",
            )
        if kind == "point":
            return (
                self.single_point_home_teaching_point_path_var.get().strip(),
                self.single_point_waypoints_path_var.get().strip(),
                self.single_point_waypoint_segment_duration_var.get().strip() or "5.0",
                self.single_point_home_return_duration_var.get().strip() or "8.0",
            )
        return (
            self.home_teaching_point_path_var.get().strip(),
            self.lift_waypoints_path_var.get().strip(),
            self.lift_waypoint_segment_duration_var.get().strip() or "5.0",
            self.home_return_duration_var.get().strip() or "8.0",
        )

    def _publish_experiment_runtime_params(
        self,
        experiment_kind: Optional[str] = None,
        waypoints_path_override: Optional[str] = None,
    ) -> None:
        namespace = self._planner_param_namespace()
        self._publish_planner_vlm_mode_runtime_param()
        home_path, waypoints_path, segment_text, home_text = self._selected_experiment_values(
            experiment_kind)
        if waypoints_path_override is not None:
            waypoints_path = waypoints_path_override
        segment_duration = float(segment_text)
        home_duration = float(home_text)
        if segment_duration <= 0.0:
            raise ValueError("间隔点规划时间必须大于 0。")
        if home_duration <= 0.0:
            raise ValueError("Home规划时间必须大于 0。")
        rospy.set_param(namespace + "/home_teaching_point_path", home_path)
        rospy.set_param(namespace + "/teaching_waypoints_path", waypoints_path)
        rospy.set_param(namespace + "/home_return_duration_sec", home_duration)
        rospy.set_param(
            namespace + "/teaching_waypoint_segment_duration_sec",
            segment_duration)
        self._set_tk_var(
            self.experiment_config_status_var,
            "已下发实验参数 / Params sent: segment {:.3g}s".format(segment_duration))

    def _single_point_target_values(self):
        position = [
            float(self.single_point_x_var.get().strip()),
            float(self.single_point_y_var.get().strip()),
            float(self.single_point_z_var.get().strip()),
        ]
        orientation = [
            float(self.single_point_qx_var.get().strip()),
            float(self.single_point_qy_var.get().strip()),
            float(self.single_point_qz_var.get().strip()),
            float(self.single_point_qw_var.get().strip()),
        ]
        duration = float(self.single_point_move_duration_var.get().strip() or "5.0")
        if duration <= 0.0:
            raise ValueError("单点优化规划时间必须大于 0。")
        quat_norm = math.sqrt(sum(value * value for value in orientation))
        if quat_norm <= 1e-9:
            raise ValueError("单点优化目标四元数不能为零。")
        orientation = [value / quat_norm for value in orientation]
        return position, orientation, duration

    def _single_point_target_values_from_point(self, point: Dict[str, Any]):
        pose = point.get("ee_pose", {}).get("pose", {})
        position_value = pose.get("position", {})
        orientation_value = pose.get("orientation", {})
        if not isinstance(position_value, dict) or not isinstance(orientation_value, dict):
            raise ValueError("选中的单点没有合法 ee_pose.pose 字段。")

        position = [
            float(position_value.get("x", 0.0)),
            float(position_value.get("y", 0.0)),
            float(position_value.get("z", 0.0)),
        ]
        orientation = [
            float(orientation_value.get("x", 0.0)),
            float(orientation_value.get("y", 0.0)),
            float(orientation_value.get("z", 0.0)),
            float(orientation_value.get("w", 1.0)),
        ]
        duration = float(self.single_point_move_duration_var.get().strip() or "5.0")
        if duration <= 0.0:
            raise ValueError("单点优化规划时间必须大于 0。")
        quat_norm = math.sqrt(sum(value * value for value in orientation))
        if quat_norm <= 1e-9:
            raise ValueError("选中的单点四元数不能为零。")
        orientation = [value / quat_norm for value in orientation]
        return position, orientation, duration

    def _single_point_whole_body_values_from_point(self, point: Dict[str, Any]) -> List[float]:
        arm_positions = point.get("arm_joint_positions")
        if arm_positions is None and isinstance(point.get("sample"), dict):
            arm_positions = point["sample"].get("arm_joint_positions")
        if not isinstance(arm_positions, list) or len(arm_positions) != 7:
            raise ValueError("选中的单点没有合法的 7 维 arm_joint_positions。")

        odom = point.get("odom")
        if odom is None and isinstance(point.get("sample"), dict):
            odom = point["sample"].get("odom")
        planar = odom.get("planar", {}) if isinstance(odom, dict) else {}
        if not isinstance(planar, dict):
            raise ValueError("选中的单点没有合法的 odom.planar。")

        return [
            float(planar.get("x", 0.0)),
            float(planar.get("y", 0.0)),
            float(planar.get("yaw", 0.0)),
        ] + [float(value) for value in arm_positions]

    def _publish_single_point_runtime_params(
        self,
        point: Optional[Dict[str, Any]] = None,
    ) -> None:
        namespace = self._planner_param_namespace()
        if point is None:
            position, orientation, duration = self._single_point_target_values()
            whole_body = None
        else:
            position, orientation, duration = self._single_point_target_values_from_point(point)
            whole_body = self._single_point_whole_body_values_from_point(point)
        rospy.set_param(namespace + "/single_point_target_position", position)
        rospy.set_param(namespace + "/single_point_target_orientation_xyzw", orientation)
        rospy.set_param(namespace + "/single_point_move_duration_sec", duration)
        self._publish_planner_vlm_mode_runtime_param()
        whole_body_param = namespace + "/single_point_target_whole_body"
        if whole_body is None:
            if rospy.has_param(whole_body_param):
                rospy.delete_param(whole_body_param)
        else:
            rospy.set_param(whole_body_param, whole_body)
        self._set_tk_var(
            self.experiment_config_status_var,
            "已下发单点优化目标 / Single-point target sent: "
            "p=[{:.3f}, {:.3f}, {:.3f}]".format(*position))

    def _write_offline_result_json_if_needed(self, config: Dict[str, Any]) -> None:
        if bool(config.get("use_api", False)):
            return

        result_json_path = str(config.get("result_json_path", "")).strip()
        if not result_json_path:
            raise ValueError("离线结果模式需要设置结果 JSON 路径。")

        wrapped_result = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "source": "gui_offline_manual",
            "model": config.get("model", "") or "offline-manual",
            "image_path": str(config.get("default_image_path", "")).strip(),
            "result": self._build_offline_result_payload(),
        }

        output_path = self._resolve_workspace_ros_path(result_json_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(wrapped_result, handle, indent=2, ensure_ascii=False)

    def _build_runtime_vlm_config(self) -> Dict:
        config = self._load_yaml_file(self.default_vlm_config_path)
        config["request_topic"] = self.vlm_request_topic
        config["result_topic"] = self.vlm_result_topic
        config["use_api"] = self.vlm_mode_var.get() == "online_api"
        config["model"] = self.vlm_model_var.get().strip()
        config["api_key"] = self.vlm_api_key_var.get()
        config["gemini_api_key"] = self.vlm_gemini_api_key_var.get()
        config["local_base_url"] = self.vlm_local_base_url_var.get().strip()
        config["image_source"] = self.vlm_image_source_var.get()
        config["realsense_image_topic"] = self.vlm_realsense_image_topic_var.get().strip()
        config["realsense_capture_service"] = self.vlm_realsense_capture_service
        config["realsense_snapshot_path"] = self._realsense_snapshot_path()
        config["default_image_path"] = (
            ""
            if self._vlm_uses_realsense()
            else self.vlm_default_image_path_var.get().strip())
        config["result_json_path"] = self.vlm_result_json_path_var.get().strip()
        config["auto_save_api_result"] = bool(self.vlm_auto_save_var.get())
        config["reuse_last_result"] = not self._vlm_force_refresh_enabled()
        return config

    def _build_runtime_planner_config(self) -> Dict:
        config = self._load_yaml_file(self.default_planner_config_path)
        config["enable_vlm_mass_request"] = True
        config["vlm_request_image_path"] = (
            "" if self._vlm_uses_realsense() else self._selected_vlm_request_image_path())
        config["vlm_mass_request_timeout_sec"] = float(
            self.planner_vlm_timeout_var.get().strip() or "90.0")
        config["vlm_timeout_uses_default_payload_mass"] = False
        config["vlm_mass_request_force_refresh"] = bool(
            self._vlm_force_refresh_enabled())
        config["vlm_offline_mode"] = self.vlm_mode_var.get() != "online_api"
        config["default_payload_mass_kg"] = float(
            self.planner_default_payload_mass_var.get().strip() or "0.0")
        config["home_teaching_point_path"] = self.home_teaching_point_path_var.get().strip()
        config["teaching_waypoints_path"] = self.lift_waypoints_path_var.get().strip()
        config["home_return_duration_sec"] = float(
            self.home_return_duration_var.get().strip() or "8.0")
        config["teaching_waypoint_segment_duration_sec"] = float(
            self.lift_waypoint_segment_duration_var.get().strip() or "5.0")
        config["move_home_teaching_point_path"] = self.move_home_teaching_point_path_var.get().strip()
        config["move_teaching_waypoints_path"] = self.move_waypoints_path_var.get().strip()
        config["move_home_return_duration_sec"] = float(
            self.move_home_return_duration_var.get().strip() or "8.0")
        config["move_teaching_waypoint_segment_duration_sec"] = float(
            self.move_waypoint_segment_duration_var.get().strip() or "5.0")
        config["single_point_home_teaching_point_path"] = (
            self.single_point_home_teaching_point_path_var.get().strip())
        config["single_point_waypoints_path"] = (
            self.single_point_waypoints_path_var.get().strip())
        config["single_point_home_return_duration_sec"] = float(
            self.single_point_home_return_duration_var.get().strip() or "8.0")
        config["single_point_teaching_waypoint_segment_duration_sec"] = float(
            self.single_point_waypoint_segment_duration_var.get().strip() or "5.0")
        single_position, single_orientation, single_duration = self._single_point_target_values()
        config["single_point_target_position"] = single_position
        config["single_point_target_orientation_xyzw"] = single_orientation
        config["single_point_move_duration_sec"] = single_duration
        return config

    def _write_runtime_vlm_config(self) -> None:
        config = self._build_runtime_vlm_config()
        self._write_offline_result_json_if_needed(config)
        self._write_yaml_file(self.runtime_vlm_config_path, config)

    def _write_runtime_planner_config(self) -> None:
        self._write_yaml_file(
            self.runtime_planner_config_path, self._build_runtime_planner_config())

    def _write_runtime_vlm_configs(self) -> None:
        self._write_runtime_vlm_config()
        self._write_runtime_planner_config()

    def _set_or_replace_launch_arg(self, args: List[str], key: str, value: str) -> List[str]:
        prefix = key + ":="
        updated_args = []
        replaced = False
        for arg in args:
            if arg.startswith(prefix):
                updated_args.append(prefix + value)
                replaced = True
            else:
                updated_args.append(arg)
        if not replaced:
            updated_args.append(prefix + value)
        return updated_args

    def _controller_mode_key(self) -> str:
        mode = self.controller_mode.get().strip().lower()
        if mode not in ("cartesian", "joint"):
            return "joint"
        return mode

    def _controller_mode_label(self) -> str:
        if self._controller_mode_key() == "joint":
            return "关节阻抗控制 / Joint impedance"
        return "笛卡尔阻抗控制 / Cartesian impedance"

    def _launch_config_for_key(self, key: str) -> LaunchConfig:
        if key == "simulation" and self._controller_mode_key() == "joint":
            return self.launch_configs.get("joint_simulation", self.launch_configs[key])
        if key == "hardware" and self._controller_mode_key() == "joint":
            return self.launch_configs.get("joint_hardware", self.launch_configs[key])
        return self.launch_configs[key]

    def _prepare_launch_args(self, key: str, args: List[str]) -> List[str]:
        prepared_args = list(args)
        if key in ("simulation", "hardware", "planner", "teaching", "home_return"):
            if key == "planner" and not any(arg.startswith("robot_id:=") for arg in prepared_args):
                robot_id = self.active_robot_id
            else:
                robot_id = self._robot_id_from_args(prepared_args)
            prepared_args = self._set_or_replace_launch_arg(
                prepared_args, "robot_id", robot_id)
            if key in ("simulation", "hardware"):
                self.active_robot_id = robot_id
            if key == "planner":
                prepared_args = self._set_or_replace_launch_arg(
                    prepared_args, "world_frame", "{}_odom".format(robot_id))
        if key == "rviz":
            rviz_config_path = self._render_robot_id_rviz_config(self.active_robot_id)
            prepared_args = self._set_or_replace_launch_arg(
                prepared_args, "robot_id", self.active_robot_id)
            prepared_args = self._set_or_replace_launch_arg(
                prepared_args, "rviz_config_file", rviz_config_path)
        if key == "franka_state_bridge":
            prepared_args = self._set_or_replace_launch_arg(
                prepared_args, "robot_id", self.active_robot_id)
        if key == "vlm":
            self._write_runtime_vlm_config()
            prepared_args = self._set_or_replace_launch_arg(
                prepared_args, "config_file", self.runtime_vlm_config_path)
        if key == "planner":
            self._write_runtime_vlm_configs()
            prepared_args = self._set_or_replace_launch_arg(
                prepared_args, "config_file", self.runtime_planner_config_path)
            prepared_args = self._set_or_replace_launch_arg(
                prepared_args, "vlm_config_file", self.runtime_vlm_config_path)
            prepared_args = self._set_or_replace_launch_arg(
                prepared_args,
                "enable_vlm_mass_request",
                "true",
            )
        return prepared_args

    def _update_vlm_reasoning_text(self, text: str) -> None:
        self.vlm_result_reasoning_text = text
        if hasattr(self, "vlm_reasoning_text"):
            self.vlm_reasoning_text.configure(state=tk.NORMAL)
            self.vlm_reasoning_text.delete("1.0", tk.END)
            self.vlm_reasoning_text.insert(tk.END, text)
            self.vlm_reasoning_text.configure(state=tk.DISABLED)

    def _vlm_result_callback(self, msg: MassEstimateResult) -> None:
        def apply_result() -> None:
            if msg.success:
                self.vlm_result_status_var.set("成功")
                self.vlm_result_source_var.set(msg.source or "-")
                self.vlm_result_model_var.set(msg.model or "-")
                self.vlm_result_mass_var.set("{:.4f} kg".format(msg.mass_kg))
                if msg.mass_kg_range:
                    self.vlm_result_range_var.set(
                        ", ".join("{:.4f}".format(v) for v in msg.mass_kg_range))
                else:
                    self.vlm_result_range_var.set("-")
                self.vlm_result_material_var.set(msg.material_guess or "-")
                self.vlm_result_confidence_var.set(msg.confidence or "-")
                self.vlm_result_image_var.set(msg.image_path or "-")
                self.vlm_result_json_var.set(msg.result_json_path or "-")
                self._update_vlm_reasoning_text(msg.reasoning or "")
                self._log(
                    "VLM 结果已更新：{:.4f} kg，来源={}。".format(
                        msg.mass_kg, msg.source or "unknown"))
            else:
                self.vlm_result_status_var.set("失败")
                self.vlm_result_source_var.set(msg.source or "-")
                self.vlm_result_model_var.set(msg.model or "-")
                self.vlm_result_mass_var.set("-")
                self.vlm_result_range_var.set("-")
                self.vlm_result_material_var.set("-")
                self.vlm_result_confidence_var.set("-")
                self.vlm_result_image_var.set(msg.image_path or "-")
                self.vlm_result_json_var.set(msg.result_json_path or "-")
                self._update_vlm_reasoning_text(msg.error_message or "")
                self._log("VLM 请求失败：{}".format(msg.error_message or "unknown error"))

        self.root.after(0, apply_result)

    def _format_float_list(self, values: List[float], precision: int = 5) -> str:
        return "[" + ", ".join(
            "{:.{}f}".format(float(value), precision) for value in values
        ) + "]"

    def _format_teaching_sample(self, sample: Dict[str, Any]) -> str:
        record_index = sample.get("record_index")
        arm_positions = sample.get("arm_joint_positions", [])
        odom_planar = sample.get("odom", {}).get("planar", {})
        odom_source = sample.get("odom", {}).get("source", "unknown")
        ee_pose = sample.get("ee_pose", {}).get("pose", {})
        ee_position = ee_pose.get("position", {})
        ee_orientation = ee_pose.get("orientation", {})
        vector = sample.get("vector", [])

        lines = []
        if record_index is not None:
            lines.append("record_index: {}".format(record_index))
        lines.append("stamp: {:.3f}".format(float(sample.get("stamp", 0.0))))
        upward_force_capacity = self.latest_teaching_directional_force_capacity
        if upward_force_capacity is None and sample.get("upward_force_capacity_N") is not None:
            upward_force_capacity = float(sample.get("upward_force_capacity_N", 0.0))
        manipulability = self.latest_teaching_manipulability
        if manipulability is None and sample.get("manipulability") is not None:
            manipulability = float(sample.get("manipulability", 0.0))
        if upward_force_capacity is not None:
            lines.append(
                "upward_force_capacity_N: {:.3f}".format(
                    upward_force_capacity))
        if manipulability is not None:
            lines.append(
                "manipulability: {:.6f}".format(
                    manipulability))
        lines.append("arm_joint_positions:")
        lines.append(self._format_float_list(arm_positions))
        lines.append(
            "odom_planar (source={}): [x={:.5f}, y={:.5f}, yaw={:.5f}]".format(
                odom_source,
                float(odom_planar.get("x", 0.0)),
                float(odom_planar.get("y", 0.0)),
                float(odom_planar.get("yaw", 0.0)),
            )
        )
        lines.append(
            "ee_pose.position: [x={:.5f}, y={:.5f}, z={:.5f}]".format(
                float(ee_position.get("x", 0.0)),
                float(ee_position.get("y", 0.0)),
                float(ee_position.get("z", 0.0)),
            )
        )
        lines.append(
            "ee_pose.orientation: [x={:.5f}, y={:.5f}, z={:.5f}, w={:.5f}]".format(
                float(ee_orientation.get("x", 0.0)),
                float(ee_orientation.get("y", 0.0)),
                float(ee_orientation.get("z", 0.0)),
                float(ee_orientation.get("w", 1.0)),
            )
        )
        lines.append("vector:")
        lines.append(self._format_float_list(vector))
        return "\n".join(lines)

    def _set_text_widget(self, widget_name: str, text: str) -> None:
        if not hasattr(self, widget_name):
            return
        widget = getattr(self, widget_name)
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        widget.configure(state=tk.DISABLED)

    def _append_text_widget(self, widget_name: str, text: str) -> None:
        if not hasattr(self, widget_name):
            return
        widget = getattr(self, widget_name)
        widget.configure(state=tk.NORMAL)
        if widget.get("1.0", tk.END).strip():
            widget.insert(tk.END, "\n\n")
        widget.insert(tk.END, text)
        widget.see(tk.END)
        widget.configure(state=tk.DISABLED)

    def _teaching_current_sample_callback(self, msg: String) -> None:
        try:
            sample = json.loads(msg.data)
            formatted = self._format_teaching_sample(sample)
        except Exception as exc:
            formatted = "示教采样解析失败 / Failed to parse teaching sample: {}".format(exc)
            sample = None

        def apply_sample() -> None:
            self.latest_teaching_sample_json = msg.data
            self.latest_teaching_sample = sample
            if sample.get("upward_force_capacity_N") is not None:
                force_capacity = float(sample.get("upward_force_capacity_N", 0.0))
                if math.isfinite(force_capacity):
                    self.latest_teaching_directional_force_capacity = force_capacity
                    self.teaching_directional_force_capacity_var.set(
                        "竖直向上力输出能力 / Upward force capacity: {:.3f} N".format(
                            force_capacity))
            if sample.get("manipulability") is not None:
                manipulability = float(sample.get("manipulability", 0.0))
                if math.isfinite(manipulability):
                    self.latest_teaching_manipulability = manipulability
                    self.teaching_manipulability_var.set(
                        "可操作度 / Manipulability: {:.6f}".format(manipulability))
            self.teaching_current_summary_var.set("当前示教采样已更新 / Current sample updated.")
            self._set_text_widget("teaching_current_text", formatted)

        self.root.after(0, apply_sample)

    def _teaching_directional_force_capacity_callback(self, msg: Float64) -> None:
        value = float(msg.data)
        if not math.isfinite(value):
            value = 0.0

        def apply_value() -> None:
            self.latest_teaching_directional_force_capacity = value
            self.teaching_directional_force_capacity_var.set(
                "竖直向上力输出能力 / Upward force capacity: {:.3f} N".format(value))

        self.root.after(0, apply_value)

    def _teaching_manipulability_callback(self, msg: Float64) -> None:
        value = float(msg.data)
        if not math.isfinite(value):
            value = 0.0

        def apply_value() -> None:
            self.latest_teaching_manipulability = value
            self.teaching_manipulability_var.set(
                "可操作度 / Manipulability: {:.6f}".format(value))

        self.root.after(0, apply_value)

    def _home_return_status_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            state = payload.get("state", "unknown")
            has_home = "true" if payload.get("has_home", False) else "false"
            text = "状态 / State: {}; has_home={}".format(state, has_home)
            if "position_error" in payload:
                text += "; pos_err={:.4f}".format(float(payload["position_error"]))
            if "yaw_error" in payload:
                text += "; yaw_err={:.4f}".format(float(payload["yaw_error"]))
        except Exception:
            text = msg.data

        self.root.after(0, lambda: self.home_return_status_var.set(text))

    @staticmethod
    def _wrench_msg_to_list(wrench) -> List[float]:
        return [
            float(wrench.force.x),
            float(wrench.force.y),
            float(wrench.force.z),
            float(wrench.torque.x),
            float(wrench.torque.y),
            float(wrench.torque.z),
        ]

    @staticmethod
    def _pose_msg_to_dict(pose) -> Dict[str, Dict[str, float]]:
        return {
            "position": {
                "x": float(pose.position.x),
                "y": float(pose.position.y),
                "z": float(pose.position.z),
            },
            "orientation": {
                "x": float(pose.orientation.x),
                "y": float(pose.orientation.y),
                "z": float(pose.orientation.z),
                "w": float(pose.orientation.w),
            },
        }

    @staticmethod
    def _skew(vector: np.ndarray) -> np.ndarray:
        return np.array(
            [
                [0.0, -vector[2], vector[1]],
                [vector[2], 0.0, -vector[0]],
                [-vector[1], vector[0], 0.0],
            ],
            dtype=float)

    @staticmethod
    def _rotation_from_quaternion(orientation: Dict[str, float]) -> np.ndarray:
        x = float(orientation.get("x", 0.0))
        y = float(orientation.get("y", 0.0))
        z = float(orientation.get("z", 0.0))
        w = float(orientation.get("w", 1.0))
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm < 1e-12:
            return np.eye(3)
        x /= norm
        y /= norm
        z /= norm
        w /= norm
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=float)

    @staticmethod
    def _pose_dict_from_planar_values(x: float, y: float, yaw: float) -> Dict[str, Any]:
        half_yaw = 0.5 * yaw
        return {
            "position": {"x": x, "y": y, "z": 0.0},
            "orientation": {
                "x": 0.0,
                "y": 0.0,
                "z": math.sin(half_yaw),
                "w": math.cos(half_yaw),
            },
        }

    @staticmethod
    def _yaw_from_quaternion_dict(orientation: Dict[str, float]) -> float:
        x = float(orientation.get("x", 0.0))
        y = float(orientation.get("y", 0.0))
        z = float(orientation.get("z", 0.0))
        w = float(orientation.get("w", 1.0))
        return math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z))

    def _placeholder_teaching_odom(self) -> Dict[str, Any]:
        return {
            "source": "not_recorded",
            "frame_id": "",
            "child_frame_id": "",
            "pose": self._pose_dict_from_planar_values(0.0, 0.0, 0.0),
            "planar": {"x": 0.0, "y": 0.0, "yaw": 0.0},
            "twist": {
                "linear": {"x": 0.0, "y": 0.0, "z": 0.0},
                "angular": {"x": 0.0, "y": 0.0, "z": 0.0},
            },
        }

    def _franka_robot_state_to_teaching_sample(self, msg: RobotState) -> Dict[str, Any]:
        stamp = msg.header.stamp.to_sec() if msg.header.stamp else rospy.Time.now().to_sec()
        pose_dict = self._pose_msg_to_dict(msg.O_T_EE)
        joint_names = list(msg.joints_states.name)
        joint_positions = [float(value) for value in msg.joints_states.position]
        arm_positions = joint_positions[:7]
        odom = deepcopy(self.latest_teaching_odom) if self.latest_teaching_odom else \
            self._placeholder_teaching_odom()
        odom_planar = odom.get("planar", {})
        position = pose_dict["position"]
        orientation = pose_dict["orientation"]
        vector = (
            arm_positions +
            [
                float(odom_planar.get("x", 0.0)),
                float(odom_planar.get("y", 0.0)),
                float(odom_planar.get("yaw", 0.0)),
            ] +
            [
                float(position.get("x", 0.0)),
                float(position.get("y", 0.0)),
                float(position.get("z", 0.0)),
                float(orientation.get("x", 0.0)),
                float(orientation.get("y", 0.0)),
                float(orientation.get("z", 0.0)),
                float(orientation.get("w", 1.0)),
            ])
        return {
            "record_index": None,
            "stamp": stamp,
            "source": "franka_state_bridge",
            "joint_names": joint_names,
            "joint_positions": joint_positions,
            "arm_joint_positions": arm_positions,
            "odom": odom,
            "ee_pose": {
                "frame_id": msg.header.frame_id,
                "pose": pose_dict,
            },
            "vector_order": [
                "arm_joint_positions[0:7]",
                "odom.x",
                "odom.y",
                "odom.yaw",
                "ee.position.x",
                "ee.position.y",
                "ee.position.z",
                "ee.orientation.x",
                "ee.orientation.y",
                "ee.orientation.z",
                "ee.orientation.w",
            ],
            "vector": vector,
        }

    def _subscribe_teaching_robot_state(self) -> None:
        topic = self.teaching_robot_state_topic_var.get().strip()
        if not topic:
            self.teaching_status_var.set("Franka robot_state topic 不能为空。")
            return
        if self.teaching_robot_state_sub is not None:
            try:
                self.teaching_robot_state_sub.unregister()
            except Exception:
                pass
            self.teaching_robot_state_sub = None
        self.teaching_robot_state_sub = rospy.Subscriber(
            topic,
            RobotState,
            self._teaching_robot_state_callback,
            queue_size=1)
        self._log("示教页订阅 Franka robot_state: {}".format(topic))

    def _subscribe_teaching_odom(self) -> None:
        topic = self.teaching_odom_topic_var.get().strip()
        if not topic:
            self.teaching_status_var.set("底盘 odom topic 不能为空。")
            return
        if self.teaching_odom_sub is not None:
            try:
                self.teaching_odom_sub.unregister()
            except Exception:
                pass
            self.teaching_odom_sub = None
        self.teaching_odom_sub = rospy.Subscriber(
            topic,
            Odometry,
            self._teaching_odom_callback,
            queue_size=1)
        self._log("示教页订阅底盘 odom: {}".format(topic))

    def _teaching_odom_callback(self, msg: Odometry) -> None:
        pose = self._pose_msg_to_dict(msg.pose.pose)
        yaw = self._yaw_from_quaternion_dict(pose.get("orientation", {}))
        self.latest_teaching_odom = {
            "source": "odom_topic",
            "frame_id": msg.header.frame_id,
            "child_frame_id": msg.child_frame_id,
            "stamp": msg.header.stamp.to_sec() if msg.header.stamp else rospy.Time.now().to_sec(),
            "pose": pose,
            "planar": {
                "x": float(msg.pose.pose.position.x),
                "y": float(msg.pose.pose.position.y),
                "yaw": yaw,
            },
            "twist": {
                "linear": {
                    "x": float(msg.twist.twist.linear.x),
                    "y": float(msg.twist.twist.linear.y),
                    "z": float(msg.twist.twist.linear.z),
                },
                "angular": {
                    "x": float(msg.twist.twist.angular.x),
                    "y": float(msg.twist.twist.angular.y),
                    "z": float(msg.twist.twist.angular.z),
                },
            },
        }

    def _teaching_set_franka_bridge_topic(self) -> None:
        self.teaching_robot_state_topic_var.set(self._payload_franka_bridge_topic())
        self._subscribe_teaching_robot_state()

    def _teaching_robot_state_callback(self, msg: RobotState) -> None:
        sample = self._franka_robot_state_to_teaching_sample(msg)
        formatted = self._format_teaching_sample(sample)
        stamp = sample.get("stamp", rospy.Time.now().to_sec())

        def apply_sample() -> None:
            self.latest_teaching_robot_state_sample = sample
            self.latest_teaching_sample = sample
            self.latest_teaching_sample_json = json.dumps(sample, ensure_ascii=False)
            self.teaching_current_summary_var.set(
                "当前 Franka robot_state 已更新 / Franka state updated: {:.3f}".format(
                    float(stamp)))
            self._set_text_widget("teaching_current_text", formatted)

        self.root.after(0, apply_sample)

    def _subscribe_payload_robot_state(self) -> None:
        topic = self.payload_robot_state_topic_var.get().strip()
        if not topic:
            self.payload_status_var.set("robot_state topic 不能为空 / Topic is empty.")
            return
        if self.payload_robot_state_sub is not None:
            try:
                self.payload_robot_state_sub.unregister()
            except Exception:
                pass
            self.payload_robot_state_sub = None
        self.payload_robot_state_sub = rospy.Subscriber(
            topic,
            RobotState,
            self._payload_robot_state_callback,
            queue_size=1)
        self.payload_status_var.set("已订阅 / Subscribed: {}".format(topic))
        self._log("负载辨识订阅 robot_state: {}".format(topic))

    def _payload_set_topic_from_active_robot(self) -> None:
        self.payload_robot_state_topic_var.set(self._default_payload_robot_state_topic())
        self._subscribe_payload_robot_state()

    def _payload_franka_bridge_topic(self) -> str:
        return "/{}/franka_state_bridge/robot_state".format(self.active_robot_id)

    def _start_franka_state_bridge(self) -> None:
        self._start_worker(self._start_franka_state_bridge_worker)

    def _start_franka_state_bridge_worker(self) -> None:
        if self._launch_process("franka_state_bridge"):
            bridge_topic = self._payload_franka_bridge_topic()

            def apply_bridge_topic() -> None:
                self.payload_robot_state_topic_var.set(bridge_topic)
                self._subscribe_payload_robot_state()
                self.teaching_robot_state_topic_var.set(bridge_topic)
                self._subscribe_teaching_robot_state()
                self.payload_status_var.set(
                    "Franka只读状态节点已启动，订阅：{}".format(bridge_topic))
                self.teaching_status_var.set(
                    "Franka只读状态节点已启动，示教订阅：{}".format(bridge_topic))

            self.root.after(0, apply_bridge_topic)
            self._log(
                "Franka 只读状态节点启动请求已发送；该节点只调用 libfranka read()，不发送控制命令。")

    def _payload_robot_state_callback(self, msg: RobotState) -> None:
        stamp = msg.header.stamp.to_sec() if msg.header.stamp else rospy.Time.now().to_sec()
        sample = {
            "stamp": stamp,
            "frame_id": msg.header.frame_id,
            "pose": self._pose_msg_to_dict(msg.O_T_EE),
            "wrench_base": self._wrench_msg_to_list(msg.est_ext_wrench_base_frame),
            "wrench_stiffness": self._wrench_msg_to_list(msg.est_ext_wrench_stiffness_frame),
            "joint_names": list(msg.joints_states.name),
            "joint_positions": [float(value) for value in msg.joints_states.position],
        }
        formatted = self._format_payload_robot_state(sample)

        def apply_sample() -> None:
            self.latest_payload_robot_state = sample
            self.payload_latest_summary_var.set(
                "当前 robot_state 已更新 / Current robot_state updated: {:.3f}".format(stamp))
            self._set_text_widget("payload_current_text", formatted)

        self.root.after(0, apply_sample)

    def _selected_payload_wrench_key(self) -> str:
        frame = self.payload_wrench_frame_var.get().strip().lower()
        if frame == "base":
            return "wrench_base"
        return "wrench_stiffness"

    def _selected_payload_wrench(self, sample: Dict[str, Any]) -> np.ndarray:
        return np.array(sample.get(self._selected_payload_wrench_key(), [0.0] * 6), dtype=float)

    def _payload_corrected_wrench(self, sample: Dict[str, Any]) -> np.ndarray:
        raw_wrench = self._selected_payload_wrench(sample)
        if self.payload_wrench_data_mode_var.get().strip().lower() == "bias_compensated":
            return raw_wrench - self.payload_wrench_bias
        return raw_wrench

    def _format_payload_robot_state(self, sample: Dict[str, Any]) -> str:
        raw_wrench = self._selected_payload_wrench(sample)
        corrected_wrench = raw_wrench - self.payload_wrench_bias
        pose = sample.get("pose", {})
        position = pose.get("position", {})
        orientation = pose.get("orientation", {})
        lines = [
            "stamp: {:.3f}".format(float(sample.get("stamp", 0.0))),
            "frame_id: {}".format(sample.get("frame_id", "")),
            "selected_wrench_frame: {}".format(self.payload_wrench_frame_var.get()),
            "wrench_data_mode: {}".format(self.payload_wrench_data_mode_var.get()),
            "raw_wrench [Fx,Fy,Fz,Tx,Ty,Tz]:",
            self._format_float_list(raw_wrench.tolist(), precision=5),
            "bias [Fx,Fy,Fz,Tx,Ty,Tz]:",
            self._format_float_list(self.payload_wrench_bias.tolist(), precision=5),
            "corrected_wrench [Fx,Fy,Fz,Tx,Ty,Tz]:",
            self._format_float_list(corrected_wrench.tolist(), precision=5),
            "O_T_EE.position: [x={:.5f}, y={:.5f}, z={:.5f}]".format(
                float(position.get("x", 0.0)),
                float(position.get("y", 0.0)),
                float(position.get("z", 0.0))),
            "O_T_EE.orientation: [x={:.5f}, y={:.5f}, z={:.5f}, w={:.5f}]".format(
                float(orientation.get("x", 0.0)),
                float(orientation.get("y", 0.0)),
                float(orientation.get("z", 0.0)),
                float(orientation.get("w", 1.0))),
        ]
        return "\n".join(lines)

    def _payload_set_current_bias(self) -> None:
        if self.latest_payload_robot_state is None:
            self.payload_status_var.set("还没有 robot_state，无法置零 / No robot_state yet.")
            return
        self.payload_wrench_bias = self._selected_payload_wrench(self.latest_payload_robot_state)
        self.payload_status_var.set(
            "已设置当前 wrench 为偏置 / Bias set: {}".format(
                self._format_float_list(self.payload_wrench_bias.tolist(), precision=4)))
        self._set_text_widget(
            "payload_current_text",
            self._format_payload_robot_state(self.latest_payload_robot_state))

    def _payload_record_current_sample(self) -> None:
        if self.latest_payload_robot_state is None:
            self.payload_status_var.set("还没有 robot_state，无法记录 / No robot_state yet.")
            return
        sample = deepcopy(self.latest_payload_robot_state)
        corrected_wrench = self._payload_corrected_wrench(sample)
        sample["record_index"] = len(self.payload_samples) + 1
        sample["wrench_frame"] = self.payload_wrench_frame_var.get().strip().lower()
        sample["wrench_data_mode"] = self.payload_wrench_data_mode_var.get().strip().lower()
        sample["wrench_bias"] = self.payload_wrench_bias.tolist()
        sample["corrected_wrench"] = corrected_wrench.tolist()
        self.payload_samples.append(sample)
        self.payload_status_var.set(
            "已记录 {} 个点 / Recorded {} samples.".format(
                len(self.payload_samples),
                len(self.payload_samples)))
        self._append_text_widget(
            "payload_samples_text",
            self._format_payload_recorded_sample(sample))

    def _format_payload_recorded_sample(self, sample: Dict[str, Any]) -> str:
        corrected = np.array(sample.get("corrected_wrench", [0.0] * 6), dtype=float)
        force_norm = float(np.linalg.norm(corrected[:3]))
        pose = sample.get("pose", {})
        position = pose.get("position", {})
        return "\n".join(
            [
                "sample #{}  t={:.3f}  frame={}".format(
                    sample.get("record_index", "?"),
                    float(sample.get("stamp", 0.0)),
                    sample.get("wrench_frame", "unknown")),
                "corrected_wrench: {}".format(
                    self._format_float_list(corrected.tolist(), precision=5)),
                "|force| = {:.5f} N".format(force_norm),
                "ee_position: [x={:.5f}, y={:.5f}, z={:.5f}]".format(
                    float(position.get("x", 0.0)),
                    float(position.get("y", 0.0)),
                    float(position.get("z", 0.0))),
            ])

    def _payload_clear_samples(self) -> None:
        self.payload_samples = []
        self.payload_result_mass_var.set("-")
        self.payload_result_com_var.set("-")
        self.payload_result_quality_var.set("-")
        self.payload_status_var.set("已清空记录 / Samples cleared.")
        self._set_text_widget("payload_samples_text", "")

    def _payload_estimate_mass_com(self) -> Optional[Dict[str, Any]]:
        if not self.payload_samples:
            self.payload_status_var.set("没有记录点 / No samples.")
            return None
        try:
            gravity = abs(float(self.payload_gravity_var.get().strip()))
        except ValueError:
            self.payload_status_var.set("重力加速度不是合法数字 / Invalid gravity.")
            return None
        if gravity < 1e-6:
            self.payload_status_var.set("重力加速度必须大于0 / Gravity must be positive.")
            return None

        matrix_blocks = []
        torque_blocks = []
        force_norms = []
        frame = self.payload_wrench_frame_var.get().strip().lower()
        for sample in self.payload_samples:
            wrench = np.array(sample.get("corrected_wrench", [0.0] * 6), dtype=float)
            force = wrench[:3]
            torque = wrench[3:]
            if not np.all(np.isfinite(wrench)):
                continue
            force_norm = float(np.linalg.norm(force))
            if force_norm < 1e-6:
                continue
            pose = sample.get("pose", {})
            rotation = self._rotation_from_quaternion(pose.get("orientation", {}))
            if frame == "base":
                matrix_blocks.append(-self._skew(force).dot(rotation))
            else:
                matrix_blocks.append(-self._skew(force))
            torque_blocks.append(torque)
            force_norms.append(force_norm)

        if not matrix_blocks:
            self.payload_status_var.set("有效力数据太小，无法估计 / Force data is too small.")
            return None

        matrix = np.vstack(matrix_blocks)
        target = np.concatenate(torque_blocks)
        com, residuals, rank, singular_values = np.linalg.lstsq(matrix, target, rcond=None)
        predicted = matrix.dot(com)
        residual_vector = predicted - target
        rms_residual = float(np.sqrt(np.mean(np.square(residual_vector)))) if residual_vector.size else 0.0
        mass = float(np.mean(force_norms) / gravity)
        mass_std = float(np.std(np.asarray(force_norms, dtype=float) / gravity))
        condition_number = (
            float(singular_values[0] / singular_values[-1])
            if singular_values.size >= 2 and singular_values[-1] > 1e-12
            else float("inf"))

        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "robot_state_topic": self.payload_robot_state_topic_var.get().strip(),
            "wrench_frame": frame,
            "gravity_acceleration_mps2": gravity,
            "sample_count": len(self.payload_samples),
            "valid_sample_count": len(force_norms),
            "mass_kg": mass,
            "mass_std_kg": mass_std,
            "center_of_mass_m": [float(value) for value in com.tolist()],
            "rms_torque_residual_Nm": rms_residual,
            "least_squares_rank": int(rank),
            "condition_number": condition_number,
            "singular_values": [float(value) for value in singular_values.tolist()],
            "samples": deepcopy(self.payload_samples),
        }
        self.latest_payload_estimate = result
        self.payload_result_mass_var.set(
            "{:.5f} kg  (std {:.5f} kg)".format(mass, mass_std))
        self.payload_result_com_var.set(
            "[x={:.5f}, y={:.5f}, z={:.5f}] m".format(
                float(com[0]),
                float(com[1]),
                float(com[2])))
        self.payload_result_quality_var.set(
            "valid_samples={}, rank={}, torque_rms={:.5f} Nm, cond={:.3g}".format(
                len(force_norms),
                int(rank),
                rms_residual,
                condition_number))
        self.payload_status_var.set(
            "估计完成 / Estimated: mass={:.5f} kg, COM={}".format(
                mass,
                self._format_float_list(com.tolist(), precision=5)))
        return result

    def _payload_export_result_json(self) -> None:
        result = self._payload_estimate_mass_com()
        if result is None:
            return
        output_dir = self._resolve_workspace_ros_path(self.payload_output_dir_var.get().strip())
        if not output_dir.is_absolute():
            output_dir = self.workspace_root / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / "payload_mass_com_{}.json".format(timestamp)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        self.payload_status_var.set("已导出 / Exported: {}".format(output_path))
        self._log("负载质量/质心辨识结果已导出：{}".format(output_path))

    def _create_scrollable_notebook_page(
        self,
        notebook: ttk.Notebook,
        title: str,
    ) -> ttk.Frame:
        outer = ttk.Frame(notebook)
        notebook.add(outer, text=title)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)

        canvas = tk.Canvas(outer, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        content = ttk.Frame(canvas, padding=16)
        window_id = canvas.create_window((0, 0), window=content, anchor="nw")

        def update_scroll_region(_event=None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def match_canvas_width(event) -> None:
            canvas.itemconfigure(window_id, width=event.width)

        content.bind("<Configure>", update_scroll_region)
        canvas.bind("<Configure>", match_canvas_width)

        for widget in (outer, canvas, scrollbar, content):
            setattr(widget, "_moca_scroll_canvas", canvas)

        return content

    def _find_scroll_canvas_for_widget(self, widget) -> Optional[tk.Canvas]:
        current = widget
        while current is not None:
            canvas = getattr(current, "_moca_scroll_canvas", None)
            if canvas is not None:
                return canvas
            current = getattr(current, "master", None)
        return None

    def _select_notebook_content_page(self, content_widget) -> bool:
        if not hasattr(self, "notebook") or content_widget is None:
            return False

        tab_ids = set(str(tab_id) for tab_id in self.notebook.tabs())
        current = content_widget
        while current is not None:
            if str(current) in tab_ids:
                self.notebook.select(current)
                return True
            current = getattr(current, "master", None)
        return False

    def _on_global_mousewheel(self, event):
        if isinstance(event.widget, tk.Text):
            return None

        canvas = self._find_scroll_canvas_for_widget(event.widget)
        if canvas is None:
            return None

        if getattr(event, "num", None) == 4:
            scroll_units = -3
        elif getattr(event, "num", None) == 5:
            scroll_units = 3
        else:
            delta = getattr(event, "delta", 0)
            if delta == 0:
                return None
            scroll_units = int(-delta / 120) if abs(delta) >= 120 else (-1 if delta > 0 else 1)

        canvas.yview_scroll(scroll_units, "units")
        return "break"

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        self.notebook = notebook

        self.root.bind_all("<MouseWheel>", self._on_global_mousewheel, add="+")
        self.root.bind_all("<Button-4>", self._on_global_mousewheel, add="+")
        self.root.bind_all("<Button-5>", self._on_global_mousewheel, add="+")

        first_page = self._create_scrollable_notebook_page(notebook, "控制台")
        self._build_first_page(first_page)

        second_page = self._create_scrollable_notebook_page(notebook, "VLM设置")
        self._build_vlm_page(second_page)

        third_page = self._create_scrollable_notebook_page(notebook, "力能力设置")
        self._build_force_capability_page(third_page)

        fourth_page = self._create_scrollable_notebook_page(notebook, "示教/Home")
        self._build_teaching_home_page(fourth_page)

        fifth_page = self._create_scrollable_notebook_page(notebook, "绘图/数据")
        self._build_plot_data_page(fifth_page)

        metric_merge_page = self._create_scrollable_notebook_page(notebook, "数据合并")
        self._build_metric_merge_page(metric_merge_page)

        sixth_page = self._create_scrollable_notebook_page(notebook, "负载辨识")
        self._build_payload_identification_page(sixth_page)

        self.experiment_page = self._create_scrollable_notebook_page(notebook, "实验配置")
        self._build_experiment_config_page(self.experiment_page)

    def _build_first_page(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(1, weight=1)

        title = ttk.Label(
            parent,
            text="MOCA 控制与规划上位机",
            font=("TkDefaultFont", 16, "bold"))
        title.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 14))

        left_frame = ttk.LabelFrame(parent, text="系统启动", padding=14)
        left_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))
        left_frame.columnconfigure(0, weight=1)
        left_frame.columnconfigure(1, weight=0)
        left_frame.columnconfigure(2, weight=1)

        self._add_action_button(
            left_frame,
            row=0,
            label="1 启动仿真控制程序\nStart Simulation",
            command=self._start_simulation,
            status_key="simulation")
        self._add_action_button(
            left_frame,
            row=1,
            label="2 连接实体机器人\nConnect Hardware",
            command=self._start_hardware,
            status_key="hardware")

        controller_mode_frame = ttk.LabelFrame(
            left_frame,
            text="控制器类型 / Controller",
            padding=8)
        controller_mode_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(8, 12))
        controller_mode_frame.columnconfigure(0, weight=1)

        cartesian_controller_button = ttk.Radiobutton(
            controller_mode_frame,
            text="笛卡尔阻抗控制器\nCartesian impedance controller",
            variable=self.controller_mode,
            value="cartesian")
        cartesian_controller_button.grid(row=0, column=0, sticky="w")

        joint_controller_button = ttk.Radiobutton(
            controller_mode_frame,
            text="关节阻抗控制器（只跟踪零空间关节目标）\nJoint impedance controller",
            variable=self.controller_mode,
            value="joint")
        joint_controller_button.grid(row=1, column=0, sticky="w", pady=(6, 0))

        self._add_action_button(
            left_frame,
            row=3,
            label="3 启动RViz\nStart RViz",
            command=self._start_rviz,
            status_key="rviz")
        self._add_action_button(
            left_frame,
            row=4,
            label="4 启动VLM\nStart VLM",
            command=self._start_vlm,
            status_key="vlm")
        self._add_action_button(
            left_frame,
            row=5,
            label="5 启动规划器\nStart Planner",
            command=self._start_planner,
            status_key="planner")

        reset_button = ttk.Button(
            left_frame,
            text="Reset到原位置\nReset to Home",
            command=self._reset_to_home)
        reset_button.grid(row=6, column=0, sticky="ew", padx=(0, 10), pady=(20, 8))

        close_all_button = ttk.Button(
            left_frame,
            text="关闭全部\nShut Down All",
            command=self._shutdown_all_processes)
        close_all_button.grid(row=7, column=0, columnspan=2, sticky="ew", padx=(0, 10), pady=8)

        right_frame = ttk.LabelFrame(parent, text="选择需要规划的轨迹", padding=14)
        right_frame.grid(row=1, column=1, sticky="nsew", padx=(8, 0), pady=(0, 8))
        right_frame.columnconfigure(0, weight=1)

        algorithm_frame = ttk.LabelFrame(
            right_frame,
            text="对比算法 / Comparison Algorithm",
            padding=8)
        algorithm_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        algorithm_frame.columnconfigure(0, weight=1)
        algorithm_combo = ttk.Combobox(
            algorithm_frame,
            textvariable=self.planning_algorithm_mode,
            values=(
                "ours",
                "manipulability_only",
                "static_force_polytope",
                "residual_force_polytope",
                "cone_intersection_volume",
                "joint_quintic_interpolation"),
            state="readonly")
        algorithm_combo.grid(row=0, column=0, sticky="ew")
        ttk.Label(
            algorithm_frame,
            text="ours 使用当前完整方法；joint_quintic_interpolation 不调用优化器，只做关节五次插值。",
            foreground="#505050").grid(row=1, column=0, sticky="w", pady=(6, 0))

        normal_button = tk.Radiobutton(
            right_frame,
            text="1 举升实验\nLift Experiment",
            variable=self.trajectory_selection,
            value="lift_experiment",
            indicatoron=False,
            width=24,
            pady=12,
            command=self._on_trajectory_selection_changed)
        normal_button.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        normal_button.bind(
            "<ButtonRelease-1>",
            lambda _event: self.root.after(0, self._open_lift_experiment_page),
            add="+")

        comparison_button = tk.Radiobutton(
            right_frame,
            text="2 两次提起对比\nTwo-Lift Compare",
            variable=self.trajectory_selection,
            value="comparison_two_lifts",
            indicatoron=False,
            width=24,
            pady=12,
            command=self._on_trajectory_selection_changed)
        comparison_button.grid(row=2, column=0, sticky="ew")

        manual_button = tk.Radiobutton(
            right_frame,
            text="3 姿态球模式\nPose Marker Mode",
            variable=self.trajectory_selection,
            value="manual_target_pose",
            indicatoron=False,
            width=24,
            pady=12,
            command=self._on_trajectory_selection_changed)
        manual_button.grid(row=3, column=0, sticky="ew", pady=(10, 0))

        move_button = tk.Radiobutton(
            right_frame,
            text="4 移动实验\nMove Experiment",
            variable=self.trajectory_selection,
            value="move_experiment",
            indicatoron=False,
            width=24,
            pady=12,
            command=self._on_trajectory_selection_changed)
        move_button.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        move_button.bind(
            "<ButtonRelease-1>",
            lambda _event: self.root.after(0, self._open_move_experiment_page),
            add="+")

        single_point_button = tk.Radiobutton(
            right_frame,
            text="5 单点优化实验\nSingle-Point Optimization",
            variable=self.trajectory_selection,
            value="single_point_optimization",
            indicatoron=False,
            width=24,
            pady=12,
            command=self._on_trajectory_selection_changed)
        single_point_button.grid(row=5, column=0, sticky="ew", pady=(10, 0))
        single_point_button.bind(
            "<ButtonRelease-1>",
            lambda _event: self.root.after(0, self._open_single_point_experiment_page),
            add="+")

        self.selection_label = ttk.Label(
            right_frame,
            text="当前选择：举升实验",
            foreground="#204060")
        self.selection_label.grid(row=6, column=0, sticky="w", pady=(16, 0))

        hint_label = ttk.Label(
            right_frame,
            text="说明：先在这里选轨迹，再点左侧“启动规划器”。\n姿态球模式下会在 RViz 中启用可拖拽目标位姿。\n如果 VLM 已启动，本次规划会自动请求质量估计。",
            justify=tk.LEFT)
        hint_label.grid(row=7, column=0, sticky="w", pady=(16, 0))

        pose_marker_status_title = ttk.Label(right_frame, text="姿态球状态：")
        pose_marker_status_title.grid(row=8, column=0, sticky="w", pady=(16, 0))

        pose_marker_status = ttk.Label(
            right_frame,
            textvariable=self.status_vars["pose_marker"],
            foreground="#206040")
        pose_marker_status.grid(row=9, column=0, sticky="w", pady=(4, 0))

        log_frame = ttk.LabelFrame(parent, text="运行日志", padding=12)
        log_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(
            log_frame,
            height=14,
            wrap=tk.WORD,
            state=tk.DISABLED,
            cursor="xterm",
            exportselection=False,
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.log_text.bind("<Control-c>", self._copy_log_selection)
        self.log_text.bind("<Control-C>", self._copy_log_selection)
        self.log_text.bind("<Control-a>", self._select_all_logs)
        self.log_text.bind("<Control-A>", self._select_all_logs)

        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        log_scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=log_scrollbar.set)

    def _build_experiment_config_page(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        title = ttk.Label(
            parent,
            text="实验配置",
            font=("TkDefaultFont", 16, "bold"))
        title.grid(row=0, column=0, sticky="w", pady=(0, 14))

        lift_frame = ttk.LabelFrame(parent, text="举升实验 / Lift Experiment", padding=14)
        lift_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 12))
        lift_frame.columnconfigure(1, weight=1)

        ttk.Label(lift_frame, text="Home 路径点文件").grid(
            row=0, column=0, sticky="w", padx=(0, 10), pady=6)
        ttk.Entry(lift_frame, textvariable=self.home_teaching_point_path_var).grid(
            row=0, column=1, sticky="ew", pady=6)

        ttk.Label(lift_frame, text="示教轨迹文件").grid(
            row=1, column=0, sticky="w", padx=(0, 10), pady=6)
        ttk.Entry(lift_frame, textvariable=self.lift_waypoints_path_var).grid(
            row=1, column=1, sticky="ew", pady=6)

        ttk.Label(lift_frame, text="间隔点规划时间 / s").grid(
            row=2, column=0, sticky="w", padx=(0, 10), pady=6)
        ttk.Entry(
            lift_frame,
            textvariable=self.lift_waypoint_segment_duration_var,
            width=12).grid(row=2, column=1, sticky="w", pady=6)

        ttk.Label(lift_frame, text="Home规划时间 / s").grid(
            row=3, column=0, sticky="w", padx=(0, 10), pady=6)
        ttk.Entry(
            lift_frame,
            textvariable=self.home_return_duration_var,
            width=12,
            state="readonly").grid(row=3, column=1, sticky="w", pady=6)

        button_frame = ttk.Frame(lift_frame)
        button_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(14, 4))
        for column in range(3):
            button_frame.columnconfigure(column, weight=1)

        ttk.Button(
            button_frame,
            text="返回home点\nReturn Home",
            command=lambda: self._reset_to_home_for_experiment("lift")).grid(
                row=0, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(
            button_frame,
            text="返回初始点\nReturn Initial",
            command=self._return_to_lift_initial_point).grid(
                row=0, column=1, sticky="ew", padx=8)
        ttk.Button(
            button_frame,
            text="开始运动\nStart Motion",
            command=self._start_lift_experiment_motion).grid(
                row=0, column=2, sticky="ew", padx=(8, 0))

        status = ttk.Label(
            lift_frame,
            textvariable=self.experiment_config_status_var,
            foreground="#204060")
        status.grid(row=5, column=0, columnspan=2, sticky="w", pady=(12, 0))

        move_frame = ttk.LabelFrame(parent, text="移动实验 / Move Experiment", padding=14)
        move_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 12))
        move_frame.columnconfigure(1, weight=1)

        ttk.Label(move_frame, text="Home 路径点文件").grid(
            row=0, column=0, sticky="w", padx=(0, 10), pady=6)
        ttk.Entry(move_frame, textvariable=self.move_home_teaching_point_path_var).grid(
            row=0, column=1, sticky="ew", pady=6)

        ttk.Label(move_frame, text="示教轨迹文件").grid(
            row=1, column=0, sticky="w", padx=(0, 10), pady=6)
        ttk.Entry(move_frame, textvariable=self.move_waypoints_path_var).grid(
            row=1, column=1, sticky="ew", pady=6)

        ttk.Label(move_frame, text="间隔点规划时间 / s").grid(
            row=2, column=0, sticky="w", padx=(0, 10), pady=6)
        ttk.Entry(
            move_frame,
            textvariable=self.move_waypoint_segment_duration_var,
            width=12).grid(row=2, column=1, sticky="w", pady=6)

        ttk.Label(move_frame, text="Home规划时间 / s").grid(
            row=3, column=0, sticky="w", padx=(0, 10), pady=6)
        ttk.Entry(
            move_frame,
            textvariable=self.move_home_return_duration_var,
            width=12).grid(row=3, column=1, sticky="w", pady=6)

        move_button_frame = ttk.Frame(move_frame)
        move_button_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(14, 4))
        for column in range(3):
            move_button_frame.columnconfigure(column, weight=1)

        ttk.Button(
            move_button_frame,
            text="返回home点\nReturn Home",
            command=lambda: self._reset_to_home_for_experiment("move")).grid(
                row=0, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(
            move_button_frame,
            text="返回初始点\nReturn Initial",
            command=self._return_to_move_initial_point).grid(
                row=0, column=1, sticky="ew", padx=8)
        ttk.Button(
            move_button_frame,
            text="下一轨迹点\nNext Waypoint",
            command=self._start_move_next_waypoint).grid(
                row=0, column=2, sticky="ew", padx=(8, 0))

        move_hint = ttk.Label(
            move_frame,
            text="说明：下一轨迹点会按 move/way_points.json 循环执行；返回 Home 或初始点后，下一次从第 1 个 waypoint 开始。",
            justify=tk.LEFT,
            foreground="#204060")
        move_hint.grid(row=5, column=0, columnspan=2, sticky="w", pady=(12, 0))

        single_frame = ttk.LabelFrame(
            parent,
            text="单点优化实验 / Single-Point Optimization",
            padding=14)
        single_frame.grid(row=3, column=0, sticky="nsew", pady=(0, 12))
        # 单点优化实验
        for column in range(8):
            single_frame.columnconfigure(column, weight=1 if column % 2 == 1 else 0)

        ttk.Label(single_frame, text="Home 路径点文件 / Home file").grid(
            row=0, column=0, columnspan=2, sticky="w", padx=(0, 10), pady=6)
        ttk.Entry(
            single_frame,
            textvariable=self.single_point_home_teaching_point_path_var).grid(
                row=0, column=2, columnspan=6, sticky="ew", pady=6)

        ttk.Label(single_frame, text="单点文件 / Point file").grid(
            row=1, column=0, columnspan=2, sticky="w", padx=(0, 10), pady=6)
        ttk.Entry(
            single_frame,
            textvariable=self.single_point_waypoints_path_var).grid(
                row=1, column=2, columnspan=6, sticky="ew", pady=6)

        ttk.Label(single_frame, text="返回 Home 时间 / Home duration (s)").grid(
            row=2, column=0, columnspan=2, sticky="w", padx=(0, 10), pady=6)
        ttk.Entry(
            single_frame,
            textvariable=self.single_point_home_return_duration_var,
            width=12).grid(row=2, column=2, sticky="w", pady=6)

        ttk.Label(single_frame, text="直达点时间 / Move-to-point segment (s)").grid(
            row=2, column=3, columnspan=3, sticky="w", padx=(10, 10), pady=6)
        ttk.Entry(
            single_frame,
            textvariable=self.single_point_waypoint_segment_duration_var,
            width=12).grid(row=2, column=6, sticky="w", pady=6)

        ttk.Label(single_frame, text="目标位置 / Position").grid(
            row=3, column=0, columnspan=8, sticky="w", pady=(10, 6))
        for column, (label, variable) in enumerate((
            ("x", self.single_point_x_var),
            ("y", self.single_point_y_var),
            ("z", self.single_point_z_var),
        )):
            ttk.Label(single_frame, text=label).grid(
                row=4, column=2 * column, sticky="w", padx=(0, 6), pady=4)
            ttk.Entry(single_frame, textvariable=variable, width=12).grid(
                row=4, column=2 * column + 1, sticky="ew", padx=(0, 10), pady=4)

        ttk.Label(single_frame, text="目标姿态四元数 / Orientation [x, y, z, w]").grid(
            row=5, column=0, columnspan=8, sticky="w", pady=(10, 6))
        for column, (label, variable) in enumerate((
            ("qx", self.single_point_qx_var),
            ("qy", self.single_point_qy_var),
            ("qz", self.single_point_qz_var),
            ("qw", self.single_point_qw_var),
        )):
            ttk.Label(single_frame, text=label).grid(
                row=6, column=2 * column, sticky="w", padx=(0, 6), pady=4)
            ttk.Entry(single_frame, textvariable=variable, width=12).grid(
                row=6, column=2 * column + 1, sticky="ew", padx=(0, 10), pady=4)

        ttk.Label(single_frame, text="优化规划时间 / Optimization duration (s)").grid(
            row=7, column=0, columnspan=2, sticky="w", padx=(0, 10), pady=(10, 4))
        ttk.Entry(
            single_frame,
            textvariable=self.single_point_move_duration_var,
            width=12).grid(row=7, column=2, sticky="w", pady=(10, 4))

        single_button_frame = ttk.Frame(single_frame)
        single_button_frame.grid(row=8, column=0, columnspan=8, sticky="ew", pady=(14, 4))
        for column in range(4):
            single_button_frame.columnconfigure(column, weight=1)
        ttk.Button(
            single_button_frame,
            text="选择下一个点\nSelect Next Point",
            command=self._select_next_single_point).grid(
                row=0, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(
            single_button_frame,
            text="移动到这个点\nMove To Point",
            command=self._move_to_selected_single_point).grid(
                row=0, column=1, sticky="ew", padx=8)
        ttk.Button(
            single_button_frame,
            text="优化这个点\nOptimize Point",
            command=self._start_single_point_experiment_motion).grid(
                row=0, column=2, sticky="ew", padx=8)
        ttk.Button(
            single_button_frame,
            text="返回home点\nReturn Home",
            command=lambda: self._reset_to_home_for_experiment("point")).grid(
                row=0, column=3, sticky="ew", padx=(8, 0))

        ttk.Label(
            single_frame,
            textvariable=self.single_point_status_var,
            foreground="#204060").grid(
                row=8, column=0, columnspan=8, sticky="w", pady=(10, 0))

        single_hint = ttk.Label(
            single_frame,
            text="说明：选择点只更新当前目标；移动到点使用五次多项式直达，不调用优化器；优化点才调用单点 whole-body 优化。",
            justify=tk.LEFT,
            foreground="#204060")
        single_hint.grid(row=9, column=0, columnspan=8, sticky="w", pady=(12, 0))

    def _build_vlm_page(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(2, weight=1)

        title = ttk.Label(
            parent,
            text="VLM 配置与结果",
            font=("TkDefaultFont", 16, "bold"))
        title.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 14))

        left_top = ttk.LabelFrame(parent, text="运行模式", padding=14)
        left_top.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))
        left_top.columnconfigure(1, weight=1)

        ttk.Radiobutton(
            left_top,
            text="在线 API 模式 / Online API Mode",
            variable=self.vlm_mode_var,
            value="online_api").grid(
                row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))
        ttk.Radiobutton(
            left_top,
            text="离线结果模式 / Offline Saved Result Mode",
            variable=self.vlm_mode_var,
            value="offline_result").grid(
                row=1, column=0, columnspan=2, sticky="w", pady=(0, 12))

        self.vlm_online_frame = ttk.LabelFrame(parent, text="在线 API 设置", padding=14)
        self.vlm_online_frame.grid(row=2, column=0, sticky="nsew", padx=(0, 8))
        self.vlm_online_frame.columnconfigure(1, weight=1)

        self._add_labeled_entry(
            self.vlm_online_frame, 0, "模型 / Model", self.vlm_model_var)
        self._add_labeled_entry(
            self.vlm_online_frame, 1, "本地服务地址 / Local Base URL",
            self.vlm_local_base_url_var)
        self._add_labeled_entry(
            self.vlm_online_frame, 2, "API Key", self.vlm_api_key_var, show="*")
        self._add_labeled_entry(
            self.vlm_online_frame, 3, "Gemini API Key",
            self.vlm_gemini_api_key_var, show="*")
        self._add_labeled_entry(
            self.vlm_online_frame, 4, "固定照片路径 / Fixed Image",
            self.vlm_default_image_path_var)
        self._add_labeled_entry(
            self.vlm_online_frame, 5, "结果 JSON 路径 / Result JSON",
            self.vlm_result_json_path_var)
        ttk.Checkbutton(
            self.vlm_online_frame,
            text="自动保存 API 结果 / Auto Save API Result",
            variable=self.vlm_auto_save_var).grid(
                row=6, column=0, columnspan=2, sticky="w", pady=(10, 0))

        image_source_frame = ttk.LabelFrame(
            self.vlm_online_frame, text="在线测试图像来源", padding=10)
        image_source_frame.grid(
            row=7, column=0, columnspan=2, sticky="ew", pady=(14, 0))
        image_source_frame.columnconfigure(0, weight=1)
        image_source_frame.columnconfigure(1, weight=1)

        ttk.Radiobutton(
            image_source_frame,
            text="固定照片 / Fixed Photo",
            variable=self.vlm_image_source_var,
            value="fixed_photo").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            image_source_frame,
            text="RealSense 相机 / RealSense Camera",
            variable=self.vlm_image_source_var,
            value="realsense").grid(row=0, column=1, sticky="w")

        self.vlm_realsense_frame = ttk.LabelFrame(
            self.vlm_online_frame, text="RealSense 预览", padding=10)
        self.vlm_realsense_frame.grid(
            row=8, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        self.vlm_realsense_frame.columnconfigure(1, weight=1)

        ttk.Label(
            self.vlm_realsense_frame,
            text="相机图像由 RealSense 节点本地窗口显示，GUI 不再接收 raw 图像。\n"
                 "Camera image is shown by the RealSense node window; GUI no longer receives raw images.",
            foreground="#204060",
            justify=tk.LEFT).grid(
                row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))
        ttk.Label(
            self.vlm_realsense_frame,
            text="快照服务 / Capture Service: {}".format(
                self.vlm_realsense_capture_service),
            foreground="#204060",
            justify=tk.LEFT).grid(
                row=1, column=0, columnspan=2, sticky="w", pady=(0, 6))

        self.vlm_realsense_status_label = ttk.Label(
            self.vlm_realsense_frame,
            textvariable=self.vlm_realsense_status_var,
            foreground="#204060",
            justify=tk.LEFT)
        self.vlm_realsense_status_label.grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

        self.vlm_offline_frame = ttk.LabelFrame(parent, text="离线结果设置", padding=14)
        self.vlm_offline_frame.grid(row=2, column=0, sticky="nsew", padx=(0, 8))
        self.vlm_offline_frame.columnconfigure(1, weight=1)
        self.vlm_offline_frame.rowconfigure(6, weight=1)

        self._add_labeled_entry(
            self.vlm_offline_frame, 0, "结果 JSON 路径 / Result JSON",
            self.vlm_result_json_path_var)
        self._add_labeled_entry(
            self.vlm_offline_frame, 1, "物体质量 / Object Mass (kg)",
            self.vlm_offline_mass_var)
        self._add_labeled_entry(
            self.vlm_offline_frame, 2, "质量范围最小值 / Range Min (kg)",
            self.vlm_offline_range_min_var)
        self._add_labeled_entry(
            self.vlm_offline_frame, 3, "质量范围最大值 / Range Max (kg)",
            self.vlm_offline_range_max_var)
        self._add_labeled_entry(
            self.vlm_offline_frame, 4, "材料猜测 / Material Guess",
            self.vlm_offline_material_var)
        self._add_labeled_entry(
            self.vlm_offline_frame, 5, "物体描述 / Object Description",
            self.vlm_offline_description_var)
        self._add_labeled_entry(
            self.vlm_offline_frame, 6, "置信度 / Confidence",
            self.vlm_offline_confidence_var)

        offline_reasoning_label = ttk.Label(
            self.vlm_offline_frame, text="离线推理说明 / Offline Reasoning")
        offline_reasoning_label.grid(row=7, column=0, sticky="nw", pady=(10, 0))

        offline_reasoning_frame = ttk.Frame(self.vlm_offline_frame)
        offline_reasoning_frame.grid(
            row=7, column=1, sticky="nsew", pady=(10, 0), padx=(10, 0))
        offline_reasoning_frame.columnconfigure(0, weight=1)
        offline_reasoning_frame.rowconfigure(0, weight=1)

        self.vlm_offline_reasoning_editor = tk.Text(
            offline_reasoning_frame,
            height=6,
            wrap=tk.WORD)
        self.vlm_offline_reasoning_editor.grid(row=0, column=0, sticky="nsew")
        offline_reasoning_scrollbar = ttk.Scrollbar(
            offline_reasoning_frame,
            orient=tk.VERTICAL,
            command=self.vlm_offline_reasoning_editor.yview)
        offline_reasoning_scrollbar.grid(row=0, column=1, sticky="ns")
        self.vlm_offline_reasoning_editor.configure(
            yscrollcommand=offline_reasoning_scrollbar.set)
        self._set_offline_reasoning_text(self.vlm_offline_reasoning_value)

        right_top = ttk.LabelFrame(parent, text="请求策略", padding=14)
        right_top.grid(row=1, column=1, sticky="nsew", padx=(8, 0), pady=(0, 8))
        right_top.columnconfigure(1, weight=1)

        ttk.Checkbutton(
            right_top,
            text="规划时请求 VLM / Request VLM During Planning",
            variable=self.planner_request_vlm_var).grid(
                row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
        estimate_policy_frame = ttk.LabelFrame(
            right_top, text="估计策略 / Estimate Policy", padding=8)
        estimate_policy_frame.grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        estimate_policy_frame.columnconfigure(0, weight=1)
        estimate_policy_frame.columnconfigure(1, weight=1)
        ttk.Radiobutton(
            estimate_policy_frame,
            text="使用最后一次估计 / Reuse Last",
            variable=self.vlm_estimate_policy_var,
            value="reuse_last").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            estimate_policy_frame,
            text="重新估计 / Re-estimate",
            variable=self.vlm_estimate_policy_var,
            value="reestimate").grid(row=0, column=1, sticky="w")
        if hasattr(self, "planner_vlm_force_refresh_var"):
            self.planner_vlm_force_refresh_var.set(
                self.vlm_estimate_policy_var.get() == "reestimate")
        self._add_labeled_entry(
            right_top, 2, "请求超时 / Timeout (s)", self.planner_vlm_timeout_var)
        self._add_labeled_entry(
            right_top, 3, "默认负载质量 / Default Payload Mass (kg)",
            self.planner_default_payload_mass_var)
        self._add_labeled_entry(
            right_top, 4, "规划请求图像 / Planner Request Image",
            self.planner_vlm_image_path_var)

        button_frame = ttk.Frame(right_top)
        button_frame.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(16, 0))
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(2, weight=1)

        ttk.Button(
            button_frame,
            text="应用设置\nApply Settings",
            command=self._apply_vlm_settings).grid(
                row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(
            button_frame,
            text="重启VLM\nRestart VLM",
            command=self._restart_vlm).grid(
                row=0, column=1, sticky="ew", padx=6)
        ttk.Button(
            button_frame,
            text="测试质量估计\nTest Estimate",
            command=self._test_vlm_request).grid(
                row=0, column=2, sticky="ew", padx=(6, 0))

        hint_label = ttk.Label(
            right_top,
            text=(
                "说明：\n"
                "1. 在线模式会调用 API；离线模式会直接读取/发布当前 JSON 里的结果。\n"
                "2. 离线模式下可直接填写质量、范围、材料和描述，再点击应用或重启 VLM。\n"
                "3. 使用最后一次估计时会立即读取结果 JSON，不重新拍照，也不调用 API。\n"
                "4. 重新估计时才会请求 RealSense/API；规划器侧的超时/图像/默认质量会在下一次启动 planner 进程时读取。\n"
                "5. 当前页面的“测试质量估计”会直接向 VLM topic 发送一次请求。"
            ),
            justify=tk.LEFT)
        hint_label.grid(row=6, column=0, columnspan=2, sticky="w", pady=(16, 0))

        right_bottom = ttk.LabelFrame(parent, text="最近一次 VLM 结果", padding=14)
        right_bottom.grid(row=2, column=1, rowspan=2, sticky="nsew", padx=(8, 0))
        right_bottom.columnconfigure(1, weight=1)
        right_bottom.rowconfigure(8, weight=1)

        self._add_result_row(right_bottom, 0, "状态 / Status", self.vlm_result_status_var)
        self._add_result_row(right_bottom, 1, "来源 / Source", self.vlm_result_source_var)
        self._add_result_row(right_bottom, 2, "模型 / Model", self.vlm_result_model_var)
        self._add_result_row(right_bottom, 3, "质量 / Mass", self.vlm_result_mass_var)
        self._add_result_row(right_bottom, 4, "范围 / Range", self.vlm_result_range_var)
        self._add_result_row(right_bottom, 5, "材料 / Material", self.vlm_result_material_var)
        self._add_result_row(right_bottom, 6, "置信度 / Confidence", self.vlm_result_confidence_var)
        self._add_result_row(right_bottom, 7, "图像 / Image", self.vlm_result_image_var)
        self._add_result_row(right_bottom, 8, "JSON / Result JSON", self.vlm_result_json_var)

        reasoning_label = ttk.Label(right_bottom, text="推理 / Reasoning")
        reasoning_label.grid(row=9, column=0, sticky="nw", pady=(10, 0))

        reasoning_frame = ttk.Frame(right_bottom)
        reasoning_frame.grid(row=9, column=1, sticky="nsew", pady=(10, 0))
        reasoning_frame.columnconfigure(0, weight=1)
        reasoning_frame.rowconfigure(0, weight=1)

        self.vlm_reasoning_text = tk.Text(
            reasoning_frame,
            height=8,
            wrap=tk.WORD,
            state=tk.DISABLED)
        self.vlm_reasoning_text.grid(row=0, column=0, sticky="nsew")
        reasoning_scrollbar = ttk.Scrollbar(
            reasoning_frame, orient=tk.VERTICAL, command=self.vlm_reasoning_text.yview)
        reasoning_scrollbar.grid(row=0, column=1, sticky="ns")
        self.vlm_reasoning_text.configure(yscrollcommand=reasoning_scrollbar.set)
        self._update_vlm_reasoning_text(self.vlm_result_reasoning_text)
        self._bind_vlm_mode_visibility()
        self._bind_vlm_image_source_visibility()
        self._update_vlm_mode_visibility()
        self._update_vlm_image_source_visibility()

    def _bind_vlm_mode_visibility(self) -> None:
        if getattr(self, "_vlm_mode_trace_registered", False):
            return
        self.vlm_mode_var.trace_add("write", self._on_vlm_mode_changed)
        self._vlm_mode_trace_registered = True

    def _on_vlm_mode_changed(self, *_args) -> None:
        self._update_vlm_mode_visibility()

    def _bind_vlm_image_source_visibility(self) -> None:
        if getattr(self, "_vlm_image_source_trace_registered", False):
            return
        self.vlm_image_source_var.trace_add("write", self._on_vlm_image_source_changed)
        self._vlm_image_source_trace_registered = True

    def _on_vlm_image_source_changed(self, *_args) -> None:
        self._update_vlm_image_source_visibility()

    def _update_vlm_image_source_visibility(self) -> None:
        if not hasattr(self, "vlm_realsense_frame"):
            return
        if self._vlm_uses_realsense():
            self.vlm_realsense_frame.grid()
        else:
            self.vlm_realsense_frame.grid_remove()

    def _update_vlm_mode_visibility(self) -> None:
        if not hasattr(self, "vlm_online_frame") or not hasattr(self, "vlm_offline_frame"):
            return

        if self.vlm_mode_var.get() == "online_api":
            self.vlm_offline_frame.grid_remove()
            self.vlm_online_frame.grid()
        else:
            self.vlm_online_frame.grid_remove()
            self.vlm_offline_frame.grid()

    def _build_force_capability_page(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(1, weight=1)

        title = ttk.Label(
            parent,
            text="力操作能力参数 / Force Capability Settings",
            font=("TkDefaultFont", 16, "bold"))
        title.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 14))

        limits_frame = ttk.LabelFrame(parent, text="Franka 7关节力矩限制", padding=14)
        limits_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))
        limits_frame.columnconfigure(1, weight=1)

        for row, joint_name in enumerate(self.force_capability_joint_names):
            label = ttk.Label(
                limits_frame,
                text="{} / Torque Limit (Nm)".format(joint_name))
            label.grid(row=row, column=0, sticky="w", pady=6)
            entry = ttk.Entry(
                limits_frame,
                textvariable=self.force_capability_torque_limit_vars[row])
            entry.grid(row=row, column=1, sticky="ew", padx=(10, 0), pady=6)

        button_frame = ttk.Frame(limits_frame)
        button_frame.grid(
            row=len(self.force_capability_joint_names),
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(18, 0))
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

        ttk.Button(
            button_frame,
            text="恢复统一默认值\nReset Defaults",
            command=self._reset_force_capability_limits_to_default).grid(
                row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(
            button_frame,
            text="更新力矩限制\nApply Torque Limits",
            command=self._apply_force_capability_settings).grid(
                row=0, column=1, sticky="ew", padx=(6, 0))

        status_label = ttk.Label(
            limits_frame,
            textvariable=self.force_capability_status_var,
            foreground="#204060",
            justify=tk.LEFT)
        status_label.grid(
            row=len(self.force_capability_joint_names) + 1,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(12, 0))

        info_frame = ttk.LabelFrame(parent, text="说明", padding=14)
        info_frame.grid(row=1, column=1, sticky="nsew", padx=(8, 0), pady=(0, 8))
        info_frame.columnconfigure(0, weight=1)

        hint_label = ttk.Label(
            info_frame,
            text=(
                "1. 这里的默认值来自 moca_trajectory_generator/config/force_capability_settings.yaml。\n"
                "2. 点击“更新力矩限制”后，GUI 会发布动态设置，并同步写入同一组 ROS 参数。\n"
                "3. 控制器会立刻用这组值做关节力矩限幅；规划器会用这组值更新后续的力能力优化。\n"
                "4. 这一页先只开放 Franka 7 个关节的力矩限制，后续其它参数可以继续往这里加。"
            ),
            justify=tk.LEFT)
        hint_label.grid(row=0, column=0, sticky="w")

    def _build_teaching_home_page(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(2, weight=1)

        title = ttk.Label(
            parent,
            text="示教记录与位置回Home / Teaching Record and Position Home Return",
            font=("TkDefaultFont", 16, "bold"))
        title.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 14))

        control_frame = ttk.LabelFrame(parent, text="节点与动作 / Nodes and Actions", padding=14)
        control_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))
        control_frame.columnconfigure(0, weight=1)
        control_frame.columnconfigure(1, weight=1)

        self._add_action_button(
            control_frame,
            row=0,
            label="启动示教采样节点\nStart Teaching Monitor",
            command=self._start_teaching_monitor,
            status_key="teaching")
        self._add_action_button(
            control_frame,
            row=1,
            label="启动位置回Home节点\nStart Home Return",
            command=self._start_home_return,
            status_key="home_return")
        franka_topic_frame = ttk.Frame(control_frame)
        franka_topic_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(12, 4))
        franka_topic_frame.columnconfigure(1, weight=1)
        ttk.Label(franka_topic_frame, text="Franka状态topic\nFranka state topic").grid(
            row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(
            franka_topic_frame,
            textvariable=self.teaching_robot_state_topic_var).grid(
                row=0, column=1, sticky="ew", padx=(0, 8))
        ttk.Button(
            franka_topic_frame,
            text="订阅",
            command=self._subscribe_teaching_robot_state).grid(
                row=0, column=2, sticky="ew")
        ttk.Label(franka_topic_frame, text="底盘Odom topic\nBase odom topic").grid(
            row=1, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        ttk.Entry(
            franka_topic_frame,
            textvariable=self.teaching_odom_topic_var).grid(
                row=1, column=1, sticky="ew", padx=(0, 8), pady=(8, 0))
        ttk.Button(
            franka_topic_frame,
            text="订阅",
            command=self._subscribe_teaching_odom).grid(
                row=1, column=2, sticky="ew", pady=(8, 0))

        ttk.Button(
            control_frame,
            text="记录当前点\nRecord Current Point",
            command=self._record_current_teaching_point).grid(
                row=3, column=0, sticky="ew", padx=(0, 10), pady=(18, 8))
        ttk.Button(
            control_frame,
            text="清除上一点\nRemove Last Point",
            command=self._remove_last_teaching_point).grid(
                row=3, column=1, sticky="ew", pady=(18, 8))
        ttk.Button(
            control_frame,
            text="清除所有点\nClear All Points",
            command=self._clear_all_teaching_points).grid(
                row=4, column=0, sticky="ew", padx=(0, 10), pady=8)
        ttk.Button(
            control_frame,
            text="保存记录\nSave Points",
            command=self._save_teaching_points).grid(
                row=4, column=1, sticky="ew", pady=8)

        path_frame = ttk.Frame(control_frame)
        path_frame.grid(row=5, column=0, columnspan=2, sticky="ew", pady=8)
        path_frame.columnconfigure(1, weight=1)
        ttk.Label(path_frame, text="保存路径\nSave path").grid(
            row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(
            path_frame,
            textvariable=self.teaching_points_output_path_var).grid(
                row=0, column=1, sticky="ew")

        ttk.Separator(control_frame).grid(
            row=6, column=0, columnspan=2, sticky="ew", pady=(12, 8))

        ttk.Button(
            control_frame,
            text="把最后一次记录设为Home\nSet Last Record as Home",
            command=self._set_latest_record_as_home).grid(
                row=7, column=0, sticky="ew", padx=(0, 10), pady=8)
        ttk.Button(
            control_frame,
            text="把当前状态设为Home\nSet Current as Home",
            command=self._set_current_state_as_home).grid(
                row=7, column=1, sticky="ew", pady=8)
        ttk.Button(
            control_frame,
            text="位置控制返回Home\nReturn Home by Position",
            command=self._start_position_home_return).grid(
                row=8, column=0, sticky="ew", padx=(0, 10), pady=8)
        ttk.Button(
            control_frame,
            text="停止位置回Home\nStop Home Return",
            command=self._stop_position_home_return).grid(
                row=8, column=1, sticky="ew", pady=8)

        status_label = ttk.Label(
            control_frame,
            textvariable=self.teaching_status_var,
            foreground="#204060",
            justify=tk.LEFT)
        status_label.grid(row=9, column=0, columnspan=2, sticky="w", pady=(12, 0))
        home_status_label = ttk.Label(
            control_frame,
            textvariable=self.home_return_status_var,
            foreground="#204060",
            justify=tk.LEFT)
        home_status_label.grid(row=10, column=0, columnspan=2, sticky="w", pady=(6, 0))
        force_capacity_label = ttk.Label(
            control_frame,
            textvariable=self.teaching_directional_force_capacity_var,
            foreground="#204060",
            justify=tk.LEFT)
        force_capacity_label.grid(row=11, column=0, columnspan=2, sticky="w", pady=(6, 0))
        manipulability_label = ttk.Label(
            control_frame,
            textvariable=self.teaching_manipulability_var,
            foreground="#204060",
            justify=tk.LEFT)
        manipulability_label.grid(row=12, column=0, columnspan=2, sticky="w", pady=(6, 0))

        hint_frame = ttk.LabelFrame(parent, text="说明 / Notes", padding=14)
        hint_frame.grid(row=1, column=1, sticky="nsew", padx=(8, 0), pady=(0, 8))
        hint_frame.columnconfigure(0, weight=1)
        hint_label = ttk.Label(
            hint_frame,
            text=(
                "1. 示教节点只读取状态，不发送运动指令，适合急停/示教模式下记录位姿。\n"
                "2. 记录向量顺序：7个关节、odom x/y/yaw、末端 pose 的 xyz + quaternion。\n"
                "3. 记录功能不依赖 Home Return 节点；没有 odom 话题时会用 moca_state 前三维作为 base pose。\n"
                "4. 位置回Home会向关节位置控制器发 JointTrajectory，底盘通过 odom PID 发 cmd_vel。\n"
                "5. 回Home前请确认关节 position controller 已启动，且底盘 cmd_vel 话题没有被其它控制器同时占用。"
            ),
            justify=tk.LEFT)
        hint_label.grid(row=0, column=0, sticky="w")

        current_frame = ttk.LabelFrame(parent, text="当前状态 / Current Sample", padding=12)
        current_frame.grid(row=2, column=0, sticky="nsew", padx=(0, 8), pady=(8, 0))
        current_frame.columnconfigure(0, weight=1)
        current_frame.rowconfigure(3, weight=1)
        ttk.Label(
            current_frame,
            textvariable=self.teaching_current_summary_var,
            foreground="#204060").grid(row=0, column=0, sticky="w", pady=(0, 6))
        ttk.Label(
            current_frame,
            textvariable=self.teaching_directional_force_capacity_var,
            foreground="#204060").grid(row=1, column=0, sticky="w", pady=(0, 6))
        ttk.Label(
            current_frame,
            textvariable=self.teaching_manipulability_var,
            foreground="#204060").grid(row=2, column=0, sticky="w", pady=(0, 6))
        self.teaching_current_text = tk.Text(
            current_frame,
            height=18,
            wrap=tk.WORD,
            state=tk.DISABLED,
            exportselection=False)
        self.teaching_current_text.grid(row=3, column=0, sticky="nsew")
        current_scrollbar = ttk.Scrollbar(
            current_frame, orient=tk.VERTICAL, command=self.teaching_current_text.yview)
        current_scrollbar.grid(row=3, column=1, sticky="ns")
        self.teaching_current_text.configure(yscrollcommand=current_scrollbar.set)

        records_frame = ttk.LabelFrame(parent, text="记录向量 / Recorded Vectors", padding=12)
        records_frame.grid(row=2, column=1, sticky="nsew", padx=(8, 0), pady=(8, 0))
        records_frame.columnconfigure(0, weight=1)
        records_frame.rowconfigure(0, weight=1)
        self.teaching_records_text = tk.Text(
            records_frame,
            height=18,
            wrap=tk.WORD,
            state=tk.DISABLED,
            exportselection=False)
        self.teaching_records_text.grid(row=0, column=0, sticky="nsew")
        records_scrollbar = ttk.Scrollbar(
            records_frame, orient=tk.VERTICAL, command=self.teaching_records_text.yview)
        records_scrollbar.grid(row=0, column=1, sticky="ns")
        self.teaching_records_text.configure(yscrollcommand=records_scrollbar.set)

    def _build_plot_data_page(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(2, weight=1)

        title = ttk.Label(
            parent,
            text="画图与数据保存 / Plotting and Data Storage",
            font=("TkDefaultFont", 16, "bold"))
        title.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 14))

        control_frame = ttk.LabelFrame(parent, text="记录控制 / Recording Control", padding=14)
        control_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))
        control_frame.columnconfigure(0, weight=1)
        control_frame.columnconfigure(1, weight=1)

        self._add_action_button(
            control_frame,
            row=0,
            label="启动绘图记录节点\nStart Plot Recorder",
            command=self._start_plot_data_recorder,
            status_key="plot_data_recorder")
        ttk.Button(
            control_frame,
            text="等待记录\nArm Waiting Record",
            command=self._wait_plot_data_recording).grid(
                row=1, column=0, sticky="ew", padx=(0, 10), pady=8)
        ttk.Button(
            control_frame,
            text="开始记录\nStart Recording",
            command=self._start_plot_data_recording).grid(
                row=1, column=1, sticky="ew", pady=8)
        ttk.Button(
            control_frame,
            text="停止并生成文件\nStop and Render",
            command=self._stop_plot_data_recording).grid(
                row=2, column=0, columnspan=2, sticky="ew", pady=8)
        ttk.Button(
            control_frame,
            text="记录单点\nRecord Point",
            command=self._record_single_plot_point).grid(
                row=3, column=0, sticky="ew", padx=(0, 10), pady=8)
        ttk.Button(
            control_frame,
            text="清除上一个单点\nUndo Point",
            command=self._undo_single_plot_point).grid(
                row=3, column=1, sticky="ew", pady=8)
        ttk.Button(
            control_frame,
            text="完成单点并出图\nFinish Point Figures",
            command=self._finish_single_plot_points).grid(
                row=4, column=0, columnspan=2, sticky="ew", pady=8)

        status_label = ttk.Label(
            control_frame,
            textvariable=self.plot_status_var,
            foreground="#204060",
            justify=tk.LEFT)
        status_label.grid(row=5, column=0, columnspan=2, sticky="w", pady=(12, 0))

        options_frame = ttk.LabelFrame(parent, text="保存内容 / Save Items", padding=14)
        options_frame.grid(row=1, column=1, sticky="nsew", padx=(8, 0), pady=(0, 8))
        options_frame.columnconfigure(0, weight=1)

        ttk.Checkbutton(
            options_frame,
            text="力输出能力 / Directional force capacity",
            variable=self.plot_save_force_capacity_var).grid(
                row=0, column=0, sticky="w", pady=6)
        ttk.Checkbutton(
            options_frame,
            text="可操作度曲线 / Manipulability curve",
            variable=self.plot_save_manipulability_var).grid(
                row=1, column=0, sticky="w", pady=6)
        ttk.Checkbutton(
            options_frame,
            text="可操作度椭球全过程视频 / Manipulability ellipsoid video",
            variable=self.plot_save_manipulability_ellipsoid_var).grid(
                row=2, column=0, sticky="w", pady=6)
        ttk.Checkbutton(
            options_frame,
            text="剩余多面体可视化 / Residual polytope animation",
            variable=self.plot_save_residual_polytope_var).grid(
                row=3, column=0, sticky="w", pady=6)
        ttk.Checkbutton(
            options_frame,
            text="对比实验完成后自动停止 / Auto-stop after comparison complete",
            variable=self.plot_auto_stop_on_comparison_complete_var).grid(
                row=4, column=0, sticky="w", pady=6)

        settings_frame = ttk.LabelFrame(parent, text="路径与时间 / Paths and Timing", padding=14)
        settings_frame.grid(row=2, column=0, sticky="nsew", padx=(0, 8), pady=(8, 0))
        settings_frame.columnconfigure(1, weight=1)

        self._add_labeled_entry(
            settings_frame,
            row=0,
            label_text="结束后继续保存时间(s)\nPost-plan save time (s)",
            variable=self.plot_end_time_var)
        self._add_labeled_entry(
            settings_frame,
            row=1,
            label_text="数据保存路径\nData output path",
            variable=self.plot_data_output_dir_var)
        self._add_labeled_entry(
            settings_frame,
            row=2,
            label_text="图片/动图保存路径\nFigure output path",
            variable=self.plot_fig_output_dir_var)

        note_frame = ttk.LabelFrame(parent, text="说明 / Notes", padding=14)
        note_frame.grid(row=2, column=1, sticky="nsew", padx=(8, 0), pady=(8, 0))
        note_frame.columnconfigure(0, weight=1)
        note_label = ttk.Label(
            note_frame,
            text=(
                "1. 所有输出文件都会自动带年月日时分秒时间戳，避免覆盖旧数据。\n"
                "2. JSON 会保存完整原始数据；CSV 方便单独画力能力、可操作度、椭球矩阵和目标轨迹。\n"
                "3. 如果控制器发布 force_polytope_vertices，就用真实顶点生成透明蓝色力多面体。\n"
                "4. 可操作度椭球视频使用控制器发布的 3x3 translational manipulability Gram 矩阵。\n"
                "5. 如果暂时没有顶点话题，动图会退化为方向力能力球壳，并在 manifest 里标注 fallback。\n"
                "6. 等待记录会在 /target_pose 开始变化时自动开始，目标稳定后再延迟保存并弹出结果。\n"
                "7. 单点按钮会冻结当前期望力、方向力能力、带宽、可操作度和力多面体；完成后生成静态图片。"
            ),
            justify=tk.LEFT)
        note_label.grid(row=0, column=0, sticky="w")

    def _build_metric_merge_page(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        title = ttk.Label(
            parent,
            text="对比实验数据合并 / Metric Merge",
            font=("TkDefaultFont", 16, "bold"))
        title.grid(row=0, column=0, sticky="w", pady=(0, 14))

        force_frame = ttk.LabelFrame(parent, text="Force Capacity", padding=14)
        force_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        self._build_metric_merge_row(
            force_frame,
            metric_key="force_capacity",
            input_a_var=self.merge_force_capacity_input_a_var,
            input_b_var=self.merge_force_capacity_input_b_var,
            label_a_var=self.merge_force_capacity_label_a_var,
            label_b_var=self.merge_force_capacity_label_b_var,
            output_var=self.merge_force_capacity_output_var)

        manipulability_frame = ttk.LabelFrame(parent, text="Manipulability", padding=14)
        manipulability_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 10))
        self._build_metric_merge_row(
            manipulability_frame,
            metric_key="manipulability",
            input_a_var=self.merge_manipulability_input_a_var,
            input_b_var=self.merge_manipulability_input_b_var,
            label_a_var=self.merge_manipulability_label_a_var,
            label_b_var=self.merge_manipulability_label_b_var,
            output_var=self.merge_manipulability_output_var)

        status_label = ttk.Label(
            parent,
            textvariable=self.metric_merge_status_var,
            foreground="#204060",
            justify=tk.LEFT)
        status_label.grid(row=3, column=0, sticky="w", pady=(6, 0))

    def _build_metric_merge_row(
        self,
        parent: ttk.LabelFrame,
        metric_key: str,
        input_a_var: tk.StringVar,
        input_b_var: tk.StringVar,
        label_a_var: tk.StringVar,
        label_b_var: tk.StringVar,
        output_var: tk.StringVar,
    ) -> None:
        parent.columnconfigure(1, weight=1)
        self._add_labeled_entry(parent, 0, "数据1 CSV\nInput 1", input_a_var)
        self._add_labeled_entry(parent, 1, "数据2 CSV\nInput 2", input_b_var)
        self._add_labeled_entry(parent, 2, "数据1标注\nLabel 1", label_a_var)
        self._add_labeled_entry(parent, 3, "数据2标注\nLabel 2", label_b_var)
        self._add_labeled_entry(parent, 4, "输出图片\nOutput PNG", output_var)
        ttk.Button(
            parent,
            text="合并生成图像\nMerge and Plot",
            command=lambda key=metric_key: self._merge_metric_csvs(key)).grid(
                row=5, column=0, columnspan=2, sticky="ew", pady=(10, 0))

    def _build_payload_identification_page(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(3, weight=1)

        title = ttk.Label(
            parent,
            text="末端负载质量与质心估计 / Payload Mass and Center of Mass",
            font=("TkDefaultFont", 16, "bold"))
        title.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 14))

        topic_frame = ttk.LabelFrame(parent, text="数据源 / Data Source", padding=14)
        topic_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))
        topic_frame.columnconfigure(1, weight=1)
        self._add_labeled_entry(
            topic_frame,
            row=0,
            label_text="HRII robot_state topic",
            variable=self.payload_robot_state_topic_var)
        ttk.Label(topic_frame, text="Wrench frame").grid(row=1, column=0, sticky="w", pady=6)
        frame_selector = ttk.Combobox(
            topic_frame,
            textvariable=self.payload_wrench_frame_var,
            values=("stiffness", "base"),
            state="readonly")
        frame_selector.grid(row=1, column=1, sticky="ew", pady=6, padx=(10, 0))
        self._add_labeled_entry(
            topic_frame,
            row=2,
            label_text="重力加速度 / Gravity (m/s^2)",
            variable=self.payload_gravity_var)
        ttk.Label(topic_frame, text="Wrench 数据模式").grid(
            row=3, column=0, sticky="w", pady=6)
        data_mode_selector = ttk.Combobox(
            topic_frame,
            textvariable=self.payload_wrench_data_mode_var,
            values=("raw", "bias_compensated"),
            state="readonly")
        data_mode_selector.grid(row=3, column=1, sticky="ew", pady=6, padx=(10, 0))
        self._add_action_button(
            topic_frame,
            row=4,
            label="启动Franka只读状态节点\nStart Franka Read-Only State",
            command=self._start_franka_state_bridge,
            status_key="franka_state_bridge")
        ttk.Button(
            topic_frame,
            text="按当前模式更新topic\nUse Current Robot/Controller Topic",
            command=self._payload_set_topic_from_active_robot).grid(
                row=5, column=0, sticky="ew", pady=(10, 0))
        ttk.Button(
            topic_frame,
            text="重新订阅\nResubscribe",
            command=self._subscribe_payload_robot_state).grid(
                row=5, column=1, sticky="ew", padx=(10, 0), pady=(10, 0))

        action_frame = ttk.LabelFrame(parent, text="采样与估计 / Sampling and Estimate", padding=14)
        action_frame.grid(row=1, column=1, sticky="nsew", padx=(8, 0), pady=(0, 8))
        action_frame.columnconfigure(0, weight=1)
        action_frame.columnconfigure(1, weight=1)
        ttk.Button(
            action_frame,
            text="置零当前力偏置\nSet Current Wrench Bias",
            command=self._payload_set_current_bias).grid(
                row=0, column=0, sticky="ew", padx=(0, 8), pady=6)
        ttk.Button(
            action_frame,
            text="记录当前点\nRecord Current Point",
            command=self._payload_record_current_sample).grid(
                row=0, column=1, sticky="ew", pady=6)
        ttk.Button(
            action_frame,
            text="计算质量和质心\nEstimate Mass and COM",
            command=self._payload_estimate_mass_com).grid(
                row=1, column=0, sticky="ew", padx=(0, 8), pady=6)
        ttk.Button(
            action_frame,
            text="清空记录\nClear Samples",
            command=self._payload_clear_samples).grid(
                row=1, column=1, sticky="ew", pady=6)
        ttk.Button(
            action_frame,
            text="导出结果JSON\nExport Result JSON",
            command=self._payload_export_result_json).grid(
                row=2, column=0, columnspan=2, sticky="ew", pady=6)
        self._add_labeled_entry(
            action_frame,
            row=3,
            label_text="输出路径\nOutput directory",
            variable=self.payload_output_dir_var)
        ttk.Label(
            action_frame,
            textvariable=self.payload_status_var,
            foreground="#204060",
            justify=tk.LEFT).grid(row=4, column=0, columnspan=2, sticky="w", pady=(10, 0))

        result_frame = ttk.LabelFrame(parent, text="估计结果 / Estimated Result", padding=14)
        result_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(8, 8))
        result_frame.columnconfigure(1, weight=1)
        self._add_result_row(
            result_frame,
            row=0,
            label_text="质量 / Mass",
            variable=self.payload_result_mass_var)
        self._add_result_row(
            result_frame,
            row=1,
            label_text="质心 / COM in selected EE frame",
            variable=self.payload_result_com_var)
        self._add_result_row(
            result_frame,
            row=2,
            label_text="拟合质量 / Fit quality",
            variable=self.payload_result_quality_var)

        current_frame = ttk.LabelFrame(parent, text="当前数据 / Current Robot State", padding=12)
        current_frame.grid(row=3, column=0, sticky="nsew", padx=(0, 8), pady=(8, 0))
        current_frame.columnconfigure(0, weight=1)
        current_frame.rowconfigure(1, weight=1)
        ttk.Label(
            current_frame,
            textvariable=self.payload_latest_summary_var,
            foreground="#204060").grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.payload_current_text = tk.Text(
            current_frame,
            height=18,
            wrap=tk.WORD,
            state=tk.DISABLED,
            exportselection=False)
        self.payload_current_text.grid(row=1, column=0, sticky="nsew")
        current_scrollbar = ttk.Scrollbar(
            current_frame, orient=tk.VERTICAL, command=self.payload_current_text.yview)
        current_scrollbar.grid(row=1, column=1, sticky="ns")
        self.payload_current_text.configure(yscrollcommand=current_scrollbar.set)

        samples_frame = ttk.LabelFrame(parent, text="记录点 / Recorded Samples", padding=12)
        samples_frame.grid(row=3, column=1, sticky="nsew", padx=(8, 0), pady=(8, 0))
        samples_frame.columnconfigure(0, weight=1)
        samples_frame.rowconfigure(0, weight=1)
        self.payload_samples_text = tk.Text(
            samples_frame,
            height=18,
            wrap=tk.WORD,
            state=tk.DISABLED,
            exportselection=False)
        self.payload_samples_text.grid(row=0, column=0, sticky="nsew")
        samples_scrollbar = ttk.Scrollbar(
            samples_frame, orient=tk.VERTICAL, command=self.payload_samples_text.yview)
        samples_scrollbar.grid(row=0, column=1, sticky="ns")
        self.payload_samples_text.configure(yscrollcommand=samples_scrollbar.set)

        note_frame = ttk.LabelFrame(parent, text="说明 / Notes", padding=14)
        note_frame.grid(row=4, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
        note_frame.columnconfigure(0, weight=1)
        ttk.Label(
            note_frame,
            text=(
                "1. 推荐先在空载或已知零载状态点击“置零当前力偏置”，再装负载记录多个姿态。\n"
                "2. 默认使用 stiffness frame 的外力估计，质心结果表达在末端/刚度坐标系下。\n"
                "3. 如果机器人侧已经补偿外力，请选择 raw；只有需要扣除当前 GUI bias 时选择 bias_compensated。\n"
                "4. 每个记录点保存当前末端位姿 O_T_EE、base/stiffness 两套外力矩和实际用于估计的 wrench。\n"
                "5. 静态数据只能可靠估质量和质心；请至少记录 4-6 个方向差异明显的姿态。"
            ),
            justify=tk.LEFT).grid(row=0, column=0, sticky="w")

    def _build_placeholder_page(self, parent: ttk.Notebook, message: str) -> ttk.Frame:
        frame = ttk.Frame(parent, padding=24)
        label = ttk.Label(frame, text=message, font=("TkDefaultFont", 12))
        label.pack(anchor="center", expand=True)
        return frame

    def _reset_force_capability_limits_to_default(self) -> None:
        for index, default_value in enumerate(self.force_capability_default_torque_limits):
            self.force_capability_torque_limit_vars[index].set(
                "{:.6g}".format(default_value))
        self.force_capability_status_var.set("已恢复为统一默认值，等待点击更新。")

    def _apply_force_capability_settings(self) -> None:
        updated_limits: List[float] = []
        try:
            for index, variable in enumerate(self.force_capability_torque_limit_vars):
                raw_value = variable.get().strip()
                try:
                    torque_limit = float(raw_value)
                except ValueError:
                    raise ValueError(
                        "{} 的力矩限制不是合法数字。".format(
                            self.force_capability_joint_names[index]))
                if torque_limit <= 0.0:
                    raise ValueError(
                        "{} 的力矩限制必须大于 0。".format(
                            self.force_capability_joint_names[index]))
                updated_limits.append(abs(torque_limit))
                variable.set("{:.6g}".format(abs(torque_limit)))
        except ValueError as exc:
            self.force_capability_status_var.set(str(exc))
            self._log(str(exc))
            return

        rospy.set_param(
            "/force_capability_settings/joint_names",
            self.force_capability_joint_names)
        rospy.set_param(
            "/force_capability_settings/arm_torque_limits",
            updated_limits)

        msg = ForceCapabilitySettings()
        msg.header.stamp = rospy.Time.now()
        msg.joint_names = list(self.force_capability_joint_names)
        msg.arm_torque_limits = list(updated_limits)
        self.force_capability_settings_pub.publish(msg)

        self.force_capability_status_var.set(
            "已更新并广播到控制器/规划器：{}".format(
                ", ".join("{:.3f}".format(value) for value in updated_limits)))
        self._log(
            "已更新关节力矩限制：{}".format(
                ", ".join(
                    "{}={:.3f}".format(name, value)
                    for name, value in zip(self.force_capability_joint_names, updated_limits)
                )))

    def _add_labeled_entry(
        self,
        parent,
        row: int,
        label_text: str,
        variable: tk.StringVar,
        show: str = "",
    ) -> None:
        label = ttk.Label(parent, text=label_text)
        label.grid(row=row, column=0, sticky="w", pady=6)
        entry = ttk.Entry(parent, textvariable=variable, show=show)
        entry.grid(row=row, column=1, sticky="ew", pady=6, padx=(10, 0))

    def _add_result_row(self, parent, row: int, label_text: str, variable: tk.StringVar) -> None:
        label = ttk.Label(parent, text=label_text)
        label.grid(row=row, column=0, sticky="nw", pady=4)
        value = ttk.Label(parent, textvariable=variable, justify=tk.LEFT)
        value.grid(row=row, column=1, sticky="w", pady=4, padx=(10, 0))

    def _add_action_button(
        self,
        parent: ttk.LabelFrame,
        row: int,
        label: str,
        command,
        status_key: str,
    ) -> None:
        button = ttk.Button(parent, text=label, command=command)
        button.grid(row=row, column=0, sticky="ew", padx=(0, 10), pady=8)

        stop_button = ttk.Button(
            parent,
            text="关闭\nStop",
            command=lambda key=status_key: self._stop_process(key))
        stop_button.grid(row=row, column=1, sticky="ew", padx=(0, 10), pady=8)

        status_label = ttk.Label(parent, textvariable=self.status_vars[status_key], width=18)
        status_label.grid(row=row, column=2, sticky="w", pady=8)

    def _on_trajectory_selection_changed(self) -> None:
        if self.trajectory_selection.get() == "comparison_two_lifts":
            self.selection_label.config(text="当前选择：两次提起对比")
            self._log("已选择轨迹：两次提起对比。")
        elif self.trajectory_selection.get() == "manual_target_pose":
            self.selection_label.config(text="当前选择：姿态球模式")
            self._log("已选择轨迹：姿态球模式。将使用 RViz 姿态球给定末端目标位姿。")
            self._start_pose_marker()
        elif self.trajectory_selection.get() == "move_experiment":
            self.selection_label.config(text="当前选择：移动实验")
            self._open_move_experiment_page(log_selection=True)
        elif self.trajectory_selection.get() == "single_point_optimization":
            self.selection_label.config(text="当前选择：单点优化实验")
            self._open_single_point_experiment_page(log_selection=True)
        else:
            self.selection_label.config(text="当前选择：举升实验")
            self._open_lift_experiment_page(log_selection=True)

    def _open_lift_experiment_page(self, log_selection: bool = False) -> None:
        if self.trajectory_selection.get() != "lift_experiment":
            return

        if log_selection:
            self._log("已选择轨迹：举升实验。已切换到实验配置页面。")
        if hasattr(self, "experiment_page"):
            if not self._select_notebook_content_page(self.experiment_page):
                self._log("实验配置页面切换失败：未找到 Notebook tab。")

    def _open_move_experiment_page(self, log_selection: bool = False) -> None:
        if self.trajectory_selection.get() != "move_experiment":
            return

        if log_selection:
            self._log("已选择轨迹：移动实验。已切换到实验配置页面。")
        if hasattr(self, "experiment_page"):
            if not self._select_notebook_content_page(self.experiment_page):
                self._log("实验配置页面切换失败：未找到 Notebook tab。")

    def _open_single_point_experiment_page(self, log_selection: bool = False) -> None:
        if self.trajectory_selection.get() != "single_point_optimization":
            return

        if log_selection:
            self._log("已选择轨迹：单点优化实验。已切换到实验配置页面。")
        if hasattr(self, "experiment_page"):
            if not self._select_notebook_content_page(self.experiment_page):
                self._log("实验配置页面切换失败：未找到 Notebook tab。")

    def _log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_queue.put("[{}] {}".format(timestamp, message))

    def _set_tk_var(self, variable, value) -> None:
        if threading.current_thread() is threading.main_thread():
            variable.set(value)
        else:
            self.root.after(0, lambda variable=variable, value=value: variable.set(value))

    def _drain_log_queue(self) -> None:
        while not self.log_queue.empty():
            line = self.log_queue.get_nowait()
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.insert(tk.END, line + "\n")
            self.log_text.see(tk.END)
            self.log_text.configure(state=tk.DISABLED)
        self.root.after(100, self._drain_log_queue)

    def _copy_log_selection(self, _event=None):
        try:
            selected_text = self.log_text.get(tk.SEL_FIRST, tk.SEL_LAST)
        except tk.TclError:
            return "break"

        self.root.clipboard_clear()
        self.root.clipboard_append(selected_text)
        return "break"

    def _select_all_logs(self, _event=None):
        self.log_text.tag_add(tk.SEL, "1.0", tk.END)
        self.log_text.mark_set(tk.INSERT, "1.0")
        self.log_text.see(tk.INSERT)
        return "break"

    def _watch_ros_shutdown(self) -> None:
        if rospy.is_shutdown():
            self._request_application_shutdown("ROS shutdown")
            return
        self.root.after(500, self._watch_ros_shutdown)

    def _refresh_process_status(self) -> None:
        for key in self.status_vars:
            process = self.processes.get(key)
            if process is None:
                continue

            return_code = process.poll()
            if return_code is None:
                self.status_vars[key].set("运行中")
            else:
                self.status_vars[key].set("已退出({})".format(return_code))

        self.root.after(500, self._refresh_process_status)

    def _is_process_running(self, key: str) -> bool:
        process = self.processes.get(key)
        return process is not None and process.poll() is None

    def _find_processes_by_exact_name(self, name: str) -> List[str]:
        try:
            result = subprocess.run(
                ["pgrep", "-a", "-x", name],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as exc:
            self._log("查询进程 {} 失败：{}".format(name, exc))
            return []

        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return lines

    def _find_process_ids_by_exact_name(self, name: str) -> List[int]:
        try:
            result = subprocess.run(
                ["pgrep", "-x", name],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as exc:
            self._log("查询进程 PID {} 失败：{}".format(name, exc))
            return []

        pids: List[int] = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                pids.append(int(line))
            except ValueError:
                continue
        return pids

    def _signal_pids(self, pids: List[int], signum: signal.Signals) -> List[int]:
        remaining_pids: List[int] = []
        for pid in pids:
            try:
                os.kill(pid, signum)
                remaining_pids.append(pid)
            except ProcessLookupError:
                continue
            except Exception as exc:
                self._log("向 PID {} 发送 {} 失败：{}".format(pid, signum.name, exc))
        return remaining_pids

    def _force_stop_processes_by_name(self, process_names: List[str]) -> bool:
        found_any = False
        for process_name in process_names:
            pids = self._find_process_ids_by_exact_name(process_name)
            if not pids:
                continue

            found_any = True
            self._log("检测到残留进程 {}: {}".format(
                process_name,
                ", ".join(str(pid) for pid in pids)))

            for signum, wait_sec in (
                (signal.SIGINT, 2.0),
                (signal.SIGTERM, 2.0),
                (signal.SIGKILL, 1.0),
            ):
                pids = self._signal_pids(pids, signum)
                if not pids:
                    break

                deadline = time.time() + wait_sec
                while time.time() < deadline:
                    still_running = [
                        pid for pid in pids if os.path.exists("/proc/{}".format(pid))
                    ]
                    if not still_running:
                        pids = []
                        break
                    time.sleep(0.1)
                    pids = still_running

            if pids:
                self._log(
                    "残留进程 {} 仍未退出，请手动检查 PID: {}".format(
                        process_name,
                        ", ".join(str(pid) for pid in pids),
                    )
                )
            else:
                self._log("残留进程 {} 已清理。".format(process_name))
        return found_any

    def _cleanup_simulation_leftovers(self) -> bool:
        return self._force_stop_processes_by_name(
            self.simulation_force_shutdown_process_names)

    def _cleanup_force_shutdown_leftovers(self) -> bool:
        return self._force_stop_processes_by_name(
            self.force_shutdown_process_names)

    def _launch_process(self, key: str) -> bool:
        config = self._launch_config_for_key(key)
        if not config.package or not config.launch_file:
            self._log("{} 的 launch 配置不完整。".format(key))
            return False

        if self._is_process_running(key):
            self._log("{} 已经在运行，不重复启动。".format(key))
            return True

        try:
            launch_args = self._prepare_launch_args(key, config.args)
        except ValueError as exc:
            self._log("{} 启动前的参数无效：{}".format(key, exc))
            self._set_tk_var(self.status_vars[key], "参数错误")
            return False
        except Exception as exc:
            self._log("{} 启动前准备运行配置失败：{}".format(key, exc))
            self._set_tk_var(self.status_vars[key], "准备失败")
            return False
        command = ["roslaunch", config.package, config.launch_file] + launch_args
        command_str = " ".join(shlex.quote(part) for part in command)
        launch_env = os.environ.copy()
        popen_kwargs = {"start_new_session": True}

        if self.launch_setup_script and os.path.isfile(self.launch_setup_script):
            shell_command = "source {} && exec {}".format(
                shlex.quote(self.launch_setup_script),
                command_str,
            )
            popen_command = ["/bin/bash", "-lc", shell_command]
            self._log("启动命令：bash -lc {}".format(shell_command))
        else:
            popen_command = command
            self._log("启动命令：{}".format(command_str))

        try:
            self.processes[key] = subprocess.Popen(
                popen_command,
                env=launch_env,
                **popen_kwargs,
            )
        except Exception as exc:
            self._log("{} 启动失败：{}".format(key, exc))
            self._set_tk_var(self.status_vars[key], "启动失败")
            return False

        self._set_tk_var(self.status_vars[key], "启动中")
        return True

    def _start_worker(self, worker) -> None:
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def _start_simulation(self) -> None:
        self._start_worker(self._start_simulation_worker)

    def _start_simulation_worker(self) -> None:
        if self._is_process_running("hardware"):
            self._log("实体机器人链路已经在运行，请不要同时再启动仿真控制程序。")
            return

        existing_gzserver = self._find_processes_by_exact_name("gzserver")
        existing_gzclient = self._find_processes_by_exact_name("gzclient")
        if existing_gzserver or existing_gzclient:
            self._log("检测到系统里已经有 Gazebo 进程在运行，先不再启动第二套仿真。")
            for line in existing_gzserver:
                self._log("已有 gzserver: {}".format(line))
            for line in existing_gzclient:
                self._log("已有 gzclient: {}".format(line))
            self._log("请先关闭旧的 Gazebo，再重新点击“启动仿真控制程序”。")
            return

        if self._launch_process("simulation"):
            self._log(
                "仿真控制程序启动请求已发送，控制器：{}。".format(
                    self._controller_mode_label()))

    def _start_hardware(self) -> None:
        self._start_worker(self._start_hardware_worker)

    def _start_hardware_worker(self) -> None:
        if self._is_process_running("simulation"):
            self._log("仿真控制程序已经在运行，请不要同时再连接实体机器人。")
            return

        if self._launch_process("hardware"):
            self._log(
                "实体机器人连接启动请求已发送，控制器：{}。".format(
                    self._controller_mode_label()))

    def _start_rviz(self) -> None:
        self._start_worker(self._start_rviz_worker)

    def _start_rviz_worker(self) -> None:
        if self._launch_process("rviz"):
            self._log("RViz 启动请求已发送。")

    def _start_vlm(self) -> None:
        self._start_worker(self._start_vlm_worker)

    def _start_vlm_worker(self) -> None:
        if self._vlm_uses_realsense() and not self._start_realsense_for_vlm():
            self._log("RealSense 启动失败，暂不启动 VLM。")
            return
        if self._launch_process("vlm"):
            self._log("VLM 节点启动请求已发送。")

    def _apply_vlm_settings(self) -> None:
        self._start_worker(self._apply_vlm_settings_worker)

    def _apply_vlm_settings_worker(self) -> None:
        try:
            self._write_runtime_vlm_configs()
        except ValueError as exc:
            self._log("VLM 设置无效：{}".format(exc))
            return
        except Exception as exc:
            self._log("写入 VLM 运行配置失败：{}".format(exc))
            return

        self._log("VLM 页面设置已写入运行时配置。")
        self._log("VLM 节点需要重启后读取新设置；规划器侧 VLM 参数会在下一次启动 planner 进程时生效。")

    def _restart_vlm(self) -> None:
        self._start_worker(self._restart_vlm_worker)

    def _restart_vlm_worker(self) -> None:
        if self._vlm_uses_realsense() and not self._start_realsense_for_vlm():
            self._log("RealSense 启动失败，暂不重启 VLM。")
            return
        try:
            self._write_runtime_vlm_config()
        except ValueError as exc:
            self._log("VLM 设置无效：{}".format(exc))
            return
        except Exception as exc:
            self._log("写入 VLM 运行配置失败：{}".format(exc))
            return
        process = self.processes.get("vlm")
        if process is not None and process.poll() is None:
            self._terminate_process("vlm", process)
        self.processes.pop("vlm", None)
        if self._launch_process("vlm"):
            self._log("VLM 节点已按当前页面设置重新启动。")

    def _test_vlm_request(self) -> None:
        self._start_worker(self._test_vlm_request_worker)

    def _test_vlm_request_worker(self) -> None:
        force_refresh = self._vlm_force_refresh_enabled()
        if force_refresh and self._vlm_uses_realsense() and not self._start_realsense_for_vlm():
            self._log("RealSense 启动失败，暂不发送 VLM 测试请求。")
            return
        try:
            self._write_runtime_vlm_config()
        except ValueError as exc:
            self._log("VLM 设置无效：{}".format(exc))
            return
        except Exception as exc:
            self._log("写入 VLM 运行配置失败：{}".format(exc))
            return
        if not self._is_process_running("vlm"):
            self._log("VLM 当前未运行，先尝试启动后再发送测试请求。")
            if not self._launch_process("vlm"):
                return
            time.sleep(1.0)
        self._publish_vlm_node_runtime_params()

        request = MassEstimateRequest()
        request.header.stamp = rospy.Time.now()
        request.header.frame_id = "world"
        request.request_id = "vlm_gui_test_{}".format(int(time.time() * 1000))
        image_path = (
            self._selected_vlm_request_image_path(
                wait_for_realsense=True,
                request_label=request.request_id)
            if force_refresh
            else "")
        request.image_path = image_path
        request.force_refresh = force_refresh
        self.vlm_request_pub.publish(request)
        self._log(
            "已发送 VLM 测试请求：request_id={} image_path={} force_refresh={}".format(
                request.request_id,
                request.image_path if request.image_path else "<empty>",
                "true" if request.force_refresh else "false",
            )
        )

    def _start_pose_marker(self) -> None:
        self._start_worker(self._start_pose_marker_worker)

    def _start_pose_marker_worker(self) -> None:
        if self._launch_process("pose_marker"):
            self._log("姿态球节点已启动或已在运行。")

    def _start_teaching_monitor(self) -> None:
        self._start_worker(self._start_teaching_monitor_worker)

    def _start_teaching_monitor_worker(self) -> None:
        teaching_started = self._launch_process("teaching")
        bridge_started = self._launch_process("franka_state_bridge")
        bridge_topic = self._payload_franka_bridge_topic()

        def apply_bridge_topic() -> None:
            self.teaching_robot_state_topic_var.set(bridge_topic)
            self._subscribe_teaching_robot_state()
            if hasattr(self, "payload_robot_state_topic_var"):
                self.payload_robot_state_topic_var.set(bridge_topic)
                self._subscribe_payload_robot_state()
            self.teaching_status_var.set(
                "示教采样已启动；Franka只读状态订阅：{}".format(bridge_topic))

        self.root.after(0, apply_bridge_topic)

        if teaching_started:
            self._log("示教采样节点启动请求已发送。")
        if bridge_started:
            self._log("Franka只读状态桥已随示教采样一起启动。")

    def _start_home_return(self) -> None:
        self._start_worker(self._start_home_return_worker)

    def _start_home_return_worker(self) -> None:
        if self._launch_process("home_return"):
            self._log("位置回Home节点启动请求已发送。")

    def _start_plot_data_recorder(self) -> None:
        self._start_worker(self._start_plot_data_recorder_worker)

    def _start_plot_data_recorder_worker(self) -> None:
        if self._launch_process("plot_data_recorder"):
            self._log("绘图数据记录节点启动请求已发送。")

    def _call_trigger_service(self, service_name: str, timeout_sec: float = 5.0):
        rospy.wait_for_service(service_name, timeout=timeout_sec)
        proxy = rospy.ServiceProxy(service_name, Trigger)
        return proxy()

    def _plot_data_recorder_namespace(self) -> str:
        service_name = self.plot_data_recorder_start_service
        suffix = "/start"
        if service_name.endswith(suffix):
            return service_name[:-len(suffix)]
        return "/moca_plot_data_recorder"

    def _apply_plot_data_recorder_params(self) -> bool:
        try:
            post_plan_duration = float(self.plot_end_time_var.get().strip())
        except ValueError:
            self.plot_status_var.set("结束时间不是合法数字 / Invalid post-plan time.")
            self._log("绘图记录参数错误：结束时间不是合法数字。")
            return False

        if post_plan_duration < 0.0:
            self.plot_status_var.set("结束时间不能小于0 / Post-plan time must be non-negative.")
            self._log("绘图记录参数错误：结束时间不能小于0。")
            return False

        data_output_dir = self.plot_data_output_dir_var.get().strip()
        fig_output_dir = self.plot_fig_output_dir_var.get().strip()
        if not data_output_dir or not fig_output_dir:
            self.plot_status_var.set("保存路径不能为空 / Output paths cannot be empty.")
            self._log("绘图记录参数错误：保存路径不能为空。")
            return False

        namespace = self._plot_data_recorder_namespace()
        rospy.set_param(
            namespace + "/save_force_capacity",
            bool(self.plot_save_force_capacity_var.get()))
        rospy.set_param(
            namespace + "/save_manipulability",
            bool(self.plot_save_manipulability_var.get()))
        rospy.set_param(
            namespace + "/save_manipulability_ellipsoid",
            bool(self.plot_save_manipulability_ellipsoid_var.get()))
        rospy.set_param(
            namespace + "/save_residual_polytope",
            bool(self.plot_save_residual_polytope_var.get()))
        rospy.set_param(
            namespace + "/auto_stop_on_comparison_complete",
            bool(self.plot_auto_stop_on_comparison_complete_var.get()))
        rospy.set_param(namespace + "/post_plan_duration_sec", post_plan_duration)
        rospy.set_param(namespace + "/data_output_dir", data_output_dir)
        rospy.set_param(namespace + "/fig_output_dir", fig_output_dir)
        return True

    def _start_plot_data_recording(self) -> None:
        self._start_worker(self._start_plot_data_recording_worker)

    def _wait_plot_data_recording(self) -> None:
        self._start_worker(self._wait_plot_data_recording_worker)

    def _wait_plot_data_recording_worker(self) -> None:
        if not self._is_process_running("plot_data_recorder"):
            self._log("绘图数据记录节点未运行，先尝试启动。")
            if not self._launch_process("plot_data_recorder"):
                return
            time.sleep(0.8)

        if not self._apply_plot_data_recorder_params():
            return

        try:
            response = self._call_trigger_service(
                self.plot_data_recorder_wait_start_service,
                timeout_sec=8.0)
        except (rospy.ROSException, rospy.ServiceException) as exc:
            self._log("等待记录启动失败：{}".format(exc))
            self.plot_status_var.set("等待记录启动失败：{}".format(exc))
            return

        if response.success:
            self._log("等待记录已武装：{}".format(response.message))
            self.plot_status_var.set(
                "等待 / Armed. 规划器开始发送目标后自动记录。{}".format(response.message))
        else:
            self._log("等待记录启动被拒绝：{}".format(response.message))
            self.plot_status_var.set("等待记录失败：{}".format(response.message))

    def _start_plot_data_recording_worker(self) -> None:
        if not self._is_process_running("plot_data_recorder"):
            self._log("绘图数据记录节点未运行，先尝试启动。")
            if not self._launch_process("plot_data_recorder"):
                return
            time.sleep(0.8)

        if not self._apply_plot_data_recorder_params():
            return

        try:
            response = self._call_trigger_service(
                self.plot_data_recorder_start_service,
                timeout_sec=8.0)
        except (rospy.ROSException, rospy.ServiceException) as exc:
            self._log("启动绘图记录失败：{}".format(exc))
            self.plot_status_var.set("启动记录失败：{}".format(exc))
            return

        if response.success:
            self._log("绘图记录已开始：{}".format(response.message))
            self.plot_status_var.set(
                "记录中 / Recording. 时间戳：{}".format(response.message))
        else:
            self._log("绘图记录启动被拒绝：{}".format(response.message))
            self.plot_status_var.set("启动记录失败：{}".format(response.message))

    def _stop_plot_data_recording(self) -> None:
        self._start_worker(self._stop_plot_data_recording_worker)

    def _stop_plot_data_recording_worker(self) -> None:
        try:
            response = self._call_trigger_service(
                self.plot_data_recorder_stop_service,
                timeout_sec=120.0)
        except (rospy.ROSException, rospy.ServiceException) as exc:
            self._log("停止并生成绘图文件失败：{}".format(exc))
            self.plot_status_var.set("停止/生成失败：{}".format(exc))
            return

        if response.success:
            self._log("绘图记录已停止，后台开始生成文件：{}".format(response.message))
            self.plot_status_var.set(
                "生成中 / Saving in background. 完成后会自动更新结果。")
        else:
            self._log("绘图记录停止失败：{}".format(response.message))
            self.plot_status_var.set("停止失败：{}".format(response.message))

    def _record_single_plot_point(self) -> None:
        self._start_worker(self._record_single_plot_point_worker)

    def _record_single_plot_point_worker(self) -> None:
        if not self._is_process_running("plot_data_recorder"):
            self._log("绘图数据记录节点未运行，先尝试启动。")
            if not self._launch_process("plot_data_recorder"):
                return
            time.sleep(0.8)

        if not self._apply_plot_data_recorder_params():
            return

        try:
            response = self._call_trigger_service(
                self.plot_data_recorder_record_point_service,
                timeout_sec=8.0)
        except (rospy.ROSException, rospy.ServiceException) as exc:
            self._log("记录单点失败：{}".format(exc))
            self.plot_status_var.set("记录单点失败：{}".format(exc))
            return

        if response.success:
            self._log("已记录单点：{}".format(response.message))
            self.plot_status_var.set("已记录单点 / Point recorded. {}".format(response.message))
        else:
            self._log("记录单点被拒绝：{}".format(response.message))
            self.plot_status_var.set("记录单点失败：{}".format(response.message))

    def _undo_single_plot_point(self) -> None:
        self._start_worker(self._undo_single_plot_point_worker)

    def _undo_single_plot_point_worker(self) -> None:
        try:
            response = self._call_trigger_service(
                self.plot_data_recorder_undo_point_service,
                timeout_sec=8.0)
        except (rospy.ROSException, rospy.ServiceException) as exc:
            self._log("清除上一个单点失败：{}".format(exc))
            self.plot_status_var.set("清除单点失败：{}".format(exc))
            return

        if response.success:
            self._log("已清除上一个单点：{}".format(response.message))
            self.plot_status_var.set("已清除上一个单点 / Last point removed. {}".format(
                response.message))
        else:
            self._log("清除上一个单点失败：{}".format(response.message))
            self.plot_status_var.set("清除单点失败：{}".format(response.message))

    def _finish_single_plot_points(self) -> None:
        self._start_worker(self._finish_single_plot_points_worker)

    def _finish_single_plot_points_worker(self) -> None:
        if not self._apply_plot_data_recorder_params():
            return

        try:
            response = self._call_trigger_service(
                self.plot_data_recorder_finish_points_service,
                timeout_sec=30.0)
        except (rospy.ROSException, rospy.ServiceException) as exc:
            self._log("完成单点出图失败：{}".format(exc))
            self.plot_status_var.set("完成单点失败：{}".format(exc))
            return

        if response.success:
            self._log("单点出图已启动：{}".format(response.message))
            self.plot_status_var.set(
                "单点图片生成中 / Saving point figures in background.")
            self._open_plot_outputs_from_service_message(response.message)
        else:
            self._log("完成单点出图失败：{}".format(response.message))
            self.plot_status_var.set("完成单点失败：{}".format(response.message))

    def _plot_data_outputs_callback(self, msg: String) -> None:
        self.root.after(0, lambda: self._handle_plot_outputs_payload(msg.data))

    def _open_plot_outputs_from_service_message(self, message: str) -> None:
        self.root.after(0, lambda: self._handle_plot_outputs_payload(message))

    def _handle_plot_outputs_payload(self, payload_text: str) -> None:
        try:
            payload = json.loads(payload_text)
        except Exception:
            return

        outputs = payload.get("outputs", payload)
        if not isinstance(outputs, dict):
            return

        manifest_path = str(outputs.get("manifest", ""))
        if manifest_path and manifest_path == self.last_opened_plot_manifest:
            return
        if manifest_path:
            self.last_opened_plot_manifest = manifest_path

        output_candidates = [
            outputs.get("force_capacity_png"),
            outputs.get("manipulability_png"),
            outputs.get("manipulability_ellipsoid_preview_png"),
            outputs.get("residual_polytope_preview_png"),
            outputs.get("single_point_metrics_png"),
        ]
        for key, value in sorted(outputs.items()):
            if key.startswith("single_point_") and key.endswith("_png"):
                output_candidates.append(value)
        saved_paths = []
        for path_text in output_candidates:
            if not path_text:
                continue
            path = str(path_text)
            if not os.path.exists(path):
                self._log("绘图输出文件不存在：{}".format(path))
                continue
            saved_paths.append(path)

        summary = payload.get("summary", "绘图数据已保存。")
        if saved_paths:
            self.plot_status_var.set(
                "{} 已保存 {} 个结果文件。".format(summary, len(saved_paths)))
            self._log("绘图/动图结果已保存：{}".format(", ".join(saved_paths)))
        else:
            self.plot_status_var.set(summary)

    def _metric_merge_vars(self, metric_key: str):
        if metric_key == "force_capacity":
            return (
                self.merge_force_capacity_input_a_var,
                self.merge_force_capacity_input_b_var,
                self.merge_force_capacity_label_a_var,
                self.merge_force_capacity_label_b_var,
                self.merge_force_capacity_output_var,
                "force_capacity",
                "Force Capacity Comparison",
            )
        return (
            self.merge_manipulability_input_a_var,
            self.merge_manipulability_input_b_var,
            self.merge_manipulability_label_a_var,
            self.merge_manipulability_label_b_var,
            self.merge_manipulability_output_var,
            "manipulability",
            "Manipulability Comparison",
        )

    def _merge_metric_csvs(self, metric_key: str) -> None:
        self._start_worker(lambda key=metric_key: self._merge_metric_csvs_worker(key))

    def _merge_metric_csvs_worker(self, metric_key: str) -> None:
        (
            input_a_var,
            input_b_var,
            label_a_var,
            label_b_var,
            output_var,
            metric_name,
            title,
        ) = self._metric_merge_vars(metric_key)
        input_a = self._resolve_workspace_ros_path(input_a_var.get().strip())
        input_b = self._resolve_workspace_ros_path(input_b_var.get().strip())
        output_path = self._resolve_workspace_ros_path(output_var.get().strip())
        label_a = label_a_var.get().strip() or "Run A"
        label_b = label_b_var.get().strip() or "Run B"
        if not input_a.is_absolute():
            input_a = self.workspace_root / input_a
        if not input_b.is_absolute():
            input_b = self.workspace_root / input_b
        if not output_path.is_absolute():
            output_path = self.workspace_root / output_path

        if not input_a.exists():
            self.metric_merge_status_var.set("数据1不存在 / Input 1 missing: {}".format(input_a))
            return
        if not input_b.exists():
            self.metric_merge_status_var.set("数据2不存在 / Input 2 missing: {}".format(input_b))
            return
        if not output_path.suffix:
            output_path = output_path / "{}_comparison.png".format(metric_name)
        if output_path.suffix.lower() != ".png":
            output_path = output_path.with_suffix(".png")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        script_path = self.workspace_root / "src/polytope_ros/scripts/compare_metric_csvs.py"
        command = [
            "python3",
            str(script_path),
            "--input-a",
            str(input_a),
            "--input-b",
            str(input_b),
            "--output-png",
            str(output_path),
            "--metric",
            metric_name,
            "--label-a",
            label_a,
            "--label-b",
            label_b,
            "--title",
            title,
        ]

        self.metric_merge_status_var.set("正在合并 {} ...".format(metric_name))
        try:
            result = subprocess.run(
                command,
                cwd=str(self.workspace_root),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True)
        except subprocess.CalledProcessError as exc:
            message = (exc.stderr or exc.stdout or str(exc)).strip()
            self.metric_merge_status_var.set("合并失败：{}".format(message))
            self._log("{} 合并失败：{}".format(metric_name, message))
            return

        merged_csv = output_path.with_name(output_path.stem + "_merged.csv")
        self.metric_merge_status_var.set(
            "已生成图片：{}\n已生成合并CSV：{}".format(output_path, merged_csv))
        self._log("{} 合并完成：{} {}".format(metric_name, output_path, result.stdout.strip()))

    def _teaching_points_output_path(self) -> Path:
        path_text = self.teaching_points_output_path_var.get().strip()
        if not path_text:
            path_text = "output/polytope_ros/teaching_points/teaching_points.json"
            self.teaching_points_output_path_var.set(path_text)

        path = self._resolve_workspace_ros_path(path_text)
        if not path.is_absolute():
            path = self.workspace_root / path
        if not path.suffix:
            path = path.with_suffix(".json")
        return path

    def _refresh_teaching_records_text(self) -> None:
        if not self.teaching_recorded_points:
            self._set_text_widget("teaching_records_text", "")
            return

        formatted_blocks = []
        for index, point in enumerate(self.teaching_recorded_points, start=1):
            sample = point.get("sample", point)
            formatted_blocks.append(
                "point_index: {}\n{}".format(
                    index,
                    self._format_teaching_sample(sample)))
        self._set_text_widget("teaching_records_text", "\n\n".join(formatted_blocks))

    def _record_current_teaching_point(self) -> None:
        if self.latest_teaching_sample is None:
            self._log(
                "没有当前 Franka robot_state/示教采样，无法记录当前点；请先启动Franka只读状态桥。")
            self.teaching_status_var.set(
                "没有当前 Franka robot_state，无法记录当前点。")
            return

        sample = deepcopy(self.latest_teaching_sample)
        point_index = len(self.teaching_recorded_points) + 1
        point = {
            "point_index": point_index,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "stamp": sample.get("stamp", rospy.Time.now().to_sec()),
            "arm_joint_positions": sample.get("arm_joint_positions", []),
            "ee_pose": sample.get("ee_pose", {}),
            "odom": sample.get("odom", {}),
            "vector": sample.get("vector", []),
            "sample": sample,
        }
        self.teaching_recorded_points.append(point)
        self.latest_recorded_teaching_sample = sample
        self.latest_recorded_teaching_sample_json = json.dumps(sample, ensure_ascii=False)
        self._refresh_teaching_records_text()
        self.teaching_status_var.set(
            "已记录当前点 {} / Recorded point {}.".format(point_index, point_index))
        self._log("已记录当前示教点 {}。".format(point_index))

    def _remove_last_teaching_point(self) -> None:
        if not self.teaching_recorded_points:
            self._log("没有记录点")
            self.teaching_status_var.set("没有记录点")
            return

        removed = self.teaching_recorded_points.pop()
        self._refresh_teaching_records_text()
        if self.teaching_recorded_points:
            last_sample = self.teaching_recorded_points[-1].get("sample", {})
            self.latest_recorded_teaching_sample = last_sample
            self.latest_recorded_teaching_sample_json = json.dumps(
                last_sample,
                ensure_ascii=False)
        else:
            self.latest_recorded_teaching_sample = None
            self.latest_recorded_teaching_sample_json = ""
        self.teaching_status_var.set(
            "已清除上一点 {} / Removed last point.".format(
                removed.get("point_index", "-")))
        self._log("已清除上一点。")

    def _clear_all_teaching_points(self) -> None:
        if not self.teaching_recorded_points:
            self._log("没有记录点")
            self.teaching_status_var.set("没有记录点")
            return

        self.teaching_recorded_points.clear()
        self.latest_recorded_teaching_sample = None
        self.latest_recorded_teaching_sample_json = ""
        self._refresh_teaching_records_text()
        self.teaching_status_var.set("已清除所有记录点。")
        self._log("已清除所有示教记录点。")

    def _save_teaching_points(self) -> None:
        if not self.teaching_recorded_points:
            self._log("没有记录点")
            self.teaching_status_var.set("没有记录点")
            return

        try:
            output_path = self._teaching_points_output_path()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "point_count": len(self.teaching_recorded_points),
                "points": self.teaching_recorded_points,
            }
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
        except Exception as exc:
            self.teaching_status_var.set("保存记录失败：{}".format(exc))
            self._log("保存示教记录失败：{}".format(exc))
            return

        self.teaching_status_var.set("已保存 {} 个点：{}".format(
            len(self.teaching_recorded_points),
            output_path))
        self._log("示教记录已保存：{}".format(output_path))

    def _record_teaching_sample(self) -> None:
        self._start_worker(self._record_teaching_sample_worker)

    def _record_teaching_sample_worker(self) -> None:
        if not self._is_process_running("teaching"):
            self._log("示教采样节点未运行，先尝试启动。")
            if not self._launch_process("teaching"):
                return
            time.sleep(0.8)

        try:
            response = self._call_trigger_service(self.teaching_record_service)
        except (rospy.ROSException, rospy.ServiceException) as exc:
            self._log("示教记录服务调用失败：{}".format(exc))
            self.teaching_status_var.set("示教记录失败：{}".format(exc))
            return

        if not response.success:
            self._log("示教记录失败：{}".format(response.message))
            self.teaching_status_var.set("示教记录失败：{}".format(response.message))
            return

        try:
            sample = json.loads(response.message)
            formatted = self._format_teaching_sample(sample)
        except Exception as exc:
            self._log("示教记录已返回，但解析失败：{}".format(exc))
            self.teaching_status_var.set("示教记录解析失败：{}".format(exc))
            return

        def apply_record() -> None:
            self.latest_recorded_teaching_sample_json = response.message
            self.latest_recorded_teaching_sample = sample
            self._append_text_widget("teaching_records_text", formatted)
            self.teaching_status_var.set(
                "已记录 sample {} / Recorded sample {}.".format(
                    sample.get("record_index", "-"),
                    sample.get("record_index", "-")))

        self.root.after(0, apply_record)
        self._log("已记录当前示教样本。")

    def _set_latest_record_as_home(self) -> None:
        self._start_worker(self._set_latest_record_as_home_worker)

    def _set_latest_record_as_home_worker(self) -> None:
        if not self.latest_recorded_teaching_sample_json:
            self._log("还没有示教记录，不能设置 Home。")
            self.teaching_status_var.set("还没有示教记录，不能设置 Home。")
            return

        if not self._is_process_running("home_return"):
            self._log("位置回Home节点未运行，先尝试启动。")
            if not self._launch_process("home_return"):
                return
            time.sleep(0.8)

        self.home_sample_pub.publish(String(data=self.latest_recorded_teaching_sample_json))
        self._log("已把最后一次示教记录发布为 Home 目标。")
        self.teaching_status_var.set("最后一次示教记录已设置为 Home。")

    def _set_current_state_as_home(self) -> None:
        self._start_worker(self._set_current_state_as_home_worker)

    def _set_current_state_as_home_worker(self) -> None:
        if not self._is_process_running("home_return"):
            self._log("位置回Home节点未运行，先尝试启动。")
            if not self._launch_process("home_return"):
                return
            time.sleep(0.8)

        try:
            response = self._call_trigger_service(self.home_return_set_current_service)
        except (rospy.ROSException, rospy.ServiceException) as exc:
            self._log("设置当前状态为 Home 失败：{}".format(exc))
            return

        if response.success:
            self._log("当前状态已设置为 Home：{}".format(response.message))
            self.teaching_status_var.set("当前状态已设置为 Home。")
        else:
            self._log("设置当前状态为 Home 被拒绝：{}".format(response.message))
            self.teaching_status_var.set("设置当前状态为 Home 失败：{}".format(response.message))

    def _start_position_home_return(self) -> None:
        self._start_worker(self._start_position_home_return_worker)

    def _start_position_home_return_worker(self) -> None:
        if not self._is_process_running("home_return"):
            self._log("位置回Home节点未运行，先尝试启动。")
            if not self._launch_process("home_return"):
                return
            time.sleep(0.8)

        try:
            response = self._call_trigger_service(self.home_return_start_service)
        except (rospy.ROSException, rospy.ServiceException) as exc:
            self._log("位置回Home启动失败：{}".format(exc))
            return

        if response.success:
            self._log("位置回Home已启动：{}".format(response.message))
        else:
            self._log("位置回Home被拒绝：{}".format(response.message))

    def _stop_position_home_return(self) -> None:
        self._start_worker(self._stop_position_home_return_worker)

    def _stop_position_home_return_worker(self) -> None:
        try:
            response = self._call_trigger_service(self.home_return_stop_service, timeout_sec=2.0)
        except (rospy.ROSException, rospy.ServiceException) as exc:
            self._log("停止位置回Home失败：{}".format(exc))
            return

        if response.success:
            self._log("位置回Home已停止。")
        else:
            self._log("停止位置回Home被拒绝：{}".format(response.message))

    def _selected_planning_request(self):
        if self.trajectory_selection.get() == "comparison_two_lifts":
            return "payload_lift", True
        if self.trajectory_selection.get() == "manual_target_pose":
            return "manual_target_pose", False
        if self.trajectory_selection.get() == "move_experiment":
            return "teaching_waypoints", False
        if self.trajectory_selection.get() == "single_point_optimization":
            return "single_point_optimization", False
        return "teaching_waypoints", False

    def _should_request_vlm_for_trajectory(self, trajectory_mode: str) -> bool:
        return trajectory_mode not in ("return_to_home", "teaching_return_initial")

    def _prepare_vlm_for_planning_request(self, log_context: str) -> None:
        force_refresh = self._vlm_force_refresh_enabled()
        if force_refresh and self._vlm_uses_realsense():
            if not self._start_realsense_for_vlm():
                self._log("RealSense 启动失败，本次{}仍将请求 VLM，失败时规划器会使用默认质量。".format(log_context))
            vlm_image_path = ""
        else:
            vlm_image_path = self._selected_vlm_request_image_path() if force_refresh else ""

        self._publish_planner_vlm_runtime_params(vlm_image_path)

        if not self._is_process_running("vlm"):
            self._log("VLM 当前未运行，先启动 VLM 节点再发送{}规划请求。".format(log_context))
            try:
                self._write_runtime_vlm_config()
            except ValueError as exc:
                self._log("VLM 设置无效，本次规划器会在 VLM 失败时使用默认质量：{}".format(exc))
            except Exception as exc:
                self._log("写入 VLM 运行配置失败，本次规划器会在 VLM 失败时使用默认质量：{}".format(exc))
            if self._launch_process("vlm"):
                time.sleep(1.0)

    def _start_planner(self) -> None:
        self._start_worker(self._start_planner_worker)

    def _start_planner_worker(self) -> None:
        if self.trajectory_selection.get() == "move_experiment":
            self._start_move_next_waypoint_worker()
            return

        trajectory_mode, comparison_force_sweep_enabled = self._selected_planning_request()
        request_vlm_mass_estimate = self._should_request_vlm_for_trajectory(trajectory_mode)
        if request_vlm_mass_estimate:
            self._prepare_vlm_for_planning_request("规划")

        if trajectory_mode == "manual_target_pose" and not self._launch_process("pose_marker"):
            self._log("姿态球节点未能启动，暂不发送手动目标规划请求。")
            return

        selected_single_point = None
        if trajectory_mode == "single_point_optimization":
            try:
                selected_single_point = self._ensure_selected_single_point()
                self._publish_single_point_runtime_params(selected_single_point)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self._log("单点优化参数错误：{}".format(exc))
                return

        if trajectory_mode.startswith("teaching_"):
            try:
                self._publish_experiment_runtime_params()
            except ValueError as exc:
                self._log("实验配置参数错误：{}".format(exc))
                return

        self._request_planner_start(
            trajectory_mode,
            comparison_force_sweep_enabled,
            request_vlm_mass_estimate,
            before_service_call=(
                self._publish_experiment_runtime_params
                if trajectory_mode.startswith("teaching_")
                else lambda: self._publish_single_point_runtime_params(selected_single_point)
                if trajectory_mode == "single_point_optimization"
                else None),
        )

    def _request_planner_start(
        self,
        trajectory_mode: str,
        comparison_force_sweep_enabled: bool,
        request_vlm_mass_estimate: bool,
        before_service_call: Optional[Callable[[], None]] = None,
        algorithm_override: Optional[str] = None,
    ) -> bool:
        if not self._launch_process("planner"):
            return False

        self._log("等待规划器服务 {} 就绪...".format(self.planner_start_service))
        try:
            rospy.wait_for_service(
                self.planner_start_service,
                timeout=self.planner_service_timeout_sec)
        except rospy.ROSException as exc:
            self._log("规划器服务未就绪：{}".format(exc))
            return False

        if before_service_call is not None:
            try:
                before_service_call()
            except Exception as exc:
                self._log("规划器运行参数下发失败：{}".format(exc))
                return False
        try:
            self._publish_planner_algorithm_runtime_param(algorithm_override)
        except Exception as exc:
            self._log("规划器算法参数下发失败：{}".format(exc))
            return False

        try:
            start_planning = rospy.ServiceProxy(
                self.planner_start_service,
                StartPlanning)
            response = start_planning(
                trajectory_mode,
                comparison_force_sweep_enabled,
                request_vlm_mass_estimate,
            )
        except rospy.ServiceException as exc:
            self._log("规划器 service 调用失败：{}".format(exc))
            return False

        if response.success:
            self._log("规划器已接收任务：{}".format(response.message))
            return True
        else:
            self._log("规划器拒绝任务：{}".format(response.message))
            return False

    def _reset_to_home(self) -> None:
        self._start_worker(self._reset_to_home_worker)

    def _reset_to_home_for_experiment(self, experiment_kind: str) -> None:
        self._start_worker(lambda: self._reset_to_home_worker(experiment_kind))

    def _reset_to_home_worker(self, experiment_kind: Optional[str] = None) -> None:
        self._log("发送 return-to-home 规划请求。")
        try:
            self._publish_experiment_runtime_params(experiment_kind)
        except ValueError as exc:
            self._log("实验配置参数错误：{}".format(exc))
            return
        if self._request_planner_start(
            "return_to_home",
            False,
            False,
            before_service_call=lambda: self._publish_experiment_runtime_params(
                experiment_kind)):
            if experiment_kind == "move" or (
                experiment_kind is None and self.trajectory_selection.get() == "move_experiment"):
                self._reset_move_next_waypoint_index("回 Home")
        else:
            self._log(
                "return-to-home 规划请求失败；旧的 /reset_to_home topic 回退逻辑未执行。")

    def _return_to_lift_initial_point(self) -> None:
        self._start_worker(self._return_to_lift_initial_point_worker)

    def _return_to_lift_initial_point_worker(self) -> None:
        try:
            self._publish_experiment_runtime_params("lift")
        except ValueError as exc:
            self._log("实验配置参数错误：{}".format(exc))
            return
        self._log("发送举升实验返回初始点规划请求。")
        self._request_planner_start(
            "teaching_return_initial",
            False,
            False,
            before_service_call=lambda: self._publish_experiment_runtime_params("lift"))

    def _start_lift_experiment_motion(self) -> None:
        self._start_worker(self._start_lift_experiment_motion_worker)

    def _start_lift_experiment_motion_worker(self) -> None:
        try:
            self._publish_experiment_runtime_params("lift")
        except ValueError as exc:
            self._log("实验配置参数错误：{}".format(exc))
            return
        request_vlm_mass_estimate = self._should_request_vlm_for_trajectory(
            "teaching_waypoints")
        if request_vlm_mass_estimate:
            self._prepare_vlm_for_planning_request("举升实验")
        self._log("发送举升实验示教轨迹规划请求。")
        self._request_planner_start(
            "teaching_waypoints",
            False,
            request_vlm_mass_estimate,
            before_service_call=lambda: self._publish_experiment_runtime_params("lift"))

    def _start_single_point_experiment_motion(self) -> None:
        self._start_worker(self._start_single_point_experiment_motion_worker)

    def _start_single_point_experiment_motion_worker(self) -> None:
        try:
            selected_point = self._ensure_selected_single_point()
            self._publish_single_point_runtime_params(selected_point)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self._log("单点优化参数错误：{}".format(exc))
            return

        request_vlm_mass_estimate = self._should_request_vlm_for_trajectory(
            "single_point_optimization")
        if request_vlm_mass_estimate:
            self._prepare_vlm_for_planning_request("单点优化")

        self._log("发送单点优化规划请求。")
        self._request_planner_start(
            "single_point_optimization",
            False,
            request_vlm_mass_estimate,
            before_service_call=lambda: self._publish_single_point_runtime_params(selected_point))

    def _load_single_point_teaching_points(self):
        path_text = self.single_point_waypoints_path_var.get().strip()
        if not path_text:
            raise ValueError("单点文件不能为空。")

        resolved_path = self._resolve_workspace_ros_path(path_text)
        with open(resolved_path, "r", encoding="utf-8") as handle:
            document = json.load(handle)

        points = document.get("points") if isinstance(document, dict) else None
        if not isinstance(points, list) or not points:
            raise ValueError("单点文件没有非空 points 数组：{}".format(resolved_path))
        return document, points, resolved_path

    def _apply_single_point_to_gui(self, point: Dict[str, Any], point_index: int, point_count: int) -> None:
        pose = point.get("ee_pose", {}).get("pose", {})
        position = pose.get("position", {})
        orientation = pose.get("orientation", {})
        if not isinstance(position, dict) or not isinstance(orientation, dict):
            raise ValueError("选中的单点没有合法 ee_pose.pose 字段。")

        self._set_tk_var(self.single_point_x_var, str(float(position.get("x", 0.0))))
        self._set_tk_var(self.single_point_y_var, str(float(position.get("y", 0.0))))
        self._set_tk_var(self.single_point_z_var, str(float(position.get("z", 0.0))))
        self._set_tk_var(self.single_point_qx_var, str(float(orientation.get("x", 0.0))))
        self._set_tk_var(self.single_point_qy_var, str(float(orientation.get("y", 0.0))))
        self._set_tk_var(self.single_point_qz_var, str(float(orientation.get("z", 0.0))))
        self._set_tk_var(self.single_point_qw_var, str(float(orientation.get("w", 1.0))))
        self.single_point_selected_point = deepcopy(point)
        self._set_tk_var(
            self.single_point_status_var,
            "当前选择第 {}/{} 个点 / Selected point {}/{}.".format(
                point_index + 1,
                point_count,
                point_index + 1,
                point_count))

    def _select_single_point_by_index(self, point_index: int, advance_after_select: bool) -> None:
        document, points, _source_path = self._load_single_point_teaching_points()
        del document
        point_count = len(points)
        selected_index = point_index % point_count
        self._apply_single_point_to_gui(points[selected_index], selected_index, point_count)
        self.single_point_selected_index = (
            (selected_index + 1) % point_count
            if advance_after_select
            else selected_index)
        self._log("已选择单点实验第 {}/{} 个点，只更新目标，不启动规划。".format(
            selected_index + 1,
            point_count))

    def _ensure_selected_single_point(self) -> Dict[str, Any]:
        if self.single_point_selected_point is None:
            self._select_single_point_by_index(self.single_point_selected_index, False)
        if self.single_point_selected_point is None:
            raise ValueError("没有选中的单点。")
        return deepcopy(self.single_point_selected_point)

    def _select_next_single_point(self) -> None:
        self._start_worker(self._select_next_single_point_worker)

    def _select_next_single_point_worker(self) -> None:
        try:
            self._select_single_point_by_index(
                self.single_point_selected_index,
                advance_after_select=True)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self._log("选择单点失败：{}".format(exc))
            self._set_tk_var(self.single_point_status_var, "选择单点失败：{}".format(exc))

    def _write_single_point_selected_waypoint_file(
        self,
        source_document: Dict[str, Any],
        point: Dict[str, Any],
        source_path: Path,
    ) -> str:
        output_document = deepcopy(source_document)
        output_document["saved_at"] = datetime.now(timezone.utc).isoformat()
        output_document["point_count"] = 1
        output_document["source_waypoints_path"] = str(source_path)
        output_document["points"] = [deepcopy(point)]

        self.single_point_single_waypoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.single_point_single_waypoint_path, "w", encoding="utf-8") as handle:
            json.dump(output_document, handle, indent=2, ensure_ascii=False)
        return str(self.single_point_single_waypoint_path)

    def _move_to_selected_single_point(self) -> None:
        self._start_worker(self._move_to_selected_single_point_worker)

    def _move_to_selected_single_point_worker(self) -> None:
        try:
            selected_point = self._ensure_selected_single_point()
            document, _points, source_path = self._load_single_point_teaching_points()
            single_waypoint_path = self._write_single_point_selected_waypoint_file(
                document,
                selected_point,
                source_path)
            self._publish_experiment_runtime_params(
                "point",
                waypoints_path_override=single_waypoint_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self._log("单点直达参数错误：{}".format(exc))
            return

        request_vlm_mass_estimate = False

        self._log("发送单点直达请求：五次多项式插值到当前选中点，不调用优化器。")
        self._request_planner_start(
            "teaching_waypoints",
            False,
            request_vlm_mass_estimate,
            before_service_call=lambda: self._publish_experiment_runtime_params(
                "point",
                waypoints_path_override=single_waypoint_path),
            algorithm_override="joint_quintic_interpolation")

    def _load_move_teaching_points(self):
        path_text = self.move_waypoints_path_var.get().strip()
        if not path_text:
            raise ValueError("移动实验示教轨迹文件不能为空。")

        resolved_path = self._resolve_workspace_ros_path(path_text)
        with open(resolved_path, "r", encoding="utf-8") as handle:
            document = json.load(handle)

        points = document.get("points") if isinstance(document, dict) else None
        if not isinstance(points, list) or not points:
            raise ValueError("移动实验示教轨迹文件没有非空 points 数组：{}".format(resolved_path))
        return document, points, resolved_path

    def _write_move_single_waypoint_file(self, source_document: Dict[str, Any], point: Dict[str, Any],
                                         point_index: int, source_path: Path) -> str:
        output_document = deepcopy(source_document)
        output_document["saved_at"] = datetime.now(timezone.utc).isoformat()
        output_document["point_count"] = 1
        output_document["source_waypoints_path"] = str(source_path)
        output_document["source_waypoint_index"] = int(point_index + 1)
        output_document["points"] = [deepcopy(point)]

        self.move_single_waypoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.move_single_waypoint_path, "w", encoding="utf-8") as handle:
            json.dump(output_document, handle, indent=2, ensure_ascii=False)
        return str(self.move_single_waypoint_path)

    def _reset_move_next_waypoint_index(self, reason: str = "") -> None:
        self.move_next_waypoint_index = 0
        if reason:
            self._log("移动实验 waypoint 已因{}清零；下一次将执行第 1 个 waypoint。".format(reason))

    def _return_to_move_initial_point(self) -> None:
        self._start_worker(self._return_to_move_initial_point_worker)

    def _return_to_move_initial_point_worker(self) -> None:
        try:
            self._publish_experiment_runtime_params("move")
        except ValueError as exc:
            self._log("实验配置参数错误：{}".format(exc))
            return
        self._log("发送移动实验返回初始点规划请求。")
        if self._request_planner_start(
            "teaching_return_initial",
            False,
            False,
            before_service_call=lambda: self._publish_experiment_runtime_params("move")):
            self._reset_move_next_waypoint_index("返回初始点")

    def _start_move_next_waypoint(self) -> None:
        with self.move_planning_lock:
            if self.move_planning_active:
                self._log("移动实验规划仍在进行，忽略重复点击“下一轨迹点”。")
                return
            self.move_planning_active = True
        self._start_worker(self._start_move_next_waypoint_worker)

    def _start_move_next_waypoint_worker(self) -> None:
        try:
            request_vlm_mass_estimate = self._should_request_vlm_for_trajectory(
                "teaching_waypoints")
            try:
                document, points, source_path = self._load_move_teaching_points()
                point_count = len(points)
                point_index = self.move_next_waypoint_index % point_count
                single_waypoint_path = self._write_move_single_waypoint_file(
                    document,
                    points[point_index],
                    point_index,
                    source_path)
                self._publish_experiment_runtime_params(
                    "move",
                    waypoints_path_override=single_waypoint_path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self._log("移动实验参数错误：{}".format(exc))
                return

            if request_vlm_mass_estimate:
                self._prepare_vlm_for_planning_request("移动实验")

            self._log(
                "发送移动实验第 {}/{} 个 waypoint 规划请求。".format(
                    point_index + 1,
                    point_count))
            if self._request_planner_start(
                "teaching_waypoints",
                False,
                request_vlm_mass_estimate,
                before_service_call=lambda: self._publish_experiment_runtime_params(
                    "move",
                    waypoints_path_override=single_waypoint_path)):
                self.move_next_waypoint_index = (point_index + 1) % point_count
                self._log("移动实验下一次将执行第 {} 个 waypoint。".format(
                    self.move_next_waypoint_index + 1))
        finally:
            with self.move_planning_lock:
                self.move_planning_active = False

    def _stop_process(self, key: str) -> None:
        self._start_worker(lambda key=key: self._stop_process_worker(key))

    def _stop_process_worker(self, key: str) -> None:
        process = self.processes.get(key)
        label = self.process_labels.get(key, key)
        if process is None:
            self.status_vars[key].set("未启动")
            self._log("{} 当前没有由上位机启动的进程。".format(label))
            return

        if process.poll() is not None:
            self.status_vars[key].set("已退出({})".format(process.returncode))
            self.processes.pop(key, None)
            self._log("{} 已经退出。".format(label))
            return

        self._terminate_process(key, process)
        if process.poll() is not None:
            self.processes.pop(key, None)

    def _terminate_process(self, key: str, process: subprocess.Popen) -> None:
        if process.poll() is not None:
            self.status_vars[key].set("已退出({})".format(process.returncode))
            return

        label = self.process_labels.get(key, key)
        try:
            process_group = os.getpgid(process.pid)
        except ProcessLookupError:
            self.status_vars[key].set("已退出")
            return

        self._log("正在关闭{}...".format(label))
        try:
            os.killpg(process_group, signal.SIGINT)
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self._log("{} 对 SIGINT 没有及时退出，继续发送 SIGTERM。".format(label))
            try:
                os.killpg(process_group, signal.SIGTERM)
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self._log("{} 仍未退出，继续发送 SIGKILL。".format(label))
                os.killpg(process_group, signal.SIGKILL)
                process.wait(timeout=2.0)
        except ProcessLookupError:
            pass

        return_code = process.poll()
        if return_code is None:
            self.status_vars[key].set("关闭中")
        else:
            self.status_vars[key].set("已退出({})".format(return_code))
            self._log("{} 已结束。".format(label))

    def _shutdown_all_processes(self) -> None:
        self._start_worker(self._shutdown_all_processes_worker)

    def _shutdown_all_processes_worker(self) -> None:
        running_items = [
            (key, process)
            for key, process in self.processes.items()
            if process.poll() is None
        ]
        if not running_items:
            if self._cleanup_force_shutdown_leftovers():
                self._log("已额外清理残留控制/仿真进程。")
            else:
                self._log("当前没有由上位机启动的进程需要关闭。")
            return

        for key, process in running_items:
            self._terminate_process(key, process)

        if self._cleanup_force_shutdown_leftovers():
            self._log("已额外清理残留控制/仿真进程。")

        self._log("全部受控进程关闭流程已完成。")

    def _handle_signal(self, signum, _frame) -> None:
        signal_name = signal.Signals(signum).name
        try:
            self.root.after(0, lambda: self._request_application_shutdown(signal_name))
        except tk.TclError:
            raise SystemExit(0)

    def _request_application_shutdown(self, reason: str) -> None:
        if self.shutdown_requested:
            return

        self.shutdown_requested = True
        self._log("收到退出请求（{}），准备关闭上位机和受控进程。".format(reason))
        self._shutdown_all_processes_worker()
        rospy.signal_shutdown(reason)
        self.root.after(100, self._finalize_gui_shutdown)

    def _finalize_gui_shutdown(self) -> None:
        try:
            self.root.quit()
            self.root.destroy()
        except tk.TclError:
            pass

    def _on_close(self) -> None:
        self._request_application_shutdown("window close")

    def run(self) -> None:
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self._request_application_shutdown("KeyboardInterrupt")


if __name__ == "__main__":
    panel = OperatorPanel()
    panel.run()
