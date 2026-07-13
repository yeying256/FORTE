#!/usr/bin/env python3

import csv
import itertools
import json
import math
import os
import threading
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rospy
import rospkg
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from moca_trajectory_generator.msg import TargetPoseCommand
from moca_vlm.msg import MassEstimateResult
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray, Int32, String
from std_srvs.srv import Trigger, TriggerResponse


class ExperimentDataRecorder:
    def __init__(self):
        self.lock = threading.Lock()
        self.recording = False
        self.waiting_for_command = False
        self.timestamp = ""
        self.start_ros_time = 0.0
        self.start_wall_time = 0.0
        self.stop_timer = None
        self.config = {}
        self.buffers = self._empty_buffers()
        self.latest_samples = {}
        self.point_samples = []
        self.latest_target_signature = None
        self.latest_target_wall_time = 0.0
        self.wait_baseline_signature = None
        self.wait_last_signature = None
        self.wait_last_change_wall_time = 0.0

        self.workspace_root = self._resolve_workspace_root()

        self.manipulability_topic = rospy.get_param(
            "~manipulability_topic", "/wb_cart_imp_controller/manipulability")
        self.manipulability_ellipsoid_topic = rospy.get_param(
            "~manipulability_ellipsoid_topic",
            "/wb_cart_imp_controller/manipulability_ellipsoid")
        self.force_capacity_topic = rospy.get_param(
            "~directional_force_capacity_topic",
            "/wb_cart_imp_controller/directional_force_capacity")
        self.desired_force_topic = rospy.get_param(
            "~desired_force_topic", "/wb_cart_imp_controller/desired_force")
        self.force_polytope_vertices_topic = rospy.get_param(
            "~force_polytope_vertices_topic",
            "/wb_cart_imp_controller/force_polytope_vertices")
        self.pose_topic = rospy.get_param("~pose_topic", "/wb_cart_imp_controller/moca_pose")
        self.pose_target_topic = rospy.get_param(
            "~pose_target_topic", "/wb_cart_imp_controller/moca_pose_target")
        self.state_topic = rospy.get_param(
            "~state_topic", "/wb_cart_imp_controller/moca_state")
        self.state_target_topic = rospy.get_param(
            "~state_target_topic", "/wb_cart_imp_controller/moca_state_target")
        self.target_pose_topic = rospy.get_param("~target_pose_topic", "/target_pose")
        self.comparison_run_state_topic = rospy.get_param(
            "~comparison_run_state_topic", "/comparison_run_state")
        self.vlm_mass_result_topic = rospy.get_param(
            "~vlm_mass_result_topic", "/vlm_mass_result")

        rospy.Subscriber(
            self.manipulability_topic, Float64, self._float_callback,
            callback_args="manipulability", queue_size=500)
        rospy.Subscriber(
            self.manipulability_ellipsoid_topic,
            Float64MultiArray,
            self._manipulability_ellipsoid_callback,
            queue_size=200)
        rospy.Subscriber(
            self.force_capacity_topic, Float64, self._float_callback,
            callback_args="force_capacity", queue_size=500)
        rospy.Subscriber(
            self.desired_force_topic, Vector3Stamped, self._desired_force_callback,
            queue_size=500)
        rospy.Subscriber(
            self.force_polytope_vertices_topic,
            Float64MultiArray,
            self._force_polytope_vertices_callback,
            queue_size=100)
        rospy.Subscriber(
            self.pose_topic, PoseStamped, self._pose_callback,
            callback_args="pose", queue_size=200)
        rospy.Subscriber(
            self.pose_target_topic, PoseStamped, self._pose_callback,
            callback_args="pose_target", queue_size=200)
        rospy.Subscriber(
            self.state_topic, JointState, self._joint_state_callback,
            callback_args="state", queue_size=200)
        rospy.Subscriber(
            self.state_target_topic, JointState, self._joint_state_callback,
            callback_args="state_target", queue_size=200)
        rospy.Subscriber(
            self.target_pose_topic, TargetPoseCommand, self._target_pose_callback,
            queue_size=500)
        rospy.Subscriber(
            self.comparison_run_state_topic, Int32, self._comparison_run_state_callback,
            queue_size=20)
        rospy.Subscriber(
            self.vlm_mass_result_topic, MassEstimateResult, self._vlm_mass_result_callback,
            queue_size=20)

        self.start_service = rospy.Service("~start", Trigger, self._start_service)
        self.wait_start_service = rospy.Service(
            "~wait_start", Trigger, self._wait_start_service)
        self.stop_service = rospy.Service("~stop", Trigger, self._stop_service)
        self.record_point_service = rospy.Service(
            "~record_point", Trigger, self._record_point_service)
        self.undo_point_service = rospy.Service(
            "~undo_point", Trigger, self._undo_point_service)
        self.finish_points_service = rospy.Service(
            "~finish_points", Trigger, self._finish_points_service)
        self.latest_outputs_pub = rospy.Publisher(
            "~latest_outputs", String, queue_size=1)
        self.wait_monitor_timer = rospy.Timer(
            rospy.Duration(0.2), self._wait_monitor_timer_callback)

        rospy.loginfo(
            "experiment_data_recorder: ready. services: %s/start, %s/wait_start, %s/stop, %s/record_point, %s/undo_point, %s/finish_points",
            rospy.get_name(),
            rospy.get_name(),
            rospy.get_name(),
            rospy.get_name(),
            rospy.get_name(),
            rospy.get_name())

    @staticmethod
    def _empty_buffers():
        return {
            "manipulability": [],
            "manipulability_ellipsoid": [],
            "force_capacity": [],
            "desired_force": [],
            "force_polytope_vertices": [],
            "pose": [],
            "pose_target": [],
            "state": [],
            "state_target": [],
            "target_pose": [],
            "comparison_run_state": [],
            "vlm_mass_result": [],
        }

    @staticmethod
    def _stamp_to_sec(stamp):
        if stamp is None or stamp == rospy.Time():
            return rospy.Time.now().to_sec()
        return stamp.to_sec()

    def _resolve_workspace_root(self):
        package_path = Path(rospkg.RosPack().get_path("polytope_ros")).resolve()
        return package_path.parents[1]

    def _resolve_output_dir(self, value):
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = self.workspace_root / path
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _relative_time(self, stamp_sec):
        if self.start_ros_time <= 0.0:
            return 0.0
        return float(stamp_sec - self.start_ros_time)

    def _record(self, key, payload):
        with self.lock:
            self.latest_samples[key] = deepcopy(payload)
            if not self.recording:
                return
            self.buffers[key].append(payload)

    def _cancel_stop_timer_locked(self):
        if self.stop_timer is not None:
            self.stop_timer.cancel()
            self.stop_timer = None

    @staticmethod
    def _signature_delta(left, right):
        if left is None or right is None:
            return float("inf")
        if len(left) != len(right):
            return float("inf")
        return float(np.max(np.abs(left - right)))

    def _target_pose_signature(self, msg):
        return np.array([
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(msg.pose.position.z),
            float(msg.pose.orientation.x),
            float(msg.pose.orientation.y),
            float(msg.pose.orientation.z),
            float(msg.pose.orientation.w),
            float(msg.twist.linear.x),
            float(msg.twist.linear.y),
            float(msg.twist.linear.z),
            float(msg.twist.angular.x),
            float(msg.twist.angular.y),
            float(msg.twist.angular.z),
            float(msg.acceleration.linear.x),
            float(msg.acceleration.linear.y),
            float(msg.acceleration.linear.z),
            float(msg.acceleration.angular.x),
            float(msg.acceleration.angular.y),
            float(msg.acceleration.angular.z),
            float(msg.desired_force.x),
            float(msg.desired_force.y),
            float(msg.desired_force.z),
            float(msg.finger_command),
        ] + [float(value) for value in msg.base_planar_positions]
          + [float(value) for value in msg.arm_joint_positions], dtype=float)

    def _begin_recording_locked(self, stamp_sec, reason):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_ros_time = stamp_sec
        self.start_wall_time = rospy.Time.now().to_sec()
        self.buffers = self._empty_buffers()
        self.recording = True
        self._cancel_stop_timer_locked()
        rospy.loginfo(
            "experiment_data_recorder: recording started with timestamp %s (%s).",
            self.timestamp,
            reason)

    def _maybe_schedule_wait_stop_locked(self, now_wall):
        if not self.waiting_for_command or not self.recording:
            return
        if self.stop_timer is not None:
            return

        stable_sec = max(0.0, float(self.config.get("wait_command_stable_sec", 0.75)))
        min_record_sec = max(0.0, float(self.config.get("wait_min_record_duration_sec", 1.0)))
        post_sec = max(0.0, float(self.config.get("post_plan_duration_sec", 5.0)))
        if now_wall - self.start_wall_time < min_record_sec:
            return
        if now_wall - self.wait_last_change_wall_time < stable_sec:
            return

        self.stop_timer = threading.Timer(
            post_sec,
            self._finish_wait_recording_from_timer)
        self.stop_timer.daemon = True
        self.stop_timer.start()
        rospy.loginfo(
            "experiment_data_recorder: target command stream is stable; will stop after %.3f s.",
            post_sec)

    def _wait_monitor_timer_callback(self, _event):
        now_wall = rospy.Time.now().to_sec()
        with self.lock:
            self._maybe_schedule_wait_stop_locked(now_wall)

    def _float_callback(self, msg, key):
        now = rospy.Time.now().to_sec()
        self._record(key, {
            "time": self._relative_time(now),
            "ros_time": now,
            "value": float(msg.data),
        })

    def _desired_force_callback(self, msg):
        stamp = self._stamp_to_sec(msg.header.stamp)
        vector = [float(msg.vector.x), float(msg.vector.y), float(msg.vector.z)]
        self._record("desired_force", {
            "time": self._relative_time(stamp),
            "ros_time": stamp,
            "frame_id": msg.header.frame_id,
            "vector": vector,
            "norm": float(np.linalg.norm(vector)),
        })

    def _force_polytope_vertices_callback(self, msg):
        data = list(float(value) for value in msg.data)
        if len(data) % 3 != 0:
            rospy.logwarn_throttle(
                2.0,
                "experiment_data_recorder: force polytope vertices message length %d is not divisible by 3.",
                len(data))
            return
        now = rospy.Time.now().to_sec()
        vertices = [
            data[index:index + 3]
            for index in range(0, len(data), 3)
        ]
        self._record("force_polytope_vertices", {
            "time": self._relative_time(now),
            "ros_time": now,
            "vertices": vertices,
        })

    def _manipulability_ellipsoid_callback(self, msg):
        data = np.asarray([float(value) for value in msg.data], dtype=float)
        if data.size != 9 or not np.all(np.isfinite(data)):
            rospy.logwarn_throttle(
                2.0,
                "experiment_data_recorder: manipulability ellipsoid message must contain 9 finite values.")
            return
        matrix = data.reshape((3, 3))
        matrix = 0.5 * (matrix + matrix.T)
        now = rospy.Time.now().to_sec()
        eigenvalues = np.linalg.eigvalsh(matrix)
        self._record("manipulability_ellipsoid", {
            "time": self._relative_time(now),
            "ros_time": now,
            "matrix": matrix.tolist(),
            "eigenvalues": eigenvalues.tolist(),
        })

    def _pose_callback(self, msg, key):
        stamp = self._stamp_to_sec(msg.header.stamp)
        pose = msg.pose
        self._record(key, {
            "time": self._relative_time(stamp),
            "ros_time": stamp,
            "frame_id": msg.header.frame_id,
            "position": [
                float(pose.position.x),
                float(pose.position.y),
                float(pose.position.z),
            ],
            "orientation": [
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            ],
        })

    def _joint_state_callback(self, msg, key):
        stamp = self._stamp_to_sec(msg.header.stamp)
        self._record(key, {
            "time": self._relative_time(stamp),
            "ros_time": stamp,
            "names": list(msg.name),
            "position": [float(value) for value in msg.position],
            "velocity": [float(value) for value in msg.velocity],
            "effort": [float(value) for value in msg.effort],
        })

    def _build_target_pose_payload(self, msg):
        stamp = self._stamp_to_sec(msg.header.stamp)
        return {
            "time": self._relative_time(stamp),
            "ros_time": stamp,
            "frame_id": msg.header.frame_id,
            "pose": {
                "position": [
                    float(msg.pose.position.x),
                    float(msg.pose.position.y),
                    float(msg.pose.position.z),
                ],
                "orientation": [
                    float(msg.pose.orientation.x),
                    float(msg.pose.orientation.y),
                    float(msg.pose.orientation.z),
                    float(msg.pose.orientation.w),
                ],
            },
            "twist": {
                "linear": [
                    float(msg.twist.linear.x),
                    float(msg.twist.linear.y),
                    float(msg.twist.linear.z),
                ],
                "angular": [
                    float(msg.twist.angular.x),
                    float(msg.twist.angular.y),
                    float(msg.twist.angular.z),
                ],
            },
            "acceleration": {
                "linear": [
                    float(msg.acceleration.linear.x),
                    float(msg.acceleration.linear.y),
                    float(msg.acceleration.linear.z),
                ],
                "angular": [
                    float(msg.acceleration.angular.x),
                    float(msg.acceleration.angular.y),
                    float(msg.acceleration.angular.z),
                ],
            },
            "desired_force": [
                float(msg.desired_force.x),
                float(msg.desired_force.y),
                float(msg.desired_force.z),
            ],
            "finger_command": float(msg.finger_command),
            "use_nullspace_joint_target": bool(msg.use_nullspace_joint_target),
            "base_planar_positions": [
                float(value) for value in msg.base_planar_positions
            ],
            "arm_joint_positions": [
                float(value) for value in msg.arm_joint_positions
            ],
        }

    def _target_pose_callback(self, msg):
        stamp = self._stamp_to_sec(msg.header.stamp)
        now_wall = rospy.Time.now().to_sec()
        signature = self._target_pose_signature(msg)
        should_record = False

        with self.lock:
            self.latest_target_signature = signature
            self.latest_target_wall_time = now_wall

            if self.waiting_for_command:
                if not self.recording:
                    if self.wait_baseline_signature is None:
                        self._begin_recording_locked(
                            stamp,
                            "first target command after wait-start")
                        self.wait_last_signature = signature
                        self.wait_last_change_wall_time = now_wall
                    else:
                        start_threshold = float(
                            self.config.get("wait_start_delta_threshold", 1e-5))
                        if self._signature_delta(
                                signature, self.wait_baseline_signature) >= start_threshold:
                            self._begin_recording_locked(
                                stamp,
                                "target command changed after wait-start")
                            self.wait_last_signature = signature
                            self.wait_last_change_wall_time = now_wall
                        else:
                            self.wait_last_signature = signature
                else:
                    change_threshold = float(
                        self.config.get("wait_change_delta_threshold", 1e-5))
                    if self._signature_delta(
                            signature, self.wait_last_signature) >= change_threshold:
                        self.wait_last_change_wall_time = now_wall
                        self.wait_last_signature = signature
                        self._cancel_stop_timer_locked()
                    self._maybe_schedule_wait_stop_locked(now_wall)

            should_record = self.recording

        if should_record:
            payload = self._build_target_pose_payload(msg)
            payload["time"] = self._relative_time(stamp)
            self._record("target_pose", payload)

    def _comparison_run_state_callback(self, msg):
        now = rospy.Time.now().to_sec()
        self._record("comparison_run_state", {
            "time": self._relative_time(now),
            "ros_time": now,
            "state": int(msg.data),
        })
        with self.lock:
            if not self.recording:
                return
            if not self.config.get("auto_stop_on_comparison_complete", True):
                return
            if int(msg.data) != 3 or self.stop_timer is not None:
                return
            delay = max(0.0, float(self.config.get("post_plan_duration_sec", 5.0)))
            self.stop_timer = threading.Timer(
                delay,
                self._finish_recording_from_timer)
            self.stop_timer.daemon = True
            self.stop_timer.start()
            rospy.loginfo(
                "experiment_data_recorder: comparison complete; will stop after %.3f s.",
                delay)

    def _vlm_mass_result_callback(self, msg):
        stamp = self._stamp_to_sec(msg.header.stamp)
        self._record("vlm_mass_result", {
            "time": self._relative_time(stamp),
            "ros_time": stamp,
            "request_id": msg.request_id,
            "success": bool(msg.success),
            "source": msg.source,
            "model": msg.model,
            "image_path": msg.image_path,
            "result_json_path": msg.result_json_path,
            "mass_kg": float(msg.mass_kg),
            "mass_kg_range": [float(value) for value in msg.mass_kg_range],
            "material_guess": msg.material_guess,
            "object_description": msg.object_description,
            "confidence": msg.confidence,
            "error_message": msg.error_message,
        })

    def _start_service(self, _request):
        with self.lock:
            if self.recording or self.waiting_for_command:
                return TriggerResponse(
                    success=True,
                    message="already active with timestamp {}".format(self.timestamp))
            self.config = self._read_runtime_config()
            self.waiting_for_command = False
            self.wait_baseline_signature = None
            self.wait_last_signature = None
            self.wait_last_change_wall_time = 0.0
            self._begin_recording_locked(rospy.Time.now().to_sec(), "manual start")
        rospy.loginfo(
            "experiment_data_recorder: recording started with timestamp %s.",
            self.timestamp)
        return TriggerResponse(
            success=True,
            message="recording started: {}".format(self.timestamp))

    def _wait_start_service(self, _request):
        with self.lock:
            if self.recording or self.waiting_for_command:
                return TriggerResponse(
                    success=True,
                    message="already active with timestamp {}".format(self.timestamp))
            self.config = self._read_runtime_config()
            self.buffers = self._empty_buffers()
            self.recording = False
            self.waiting_for_command = True
            self.timestamp = ""
            self.start_ros_time = 0.0
            self.start_wall_time = 0.0
            self.wait_last_signature = None
            self.wait_last_change_wall_time = 0.0
            self._cancel_stop_timer_locked()

            recent_timeout = max(
                0.0,
                float(self.config.get("wait_recent_target_timeout_sec", 1.0)))
            now_wall = rospy.Time.now().to_sec()
            if self.latest_target_signature is not None and \
                    now_wall - self.latest_target_wall_time <= recent_timeout:
                self.wait_baseline_signature = np.array(
                    self.latest_target_signature,
                    dtype=float)
                baseline_state = "with current /target_pose baseline"
            else:
                self.wait_baseline_signature = None
                baseline_state = "without current /target_pose baseline"

        rospy.loginfo(
            "experiment_data_recorder: armed waiting recording %s.",
            baseline_state)
        return TriggerResponse(
            success=True,
            message="waiting for target command start ({})".format(baseline_state))

    def _stop_service(self, _request):
        success, message = self._finish_recording("manual stop")
        return TriggerResponse(success=success, message=message)

    def _record_point_service(self, _request):
        with self.lock:
            config = self._read_runtime_config()
            required_keys = [
                "desired_force",
                "force_capacity",
                "manipulability",
                "manipulability_ellipsoid",
                "force_polytope_vertices",
            ]
            missing = [
                key for key in required_keys
                if key not in self.latest_samples
            ]
            if missing:
                return TriggerResponse(
                    success=False,
                    message="missing latest samples: {}".format(", ".join(missing)))

            point = {
                "index": len(self.point_samples) + 1,
                "recorded_at": datetime.now().isoformat(),
                "desired_force_radius_N": float(config.get("desired_force_radius_N", 0.0)),
                "desired_force": deepcopy(self.latest_samples["desired_force"]),
                "force_capacity": deepcopy(self.latest_samples["force_capacity"]),
                "manipulability": deepcopy(self.latest_samples["manipulability"]),
                "manipulability_ellipsoid": deepcopy(
                    self.latest_samples["manipulability_ellipsoid"]),
                "force_polytope_vertices": deepcopy(
                    self.latest_samples["force_polytope_vertices"]),
            }
            self.point_samples.append(point)
            count = len(self.point_samples)

        return TriggerResponse(
            success=True,
            message="recorded single-point sample {} (total {})".format(
                point["index"],
                count))

    def _undo_point_service(self, _request):
        with self.lock:
            if not self.point_samples:
                return TriggerResponse(success=False, message="no single-point samples")
            removed = self.point_samples.pop()
            for index, point in enumerate(self.point_samples, start=1):
                point["index"] = index
            count = len(self.point_samples)
        return TriggerResponse(
            success=True,
            message="removed single-point sample {} (remaining {})".format(
                removed.get("index", "?"),
                count))

    def _finish_points_service(self, _request):
        with self.lock:
            if not self.point_samples:
                return TriggerResponse(
                    success=False,
                    message="no single-point samples to render")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config = self._read_runtime_config()
            points = deepcopy(self.point_samples)
            self.point_samples = []

        save_thread = threading.Thread(
            target=self._save_single_point_outputs_background,
            args=(timestamp, config, points),
            daemon=True)
        save_thread.start()
        payload = {
            "success": True,
            "reason": "single point finish",
            "saving": True,
            "timestamp": timestamp,
            "summary": "saving {} single-point samples in background".format(len(points)),
            "outputs": {},
        }
        return TriggerResponse(success=True, message=json.dumps(payload, ensure_ascii=False))

    def _finish_recording_from_timer(self):
        self._finish_recording("comparison complete + post duration")

    def _finish_wait_recording_from_timer(self):
        self._finish_recording("target command stable + post duration")

    def _read_runtime_config(self):
        desired_force_radius = float(
            rospy.get_param("/target_pose_generator/desired_force_radius_N", 0.0))
        return {
            "save_force_capacity": bool(rospy.get_param("~save_force_capacity", True)),
            "save_manipulability": bool(rospy.get_param("~save_manipulability", True)),
            "save_manipulability_ellipsoid": bool(
                rospy.get_param("~save_manipulability_ellipsoid", True)),
            "save_residual_polytope": bool(
                rospy.get_param("~save_residual_polytope", True)),
            "post_plan_duration_sec": float(
                rospy.get_param("~post_plan_duration_sec", 5.0)),
            "auto_stop_on_comparison_complete": bool(
                rospy.get_param("~auto_stop_on_comparison_complete", True)),
            "wait_start_delta_threshold": float(
                rospy.get_param("~wait_start_delta_threshold", 1e-5)),
            "wait_change_delta_threshold": float(
                rospy.get_param("~wait_change_delta_threshold", 1e-5)),
            "wait_command_stable_sec": float(
                rospy.get_param("~wait_command_stable_sec", 0.75)),
            "wait_min_record_duration_sec": float(
                rospy.get_param("~wait_min_record_duration_sec", 1.0)),
            "wait_recent_target_timeout_sec": float(
                rospy.get_param("~wait_recent_target_timeout_sec", 1.0)),
            "data_output_dir": rospy.get_param(
                "~data_output_dir", "output/polytope_ros/data"),
            "fig_output_dir": rospy.get_param(
                "~fig_output_dir", "output/polytope_ros/fig"),
            "animation_fps": float(rospy.get_param("~animation_fps", 10.0)),
            "animation_max_frames": int(rospy.get_param("~animation_max_frames", 180)),
            "desired_force_radius_N": max(0.0, desired_force_radius),
            "gravity_acceleration_mps2": float(
                rospy.get_param("/target_pose_generator/gravity_acceleration_mps2", 9.81)),
            "fallback_polytope_radius_N": float(
                rospy.get_param("~fallback_polytope_radius_N", 80.0)),
            "topics": {
                "manipulability": self.manipulability_topic,
                "manipulability_ellipsoid": self.manipulability_ellipsoid_topic,
                "force_capacity": self.force_capacity_topic,
                "desired_force": self.desired_force_topic,
                "force_polytope_vertices": self.force_polytope_vertices_topic,
                "pose": self.pose_topic,
                "pose_target": self.pose_target_topic,
                "state": self.state_topic,
                "state_target": self.state_target_topic,
                "target_pose": self.target_pose_topic,
                "comparison_run_state": self.comparison_run_state_topic,
                "vlm_mass_result": self.vlm_mass_result_topic,
            },
        }

    def _finish_recording(self, reason):
        with self.lock:
            if not self.recording:
                if self.waiting_for_command:
                    self.waiting_for_command = False
                    self._cancel_stop_timer_locked()
                    return True, "waiting recording cancelled"
                return False, "not recording"
            self.recording = False
            self.waiting_for_command = False
            self.wait_baseline_signature = None
            self.wait_last_signature = None
            self.wait_last_change_wall_time = 0.0
            self._cancel_stop_timer_locked()
            timestamp = self.timestamp
            config = deepcopy(self.config)
            buffers = self.buffers
            self.buffers = self._empty_buffers()

        save_thread = threading.Thread(
            target=self._save_outputs_background,
            args=(timestamp, config, buffers, reason),
            daemon=True)
        save_thread.start()

        payload = {
            "success": True,
            "reason": reason,
            "saving": True,
            "timestamp": timestamp,
            "summary": "recording stopped; saving plots/videos in background for {}".format(
                timestamp),
            "outputs": {},
        }
        rospy.loginfo("experiment_data_recorder: %s", payload["summary"])
        return True, json.dumps(payload, ensure_ascii=False)

    def _save_outputs_background(self, timestamp, config, buffers, reason):
        try:
            outputs = self._save_outputs(timestamp, config, buffers, reason)
            payload = {
                "success": True,
                "reason": reason,
                "saving": False,
                "timestamp": timestamp,
                "summary": "saved {} files under data={} fig={}".format(
                    len(outputs),
                    outputs.get("data_dir", "-"),
                    outputs.get("fig_dir", "-")),
                "outputs": outputs,
            }
            rospy.loginfo("experiment_data_recorder: %s", payload["summary"])
        except Exception as exc:  # pylint: disable=broad-except
            payload = {
                "success": False,
                "reason": reason,
                "saving": False,
                "timestamp": timestamp,
                "summary": "failed to save outputs for {}: {}".format(timestamp, exc),
                "outputs": {},
                "error": str(exc),
            }
            rospy.logerr("experiment_data_recorder: %s", payload["summary"])

        self.latest_outputs_pub.publish(
            String(data=json.dumps(payload, ensure_ascii=False)))

    def _save_single_point_outputs_background(self, timestamp, config, points):
        try:
            outputs = self._save_single_point_outputs(timestamp, config, points)
            payload = {
                "success": True,
                "reason": "single point finish",
                "saving": False,
                "timestamp": timestamp,
                "summary": "saved {} single-point output files under data={} fig={}".format(
                    len(outputs),
                    outputs.get("data_dir", "-"),
                    outputs.get("fig_dir", "-")),
                "outputs": outputs,
            }
            rospy.loginfo("experiment_data_recorder: %s", payload["summary"])
        except Exception as exc:  # pylint: disable=broad-except
            payload = {
                "success": False,
                "reason": "single point finish",
                "saving": False,
                "timestamp": timestamp,
                "summary": "failed to save single-point outputs for {}: {}".format(
                    timestamp,
                    exc),
                "outputs": {},
                "error": str(exc),
            }
            rospy.logerr("experiment_data_recorder: %s", payload["summary"])

        self.latest_outputs_pub.publish(
            String(data=json.dumps(payload, ensure_ascii=False)))

    def _save_single_point_outputs(self, timestamp, config, points):
        data_dir = self._resolve_output_dir(config["data_output_dir"])
        fig_dir = self._resolve_output_dir(config["fig_output_dir"])
        outputs = {
            "data_dir": str(data_dir),
            "fig_dir": str(fig_dir),
        }

        payload = {
            "timestamp": timestamp,
            "reason": "single point finish",
            "config": config,
            "points": points,
        }
        json_path = data_dir / "single_point_samples_{}.json".format(timestamp)
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        outputs["single_point_json"] = str(json_path)

        csv_path = data_dir / "single_point_metrics_{}.csv".format(timestamp)
        self._write_single_point_metrics_csv(csv_path, points)
        outputs["single_point_metrics_csv"] = str(csv_path)

        metrics_png = fig_dir / "single_point_metrics_{}.png".format(timestamp)
        self._plot_single_point_metrics(metrics_png, points)
        outputs["single_point_metrics_png"] = str(metrics_png)

        for point in points:
            point_index = int(point.get("index", 0))
            ellipsoid_png = fig_dir / "single_point_{:02d}_manipulability_ellipsoid_{}.png".format(
                point_index,
                timestamp)
            self._render_single_point_ellipsoid(ellipsoid_png, point)
            outputs["single_point_{:02d}_ellipsoid_png".format(point_index)] = str(
                ellipsoid_png)

            polytope_png = fig_dir / "single_point_{:02d}_residual_polytope_{}.png".format(
                point_index,
                timestamp)
            self._render_single_point_polytope(polytope_png, point, config)
            outputs["single_point_{:02d}_polytope_png".format(point_index)] = str(
                polytope_png)

        manifest_path = data_dir / "single_point_manifest_{}.json".format(timestamp)
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(outputs, handle, indent=2, ensure_ascii=False)
        outputs["manifest"] = str(manifest_path)
        return outputs

    def _save_outputs(self, timestamp, config, buffers, reason):
        data_dir = self._resolve_output_dir(config["data_output_dir"])
        fig_dir = self._resolve_output_dir(config["fig_output_dir"])
        outputs = {
            "data_dir": str(data_dir),
            "fig_dir": str(fig_dir),
        }

        payload = {
            "timestamp": timestamp,
            "reason": reason,
            "config": config,
            "data": buffers,
        }
        json_path = data_dir / "moca_plot_data_{}.json".format(timestamp)
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        outputs["json"] = str(json_path)

        if config["save_force_capacity"]:
            path = data_dir / "force_capacity_{}.csv".format(timestamp)
            self._write_scalar_csv(path, buffers["force_capacity"])
            outputs["force_capacity_csv"] = str(path)
            fig_path = fig_dir / "force_capacity_{}.png".format(timestamp)
            self._plot_scalar_series(
                fig_path,
                buffers["force_capacity"],
                "Directional Force Capacity",
                "Force capacity / N",
                overlay=buffers["desired_force"],
                overlay_label="Desired force norm / N",
                overlay_band_radius=config.get("desired_force_radius_N", 0.0),
                vlm_results=buffers["vlm_mass_result"],
                gravity=config.get("gravity_acceleration_mps2", 9.81))
            outputs["force_capacity_png"] = str(fig_path)

        if config["save_manipulability"]:
            path = data_dir / "manipulability_{}.csv".format(timestamp)
            self._write_scalar_csv(path, buffers["manipulability"])
            outputs["manipulability_csv"] = str(path)
            fig_path = fig_dir / "manipulability_{}.png".format(timestamp)
            self._plot_scalar_series(
                fig_path,
                buffers["manipulability"],
                "Manipulability",
                "Manipulability")
            outputs["manipulability_png"] = str(fig_path)

        if config["save_manipulability_ellipsoid"]:
            ellipsoid_csv = data_dir / "manipulability_ellipsoid_{}.csv".format(timestamp)
            self._write_matrix_csv(ellipsoid_csv, buffers["manipulability_ellipsoid"])
            outputs["manipulability_ellipsoid_csv"] = str(ellipsoid_csv)
            ellipsoid_gif = fig_dir / "manipulability_ellipsoid_{}.gif".format(timestamp)
            ellipsoid_preview = fig_dir / "manipulability_ellipsoid_preview_{}.png".format(timestamp)
            ellipsoid_mp4 = fig_dir / "manipulability_ellipsoid_{}.mp4".format(timestamp)
            self._render_manipulability_ellipsoid_animation(
                ellipsoid_gif,
                buffers,
                config,
                preview_path=ellipsoid_preview,
                mp4_path=ellipsoid_mp4)
            if ellipsoid_preview.exists():
                outputs["manipulability_ellipsoid_preview_png"] = str(ellipsoid_preview)
            if ellipsoid_gif.exists():
                outputs["manipulability_ellipsoid_gif"] = str(ellipsoid_gif)
            if ellipsoid_mp4.exists():
                outputs["manipulability_ellipsoid_mp4"] = str(ellipsoid_mp4)

        desired_force_csv = data_dir / "desired_force_{}.csv".format(timestamp)
        self._write_desired_force_csv(desired_force_csv, buffers["desired_force"])
        outputs["desired_force_csv"] = str(desired_force_csv)
        vlm_result_csv = data_dir / "vlm_mass_result_{}.csv".format(timestamp)
        self._write_vlm_mass_result_csv(vlm_result_csv, buffers["vlm_mass_result"])
        outputs["vlm_mass_result_csv"] = str(vlm_result_csv)
        state_csv = data_dir / "moca_state_{}.csv".format(timestamp)
        self._write_joint_state_csv(state_csv, buffers["state"])
        outputs["moca_state_csv"] = str(state_csv)
        target_pose_csv = data_dir / "target_pose_{}.csv".format(timestamp)
        self._write_target_pose_csv(target_pose_csv, buffers["target_pose"])
        outputs["target_pose_csv"] = str(target_pose_csv)

        if config["save_residual_polytope"]:
            vertices_csv = data_dir / "force_polytope_vertices_{}.csv".format(timestamp)
            self._write_vertices_csv(vertices_csv, buffers["force_polytope_vertices"])
            outputs["force_polytope_vertices_csv"] = str(vertices_csv)
            gif_path = fig_dir / "residual_polytope_{}.gif".format(timestamp)
            preview_path = fig_dir / "residual_polytope_preview_{}.png".format(timestamp)
            mp4_path = fig_dir / "residual_polytope_{}.mp4".format(timestamp)
            source = self._render_residual_polytope_animation(
                gif_path,
                buffers,
                config,
                preview_path=preview_path,
                mp4_path=mp4_path)
            if preview_path.exists():
                outputs["residual_polytope_preview_png"] = str(preview_path)
            if gif_path.exists():
                outputs["residual_polytope_gif"] = str(gif_path)
            if mp4_path.exists():
                outputs["residual_polytope_mp4"] = str(mp4_path)
            outputs["residual_polytope_source"] = source

        manifest_path = data_dir / "manifest_{}.json".format(timestamp)
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(outputs, handle, indent=2, ensure_ascii=False)
        outputs["manifest"] = str(manifest_path)
        return outputs

    @staticmethod
    def _write_scalar_csv(path, rows):
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time", "ros_time", "value"])
            for row in rows:
                writer.writerow([row["time"], row["ros_time"], row["value"]])

    @staticmethod
    def _write_desired_force_csv(path, rows):
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time", "ros_time", "frame_id", "fx", "fy", "fz", "norm"])
            for row in rows:
                vector = row["vector"]
                writer.writerow([
                    row["time"], row["ros_time"], row["frame_id"],
                    vector[0], vector[1], vector[2], row["norm"]])

    @staticmethod
    def _write_vlm_mass_result_csv(path, rows):
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "time", "ros_time", "request_id", "success", "source", "model",
                "mass_kg", "mass_kg_min", "mass_kg_max", "confidence",
                "image_path", "result_json_path", "error_message"])
            for row in rows:
                mass_range = row.get("mass_kg_range", [])
                writer.writerow([
                    row.get("time", 0.0),
                    row.get("ros_time", 0.0),
                    row.get("request_id", ""),
                    row.get("success", False),
                    row.get("source", ""),
                    row.get("model", ""),
                    row.get("mass_kg", 0.0),
                    mass_range[0] if len(mass_range) >= 1 else "",
                    mass_range[1] if len(mass_range) >= 2 else "",
                    row.get("confidence", ""),
                    row.get("image_path", ""),
                    row.get("result_json_path", ""),
                    row.get("error_message", ""),
                ])

    @staticmethod
    def _write_joint_state_csv(path, rows):
        if not rows:
            with open(path, "w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(["time", "ros_time"])
            return
        names = rows[0].get("names", [])
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["time", "ros_time"] +
                ["pos_{}".format(name) for name in names] +
                ["vel_{}".format(name) for name in names] +
                ["eff_{}".format(name) for name in names])
            for row in rows:
                writer.writerow(
                    [row["time"], row["ros_time"]] +
                    row.get("position", []) +
                    row.get("velocity", []) +
                    row.get("effort", []))

    @staticmethod
    def _write_target_pose_csv(path, rows):
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "time", "ros_time", "frame_id",
                "px", "py", "pz", "qx", "qy", "qz", "qw",
                "fx", "fy", "fz",
                "base_x", "base_y", "base_yaw",
                "q1", "q2", "q3", "q4", "q5", "q6", "q7"])
            for row in rows:
                writer.writerow(
                    [row["time"], row["ros_time"], row["frame_id"]] +
                    row["pose"]["position"] +
                    row["pose"]["orientation"] +
                    row["desired_force"] +
                    row["base_planar_positions"] +
                    row["arm_joint_positions"])

    @staticmethod
    def _write_vertices_csv(path, rows):
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frame_index", "time", "ros_time", "vertex_index", "x", "y", "z"])
            for frame_index, row in enumerate(rows):
                for vertex_index, vertex in enumerate(row["vertices"]):
                    writer.writerow([
                        frame_index,
                        row["time"],
                        row["ros_time"],
                        vertex_index,
                        vertex[0],
                        vertex[1],
                        vertex[2],
                    ])

    @staticmethod
    def _write_matrix_csv(path, rows):
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "frame_index", "time", "ros_time",
                "m00", "m01", "m02",
                "m10", "m11", "m12",
                "m20", "m21", "m22",
                "eig0", "eig1", "eig2"])
            for frame_index, row in enumerate(rows):
                matrix = np.asarray(row.get("matrix", np.zeros((3, 3))), dtype=float)
                if matrix.shape != (3, 3):
                    matrix = np.zeros((3, 3), dtype=float)
                eigenvalues = list(row.get("eigenvalues", []))
                if len(eigenvalues) != 3:
                    eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T)).tolist()
                writer.writerow(
                    [frame_index, row["time"], row["ros_time"]] +
                    matrix.reshape(-1).tolist() +
                    eigenvalues)

    @staticmethod
    def _write_single_point_metrics_csv(path, points):
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "point_index",
                "recorded_at",
                "desired_fx",
                "desired_fy",
                "desired_fz",
                "desired_force_norm_N",
                "desired_force_radius_N",
                "directional_force_capacity_N",
                "manipulability",
            ])
            for point in points:
                desired = point.get("desired_force", {})
                vector = list(desired.get("vector", [0.0, 0.0, 0.0]))
                while len(vector) < 3:
                    vector.append(0.0)
                writer.writerow([
                    point.get("index", ""),
                    point.get("recorded_at", ""),
                    vector[0],
                    vector[1],
                    vector[2],
                    desired.get("norm", 0.0),
                    point.get("desired_force_radius_N", 0.0),
                    point.get("force_capacity", {}).get("value", 0.0),
                    point.get("manipulability", {}).get("value", 0.0),
                ])

    @staticmethod
    def _plot_single_point_metrics(path, points):
        indices = np.asarray([int(point.get("index", index + 1))
                              for index, point in enumerate(points)], dtype=float)
        force_capacity = np.asarray([
            float(point.get("force_capacity", {}).get("value", np.nan))
            for point in points
        ], dtype=float)
        desired_norm = np.asarray([
            float(point.get("desired_force", {}).get("norm", np.nan))
            for point in points
        ], dtype=float)
        desired_radius = np.asarray([
            max(0.0, float(point.get("desired_force_radius_N", 0.0)))
            for point in points
        ], dtype=float)
        manipulability = np.asarray([
            float(point.get("manipulability", {}).get("value", np.nan))
            for point in points
        ], dtype=float)

        figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        force_axis, manipulability_axis = axes

        force_axis.scatter(
            indices,
            force_capacity,
            s=58,
            color="#1f77b4",
            label="Directional force capacity / N",
            zorder=3)
        force_axis.errorbar(
            indices,
            desired_norm,
            yerr=desired_radius,
            fmt="o",
            markersize=5,
            color="tab:orange",
            ecolor=(1.0, 0.55, 0.05, 0.32),
            elinewidth=8.0,
            capsize=5,
            label="Desired force norm +/- radius / N",
            zorder=2)
        force_axis.set_ylabel("Force / N")
        force_axis.set_title("Single-Point Force Capacity")
        force_axis.grid(True, alpha=0.3)
        force_axis.legend(loc="best")

        manipulability_axis.scatter(
            indices,
            manipulability,
            s=58,
            color="#2ca02c",
            label="Manipulability",
            zorder=3)
        manipulability_axis.set_xlabel("Point index")
        manipulability_axis.set_ylabel("Manipulability")
        manipulability_axis.set_title("Single-Point Manipulability")
        manipulability_axis.grid(True, alpha=0.3)
        manipulability_axis.legend(loc="best")
        manipulability_axis.set_xticks(indices)

        figure.tight_layout()
        figure.savefig(path, dpi=180)
        plt.close(figure)

    @staticmethod
    def _plot_scalar_series(
            path, rows, title, ylabel, overlay=None, overlay_label="",
            overlay_band_radius=0.0, vlm_results=None, gravity=9.81):
        all_rows = []
        all_rows.extend(rows or [])
        all_rows.extend(overlay or [])
        all_rows.extend(vlm_results or [])
        base_ros_time = None
        ros_times = [
            float(row.get("ros_time", 0.0))
            for row in all_rows
            if "ros_time" in row
        ]
        if ros_times:
            base_ros_time = min(ros_times)

        def series_xy(series, value_key):
            if not series:
                return [], np.asarray([], dtype=float)
            sorted_rows = sorted(
                series,
                key=lambda row: float(row.get("ros_time", row.get("time", 0.0))))
            if base_ros_time is not None and any("ros_time" in row for row in sorted_rows):
                times = [
                    float(row.get("ros_time", base_ros_time)) - base_ros_time
                    for row in sorted_rows
                ]
            else:
                times = [float(row.get("time", 0.0)) for row in sorted_rows]
            values = np.asarray([
                float(row.get(value_key, 0.0))
                for row in sorted_rows
            ], dtype=float)
            return times, values

        figure, axis = plt.subplots(figsize=(10, 5))
        if rows:
            row_times, row_values = series_xy(rows, "value")
            axis.plot(
                row_times,
                row_values,
                linewidth=2.0,
                label=ylabel)
        if overlay:
            overlay_times, overlay_values = series_xy(overlay, "norm")
            band_radius = max(0.0, float(overlay_band_radius))
            if band_radius > 0.0:
                axis.fill_between(
                    overlay_times,
                    np.maximum(0.0, overlay_values - band_radius),
                    overlay_values + band_radius,
                    color="tab:orange",
                    alpha=0.18,
                    linewidth=0.0,
                    label="Desired force radius +/- {:.1f} N".format(band_radius))
            axis.plot(
                overlay_times,
                overlay_values,
                linewidth=1.6,
                linestyle="--",
                color="tab:orange",
                label=overlay_label)
        if vlm_results:
            success_rows = [
                row for row in vlm_results
                if row.get("success", False) and math.isfinite(float(row.get("mass_kg", 0.0)))
            ]
            for index, row in enumerate(sorted(
                    success_rows,
                    key=lambda item: float(item.get("ros_time", item.get("time", 0.0))))):
                if base_ros_time is not None:
                    marker_time = float(row.get("ros_time", base_ros_time)) - base_ros_time
                else:
                    marker_time = float(row.get("time", 0.0))
                expected_force = float(row.get("mass_kg", 0.0)) * float(gravity)
                label = "VLM {:.2f} kg ({:.1f} N)".format(
                    float(row.get("mass_kg", 0.0)),
                    expected_force)
                color = "tab:red" if index == 0 else "#aa3344"
                axis.axvline(
                    marker_time,
                    color=color,
                    linestyle=":",
                    linewidth=1.5,
                    alpha=0.75,
                    label=label)
                axis.scatter(
                    [marker_time],
                    [expected_force],
                    color=color,
                    marker="x",
                    s=64,
                    zorder=5)
        axis.set_title(title)
        axis.set_xlabel("Time / s")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        if rows or overlay:
            axis.legend()
        figure.tight_layout()
        figure.savefig(path, dpi=180)
        plt.close(figure)

    @staticmethod
    def _interpolate_force_at_time(desired_force_rows, time_sec):
        if not desired_force_rows:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        nearest = min(
            desired_force_rows,
            key=lambda row: abs(float(row["time"]) - float(time_sec)))
        vector = np.array(nearest["vector"], dtype=float)
        if not np.all(np.isfinite(vector)) or np.linalg.norm(vector) < 1e-9:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        return vector

    @staticmethod
    def _interpolate_capacity_at_time(force_capacity_rows, time_sec, fallback_radius):
        if not force_capacity_rows:
            return float(fallback_radius)
        nearest = min(
            force_capacity_rows,
            key=lambda row: abs(float(row["time"]) - float(time_sec)))
        value = float(nearest["value"])
        if not math.isfinite(value) or value <= 0.0:
            return float(fallback_radius)
        return value

    @staticmethod
    def _make_sphere(center, radius, resolution=18):
        u = np.linspace(0.0, 2.0 * np.pi, resolution)
        v = np.linspace(0.0, np.pi, resolution)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
        return x, y, z

    @staticmethod
    def _make_ellipsoid(matrix, resolution=26):
        matrix = np.asarray(matrix, dtype=float)
        matrix = 0.5 * (matrix + matrix.T)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            matrix = np.eye(3)
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        radii = np.sqrt(eigenvalues)
        if not np.all(np.isfinite(radii)) or float(np.max(radii)) < 1e-9:
            radii = np.array([1e-3, 1e-3, 1e-3], dtype=float)

        u = np.linspace(0.0, 2.0 * np.pi, resolution)
        v = np.linspace(0.0, np.pi, resolution)
        unit = np.vstack([
            np.outer(np.cos(u), np.sin(v)).reshape(1, -1),
            np.outer(np.sin(u), np.sin(v)).reshape(1, -1),
            np.outer(np.ones_like(u), np.cos(v)).reshape(1, -1),
        ])
        points = eigenvectors.dot(np.diag(radii)).dot(unit)
        shape = (resolution, resolution)
        return (
            points[0].reshape(shape),
            points[1].reshape(shape),
            points[2].reshape(shape),
            radii,
            eigenvectors)

    @staticmethod
    def _set_equal_axes(axis, limit):
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
        axis.set_zlim(-limit, limit)
        if hasattr(axis, "set_box_aspect"):
            axis.set_box_aspect([1.0, 1.0, 1.0])

    @staticmethod
    def _finite_vertices(vertices):
        vertices = np.asarray(vertices, dtype=float)
        if vertices.size == 0:
            return np.empty((0, 3), dtype=float)
        vertices = np.reshape(vertices, (-1, 3))
        return vertices[np.all(np.isfinite(vertices), axis=1)]

    @staticmethod
    def _draw_polytope_wireframe(axis, vertices, color="#1f77b4", alpha=0.55):
        if vertices.shape[0] < 2:
            return
        center = np.mean(vertices, axis=0)
        for vertex in vertices:
            axis.plot(
                [center[0], vertex[0]],
                [center[1], vertex[1]],
                [center[2], vertex[2]],
                color=color,
                alpha=alpha,
                linewidth=0.8)

    @staticmethod
    def _polytope_wire_segments(vertices, ConvexHull=None):
        vertices = ExperimentDataRecorder._unique_vertices(np.asarray(vertices, dtype=float))
        if vertices.shape[0] < 2:
            return []
        if vertices.shape[0] < 4:
            center = np.mean(vertices, axis=0)
            return [(center, vertex) for vertex in vertices]

        try:
            if ConvexHull is not None:
                hull = ConvexHull(vertices)
                faces = [vertices[simplex] for simplex in hull.simplices]
            else:
                faces = ExperimentDataRecorder._compute_fallback_convex_hull_faces(vertices)
        except Exception:  # pylint: disable=broad-except
            faces = []

        edge_indices = set()
        for face in faces:
            for first_vertex, second_vertex in (
                    (face[0], face[1]),
                    (face[1], face[2]),
                    (face[2], face[0])):
                first_matches = np.where(np.all(np.isclose(vertices, first_vertex, atol=1e-8), axis=1))[0]
                second_matches = np.where(np.all(np.isclose(vertices, second_vertex, atol=1e-8), axis=1))[0]
                if first_matches.size == 0 or second_matches.size == 0:
                    continue
                edge_indices.add(tuple(sorted((int(first_matches[0]), int(second_matches[0])))))

        if edge_indices:
            return [(vertices[first], vertices[second]) for first, second in sorted(edge_indices)]

        center = np.mean(vertices, axis=0)
        return [(center, vertex) for vertex in vertices]

    @staticmethod
    def _unique_vertices(vertices, decimals=8):
        if vertices.size == 0:
            return vertices
        _, indices = np.unique(np.round(vertices, decimals=decimals), axis=0, return_index=True)
        return vertices[np.sort(indices)]

    @staticmethod
    def _order_coplanar_vertices(vertices, indices, normal):
        points = vertices[list(indices)]
        center = np.mean(points, axis=0)
        normal = np.asarray(normal, dtype=float)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-12:
            return list(indices)
        normal = normal / normal_norm

        basis_u = None
        for point in points:
            candidate = point - center
            candidate -= normal * np.dot(candidate, normal)
            if np.linalg.norm(candidate) > 1e-9:
                basis_u = candidate / np.linalg.norm(candidate)
                break
        if basis_u is None:
            return list(indices)

        basis_v = np.cross(normal, basis_u)
        ordered = []
        for index in indices:
            rel = vertices[index] - center
            angle = math.atan2(np.dot(rel, basis_v), np.dot(rel, basis_u))
            ordered.append((angle, index))
        ordered.sort()
        return [index for _angle, index in ordered]

    @staticmethod
    def _compute_fallback_convex_hull_faces(vertices, tol=1e-7):
        vertices = ExperimentDataRecorder._unique_vertices(np.asarray(vertices, dtype=float))
        if vertices.shape[0] < 4:
            return []
        if np.linalg.matrix_rank(vertices - np.mean(vertices, axis=0), tol=tol) < 3:
            return []

        planes = {}
        vertex_count = vertices.shape[0]
        scale = max(1.0, float(np.nanmax(np.linalg.norm(vertices, axis=1))))
        plane_tol = tol * scale
        for triplet in itertools.combinations(range(vertex_count), 3):
            a, b, c = vertices[list(triplet)]
            normal = np.cross(b - a, c - a)
            normal_norm = np.linalg.norm(normal)
            if normal_norm < plane_tol:
                continue
            normal = normal / normal_norm
            signed_distances = (vertices - a).dot(normal)
            if np.all(signed_distances <= plane_tol) or np.all(signed_distances >= -plane_tol):
                if np.sum(np.abs(signed_distances) <= plane_tol) < 3:
                    continue
                if signed_distances.sum() > 0.0:
                    normal = -normal
                    signed_distances = -signed_distances
                coplanar = frozenset(np.where(np.abs(signed_distances) <= plane_tol)[0].tolist())
                if len(coplanar) < 3:
                    continue
                key = tuple(sorted(coplanar))
                planes[key] = (coplanar, normal)

        faces = []
        for coplanar, normal in planes.values():
            ordered = ExperimentDataRecorder._order_coplanar_vertices(vertices, list(coplanar), normal)
            if len(ordered) < 3:
                continue
            for local_index in range(1, len(ordered) - 1):
                faces.append(vertices[[ordered[0], ordered[local_index], ordered[local_index + 1]]])
        return faces

    @staticmethod
    def _draw_convex_polytope(axis, vertices, ConvexHull):
        vertices = ExperimentDataRecorder._unique_vertices(vertices)
        axis.scatter(
            vertices[:, 0], vertices[:, 1], vertices[:, 2],
            color="#0b4f9c", s=13, alpha=0.75)
        if vertices.shape[0] < 4:
            ExperimentDataRecorder._draw_polytope_wireframe(axis, vertices)
            return False
        try:
            if ConvexHull is not None:
                hull = ConvexHull(vertices)
                faces = [vertices[simplex] for simplex in hull.simplices]
            else:
                faces = ExperimentDataRecorder._compute_fallback_convex_hull_faces(vertices)
            if not faces:
                ExperimentDataRecorder._draw_polytope_wireframe(axis, vertices)
                return False
            collection = Poly3DCollection(
                faces,
                alpha=0.22,
                facecolor="#4da6ff",
                edgecolor="#0b4f9c",
                linewidth=0.55)
            axis.add_collection3d(collection)
            drawn_edges = set()
            for face in faces:
                for first_vertex, second_vertex in (
                        (face[0], face[1]),
                        (face[1], face[2]),
                        (face[2], face[0])):
                    first_matches = np.where(np.all(np.isclose(vertices, first_vertex, atol=1e-8), axis=1))[0]
                    second_matches = np.where(np.all(np.isclose(vertices, second_vertex, atol=1e-8), axis=1))[0]
                    if first_matches.size == 0 or second_matches.size == 0:
                        continue
                    edge = tuple(sorted((int(first_matches[0]), int(second_matches[0]))))
                    if edge in drawn_edges:
                        continue
                    drawn_edges.add(edge)
                    start = vertices[edge[0]]
                    end = vertices[edge[1]]
                    axis.plot(
                        [start[0], end[0]],
                        [start[1], end[1]],
                        [start[2], end[2]],
                        color="#0b4f9c",
                        alpha=0.65,
                        linewidth=0.7)
            return True
        except Exception as exc:  # pylint: disable=broad-except
            rospy.logwarn_throttle(
                5.0,
                "experiment_data_recorder: ConvexHull failed, drawing wireframe fallback: %s",
                exc)
            ExperimentDataRecorder._draw_polytope_wireframe(axis, vertices)
            return False

    def _render_single_point_ellipsoid(self, path, point):
        matrix = np.asarray(
            point.get("manipulability_ellipsoid", {}).get("matrix", np.eye(3)),
            dtype=float)
        x_values, y_values, z_values, radii, eigenvectors = self._make_ellipsoid(matrix)
        axis_limit = max(1e-3, float(np.nanmax(radii)) * 1.35)
        view_elev_deg = 24.0
        view_azim_deg = -42.0
        view_elev = math.radians(view_elev_deg)
        view_azim = math.radians(view_azim_deg)
        camera_direction = np.array([
            math.cos(view_elev) * math.cos(view_azim),
            math.cos(view_elev) * math.sin(view_azim),
            math.sin(view_elev),
        ])
        projection_colors = {
            "yz": (0.95, 0.20, 0.18, 0.38),
            "xz": (0.12, 0.55, 0.22, 0.38),
            "xy": (0.18, 0.32, 0.78, 0.38),
        }

        def add_coordinate_plane_projections():
            x_plane = -math.copysign(axis_limit, camera_direction[0])
            y_plane = -math.copysign(axis_limit, camera_direction[1])
            z_plane = -math.copysign(axis_limit, camera_direction[2])

            projection_specs = [
                (
                    "yz",
                    [
                        np.column_stack((
                            np.full_like(y_values[row_index, :], x_plane),
                            y_values[row_index, :],
                            z_values[row_index, :]))
                        for row_index in range(y_values.shape[0])
                    ] + [
                        np.column_stack((
                            np.full_like(y_values[:, col_index], x_plane),
                            y_values[:, col_index],
                            z_values[:, col_index]))
                        for col_index in range(y_values.shape[1])
                    ]),
                (
                    "xz",
                    [
                        np.column_stack((
                            x_values[row_index, :],
                            np.full_like(x_values[row_index, :], y_plane),
                            z_values[row_index, :]))
                        for row_index in range(x_values.shape[0])
                    ] + [
                        np.column_stack((
                            x_values[:, col_index],
                            np.full_like(x_values[:, col_index], y_plane),
                            z_values[:, col_index]))
                        for col_index in range(x_values.shape[1])
                    ]),
                (
                    "xy",
                    [
                        np.column_stack((
                            x_values[row_index, :],
                            y_values[row_index, :],
                            np.full_like(x_values[row_index, :], z_plane)))
                        for row_index in range(x_values.shape[0])
                    ] + [
                        np.column_stack((
                            x_values[:, col_index],
                            y_values[:, col_index],
                            np.full_like(x_values[:, col_index], z_plane)))
                        for col_index in range(x_values.shape[1])
                    ]),
            ]

            for plane_key, segments in projection_specs:
                axis.add_collection3d(
                    Line3DCollection(
                        segments,
                        colors=[projection_colors[plane_key]] * len(segments),
                        linewidths=0.42,
                        linestyles="solid"))

        figure = plt.figure(figsize=(7, 7))
        axis = figure.add_subplot(111, projection="3d")
        add_coordinate_plane_projections()
        axis.plot_surface(
            x_values,
            y_values,
            z_values,
            color=(0.30, 0.62, 0.95, 0.22),
            linewidth=0.0,
            antialiased=True,
            shade=False)

        line_segments = []
        for row_index in range(x_values.shape[0]):
            line_segments.append(np.column_stack((
                x_values[row_index, :],
                y_values[row_index, :],
                z_values[row_index, :])))
        for col_index in range(x_values.shape[1]):
            line_segments.append(np.column_stack((
                x_values[:, col_index],
                y_values[:, col_index],
                z_values[:, col_index])))
        axis.add_collection3d(
            Line3DCollection(
                line_segments,
                colors=[(0.05, 0.18, 0.32, 0.52)] * len(line_segments),
                linewidths=0.7))

        colors = ["#d62728", "#2ca02c", "#ff7f0e"]
        labels = ["major", "middle", "minor"]
        order = np.argsort(radii)[::-1]
        for draw_index, axis_index in enumerate(order):
            radius = float(radii[axis_index])
            direction = eigenvectors[:, axis_index]
            endpoint = direction * radius
            axis.plot(
                [-endpoint[0], endpoint[0]],
                [-endpoint[1], endpoint[1]],
                [-endpoint[2], endpoint[2]],
                color=colors[draw_index],
                linewidth=2.2,
                label="{} axis {:.4g}".format(labels[draw_index], radius))

        self._set_equal_axes(axis, axis_limit)
        axis.view_init(elev=view_elev_deg, azim=view_azim_deg)
        axis.set_xlabel("vx direction")
        axis.set_ylabel("vy direction")
        axis.set_zlabel("vz direction")
        axis.set_title(
            "Point {} Manipulability Ellipsoid".format(point.get("index", "?")))
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper right")
        figure.tight_layout()
        figure.savefig(path, dpi=180)
        plt.close(figure)

    def _render_single_point_polytope(self, path, point, config):
        try:
            from scipy.spatial import ConvexHull  # pylint: disable=import-outside-toplevel
        except Exception:  # pylint: disable=broad-except
            ConvexHull = None

        desired_force = np.asarray(
            point.get("desired_force", {}).get("vector", [0.0, 0.0, 0.0]),
            dtype=float)
        if desired_force.shape != (3,) or not np.all(np.isfinite(desired_force)):
            desired_force = np.zeros(3, dtype=float)
        desired_norm = float(np.linalg.norm(desired_force))
        desired_radius = max(0.0, float(point.get("desired_force_radius_N", 0.0)))
        ball_radius = max(0.1, desired_radius)
        vertices = self._finite_vertices(
            point.get("force_polytope_vertices", {}).get("vertices", []))
        view_elev_deg = 24.0
        view_azim_deg = -42.0
        view_elev = math.radians(view_elev_deg)
        view_azim = math.radians(view_azim_deg)
        camera_direction = np.array([
            math.cos(view_elev) * math.cos(view_azim),
            math.cos(view_elev) * math.sin(view_azim),
            math.sin(view_elev),
        ])
        plane_projection_colors = {
            "yz": (0.85, 0.18, 0.16, 0.34),
            "xz": (0.12, 0.50, 0.18, 0.34),
            "xy": (0.16, 0.28, 0.72, 0.34),
        }

        def projection_plane_values(limit):
            return {
                "yz": -math.copysign(limit, camera_direction[0]),
                "xz": -math.copysign(limit, camera_direction[1]),
                "xy": -math.copysign(limit, camera_direction[2]),
            }

        def project_segment(segment, plane_key, plane_values):
            projected = np.asarray(segment, dtype=float).copy()
            if plane_key == "yz":
                projected[:, 0] = plane_values[plane_key]
            elif plane_key == "xz":
                projected[:, 1] = plane_values[plane_key]
            else:
                projected[:, 2] = plane_values[plane_key]
            return projected

        def project_points(points, plane_key, plane_values):
            projected = np.asarray(points, dtype=float).copy()
            if plane_key == "yz":
                projected[:, 0] = plane_values[plane_key]
            elif plane_key == "xz":
                projected[:, 1] = plane_values[plane_key]
            else:
                projected[:, 2] = plane_values[plane_key]
            return projected

        def plane_points_to_2d(points, plane_key):
            points = np.asarray(points, dtype=float)
            if plane_key == "yz":
                return points[:, [1, 2]]
            if plane_key == "xz":
                return points[:, [0, 2]]
            return points[:, [0, 1]]

        def plane_points_from_2d(points_2d, plane_key, plane_value):
            points_2d = np.asarray(points_2d, dtype=float)
            if plane_key == "yz":
                return np.column_stack((
                    np.full(points_2d.shape[0], plane_value),
                    points_2d[:, 0],
                    points_2d[:, 1]))
            if plane_key == "xz":
                return np.column_stack((
                    points_2d[:, 0],
                    np.full(points_2d.shape[0], plane_value),
                    points_2d[:, 1]))
            return np.column_stack((
                points_2d[:, 0],
                points_2d[:, 1],
                np.full(points_2d.shape[0], plane_value)))

        def order_projected_polygon(points_2d):
            points_2d = np.asarray(points_2d, dtype=float)
            if points_2d.shape[0] < 3:
                return points_2d
            try:
                if ConvexHull is not None:
                    hull = ConvexHull(points_2d)
                    return points_2d[hull.vertices]
            except Exception:  # pylint: disable=broad-except
                pass
            center = np.mean(points_2d, axis=0)
            angles = np.arctan2(points_2d[:, 1] - center[1], points_2d[:, 0] - center[0])
            return points_2d[np.argsort(angles)]

        def blend_projection_color(color, plane_key, alpha_scale=1.0):
            base_color = plane_projection_colors[plane_key]
            return (
                0.55 * color[0] + 0.45 * base_color[0],
                0.55 * color[1] + 0.45 * base_color[1],
                0.55 * color[2] + 0.45 * base_color[2],
                min(color[3], base_color[3]) * alpha_scale)

        def add_projected_segments(segments, color, limit, linewidth=1.15):
            if not segments:
                return
            plane_values = projection_plane_values(limit)
            for plane_key in ("yz", "xz", "xy"):
                projected_segments = [
                    project_segment(segment, plane_key, plane_values)
                    for segment in segments
                ]
                blended_color = blend_projection_color(color, plane_key)
                axis.add_collection3d(
                    Line3DCollection(
                        projected_segments,
                        colors=[blended_color] * len(projected_segments),
                        linewidths=linewidth))

        def add_projected_polygon(points, color, limit, edge_linewidth=1.8, fill_alpha=0.14):
            points = ExperimentDataRecorder._unique_vertices(
                np.asarray(points, dtype=float))
            if points.shape[0] < 3:
                return
            plane_values = projection_plane_values(limit)
            for plane_key in ("yz", "xz", "xy"):
                projected = project_points(points, plane_key, plane_values)
                polygon_2d = order_projected_polygon(
                    plane_points_to_2d(projected, plane_key))
                if polygon_2d.shape[0] < 3:
                    continue
                polygon_3d = plane_points_from_2d(
                    polygon_2d,
                    plane_key,
                    plane_values[plane_key])
                face_color = blend_projection_color(color, plane_key, fill_alpha / max(color[3], 1e-9))
                edge_color = blend_projection_color(color, plane_key, 1.0)
                axis.add_collection3d(
                    Poly3DCollection(
                        [polygon_3d],
                        facecolors=[face_color],
                        edgecolors=[edge_color],
                        linewidths=edge_linewidth))

        def add_projected_force_ball(center, radius, color, limit, resolution=72):
            center = np.asarray(center, dtype=float)
            theta = np.linspace(0.0, 2.0 * np.pi, resolution, endpoint=False)
            unit_circle = np.column_stack((np.cos(theta), np.sin(theta))) * float(radius)
            plane_values = projection_plane_values(limit)
            for plane_key in ("yz", "xz", "xy"):
                projected_center = project_points(
                    center.reshape(1, 3),
                    plane_key,
                    plane_values)[0]
                center_2d = plane_points_to_2d(projected_center.reshape(1, 3), plane_key)[0]
                circle_2d = center_2d.reshape(1, 2) + unit_circle
                circle_3d = plane_points_from_2d(
                    circle_2d,
                    plane_key,
                    plane_values[plane_key])
                face_color = blend_projection_color(color, plane_key, 0.42)
                edge_color = blend_projection_color(color, plane_key, 0.95)
                axis.add_collection3d(
                    Poly3DCollection(
                        [circle_3d],
                        facecolors=[face_color],
                        edgecolors=[edge_color],
                        linewidths=1.65))

        def make_surface_wire_segments(x_values, y_values, z_values, stride=3):
            segments = []
            for row_index in range(0, x_values.shape[0], stride):
                segments.append(np.column_stack((
                    x_values[row_index, :],
                    y_values[row_index, :],
                    z_values[row_index, :])))
            for col_index in range(0, x_values.shape[1], stride):
                segments.append(np.column_stack((
                    x_values[:, col_index],
                    y_values[:, col_index],
                    z_values[:, col_index])))
            return segments

        figure = plt.figure(figsize=(7, 7))
        axis = figure.add_subplot(111, projection="3d")
        limit_candidates = [20.0, desired_norm + ball_radius]
        if vertices.size:
            limit_candidates.append(float(np.nanmax(np.abs(vertices))) * 1.18)
        else:
            radius = max(
                1.0,
                float(point.get("force_capacity", {}).get(
                    "value",
                    config.get("fallback_polytope_radius_N", 80.0))))
            limit_candidates.append(radius * 1.25)
        limit_candidates.append(float(np.nanmax(np.abs(desired_force))) + ball_radius)
        limit = max(limit_candidates)

        if vertices.size:
            polytope_segments = self._polytope_wire_segments(vertices, ConvexHull)
            add_projected_polygon(
                vertices,
                (0.05, 0.25, 0.50, 0.50),
                limit)
            add_projected_segments(
                polytope_segments,
                (0.05, 0.25, 0.50, 0.52),
                limit,
                linewidth=1.15)
            self._draw_convex_polytope(axis, vertices, ConvexHull)
        else:
            sphere = self._make_sphere(np.zeros(3), radius, resolution=18)
            add_projected_segments(
                make_surface_wire_segments(sphere[0], sphere[1], sphere[2], stride=3),
                (0.05, 0.25, 0.50, 0.30),
                limit,
                linewidth=1.0)
            axis.plot_surface(
                sphere[0],
                sphere[1],
                sphere[2],
                color="#4da6ff",
                alpha=0.18,
                linewidth=0.0)

        if desired_norm > 1e-9:
            axis.quiver(
                0.0, 0.0, 0.0,
                desired_force[0], desired_force[1], desired_force[2],
                color="#d62728",
                linewidth=2.5,
                arrow_length_ratio=0.12,
                label="Desired force")
        ball = self._make_sphere(desired_force, ball_radius, resolution=14)
        add_projected_segments(
            [(np.zeros(3), desired_force)],
            (0.82, 0.08, 0.08, 0.70),
            limit,
            linewidth=1.35)
        add_projected_force_ball(
            desired_force,
            ball_radius,
            (0.12, 0.58, 0.16, 0.72),
            limit)
        add_projected_segments(
            make_surface_wire_segments(ball[0], ball[1], ball[2], stride=2),
            (0.12, 0.58, 0.16, 0.52),
            limit,
            linewidth=0.95)
        axis.plot_surface(
            ball[0],
            ball[1],
            ball[2],
            color="#2ca02c",
            alpha=0.42,
            linewidth=0.0)

        self._set_equal_axes(axis, limit)
        axis.view_init(elev=view_elev_deg, azim=view_azim_deg)
        axis.set_xlabel("Fx / N")
        axis.set_ylabel("Fy / N")
        axis.set_zlabel("Fz / N")
        axis.set_title(
            "Point {} Residual Force Polytope\nDesired {:.2f} N, radius {:.2f} N".format(
                point.get("index", "?"),
                desired_norm,
                desired_radius))
        axis.grid(True, alpha=0.25)
        figure.tight_layout()
        figure.savefig(path, dpi=180)
        plt.close(figure)

    def _select_animation_rows(self, rows, max_frames):
        if len(rows) <= max_frames:
            return rows
        indices = np.linspace(0, len(rows) - 1, max_frames).astype(int)
        return [rows[index] for index in indices]

    def _render_residual_polytope_animation(
            self, path, buffers, config, preview_path=None, mp4_path=None):
        vertex_rows = buffers["force_polytope_vertices"]
        source = "force_polytope_vertices"
        if vertex_rows:
            frame_rows = self._select_animation_rows(
                vertex_rows,
                max(1, int(config.get("animation_max_frames", 180))))
        else:
            source = "directional_capacity_fallback"
            reference_rows = buffers["desired_force"] or buffers["force_capacity"]
            frame_rows = self._select_animation_rows(
                reference_rows,
                max(1, int(config.get("animation_max_frames", 180))))

        if not frame_rows:
            frame_rows = [{"time": 0.0, "vertices": []}]

        figure = plt.figure(figsize=(7, 7))
        axis = figure.add_subplot(111, projection="3d")
        view_elev_deg = 24.0
        view_azim_deg = -42.0
        view_elev = math.radians(view_elev_deg)
        view_azim = math.radians(view_azim_deg)
        camera_direction = np.array([
            math.cos(view_elev) * math.cos(view_azim),
            math.cos(view_elev) * math.sin(view_azim),
            math.sin(view_elev),
        ])
        plane_projection_colors = {
            "yz": (0.85, 0.18, 0.16, 0.34),
            "xz": (0.12, 0.50, 0.18, 0.34),
            "xy": (0.16, 0.28, 0.72, 0.34),
        }

        try:
            from scipy.spatial import ConvexHull  # pylint: disable=import-outside-toplevel
        except Exception:  # pylint: disable=broad-except
            ConvexHull = None

        def projection_plane_values(limit):
            return {
                "yz": -math.copysign(limit, camera_direction[0]),
                "xz": -math.copysign(limit, camera_direction[1]),
                "xy": -math.copysign(limit, camera_direction[2]),
            }

        def project_segment(segment, plane_key, plane_values):
            projected = np.asarray(segment, dtype=float).copy()
            if plane_key == "yz":
                projected[:, 0] = plane_values[plane_key]
            elif plane_key == "xz":
                projected[:, 1] = plane_values[plane_key]
            else:
                projected[:, 2] = plane_values[plane_key]
            return projected

        def project_points(points, plane_key, plane_values):
            projected = np.asarray(points, dtype=float).copy()
            if plane_key == "yz":
                projected[:, 0] = plane_values[plane_key]
            elif plane_key == "xz":
                projected[:, 1] = plane_values[plane_key]
            else:
                projected[:, 2] = plane_values[plane_key]
            return projected

        def plane_points_to_2d(points, plane_key):
            points = np.asarray(points, dtype=float)
            if plane_key == "yz":
                return points[:, [1, 2]]
            if plane_key == "xz":
                return points[:, [0, 2]]
            return points[:, [0, 1]]

        def plane_points_from_2d(points_2d, plane_key, plane_value):
            points_2d = np.asarray(points_2d, dtype=float)
            if plane_key == "yz":
                return np.column_stack((
                    np.full(points_2d.shape[0], plane_value),
                    points_2d[:, 0],
                    points_2d[:, 1]))
            if plane_key == "xz":
                return np.column_stack((
                    points_2d[:, 0],
                    np.full(points_2d.shape[0], plane_value),
                    points_2d[:, 1]))
            return np.column_stack((
                points_2d[:, 0],
                points_2d[:, 1],
                np.full(points_2d.shape[0], plane_value)))

        def order_projected_polygon(points_2d):
            points_2d = np.asarray(points_2d, dtype=float)
            if points_2d.shape[0] < 3:
                return points_2d
            try:
                if ConvexHull is not None:
                    hull = ConvexHull(points_2d)
                    return points_2d[hull.vertices]
            except Exception:  # pylint: disable=broad-except
                pass
            center = np.mean(points_2d, axis=0)
            angles = np.arctan2(points_2d[:, 1] - center[1], points_2d[:, 0] - center[0])
            return points_2d[np.argsort(angles)]

        def blend_projection_color(color, plane_key, alpha_scale=1.0):
            base_color = plane_projection_colors[plane_key]
            return (
                0.55 * color[0] + 0.45 * base_color[0],
                0.55 * color[1] + 0.45 * base_color[1],
                0.55 * color[2] + 0.45 * base_color[2],
                min(color[3], base_color[3]) * alpha_scale)

        def add_projected_segments(segments, color, limit, linewidth=1.15):
            if not segments:
                return
            plane_values = projection_plane_values(limit)
            for plane_key in ("yz", "xz", "xy"):
                projected_segments = [
                    project_segment(segment, plane_key, plane_values)
                    for segment in segments
                ]
                blended_color = blend_projection_color(color, plane_key)
                axis.add_collection3d(
                    Line3DCollection(
                        projected_segments,
                        colors=[blended_color] * len(projected_segments),
                        linewidths=linewidth))

        def add_projected_polygon(points, color, limit, edge_linewidth=1.8, fill_alpha=0.14):
            points = ExperimentDataRecorder._unique_vertices(
                np.asarray(points, dtype=float))
            if points.shape[0] < 3:
                return
            plane_values = projection_plane_values(limit)
            for plane_key in ("yz", "xz", "xy"):
                projected = project_points(points, plane_key, plane_values)
                polygon_2d = order_projected_polygon(
                    plane_points_to_2d(projected, plane_key))
                if polygon_2d.shape[0] < 3:
                    continue
                polygon_3d = plane_points_from_2d(
                    polygon_2d,
                    plane_key,
                    plane_values[plane_key])
                face_color = blend_projection_color(color, plane_key, fill_alpha / max(color[3], 1e-9))
                edge_color = blend_projection_color(color, plane_key, 1.0)
                axis.add_collection3d(
                    Poly3DCollection(
                        [polygon_3d],
                        facecolors=[face_color],
                        edgecolors=[edge_color],
                        linewidths=edge_linewidth))

        def add_projected_force_ball(center, radius, color, limit, resolution=72):
            center = np.asarray(center, dtype=float)
            theta = np.linspace(0.0, 2.0 * np.pi, resolution, endpoint=False)
            unit_circle = np.column_stack((np.cos(theta), np.sin(theta))) * float(radius)
            plane_values = projection_plane_values(limit)
            for plane_key in ("yz", "xz", "xy"):
                projected_center = project_points(
                    center.reshape(1, 3),
                    plane_key,
                    plane_values)[0]
                center_2d = plane_points_to_2d(projected_center.reshape(1, 3), plane_key)[0]
                circle_2d = center_2d.reshape(1, 2) + unit_circle
                circle_3d = plane_points_from_2d(
                    circle_2d,
                    plane_key,
                    plane_values[plane_key])
                face_color = blend_projection_color(color, plane_key, 0.42)
                edge_color = blend_projection_color(color, plane_key, 0.95)
                axis.add_collection3d(
                    Poly3DCollection(
                        [circle_3d],
                        facecolors=[face_color],
                        edgecolors=[edge_color],
                        linewidths=1.65))

        def make_surface_wire_segments(x_values, y_values, z_values, stride=3):
            segments = []
            for row_index in range(0, x_values.shape[0], stride):
                segments.append(np.column_stack((
                    x_values[row_index, :],
                    y_values[row_index, :],
                    z_values[row_index, :])))
            for col_index in range(0, x_values.shape[1], stride):
                segments.append(np.column_stack((
                    x_values[:, col_index],
                    y_values[:, col_index],
                    z_values[:, col_index])))
            return segments

        def draw(frame):
            axis.clear()
            time_sec = float(frame.get("time", 0.0))
            desired_force = self._interpolate_force_at_time(
                buffers["desired_force"], time_sec)
            desired_norm = np.linalg.norm(desired_force)
            if desired_norm < 1e-9:
                direction = np.array([0.0, 0.0, 1.0], dtype=float)
                desired_force = direction * 1e-9
            else:
                direction = desired_force / desired_norm

            vertices = self._finite_vertices(frame.get("vertices", []))
            desired_force_radius = float(config.get("desired_force_radius_N", 0.0))
            limit_candidates = [20.0, desired_norm + desired_force_radius]
            fallback_shell = None
            if vertices.size:
                limit_candidates.append(float(np.nanmax(np.abs(vertices))) * 1.18)
            else:
                radius = self._interpolate_capacity_at_time(
                    buffers["force_capacity"],
                    time_sec,
                    config.get("fallback_polytope_radius_N", 80.0))
                limit_candidates.append(radius * 1.25)
                fallback_shell = self._make_sphere(np.zeros(3), radius)

            ball_radius = max(0.1, desired_force_radius)
            arrow_tip = desired_force
            limit_candidates.append(float(np.nanmax(np.abs(arrow_tip))) + ball_radius)
            limit = max(limit_candidates)

            if vertices.size:
                polytope_segments = self._polytope_wire_segments(vertices, ConvexHull)
                add_projected_polygon(
                    vertices,
                    (0.05, 0.25, 0.50, 0.50),
                    limit)
                add_projected_segments(
                    polytope_segments,
                    (0.05, 0.25, 0.50, 0.52),
                    limit,
                    linewidth=1.15)
                hull_drawn = self._draw_convex_polytope(axis, vertices, ConvexHull)
                if not hull_drawn:
                    radius = max(1.0, float(np.nanmax(np.linalg.norm(vertices, axis=1))))
                    shell = self._make_sphere(np.zeros(3), radius, resolution=16)
                    add_projected_segments(
                        make_surface_wire_segments(shell[0], shell[1], shell[2], stride=3),
                        (0.05, 0.25, 0.50, 0.28),
                        limit,
                        linewidth=1.0)
                    axis.plot_surface(
                        shell[0],
                        shell[1],
                        shell[2],
                        color="#4da6ff",
                        alpha=0.08,
                        linewidth=0.0)
            else:
                sphere = fallback_shell
                add_projected_segments(
                    make_surface_wire_segments(sphere[0], sphere[1], sphere[2], stride=3),
                    (0.05, 0.25, 0.50, 0.30),
                    limit,
                    linewidth=1.0)
                axis.plot_surface(
                    sphere[0],
                    sphere[1],
                    sphere[2],
                    color="#4da6ff",
                    alpha=0.18,
                    linewidth=0.0)

            add_projected_segments(
                [(np.zeros(3), arrow_tip)],
                (0.82, 0.08, 0.08, 0.70),
                limit,
                linewidth=1.35)
            axis.quiver(
                0.0, 0.0, 0.0,
                arrow_tip[0], arrow_tip[1], arrow_tip[2],
                color="#d62728",
                linewidth=2.5,
                arrow_length_ratio=0.12)
            ball = self._make_sphere(arrow_tip, ball_radius, resolution=12)
            add_projected_force_ball(
                arrow_tip,
                ball_radius,
                (0.12, 0.58, 0.16, 0.72),
                limit)
            add_projected_segments(
                make_surface_wire_segments(ball[0], ball[1], ball[2], stride=2),
                (0.12, 0.58, 0.16, 0.52),
                limit,
                linewidth=0.95)
            axis.plot_surface(
                ball[0],
                ball[1],
                ball[2],
                color="#2ca02c",
                alpha=0.55,
                linewidth=0.0)

            self._set_equal_axes(axis, limit)
            axis.set_proj_type("persp")
            axis.view_init(elev=view_elev_deg, azim=view_azim_deg)
            axis.set_xlabel("Fx / N")
            axis.set_ylabel("Fy / N")
            axis.set_zlabel("Fz / N")
            axis.set_title(
                "Residual Force Polytope  t={:.2f}s\nsource={}".format(
                    time_sec,
                    source))
            axis.grid(True, alpha=0.25)

        animation = FuncAnimation(
            figure,
            draw,
            frames=frame_rows,
            interval=1000.0 / max(1.0, float(config.get("animation_fps", 10.0))),
            repeat=False)
        if preview_path is not None:
            draw(frame_rows[len(frame_rows) // 2])
            figure.tight_layout()
            figure.savefig(preview_path, dpi=180)
        animation.save(
            str(path),
            writer=PillowWriter(fps=max(1, int(config.get("animation_fps", 10.0)))))
        if mp4_path is not None:
            try:
                from matplotlib.animation import FFMpegWriter  # pylint: disable=import-outside-toplevel
                animation.save(
                    str(mp4_path),
                    writer=FFMpegWriter(
                        fps=max(1, int(config.get("animation_fps", 10.0))),
                        bitrate=1800))
            except Exception as exc:  # pylint: disable=broad-except
                rospy.logwarn(
                    "experiment_data_recorder: MP4 residual polytope export skipped: %s",
                    exc)
                try:
                    if mp4_path.exists():
                        mp4_path.unlink()
                except OSError:
                    pass
        plt.close(figure)
        return source

    def _render_manipulability_ellipsoid_animation(
            self, path, buffers, config, preview_path=None, mp4_path=None):
        ellipsoid_rows = buffers["manipulability_ellipsoid"]
        frame_rows = self._select_animation_rows(
            ellipsoid_rows,
            max(1, int(config.get("animation_max_frames", 180))))
        if not frame_rows:
            frame_rows = [{
                "time": 0.0,
                "matrix": np.eye(3).tolist(),
                "eigenvalues": [1.0, 1.0, 1.0],
            }]

        max_radius = 1e-3
        for row in frame_rows:
            matrix = np.asarray(row.get("matrix", np.eye(3)), dtype=float)
            if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
                continue
            eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
            radius = float(np.sqrt(max(0.0, np.nanmax(eigenvalues))))
            if math.isfinite(radius):
                max_radius = max(max_radius, radius)
        axis_limit = max(1e-3, max_radius * 1.25)

        figure = plt.figure(figsize=(7, 7))
        axis = figure.add_subplot(111, projection="3d")
        view_elev_deg = 24.0
        view_azim_deg = -42.0
        view_elev = math.radians(view_elev_deg)
        view_azim = math.radians(view_azim_deg)
        camera_direction = np.array([
            math.cos(view_elev) * math.cos(view_azim),
            math.cos(view_elev) * math.sin(view_azim),
            math.sin(view_elev),
        ])
        near_color = np.array([0.98, 0.56, 0.18, 0.95])
        far_color = np.array([0.03, 0.12, 0.28, 0.72])
        projection_colors = {
            "yz": (0.95, 0.20, 0.18, 0.38),
            "xz": (0.12, 0.55, 0.22, 0.38),
            "xy": (0.18, 0.32, 0.78, 0.38),
        }

        def add_coordinate_plane_projections(x_values, y_values, z_values):
            x_plane = -math.copysign(axis_limit, camera_direction[0])
            y_plane = -math.copysign(axis_limit, camera_direction[1])
            z_plane = -math.copysign(axis_limit, camera_direction[2])

            projection_specs = [
                (
                    "yz",
                    [
                        np.column_stack((
                            np.full_like(y_values[row_index, :], x_plane),
                            y_values[row_index, :],
                            z_values[row_index, :]))
                        for row_index in range(y_values.shape[0])
                    ] + [
                        np.column_stack((
                            np.full_like(y_values[:, col_index], x_plane),
                            y_values[:, col_index],
                            z_values[:, col_index]))
                        for col_index in range(y_values.shape[1])
                    ]),
                (
                    "xz",
                    [
                        np.column_stack((
                            x_values[row_index, :],
                            np.full_like(x_values[row_index, :], y_plane),
                            z_values[row_index, :]))
                        for row_index in range(x_values.shape[0])
                    ] + [
                        np.column_stack((
                            x_values[:, col_index],
                            np.full_like(x_values[:, col_index], y_plane),
                            z_values[:, col_index]))
                        for col_index in range(x_values.shape[1])
                    ]),
                (
                    "xy",
                    [
                        np.column_stack((
                            x_values[row_index, :],
                            y_values[row_index, :],
                            np.full_like(x_values[row_index, :], z_plane)))
                        for row_index in range(x_values.shape[0])
                    ] + [
                        np.column_stack((
                            x_values[:, col_index],
                            y_values[:, col_index],
                            np.full_like(x_values[:, col_index], z_plane)))
                        for col_index in range(x_values.shape[1])
                    ]),
            ]

            for plane_key, segments in projection_specs:
                axis.add_collection3d(
                    Line3DCollection(
                        segments,
                        colors=[projection_colors[plane_key]] * len(segments),
                        linewidths=0.42,
                        linestyles="solid"))

        def draw(frame):
            axis.clear()
            time_sec = float(frame.get("time", 0.0))
            matrix = np.asarray(frame.get("matrix", np.eye(3)), dtype=float)
            x, y, z, radii, eigenvectors = self._make_ellipsoid(matrix)

            add_coordinate_plane_projections(x, y, z)

            axis.plot_surface(
                x,
                y,
                z,
                color=(0.30, 0.62, 0.95, 0.18),
                linewidth=0.0,
                antialiased=True,
                shade=False)

            depth = (
                camera_direction[0] * x +
                camera_direction[1] * y +
                camera_direction[2] * z)
            depth_min = float(np.nanmin(depth))
            depth_max = float(np.nanmax(depth))
            depth_span = max(1e-9, depth_max - depth_min)
            line_segments = []
            line_colors = []
            for line_index in range(x.shape[0]):
                line = np.column_stack((x[line_index, :], y[line_index, :], z[line_index, :]))
                closeness = (float(np.nanmean(depth[line_index, :])) - depth_min) / depth_span
                line_segments.append(line)
                line_colors.append(far_color * (1.0 - closeness) + near_color * closeness)
            for line_index in range(x.shape[1]):
                line = np.column_stack((x[:, line_index], y[:, line_index], z[:, line_index]))
                closeness = (float(np.nanmean(depth[:, line_index])) - depth_min) / depth_span
                line_segments.append(line)
                line_colors.append(far_color * (1.0 - closeness) + near_color * closeness)
            axis.add_collection3d(
                Line3DCollection(
                    line_segments,
                    colors=line_colors,
                    linewidths=0.8))

            colors = ["#d62728", "#2ca02c", "#ff7f0e"]
            labels = ["major", "middle", "minor"]
            order = np.argsort(radii)[::-1]
            for draw_index, axis_index in enumerate(order):
                radius = float(radii[axis_index])
                direction = eigenvectors[:, axis_index]
                endpoint = direction * radius
                axis.plot(
                    [-endpoint[0], endpoint[0]],
                    [-endpoint[1], endpoint[1]],
                    [-endpoint[2], endpoint[2]],
                    color=colors[draw_index],
                    linewidth=2.2,
                    label="{} axis {:.4g}".format(labels[draw_index], radius))

            self._set_equal_axes(axis, axis_limit)
            axis.set_proj_type("persp")
            axis.view_init(elev=view_elev_deg, azim=view_azim_deg)
            axis.set_xlabel("vx direction")
            axis.set_ylabel("vy direction")
            axis.set_zlabel("vz direction")
            axis.set_title("Manipulability Ellipsoid  t={:.2f}s".format(time_sec))
            axis.grid(True, alpha=0.25)
            axis.xaxis.pane.set_facecolor((0.96, 0.98, 1.0, 0.30))
            axis.yaxis.pane.set_facecolor((0.96, 0.98, 1.0, 0.30))
            axis.zaxis.pane.set_facecolor((0.96, 0.98, 1.0, 0.30))
            axis.legend(loc="upper right")

        animation = FuncAnimation(
            figure,
            draw,
            frames=frame_rows,
            interval=1000.0 / max(1.0, float(config.get("animation_fps", 10.0))),
            repeat=False)
        if preview_path is not None:
            draw(frame_rows[len(frame_rows) // 2])
            figure.tight_layout()
            figure.savefig(preview_path, dpi=180)
        animation.save(
            str(path),
            writer=PillowWriter(fps=max(1, int(config.get("animation_fps", 10.0)))))
        if mp4_path is not None:
            try:
                from matplotlib.animation import FFMpegWriter  # pylint: disable=import-outside-toplevel
                animation.save(
                    str(mp4_path),
                    writer=FFMpegWriter(
                        fps=max(1, int(config.get("animation_fps", 10.0))),
                        bitrate=1800))
            except Exception as exc:  # pylint: disable=broad-except
                rospy.logwarn(
                    "experiment_data_recorder: MP4 manipulability ellipsoid export skipped: %s",
                    exc)
                try:
                    if mp4_path.exists():
                        mp4_path.unlink()
                except OSError:
                    pass
        plt.close(figure)


def main():
    rospy.init_node("moca_plot_data_recorder")
    ExperimentDataRecorder()
    rospy.spin()


if __name__ == "__main__":
    main()
