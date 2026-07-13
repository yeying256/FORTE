#!/usr/bin/env python3

import os
import threading

import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3Stamped
from matplotlib.lines import Line2D
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Int32


DEFAULT_OUTPUT_DIR = (
    os.path.join(os.path.expanduser("~"), ".ros", "moca_polytope")
)


class ForceSweepComparisonPlotter:
    def __init__(self):
        self.run_state_topic = rospy.get_param(
            "~comparison_run_state_topic", "/comparison_run_state"
        )
        self.manipulability_topic = rospy.get_param(
            "~manipulability_topic", "/wb_cart_imp_controller/manipulability"
        )
        self.directional_force_capacity_topic = rospy.get_param(
            "~directional_force_capacity_topic",
            "/wb_cart_imp_controller/directional_force_capacity",
        )
        self.desired_force_topic = rospy.get_param(
            "~desired_force_topic", "/wb_cart_imp_controller/desired_force"
        )
        self.pose_topic = rospy.get_param("~pose_topic", "/wb_cart_imp_controller/moca_pose")
        self.pose_target_topic = rospy.get_param(
            "~pose_target_topic", "/wb_cart_imp_controller/moca_pose_target"
        )
        self.state_topic = rospy.get_param("~state_topic", "/wb_cart_imp_controller/moca_state")
        self.state_target_topic = rospy.get_param(
            "~state_target_topic", "/wb_cart_imp_controller/moca_state_target"
        )
        self.base_cmd_velocity_topic = rospy.get_param(
            "~base_cmd_velocity_topic", "/wb_cart_imp_controller/base_cmd_velocity"
        )
        self.first_force_magnitude = rospy.get_param(
            "~comparison_first_lift_force_magnitude", 10.0
        )
        self.second_force_magnitude = rospy.get_param(
            "~comparison_second_lift_force_magnitude", 80.0
        )
        self.task_force_mode = rospy.get_param("~task_force_mode", "legacy_fixed")
        self.show_window = rospy.get_param("~show_window", True)
        self.render_rate_hz = rospy.get_param("~render_rate_hz", 5.0)
        self.combined_output_png_path = rospy.get_param(
            "~combined_output_png_path",
            os.path.join(DEFAULT_OUTPUT_DIR, "force_sweep_comparison.png"),
        )
        self.manipulability_output_png_path = rospy.get_param(
            "~manipulability_output_png_path",
            os.path.join(
                DEFAULT_OUTPUT_DIR,
                "force_sweep_manipulability_comparison.png",
            ),
        )
        self.force_output_png_path = rospy.get_param(
            "~force_output_png_path",
            os.path.join(
                DEFAULT_OUTPUT_DIR,
                "force_sweep_directional_force_comparison.png",
            ),
        )

        self.lock = threading.Lock()
        self.current_run_state = 0
        self.saved = False
        self.frozen = False
        self.run_data = {
            1: {
                "start_time": None,
                "manipulability": [],
                "force_capacity": [],
                "desired_force": [],
                "pose_actual": [],
                "pose_target": [],
                "orientation_actual": [],
                "orientation_target": [],
                "arm_actual": [],
                "arm_target": [],
                "base_position": [],
                "base_cmd_velocity": [],
            },
            2: {
                "start_time": None,
                "manipulability": [],
                "force_capacity": [],
                "desired_force": [],
                "pose_actual": [],
                "pose_target": [],
                "orientation_actual": [],
                "orientation_target": [],
                "arm_actual": [],
                "arm_target": [],
                "base_position": [],
                "base_cmd_velocity": [],
            },
        }

        self.figure, self.axes = plt.subplots(6, 1, figsize=(12, 24), sharex=False)
        self.joint_actual_axes = [self.axes[2].twinx(), self.axes[3].twinx()]
        self.figure.suptitle("Force Sweep Comparison", fontsize=14)
        for axis in self.axes:
            axis.grid(True, linestyle="--", alpha=0.3)
        if self.show_window and os.environ.get("DISPLAY"):
            plt.ion()
            plt.show(block=False)
            self.figure.canvas.draw_idle()
            self.figure.canvas.flush_events()
            plt.pause(0.001)

        rospy.Subscriber(self.run_state_topic, Int32, self.run_state_callback, queue_size=20)
        rospy.Subscriber(
            self.manipulability_topic, Float64, self.manipulability_callback, queue_size=200
        )
        rospy.Subscriber(
            self.directional_force_capacity_topic,
            Float64,
            self.force_capacity_callback,
            queue_size=200,
        )
        rospy.Subscriber(
            self.desired_force_topic,
            Vector3Stamped,
            self.desired_force_callback,
            queue_size=200,
        )
        rospy.Subscriber(self.pose_target_topic, PoseStamped, self.target_pose_callback, queue_size=200)
        rospy.Subscriber(self.pose_topic, PoseStamped, self.pose_callback, queue_size=200)
        rospy.Subscriber(self.state_topic, JointState, self.state_callback, queue_size=200)
        rospy.Subscriber(
            self.state_target_topic, JointState, self.target_state_callback, queue_size=200
        )
        rospy.Subscriber(
            self.base_cmd_velocity_topic,
            TwistStamped,
            self.base_cmd_velocity_callback,
            queue_size=200,
        )

        rospy.loginfo(
            "force_sweep_comparison_plotter: listening to %s, %s, %s, %s, %s, %s, %s, %s, %s",
            self.run_state_topic,
            self.manipulability_topic,
            self.directional_force_capacity_topic,
            self.desired_force_topic,
            self.pose_topic,
            self.pose_target_topic,
            self.state_topic,
            self.state_target_topic,
            self.base_cmd_velocity_topic,
        )

    def normalize_quaternion_xyzw(self, quat):
        quat = np.asarray(quat, dtype=float)
        norm = np.linalg.norm(quat)
        if norm < 1e-12:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        return quat / norm

    def quaternion_multiply_xyzw(self, q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ],
            dtype=float,
        )

    def quaternion_inverse_xyzw(self, quat):
        x, y, z, w = self.normalize_quaternion_xyzw(quat)
        return np.array([-x, -y, -z, w], dtype=float)

    def quaternion_to_axis_angle_error(self, actual_quat, target_quat):
        actual_quat = self.normalize_quaternion_xyzw(actual_quat)
        target_quat = self.normalize_quaternion_xyzw(target_quat)

        if float(np.dot(actual_quat, target_quat)) < 0.0:
            actual_quat = -actual_quat

        error_quat = self.quaternion_multiply_xyzw(
            actual_quat,
            self.quaternion_inverse_xyzw(target_quat),
        )
        error_quat = self.normalize_quaternion_xyzw(error_quat)

        vector = error_quat[:3]
        scalar = float(np.clip(error_quat[3], -1.0, 1.0))
        vector_norm = np.linalg.norm(vector)
        if vector_norm < 1e-12:
            return np.zeros(3, dtype=float)

        angle = 2.0 * np.arctan2(vector_norm, scalar)
        axis = vector / vector_norm
        return axis * angle

    def run_state_callback(self, msg):
        with self.lock:
            self.current_run_state = int(msg.data)

    def append_metric(self, key, value):
        stamp = rospy.Time.now().to_sec()
        with self.lock:
            if self.current_run_state not in self.run_data:
                return

            run_entry = self.run_data[self.current_run_state]
            if run_entry["start_time"] is None:
                run_entry["start_time"] = stamp

            run_entry[key].append((stamp - run_entry["start_time"], float(value)))

    def manipulability_callback(self, msg):
        self.append_metric("manipulability", msg.data)

    def force_capacity_callback(self, msg):
        self.append_metric("force_capacity", msg.data)

    def desired_force_callback(self, msg):
        stamp = rospy.Time.now().to_sec()
        with self.lock:
            if self.current_run_state not in self.run_data:
                return

            run_entry = self.run_data[self.current_run_state]
            if run_entry["start_time"] is None:
                run_entry["start_time"] = stamp

            relative_time = stamp - run_entry["start_time"]
            desired_force = np.asarray(
                [msg.vector.x, msg.vector.y, msg.vector.z], dtype=float
            )
            run_entry["desired_force"].append((relative_time, desired_force))

    def target_state_callback(self, msg):
        if len(msg.position) < 10:
            return
        stamp = self.append_state_target(np.asarray(msg.position[3:10], dtype=float))
        return stamp

    def target_pose_callback(self, msg):
        target_position = np.asarray(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=float
        )
        target_orientation = np.asarray(
            [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ],
            dtype=float,
        )
        stamp = rospy.Time.now().to_sec()
        with self.lock:
            if self.current_run_state not in self.run_data:
                return
            run_entry = self.run_data[self.current_run_state]
            if run_entry["start_time"] is None:
                run_entry["start_time"] = stamp
            run_entry["latest_pose_target"] = target_position
            run_entry["latest_orientation_target"] = target_orientation

    def pose_callback(self, msg):
        stamp = rospy.Time.now().to_sec()
        with self.lock:
            if self.current_run_state not in self.run_data:
                return

            run_entry = self.run_data[self.current_run_state]
            if run_entry["start_time"] is None:
                run_entry["start_time"] = stamp

            relative_time = stamp - run_entry["start_time"]
            actual_position = np.asarray(
                [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=float
            )
            actual_orientation = np.asarray(
                [
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                    msg.pose.orientation.w,
                ],
                dtype=float,
            )
            target_position = np.asarray(
                run_entry.get("latest_pose_target", actual_position.copy()), dtype=float
            )
            target_orientation = np.asarray(
                run_entry.get("latest_orientation_target", actual_orientation.copy()),
                dtype=float,
            )
            run_entry["pose_actual"].append((relative_time, actual_position))
            run_entry["pose_target"].append((relative_time, target_position))
            run_entry["orientation_actual"].append((relative_time, actual_orientation))
            run_entry["orientation_target"].append((relative_time, target_orientation))

    def append_state_target(self, target_value):
        stamp = rospy.Time.now().to_sec()
        with self.lock:
            if self.current_run_state not in self.run_data:
                return None
            run_entry = self.run_data[self.current_run_state]
            if run_entry["start_time"] is None:
                run_entry["start_time"] = stamp
            run_entry["latest_arm_target"] = np.asarray(target_value, dtype=float)
        return stamp

    def state_callback(self, msg):
        if len(msg.position) < 10:
            return
        stamp = rospy.Time.now().to_sec()
        with self.lock:
            if self.current_run_state not in self.run_data:
                return

            run_entry = self.run_data[self.current_run_state]
            if run_entry["start_time"] is None:
                run_entry["start_time"] = stamp

            relative_time = stamp - run_entry["start_time"]
            base_position = np.asarray(msg.position[0:3], dtype=float)
            arm_actual = np.asarray(msg.position[3:10], dtype=float)
            arm_target = np.asarray(
                run_entry.get("latest_arm_target", arm_actual.copy()), dtype=float
            )
            run_entry["base_position"].append((relative_time, base_position))
            run_entry["arm_actual"].append((relative_time, arm_actual))
            run_entry["arm_target"].append((relative_time, arm_target))

    def base_cmd_velocity_callback(self, msg):
        stamp = rospy.Time.now().to_sec()
        with self.lock:
            if self.current_run_state not in self.run_data:
                return

            run_entry = self.run_data[self.current_run_state]
            if run_entry["start_time"] is None:
                run_entry["start_time"] = stamp

            relative_time = stamp - run_entry["start_time"]
            base_cmd_velocity = np.asarray(
                [
                    msg.twist.linear.x,
                    msg.twist.linear.y,
                    msg.twist.angular.z,
                ],
                dtype=float,
            )
            run_entry["base_cmd_velocity"].append((relative_time, base_cmd_velocity))

    def save_figure(self, fig, output_path):
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        fig.savefig(output_path, dpi=180)

    def save_outputs(self, run_data):
        self.save_figure(self.figure, self.combined_output_png_path)

        manip_fig, manip_ax = plt.subplots(1, 1, figsize=(10, 5))
        self.draw_metric_axis(
            manip_ax,
            run_data,
            metric_key="manipulability",
            title="Manipulability Comparison",
            ylabel="manipulability",
        )
        manip_fig.tight_layout()
        self.save_figure(manip_fig, self.manipulability_output_png_path)
        plt.close(manip_fig)

        force_fig, force_ax = plt.subplots(1, 1, figsize=(10, 5))
        self.draw_force_axis(force_ax, run_data)
        force_ax.set_xlabel("time [s]")
        force_fig.tight_layout()
        self.save_figure(force_fig, self.force_output_png_path)
        plt.close(force_fig)

        rospy.loginfo(
            "force_sweep_comparison_plotter: saved %s, %s, %s",
            self.combined_output_png_path,
            self.manipulability_output_png_path,
            self.force_output_png_path,
        )

    def draw_metric_axis(self, axis, run_data, metric_key, title, ylabel):
        axis.clear()
        run_specs = (
            (1, self.run_label(1), "tab:blue"),
            (2, self.run_label(2), "tab:red"),
        )

        for run_index, run_label, color in run_specs:
            samples = run_data[run_index][metric_key]
            if not samples:
                continue

            times = [sample[0] for sample in samples]
            values = [sample[1] for sample in samples]
            axis.plot(
                times,
                values,
                color=color,
                linewidth=2.0,
                label=run_label,
            )

        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(True, linestyle="--", alpha=0.3)
        axis.legend(loc="best")

    def run_label(self, run_index):
        if self.task_force_mode == "legacy_fixed":
            force_value = (
                self.first_force_magnitude if run_index == 1 else self.second_force_magnitude
            )
            return f"{force_value:.0f} N"
        return f"run {run_index}"

    def draw_force_axis(self, axis, run_data):
        axis.clear()
        run_specs = (
            (1, self.run_label(1), "tab:blue"),
            (2, self.run_label(2), "tab:red"),
        )

        for run_index, run_label, color in run_specs:
            force_samples = run_data[run_index]["force_capacity"]
            if force_samples:
                force_times = [sample[0] for sample in force_samples]
                force_values = [sample[1] for sample in force_samples]
                axis.plot(
                    force_times,
                    force_values,
                    color=color,
                    linewidth=2.0,
                    label=f"{run_label} capacity",
                )

            desired_force_samples = run_data[run_index]["desired_force"]
            if desired_force_samples:
                desired_force_times = [sample[0] for sample in desired_force_samples]
                desired_force_values = [
                    float(np.linalg.norm(sample[1])) for sample in desired_force_samples
                ]
                axis.plot(
                    desired_force_times,
                    desired_force_values,
                    linestyle="--",
                    color=color,
                    linewidth=1.8,
                    label=f"{run_label} desired",
                )

        axis.set_title("Directional Force Capability and Desired Force Comparison")
        axis.set_ylabel("force [N]")
        axis.grid(True, linestyle="--", alpha=0.3)
        axis.legend(loc="best")

    def draw_joint_axis(self, target_axis, actual_axis, run_data, run_index, force_value):
        target_axis.clear()
        actual_axis.clear()
        joint_colors = plt.cm.tab10(np.linspace(0.0, 1.0, 7))
        actual_samples = run_data[run_index]["arm_actual"]
        target_samples = run_data[run_index]["arm_target"]
        if actual_samples and target_samples:
            actual_times = np.asarray([sample[0] for sample in actual_samples], dtype=float)
            actual_values = np.asarray([sample[1] for sample in actual_samples], dtype=float)
            target_times = np.asarray([sample[0] for sample in target_samples], dtype=float)
            target_values = np.asarray([sample[1] for sample in target_samples], dtype=float)

            actual_reference = actual_values[0:1, :]
            target_reference = target_values[0:1, :]
            actual_values = actual_values - actual_reference
            target_values = target_values - target_reference

            for joint_index in range(7):
                color = joint_colors[joint_index]
                actual_axis.plot(
                    actual_times,
                    actual_values[:, joint_index],
                    color=color,
                    linewidth=1.8,
                )
                target_axis.plot(
                    target_times,
                    target_values[:, joint_index],
                    linestyle="--",
                    color=color,
                    linewidth=1.5,
                )

            target_min = float(np.min(target_values))
            target_max = float(np.max(target_values))
            target_scale = max(abs(target_min), abs(target_max), 1e-3)
            target_axis.set_ylim(
                target_min - 0.08 * target_scale,
                target_max + 0.08 * target_scale,
            )

            actual_min = float(np.min(actual_values))
            actual_max = float(np.max(actual_values))
            actual_span = max(actual_max - actual_min, 1e-5)
            actual_pad = max(5e-4, 0.12 * actual_span)
            actual_axis.set_ylim(actual_min - actual_pad, actual_max + actual_pad)

        joint_handles = [
            Line2D([0], [0], color=joint_colors[i], linewidth=2.0) for i in range(7)
        ]
        joint_labels = [f"joint {i + 1}" for i in range(7)]
        joint_legend = target_axis.legend(
            joint_handles, joint_labels, loc="upper left", ncol=4, fontsize=9
        )
        target_axis.add_artist(joint_legend)

        style_handles = [
            Line2D([0], [0], color="black", linewidth=2.0, linestyle="-"),
            Line2D([0], [0], color="black", linewidth=2.0, linestyle="--"),
        ]
        style_labels = [
            "actual",
            "nullspace target",
        ]
        target_axis.legend(style_handles, style_labels, loc="upper right", fontsize=9)
        run_label = self.run_label(run_index)
        target_axis.set_title(
            f"Arm Joint Delta from Run Start: Actual vs Nullspace Target ({run_label})"
        )
        target_axis.set_ylabel("target joint delta [rad]")
        actual_axis.set_ylabel("actual joint delta [rad]")
        target_axis.set_xlabel("time [s]")
        target_axis.grid(True, linestyle="--", alpha=0.3)
        target_axis.tick_params(axis="y", colors="tab:gray")
        actual_axis.tick_params(axis="y", colors="black")
        target_axis.yaxis.label.set_color("tab:gray")
        actual_axis.yaxis.label.set_color("black")

    def draw_position_error_axis(self, axis, run_data, run_index, force_value):
        axis.clear()
        actual_samples = run_data[run_index]["pose_actual"]
        target_samples = run_data[run_index]["pose_target"]
        if actual_samples and target_samples:
            actual_times = np.asarray([sample[0] for sample in actual_samples], dtype=float)
            actual_values = np.asarray([sample[1] for sample in actual_samples], dtype=float)
            target_values = np.asarray([sample[1] for sample in target_samples], dtype=float)
            position_error = actual_values - target_values
            error_norm = np.linalg.norm(position_error, axis=1)

            axis.plot(
                actual_times,
                position_error[:, 0],
                color="tab:red",
                linewidth=1.8,
                label="x error",
            )
            axis.plot(
                actual_times,
                position_error[:, 1],
                color="tab:green",
                linewidth=1.8,
                label="y error",
            )
            axis.plot(
                actual_times,
                position_error[:, 2],
                color="tab:blue",
                linewidth=1.8,
                label="z error",
            )
            axis.plot(
                actual_times,
                error_norm,
                color="black",
                linewidth=2.0,
                linestyle="--",
                label="error norm",
            )

        axis.set_title(f"End-Effector Position Error ({self.run_label(run_index)})")
        axis.set_ylabel("position error [m]")
        axis.set_xlabel("time [s]")
        axis.grid(True, linestyle="--", alpha=0.3)
        axis.legend(loc="best", fontsize=9)

    def draw_orientation_error_axis(self, axis, run_data, run_index, force_value):
        axis.clear()
        actual_samples = run_data[run_index]["orientation_actual"]
        target_samples = run_data[run_index]["orientation_target"]
        if actual_samples and target_samples:
            actual_times = np.asarray([sample[0] for sample in actual_samples], dtype=float)
            actual_values = np.asarray([sample[1] for sample in actual_samples], dtype=float)
            target_values = np.asarray([sample[1] for sample in target_samples], dtype=float)
            axis_angle_error = np.asarray(
                [
                    self.quaternion_to_axis_angle_error(actual_quat, target_quat)
                    for actual_quat, target_quat in zip(actual_values, target_values)
                ],
                dtype=float,
            )
            error_norm = np.linalg.norm(axis_angle_error, axis=1)

            axis.plot(
                actual_times,
                axis_angle_error[:, 0],
                color="tab:red",
                linewidth=1.8,
                label="rx error",
            )
            axis.plot(
                actual_times,
                axis_angle_error[:, 1],
                color="tab:green",
                linewidth=1.8,
                label="ry error",
            )
            axis.plot(
                actual_times,
                axis_angle_error[:, 2],
                color="tab:blue",
                linewidth=1.8,
                label="rz error",
            )
            axis.plot(
                actual_times,
                error_norm,
                color="black",
                linewidth=2.0,
                linestyle="--",
                label="error norm",
            )

        axis.set_title(
            f"End-Effector Orientation Axis-Angle Error ({self.run_label(run_index)})"
        )
        axis.set_ylabel("axis-angle error [rad]")
        axis.set_xlabel("time [s]")
        axis.grid(True, linestyle="--", alpha=0.3)
        axis.legend(loc="best", fontsize=9)

    def draw_base_motion_axis(self, velocity_axis, position_axis, run_data, run_index, force_value):
        velocity_axis.clear()
        position_axis.clear()
        velocity_samples = run_data[run_index]["base_cmd_velocity"]
        position_samples = run_data[run_index]["base_position"]

        command_handles = []
        position_handles = []

        if velocity_samples:
            velocity_times = np.asarray([sample[0] for sample in velocity_samples], dtype=float)
            velocity_values = np.asarray([sample[1] for sample in velocity_samples], dtype=float)
            command_handles.append(
                velocity_axis.plot(
                    velocity_times,
                    velocity_values[:, 0],
                    color="tab:red",
                    linewidth=1.8,
                    label="cmd vx",
                )[0]
            )
            command_handles.append(
                velocity_axis.plot(
                    velocity_times,
                    velocity_values[:, 1],
                    color="tab:green",
                    linewidth=1.8,
                    label="cmd vy",
                )[0]
            )
            command_handles.append(
                velocity_axis.plot(
                    velocity_times,
                    velocity_values[:, 2],
                    color="tab:blue",
                    linewidth=1.8,
                    label="cmd wz",
                )[0]
            )

        if position_samples:
            position_times = np.asarray([sample[0] for sample in position_samples], dtype=float)
            position_values = np.asarray([sample[1] for sample in position_samples], dtype=float)
            position_reference = position_values[0:1, :]
            position_delta = position_values - position_reference
            position_handles.append(
                position_axis.plot(
                    position_times,
                    position_delta[:, 0],
                    color="tab:red",
                    linewidth=1.8,
                    linestyle="--",
                    label="base x delta",
                )[0]
            )
            position_handles.append(
                position_axis.plot(
                    position_times,
                    position_delta[:, 1],
                    color="tab:green",
                    linewidth=1.8,
                    linestyle="--",
                    label="base y delta",
                )[0]
            )
            position_handles.append(
                position_axis.plot(
                    position_times,
                    position_delta[:, 2],
                    color="tab:blue",
                    linewidth=1.8,
                    linestyle="--",
                    label="base yaw delta",
                )[0]
            )

        velocity_axis.set_title(
            f"Base Command Velocity vs Actual Base Position Delta ({self.run_label(run_index)})"
        )
        velocity_axis.set_ylabel("command velocity [m/s, rad/s]")
        position_axis.set_ylabel("base delta [m, rad]")
        velocity_axis.set_xlabel("time [s]")
        velocity_axis.grid(True, linestyle="--", alpha=0.3)
        velocity_axis.tick_params(axis="y", colors="black")
        position_axis.tick_params(axis="y", colors="tab:gray")
        velocity_axis.yaxis.label.set_color("black")
        position_axis.yaxis.label.set_color("tab:gray")

        legend_handles = command_handles + position_handles
        legend_labels = [handle.get_label() for handle in legend_handles]
        if legend_handles:
            velocity_axis.legend(legend_handles, legend_labels, loc="best", fontsize=9)

    def draw(self):
        with self.lock:
            run_data = {
                1: {
                    "manipulability": list(self.run_data[1]["manipulability"]),
                    "force_capacity": list(self.run_data[1]["force_capacity"]),
                    "desired_force": list(self.run_data[1]["desired_force"]),
                    "pose_actual": list(self.run_data[1]["pose_actual"]),
                    "pose_target": list(self.run_data[1]["pose_target"]),
                    "orientation_actual": list(self.run_data[1]["orientation_actual"]),
                    "orientation_target": list(self.run_data[1]["orientation_target"]),
                    "arm_actual": list(self.run_data[1]["arm_actual"]),
                    "arm_target": list(self.run_data[1]["arm_target"]),
                    "base_position": list(self.run_data[1]["base_position"]),
                    "base_cmd_velocity": list(self.run_data[1]["base_cmd_velocity"]),
                },
                2: {
                    "manipulability": list(self.run_data[2]["manipulability"]),
                    "force_capacity": list(self.run_data[2]["force_capacity"]),
                    "desired_force": list(self.run_data[2]["desired_force"]),
                    "pose_actual": list(self.run_data[2]["pose_actual"]),
                    "pose_target": list(self.run_data[2]["pose_target"]),
                    "orientation_actual": list(self.run_data[2]["orientation_actual"]),
                    "orientation_target": list(self.run_data[2]["orientation_target"]),
                    "arm_actual": list(self.run_data[2]["arm_actual"]),
                    "arm_target": list(self.run_data[2]["arm_target"]),
                    "base_position": list(self.run_data[2]["base_position"]),
                    "base_cmd_velocity": list(self.run_data[2]["base_cmd_velocity"]),
                },
                "state": self.current_run_state,
            }

        if (
            not run_data[1]["manipulability"]
            and not run_data[2]["manipulability"]
            and not run_data[1]["force_capacity"]
            and not run_data[2]["force_capacity"]
            and not run_data[1]["desired_force"]
            and not run_data[2]["desired_force"]
            and not run_data[1]["pose_actual"]
            and not run_data[2]["pose_actual"]
            and not run_data[1]["orientation_actual"]
            and not run_data[2]["orientation_actual"]
            and not run_data[1]["arm_actual"]
            and not run_data[2]["arm_actual"]
            and not run_data[1]["base_position"]
            and not run_data[2]["base_position"]
            and not run_data[1]["base_cmd_velocity"]
            and not run_data[2]["base_cmd_velocity"]
        ):
            self.axes[0].set_title("Manipulability Comparison")
            self.axes[1].set_title("Directional Force Capability and Desired Force Comparison")
            self.axes[2].set_title(
                f"Arm Joint Delta from Run Start: Actual vs Nullspace Target ({self.run_label(1)})"
            )
            self.axes[3].set_title(
                f"Arm Joint Delta from Run Start: Actual vs Nullspace Target ({self.run_label(2)})"
            )
            self.axes[4].set_title(
                f"End-Effector Position Error ({self.run_label(1)})"
            )
            self.axes[5].set_title(
                f"End-Effector Position Error ({self.run_label(2)})"
            )
            self.figure.tight_layout()
            if self.show_window and os.environ.get("DISPLAY"):
                self.figure.canvas.draw_idle()
                self.figure.canvas.flush_events()
                plt.pause(0.001)
            return

        self.draw_metric_axis(
            self.axes[0],
            run_data,
            metric_key="manipulability",
            title="Manipulability Comparison",
            ylabel="manipulability",
        )
        self.draw_force_axis(self.axes[1], run_data)
        self.axes[1].set_xlabel("time [s]")
        self.draw_joint_axis(
            self.axes[2],
            self.joint_actual_axes[0],
            run_data,
            run_index=1,
            force_value=self.first_force_magnitude,
        )
        self.draw_joint_axis(
            self.axes[3],
            self.joint_actual_axes[1],
            run_data,
            run_index=2,
            force_value=self.second_force_magnitude,
        )
        self.draw_position_error_axis(
            self.axes[4],
            run_data,
            run_index=1,
            force_value=self.first_force_magnitude,
        )
        self.draw_position_error_axis(
            self.axes[5],
            run_data,
            run_index=2,
            force_value=self.second_force_magnitude,
        )
        self.figure.tight_layout()

        if run_data["state"] == 3 and not self.saved:
            self.save_outputs(run_data)
            self.saved = True
            self.frozen = True

        if self.show_window and os.environ.get("DISPLAY"):
            self.figure.canvas.draw_idle()
            self.figure.canvas.flush_events()
            plt.pause(0.001)

    def spin(self):
        rate = rospy.Rate(max(1.0, self.render_rate_hz))
        while not rospy.is_shutdown():
            if not self.frozen:
                self.draw()
            rate.sleep()


def main():
    rospy.init_node("force_sweep_comparison_plotter")
    plotter = ForceSweepComparisonPlotter()
    plotter.spin()


if __name__ == "__main__":
    main()
