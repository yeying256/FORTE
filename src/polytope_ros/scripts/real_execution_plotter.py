#!/usr/bin/env python3

import collections
import os
import threading

import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rospy
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Float64


DEFAULT_OUTPUT_PATH = (
    os.path.join(os.path.expanduser("~"), ".ros", "moca_polytope", "real_execution_plot.png")
)


class RealExecutionPlotter:
    def __init__(self):
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
        self.output_png_path = rospy.get_param("~output_png_path", DEFAULT_OUTPUT_PATH)
        self.show_window = rospy.get_param("~show_window", True)
        self.render_rate_hz = rospy.get_param("~render_rate_hz", 5.0)
        self.history_duration_sec = rospy.get_param("~history_duration_sec", 30.0)
        self.trajectory_mode = rospy.get_param("~trajectory_mode", "test_lift")
        self.trajectory_start_delay_sec = rospy.get_param("~trajectory_start_delay_sec", 1.0)
        self.pre_trajectory_home_settle_duration_sec = rospy.get_param(
            "~pre_trajectory_home_settle_duration_sec", 0.0
        )
        self.circle_frequency_hz = rospy.get_param("~circle_frequency_hz", 0.05)
        self.circle_desired_force_magnitude = rospy.get_param(
            "~circle_desired_force_magnitude", 70.0
        )
        self.test_forward_duration_sec = rospy.get_param("~test_forward_duration_sec", 5.0)
        self.test_lift_duration_sec = rospy.get_param("~test_lift_duration_sec", 5.0)
        self.payload_move_above_duration_sec = rospy.get_param(
            "~payload_move_above_duration_sec", 5.0
        )
        self.payload_descend_duration_sec = rospy.get_param(
            "~payload_descend_duration_sec", 3.0
        )
        self.payload_close_duration_sec = rospy.get_param(
            "~payload_close_duration_sec", 1.5
        )
        self.payload_lift_duration_sec = rospy.get_param(
            "~payload_lift_duration_sec", 4.0
        )
        self.lift_desired_force_magnitude = rospy.get_param(
            "~lift_desired_force_magnitude", 70.0
        )
        self.enable_force_capability_optimization = rospy.get_param(
            "~enable_force_capability_optimization", False
        )
        self.enable_vlm_mass_request = rospy.get_param(
            "~enable_vlm_mass_request", False
        )
        self.vlm_mass_request_timeout_sec = rospy.get_param(
            "~vlm_mass_request_timeout_sec", 0.0
        )
        self.save_first_cycle_metrics = rospy.get_param("~save_first_cycle_metrics", True)
        self.auto_save_and_freeze = rospy.get_param("~auto_save_and_freeze", True)
        self.freeze_after_duration_sec = rospy.get_param("~freeze_after_duration_sec", 0.0)
        self.freeze_extra_duration_sec = rospy.get_param(
            "~freeze_extra_duration_sec", 0.0
        )

        self.lock = threading.Lock()
        self.manipulability_history = collections.deque()
        self.directional_force_capacity_history = collections.deque()
        self.desired_force_history = collections.deque()
        self.metric_start_time = None
        self.first_cycle_metrics_saved = False
        self.main_figure_saved = False
        self.frozen = False

        output_dir = os.path.dirname(self.output_png_path)
        self.first_cycle_manipulability_png_path = rospy.get_param(
            "~first_cycle_manipulability_png_path",
            os.path.join(output_dir, "first_cycle_manipulability.png"),
        )
        self.first_cycle_remaining_force_png_path = rospy.get_param(
            "~first_cycle_remaining_force_png_path",
            os.path.join(output_dir, "first_cycle_remaining_force.png"),
        )

        self.figure, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=False)
        self.force_axis = axes[0]
        self.manipulability_axis = axes[1]
        if self.show_window and os.environ.get("DISPLAY"):
            plt.ion()
            plt.show(block=False)

        rospy.Subscriber(
            self.manipulability_topic, Float64, self.manipulability_callback, queue_size=200
        )
        rospy.Subscriber(
            self.directional_force_capacity_topic,
            Float64,
            self.directional_force_capacity_callback,
            queue_size=200,
        )
        rospy.Subscriber(
            self.desired_force_topic,
            Vector3Stamped,
            self.desired_force_callback,
            queue_size=200,
        )

        rospy.loginfo(
            "real_execution_plotter: listening to %s, %s, %s",
            self.manipulability_topic,
            self.directional_force_capacity_topic,
            self.desired_force_topic,
        )

    def ros_time_to_sec(self, stamp):
        if stamp == rospy.Time():
            return rospy.Time.now().to_sec()
        return stamp.to_sec()

    def trim_history(self, history, latest_time):
        min_time = latest_time - self.history_duration_sec
        while history and history[0]["time"] < min_time:
            history.popleft()

    def manipulability_callback(self, msg):
        stamp = rospy.Time.now().to_sec()
        with self.lock:
            if self.metric_start_time is None:
                self.metric_start_time = stamp
            if self.manipulability_history and abs(
                self.manipulability_history[-1]["time"] - stamp
            ) < 1e-3:
                self.manipulability_history[-1]["manipulability"] = float(msg.data)
            else:
                self.manipulability_history.append(
                    {
                        "time": stamp,
                        "manipulability": float(msg.data),
                    }
                )
            self.trim_history(self.manipulability_history, stamp)

    def directional_force_capacity_callback(self, msg):
        stamp = rospy.Time.now().to_sec()
        with self.lock:
            if self.metric_start_time is None:
                self.metric_start_time = stamp
            self.directional_force_capacity_history.append(
                {"time": stamp, "force_capacity": float(msg.data)}
            )
            self.trim_history(self.directional_force_capacity_history, stamp)

    def desired_force_callback(self, msg):
        stamp = self.ros_time_to_sec(msg.header.stamp)
        with self.lock:
            if self.metric_start_time is None:
                self.metric_start_time = stamp
            self.desired_force_history.append(
                {
                    "time": stamp,
                    "value": np.asarray(
                        [msg.vector.x, msg.vector.y, msg.vector.z], dtype=float
                    ),
                }
            )
            self.trim_history(self.desired_force_history, stamp)

    def first_cycle_duration_sec(self):
        startup_buffer = (
            self.pre_trajectory_home_settle_duration_sec + self.trajectory_start_delay_sec
        )
        if self.enable_vlm_mass_request:
            startup_buffer += self.vlm_mass_request_timeout_sec

        if self.trajectory_mode == "circle":
            return (
                startup_buffer
                + max(1e-6, 1.0 / self.circle_frequency_hz)
                + self.freeze_extra_duration_sec
            )
        if self.trajectory_mode == "payload_lift":
            return (
                startup_buffer
                + self.payload_move_above_duration_sec
                + self.payload_descend_duration_sec
                + self.payload_close_duration_sec
                + self.payload_lift_duration_sec
                + self.freeze_extra_duration_sec
            )
        return (
            startup_buffer
            + self.test_forward_duration_sec
            + self.test_lift_duration_sec
            + self.freeze_extra_duration_sec
        )

    def desired_force_magnitude_at(self, relative_time):
        if not self.enable_force_capability_optimization:
            return 0.0

        if self.trajectory_mode == "circle":
            active_time = max(0.0, relative_time - self.trajectory_start_delay_sec)
            return self.circle_desired_force_magnitude if active_time >= 0.0 else 0.0

        if relative_time < self.trajectory_start_delay_sec + self.test_forward_duration_sec:
            return 0.0
        return self.lift_desired_force_magnitude

    def maybe_save_first_cycle_metric_figures(
        self,
        manipulability_samples,
        directional_force_capacity_samples,
        desired_force_samples,
    ):
        if not self.save_first_cycle_metrics or self.first_cycle_metrics_saved:
            return
        if self.metric_start_time is None:
            return
        if not manipulability_samples or not directional_force_capacity_samples:
            return

        latest_time = max(
            manipulability_samples[-1]["time"], directional_force_capacity_samples[-1]["time"]
        )
        cycle_duration = self.first_cycle_duration_sec()
        if latest_time - self.metric_start_time < cycle_duration:
            return

        manipulability_cycle = [
            sample
            for sample in manipulability_samples
            if sample["time"] - self.metric_start_time <= cycle_duration
        ]
        force_cycle = [
            sample
            for sample in directional_force_capacity_samples
            if sample["time"] - self.metric_start_time <= cycle_duration
        ]
        if not manipulability_cycle or not force_cycle:
            return

        output_dir = os.path.dirname(self.output_png_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        manipulability_times = np.asarray(
            [sample["time"] - self.metric_start_time for sample in manipulability_cycle],
            dtype=float,
        )
        manipulability_values = np.asarray(
            [sample["manipulability"] for sample in manipulability_cycle], dtype=float
        )

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(
            manipulability_times,
            manipulability_values,
            color="tab:purple",
            linewidth=2.0,
        )
        ax.set_title("First Cycle Manipulability")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("manipulability")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.first_cycle_manipulability_png_path, dpi=180)
        plt.close(fig)

        force_times = np.asarray(
            [sample["time"] - self.metric_start_time for sample in force_cycle], dtype=float
        )
        force_capacity_values = np.asarray(
            [sample["force_capacity"] for sample in force_cycle], dtype=float
        )
        if desired_force_samples:
            desired_force_cycle = [
                sample
                for sample in desired_force_samples
                if sample["time"] - self.metric_start_time <= cycle_duration
            ]
        else:
            desired_force_cycle = []

        if desired_force_cycle:
            desired_force_times = np.asarray(
                [sample["time"] - self.metric_start_time for sample in desired_force_cycle],
                dtype=float,
            )
            desired_force_values_raw = np.asarray(
                [np.linalg.norm(sample["value"]) for sample in desired_force_cycle],
                dtype=float,
            )
            if desired_force_values_raw.size == 1:
                desired_force_values = np.full_like(
                    force_times, desired_force_values_raw[0], dtype=float
                )
            else:
                desired_force_values = np.interp(
                    force_times,
                    desired_force_times,
                    desired_force_values_raw,
                    left=desired_force_values_raw[0],
                    right=desired_force_values_raw[-1],
                )
        else:
            desired_force_values = np.asarray(
                [self.desired_force_magnitude_at(time_value) for time_value in force_times],
                dtype=float,
            )
        remaining_force_values = force_capacity_values - desired_force_values

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(
            force_times,
            remaining_force_values,
            color="tab:blue",
            linewidth=2.0,
            label="remaining force",
        )
        ax.plot(
            force_times,
            force_capacity_values,
            color="tab:red",
            linewidth=1.8,
            label="force capacity",
        )
        ax.plot(
            force_times,
            desired_force_values,
            "--",
            color="tab:orange",
            linewidth=1.8,
            label="desired force",
        )
        ax.set_title("First Cycle Remaining Output Force")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("force [N]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(self.first_cycle_remaining_force_png_path, dpi=180)
        plt.close(fig)

        self.first_cycle_metrics_saved = True
        rospy.loginfo(
            "real_execution_plotter: saved first-cycle metric figures to %s and %s",
            self.first_cycle_manipulability_png_path,
            self.first_cycle_remaining_force_png_path,
        )

    def freeze_duration_sec(self):
        if self.freeze_after_duration_sec > 0.0:
            return self.freeze_after_duration_sec
        return self.first_cycle_duration_sec()

    def latest_metric_time(self, manipulability_samples, directional_force_capacity_samples):
        latest_time = None
        if manipulability_samples:
            latest_time = manipulability_samples[-1]["time"]
        if directional_force_capacity_samples:
            if latest_time is None:
                latest_time = directional_force_capacity_samples[-1]["time"]
            else:
                latest_time = max(latest_time, directional_force_capacity_samples[-1]["time"])
        return latest_time

    def should_freeze(self, manipulability_samples, directional_force_capacity_samples):
        if not self.auto_save_and_freeze or self.frozen or self.metric_start_time is None:
            return False
        latest_time = self.latest_metric_time(
            manipulability_samples, directional_force_capacity_samples
        )
        if latest_time is None:
            return False
        return latest_time - self.metric_start_time >= self.freeze_duration_sec()

    def save_main_figure_if_needed(self):
        if self.main_figure_saved:
            return
        output_dir = os.path.dirname(self.output_png_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        self.figure.savefig(self.output_png_path, dpi=180)
        self.main_figure_saved = True
        rospy.loginfo(
            "real_execution_plotter: saved frozen main figure to %s",
            self.output_png_path,
        )

    def render(self):
        if self.frozen:
            return

        with self.lock:
            manipulability_samples = list(self.manipulability_history)
            directional_force_capacity_samples = list(self.directional_force_capacity_history)
            desired_force_samples = list(self.desired_force_history)

        if (
            not manipulability_samples
            and not directional_force_capacity_samples
            and not desired_force_samples
        ):
            return

        self.force_axis.clear()
        self.manipulability_axis.clear()

        first_time = None
        if directional_force_capacity_samples:
            first_time = directional_force_capacity_samples[0]["time"]
        if manipulability_samples:
            if first_time is None:
                first_time = manipulability_samples[0]["time"]
            else:
                first_time = min(first_time, manipulability_samples[0]["time"])
        if desired_force_samples:
            if first_time is None:
                first_time = desired_force_samples[0]["time"]
            else:
                first_time = min(first_time, desired_force_samples[0]["time"])

        legend_handles = []
        legend_labels = []

        if directional_force_capacity_samples:
            force_capacity_times = np.asarray(
                [sample["time"] - first_time for sample in directional_force_capacity_samples],
                dtype=float,
            )
            force_capacity_values = np.asarray(
                [sample["force_capacity"] for sample in directional_force_capacity_samples],
                dtype=float,
            )
            force_line = self.force_axis.plot(
                force_capacity_times,
                force_capacity_values,
                color="tab:red",
                linewidth=2.2,
                label="max directional force capability",
            )[0]
            legend_handles.append(force_line)
            legend_labels.append("max directional force capability")

        if desired_force_samples:
            desired_force_times = np.asarray(
                [sample["time"] - first_time for sample in desired_force_samples],
                dtype=float,
            )
            desired_force_vectors = np.asarray(
                [sample["value"] for sample in desired_force_samples], dtype=float
            )
            desired_force_magnitudes = np.linalg.norm(desired_force_vectors, axis=1)
            desired_force_line = self.force_axis.plot(
                desired_force_times,
                desired_force_magnitudes,
                "--",
                color="tab:orange",
                linewidth=2.0,
                label="desired force magnitude",
            )[0]
            legend_handles.append(desired_force_line)
            legend_labels.append("desired force magnitude")

        if manipulability_samples:
            manipulability_times = np.asarray(
                [sample["time"] - first_time for sample in manipulability_samples],
                dtype=float,
            )
            manipulability_values = np.asarray(
                [sample["manipulability"] for sample in manipulability_samples],
                dtype=float,
            )
            manipulability_line = self.manipulability_axis.plot(
                manipulability_times,
                manipulability_values,
                color="tab:purple",
                linewidth=2.2,
                label="manipulability",
            )[0]
            legend_handles.append(manipulability_line)
            legend_labels.append("manipulability")

        self.force_axis.set_title(
            "Directional Force Capability and Desired Force"
        )
        self.force_axis.set_xlabel("time [s]")
        self.force_axis.set_ylabel("force [N]")
        self.force_axis.grid(True, alpha=0.3)
        if legend_handles:
            self.force_axis.legend(legend_handles, legend_labels, loc="best")

        if manipulability_samples:
            manipulability_times = np.asarray(
                [sample["time"] - first_time for sample in manipulability_samples],
                dtype=float,
            )
            manipulability_values = np.asarray(
                [sample["manipulability"] for sample in manipulability_samples],
                dtype=float,
            )
            self.manipulability_axis.plot(
                manipulability_times,
                manipulability_values,
                color="tab:purple",
                linewidth=2.2,
                label="manipulability",
            )
        self.manipulability_axis.set_title("Manipulability")
        self.manipulability_axis.set_xlabel("time [s]")
        self.manipulability_axis.set_ylabel("manipulability")
        self.manipulability_axis.grid(True, alpha=0.3)
        if manipulability_samples:
            self.manipulability_axis.legend(loc="best")

        self.figure.tight_layout()

        if self.show_window and os.environ.get("DISPLAY"):
            self.figure.canvas.draw_idle()
            self.figure.canvas.flush_events()
            plt.pause(0.001)

        if self.should_freeze(manipulability_samples, directional_force_capacity_samples):
            self.save_main_figure_if_needed()
            self.maybe_save_first_cycle_metric_figures(
                manipulability_samples,
                directional_force_capacity_samples,
                desired_force_samples,
            )
            self.frozen = True
            rospy.loginfo("real_execution_plotter: figure frozen after auto-save.")

    def spin(self):
        rate = rospy.Rate(self.render_rate_hz)
        while not rospy.is_shutdown():
            self.render()
            rate.sleep()


def main():
    rospy.init_node("real_execution_plotter")
    plotter = RealExecutionPlotter()
    plotter.spin()


if __name__ == "__main__":
    main()
