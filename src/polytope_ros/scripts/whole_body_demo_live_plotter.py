#!/usr/bin/env python3

import os
import sys

import matplotlib

if "--show_window" not in sys.argv or not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray


EXPECTED_SAMPLE_SIZE = 20


class WholeBodyDemoLivePlotter:
    def __init__(self):
        self.sample_topic = rospy.get_param("~sample_topic", "~demo_sample")
        self.output_png_path = rospy.get_param(
            "~output_png_path",
            os.path.join("output", "polytope_ros", "whole_body_demo_live.png"),
        )
        self.show_window = rospy.get_param("~show_window", True)
        self.render_rate_hz = rospy.get_param("~render_rate_hz", 5.0)

        self.samples = []
        self.figure = None
        self.axes = None
        self.last_rendered_count = -1

        rospy.Subscriber(self.sample_topic, Float64MultiArray, self.sample_callback, queue_size=50)

        if self.show_window and os.environ.get("DISPLAY"):
            plt.ion()

        rospy.loginfo(
            "whole_body_demo_live_plotter: listening to %s", self.sample_topic
        )

    def sample_callback(self, msg):
        if len(msg.data) < EXPECTED_SAMPLE_SIZE:
            rospy.logwarn_throttle(
                2.0,
                "whole_body_demo_live_plotter: expected at least %d values, got %d",
                EXPECTED_SAMPLE_SIZE,
                len(msg.data),
            )
            return
        self.samples.append(list(msg.data[:EXPECTED_SAMPLE_SIZE]))

    def ensure_figure(self):
        if self.figure is None:
            self.figure, self.axes = plt.subplots(2, 2, figsize=(14, 10))

    def render(self):
        if not self.samples or self.last_rendered_count == len(self.samples):
            return

        self.ensure_figure()
        self.last_rendered_count = len(self.samples)

        data = np.asarray(self.samples, dtype=float)
        time_sec = data[:, 0]
        target_xyz = data[:, 1:4]
        optimized_xyz = data[:, 4:7]
        base_state = data[:, 7:10]
        pose_error_mm = data[:, 10] * 1000.0
        force_capacity = data[:, 11]
        objective_value = data[:, 12]

        error_xyz_mm = (optimized_xyz - target_xyz) * 1000.0
        base_yaw_deg = np.rad2deg(base_state[:, 2])

        for axis_row in self.axes:
            for axis in axis_row:
                axis.cla()

        ax = self.axes[0, 0]
        ax.plot(target_xyz[:, 0], target_xyz[:, 1], "--", color="tab:orange", linewidth=2.0, label="target")
        ax.plot(optimized_xyz[:, 0], optimized_xyz[:, 1], "-", color="tab:blue", linewidth=2.0, label="optimized")
        ax.set_title("XY Tracking")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True, alpha=0.3)
        ax.axis("equal")
        ax.legend(loc="best")

        ax = self.axes[0, 1]
        ax.plot(time_sec, error_xyz_mm[:, 0], color="tab:red", label="ex")
        ax.plot(time_sec, error_xyz_mm[:, 1], color="tab:green", label="ey")
        ax.plot(time_sec, error_xyz_mm[:, 2], color="tab:blue", label="ez")
        ax.plot(time_sec, pose_error_mm, color="tab:purple", linewidth=2.0, label="norm")
        ax.set_title("Tracking Error")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("error [mm]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        ax = self.axes[1, 0]
        ax.plot(time_sec, base_state[:, 0], color="tab:blue", label="base x")
        ax.plot(time_sec, base_state[:, 1], color="tab:orange", label="base y")
        ax.set_title("Base Motion")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("translation [m]")
        ax.grid(True, alpha=0.3)

        yaw_axis = ax.twinx()
        yaw_axis.plot(time_sec, base_yaw_deg, color="tab:brown", linestyle="--", label="base yaw")
        yaw_axis.set_ylabel("yaw [deg]", color="tab:brown")
        yaw_axis.tick_params(axis="y", labelcolor="tab:brown")

        handles_1, labels_1 = ax.get_legend_handles_labels()
        handles_2, labels_2 = yaw_axis.get_legend_handles_labels()
        ax.legend(handles_1 + handles_2, labels_1 + labels_2, loc="best")

        ax = self.axes[1, 1]
        ax.plot(time_sec, force_capacity, color="tab:green", linewidth=2.0, label="force capacity")
        ax.set_title("Optimization Metrics")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("force capacity [N]", color="tab:green")
        ax.tick_params(axis="y", labelcolor="tab:green")
        ax.grid(True, alpha=0.3)

        objective_axis = ax.twinx()
        objective_axis.plot(
            time_sec,
            objective_value,
            color="tab:gray",
            linestyle="--",
            linewidth=1.8,
            label="objective",
        )
        objective_axis.set_ylabel("objective value", color="tab:gray")
        objective_axis.tick_params(axis="y", labelcolor="tab:gray")

        handles_1, labels_1 = ax.get_legend_handles_labels()
        handles_2, labels_2 = objective_axis.get_legend_handles_labels()
        ax.legend(handles_1 + handles_2, labels_1 + labels_2, loc="best")

        max_error = float(np.max(pose_error_mm))
        rms_error = float(np.sqrt(np.mean(np.square(pose_error_mm))))
        max_base_translation = float(np.max(np.linalg.norm(base_state[:, :2], axis=1)))
        self.figure.suptitle(
            f"RMS error = {rms_error:.3f} mm | "
            f"Max error = {max_error:.3f} mm | "
            f"Max base shift = {max_base_translation:.3f} m",
            fontsize=13,
        )
        self.figure.tight_layout(rect=[0.0, 0.03, 1.0, 0.95])

        output_dir = os.path.dirname(self.output_png_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        self.figure.savefig(self.output_png_path, dpi=180)

        if self.show_window and os.environ.get("DISPLAY"):
            self.figure.canvas.draw_idle()
            self.figure.canvas.flush_events()

    def spin(self):
        rate = rospy.Rate(self.render_rate_hz)
        while not rospy.is_shutdown():
            self.render()
            rate.sleep()
        self.render()


def main():
    rospy.init_node("whole_body_demo_live_plotter")
    plotter = WholeBodyDemoLivePlotter()
    plotter.spin()


if __name__ == "__main__":
    main()
