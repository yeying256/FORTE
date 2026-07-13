#!/usr/bin/env python3

import os
import sys
import threading
from math import sqrt

import rospy
from geometry_msgs.msg import PoseStamped
from moca_trajectory_generator.msg import TargetPoseCommand


STATE_TOPIC = "/wb_cart_imp_controller/moca_pose"
TARGET_TOPIC = "/target_pose"
TRAJECTORY_FREQUENCY_HZ = 0.05
CYCLE_DURATION_SEC = 1.0 / TRAJECTORY_FREQUENCY_HZ
OUTPUT_DIR = "/tmp/moca_cycle_plots"


def _has_display():
    return bool(os.environ.get("DISPLAY"))


class MatplotlibBackend:
    def __init__(self):
        import matplotlib.pyplot as plt

        self.plt = plt
        self.show_window = _has_display()
        self.plt.ion()
        self.figure, self.axes = self.plt.subplots(3, 2, figsize=(14, 11))
        self.figure.suptitle("MOCA Real-Time State by Cycle")
        if self.show_window:
            self.plt.show(block=False)

    def render(self, cycle_data):
        cycle_index = cycle_data["cycle_index"]
        times = cycle_data["times"]
        actual_x_values = cycle_data["actual_x_values"]
        actual_y_values = cycle_data["actual_y_values"]
        actual_z_values = cycle_data["actual_z_values"]
        target_x_values = cycle_data["target_x_values"]
        target_y_values = cycle_data["target_y_values"]
        target_z_values = cycle_data["target_z_values"]

        error_x_mm = [
            1000.0 * (actual_value - target_value)
            for actual_value, target_value in zip(actual_x_values, target_x_values)
        ]
        error_y_mm = [
            1000.0 * (actual_value - target_value)
            for actual_value, target_value in zip(actual_y_values, target_y_values)
        ]
        error_z_mm = [
            1000.0 * (actual_value - target_value)
            for actual_value, target_value in zip(actual_z_values, target_z_values)
        ]
        error_norm_mm = [
            sqrt(dx * dx + dy * dy + dz * dz)
            for dx, dy, dz in zip(error_x_mm, error_y_mm, error_z_mm)
        ]
        error_rms_mm = sqrt(sum(value * value for value in error_norm_mm) / len(error_norm_mm))
        error_max_mm = max(error_norm_mm)

        ax_xy = self.axes[0][0]
        ax_x = self.axes[0][1]
        ax_y = self.axes[1][0]
        ax_z = self.axes[1][1]
        ax_err_norm = self.axes[2][0]
        ax_err_xyz = self.axes[2][1]

        for axis in (ax_xy, ax_x, ax_y, ax_z, ax_err_norm, ax_err_xyz):
            axis.clear()

        ax_xy.plot(actual_x_values, actual_y_values, color="tab:blue", linewidth=2.0, label="actual")
        ax_xy.plot(
            target_x_values,
            target_y_values,
            color="tab:orange",
            linewidth=2.0,
            linestyle="--",
            label="target",
        )
        ax_xy.scatter(actual_x_values[0], actual_y_values[0], color="tab:green", label="actual start")
        ax_xy.scatter(actual_x_values[-1], actual_y_values[-1], color="tab:red", label="actual end")
        ax_xy.set_title("XY Trajectory")
        ax_xy.set_xlabel("x [m]")
        ax_xy.set_ylabel("y [m]")
        ax_xy.grid(True)
        ax_xy.legend(loc="best")

        ax_x.plot(times, actual_x_values, color="tab:blue", linewidth=2.0, label="actual")
        ax_x.plot(times, target_x_values, color="tab:orange", linewidth=2.0, linestyle="--", label="target")
        ax_x.set_title("x(t)")
        ax_x.set_xlabel("time [s]")
        ax_x.set_ylabel("x [m]")
        ax_x.grid(True)
        ax_x.legend(loc="best")

        ax_y.plot(times, actual_y_values, color="tab:blue", linewidth=2.0, label="actual")
        ax_y.plot(times, target_y_values, color="tab:orange", linewidth=2.0, linestyle="--", label="target")
        ax_y.set_title("y(t)")
        ax_y.set_xlabel("time [s]")
        ax_y.set_ylabel("y [m]")
        ax_y.grid(True)
        ax_y.legend(loc="best")

        ax_z.plot(times, actual_z_values, color="tab:blue", linewidth=2.0, label="actual")
        ax_z.plot(times, target_z_values, color="tab:orange", linewidth=2.0, linestyle="--", label="target")
        ax_z.set_title("z(t)")
        ax_z.set_xlabel("time [s]")
        ax_z.set_ylabel("z [m]")
        ax_z.grid(True)
        ax_z.legend(loc="best")

        ax_err_norm.plot(times, error_norm_mm, color="tab:red", linewidth=2.0)
        ax_err_norm.set_title("Position Error Norm")
        ax_err_norm.set_xlabel("time [s]")
        ax_err_norm.set_ylabel("error [mm]")
        ax_err_norm.grid(True)

        ax_err_xyz.plot(times, error_x_mm, color="tab:blue", linewidth=2.0, label="ex")
        ax_err_xyz.plot(times, error_y_mm, color="tab:orange", linewidth=2.0, label="ey")
        ax_err_xyz.plot(times, error_z_mm, color="tab:green", linewidth=2.0, label="ez")
        ax_err_xyz.set_title("Axis Errors")
        ax_err_xyz.set_xlabel("time [s]")
        ax_err_xyz.set_ylabel("error [mm]")
        ax_err_xyz.grid(True)
        ax_err_xyz.legend(loc="best")

        self.figure.suptitle(
            "MOCA Real-Time State by Cycle - "
            f"cycle {cycle_index} ({len(times)} samples), "
            f"RMS error {error_rms_mm:.2f} mm, max error {error_max_mm:.2f} mm"
        )
        self.figure.tight_layout()
        self.figure.canvas.draw()

        output_path = os.path.join(OUTPUT_DIR, f"cycle_{cycle_index:03d}.png")
        self.figure.savefig(output_path, dpi=160)
        return output_path

    def process_events(self):
        if self.show_window:
            self.figure.canvas.flush_events()
            self.plt.pause(0.001)


class OpenCvBackend:
    def __init__(self):
        import cv2
        import numpy as np

        self.cv2 = cv2
        self.np = np
        self.show_window = _has_display()
        self.window_name = "MOCA Real-Time State by Cycle"

    def _draw_panel(self, image, rect, title, x_series_list, y_series_list, colors, labels, x_label, y_label):
        left, top, width, height = rect
        margin = 50
        plot_left = left + margin
        plot_top = top + 30
        plot_right = left + width - 20
        plot_bottom = top + height - 40

        self.cv2.rectangle(image, (left, top), (left + width, top + height), (0, 0, 0), 1)
        self.cv2.putText(
            image, title, (left + 10, top + 20), self.cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
        )
        self.cv2.putText(
            image,
            x_label,
            (plot_left, top + height - 10),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (80, 80, 80),
            1,
        )
        self.cv2.putText(
            image,
            y_label,
            (left + 5, plot_top + 10),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (80, 80, 80),
            1,
        )

        if not x_series_list or not y_series_list:
            return

        valid_pairs = [
            (xs, ys)
            for xs, ys in zip(x_series_list, y_series_list)
            if len(xs) >= 2 and len(ys) >= 2
        ]
        if not valid_pairs:
            return

        x_min = min(min(xs) for xs, _ in valid_pairs)
        x_max = max(max(xs) for xs, _ in valid_pairs)
        y_min = min(min(ys) for _, ys in valid_pairs)
        y_max = max(max(ys) for _, ys in valid_pairs)

        if abs(x_max - x_min) < 1e-9:
            x_min -= 0.5
            x_max += 0.5
        if abs(y_max - y_min) < 1e-9:
            y_min -= 0.5
            y_max += 0.5

        x_pad = 0.05 * (x_max - x_min)
        y_pad = 0.05 * (y_max - y_min)
        x_min -= x_pad
        x_max += x_pad
        y_min -= y_pad
        y_max += y_pad

        def project(x_value, y_value):
            x_norm = (x_value - x_min) / (x_max - x_min)
            y_norm = (y_value - y_min) / (y_max - y_min)
            px = int(plot_left + x_norm * (plot_right - plot_left))
            py = int(plot_bottom - y_norm * (plot_bottom - plot_top))
            return px, py

        self.cv2.rectangle(
            image, (plot_left, plot_top), (plot_right, plot_bottom), (220, 220, 220), 1
        )

        legend_x = plot_right - 120
        legend_y = plot_top + 18
        for xs, ys, color, label in zip(x_series_list, y_series_list, colors, labels):
            if len(xs) < 2 or len(ys) < 2:
                continue

            points = [project(x_value, y_value) for x_value, y_value in zip(xs, ys)]
            for first, second in zip(points[:-1], points[1:]):
                self.cv2.line(image, first, second, color, 2)

            self.cv2.circle(image, points[0], 4, color, -1)
            self.cv2.circle(image, points[-1], 4, color, 1)

            self.cv2.line(image, (legend_x, legend_y - 5), (legend_x + 20, legend_y - 5), color, 2)
            self.cv2.putText(
                image, label, (legend_x + 25, legend_y), self.cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1
            )
            legend_y += 18

    def render(self, cycle_data):
        cycle_index = cycle_data["cycle_index"]
        times = cycle_data["times"]
        actual_x_values = cycle_data["actual_x_values"]
        actual_y_values = cycle_data["actual_y_values"]
        actual_z_values = cycle_data["actual_z_values"]
        target_x_values = cycle_data["target_x_values"]
        target_y_values = cycle_data["target_y_values"]
        target_z_values = cycle_data["target_z_values"]

        error_x_mm = [
            1000.0 * (actual_value - target_value)
            for actual_value, target_value in zip(actual_x_values, target_x_values)
        ]
        error_y_mm = [
            1000.0 * (actual_value - target_value)
            for actual_value, target_value in zip(actual_y_values, target_y_values)
        ]
        error_z_mm = [
            1000.0 * (actual_value - target_value)
            for actual_value, target_value in zip(actual_z_values, target_z_values)
        ]
        error_norm_mm = [
            sqrt(dx * dx + dy * dy + dz * dz)
            for dx, dy, dz in zip(error_x_mm, error_y_mm, error_z_mm)
        ]
        error_rms_mm = sqrt(sum(value * value for value in error_norm_mm) / len(error_norm_mm))
        error_max_mm = max(error_norm_mm)

        image = self.np.full((1100, 1200, 3), 255, dtype=self.np.uint8)
        self.cv2.putText(
            image,
            (
                f"MOCA Real-Time State by Cycle - cycle {cycle_index} ({len(times)} samples), "
                f"RMS {error_rms_mm:.2f} mm, max {error_max_mm:.2f} mm"
            ),
            (30, 35),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )

        panels = [
            (
                (40, 70, 540, 280),
                "XY Trajectory",
                [actual_x_values, target_x_values],
                [actual_y_values, target_y_values],
                [(255, 0, 0), (0, 140, 255)],
                ["actual", "target"],
                "x [m]",
                "y [m]",
            ),
            (
                (620, 70, 540, 280),
                "x(t)",
                [times, times],
                [actual_x_values, target_x_values],
                [(255, 0, 0), (0, 140, 255)],
                ["actual", "target"],
                "time [s]",
                "x [m]",
            ),
            (
                (40, 390, 540, 280),
                "y(t)",
                [times, times],
                [actual_y_values, target_y_values],
                [(255, 0, 0), (0, 140, 255)],
                ["actual", "target"],
                "time [s]",
                "y [m]",
            ),
            (
                (620, 390, 540, 280),
                "z(t)",
                [times, times],
                [actual_z_values, target_z_values],
                [(255, 0, 0), (0, 140, 255)],
                ["actual", "target"],
                "time [s]",
                "z [m]",
            ),
            (
                (40, 710, 540, 280),
                "Position Error Norm",
                [times],
                [error_norm_mm],
                [(0, 0, 255)],
                ["|e|"],
                "time [s]",
                "error [mm]",
            ),
            (
                (620, 710, 540, 280),
                "Axis Errors",
                [times, times, times],
                [error_x_mm, error_y_mm, error_z_mm],
                [(255, 0, 0), (0, 140, 255), (0, 170, 0)],
                ["ex", "ey", "ez"],
                "time [s]",
                "error [mm]",
            ),
        ]

        for rect, title, x_series_list, y_series_list, colors, labels, x_label, y_label in panels:
            self._draw_panel(image, rect, title, x_series_list, y_series_list, colors, labels, x_label, y_label)

        output_path = os.path.join(OUTPUT_DIR, f"cycle_{cycle_index:03d}.png")
        self.cv2.imwrite(output_path, image)

        if self.show_window:
            self.cv2.imshow(self.window_name, image)

        return output_path

    def process_events(self):
        if self.show_window:
            self.cv2.waitKey(1)


def create_backend():
    errors = []

    try:
        return MatplotlibBackend(), "matplotlib"
    except Exception as exc:
        errors.append(f"matplotlib backend unavailable: {exc}")

    try:
        return OpenCvBackend(), "opencv"
    except Exception as exc:
        errors.append(f"opencv backend unavailable: {exc}")

    raise RuntimeError("; ".join(errors))


class CycleStatePlotter:
    def __init__(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        self.backend, backend_name = create_backend()
        self.data_lock = threading.Lock()
        self.cycle_index = 0
        self.cycle_start_time = None
        self.latest_target_position = None
        self.times = []
        self.actual_x_values = []
        self.actual_y_values = []
        self.actual_z_values = []
        self.target_x_values = []
        self.target_y_values = []
        self.target_z_values = []
        self.pending_cycles = []

        rospy.Subscriber(TARGET_TOPIC, TargetPoseCommand, self.target_callback, queue_size=50)
        rospy.Subscriber(STATE_TOPIC, PoseStamped, self.pose_callback, queue_size=200)
        rospy.loginfo(
            "cycle_state_plotter: listening to %s and %s with %s backend on %s",
            STATE_TOPIC,
            TARGET_TOPIC,
            backend_name,
            sys.executable,
        )

    def target_callback(self, msg):
        with self.data_lock:
            self.latest_target_position = (
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            )

    def pose_callback(self, msg):
        stamp = msg.header.stamp.to_sec() if msg.header.stamp != rospy.Time() else rospy.Time.now().to_sec()

        with self.data_lock:
            if self.cycle_start_time is None:
                self.cycle_start_time = stamp

            relative_time = stamp - self.cycle_start_time
            actual_x = msg.pose.position.x
            actual_y = msg.pose.position.y
            actual_z = msg.pose.position.z
            if self.latest_target_position is None:
                target_x, target_y, target_z = actual_x, actual_y, actual_z
            else:
                target_x, target_y, target_z = self.latest_target_position

            self.times.append(relative_time)
            self.actual_x_values.append(actual_x)
            self.actual_y_values.append(actual_y)
            self.actual_z_values.append(actual_z)
            self.target_x_values.append(target_x)
            self.target_y_values.append(target_y)
            self.target_z_values.append(target_z)

            if relative_time >= CYCLE_DURATION_SEC:
                self.queue_cycle_for_render()
                self.reset_cycle(stamp)

    def queue_cycle_for_render(self):
        if len(self.times) < 2:
            return

        self.cycle_index += 1
        self.pending_cycles.append(
            {
                "cycle_index": self.cycle_index,
                "times": list(self.times),
                "actual_x_values": list(self.actual_x_values),
                "actual_y_values": list(self.actual_y_values),
                "actual_z_values": list(self.actual_z_values),
                "target_x_values": list(self.target_x_values),
                "target_y_values": list(self.target_y_values),
                "target_z_values": list(self.target_z_values),
            }
        )

    def render_pending_cycles(self):
        while True:
            with self.data_lock:
                if not self.pending_cycles:
                    return
                cycle_data = self.pending_cycles.pop(0)

            output_path = self.backend.render(cycle_data)
            rospy.loginfo("cycle_state_plotter: wrote %s", output_path)

    def reset_cycle(self, current_stamp):
        last_actual_x = self.actual_x_values[-1]
        last_actual_y = self.actual_y_values[-1]
        last_actual_z = self.actual_z_values[-1]
        last_target_x = self.target_x_values[-1]
        last_target_y = self.target_y_values[-1]
        last_target_z = self.target_z_values[-1]

        self.cycle_start_time = current_stamp
        self.times = [0.0]
        self.actual_x_values = [last_actual_x]
        self.actual_y_values = [last_actual_y]
        self.actual_z_values = [last_actual_z]
        self.target_x_values = [last_target_x]
        self.target_y_values = [last_target_y]
        self.target_z_values = [last_target_z]

    def spin(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            self.render_pending_cycles()
            self.backend.process_events()
            rate.sleep()


def main():
    rospy.init_node("cycle_state_plotter")
    plotter = CycleStatePlotter()
    plotter.spin()


if __name__ == "__main__":
    main()
