#!/usr/bin/env python3

import glob
import importlib.util
import os

import matplotlib
import numpy as np
import rospy

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt


def load_demo_plotter_module():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(script_dir, "whole_body_demo_plotter.py")
    spec = importlib.util.spec_from_file_location(
        "whole_body_demo_plotter_helper", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load whole_body_demo_plotter from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PlannerMetricsPlotter:
    def __init__(self):
        self.csv_path = rospy.get_param("~planner_metrics_csv_path", "")
        self.output_png_path = rospy.get_param(
            "~planner_metrics_output_png_path",
            os.path.join("output", "polytope_ros", "planned_whole_body_metrics.png"),
        )
        self.show_window = bool(rospy.get_param("~show_window", True))
        self.render_rate_hz = float(rospy.get_param("~render_rate_hz", 1.0))
        self.watch_comparison_runs = bool(rospy.get_param("~watch_comparison_runs", True))
        self.comparison_force_sweep_enabled = bool(
            rospy.get_param("~comparison_force_sweep_enabled", False)
        )
        self.expected_run_count = int(
            rospy.get_param(
                "~planner_metrics_expected_run_count",
                2 if self.comparison_force_sweep_enabled else 1,
            )
        )

        if not self.csv_path:
            raise RuntimeError("~planner_metrics_csv_path is empty.")

        self.demo_plotter = load_demo_plotter_module()
        self.last_signature = None
        self.figure_name = "Planner Cost Breakdown"

    def build_watched_csv_paths(self):
        if self.comparison_force_sweep_enabled and self.watch_comparison_runs:
            root, extension = os.path.splitext(self.csv_path)
            run_paths = sorted(glob.glob(f"{root}_run*{extension}"))
            if len(run_paths) < max(self.expected_run_count, 1):
                return []
            return run_paths[: self.expected_run_count]

        return [self.csv_path]

    def extract_run_label(self, csv_path, index):
        base_root, _ = os.path.splitext(os.path.basename(self.csv_path))
        csv_root, _ = os.path.splitext(os.path.basename(csv_path))
        if csv_root.startswith(base_root):
            suffix = csv_root[len(base_root) :]
            if suffix.startswith("_run"):
                return suffix[1:]
        return f"plan_{index + 1}"

    def build_signature(self, csv_paths):
        signature_items = []
        for csv_path in csv_paths:
            if not os.path.exists(csv_path) or os.path.getsize(csv_path) <= 0:
                continue
            signature_items.append(
                (csv_path, os.path.getmtime(csv_path), os.path.getsize(csv_path))
            )
        return tuple(signature_items)

    def plot_cost_axis(self, axis, time_sec, data, columns, title, ylabel):
        plotted_values = []
        for column_name, label in columns:
            if self.demo_plotter.has_col(data, column_name):
                values = self.demo_plotter.col(data, column_name)
                plotted_values.append(values)
                axis.plot(time_sec, values, linewidth=1.8, label=label)
        axis.set_title(title)
        axis.set_xlabel("time [s]")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        if plotted_values:
            self.demo_plotter.symlog_if_needed(axis, np.concatenate(plotted_values))
            axis.legend(loc="best", ncol=2)
        else:
            axis.text(0.5, 0.5, "cost terms not available", ha="center", va="center")

    def plot_cost_panels(self, plotted_items):
        use_interactive_window = self.show_window and bool(os.environ.get("DISPLAY"))
        if not use_interactive_window:
            plt.switch_backend("Agg")

        figure_height = max(4.5 * len(plotted_items), 5.0)
        try:
            if use_interactive_window:
                fig = plt.figure(num=self.figure_name)
                fig.set_size_inches(18, figure_height, forward=True)
                fig.clf()
                axes = fig.subplots(len(plotted_items), 2, squeeze=False)
            else:
                fig, axes = plt.subplots(
                    len(plotted_items),
                    2,
                    figsize=(18, figure_height),
                    squeeze=False,
                )
        except Exception:
            plt.switch_backend("Agg")
            fig, axes = plt.subplots(
                len(plotted_items),
                2,
                figsize=(18, figure_height),
                squeeze=False,
            )

        before_cost_columns = [
            ("before_capability_cost", "capability"),
            ("before_manipulability_cost", "manip"),
            ("before_joint_limit_cost", "joint_limit"),
            ("before_smoothness_cost", "arm_smooth"),
            ("before_nominal_cost", "arm_nominal"),
            ("before_base_smoothness_cost", "base_smooth"),
            ("before_base_nominal_cost", "base_nominal"),
            ("before_base_spectral_energy_cost", "base_spectral"),
            ("before_collision_cost", "collision"),
        ]
        after_cost_columns = [
            ("capability_cost", "capability"),
            ("manipulability_cost", "manip"),
            ("joint_limit_cost", "joint_limit"),
            ("smoothness_cost", "arm_smooth"),
            ("nominal_cost", "arm_nominal"),
            ("base_smoothness_cost", "base_smooth"),
            ("base_nominal_cost", "base_nominal"),
            ("base_spectral_energy_cost", "base_spectral"),
            ("collision_cost", "collision"),
        ]

        title_lines = []
        for row_index, (run_label, csv_path, data) in enumerate(plotted_items):
            time_sec = self.demo_plotter.col(data, "time_sec")
            before_axis = axes[row_index, 0]
            after_axis = axes[row_index, 1]
            before_title = (
                "Before Optimization Costs"
                if len(plotted_items) == 1
                else f"{run_label} Before Optimization Costs"
            )
            after_title = (
                "After Optimization Costs"
                if len(plotted_items) == 1
                else f"{run_label} After Optimization Costs"
            )

            self.plot_cost_axis(
                before_axis,
                time_sec,
                data,
                before_cost_columns,
                before_title,
                "cost value",
            )
            self.plot_cost_axis(
                after_axis,
                time_sec,
                data,
                after_cost_columns,
                after_title,
                "cost value",
            )
            title_lines.append(f"{run_label}: {os.path.basename(csv_path)}")

        fig.suptitle("Planner Cost Breakdown\n" + " | ".join(title_lines), fontsize=14)
        os.makedirs(os.path.dirname(self.output_png_path) or ".", exist_ok=True)
        fig.tight_layout(rect=[0.0, 0.03, 1.0, 0.95])
        fig.savefig(self.output_png_path, dpi=180)
        rospy.loginfo("planner_metrics_plotter: wrote %s", self.output_png_path)

        if use_interactive_window:
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.show(block=False)
            plt.pause(0.001)
        else:
            plt.close(fig)

    def maybe_plot(self):
        csv_paths = self.build_watched_csv_paths()
        signature = self.build_signature(csv_paths)
        if not signature or signature == self.last_signature:
            return

        plotted_items = []
        for index, csv_path in enumerate(csv_paths):
            if not os.path.exists(csv_path) or os.path.getsize(csv_path) <= 0:
                continue
            data = self.demo_plotter.load_structured_csv(csv_path)
            run_label = self.extract_run_label(csv_path, index)
            plotted_items.append((run_label, csv_path, data))

        if not plotted_items:
            return

        self.plot_cost_panels(plotted_items)
        self.last_signature = signature

    def spin(self):
        rate = rospy.Rate(max(self.render_rate_hz, 0.2))
        while not rospy.is_shutdown():
            try:
                self.maybe_plot()
                rate.sleep()
            except rospy.ROSInterruptException:
                break
            except Exception as exc:
                rospy.logwarn("planner_metrics_plotter: failed to update plot: %s", exc)


def main():
    rospy.init_node("planner_metrics_plotter")
    plotter = PlannerMetricsPlotter()
    plotter.spin()


if __name__ == "__main__":
    main()
