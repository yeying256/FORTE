#!/usr/bin/env python3

import argparse
import os
import time

import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def wait_for_csv(path, timeout_sec):
    start_time = time.time()
    while True:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return True
        if time.time() - start_time > timeout_sec:
            return False
        time.sleep(0.2)


def load_structured_csv(path):
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    if data.size == 0:
        raise RuntimeError(f"CSV is empty: {path}")
    return np.atleast_1d(data)


def col(data, name):
    return np.atleast_1d(data[name]).astype(float)


def has_col(data, name):
    return name in data.dtype.names


def symlog_if_needed(axis, values, linthresh=1e-3):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return
    if np.all(np.isfinite(values)) and np.max(np.abs(values)) > 10.0 * linthresh:
        axis.set_yscale("symlog", linthresh=linthresh)


def build_summary_text(data):
    pose_error_mm = col(data, "pose_error_norm") * 1000.0
    base_x = col(data, "base_x")
    base_y = col(data, "base_y")
    base_yaw_deg = np.rad2deg(col(data, "base_yaw"))
    force_capacity = col(data, "force_capacity")

    rms_error_mm = float(np.sqrt(np.mean(np.square(pose_error_mm))))
    max_error_mm = float(np.max(pose_error_mm))
    max_base_translation = float(np.max(np.sqrt(np.square(base_x) + np.square(base_y))))
    max_base_yaw_deg = float(np.max(np.abs(base_yaw_deg)))
    min_force_capacity = float(np.min(force_capacity))

    return (
        f"RMS error = {rms_error_mm:.3f} mm | "
        f"Max error = {max_error_mm:.3f} mm | "
        f"Max base shift = {max_base_translation:.3f} m | "
        f"Max yaw = {max_base_yaw_deg:.2f} deg | "
        f"Min force capacity = {min_force_capacity:.2f} N"
    )


def plot_demo(data, output_png_path, show_window):
    if not show_window:
        plt.switch_backend("Agg")

    time_sec = col(data, "time_sec")
    target_x = col(data, "target_x")
    target_y = col(data, "target_y")
    target_z = col(data, "target_z")
    optimized_x = col(data, "optimized_ee_x")
    optimized_y = col(data, "optimized_ee_y")
    optimized_z = col(data, "optimized_ee_z")
    base_x = col(data, "base_x")
    base_y = col(data, "base_y")
    base_yaw_deg = np.rad2deg(col(data, "base_yaw"))
    pose_error_mm = col(data, "pose_error_norm") * 1000.0
    required_force = col(data, "required_force") if has_col(data, "required_force") else None
    force_capacity = col(data, "force_capacity")
    objective_value = col(data, "objective_value")

    error_x_mm = (optimized_x - target_x) * 1000.0
    error_y_mm = (optimized_y - target_y) * 1000.0
    error_z_mm = (optimized_z - target_z) * 1000.0

    try:
        fig, axes = plt.subplots(5, 2, figsize=(18, 25))
    except Exception:
        plt.switch_backend("Agg")
        fig, axes = plt.subplots(5, 2, figsize=(18, 25))
    fig.suptitle(build_summary_text(data), fontsize=14)

    ax = axes[0, 0]
    ax.plot(target_x, target_y, "--", color="tab:orange", linewidth=2.0, label="target")
    ax.plot(optimized_x, optimized_y, "-", color="tab:blue", linewidth=2.0, label="optimized")
    ax.set_title("End-Effector XY Trajectory")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    ax.legend(loc="best")

    ax = axes[0, 1]
    ax.plot(time_sec, target_x, "--", color="tab:red", label="target x")
    ax.plot(time_sec, optimized_x, "-", color="tab:red", label="optimized x")
    ax.plot(time_sec, target_y, "--", color="tab:green", label="target y")
    ax.plot(time_sec, optimized_y, "-", color="tab:green", label="optimized y")
    ax.plot(time_sec, target_z, "--", color="tab:blue", label="target z")
    ax.plot(time_sec, optimized_z, "-", color="tab:blue", label="optimized z")
    ax.set_title("Cartesian Tracking")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position [m]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncol=2)

    ax = axes[1, 0]
    ax.plot(time_sec, error_x_mm, color="tab:red", label="ex")
    ax.plot(time_sec, error_y_mm, color="tab:green", label="ey")
    ax.plot(time_sec, error_z_mm, color="tab:blue", label="ez")
    ax.set_title("Axis Errors")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("error [mm]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[1, 1]
    ax.plot(time_sec, pose_error_mm, color="tab:purple", linewidth=2.0)
    ax.set_title("Pose Error Norm")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("error norm [mm]")
    ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    ax.plot(time_sec, base_x, color="tab:blue", label="base x")
    ax.plot(time_sec, base_y, color="tab:orange", label="base y")
    ax.set_title("Base Translation")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position [m]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[2, 1]
    ax.plot(time_sec, base_yaw_deg, color="tab:brown")
    ax.set_title("Base Yaw")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("yaw [deg]")
    ax.grid(True, alpha=0.3)

    ax = axes[3, 0]
    for joint_index in range(7):
        ax.plot(
            time_sec,
            col(data, f"arm_q{joint_index + 1}"),
            linewidth=1.6,
            label=f"q{joint_index + 1}",
        )
    ax.set_title("Arm Joint Trajectory")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("joint position [rad]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncol=4)

    ax = axes[3, 1]
    ax.plot(time_sec, force_capacity, color="tab:green", linewidth=2.0, label="force capacity")
    if required_force is not None:
        ax.plot(
            time_sec,
            required_force,
            color="tab:red",
            linewidth=1.8,
            linestyle=":",
            label="required force",
        )
    ax.set_title("Optimization Metrics")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("force capacity [N]", color="tab:green")
    ax.tick_params(axis="y", labelcolor="tab:green")
    ax.grid(True, alpha=0.3)

    objective_ax = ax.twinx()
    objective_ax.plot(
        time_sec,
        objective_value,
        color="tab:gray",
        linewidth=1.8,
        linestyle="--",
        label="objective",
    )
    objective_ax.set_ylabel("objective value", color="tab:gray")
    objective_ax.tick_params(axis="y", labelcolor="tab:gray")

    handles_1, labels_1 = ax.get_legend_handles_labels()
    handles_2, labels_2 = objective_ax.get_legend_handles_labels()
    ax.legend(handles_1 + handles_2, labels_1 + labels_2, loc="best")

    raw_cost_columns = [
        ("capability_cost", "capability"),
        ("manipulability_cost", "manip"),
        ("joint_limit_cost", "joint_limit"),
        ("smoothness_cost", "arm_smooth"),
        ("nominal_cost", "arm_nominal"),
        ("base_smoothness_cost", "base_smooth"),
        ("base_nominal_cost", "base_nominal"),
        ("base_spectral_energy_cost", "base_spectral"),
    ]
    ax = axes[4, 0]
    raw_values_for_scale = []
    for column_name, label in raw_cost_columns:
        if has_col(data, column_name):
            values = col(data, column_name)
            raw_values_for_scale.append(values)
            ax.plot(time_sec, values, linewidth=1.8, label=label)
    ax.set_title("Raw Cost Terms")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("raw cost")
    ax.grid(True, alpha=0.3)
    if raw_values_for_scale:
        symlog_if_needed(ax, np.concatenate(raw_values_for_scale))
        ax.legend(loc="best", ncol=2)
    else:
        ax.text(0.5, 0.5, "raw cost terms not available", ha="center", va="center")

    weighted_cost_columns = [
        ("weighted_capability", "w*capability"),
        ("weighted_manipulability", "w*manip"),
        ("weighted_joint_limit", "w*joint_limit"),
        ("weighted_smoothness", "w*arm_smooth"),
        ("weighted_nominal", "w*arm_nominal"),
        ("weighted_base_smoothness", "w*base_smooth"),
        ("weighted_base_nominal", "w*base_nominal"),
        ("weighted_base_spectral_energy", "w*base_spectral"),
    ]
    ax = axes[4, 1]
    weighted_values_for_scale = []
    for column_name, label in weighted_cost_columns:
        if has_col(data, column_name):
            values = col(data, column_name)
            weighted_values_for_scale.append(values)
            ax.plot(time_sec, values, linewidth=1.8, label=label)
    ax.set_title("Weighted Cost Contributions")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("weighted contribution")
    ax.grid(True, alpha=0.3)
    if weighted_values_for_scale:
        symlog_if_needed(ax, np.concatenate(weighted_values_for_scale))
        ax.legend(loc="best", ncol=2)
    else:
        ax.text(0.5, 0.5, "weighted contributions not available", ha="center", va="center")

    os.makedirs(os.path.dirname(output_png_path) or ".", exist_ok=True)
    fig.tight_layout(rect=[0.0, 0.03, 1.0, 0.97])
    fig.savefig(output_png_path, dpi=180)
    print(f"whole_body_demo_plotter: wrote {output_png_path}")

    if show_window and os.environ.get("DISPLAY"):
        plt.show()
    else:
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize whole-body demo CSV output."
    )
    parser.add_argument(
        "--csv_path",
        default=os.path.join("output", "polytope_ros", "whole_body_demo.csv"),
        help="Path to whole_body_demo CSV file.",
    )
    parser.add_argument(
        "--output_png_path",
        default=os.path.join("output", "polytope_ros", "whole_body_demo_plot.png"),
        help="Path to output plot image.",
    )
    parser.add_argument(
        "--wait_timeout_sec",
        type=float,
        default=1.0,
        help="How long to wait for the CSV to appear.",
    )
    parser.add_argument(
        "--show_window",
        action="store_true",
        help="Display the plot window if DISPLAY is available.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not wait_for_csv(args.csv_path, args.wait_timeout_sec):
        raise RuntimeError(f"Timed out waiting for CSV: {args.csv_path}")

    data = load_structured_csv(args.csv_path)
    plot_demo(data, args.output_png_path, args.show_window)


if __name__ == "__main__":
    main()
