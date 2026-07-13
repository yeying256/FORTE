#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _float_or_nan(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_metric_csv(path, value_column=None):
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(str(path))

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("{} has no CSV header".format(path))

        fields = set(reader.fieldnames)
        if "time" not in fields:
            raise ValueError("{} is missing required 'time' column".format(path))

        if value_column:
            selected_value_column = value_column
        elif "value" in fields:
            selected_value_column = "value"
        elif "norm" in fields:
            selected_value_column = "norm"
        else:
            raise ValueError(
                "{} must contain either a 'value' column or a 'norm' column".format(path))

        rows = []
        for row in reader:
            time_value = _float_or_nan(row.get("time"))
            metric_value = _float_or_nan(row.get(selected_value_column))
            if np.isfinite(time_value) and np.isfinite(metric_value):
                rows.append((time_value, metric_value))

    if not rows:
        raise ValueError("{} has no finite samples".format(path))

    data = np.asarray(rows, dtype=float)
    data[:, 0] -= data[0, 0]
    return data, selected_value_column


def write_merged_csv(path, metric_name, data_a, data_b, label_a, label_b):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "time", metric_name])
        for time_value, value in data_a:
            writer.writerow([label_a, time_value, value])
        for time_value, value in data_b:
            writer.writerow([label_b, time_value, value])


def plot_comparison(output_png, metric_name, data_a, data_b, label_a, label_b, title=None):
    output_png = Path(output_png).expanduser()
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(11, 5.8))
    axis.plot(data_a[:, 0], data_a[:, 1], linewidth=2.2, label=label_a)
    axis.plot(data_b[:, 0], data_b[:, 1], linewidth=2.2, label=label_b)

    axis.set_title(title or "{} comparison".format(metric_name.replace("_", " ").title()))
    axis.set_xlabel("time [s]")
    axis.set_ylabel(metric_name.replace("_", " "))
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")

    stats_text = (
        "{}: min={:.4g}, max={:.4g}\n"
        "{}: min={:.4g}, max={:.4g}"
    ).format(
        label_a,
        float(np.min(data_a[:, 1])),
        float(np.max(data_a[:, 1])),
        label_b,
        float(np.min(data_b[:, 1])),
        float(np.max(data_b[:, 1])),
    )
    axis.text(
        0.01,
        0.02,
        stats_text,
        transform=axis.transAxes,
        fontsize=9,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.8})

    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def default_merged_csv_path(output_png):
    output_png = Path(output_png).expanduser()
    return output_png.with_name(output_png.stem + "_merged.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Merge and plot two MOCA scalar metric CSV files.")
    parser.add_argument("--input-a", required=True, help="First CSV path.")
    parser.add_argument("--input-b", required=True, help="Second CSV path.")
    parser.add_argument("--output-png", required=True, help="Output PNG path.")
    parser.add_argument("--metric", default="metric", help="Metric name for labels.")
    parser.add_argument("--value-column", default="", help="CSV value column override.")
    parser.add_argument("--label-a", default="Run 1", help="Legend label for first CSV.")
    parser.add_argument("--label-b", default="Run 2", help="Legend label for second CSV.")
    parser.add_argument("--title", default="", help="Plot title override.")
    parser.add_argument(
        "--merged-csv",
        default="",
        help="Optional combined CSV path. Defaults to output PNG stem + _merged.csv.")
    args = parser.parse_args()

    value_column = args.value_column.strip() or None
    data_a, used_column_a = load_metric_csv(args.input_a, value_column)
    data_b, used_column_b = load_metric_csv(args.input_b, value_column)
    if used_column_a != used_column_b:
        raise ValueError(
            "input files use different value columns: {} vs {}".format(
                used_column_a, used_column_b))

    output_png = Path(args.output_png).expanduser()
    metric_name = args.metric.strip() or used_column_a
    plot_comparison(
        output_png,
        metric_name,
        data_a,
        data_b,
        args.label_a,
        args.label_b,
        title=args.title.strip() or None)

    merged_csv = Path(args.merged_csv).expanduser() if args.merged_csv else default_merged_csv_path(output_png)
    write_merged_csv(merged_csv, metric_name, data_a, data_b, args.label_a, args.label_b)
    print("wrote {}".format(output_png))
    print("wrote {}".format(merged_csv))


if __name__ == "__main__":
    main()
