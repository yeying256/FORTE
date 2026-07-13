#!/usr/bin/env python3

import os
import itertools
import math
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import rospy
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    HAS_MPL_3D = True
except Exception:
    HAS_MPL_3D = False

try:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"A NumPy version >=1\.19\.5 and <1\.27\.0 is required for this version of SciPy.*",
            category=UserWarning,
        )
        from scipy.spatial import ConvexHull
except Exception:
    ConvexHull = None


def wait_for_csv(path, timeout_sec):
    start_time = time.time()
    while not rospy.is_shutdown():
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return True
        if time.time() - start_time > timeout_sec:
            return False
        time.sleep(0.2)
    return False


def load_vertices(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def set_equal_limits_2d(axis, xs, ys):
    max_abs = max(float(np.max(np.abs(xs))), float(np.max(np.abs(ys))), 1e-3)
    axis.set_xlim([-max_abs, max_abs])
    axis.set_ylim([-max_abs, max_abs])


def unique_vertices(vertices, decimals=8):
    if vertices.size == 0:
        return vertices
    _, indices = np.unique(np.round(vertices, decimals=decimals), axis=0, return_index=True)
    return vertices[np.sort(indices)]


def order_coplanar_vertices(vertices, indices, normal):
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
        ordered.append((math.atan2(np.dot(rel, basis_v), np.dot(rel, basis_u)), index))
    ordered.sort()
    return [index for _angle, index in ordered]


def compute_fallback_convex_hull_faces(vertices, tol=1e-7):
    vertices = unique_vertices(np.asarray(vertices, dtype=float))
    if vertices.shape[0] < 4:
        return []
    if np.linalg.matrix_rank(vertices - np.mean(vertices, axis=0), tol=tol) < 3:
        return []

    planes = {}
    scale = max(1.0, float(np.nanmax(np.linalg.norm(vertices, axis=1))))
    plane_tol = tol * scale
    for triplet in itertools.combinations(range(vertices.shape[0]), 3):
        a, b, c = vertices[list(triplet)]
        normal = np.cross(b - a, c - a)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < plane_tol:
            continue
        normal = normal / normal_norm
        signed_distances = (vertices - a).dot(normal)
        if np.all(signed_distances <= plane_tol) or np.all(signed_distances >= -plane_tol):
            if signed_distances.sum() > 0.0:
                normal = -normal
                signed_distances = -signed_distances
            coplanar = frozenset(np.where(np.abs(signed_distances) <= plane_tol)[0].tolist())
            if len(coplanar) < 3:
                continue
            planes[tuple(sorted(coplanar))] = (coplanar, normal)

    faces = []
    for coplanar, normal in planes.values():
        ordered = order_coplanar_vertices(vertices, list(coplanar), normal)
        for local_index in range(1, len(ordered) - 1):
            faces.append(vertices[[ordered[0], ordered[local_index], ordered[local_index + 1]]])
    return faces


def plot_polytope_2d_fallback(vertices, output_png_path, title, show_window):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    projections = [
        (0, 1, "XY Projection", "Fx [N]", "Fy [N]"),
        (0, 2, "XZ Projection", "Fx [N]", "Fz [N]"),
        (1, 2, "YZ Projection", "Fy [N]", "Fz [N]"),
    ]

    for axis, (i, j, subtitle, xlabel, ylabel) in zip(axes, projections):
        xs = vertices[:, i]
        ys = vertices[:, j]
        axis.scatter(xs, ys, color="tab:red", s=30, label="vertices")

        if ConvexHull is not None and vertices.shape[0] >= 3:
            points_2d = np.column_stack([xs, ys])
            try:
                hull = ConvexHull(points_2d)
                hull_loop = np.append(hull.vertices, hull.vertices[0])
                axis.plot(points_2d[hull_loop, 0], points_2d[hull_loop, 1], color="tab:blue", alpha=0.8)
            except Exception as exc:
                rospy.logwarn("polytope_plotter: 2D hull failed on %s: %s", subtitle, exc)

        axis.set_title(subtitle)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")
        set_equal_limits_2d(axis, xs, ys)

    fig.suptitle(f"{title} (2D fallback)")
    fig.tight_layout()
    fig.savefig(output_png_path, dpi=180)
    rospy.loginfo("polytope_plotter: wrote %s using 2D fallback", output_png_path)

    if show_window and os.environ.get("DISPLAY"):
        plt.show()
    else:
        plt.close(fig)


def plot_polytope(vertices, output_png_path, title, show_window):
    vertices = unique_vertices(vertices)
    output_dir = os.path.dirname(output_png_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if not HAS_MPL_3D:
        rospy.logwarn("polytope_plotter: matplotlib 3D toolkit unavailable, using 2D fallback.")
        plot_polytope_2d_fallback(vertices, output_png_path, title, show_window)
        return

    fig = plt.figure(figsize=(10, 8))
    try:
        ax = fig.add_subplot(111, projection="3d")
    except Exception as exc:
        plt.close(fig)
        rospy.logwarn("polytope_plotter: failed to create 3D axes (%s), using 2D fallback.", exc)
        plot_polytope_2d_fallback(vertices, output_png_path, title, show_window)
        return

    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color="tab:red", s=35, label="vertices")

    if vertices.shape[0] >= 4:
        try:
            faces = []
            if ConvexHull is not None:
                hull = ConvexHull(vertices)
                faces = [vertices[simplex] for simplex in hull.simplices]
            if not faces:
                faces = compute_fallback_convex_hull_faces(vertices)
            if faces:
                ax.add_collection3d(Poly3DCollection(
                    faces,
                    alpha=0.22,
                    facecolor="tab:blue",
                    edgecolor="tab:blue",
                    linewidth=0.6))
        except Exception as exc:
            rospy.logwarn("polytope_plotter: hull failed, fallback to scatter only: %s", exc)

    ax.set_xlabel("Fx [N]")
    ax.set_ylabel("Fy [N]")
    ax.set_zlabel("Fz [N]")
    ax.set_title(title)
    ax.legend(loc="best")

    if vertices.size > 0:
        max_abs = float(np.max(np.abs(vertices)))
        if max_abs > 1e-9:
            ax.set_xlim([-max_abs, max_abs])
            ax.set_ylim([-max_abs, max_abs])
            ax.set_zlim([-max_abs, max_abs])

    fig.tight_layout()
    fig.savefig(output_png_path, dpi=180)
    rospy.loginfo("polytope_plotter: wrote %s", output_png_path)

    if show_window and os.environ.get("DISPLAY"):
        plt.show()
    else:
        plt.close(fig)


def main():
    rospy.init_node("polytope_plotter")

    vertices_csv_path = rospy.get_param("~vertices_csv_path", "/tmp/polytope_vertices.csv")
    output_png_path = rospy.get_param("~output_png_path", "/tmp/polytope_plot.png")
    wait_timeout_sec = rospy.get_param("~wait_timeout_sec", 10.0)
    show_window = rospy.get_param("~show_window", True)
    plot_title = rospy.get_param("~plot_title", "3D Force Polytope")

    if not wait_for_csv(vertices_csv_path, wait_timeout_sec):
        rospy.logerr("polytope_plotter: timed out waiting for %s", vertices_csv_path)
        return

    vertices = load_vertices(vertices_csv_path)
    if vertices.shape[1] != 3:
        rospy.logerr("polytope_plotter: expected 3 columns, got %s", vertices.shape)
        return

    plot_polytope(vertices, output_png_path, plot_title, show_window)


if __name__ == "__main__":
    main()
