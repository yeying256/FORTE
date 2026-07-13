#!/usr/bin/env python3
"""Render the MOCA RViz config for the requested robot_id, then exec RViz."""

import argparse
import os
import re
import sys
from pathlib import Path


def sanitized_rviz_environment() -> dict:
    """Keep ROS paths, but stop external Qt installs from overriding RViz."""
    environment = os.environ.copy()

    library_paths = environment.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    kept_library_paths = [
        path
        for path in library_paths
        if path and "/opt/Qt" not in path and "Qt5" not in path
    ]
    if kept_library_paths:
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(kept_library_paths)
    else:
        environment.pop("LD_LIBRARY_PATH", None)

    for variable in (
        "QT_PLUGIN_PATH",
        "QT_QPA_PLATFORM_PLUGIN_PATH",
        "QML2_IMPORT_PATH",
    ):
        environment.pop(variable, None)

    return environment


def render_config(template_text: str, robot_id: str) -> str:
    robot_description = "/{}/robot_description".format(robot_id)
    fixed_frame = "{}_odom".format(robot_id)

    rendered = re.sub(
        r"Robot Description: /[^/\s]+/robot_description",
        "Robot Description: {}".format(robot_description),
        template_text,
    )
    rendered = re.sub(
        r"Fixed Frame: [A-Za-z0-9_]+_odom",
        "Fixed Frame: {}".format(fixed_frame),
        rendered,
    )
    return rendered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start RViz with a robot_id-specific MOCA config.")
    parser.add_argument("--robot-id", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--output", required=True)
    args, _ = parser.parse_known_args()
    return args


def main() -> int:
    args = parse_args()
    template_path = Path(args.template)
    output_path = Path(args.output)

    template_text = template_path.read_text()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_config(template_text, args.robot_id))

    os.execvpe("rviz", ["rviz", "-d", str(output_path)], sanitized_rviz_environment())
    return 1


if __name__ == "__main__":
    sys.exit(main())
