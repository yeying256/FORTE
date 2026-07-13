#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run the MOCA MuJoCo environment.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the MuJoCo environment config (.json/.yaml/.yml).",
    )
    return parser.parse_args()


def ensure_ros_python_paths():
    try:
        import rospy  # noqa: F401
        from sensor_msgs.msg import Image  # noqa: F401
        return
    except Exception:
        pass

    workspace_root = Path(__file__).resolve().parents[3]
    candidate_paths = []

    env_pythonpath = os.environ.get("PYTHONPATH", "")
    for entry in env_pythonpath.split(os.pathsep):
        if entry:
            candidate_paths.append(Path(entry))

    candidate_paths.extend(
        [
            workspace_root / "devel/lib/python3/dist-packages",
            Path("/opt/ros/noetic/lib/python3/dist-packages"),
            Path("/usr/lib/python3/dist-packages"),
        ]
    )

    for path in candidate_paths:
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.append(path_str)


def main():
    args = parse_args()
    ensure_ros_python_paths()

    from moca_mujoco.environment import MocaMujocoEnvironment

    env = None
    try:
        env = MocaMujocoEnvironment(args.config)
        env.starting()
        env.update()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"moca_mujoco failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if env is not None:
            env.end()

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
