#!/usr/bin/env python3

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

import moca_mujoco.environment as environment_module


class _DummyBridge:
    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def poll_connection(self):
        pass

    @property
    def is_connected(self):
        return False

    @property
    def has_received_command(self):
        return False

    def fill_state(self, **kwargs):
        pass

    def send_state(self):
        pass


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a MOCA URDF into a MuJoCo MJCF XML scene."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Base MuJoCo config (.json/.yaml/.yml).",
    )
    parser.add_argument(
        "--urdf",
        help="Override the URDF path from the config.",
    )
    parser.add_argument(
        "--output-xml",
        help="Output MJCF XML path. Defaults to the path in the config.",
    )
    parser.add_argument(
        "--use-source-visual-meshes",
        dest="use_source_visual_meshes",
        action="store_true",
        help="Prefer source URDF visual meshes when possible.",
    )
    parser.add_argument(
        "--no-source-visual-meshes",
        dest="use_source_visual_meshes",
        action="store_false",
        help="Drop URDF visual meshes and keep the simplified collision-style scene generation.",
    )
    parser.set_defaults(use_source_visual_meshes=None)
    parser.add_argument(
        "--enable-payload",
        dest="payload_enabled",
        action="store_true",
        help="Enable the configured payload object in the generated scene.",
    )
    parser.add_argument(
        "--disable-payload",
        dest="payload_enabled",
        action="store_false",
        help="Disable the configured payload object in the generated scene.",
    )
    parser.set_defaults(payload_enabled=None)
    return parser.parse_args()


def _load_config(config_path: Path):
    suffix = config_path.suffix.lower()
    with config_path.open("r", encoding="utf-8") as handle:
        if suffix == ".json":
            return json.load(handle)
        if suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError(
                    "PyYAML is not available in the active Python environment."
                )
            return yaml.safe_load(handle)
    raise RuntimeError(f"Unsupported config format: {config_path}")


def _write_temp_json_config(config: dict) -> str:
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        delete=False,
        encoding="utf-8",
    ) as handle:
        json.dump(config, handle, indent=2)
        handle.flush()
        return handle.name


def main():
    args = _parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = _load_config(config_path)

    mujoco_cfg = config.setdefault("mujoco", {})
    mujoco_cfg["headless"] = True
    mujoco_cfg["auto_generate_scene"] = True

    if args.urdf:
        mujoco_cfg["urdf_path"] = str(Path(args.urdf).expanduser().resolve())

    if args.output_xml:
        mujoco_cfg["generated_scene_path"] = str(
            Path(args.output_xml).expanduser().resolve()
        )

    if args.use_source_visual_meshes is not None:
        mujoco_cfg["use_source_visual_meshes"] = bool(args.use_source_visual_meshes)

    if args.payload_enabled is not None:
        mujoco_cfg.setdefault("payload", {})
        mujoco_cfg["payload"]["enabled"] = bool(args.payload_enabled)

    original_bridge = environment_module.TcpBridge
    environment_module.TcpBridge = _DummyBridge

    temp_config_path = None
    env = None
    try:
        temp_config_path = _write_temp_json_config(config)
        env = environment_module.MocaMujocoEnvironment(temp_config_path)

        scene_path = Path(env.scene_path).resolve()
        resolved_urdf_path = scene_path.with_name(f"{scene_path.stem}_resolved.urdf")
        compiled_xml_path = scene_path.with_name(f"{scene_path.stem}_compiled.xml")

        print(f"Source URDF: {Path(mujoco_cfg['urdf_path']).resolve()}")
        print(f"Resolved URDF: {resolved_urdf_path}")
        print(f"Compiled MJCF: {compiled_xml_path}")
        print(f"Final MJCF scene: {scene_path}")
        return 0
    except Exception as exc:
        print(f"convert_urdf_to_mjcf failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if env is not None:
            env.end()
        if temp_config_path is not None:
            try:
                os.remove(temp_config_path)
            except OSError:
                pass
        environment_module.TcpBridge = original_bridge


if __name__ == "__main__":
    sys.exit(main())
