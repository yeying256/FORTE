import importlib
import json
import math
import os
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import numpy as np
try:
    import yaml
except ImportError:
    yaml = None

from .protocol import (
    ArmCommandInterface,
    CommandType,
    MobileCommandInterface,
    TcpBridge,
)


class MocaMujocoEnvironment:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path).expanduser().resolve()
        self.config = self._load_config(self.config_path)

        self.mujoco = self._import_optional_module("mujoco")
        self.viewer_module = None
        try:
            self.viewer_module = importlib.import_module("mujoco.viewer")
        except ImportError:
            self.viewer_module = None

        mujoco_cfg = self.config["mujoco"]
        comm_cfg = self.config["communication"]["tcp"]

        config_dir = self.config_path.parent
        self.urdf_path = self._resolve_config_path(
            mujoco_cfg["urdf_path"], base_dir=config_dir
        )
        self.generated_scene_path = self._resolve_config_path(
            mujoco_cfg["generated_scene_path"], base_dir=config_dir
        )
        self.auto_generate_scene = bool(mujoco_cfg.get("auto_generate_scene", True))
        self.headless = bool(mujoco_cfg.get("headless", False))
        self.use_source_visual_meshes = bool(
            mujoco_cfg.get("use_source_visual_meshes", True)
        )
        self.realtime_rate = float(mujoco_cfg.get("realtime_rate", 1.0))
        self.target_frequency = float(mujoco_cfg.get("target_frequency", 1000.0))
        self.dt = 1.0 / max(1.0, self.target_frequency)

        self.freejoint_name = str(mujoco_cfg.get("freejoint_name", "moca_base_freejoint"))
        self.root_body_name = str(mujoco_cfg.get("root_body_name", "moca_base_footprint"))
        self.ee_body_name = str(mujoco_cfg.get("ee_body_name", "moca_franka_EE"))

        self.init_world_pose = np.asarray(
            mujoco_cfg.get("init_world_pose", [0.0, 0.0, 0.0, 0.0]),
            dtype=np.float64,
        ).reshape(-1)
        self.init_joint_positions = np.asarray(
            mujoco_cfg["init_joint_positions"],
            dtype=np.float64,
        ).reshape(-1)
        self.arm_joint_names = list(mujoco_cfg["arm_joint_names"])
        self.finger_joint_names = list(mujoco_cfg["finger_joint_names"])

        self.mobile_admittance_mass = np.asarray(
            mujoco_cfg.get("mobile_admittance_mass", [105.0, 105.0, 20.0]),
            dtype=np.float64,
        ).reshape(3)
        self.mobile_admittance_damping = np.asarray(
            mujoco_cfg.get("mobile_admittance_damping", [180.0, 180.0, 30.0]),
            dtype=np.float64,
        ).reshape(3)
        self.mobile_velocity_limits = np.asarray(
            mujoco_cfg.get("mobile_velocity_limits", [0.5, 0.5, 1.0]),
            dtype=np.float64,
        ).reshape(3)
        payload_cfg = mujoco_cfg.get("payload", {})
        self.payload_enabled = bool(payload_cfg.get("enabled", True))
        self.payload_name = str(payload_cfg.get("name", "lift_payload"))
        self.payload_position = np.asarray(
            payload_cfg.get("position", [0.78, 0.0, 0.12]),
            dtype=np.float64,
        ).reshape(3)
        self.payload_yaw = float(payload_cfg.get("yaw", 0.0))
        self.payload_box_half_extents = np.asarray(
            payload_cfg.get("box_half_extents", [0.12, 0.08, 0.12]),
            dtype=np.float64,
        ).reshape(3)
        self.payload_box_mass = float(payload_cfg.get("box_mass", 18.0))
        self.payload_box_rgba = np.asarray(
            payload_cfg.get("box_rgba", [0.45, 0.30, 0.18, 1.0]),
            dtype=np.float64,
        ).reshape(4)
        self.payload_handle_span = float(payload_cfg.get("handle_span", 0.12))
        self.payload_handle_height = float(payload_cfg.get("handle_height", 0.10))
        self.payload_handle_radius = float(payload_cfg.get("handle_radius", 0.012))
        self.payload_handle_mass = float(payload_cfg.get("handle_mass", 0.6))
        self.payload_handle_rgba = np.asarray(
            payload_cfg.get("handle_rgba", [0.15, 0.15, 0.15, 1.0]),
            dtype=np.float64,
        ).reshape(4)
        self.payload_friction = np.asarray(
            payload_cfg.get("friction", [1.4, 0.08, 0.02]),
            dtype=np.float64,
        ).reshape(3)
        self.payload_solref = np.asarray(
            payload_cfg.get("solref", [-8000.0, -200.0]),
            dtype=np.float64,
        ).reshape(2)
        self.payload_solimp = np.asarray(
            payload_cfg.get("solimp", [0.95, 0.995, 0.001]),
            dtype=np.float64,
        ).reshape(3)
        gripper_helper_cfg = mujoco_cfg.get("gripper_helper", {})
        self.gripper_helper_enabled = bool(gripper_helper_cfg.get("enabled", True))
        self.gripper_helper_pad_half_extents = np.asarray(
            gripper_helper_cfg.get("pad_half_extents", [0.006, 0.0025, 0.028]),
            dtype=np.float64,
        ).reshape(3)
        self.gripper_helper_pad_center = np.asarray(
            gripper_helper_cfg.get("pad_center", [0.0, 0.0, 0.035]),
            dtype=np.float64,
        ).reshape(3)
        self.gripper_helper_inner_y = float(
            gripper_helper_cfg.get("pad_inner_y", 0.0025)
        )
        self.gripper_helper_rgba = np.asarray(
            gripper_helper_cfg.get("pad_rgba", [0.18, 0.18, 0.18, 1.0]),
            dtype=np.float64,
        ).reshape(4)
        self.gripper_helper_friction = np.asarray(
            gripper_helper_cfg.get("pad_friction", [2.0, 0.08, 0.01]),
            dtype=np.float64,
        ).reshape(3)
        self.gripper_helper_solref = np.asarray(
            gripper_helper_cfg.get("solref", [-4000.0, -120.0]),
            dtype=np.float64,
        ).reshape(2)
        self.gripper_helper_solimp = np.asarray(
            gripper_helper_cfg.get("solimp", [0.95, 0.995, 0.001]),
            dtype=np.float64,
        ).reshape(3)
        grasp_assist_cfg = mujoco_cfg.get("grasp_assist", {})
        self.grasp_assist_enabled = bool(grasp_assist_cfg.get("enabled", True))
        self.grasp_assist_attach_distance = float(
            grasp_assist_cfg.get("attach_distance", 0.03)
        )
        self.grasp_assist_attach_finger_position = float(
            grasp_assist_cfg.get("attach_finger_position", 0.014)
        )
        self.grasp_assist_release_finger_position = float(
            grasp_assist_cfg.get("release_finger_position", 0.020)
        )
        fixed_camera_cfg = mujoco_cfg.get("fixed_camera", {})
        self.fixed_camera_enabled = bool(fixed_camera_cfg.get("enabled", True))
        self.fixed_camera_preview_in_viewer = bool(
            fixed_camera_cfg.get("preview_in_viewer", False)
        )
        self.fixed_camera_publish_to_ros = bool(
            fixed_camera_cfg.get("publish_to_ros", False)
        )
        self.fixed_camera_name = str(
            fixed_camera_cfg.get("name", "payload_fixed_camera")
        )
        self.fixed_camera_ros_frame_id = str(
            fixed_camera_cfg.get("frame_id", self.fixed_camera_name)
        )
        self.fixed_camera_fovy = float(fixed_camera_cfg.get("fovy", 45.0))
        self.fixed_camera_ros_width = int(fixed_camera_cfg.get("width", 640))
        self.fixed_camera_ros_height = int(fixed_camera_cfg.get("height", 480))
        self.fixed_camera_ros_publish_rate_hz = float(
            fixed_camera_cfg.get("publish_rate_hz", 10.0)
        )
        self.fixed_camera_ros_publish_camera_info = bool(
            fixed_camera_cfg.get("publish_camera_info", True)
        )
        self.fixed_camera_ros_topic = str(
            fixed_camera_cfg.get("ros_topic", "~fixed_camera/image_raw")
        )
        self.fixed_camera_ros_camera_info_topic = str(
            fixed_camera_cfg.get("camera_info_topic", "~fixed_camera/camera_info")
        )
        self.fixed_camera_up = np.asarray(
            fixed_camera_cfg.get("up", [0.0, 0.0, 1.0]),
            dtype=np.float64,
        ).reshape(3)
        self.fixed_camera_lookat_offset = np.asarray(
            fixed_camera_cfg.get("lookat_offset", [0.0, 0.0, 0.10]),
            dtype=np.float64,
        ).reshape(3)
        self.fixed_camera_relative_position = np.asarray(
            fixed_camera_cfg.get("relative_position", [-1.1, -1.4, 0.9]),
            dtype=np.float64,
        ).reshape(3)
        fixed_camera_position = fixed_camera_cfg.get("position", [])
        self.fixed_camera_position_override = (
            np.asarray(fixed_camera_position, dtype=np.float64).reshape(3)
            if len(fixed_camera_position) == 3
            else None
        )
        fixed_camera_lookat = fixed_camera_cfg.get("lookat", [])
        self.fixed_camera_lookat_override = (
            np.asarray(fixed_camera_lookat, dtype=np.float64).reshape(3)
            if len(fixed_camera_lookat) == 3
            else None
        )

        self.bridge = TcpBridge(
            ip=str(comm_cfg["ip"]),
            port=int(comm_cfg["port"]),
            target_frequency=float(comm_cfg.get("target_frequency", self.target_frequency)),
        )

        self.model = None
        self.data = None
        self.viewer = None
        self.renderer = None
        self.scene_path = None
        self.rospy = None
        self.ros_image_msg_type = None
        self.ros_camera_info_msg_type = None
        self.fixed_camera_image_publisher = None
        self.fixed_camera_camera_info_publisher = None
        self.fixed_camera_info_msg = None
        self.fixed_camera_last_ros_publish_time = 0.0

        self.arm_joint_ids = []
        self.arm_qpos_adrs = []
        self.arm_dof_adrs = []
        self.finger_joint_ids = []
        self.finger_qpos_adrs = []
        self.finger_dof_adrs = []
        self.freejoint_id = -1
        self.freejoint_qpos_adr = -1
        self.freejoint_dof_adr = -1
        self.root_body_id = -1
        self.ee_body_id = -1
        self.payload_body_id = -1
        self.payload_freejoint_id = -1
        self.payload_freejoint_qpos_adr = -1
        self.payload_freejoint_dof_adr = -1
        self.payload_handle_site_id = -1
        self.left_pad_geom_id = -1
        self.right_pad_geom_id = -1
        self.fixed_camera_id = -1
        self.has_free_base = False
        self.warned_no_free_base = False

        self.base_state = np.zeros(3, dtype=np.float64)
        self.base_velocity_cmd_body = np.zeros(3, dtype=np.float64)
        self.base_velocity_world = np.zeros(3, dtype=np.float64)
        self.arm_torque_command = np.zeros(7, dtype=np.float64)
        self.payload_attached = False
        self.payload_attached_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.waiting_for_connection = True
        self.waiting_for_first_command = True

        self._prepare_scene_and_model()

    def _load_config(self, config_path: Path):
        suffix = config_path.suffix.lower()
        with config_path.open("r", encoding="utf-8") as handle:
            if suffix == ".json":
                return json.load(handle)
            if suffix in {".yaml", ".yml"}:
                if yaml is None:
                    raise RuntimeError(
                        "PyYAML is not installed in the active Python environment. "
                        "Use the provided JSON config or install pyyaml."
                    )
                return yaml.safe_load(handle)

        raise RuntimeError(
            f"Unsupported config format for {config_path}. Use .json, .yaml, or .yml."
        )

    def _resolve_config_path(self, value: str, base_dir: Optional[Path] = None) -> Path:
        path_text = os.path.expandvars(str(value)).strip()
        if not path_text:
            raise ValueError("Empty path in MuJoCo configuration.")

        if path_text.startswith("package://"):
            return self._resolve_ros_package_path(path_text[len("package://"):])

        if path_text.startswith("$(find "):
            close_idx = path_text.find(")")
            if close_idx > 0:
                package_name = path_text[len("$(find "):close_idx].strip()
                relative_path = path_text[close_idx + 1:].lstrip("/")
                resource = (
                    f"{package_name}/{relative_path}" if relative_path else package_name
                )
                return self._resolve_ros_package_path(resource)

        path = Path(path_text).expanduser()
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        return path.resolve()

    @staticmethod
    def _resolve_ros_package_path(package_resource: str) -> Path:
        package_name, _, relative_path = package_resource.partition("/")
        if not package_name:
            raise ValueError(f"Invalid ROS package path: {package_resource}")

        try:
            import rospkg
        except ImportError as exc:
            raise RuntimeError(
                "Resolving package:// paths requires python3-rospkg."
            ) from exc

        package_path = Path(rospkg.RosPack().get_path(package_name))
        return (package_path / relative_path).expanduser().resolve()

    def _import_optional_module(self, module_name: str):
        try:
            return importlib.import_module(module_name)
        except ImportError as exc:
            raise RuntimeError(
                f"MuJoCo Python package '{module_name}' is not installed on this machine. "
                "Please install MuJoCo first, then rerun this environment."
            ) from exc

    def _prepare_scene_and_model(self):
        if self.auto_generate_scene:
            self.scene_path = self._generate_freebase_scene_from_urdf()
        else:
            if not self.generated_scene_path.exists():
                raise FileNotFoundError(
                    "Configured MuJoCo scene file does not exist: "
                    f"{self.generated_scene_path}. Generate it first or enable auto_generate_scene."
                )
            self.scene_path = self.generated_scene_path.resolve()

        self.model = self.mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.model.opt.timestep = self.dt
        self.data = self.mujoco.MjData(self.model)
        self._resolve_model_indices()
        self._initialize_fixed_camera_renderer()

        if not self.headless and self.viewer_module is not None:
            self.viewer = self.viewer_module.launch_passive(self.model, self.data)
            if self.fixed_camera_preview_in_viewer:
                self._configure_viewer_camera()

    def _generate_freebase_scene_from_urdf(self) -> Path:
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")

        self.generated_scene_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_urdf_path = self._write_resolved_urdf_for_mujoco()

        compiled_model = self.mujoco.MjModel.from_xml_path(str(resolved_urdf_path))
        compiled_xml_path = self.generated_scene_path.with_name(
            f"{self.generated_scene_path.stem}_compiled.xml"
        )
        self.mujoco.mj_saveLastXML(str(compiled_xml_path), compiled_model)

        tree = ET.parse(compiled_xml_path)
        root = tree.getroot()
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise RuntimeError("Generated MuJoCo XML has no worldbody.")

        existing_children = list(worldbody)
        if not existing_children:
            raise RuntimeError("Generated MuJoCo XML worldbody is empty.")

        for child in existing_children:
            worldbody.remove(child)

        robot_root = ET.Element("body", {"name": self.root_body_name})
        freejoint_elem = ET.Element("freejoint", {"name": self.freejoint_name})
        robot_root.append(freejoint_elem)

        for child in existing_children:
            robot_root.append(child)

        self._add_gripper_helper_geoms(robot_root)

        worldbody.append(robot_root)

        if worldbody.find("./geom[@name='ground']") is None:
            ground = ET.Element(
                "geom",
                {
                    "name": "ground",
                    "type": "plane",
                    "size": "2 2 0.05",
                    "rgba": "0.85 0.88 0.92 1",
                },
            )
            worldbody.insert(0, ground)

        if worldbody.find("./light[@name='sun']") is None:
            light = ET.Element(
                "light",
                {
                    "name": "sun",
                    "pos": "0 0 3",
                    "dir": "0 0 -1",
                    "diffuse": "1 1 1",
                },
            )
            worldbody.insert(0, light)

        payload_body = self._build_payload_body()
        if payload_body is not None:
            worldbody.append(payload_body)

        fixed_camera = self._build_fixed_camera()
        if fixed_camera is not None:
            worldbody.append(fixed_camera)

        tree.write(self.generated_scene_path, encoding="utf-8", xml_declaration=True)
        print(f"Generated free-base MuJoCo scene at {self.generated_scene_path}")
        return self.generated_scene_path

    def _write_resolved_urdf_for_mujoco(self) -> Path:
        resolved_urdf_path = self.generated_scene_path.with_name(
            f"{self.generated_scene_path.stem}_resolved.urdf"
        )
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()

        for link in root.findall("link"):
            for visual in list(link.findall("visual")):
                if not self.use_source_visual_meshes:
                    link.remove(visual)
                    continue

                if not self._rewrite_geometry_mesh_filenames_for_mujoco(
                    visual,
                    link_name=link.get("name", "<unnamed_link>"),
                    mesh_role="visual",
                ):
                    link.remove(visual)

            for collision in list(link.findall("collision")):
                if not self._rewrite_geometry_mesh_filenames_for_mujoco(
                    collision,
                    link_name=link.get("name", "<unnamed_link>"),
                    mesh_role="collision",
                ):
                    link.remove(collision)

        tree.write(resolved_urdf_path, encoding="utf-8", xml_declaration=True)
        print(f"Resolved MuJoCo URDF written to {resolved_urdf_path}")
        return resolved_urdf_path

    def _rewrite_geometry_mesh_filenames_for_mujoco(
        self,
        geometry_container: ET.Element,
        link_name: str,
        mesh_role: str,
    ) -> bool:
        mesh_elements = list(geometry_container.findall("./geometry/mesh"))
        if not mesh_elements:
            return True

        for mesh in mesh_elements:
            filename = mesh.get("filename")
            if not filename:
                continue

            try:
                resolved_path = self._resolve_mesh_path_for_mujoco(
                    filename,
                    mesh_role=mesh_role,
                )
                staged_mesh_path = self._stage_mesh_for_mujoco(resolved_path)
                mesh.set("filename", staged_mesh_path.name)
            except FileNotFoundError as exc:
                print(
                    f"Skipping {mesh_role} mesh for link '{link_name}': {exc}"
                )
                return False

        return True

    def _resolve_mesh_path_for_mujoco(self, filename: str, mesh_role: str = "collision") -> Path:
        normalized = filename.strip()
        if normalized.startswith("package://"):
            normalized = normalized[len("package://") :]
        elif normalized.startswith("file://"):
            normalized = normalized[len("file://") :]

        mesh_path = Path(normalized)
        if not mesh_path.is_absolute():
            mesh_path = (self.urdf_path.parent / mesh_path).resolve()

        supported_suffixes = {".stl", ".obj", ".msh"}
        if mesh_path.exists() and mesh_path.suffix.lower() in supported_suffixes:
            return mesh_path

        candidate_paths = []
        if mesh_path.suffix.lower() == ".dae":
            candidate_paths.extend(
                [
                    mesh_path.with_suffix(".stl"),
                    mesh_path.with_suffix(".obj"),
                    mesh_path.with_suffix(".msh"),
                ]
            )

            mesh_path_str = str(mesh_path)
            if f"{os.sep}visual{os.sep}" in mesh_path_str:
                collision_peer = Path(
                    mesh_path_str.replace(
                        f"{os.sep}visual{os.sep}",
                        f"{os.sep}collision{os.sep}",
                    )
                )
                candidate_paths.extend(
                    [
                        collision_peer.with_suffix(".stl"),
                        collision_peer.with_suffix(".obj"),
                        collision_peer.with_suffix(".msh"),
                    ]
                )
            elif f"{os.sep}collision{os.sep}" in mesh_path_str:
                visual_peer = Path(
                    mesh_path_str.replace(
                        f"{os.sep}collision{os.sep}",
                        f"{os.sep}visual{os.sep}",
                    )
                )
                candidate_paths.extend(
                    [
                        visual_peer.with_suffix(".stl"),
                        visual_peer.with_suffix(".obj"),
                        visual_peer.with_suffix(".msh"),
                    ]
                )

        for candidate in candidate_paths:
            if candidate.exists() and candidate.suffix.lower() in supported_suffixes:
                return candidate

        raise FileNotFoundError(
            f"no MuJoCo-compatible mesh found for {mesh_role} '{filename}' "
            f"(resolved to {mesh_path})."
        )

    def _stage_mesh_for_mujoco(self, mesh_path: Path) -> Path:
        if not mesh_path.exists():
            raise FileNotFoundError(f"MuJoCo mesh not found after resolution: {mesh_path}")

        staged_mesh_path = self.generated_scene_path.parent / mesh_path.name
        if staged_mesh_path.resolve() != mesh_path.resolve():
            shutil.copy2(mesh_path, staged_mesh_path)
        return staged_mesh_path

    def _resolve_model_indices(self):
        self.root_body_id = self._name_to_id(self.mujoco.mjtObj.mjOBJ_BODY, self.root_body_name)
        self.ee_body_id = self._name_to_id(self.mujoco.mjtObj.mjOBJ_BODY, self.ee_body_name)

        self.freejoint_id = self._name_to_id(
            self.mujoco.mjtObj.mjOBJ_JOINT,
            self.freejoint_name,
            allow_missing=True,
        )
        if self.freejoint_id >= 0:
            self.freejoint_qpos_adr = int(self.model.jnt_qposadr[self.freejoint_id])
            self.freejoint_dof_adr = int(self.model.jnt_dofadr[self.freejoint_id])
            self.has_free_base = True
        else:
            print(
                "MuJoCo model has no free base joint. The environment will keep a virtual base state, "
                "but full mobile-base kinematics in the viewer will require a generated freejoint scene."
            )

        self.payload_body_id = self._name_to_id(
            self.mujoco.mjtObj.mjOBJ_BODY,
            self.payload_name,
            allow_missing=True,
        )
        self.payload_freejoint_id = self._name_to_id(
            self.mujoco.mjtObj.mjOBJ_JOINT,
            f"{self.payload_name}_freejoint",
            allow_missing=True,
        )
        if self.payload_freejoint_id >= 0:
            self.payload_freejoint_qpos_adr = int(
                self.model.jnt_qposadr[self.payload_freejoint_id]
            )
            self.payload_freejoint_dof_adr = int(
                self.model.jnt_dofadr[self.payload_freejoint_id]
            )
        self.payload_handle_site_id = self._name_to_id(
            self.mujoco.mjtObj.mjOBJ_SITE,
            f"{self.payload_name}_handle_grasp",
            allow_missing=True,
        )
        self.left_pad_geom_id = self._name_to_id(
            self.mujoco.mjtObj.mjOBJ_GEOM,
            "moca_gripper_left_pad",
            allow_missing=True,
        )
        self.right_pad_geom_id = self._name_to_id(
            self.mujoco.mjtObj.mjOBJ_GEOM,
            "moca_gripper_right_pad",
            allow_missing=True,
        )
        self.fixed_camera_id = self._name_to_id(
            self.mujoco.mjtObj.mjOBJ_CAMERA,
            self.fixed_camera_name,
            allow_missing=True,
        )

        self.arm_joint_ids = [
            self._name_to_id(self.mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.arm_joint_names
        ]
        self.arm_qpos_adrs = [int(self.model.jnt_qposadr[jid]) for jid in self.arm_joint_ids]
        self.arm_dof_adrs = [int(self.model.jnt_dofadr[jid]) for jid in self.arm_joint_ids]

        self.finger_joint_ids = [
            self._name_to_id(self.mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.finger_joint_names
        ]
        self.finger_qpos_adrs = [int(self.model.jnt_qposadr[jid]) for jid in self.finger_joint_ids]
        self.finger_dof_adrs = [int(self.model.jnt_dofadr[jid]) for jid in self.finger_joint_ids]

    def _name_to_id(self, obj_type, name: str, allow_missing: bool = False) -> int:
        obj_id = int(self.mujoco.mj_name2id(self.model, obj_type, name))
        if obj_id < 0 and not allow_missing:
            raise RuntimeError(f"MuJoCo object '{name}' not found in model {self.scene_path}.")
        return obj_id

    def _vector_to_mjcf_string(self, values: np.ndarray) -> str:
        return " ".join(f"{float(value):.8g}" for value in np.asarray(values).reshape(-1))

    def _sanitize_payload_config(self):
        self.payload_box_half_extents = np.maximum(self.payload_box_half_extents, 1e-3)
        self.payload_box_mass = max(1e-3, self.payload_box_mass)
        self.payload_handle_span = max(2e-2, self.payload_handle_span)
        self.payload_handle_height = max(2e-2, self.payload_handle_height)
        self.payload_handle_radius = max(2e-3, self.payload_handle_radius)
        self.payload_handle_mass = max(1e-4, self.payload_handle_mass)

        # Keep the handle narrower than the payload body so the grasp stays realistic.
        max_span = max(2.0 * self.payload_handle_radius, 1.8 * self.payload_box_half_extents[0])
        self.payload_handle_span = min(self.payload_handle_span, max_span)

    def _build_payload_body(self) -> Optional[ET.Element]:
        if not self.payload_enabled:
            return None

        self._sanitize_payload_config()

        hx, hy, hz = [float(value) for value in self.payload_box_half_extents]
        handle_half_span = 0.5 * self.payload_handle_span
        handle_top_z = hz + self.payload_handle_height
        handle_post_mass = 0.25 * self.payload_handle_mass
        handle_top_mass = 0.50 * self.payload_handle_mass

        payload_body = ET.Element(
            "body",
            {
                "name": self.payload_name,
                "pos": self._vector_to_mjcf_string(self.payload_position),
                "quat": self._vector_to_mjcf_string(
                    self._yaw_to_quaternion_wxyz(self.payload_yaw)
                ),
            },
        )
        payload_body.append(
            ET.Element(
                "freejoint",
                {"name": f"{self.payload_name}_freejoint"},
            )
        )
        payload_body.append(
            ET.Element(
                "geom",
                {
                    "name": f"{self.payload_name}_body",
                    "type": "box",
                    "size": self._vector_to_mjcf_string(self.payload_box_half_extents),
                    "mass": f"{self.payload_box_mass:.8g}",
                    "rgba": self._vector_to_mjcf_string(self.payload_box_rgba),
                    "friction": self._vector_to_mjcf_string(self.payload_friction),
                    "solref": self._vector_to_mjcf_string(self.payload_solref),
                    "solimp": self._vector_to_mjcf_string(self.payload_solimp),
                },
            )
        )

        handle_geoms = (
            (
                f"{self.payload_name}_handle_left",
                (-handle_half_span, 0.0, hz, -handle_half_span, 0.0, handle_top_z),
                handle_post_mass,
            ),
            (
                f"{self.payload_name}_handle_right",
                (handle_half_span, 0.0, hz, handle_half_span, 0.0, handle_top_z),
                handle_post_mass,
            ),
            (
                f"{self.payload_name}_handle_top",
                (-handle_half_span, 0.0, handle_top_z, handle_half_span, 0.0, handle_top_z),
                handle_top_mass,
            ),
        )
        for geom_name, fromto, geom_mass in handle_geoms:
            payload_body.append(
                ET.Element(
                    "geom",
                    {
                        "name": geom_name,
                        "type": "capsule",
                        "fromto": self._vector_to_mjcf_string(np.asarray(fromto)),
                        "size": f"{self.payload_handle_radius:.8g}",
                        "mass": f"{geom_mass:.8g}",
                        "rgba": self._vector_to_mjcf_string(self.payload_handle_rgba),
                        "friction": self._vector_to_mjcf_string(self.payload_friction),
                        "solref": self._vector_to_mjcf_string(self.payload_solref),
                        "solimp": self._vector_to_mjcf_string(self.payload_solimp),
                    },
                )
            )

        payload_body.append(
            ET.Element(
                "site",
                {
                    "name": f"{self.payload_name}_handle_grasp",
                    "pos": self._vector_to_mjcf_string(
                        np.array([0.0, 0.0, handle_top_z], dtype=np.float64)
                    ),
                    "size": "0.008",
                    "rgba": "0.85 0.1 0.1 0.7",
                },
            )
        )

        print(
            "Added MuJoCo payload "
            f"'{self.payload_name}' at {self.payload_position.tolist()} with "
            f"box_mass={self.payload_box_mass:.2f} kg and handle_mass={self.payload_handle_mass:.2f} kg."
        )
        return payload_body

    def _add_gripper_helper_geoms(self, robot_root: ET.Element):
        if not self.gripper_helper_enabled:
            return

        self.gripper_helper_pad_half_extents = np.maximum(
            self.gripper_helper_pad_half_extents,
            1e-4,
        )
        self.gripper_helper_inner_y = max(
            1e-4,
            abs(self.gripper_helper_inner_y),
        )

        left_finger_body = robot_root.find(
            ".//body[@name='moca_franka_franka_gripper_leftfinger']"
        )
        right_finger_body = robot_root.find(
            ".//body[@name='moca_franka_franka_gripper_rightfinger']"
        )

        if left_finger_body is None or right_finger_body is None:
            print(
                "Warning: gripper helper geoms were requested, but finger bodies were not found in the MuJoCo scene."
            )
            return

        self._append_gripper_pad_geom(
            left_finger_body,
            geom_name="moca_gripper_left_pad",
            y_sign=1.0,
        )
        self._append_gripper_pad_geom(
            right_finger_body,
            geom_name="moca_gripper_right_pad",
            y_sign=-1.0,
        )
        print("Added MuJoCo gripper helper pad geoms for stable pinch grasping.")

    def _append_gripper_pad_geom(
        self,
        finger_body: ET.Element,
        geom_name: str,
        y_sign: float,
    ):
        pad_center = self.gripper_helper_pad_center.copy()
        pad_center[1] = y_sign * self.gripper_helper_inner_y
        finger_body.append(
            ET.Element(
                "geom",
                {
                    "name": geom_name,
                    "type": "box",
                    "pos": self._vector_to_mjcf_string(pad_center),
                    "size": self._vector_to_mjcf_string(
                        self.gripper_helper_pad_half_extents
                    ),
                    "rgba": self._vector_to_mjcf_string(self.gripper_helper_rgba),
                    "friction": self._vector_to_mjcf_string(
                        self.gripper_helper_friction
                    ),
                    "solref": self._vector_to_mjcf_string(
                        self.gripper_helper_solref
                    ),
                    "solimp": self._vector_to_mjcf_string(
                        self.gripper_helper_solimp
                    ),
                },
            )
        )

    def _yaw_to_quaternion_wxyz(self, yaw: float) -> np.ndarray:
        half_yaw = 0.5 * float(yaw)
        return np.array(
            [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)],
            dtype=np.float64,
        )

    def _resolve_fixed_camera_lookat(self) -> np.ndarray:
        if self.fixed_camera_lookat_override is not None:
            return self.fixed_camera_lookat_override.copy()

        if self.payload_enabled:
            lookat = self.payload_position.copy()
            lookat[2] += self.payload_box_half_extents[2] + self.payload_handle_height
            return lookat + self.fixed_camera_lookat_offset

        return np.array([1.2, 0.0, 0.5], dtype=np.float64)

    def _resolve_fixed_camera_position(self, lookat: np.ndarray) -> np.ndarray:
        if self.fixed_camera_position_override is not None:
            return self.fixed_camera_position_override.copy()
        return lookat + self.fixed_camera_relative_position

    def _compute_camera_xyaxes(
        self,
        position: np.ndarray,
        lookat: np.ndarray,
    ) -> np.ndarray:
        forward = np.asarray(lookat, dtype=np.float64) - np.asarray(
            position, dtype=np.float64
        )
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-9:
            forward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            forward /= forward_norm

        up = self.fixed_camera_up.copy()
        up_norm = np.linalg.norm(up)
        if up_norm < 1e-9:
            up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            up /= up_norm

        if abs(float(np.dot(forward, up))) > 0.98:
            up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        z_axis = -forward
        x_axis = np.cross(up, z_axis)
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-9:
            up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            x_axis = np.cross(up, z_axis)
            x_norm = np.linalg.norm(x_axis)
        x_axis /= max(x_norm, 1e-9)
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= max(np.linalg.norm(y_axis), 1e-9)
        return np.concatenate([x_axis, y_axis])

    def _build_fixed_camera(self) -> Optional[ET.Element]:
        if not self.fixed_camera_enabled:
            return None

        lookat = self._resolve_fixed_camera_lookat()
        position = self._resolve_fixed_camera_position(lookat)
        xyaxes = self._compute_camera_xyaxes(position, lookat)

        camera = ET.Element(
            "camera",
            {
                "name": self.fixed_camera_name,
                "mode": "fixed",
                "pos": self._vector_to_mjcf_string(position),
                "xyaxes": self._vector_to_mjcf_string(xyaxes),
                "fovy": f"{self.fixed_camera_fovy:.8g}",
            },
        )
        print(
            f"Added MuJoCo fixed camera '{self.fixed_camera_name}' at "
            f"{position.tolist()} looking at {lookat.tolist()}."
        )
        return camera

    def _quat_wxyz_to_yaw(self, quat_wxyz: np.ndarray) -> float:
        w, x, y, z = [float(v) for v in quat_wxyz]
        return math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )

    def _wrap_to_pi(self, angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def _parse_home_world_pose(self) -> np.ndarray:
        raw_pose = self.init_world_pose.reshape(-1)
        if raw_pose.size == 3:
            return np.array([raw_pose[0], raw_pose[1], 0.0], dtype=np.float64)
        if raw_pose.size == 4:
            return np.array([raw_pose[0], raw_pose[1], raw_pose[3]], dtype=np.float64)
        if raw_pose.size == 7:
            yaw = self._quat_wxyz_to_yaw(raw_pose[3:])
            return np.array([raw_pose[0], raw_pose[1], yaw], dtype=np.float64)
        raise ValueError(
            "init_world_pose must be [x, y, z], [x, y, z, yaw], or [x, y, z, qw, qx, qy, qz]."
        )

    def starting(self):
        self._initialize_ros_camera_publishers()
        self.bridge.start()
        self.reset_to_home()
        print("MuJoCo environment initialized and waiting for TCP/IP control.")

    def update(self):
        period = 1.0 / max(1.0, self.target_frequency)
        while not self._should_stop():
            loop_start = time.perf_counter()
            self.bridge.poll_connection()

            if not self.bridge.is_connected:
                if self.waiting_for_connection:
                    print("Waiting for TCP/IP client before stepping MuJoCo world.")
                    self.waiting_for_connection = False
                self.waiting_for_first_command = True
                self.base_velocity_cmd_body.fill(0.0)
                self._publish_state_only()
            elif not self.bridge.has_received_command:
                if self.waiting_for_first_command:
                    print(
                        "TCP/IP client connected. Waiting for first control command before stepping MuJoCo world."
                    )
                    self.waiting_for_first_command = False
                    self.reset_to_home()
                self.waiting_for_connection = True
                self.base_velocity_cmd_body.fill(0.0)
                self._publish_state_only()
            else:
                if not self.waiting_for_first_command:
                    print("First control command received. MuJoCo world stepping enabled.")
                    self.waiting_for_first_command = True
                command = self.bridge.get_command()
                self.apply_command(*command)
                self._step_simulation()
                self._publish_state_only()

            elapsed = time.perf_counter() - loop_start
            desired_period = period / max(1e-6, self.realtime_rate)
            sleep_time = desired_period - elapsed
            if sleep_time > 0.0:
                time.sleep(sleep_time)

    def end(self):
        self.bridge.stop()
        if self.renderer is not None:
            try:
                self.renderer.close()
            except Exception:
                pass
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass

    def apply_command(
        self,
        command_type: CommandType,
        mobile_cmd: np.ndarray,
        arm_cmd: np.ndarray,
        finger: float,
        mobile_interface: MobileCommandInterface,
        arm_interface: ArmCommandInterface,
    ):
        if command_type == CommandType.RESET_TO_HOME:
            self.reset_to_home()
            return

        self.apply_arm_command(arm_cmd, finger, arm_interface)
        self.apply_mobile_command(mobile_cmd, mobile_interface)

    def reset_to_home(self):
        self.base_state = self._parse_home_world_pose()
        self.base_velocity_cmd_body.fill(0.0)
        self.base_velocity_world.fill(0.0)
        self.arm_torque_command.fill(0.0)
        self.payload_attached = False

        self.data.qfrc_applied[:] = 0.0
        self.data.qvel[:] = 0.0

        for i, qpos_adr in enumerate(self.arm_qpos_adrs):
            self.data.qpos[qpos_adr] = self.init_joint_positions[i]
        for i, qpos_adr in enumerate(self.finger_qpos_adrs):
            finger_index = min(7 + i, len(self.init_joint_positions) - 1)
            self.data.qpos[qpos_adr] = self.init_joint_positions[finger_index]

        self._sync_free_base_to_model()
        self._reset_payload_to_initial_pose()
        self.mujoco.mj_forward(self.model, self.data)
        self._sync_viewer()

    def _reset_payload_to_initial_pose(self):
        if not self.payload_enabled or self.payload_freejoint_qpos_adr < 0:
            return

        payload_quat_wxyz = self._yaw_to_quaternion_wxyz(self.payload_yaw)

        self.data.qpos[self.payload_freejoint_qpos_adr + 0] = self.payload_position[0]
        self.data.qpos[self.payload_freejoint_qpos_adr + 1] = self.payload_position[1]
        self.data.qpos[self.payload_freejoint_qpos_adr + 2] = self.payload_position[2]
        self.data.qpos[self.payload_freejoint_qpos_adr + 3] = payload_quat_wxyz[0]
        self.data.qpos[self.payload_freejoint_qpos_adr + 4] = payload_quat_wxyz[1]
        self.data.qpos[self.payload_freejoint_qpos_adr + 5] = payload_quat_wxyz[2]
        self.data.qpos[self.payload_freejoint_qpos_adr + 6] = payload_quat_wxyz[3]

        self.data.qvel[self.payload_freejoint_dof_adr + 0] = 0.0
        self.data.qvel[self.payload_freejoint_dof_adr + 1] = 0.0
        self.data.qvel[self.payload_freejoint_dof_adr + 2] = 0.0
        self.data.qvel[self.payload_freejoint_dof_adr + 3] = 0.0
        self.data.qvel[self.payload_freejoint_dof_adr + 4] = 0.0
        self.data.qvel[self.payload_freejoint_dof_adr + 5] = 0.0

    def apply_arm_command(
        self,
        arm_cmd: np.ndarray,
        finger: float,
        arm_interface: ArmCommandInterface,
    ):
        arm_cmd = np.asarray(arm_cmd, dtype=np.float64).reshape(7)
        current_arm_positions = self._get_arm_positions()

        if arm_interface == ArmCommandInterface.POSITION:
            for i, qpos_adr in enumerate(self.arm_qpos_adrs):
                self.data.qpos[qpos_adr] = arm_cmd[i]
                self.data.qvel[self.arm_dof_adrs[i]] = 0.0
            self.arm_torque_command.fill(0.0)
        elif arm_interface == ArmCommandInterface.VELOCITY:
            integrated_positions = current_arm_positions + arm_cmd * self.dt
            for i, qpos_adr in enumerate(self.arm_qpos_adrs):
                self.data.qpos[qpos_adr] = integrated_positions[i]
                self.data.qvel[self.arm_dof_adrs[i]] = arm_cmd[i]
            self.arm_torque_command.fill(0.0)
        elif arm_interface == ArmCommandInterface.TORQUE:
            self.arm_torque_command = arm_cmd.copy()
        else:
            self.arm_torque_command.fill(0.0)

        self._apply_finger_targets(finger)
        self.mujoco.mj_forward(self.model, self.data)

    def apply_mobile_command(
        self,
        mobile_cmd: np.ndarray,
        mobile_interface: MobileCommandInterface,
    ):
        mobile_cmd = np.asarray(mobile_cmd, dtype=np.float64).reshape(3)

        if mobile_interface == MobileCommandInterface.VELOCITY:
            self.base_velocity_cmd_body = mobile_cmd.copy()
        elif mobile_interface == MobileCommandInterface.WRENCH:
            cmd_acc = (
                mobile_cmd - self.mobile_admittance_damping * self.base_velocity_cmd_body
            ) / self.mobile_admittance_mass
            self.base_velocity_cmd_body += cmd_acc * self.dt
        elif mobile_interface == MobileCommandInterface.POSITION_WORLD:
            self.base_state = mobile_cmd.copy()
            self.base_state[2] = self._wrap_to_pi(self.base_state[2])
            self.base_velocity_cmd_body.fill(0.0)
            self.base_velocity_world.fill(0.0)
            return
        else:
            self.base_velocity_cmd_body.fill(0.0)

        self.base_velocity_cmd_body = np.clip(
            self.base_velocity_cmd_body,
            -self.mobile_velocity_limits,
            self.mobile_velocity_limits,
        )

    def _step_simulation(self):
        self._integrate_base_state()
        self._sync_free_base_to_model()

        self.data.qfrc_applied[:] = 0.0
        for i, dof_adr in enumerate(self.arm_dof_adrs):
            self.data.qfrc_applied[dof_adr] = self.arm_torque_command[i]

        self.mujoco.mj_step(self.model, self.data)
        self._sync_free_base_to_model()
        self._update_grasp_assist()
        self.mujoco.mj_forward(self.model, self.data)
        self._sync_viewer()

    def _integrate_base_state(self):
        yaw = self.base_state[2]
        vx_body, vy_body, wz = self.base_velocity_cmd_body
        vx_world = math.cos(yaw) * vx_body - math.sin(yaw) * vy_body
        vy_world = math.sin(yaw) * vx_body + math.cos(yaw) * vy_body

        self.base_state[0] += vx_world * self.dt
        self.base_state[1] += vy_world * self.dt
        self.base_state[2] = self._wrap_to_pi(self.base_state[2] + wz * self.dt)

        self.base_velocity_world[0] = vx_world
        self.base_velocity_world[1] = vy_world
        self.base_velocity_world[2] = wz

    def _sync_free_base_to_model(self):
        if not self.has_free_base:
            if not self.warned_no_free_base:
                print(
                    "Free base joint is missing; base motion will remain virtual in state feedback."
                )
                self.warned_no_free_base = True
            return

        quat_wxyz = self._yaw_to_quaternion_wxyz(self.base_state[2])
        qpos = self.data.qpos
        qvel = self.data.qvel

        qpos[self.freejoint_qpos_adr + 0] = self.base_state[0]
        qpos[self.freejoint_qpos_adr + 1] = self.base_state[1]
        qpos[self.freejoint_qpos_adr + 2] = 0.0
        qpos[self.freejoint_qpos_adr + 3] = quat_wxyz[0]
        qpos[self.freejoint_qpos_adr + 4] = quat_wxyz[1]
        qpos[self.freejoint_qpos_adr + 5] = quat_wxyz[2]
        qpos[self.freejoint_qpos_adr + 6] = quat_wxyz[3]

        qvel[self.freejoint_dof_adr + 0] = self.base_velocity_world[0]
        qvel[self.freejoint_dof_adr + 1] = self.base_velocity_world[1]
        qvel[self.freejoint_dof_adr + 2] = 0.0
        qvel[self.freejoint_dof_adr + 3] = 0.0
        qvel[self.freejoint_dof_adr + 4] = 0.0
        qvel[self.freejoint_dof_adr + 5] = self.base_velocity_world[2]

    def _apply_finger_targets(self, finger: float):
        for qpos_adr in self.finger_qpos_adrs:
            self.data.qpos[qpos_adr] = finger
        for dof_adr in self.finger_dof_adrs:
            self.data.qvel[dof_adr] = 0.0

    def _quat_wxyz_to_rotation_matrix(self, quat_wxyz: np.ndarray) -> np.ndarray:
        w, x, y, z = [float(v) for v in quat_wxyz]
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    def _get_grasp_center_world(self) -> Optional[np.ndarray]:
        if self.left_pad_geom_id < 0 or self.right_pad_geom_id < 0:
            return None
        return 0.5 * (
            np.array(self.data.geom_xpos[self.left_pad_geom_id], dtype=np.float64) +
            np.array(self.data.geom_xpos[self.right_pad_geom_id], dtype=np.float64)
        )

    def _update_grasp_assist(self):
        if (
            not self.grasp_assist_enabled
            or not self.payload_enabled
            or self.payload_freejoint_qpos_adr < 0
            or self.payload_handle_site_id < 0
        ):
            return

        finger_position = self._get_finger_position()
        grasp_center_world = self._get_grasp_center_world()
        if grasp_center_world is None:
            return

        payload_handle_world = np.array(
            self.data.site_xpos[self.payload_handle_site_id],
            dtype=np.float64,
        )

        if self.payload_attached:
            if finger_position > self.grasp_assist_release_finger_position:
                self.payload_attached = False
                return

            handle_local = np.array(
                [0.0, 0.0, self.payload_box_half_extents[2] + self.payload_handle_height],
                dtype=np.float64,
            )
            rotation = self._quat_wxyz_to_rotation_matrix(self.payload_attached_quat)
            payload_position = grasp_center_world - rotation.dot(handle_local)

            self.data.qpos[self.payload_freejoint_qpos_adr + 0] = payload_position[0]
            self.data.qpos[self.payload_freejoint_qpos_adr + 1] = payload_position[1]
            self.data.qpos[self.payload_freejoint_qpos_adr + 2] = payload_position[2]
            self.data.qpos[self.payload_freejoint_qpos_adr + 3] = self.payload_attached_quat[0]
            self.data.qpos[self.payload_freejoint_qpos_adr + 4] = self.payload_attached_quat[1]
            self.data.qpos[self.payload_freejoint_qpos_adr + 5] = self.payload_attached_quat[2]
            self.data.qpos[self.payload_freejoint_qpos_adr + 6] = self.payload_attached_quat[3]

            self.data.qvel[self.payload_freejoint_dof_adr + 0] = 0.0
            self.data.qvel[self.payload_freejoint_dof_adr + 1] = 0.0
            self.data.qvel[self.payload_freejoint_dof_adr + 2] = 0.0
            self.data.qvel[self.payload_freejoint_dof_adr + 3] = 0.0
            self.data.qvel[self.payload_freejoint_dof_adr + 4] = 0.0
            self.data.qvel[self.payload_freejoint_dof_adr + 5] = 0.0
            return

        distance = np.linalg.norm(grasp_center_world - payload_handle_world)
        if (
            finger_position <= self.grasp_assist_attach_finger_position
            and distance <= self.grasp_assist_attach_distance
        ):
            self.payload_attached = True
            self.payload_attached_quat = self.data.qpos[
                self.payload_freejoint_qpos_adr + 3 : self.payload_freejoint_qpos_adr + 7
            ].copy()

    def _get_arm_positions(self) -> np.ndarray:
        return np.array([self.data.qpos[idx] for idx in self.arm_qpos_adrs], dtype=np.float64)

    def _get_arm_velocities(self) -> np.ndarray:
        return np.array([self.data.qvel[idx] for idx in self.arm_dof_adrs], dtype=np.float64)

    def _get_finger_position(self) -> float:
        if not self.finger_qpos_adrs:
            return 0.0
        return float(self.data.qpos[self.finger_qpos_adrs[0]])

    def _publish_state_only(self):
        arm_positions = self._get_arm_positions()
        arm_velocities = self._get_arm_velocities()
        finger_position = self._get_finger_position()

        self.bridge.fill_state(
            mobile_positions=self.base_state,
            arm_positions=arm_positions,
            mobile_velocities=self.base_velocity_world,
            arm_velocities=arm_velocities,
            finger_position=finger_position,
        )
        self.bridge.send_state()
        self._publish_fixed_camera_ros()
        self._sync_viewer()

    def _sync_viewer(self):
        if self.viewer is None:
            return
        try:
            self.viewer.sync()
        except Exception:
            pass

    def _configure_viewer_camera(self):
        if self.viewer is None or not self.fixed_camera_enabled or self.fixed_camera_id < 0:
            return
        try:
            self.viewer.cam.type = self.mujoco.mjtCamera.mjCAMERA_FIXED
            self.viewer.cam.fixedcamid = int(self.fixed_camera_id)
            self.viewer.sync()
            print(
                f"MuJoCo viewer switched to fixed camera '{self.fixed_camera_name}' "
                f"(id={self.fixed_camera_id})."
            )
        except Exception as exc:
            print(f"Warning: failed to activate fixed camera view: {exc}")

    def _initialize_fixed_camera_renderer(self):
        if (
            not self.fixed_camera_enabled
            or not self.fixed_camera_publish_to_ros
            or self.fixed_camera_id < 0
        ):
            return
        try:
            self.renderer = self.mujoco.Renderer(
                self.model,
                height=max(1, self.fixed_camera_ros_height),
                width=max(1, self.fixed_camera_ros_width),
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize the MuJoCo offscreen renderer for the fixed camera."
            ) from exc

    def _initialize_ros_camera_publishers(self):
        if not self.fixed_camera_publish_to_ros:
            return
        if not self.fixed_camera_enabled or self.fixed_camera_id < 0:
            print(
                "Warning: fixed camera ROS publishing is enabled, but the fixed camera "
                "is not available in the MuJoCo model."
            )
            return

        self.rospy = self._import_optional_module("rospy")
        sensor_msgs = self._import_optional_module("sensor_msgs.msg")
        self.ros_image_msg_type = sensor_msgs.Image
        self.ros_camera_info_msg_type = sensor_msgs.CameraInfo

        rospy_core = getattr(self.rospy, "core", None)
        rospy_is_initialized = (
            getattr(rospy_core, "is_initialized", lambda: False)()
            if rospy_core is not None
            else False
        )
        if not rospy_is_initialized:
            self.rospy.init_node("moca_mujoco", anonymous=False, disable_signals=True)

        self.fixed_camera_image_publisher = self.rospy.Publisher(
            self.fixed_camera_ros_topic,
            self.ros_image_msg_type,
            queue_size=1,
        )
        if self.fixed_camera_ros_publish_camera_info:
            self.fixed_camera_camera_info_publisher = self.rospy.Publisher(
                self.fixed_camera_ros_camera_info_topic,
                self.ros_camera_info_msg_type,
                queue_size=1,
            )
            self.fixed_camera_info_msg = self._build_fixed_camera_info_msg()

        resolved_topic = self.fixed_camera_image_publisher.resolved_name
        print(
            f"MuJoCo fixed camera '{self.fixed_camera_name}' publishing to ROS topic "
            f"'{resolved_topic}'."
        )

    def _build_fixed_camera_info_msg(self):
        if self.ros_camera_info_msg_type is None:
            return None

        info_msg = self.ros_camera_info_msg_type()
        info_msg.header.frame_id = self.fixed_camera_ros_frame_id
        info_msg.width = int(self.fixed_camera_ros_width)
        info_msg.height = int(self.fixed_camera_ros_height)
        info_msg.distortion_model = "plumb_bob"
        info_msg.D = [0.0] * 5

        fovy_rad = math.radians(max(1e-6, self.fixed_camera_fovy))
        fy = 0.5 * float(self.fixed_camera_ros_height) / math.tan(0.5 * fovy_rad)
        fx = fy
        cx = 0.5 * (float(self.fixed_camera_ros_width) - 1.0)
        cy = 0.5 * (float(self.fixed_camera_ros_height) - 1.0)

        info_msg.K = [
            fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0,
        ]
        info_msg.R = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]
        info_msg.P = [
            fx, 0.0, cx, 0.0,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]
        return info_msg

    def _publish_fixed_camera_ros(self):
        if (
            not self.fixed_camera_publish_to_ros
            or self.renderer is None
            or self.rospy is None
            or self.fixed_camera_image_publisher is None
        ):
            return

        publish_period = (
            1.0 / self.fixed_camera_ros_publish_rate_hz
            if self.fixed_camera_ros_publish_rate_hz > 0.0
            else 0.0
        )
        now_wall = time.perf_counter()
        if (
            publish_period > 0.0
            and self.fixed_camera_last_ros_publish_time > 0.0
            and (now_wall - self.fixed_camera_last_ros_publish_time) < publish_period
        ):
            return

        try:
            self.renderer.update_scene(self.data, camera=self.fixed_camera_name)
            pixels = self.renderer.render()
        except Exception as exc:
            print(f"Warning: failed to render fixed camera image: {exc}")
            return

        if pixels is None:
            return

        pixels = np.ascontiguousarray(pixels, dtype=np.uint8)
        if pixels.ndim != 3 or pixels.shape[2] != 3:
            print(
                "Warning: fixed camera ROS publishing expected an RGB image, "
                f"but got shape {pixels.shape}."
            )
            return

        stamp = self.rospy.Time.now()
        image_msg = self.ros_image_msg_type()
        image_msg.header.stamp = stamp
        image_msg.header.frame_id = self.fixed_camera_ros_frame_id
        image_msg.height = int(pixels.shape[0])
        image_msg.width = int(pixels.shape[1])
        image_msg.encoding = "rgb8"
        image_msg.is_bigendian = False
        image_msg.step = int(pixels.shape[1] * pixels.shape[2])
        image_msg.data = pixels.tobytes()
        self.fixed_camera_image_publisher.publish(image_msg)

        if (
            self.fixed_camera_camera_info_publisher is not None
            and self.fixed_camera_info_msg is not None
        ):
            self.fixed_camera_info_msg.header.stamp = stamp
            self.fixed_camera_info_msg.header.frame_id = self.fixed_camera_ros_frame_id
            self.fixed_camera_camera_info_publisher.publish(self.fixed_camera_info_msg)

        self.fixed_camera_last_ros_publish_time = now_wall

    def _should_stop(self) -> bool:
        if self.rospy is not None:
            try:
                if self.rospy.is_shutdown():
                    return True
            except Exception:
                pass
        if self.viewer is not None:
            try:
                return not self.viewer.is_running()
            except Exception:
                return False
        return False
