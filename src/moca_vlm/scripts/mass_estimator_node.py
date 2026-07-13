#!/usr/bin/env python3

import base64
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import rospy

from moca_realsense.srv import CaptureImage
from moca_vlm.msg import MassEstimateRequest, MassEstimateResult


def _default_repo_root() -> Path:
    try:
        import rospkg

        return Path(rospkg.RosPack().get_path("moca_vlm")).resolve().parents[1]
    except Exception:  # noqa: BLE001
        pass

    return Path(__file__).resolve().parents[3]


def _resolve_ros_package_path(package_resource: str) -> Path:
    package_name, _, relative_path = package_resource.partition("/")
    if not package_name:
        raise ValueError(f"Invalid ROS package path: {package_resource}")

    try:
        import rospkg
    except ImportError as exc:
        raise RuntimeError("Resolving package paths requires python3-rospkg.") from exc

    package_path = Path(rospkg.RosPack().get_path(package_name))
    return (package_path / relative_path).expanduser().resolve()


def _resolve_config_path(value: str, *, allow_empty: bool = True) -> str:
    path_text = os.path.expandvars(str(value)).strip()
    if not path_text:
        if allow_empty:
            return ""
        raise ValueError("Empty path in moca_vlm configuration.")

    if path_text.startswith("package://"):
        return str(_resolve_ros_package_path(path_text[len("package://"):]))

    if path_text.startswith("$(find "):
        close_idx = path_text.find(")")
        if close_idx > 0:
            package_name = path_text[len("$(find "):close_idx].strip()
            relative_path = path_text[close_idx + 1:].lstrip("/")
            resource = f"{package_name}/{relative_path}" if relative_path else package_name
            return str(_resolve_ros_package_path(resource))

    return str(Path(path_text).expanduser().resolve())


def _resolve_executable(value: str) -> str:
    executable = os.path.expandvars(str(value)).strip()
    if executable.startswith("package://") or executable.startswith("$(find "):
        return _resolve_config_path(executable, allow_empty=False)
    if "/" in executable or executable.startswith("~"):
        return str(Path(executable).expanduser().resolve())
    return executable


class MassEstimatorNode:
    def __init__(self) -> None:
        self._request_topic = rospy.get_param("~request_topic", "/vlm_mass_request")
        self._result_topic = rospy.get_param("~result_topic", "/vlm_mass_result")
        self._use_api = bool(rospy.get_param("~use_api", False))
        self._model = rospy.get_param("~model", "gpt-4o")
        self._image_source = rospy.get_param("~image_source", "fixed")
        self._api_key = rospy.get_param("~api_key", "")
        self._gemini_api_key = rospy.get_param("~gemini_api_key", "")
        self._local_base_url = rospy.get_param("~local_base_url", "")
        self._default_image_path = _resolve_config_path(
            rospy.get_param("~default_image_path", "")
        )
        self._realsense_capture_service = rospy.get_param(
            "~realsense_capture_service", "/moca_realsense/capture_image"
        )
        self._realsense_capture_timeout_sec = float(
            rospy.get_param("~realsense_capture_timeout_sec", 5.0)
        )
        self._result_json_path = _resolve_config_path(
            rospy.get_param("~result_json_path", "")
        )
        self._auto_save_api_result = bool(rospy.get_param("~auto_save_api_result", True))
        self._reuse_last_result = bool(rospy.get_param("~reuse_last_result", True))
        self._api_python_executable = _resolve_executable(
            rospy.get_param("~api_python_executable", sys.executable)
        )
        self._api_helper_script_path = _resolve_config_path(
            rospy.get_param(
                "~api_helper_script_path",
                str(Path(__file__).resolve().with_name("mass_estimator_api_helper.py")),
            ),
            allow_empty=False,
        )
        self._vendor_python_path = _resolve_config_path(
            rospy.get_param(
                "~vendor_python_path",
                "$(find moca_vlm)/vendor/python",
            )
        )

        repo_root = _default_repo_root()
        manipulicity_repo_path = rospy.get_param("~manipulicity_repo_path", "")
        self._manipulicity_repo_path = (
            Path(_resolve_config_path(manipulicity_repo_path, allow_empty=False))
            if manipulicity_repo_path
            else repo_root / "manipulicity"
        )
        self._mass_estimator_class = None
        self._estimator = None

        self._result_pub = rospy.Publisher(
            self._result_topic, MassEstimateResult, queue_size=1, latch=True
        )
        self._request_sub = rospy.Subscriber(
            self._request_topic, MassEstimateRequest, self._request_callback, queue_size=1
        )

        rospy.loginfo(
            "moca_vlm: ready on request topic '%s' and result topic '%s' (use_api=%s, model=%s, image_source=%s).",
            self._request_topic,
            self._result_topic,
            "true" if self._use_api else "false",
            self._model,
            self._image_source,
        )

    def _request_callback(self, msg: MassEstimateRequest) -> None:
        self._reuse_last_result = bool(
            rospy.get_param("~reuse_last_result", self._reuse_last_result))
        image_path = msg.image_path.strip() if msg.image_path.strip() else self._default_image_path
        image_path = _resolve_config_path(image_path) if image_path else ""
        result_msg = MassEstimateResult()
        result_msg.header.stamp = rospy.Time.now()
        result_msg.header.frame_id = "world"
        result_msg.request_id = msg.request_id
        result_msg.model = self._model
        result_msg.image_path = image_path
        result_msg.result_json_path = self._result_json_path

        try:
            if self._use_api and self._reuse_last_result and not msg.force_refresh:
                result_data = self._load_saved_result(image_path)
                result_msg.source = "saved_json_reuse"
            elif self._use_api:
                image_bytes = None
                image_reference = image_path
                if not image_path and self._image_source == "realsense":
                    image_bytes, image_reference = self._capture_realsense_image()
                    result_msg.image_path = image_reference
                if not image_path and image_bytes is None:
                    raise ValueError(
                        "No input image path was provided in the request or config."
                    )

                result_data = self._estimate_from_api(
                    image_path,
                    image_bytes=image_bytes)
                result_msg.source = "api"

                if self._auto_save_api_result and self._result_json_path:
                    self._save_api_result(result_data, image_reference)
            else:
                result_data = self._load_saved_result(image_path)
                result_msg.source = "saved_json"

            self._fill_result_message(result_msg, result_data)
            result_msg.success = True

            rospy.loginfo(
                "moca_vlm: published mass estimate %.4f kg from %s for request '%s' (json=%s).",
                result_msg.mass_kg,
                result_msg.source,
                result_msg.request_id,
                result_msg.result_json_path,
            )
        except Exception as exc:  # noqa: BLE001
            result_msg.success = False
            result_msg.error_message = str(exc)
            rospy.logerr(
                "moca_vlm: failed to process request '%s': %s",
                msg.request_id,
                result_msg.error_message,
            )

        self._result_pub.publish(result_msg)

    def _capture_realsense_image(self):
        try:
            rospy.wait_for_service(
                self._realsense_capture_service,
                timeout=self._realsense_capture_timeout_sec)
        except rospy.ROSException as exc:
            raise TimeoutError(
                "Timed out waiting for RealSense capture service '{}': {}".format(
                    self._realsense_capture_service, exc)) from exc

        proxy = rospy.ServiceProxy(self._realsense_capture_service, CaptureImage)
        response = proxy()
        if not response.success:
            raise RuntimeError(
                "RealSense capture failed: {}".format(response.error_message))
        if not response.image.data:
            raise RuntimeError("RealSense capture returned an empty image.")

        stamp = response.image.header.stamp
        image_reference = "realsense://{}#{}".format(
            self._realsense_capture_service,
            stamp.to_nsec() if stamp else rospy.Time.now().to_nsec())
        return bytes(response.image.data), image_reference

    def _estimate_from_api(
        self,
        image_path: str,
        *,
        image_bytes: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        command = [
            self._api_python_executable,
            self._api_helper_script_path,
            "--manipulicity_repo_path",
            str(self._manipulicity_repo_path),
            "--model",
            self._model,
        ]
        helper_input = None
        if image_bytes is not None:
            command.append("--image_base64_stdin")
            helper_input = base64.b64encode(image_bytes).decode("ascii")
        else:
            command.extend(["--image", image_path])
        if self._vendor_python_path:
            command.extend(["--vendor_python_path", self._vendor_python_path])
        if self._api_key:
            command.extend(["--api_key", self._api_key])
        if self._gemini_api_key:
            command.extend(["--gemini_api_key", self._gemini_api_key])
        if self._local_base_url:
            command.extend(["--local_base_url", self._local_base_url])

        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            input=helper_input,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            details = stderr if stderr else stdout
            raise RuntimeError(
                "external VLM helper failed"
                f" (returncode={completed.returncode}): {details}"
            )

        try:
            return self._coerce_result_dict(json.loads(completed.stdout))
        except json.JSONDecodeError as exc:
            raise ValueError(
                "external VLM helper did not return valid JSON on stdout:\n"
                f"{completed.stdout}"
            ) from exc

    def _load_mass_estimator_class(self):
        if self._mass_estimator_class is not None:
            return self._mass_estimator_class

        if not self._manipulicity_repo_path.exists():
            raise FileNotFoundError(
                f"manipulicity repo was not found at: {self._manipulicity_repo_path}"
            )

        sys.path.insert(0, str(self._manipulicity_repo_path))
        from mass_estimator import MassEstimator  # pylint: disable=import-error

        self._mass_estimator_class = MassEstimator
        return self._mass_estimator_class

    def _save_api_result(self, result_data: Dict[str, Any], image_path: str) -> None:
        if not self._result_json_path:
            return

        out_path = Path(self._result_json_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        wrapped_result = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "source": "api",
            "model": self._model,
            "image_path": image_path,
            "result": result_data,
        }
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(wrapped_result, handle, indent=2, ensure_ascii=False)

    def _load_saved_result(self, requested_image_path: str) -> Dict[str, Any]:
        if not self._result_json_path:
            raise ValueError("result_json_path is empty, so there is no saved result to load.")

        result_path = Path(self._result_json_path)
        if not result_path.exists():
            raise FileNotFoundError(f"Saved result JSON was not found: {result_path}")

        with open(result_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        if isinstance(data, dict) and "result" in data and isinstance(data["result"], dict):
            self._result_json_path = str(result_path.resolve())
            return self._coerce_result_dict(data["result"])

        if isinstance(data, dict) and "mass_kg" in data:
            self._result_json_path = str(result_path.resolve())
            return self._coerce_result_dict(data)

        if isinstance(data, list):
            selected = self._select_record_from_batch_json(data, requested_image_path)
            self._result_json_path = str(result_path.resolve())
            return selected

        raise ValueError(
            f"Unsupported JSON format in saved result file: {result_path}"
        )

    def _select_record_from_batch_json(
        self, records: List[Dict[str, Any]], requested_image_path: str
    ) -> Dict[str, Any]:
        requested_name = Path(requested_image_path).name if requested_image_path else ""

        for record in records:
            if record.get("status") != "ok":
                continue
            if requested_name and record.get("image") == requested_name:
                return self._coerce_result_dict(record)

        for record in records:
            if record.get("status") == "ok":
                return self._coerce_result_dict(record)

        raise ValueError("No successful mass estimate record was found in the saved JSON file.")

    @staticmethod
    def _coerce_result_dict(result_data: Dict[str, Any]) -> Dict[str, Any]:
        required_keys = {
            "mass_kg",
            "mass_kg_range",
            "material_guess",
            "object_description",
            "confidence",
            "reasoning",
        }
        missing = required_keys - set(result_data.keys())
        if missing:
            raise ValueError(f"Saved result is missing required keys: {sorted(missing)}")

        coerced = dict(result_data)
        coerced["mass_kg"] = float(coerced["mass_kg"])
        coerced["mass_kg_range"] = [float(v) for v in coerced["mass_kg_range"]]
        return coerced

    @staticmethod
    def _fill_result_message(msg: MassEstimateResult, result_data: Dict[str, Any]) -> None:
        msg.mass_kg = float(result_data["mass_kg"])
        msg.mass_kg_range = [float(v) for v in result_data["mass_kg_range"]]
        msg.material_guess = str(result_data["material_guess"])
        msg.object_description = str(result_data["object_description"])
        msg.confidence = str(result_data["confidence"])
        msg.reasoning = str(result_data["reasoning"])


def main() -> None:
    rospy.init_node("moca_vlm_mass_estimator")
    MassEstimatorNode()
    rospy.spin()


if __name__ == "__main__":
    main()
