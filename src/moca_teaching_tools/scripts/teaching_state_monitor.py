#!/usr/bin/env python3

import json
import math
import threading
from typing import Any, Dict, List, Optional

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, String
from std_srvs.srv import Trigger, TriggerResponse


def _yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _pose_to_dict(pose) -> Dict[str, Any]:
    return {
        "position": {
            "x": pose.position.x,
            "y": pose.position.y,
            "z": pose.position.z,
        },
        "orientation": {
            "x": pose.orientation.x,
            "y": pose.orientation.y,
            "z": pose.orientation.z,
            "w": pose.orientation.w,
        },
    }


def _pose_dict_from_planar(x: float, y: float, yaw: float) -> Dict[str, Any]:
    half_yaw = 0.5 * yaw
    return {
        "position": {
            "x": x,
            "y": y,
            "z": 0.0,
        },
        "orientation": {
            "x": 0.0,
            "y": 0.0,
            "z": math.sin(half_yaw),
            "w": math.cos(half_yaw),
        },
    }


def _stamp_to_float(stamp: rospy.Time) -> float:
    return float(stamp.to_sec()) if stamp is not None else 0.0


class TeachingStateMonitor:
    def __init__(self) -> None:
        self.state_topic = rospy.get_param("~state_topic", "/wb_cart_imp_controller/moca_state")
        self.ee_pose_topic = rospy.get_param("~ee_pose_topic", "/wb_cart_imp_controller/moca_pose")
        self.odom_topic = rospy.get_param("~odom_topic", "/moca_white/robotnik_base_control/odom")
        self.current_sample_topic = rospy.get_param(
            "~current_sample_topic", "/moca_teaching/current_sample")
        self.recorded_sample_topic = rospy.get_param(
            "~recorded_sample_topic", "/moca_teaching/recorded_sample")
        self.upward_force_capacity_topic = rospy.get_param(
            "~upward_force_capacity_topic", "/moca_teaching/upward_force_capacity")
        self.manipulability_topic = rospy.get_param(
            "~manipulability_topic", "/moca_teaching/manipulability")
        self.publish_rate_hz = float(rospy.get_param("~publish_rate_hz", 10.0))
        self.arm_joint_count = int(rospy.get_param("~arm_joint_count", 7))
        self.allow_state_odom_fallback = bool(
            rospy.get_param("~allow_state_odom_fallback", True))

        self._lock = threading.Lock()
        self._state: Optional[JointState] = None
        self._odom: Optional[Odometry] = None
        self._ee_pose: Optional[PoseStamped] = None
        self._upward_force_capacity: Optional[float] = None
        self._manipulability: Optional[float] = None
        self._record_count = 0

        self._current_pub = rospy.Publisher(
            self.current_sample_topic, String, queue_size=1, latch=True)
        self._recorded_pub = rospy.Publisher(
            self.recorded_sample_topic, String, queue_size=10, latch=True)

        rospy.Subscriber(self.state_topic, JointState, self._state_callback, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self._odom_callback, queue_size=1)
        rospy.Subscriber(self.ee_pose_topic, PoseStamped, self._ee_pose_callback, queue_size=1)
        rospy.Subscriber(
            self.upward_force_capacity_topic,
            Float64,
            self._upward_force_capacity_callback,
            queue_size=1)
        rospy.Subscriber(
            self.manipulability_topic,
            Float64,
            self._manipulability_callback,
            queue_size=1)

        self._record_service = rospy.Service("~record", Trigger, self._record_callback)
        self._timer = rospy.Timer(
            rospy.Duration(1.0 / max(self.publish_rate_hz, 1.0)),
            self._publish_current_sample,
        )

        rospy.loginfo(
            "moca_teaching_tools: monitoring state='%s', odom='%s', ee_pose='%s', "
            "allow_state_odom_fallback=%s.",
            self.state_topic,
            self.odom_topic,
            self.ee_pose_topic,
            "true" if self.allow_state_odom_fallback else "false",
        )

    def _state_callback(self, msg: JointState) -> None:
        with self._lock:
            self._state = msg

    def _odom_callback(self, msg: Odometry) -> None:
        with self._lock:
            self._odom = msg

    def _ee_pose_callback(self, msg: PoseStamped) -> None:
        with self._lock:
            self._ee_pose = msg

    def _upward_force_capacity_callback(self, msg: Float64) -> None:
        value = float(msg.data)
        with self._lock:
            self._upward_force_capacity = value if math.isfinite(value) else 0.0

    def _manipulability_callback(self, msg: Float64) -> None:
        value = float(msg.data)
        with self._lock:
            self._manipulability = value if math.isfinite(value) else 0.0

    def _missing_inputs(self) -> List[str]:
        missing = []
        if self._state is None:
            missing.append(self.state_topic)
        if self._odom is None and not self.allow_state_odom_fallback:
            missing.append(self.odom_topic)
        if self._ee_pose is None:
            missing.append(self.ee_pose_topic)
        return missing

    def _build_sample_locked(self, record_index: Optional[int] = None) -> Dict[str, Any]:
        missing = self._missing_inputs()
        if missing:
            raise RuntimeError("waiting for topics: {}".format(", ".join(missing)))

        assert self._state is not None
        assert self._ee_pose is not None

        state = self._state
        ee_pose = self._ee_pose
        arm_count = max(0, self.arm_joint_count)
        arm_positions = list(state.position[-arm_count:]) if arm_count else []
        base_positions_from_state = list(state.position[:3]) if len(state.position) >= 3 else []
        ee_pose_dict = _pose_to_dict(ee_pose.pose)
        odom = self._odom

        if odom is not None:
            odom_x = odom.pose.pose.position.x
            odom_y = odom.pose.pose.position.y
            odom_yaw = _yaw_from_quaternion(odom.pose.pose.orientation)
            odom_frame_id = odom.header.frame_id
            odom_child_frame_id = odom.child_frame_id
            odom_stamp = _stamp_to_float(odom.header.stamp)
            odom_pose_dict = _pose_to_dict(odom.pose.pose)
            odom_twist_linear = {
                "x": odom.twist.twist.linear.x,
                "y": odom.twist.twist.linear.y,
                "z": odom.twist.twist.linear.z,
            }
            odom_twist_angular = {
                "x": odom.twist.twist.angular.x,
                "y": odom.twist.twist.angular.y,
                "z": odom.twist.twist.angular.z,
            }
            odom_source = "odom_topic"
        elif self.allow_state_odom_fallback and len(state.position) >= 3:
            odom_x = state.position[0]
            odom_y = state.position[1]
            odom_yaw = state.position[2]
            odom_frame_id = state.header.frame_id or "state_fallback"
            odom_child_frame_id = "base_from_state"
            odom_stamp = _stamp_to_float(state.header.stamp)
            odom_pose_dict = _pose_dict_from_planar(odom_x, odom_y, odom_yaw)
            odom_twist_linear = {"x": 0.0, "y": 0.0, "z": 0.0}
            odom_twist_angular = {"x": 0.0, "y": 0.0, "z": 0.0}
            odom_source = "state_fallback"
        else:
            raise RuntimeError(
                "waiting for odom topic: {} or state with base_x/base_y/base_yaw".format(
                    self.odom_topic))

        vector = (
            arm_positions
            + [
                odom_x,
                odom_y,
                odom_yaw,
                ee_pose.pose.position.x,
                ee_pose.pose.position.y,
                ee_pose.pose.position.z,
                ee_pose.pose.orientation.x,
                ee_pose.pose.orientation.y,
                ee_pose.pose.orientation.z,
                ee_pose.pose.orientation.w,
            ]
        )

        sample = {
            "record_index": record_index,
            "stamp": rospy.Time.now().to_sec(),
            "state_stamp": _stamp_to_float(state.header.stamp),
            "odom_stamp": odom_stamp,
            "ee_pose_stamp": _stamp_to_float(ee_pose.header.stamp),
            "joint_names": list(state.name),
            "joint_positions": list(state.position),
            "arm_joint_positions": arm_positions,
            "base_positions_from_state": base_positions_from_state,
            "odom": {
                "source": odom_source,
                "frame_id": odom_frame_id,
                "child_frame_id": odom_child_frame_id,
                "pose": odom_pose_dict,
                "planar": {
                    "x": odom_x,
                    "y": odom_y,
                    "yaw": odom_yaw,
                },
                "twist": {
                    "linear": odom_twist_linear,
                    "angular": odom_twist_angular,
                },
            },
            "ee_pose": {
                "frame_id": ee_pose.header.frame_id,
                "pose": ee_pose_dict,
            },
            "upward_force_capacity_N": self._upward_force_capacity,
            "manipulability": self._manipulability,
            "vector_order": [
                "arm_joint_positions[0:7]",
                "odom.x",
                "odom.y",
                "odom.yaw",
                "ee.position.x",
                "ee.position.y",
                "ee.position.z",
                "ee.orientation.x",
                "ee.orientation.y",
                "ee.orientation.z",
                "ee.orientation.w",
            ],
            "vector": vector,
        }
        return sample

    def _sample_json(self, sample: Dict[str, Any]) -> str:
        return json.dumps(sample, ensure_ascii=False, separators=(",", ":"))

    def _publish_current_sample(self, _event) -> None:
        with self._lock:
            try:
                sample = self._build_sample_locked()
            except RuntimeError:
                return
        self._current_pub.publish(String(data=self._sample_json(sample)))

    def _record_callback(self, _request) -> TriggerResponse:
        with self._lock:
            try:
                next_record_count = self._record_count + 1
                sample = self._build_sample_locked(next_record_count)
                self._record_count = next_record_count
            except RuntimeError as exc:
                return TriggerResponse(success=False, message=str(exc))

        sample_text = self._sample_json(sample)
        self._recorded_pub.publish(String(data=sample_text))
        rospy.loginfo("moca_teaching_tools: recorded sample %d.", self._record_count)
        return TriggerResponse(success=True, message=sample_text)


def main() -> None:
    rospy.init_node("moca_teaching_state_monitor")
    TeachingStateMonitor()
    rospy.spin()


if __name__ == "__main__":
    main()
