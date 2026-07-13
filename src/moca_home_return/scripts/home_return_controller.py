#!/usr/bin/env python3

import json
import math
import threading
from typing import Any, Dict, List, Optional, Tuple

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


def _yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _clamp(value: float, limit: float) -> float:
    limit = abs(limit)
    return max(-limit, min(limit, value))


def _clamp_norm(x: float, y: float, max_norm: float) -> Tuple[float, float]:
    max_norm = abs(max_norm)
    norm = math.hypot(x, y)
    if max_norm <= 0.0 or norm <= max_norm or norm <= 1.0e-12:
        return x, y
    scale = max_norm / norm
    return x * scale, y * scale


class HomeReturnController:
    def __init__(self) -> None:
        self.state_topic = rospy.get_param("~state_topic", "/wb_cart_imp_controller/moca_state")
        self.odom_topic = rospy.get_param("~odom_topic", "/moca_white/robotnik_base_control/odom")
        self.arm_joint_command_topic = rospy.get_param(
            "~arm_joint_command_topic", "/moca_white/joint_position_controller/command")
        self.base_cmd_vel_topic = rospy.get_param(
            "~base_cmd_vel_topic", "/moca_white/robotnik_base_control/cmd_vel")
        self.home_sample_topic = rospy.get_param("~home_sample_topic", "/moca_home_return/home_sample")
        self.status_topic = rospy.get_param("~status_topic", "/moca_home_return/status")

        self.control_rate_hz = float(rospy.get_param("~control_rate_hz", 30.0))
        self.arm_joint_count = int(rospy.get_param("~arm_joint_count", 7))
        self.arm_motion_duration_sec = float(rospy.get_param("~arm_motion_duration_sec", 8.0))
        self.base_timeout_sec = float(rospy.get_param("~base_timeout_sec", 30.0))
        self.base_kp_xy = float(rospy.get_param("~base_kp_xy", 0.6))
        self.base_ki_xy = float(rospy.get_param("~base_ki_xy", 0.0))
        self.base_kd_xy = float(rospy.get_param("~base_kd_xy", 0.05))
        self.base_kp_yaw = float(rospy.get_param("~base_kp_yaw", 0.8))
        self.base_ki_yaw = float(rospy.get_param("~base_ki_yaw", 0.0))
        self.base_kd_yaw = float(rospy.get_param("~base_kd_yaw", 0.05))
        self.base_integral_limit_xy = float(rospy.get_param("~base_integral_limit_xy", 0.5))
        self.base_integral_limit_yaw = float(rospy.get_param("~base_integral_limit_yaw", 0.5))
        self.base_max_linear_velocity = float(rospy.get_param("~base_max_linear_velocity", 0.25))
        self.base_max_angular_velocity = float(rospy.get_param("~base_max_angular_velocity", 0.35))
        self.base_position_tolerance = float(rospy.get_param("~base_position_tolerance", 0.02))
        self.base_yaw_tolerance = float(rospy.get_param("~base_yaw_tolerance", 0.03))
        self.goal_hold_time_sec = float(rospy.get_param("~goal_hold_time_sec", 0.5))
        self.cmd_vel_in_base_frame = bool(rospy.get_param("~cmd_vel_in_base_frame", True))

        self._lock = threading.Lock()
        self._state: Optional[JointState] = None
        self._odom: Optional[Odometry] = None
        self._home_arm_positions: List[float] = []
        self._home_base_pose: List[float] = []
        self._active = False
        self._last_time: Optional[rospy.Time] = None
        self._start_time: Optional[rospy.Time] = None
        self._goal_enter_time: Optional[rospy.Time] = None
        self._integral_x = 0.0
        self._integral_y = 0.0
        self._integral_yaw = 0.0
        self._last_error_x = 0.0
        self._last_error_y = 0.0
        self._last_error_yaw = 0.0

        self._load_home_params()

        self._arm_pub = rospy.Publisher(
            self.arm_joint_command_topic, JointTrajectory, queue_size=1, latch=True)
        self._base_pub = rospy.Publisher(self.base_cmd_vel_topic, Twist, queue_size=1)
        self._status_pub = rospy.Publisher(self.status_topic, String, queue_size=1, latch=True)

        rospy.Subscriber(self.state_topic, JointState, self._state_callback, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self._odom_callback, queue_size=1)
        rospy.Subscriber(self.home_sample_topic, String, self._home_sample_callback, queue_size=1)

        self._start_service = rospy.Service("~start", Trigger, self._start_callback)
        self._stop_service = rospy.Service("~stop", Trigger, self._stop_callback)
        self._set_current_service = rospy.Service(
            "~set_home_from_current", Trigger, self._set_home_from_current_callback)

        self._timer = rospy.Timer(
            rospy.Duration(1.0 / max(self.control_rate_hz, 1.0)),
            self._control_timer_callback,
        )

        rospy.loginfo(
            "moca_home_return: state='%s', odom='%s', arm_command='%s', base_cmd='%s'.",
            self.state_topic,
            self.odom_topic,
            self.arm_joint_command_topic,
            self.base_cmd_vel_topic,
        )
        self._publish_status("idle")

    def _load_home_params(self) -> None:
        arm_positions = rospy.get_param("~home_arm_joint_positions", [])
        base_pose = rospy.get_param("~home_base_pose", [])
        self._home_arm_positions = [float(v) for v in arm_positions] if arm_positions else []
        self._home_base_pose = [float(v) for v in base_pose] if base_pose else []

    def _state_callback(self, msg: JointState) -> None:
        with self._lock:
            self._state = msg

    def _odom_callback(self, msg: Odometry) -> None:
        with self._lock:
            self._odom = msg

    def _home_sample_callback(self, msg: String) -> None:
        try:
            sample = json.loads(msg.data)
            arm_positions, base_pose = self._target_from_sample(sample)
        except Exception as exc:  # noqa: BLE001
            rospy.logwarn("moca_home_return: rejected home sample: %s", exc)
            self._publish_status("home_sample_rejected", {"error": str(exc)})
            return

        with self._lock:
            self._home_arm_positions = arm_positions
            self._home_base_pose = base_pose
        rospy.loginfo("moca_home_return: updated home sample from topic.")
        self._publish_status("home_updated_from_sample")

    def _target_from_sample(self, sample: Dict[str, Any]) -> Tuple[List[float], List[float]]:
        arm_positions = sample.get("arm_joint_positions", [])
        odom_planar = sample.get("odom", {}).get("planar", {})
        base_pose = [
            odom_planar.get("x"),
            odom_planar.get("y"),
            odom_planar.get("yaw"),
        ]

        if len(arm_positions) != self.arm_joint_count:
            raise ValueError(
                "home sample has {} arm joints, expected {}".format(
                    len(arm_positions), self.arm_joint_count))
        if any(value is None for value in base_pose):
            raise ValueError("home sample does not contain odom.planar x/y/yaw")

        return [float(v) for v in arm_positions], [float(v) for v in base_pose]

    def _current_arm_positions_locked(self) -> List[float]:
        if self._state is None:
            raise RuntimeError("waiting for state topic: {}".format(self.state_topic))
        if len(self._state.position) < self.arm_joint_count:
            raise RuntimeError(
                "state topic has {} positions, expected at least {}".format(
                    len(self._state.position), self.arm_joint_count))
        return list(self._state.position[-self.arm_joint_count:])

    def _current_base_pose_locked(self) -> List[float]:
        if self._odom is None:
            raise RuntimeError("waiting for odom topic: {}".format(self.odom_topic))
        pose = self._odom.pose.pose
        return [pose.position.x, pose.position.y, _yaw_from_quaternion(pose.orientation)]

    def _set_home_from_current_callback(self, _request) -> TriggerResponse:
        with self._lock:
            try:
                self._home_arm_positions = self._current_arm_positions_locked()
                self._home_base_pose = self._current_base_pose_locked()
            except RuntimeError as exc:
                return TriggerResponse(success=False, message=str(exc))

        self._publish_status("home_updated_from_current")
        return TriggerResponse(
            success=True,
            message=json.dumps(
                {
                    "arm_joint_positions": self._home_arm_positions,
                    "base_pose": self._home_base_pose,
                },
                ensure_ascii=False,
            ),
        )

    def _start_callback(self, _request) -> TriggerResponse:
        with self._lock:
            if len(self._home_arm_positions) != self.arm_joint_count:
                return TriggerResponse(
                    success=False,
                    message="home arm target is not set; record/set a teaching sample first",
                )
            if len(self._home_base_pose) != 3:
                return TriggerResponse(
                    success=False,
                    message="home base target is not set; record/set a teaching sample first",
                )
            if self._state is None or self._odom is None:
                return TriggerResponse(
                    success=False,
                    message="waiting for current state and odom before starting home return",
                )

            arm_positions = list(self._home_arm_positions)
            joint_names = list(self._state.name[-self.arm_joint_count:])
            self._reset_pid_locked()
            self._active = True
            self._start_time = rospy.Time.now()
            self._last_time = self._start_time
            self._goal_enter_time = None

        self._publish_arm_trajectory(arm_positions, joint_names)
        self._publish_status("running")
        return TriggerResponse(success=True, message="home return started")

    def _stop_callback(self, _request) -> TriggerResponse:
        with self._lock:
            self._active = False
            self._goal_enter_time = None
        self._publish_zero_base_velocity()
        self._publish_status("stopped")
        return TriggerResponse(success=True, message="home return stopped")

    def _reset_pid_locked(self) -> None:
        self._integral_x = 0.0
        self._integral_y = 0.0
        self._integral_yaw = 0.0
        self._last_error_x = 0.0
        self._last_error_y = 0.0
        self._last_error_yaw = 0.0

    def _publish_arm_trajectory(self, positions: List[float], joint_names: List[str]) -> None:
        trajectory = JointTrajectory()
        trajectory.header.stamp = rospy.Time.now()
        trajectory.joint_names = joint_names
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = rospy.Duration(max(self.arm_motion_duration_sec, 0.1))
        trajectory.points.append(point)
        self._arm_pub.publish(trajectory)
        rospy.loginfo(
            "moca_home_return: published arm position trajectory to %s.",
            self.arm_joint_command_topic,
        )

    def _publish_zero_base_velocity(self) -> None:
        self._base_pub.publish(Twist())

    def _publish_status(self, state: str, extra: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            "state": state,
            "stamp": rospy.Time.now().to_sec(),
            "has_home": (
                len(self._home_arm_positions) == self.arm_joint_count
                and len(self._home_base_pose) == 3
            ),
        }
        if extra:
            payload.update(extra)
        self._status_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    def _control_timer_callback(self, event) -> None:
        with self._lock:
            if not self._active:
                return
            if self._odom is None:
                return
            if len(self._home_base_pose) != 3:
                return

            now = event.current_real
            dt = 1.0 / max(self.control_rate_hz, 1.0)
            if self._last_time is not None:
                measured_dt = (now - self._last_time).to_sec()
                if measured_dt > 1.0e-4:
                    dt = measured_dt
            self._last_time = now

            current_pose = self._current_base_pose_locked()
            target_x, target_y, target_yaw = self._home_base_pose
            err_x = target_x - current_pose[0]
            err_y = target_y - current_pose[1]
            err_yaw = _wrap_angle(target_yaw - current_pose[2])

            elapsed = (now - self._start_time).to_sec() if self._start_time else 0.0
            pos_err = math.hypot(err_x, err_y)
            if pos_err <= self.base_position_tolerance and abs(err_yaw) <= self.base_yaw_tolerance:
                if self._goal_enter_time is None:
                    self._goal_enter_time = now
                hold_time = (now - self._goal_enter_time).to_sec()
                if hold_time >= self.goal_hold_time_sec:
                    self._active = False
                    command = Twist()
                    status = {
                        "position_error": pos_err,
                        "yaw_error": err_yaw,
                        "elapsed": elapsed,
                    }
                else:
                    command = Twist()
                    status = None
            else:
                self._goal_enter_time = None
                self._integral_x = _clamp(
                    self._integral_x + err_x * dt, self.base_integral_limit_xy)
                self._integral_y = _clamp(
                    self._integral_y + err_y * dt, self.base_integral_limit_xy)
                self._integral_yaw = _clamp(
                    self._integral_yaw + err_yaw * dt, self.base_integral_limit_yaw)

                der_x = (err_x - self._last_error_x) / dt
                der_y = (err_y - self._last_error_y) / dt
                der_yaw = _wrap_angle(err_yaw - self._last_error_yaw) / dt
                self._last_error_x = err_x
                self._last_error_y = err_y
                self._last_error_yaw = err_yaw

                vx_world = (
                    self.base_kp_xy * err_x
                    + self.base_ki_xy * self._integral_x
                    + self.base_kd_xy * der_x
                )
                vy_world = (
                    self.base_kp_xy * err_y
                    + self.base_ki_xy * self._integral_y
                    + self.base_kd_xy * der_y
                )
                wz = (
                    self.base_kp_yaw * err_yaw
                    + self.base_ki_yaw * self._integral_yaw
                    + self.base_kd_yaw * der_yaw
                )
                vx_world, vy_world = _clamp_norm(
                    vx_world, vy_world, self.base_max_linear_velocity)
                wz = _clamp(wz, self.base_max_angular_velocity)

                if self.cmd_vel_in_base_frame:
                    yaw = current_pose[2]
                    cos_yaw = math.cos(yaw)
                    sin_yaw = math.sin(yaw)
                    vx_cmd = cos_yaw * vx_world + sin_yaw * vy_world
                    vy_cmd = -sin_yaw * vx_world + cos_yaw * vy_world
                else:
                    vx_cmd = vx_world
                    vy_cmd = vy_world

                command = Twist()
                command.linear.x = vx_cmd
                command.linear.y = vy_cmd
                command.angular.z = wz
                status = None

            if elapsed > self.base_timeout_sec:
                self._active = False
                command = Twist()
                status = {
                    "position_error": pos_err,
                    "yaw_error": err_yaw,
                    "elapsed": elapsed,
                    "timeout": True,
                }

        self._base_pub.publish(command)
        if status is not None:
            state = "reached" if not status.get("timeout", False) else "timeout"
            self._publish_status(state, status)


def main() -> None:
    rospy.init_node("moca_home_return")
    HomeReturnController()
    rospy.spin()


if __name__ == "__main__":
    main()
