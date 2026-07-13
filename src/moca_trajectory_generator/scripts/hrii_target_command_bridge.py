#!/usr/bin/env python3

import math

import rospy
from geometry_msgs.msg import PoseStamped, Vector3Stamped, Wrench
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String

from moca_trajectory_generator.msg import TargetPoseCommand


WHOLE_BODY_JOINT_NAMES = [
    "base_x",
    "base_y",
    "base_yaw",
    "moca_franka_joint1",
    "moca_franka_joint2",
    "moca_franka_joint3",
    "moca_franka_joint4",
    "moca_franka_joint5",
    "moca_franka_joint6",
    "moca_franka_joint7",
]


def _all_finite(values):
    return all(math.isfinite(value) for value in values)


def _pose_is_finite(pose):
    return _all_finite(
        [
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]
    )


def _wrap_to_pi(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _clamp(value, lower, upper):
    return max(lower, min(upper, value))


class HRIITargetCommandBridge:
    def __init__(self):
        target_pose_topic = rospy.get_param("~target_pose_topic", "/target_pose")

        self.old_pose_topic = rospy.get_param(
            "~old_pose_topic",
            "/moca_red/cartesian_impedance_controller/moca_pose",
        )
        self.old_state_topic = rospy.get_param(
            "~old_state_topic",
            "/moca_red/cartesian_impedance_controller/moca_state",
        )

        self.compat_pose_topic = rospy.get_param(
            "~compat_pose_topic", "/wb_cart_imp_controller/moca_pose"
        )
        self.compat_state_topic = rospy.get_param(
            "~compat_state_topic", "/wb_cart_imp_controller/moca_state"
        )
        self.compat_pose_target_topic = rospy.get_param(
            "~compat_pose_target_topic", "/wb_cart_imp_controller/moca_pose_target"
        )
        self.compat_state_target_topic = rospy.get_param(
            "~compat_state_target_topic", "/wb_cart_imp_controller/moca_state_target"
        )
        self.compat_desired_force_topic = rospy.get_param(
            "~compat_desired_force_topic", "/wb_cart_imp_controller/desired_force"
        )

        self.equilibrium_pose_topic = rospy.get_param(
            "~equilibrium_pose_topic",
            "/moca_red/cartesian_impedance_controller/equilibrium_pose",
        )
        self.mobile_base_des_joint_pos_topic = rospy.get_param(
            "~mobile_base_des_joint_pos_topic",
            "/moca_red/cartesian_impedance_controller/mobile_base_des_joint_pos",
        )
        self.arm_des_joint_pos_topic = rospy.get_param(
            "~arm_des_joint_pos_topic",
            "/moca_red/cartesian_impedance_controller/arm_des_joint_pos",
        )
        self.moca_des_joint_pos_topic = rospy.get_param(
            "~moca_des_joint_pos_topic",
            "/moca_red/cartesian_impedance_controller/moca_des_joint_pos",
        )
        self.nullspace_stiffness_matrix_topic = rospy.get_param(
            "~nullspace_stiffness_matrix_topic",
            "/moca_red/cartesian_impedance_controller/nullspace_stiffness_matrix",
        )
        self.weighted_jacobian_topic = rospy.get_param(
            "~weighted_jacobian_topic",
            "/moca_red/cartesian_impedance_controller/weighted_jacobian",
        )
        self.external_base_wrench_topic = rospy.get_param(
            "~external_base_wrench_topic",
            "/moca_red/ext_torque",
        )
        self.base_nullspace_linear_stiffness = float(
            rospy.get_param("~base_nullspace_linear_stiffness", 400.0)
        )
        self.base_nullspace_yaw_stiffness = float(
            rospy.get_param("~base_nullspace_yaw_stiffness", 100.0)
        )
        self.arm_nullspace_joint_stiffness = float(
            rospy.get_param("~arm_nullspace_joint_stiffness", 10.0)
        )
        self.active_weighted_jacobian_mode = rospy.get_param(
            "~active_weighted_jacobian_mode", "base"
        )
        self.idle_weighted_jacobian_mode = rospy.get_param(
            "~idle_weighted_jacobian_mode", "equal"
        )
        self.enable_base_planar_wrench_tracking = bool(
            rospy.get_param("~enable_base_planar_wrench_tracking", False)
        )
        self.base_planar_kp_xy = float(
            rospy.get_param("~base_planar_kp_xy", 250.0)
        )
        self.base_planar_kd_xy = float(
            rospy.get_param("~base_planar_kd_xy", 80.0)
        )
        self.base_planar_kp_yaw = float(
            rospy.get_param("~base_planar_kp_yaw", 120.0)
        )
        self.base_planar_kd_yaw = float(
            rospy.get_param("~base_planar_kd_yaw", 20.0)
        )
        self.base_planar_max_force_xy = float(
            rospy.get_param("~base_planar_max_force_xy", 300.0)
        )
        self.base_planar_max_torque_yaw = float(
            rospy.get_param("~base_planar_max_torque_yaw", 80.0)
        )

        self.current_base_positions = [0.0, 0.0, 0.0]
        self.current_base_velocities = [0.0, 0.0, 0.0]
        self.have_current_base_state = False

        self.pose_pub = rospy.Publisher(self.compat_pose_topic, PoseStamped, queue_size=10)
        self.state_pub = rospy.Publisher(self.compat_state_topic, JointState, queue_size=10)
        self.pose_target_pub = rospy.Publisher(
            self.compat_pose_target_topic, PoseStamped, queue_size=10
        )
        self.state_target_pub = rospy.Publisher(
            self.compat_state_target_topic, JointState, queue_size=10
        )
        self.desired_force_pub = rospy.Publisher(
            self.compat_desired_force_topic, Vector3Stamped, queue_size=10
        )

        self.equilibrium_pose_pub = rospy.Publisher(
            self.equilibrium_pose_topic, PoseStamped, queue_size=10
        )
        self.mobile_base_joint_pub = rospy.Publisher(
            self.mobile_base_des_joint_pos_topic, PoseStamped, queue_size=10
        )
        self.arm_joint_pub = rospy.Publisher(
            self.arm_des_joint_pos_topic, Float64MultiArray, queue_size=10
        )
        self.moca_joint_pub = rospy.Publisher(
            self.moca_des_joint_pos_topic, Float64MultiArray, queue_size=10
        )
        self.nullspace_stiffness_pub = rospy.Publisher(
            self.nullspace_stiffness_matrix_topic,
            Float64MultiArray,
            queue_size=1,
            latch=True,
        )
        self.weighted_jacobian_pub = rospy.Publisher(
            self.weighted_jacobian_topic,
            String,
            queue_size=1,
            latch=True,
        )
        self.external_base_wrench_pub = rospy.Publisher(
            self.external_base_wrench_topic,
            Wrench,
            queue_size=10,
        )

        rospy.Subscriber(target_pose_topic, TargetPoseCommand, self.target_pose_callback, queue_size=10)
        rospy.Subscriber(self.old_pose_topic, PoseStamped, self.old_pose_callback, queue_size=50)
        rospy.Subscriber(self.old_state_topic, JointState, self.old_state_callback, queue_size=50)

        self._publish_nullspace_stiffness_matrix()
        self._publish_weighted_jacobian_mode(self.idle_weighted_jacobian_mode)
        self._publish_external_base_wrench([0.0, 0.0, 0.0], False)

        rospy.loginfo(
            "hrii_target_command_bridge: bridging %s -> %s and %s",
            target_pose_topic,
            self.equilibrium_pose_topic,
            self.moca_des_joint_pos_topic,
        )

    def _publish_nullspace_stiffness_matrix(self):
        stiffness_msg = Float64MultiArray()
        stiffness_msg.data = [
            self.base_nullspace_linear_stiffness,
            self.base_nullspace_linear_stiffness,
            self.base_nullspace_yaw_stiffness,
        ] + [self.arm_nullspace_joint_stiffness] * 7
        self.nullspace_stiffness_pub.publish(stiffness_msg)

    def _publish_weighted_jacobian_mode(self, mode):
        mode_msg = String()
        mode_msg.data = str(mode)
        self.weighted_jacobian_pub.publish(mode_msg)

    def old_pose_callback(self, msg: PoseStamped):
        self.pose_pub.publish(msg)

    def old_state_callback(self, msg: JointState):
        expected_size = len(WHOLE_BODY_JOINT_NAMES)
        if len(msg.position) < expected_size:
            rospy.logwarn_throttle(
                1.0,
                "hrii_target_command_bridge: ignoring moca_state with %d positions, expected %d",
                len(msg.position),
                expected_size,
            )
            return

        bridged = JointState()
        bridged.header = msg.header
        bridged.name = WHOLE_BODY_JOINT_NAMES[:]
        bridged.position = list(msg.position[:expected_size])
        if len(msg.velocity) >= expected_size:
            bridged.velocity = list(msg.velocity[:expected_size])
        else:
            bridged.velocity = [0.0] * expected_size
        if len(msg.effort) >= expected_size:
            bridged.effort = list(msg.effort[:expected_size])
        else:
            bridged.effort = [0.0] * expected_size
        self.state_pub.publish(bridged)

        self.current_base_positions = list(bridged.position[:3])
        self.current_base_velocities = list(bridged.velocity[:3])
        self.have_current_base_state = True

    def _publish_external_base_wrench(self, desired_base_positions, active):
        wrench_msg = Wrench()

        if active:
            if not self.have_current_base_state:
                rospy.logwarn_throttle(
                    1.0,
                    "hrii_target_command_bridge: base wrench tracking is active but no current whole-body state has been received yet.",
                )
            else:
                x_error = desired_base_positions[0] - self.current_base_positions[0]
                y_error = desired_base_positions[1] - self.current_base_positions[1]
                yaw_error = _wrap_to_pi(
                    desired_base_positions[2] - self.current_base_positions[2]
                )

                wrench_msg.force.x = _clamp(
                    self.base_planar_kp_xy * x_error
                    - self.base_planar_kd_xy * self.current_base_velocities[0],
                    -self.base_planar_max_force_xy,
                    self.base_planar_max_force_xy,
                )
                wrench_msg.force.y = _clamp(
                    self.base_planar_kp_xy * y_error
                    - self.base_planar_kd_xy * self.current_base_velocities[1],
                    -self.base_planar_max_force_xy,
                    self.base_planar_max_force_xy,
                )
                wrench_msg.torque.z = _clamp(
                    self.base_planar_kp_yaw * yaw_error
                    - self.base_planar_kd_yaw * self.current_base_velocities[2],
                    -self.base_planar_max_torque_yaw,
                    self.base_planar_max_torque_yaw,
                )

        self.external_base_wrench_pub.publish(wrench_msg)

    def target_pose_callback(self, msg: TargetPoseCommand):
        base_positions = list(msg.base_planar_positions)
        arm_positions = list(msg.arm_joint_positions)

        if (
            not _pose_is_finite(msg.pose)
            or not _all_finite([msg.desired_force.x, msg.desired_force.y, msg.desired_force.z])
            or not _all_finite(base_positions)
            or not _all_finite(arm_positions)
        ):
            rospy.logwarn_throttle(
                1.0,
                "hrii_target_command_bridge: rejecting target command with non-finite pose, force, or joint target.",
            )
            return

        pose_msg = PoseStamped()
        pose_msg.header = msg.header
        pose_msg.pose = msg.pose
        self.equilibrium_pose_pub.publish(pose_msg)
        self.pose_target_pub.publish(pose_msg)

        force_msg = Vector3Stamped()
        force_msg.header = msg.header
        force_msg.vector = msg.desired_force
        self.desired_force_pub.publish(force_msg)

        target_state = JointState()
        target_state.header = msg.header
        target_state.name = WHOLE_BODY_JOINT_NAMES[:]
        target_state.position = base_positions + arm_positions
        target_state.velocity = [0.0] * len(WHOLE_BODY_JOINT_NAMES)
        target_state.effort = [0.0] * len(WHOLE_BODY_JOINT_NAMES)
        self.state_target_pub.publish(target_state)

        self._publish_external_base_wrench(
            base_positions,
            bool(
                msg.use_nullspace_joint_target
                and self.enable_base_planar_wrench_tracking
            ),
        )

        if not msg.use_nullspace_joint_target:
            self._publish_weighted_jacobian_mode(self.idle_weighted_jacobian_mode)
            return

        self._publish_nullspace_stiffness_matrix()
        self._publish_weighted_jacobian_mode(self.active_weighted_jacobian_mode)

        base_pose_msg = PoseStamped()
        base_pose_msg.header = msg.header
        base_pose_msg.pose.position.x = base_positions[0]
        base_pose_msg.pose.position.y = base_positions[1]
        base_pose_msg.pose.position.z = 0.0

        yaw = base_positions[2]
        base_pose_msg.pose.orientation.z = math.sin(0.5 * yaw)
        base_pose_msg.pose.orientation.w = math.cos(0.5 * yaw)
        self.mobile_base_joint_pub.publish(base_pose_msg)

        arm_msg = Float64MultiArray()
        arm_msg.data = arm_positions
        self.arm_joint_pub.publish(arm_msg)

        whole_body_msg = Float64MultiArray()
        whole_body_msg.data = base_positions + arm_positions
        self.moca_joint_pub.publish(whole_body_msg)


if __name__ == "__main__":
    rospy.init_node("hrii_target_command_bridge")
    HRIITargetCommandBridge()
    rospy.spin()
