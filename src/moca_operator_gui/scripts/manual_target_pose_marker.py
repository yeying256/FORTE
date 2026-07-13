#!/usr/bin/env python3

import copy
import os

import rospy
from geometry_msgs.msg import Pose, PoseStamped
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from visualization_msgs.msg import (
    InteractiveMarker,
    InteractiveMarkerControl,
    InteractiveMarkerFeedback,
    Marker,
)


class ManualTargetPoseMarker:
    def __init__(self) -> None:
        rospy.init_node("moca_manual_target_pose_marker", anonymous=False)

        self.server_name = rospy.get_param("~server_name", "moca_manual_target_pose_marker")
        self.marker_name = rospy.get_param("~marker_name", "manual_target_pose")
        self.marker_description = rospy.get_param("~marker_description", "MOCA Manual Target")
        default_fixed_frame = "{}_odom".format(os.environ.get("ROBOT_ID", "moca_white"))
        self.fixed_frame = rospy.get_param("~fixed_frame", default_fixed_frame)
        self.seed_pose_topic = rospy.get_param("~seed_pose_topic", "/wb_cart_imp_controller/moca_pose")
        self.manual_target_pose_topic = rospy.get_param(
            "~manual_target_pose_topic", "/moca_manual_target_pose")
        self.marker_scale = float(rospy.get_param("~marker_scale", 0.28))
        self.sphere_scale = float(rospy.get_param("~sphere_scale", 0.09))

        self.server = InteractiveMarkerServer(self.server_name)
        self.target_pose_pub = rospy.Publisher(
            self.manual_target_pose_topic, PoseStamped, queue_size=1, latch=True)
        self.seed_pose_sub = rospy.Subscriber(
            self.seed_pose_topic, PoseStamped, self._seed_pose_callback, queue_size=1)

        self.current_pose = Pose()
        self.current_pose.orientation.w = 1.0
        self.marker_initialized = False
        self.user_modified_marker = False

        self._update_marker()
        rospy.loginfo(
            "manual_target_pose_marker: publishing manual target poses on %s in frame %s.",
            self.manual_target_pose_topic,
            self.fixed_frame,
        )

    def _make_sphere_marker(self) -> Marker:
        marker = Marker()
        marker.type = Marker.SPHERE
        marker.scale.x = self.sphere_scale
        marker.scale.y = self.sphere_scale
        marker.scale.z = self.sphere_scale
        marker.color.r = 0.15
        marker.color.g = 0.75
        marker.color.b = 0.95
        marker.color.a = 0.9
        return marker

    def _make_axis_control(self, name, orientation, mode) -> InteractiveMarkerControl:
        control = InteractiveMarkerControl()
        control.orientation.w = orientation[0]
        control.orientation.x = orientation[1]
        control.orientation.y = orientation[2]
        control.orientation.z = orientation[3]
        control.name = name
        control.interaction_mode = mode
        control.orientation_mode = InteractiveMarkerControl.FIXED
        return control

    def _build_interactive_marker(self) -> InteractiveMarker:
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = self.fixed_frame
        int_marker.name = self.marker_name
        int_marker.description = self.marker_description
        int_marker.scale = self.marker_scale
        int_marker.pose = copy.deepcopy(self.current_pose)

        visual_control = InteractiveMarkerControl()
        visual_control.always_visible = True
        visual_control.interaction_mode = InteractiveMarkerControl.MOVE_ROTATE_3D
        visual_control.markers.append(self._make_sphere_marker())
        int_marker.controls.append(visual_control)

        axes = [
            ((1.0, 1.0, 0.0, 0.0), "rotate_x", InteractiveMarkerControl.ROTATE_AXIS),
            ((1.0, 1.0, 0.0, 0.0), "move_x", InteractiveMarkerControl.MOVE_AXIS),
            ((1.0, 0.0, 1.0, 0.0), "rotate_z", InteractiveMarkerControl.ROTATE_AXIS),
            ((1.0, 0.0, 1.0, 0.0), "move_z", InteractiveMarkerControl.MOVE_AXIS),
            ((1.0, 0.0, 0.0, 1.0), "rotate_y", InteractiveMarkerControl.ROTATE_AXIS),
            ((1.0, 0.0, 0.0, 1.0), "move_y", InteractiveMarkerControl.MOVE_AXIS),
        ]
        for orientation, name, mode in axes:
            int_marker.controls.append(self._make_axis_control(name, orientation, mode))

        return int_marker

    def _update_marker(self) -> None:
        self.server.insert(self._build_interactive_marker(), self._process_feedback)
        self.server.applyChanges()

    def _publish_current_pose(self) -> None:
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.fixed_frame
        msg.pose = copy.deepcopy(self.current_pose)
        self.target_pose_pub.publish(msg)

    def _seed_pose_callback(self, msg: PoseStamped) -> None:
        if self.user_modified_marker and self.marker_initialized:
            return

        if msg.header.frame_id:
            self.fixed_frame = msg.header.frame_id
        self.current_pose = copy.deepcopy(msg.pose)
        self.marker_initialized = True
        self._update_marker()
        self._publish_current_pose()

    def _process_feedback(self, feedback: InteractiveMarkerFeedback) -> None:
        if feedback.event_type not in (
            InteractiveMarkerFeedback.POSE_UPDATE,
            InteractiveMarkerFeedback.MOUSE_UP,
        ):
            return

        self.user_modified_marker = True
        self.current_pose = copy.deepcopy(feedback.pose)
        self._publish_current_pose()

    def spin(self) -> None:
        rospy.spin()


if __name__ == "__main__":
    ManualTargetPoseMarker().spin()
