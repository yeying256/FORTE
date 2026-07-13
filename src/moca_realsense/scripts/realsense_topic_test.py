#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, PointCloud2


class RealSenseTopicTest:
    def __init__(self):
        self.color_msg = None
        self.depth_msg = None
        self.cloud_msg = None
        self.color_topic = rospy.get_param("~color_topic", "/moca_realsense/color/image_raw")
        self.depth_topic = rospy.get_param("~depth_topic", "/moca_realsense/depth/image_raw")
        self.cloud_topic = rospy.get_param("~cloud_topic", "/moca_realsense/depth/color/points")
        rospy.Subscriber(self.color_topic, Image, self._color_cb, queue_size=1)
        rospy.Subscriber(self.depth_topic, Image, self._depth_cb, queue_size=1)
        rospy.Subscriber(self.cloud_topic, PointCloud2, self._cloud_cb, queue_size=1)

    def _color_cb(self, msg):
        self.color_msg = msg

    def _depth_cb(self, msg):
        self.depth_msg = msg

    def _cloud_cb(self, msg):
        self.cloud_msg = msg

    def spin(self):
        timeout = rospy.Duration(rospy.get_param("~timeout", 10.0))
        deadline = rospy.Time.now() + timeout
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            if self.color_msg and self.depth_msg and self.cloud_msg:
                rospy.loginfo(
                    "moca_realsense test ok: color=%dx%d %s, depth=%dx%d %s, cloud=%dx%d frame=%s",
                    self.color_msg.width,
                    self.color_msg.height,
                    self.color_msg.encoding,
                    self.depth_msg.width,
                    self.depth_msg.height,
                    self.depth_msg.encoding,
                    self.cloud_msg.width,
                    self.cloud_msg.height,
                    self.cloud_msg.header.frame_id,
                )
                return
            rate.sleep()

        missing = []
        if self.color_msg is None:
            missing.append(self.color_topic)
        if self.depth_msg is None:
            missing.append(self.depth_topic)
        if self.cloud_msg is None:
            missing.append(self.cloud_topic)
        rospy.logerr("moca_realsense test failed: missing topics: %s", ", ".join(missing))


if __name__ == "__main__":
    rospy.init_node("moca_realsense_topic_test")
    RealSenseTopicTest().spin()
