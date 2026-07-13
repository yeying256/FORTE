cmake_minimum_required(VERSION 3.0.2)
project(moca_vlm)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  moca_realsense
  roscpp
  rospy
  sensor_msgs
  std_msgs
)

add_message_files(
  FILES
    MassEstimateRequest.msg
    MassEstimateResult.msg
)

generate_messages(
  DEPENDENCIES
    std_msgs
)

catkin_package(
  CATKIN_DEPENDS
    message_runtime
    moca_realsense
    roscpp
    rospy
    sensor_msgs
    std_msgs
)

catkin_install_python(PROGRAMS
  scripts/mass_estimator_api_helper.py
  scripts/mass_estimator_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
