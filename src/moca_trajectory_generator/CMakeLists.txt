cmake_minimum_required(VERSION 3.0.2)
project(moca_trajectory_generator)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  geometry_msgs
  hrii_robot_msgs
  moca_vlm
  polytope_ros
  roscpp
  roslib
  rospy
  sensor_msgs
  std_msgs
  tf
  visualization_msgs
)

find_package(pinocchio REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP REQUIRED jsoncpp)

add_message_files(
  FILES
    ForceCapabilitySettings.msg
    TargetPoseCommand.msg
)

add_service_files(
  FILES
    StartPlanning.srv
)

generate_messages(
  DEPENDENCIES
    geometry_msgs
    std_msgs
)

catkin_package(
  CATKIN_DEPENDS
    message_runtime
    geometry_msgs
    hrii_robot_msgs
    moca_vlm
    polytope_ros
    roscpp
    roslib
    rospy
    sensor_msgs
    std_msgs
    tf
    visualization_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${JSONCPP_INCLUDE_DIRS}
  /usr/include/eigen3
)

add_executable(target_pose_generator_node
  src/target_pose_generator.cpp
)

add_dependencies(target_pose_generator_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(target_pose_generator_node
  ${catkin_LIBRARIES}
  ${JSONCPP_LIBRARIES}
  polytope_ros_core
  pinocchio::pinocchio
)

catkin_install_python(PROGRAMS
  scripts/cycle_state_plotter.py
  scripts/hrii_target_command_bridge.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
