cmake_minimum_required(VERSION 3.0.2)
project(moca_operator_gui)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  hrii_robot_msgs
  interactive_markers
  moca_home_return
  moca_teaching_tools
  moca_trajectory_generator
  moca_vlm
  roscpp
  rospy
  sensor_msgs
  std_msgs
  std_srvs
  visualization_msgs
)
find_package(Eigen3 REQUIRED)
find_package(Franka 0.7.0 REQUIRED)

catkin_package(
  CATKIN_DEPENDS
    geometry_msgs
    hrii_robot_msgs
    interactive_markers
    moca_home_return
    moca_teaching_tools
    moca_trajectory_generator
    moca_vlm
    roscpp
    rospy
    sensor_msgs
    std_msgs
    std_srvs
    visualization_msgs
  DEPENDS
    Franka
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

add_executable(franka_state_bridge_node
  src/franka_state_bridge_node.cpp
)
add_dependencies(franka_state_bridge_node
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(franka_state_bridge_node
  ${catkin_LIBRARIES}
  ${Franka_LIBRARIES}
)

catkin_install_python(PROGRAMS
  scripts/manual_target_pose_marker.py
  scripts/operator_panel_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS franka_state_bridge_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
