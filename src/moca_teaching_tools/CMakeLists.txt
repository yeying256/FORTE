cmake_minimum_required(VERSION 3.0.2)
project(moca_teaching_tools)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  hrii_robot_msgs
  nav_msgs
  robot_dynamic_pin
  roscpp
  roslib
  rospy
  sensor_msgs
  std_msgs
  std_srvs
  tcpip_polytope
)

find_package(Eigen3 REQUIRED)
find_package(pinocchio REQUIRED)

catkin_package(
  CATKIN_DEPENDS geometry_msgs hrii_robot_msgs nav_msgs robot_dynamic_pin roscpp roslib rospy sensor_msgs std_msgs std_srvs tcpip_polytope
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${pinocchio_INCLUDE_DIRS}
)

add_executable(moca_teaching_upward_force_capacity_node
  src/upward_force_capacity_node.cpp
)
target_link_libraries(moca_teaching_upward_force_capacity_node
  moca_dynamic
  ${catkin_LIBRARIES}
  ${pinocchio_LIBRARIES}
)

catkin_install_python(PROGRAMS
  scripts/teaching_state_monitor.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS moca_teaching_upward_force_capacity_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
