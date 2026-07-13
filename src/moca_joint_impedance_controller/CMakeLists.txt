cmake_minimum_required(VERSION 3.0.2)
project(moca_joint_impedance_controller)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  roslib
  polytope_ros
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  tf
  visualization_msgs
  moca_trajectory_generator
  hrii_mor_interface
  hrii_ra_interface
  robot_dynamic_pin
  tcpip_polytope
)

find_package(Eigen3 REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS
    roscpp
    roslib
    polytope_ros
    geometry_msgs
    nav_msgs
    sensor_msgs
    std_msgs
    tf
    visualization_msgs
    moca_trajectory_generator
    hrii_mor_interface
    hrii_ra_interface
    robot_dynamic_pin
    tcpip_polytope
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}_lib
  src/joint_impedance_controller.cpp
)

add_dependencies(${PROJECT_NAME}_lib
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(${PROJECT_NAME}_lib
  moca_dynamic
  lib_tcpip
  polytope_ros_core
  ${catkin_LIBRARIES}
)

add_executable(${PROJECT_NAME}_node
  src/main.cpp
)

add_dependencies(${PROJECT_NAME}_node
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(${PROJECT_NAME}_node
  ${PROJECT_NAME}_lib
  ${catkin_LIBRARIES}
)
