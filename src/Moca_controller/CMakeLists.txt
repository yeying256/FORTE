cmake_minimum_required(VERSION 3.0.2)
project(Moca_controller)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  polytope_ros
  roscpp
  roslib
  geometry_msgs
  nav_msgs
  moca_trajectory_generator
  sensor_msgs
  std_msgs
  visualization_msgs
  realtime_tools
  tf
  eigen_conversions
  hrii_robot_msgs
  hrii_mor_interface
  hrii_ra_interface
  hrii_utils

  robot_dynamic_pin
  tcpip_polytope
)

find_package(matlogger2 REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS
    roscpp
    roslib
    polytope_ros
    sensor_msgs
    geometry_msgs
    nav_msgs
    moca_trajectory_generator
    std_msgs
    visualization_msgs
    realtime_tools
    tf
    eigen_conversions
    hrii_robot_msgs
    hrii_mor_interface
    hrii_ra_interface
    hrii_utils

    robot_dynamic_pin
    tcpip_polytope

)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${matlogger2_INCLUDE_DIRS}
  # ${matlogger2_LIBRARIES}
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(
# include
#   ${catkin_INCLUDE_DIRS}
# )

## Declare a C++ library
add_library(Moca_controller_lib
  src/Moca_controller.cpp
)

add_dependencies(Moca_controller_lib
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(Moca_controller_lib
  moca_dynamic
  lib_tcpip
  polytope_ros_core
  ${matlogger2_LIBRARIES}
  ${catkin_LIBRARIES}
)

add_executable(Moca_controller_node src/main.cpp)

add_dependencies(Moca_controller_node
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(Moca_controller_node
  Moca_controller_lib
  ${catkin_LIBRARIES}
)

add_library(Moca_position_controller_lib
  src/Moca_position_controller.cpp
)

add_dependencies(Moca_position_controller_lib
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(Moca_position_controller_lib
  moca_dynamic
  lib_tcpip
  ${catkin_LIBRARIES}
)

add_executable(Moca_position_controller_node src/position_main.cpp)

add_dependencies(Moca_position_controller_node
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(Moca_position_controller_node
  Moca_position_controller_lib
  moca_dynamic
  lib_tcpip
  ${catkin_LIBRARIES}
)

catkin_install_python(PROGRAMS
  scripts/moca_rviz_with_robot_id.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
