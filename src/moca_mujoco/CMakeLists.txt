cmake_minimum_required(VERSION 3.0.2)
project(moca_mujoco)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS rospy sensor_msgs
)

catkin_install_python(PROGRAMS
  scripts/moca_mujoco_main.py
  scripts/convert_urdf_to_mjcf.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
  config
  launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
