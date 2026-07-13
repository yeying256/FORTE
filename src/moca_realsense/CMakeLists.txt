cmake_minimum_required(VERSION 3.0.2)
project(moca_realsense)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
  roscpp
  rospy
  sensor_msgs
  tf2
  tf2_ros
)

find_package(realsense2 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core highgui imgcodecs imgproc)

add_service_files(
  FILES
    CaptureImage.srv
)

generate_messages(
  DEPENDENCIES
    sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs message_runtime roscpp rospy sensor_msgs tf2 tf2_ros
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(moca_realsense_node src/realsense_camera_node.cpp)
add_dependencies(moca_realsense_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(moca_realsense_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

if(TARGET realsense2::realsense2)
  target_link_libraries(moca_realsense_node realsense2::realsense2)
else()
  target_link_libraries(moca_realsense_node ${realsense2_LIBRARY})
endif()

catkin_install_python(PROGRAMS
  scripts/realsense_topic_test.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS moca_realsense_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch rviz
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
