cmake_minimum_required(VERSION 3.0.2)
project(polytope_ros)

add_compile_options(-std=c++14)

set(CASADI_PREFIX "" CACHE PATH "Prefix where CasADi is installed")
if(NOT CASADI_PREFIX)
  if(DEFINED ENV{CASADI_PREFIX} AND EXISTS "$ENV{CASADI_PREFIX}/lib/cmake/casadi/casadi-config.cmake")
    set(CASADI_PREFIX "$ENV{CASADI_PREFIX}")
  endif()
endif()

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
  std_srvs
)

find_package(pinocchio REQUIRED) 
find_package(Eigen3 REQUIRED)
if(CASADI_PREFIX)
  set(_casadi_config_dir "${CASADI_PREFIX}/lib/cmake/casadi")
  set(_casadi_runtime_lib_dir "${CASADI_PREFIX}/lib")
  find_package(casadi CONFIG REQUIRED
    PATHS ${_casadi_config_dir}
    NO_DEFAULT_PATH)
else()
  find_package(casadi CONFIG REQUIRED)
  if(casadi_DIR)
    get_filename_component(_casadi_runtime_lib_dir "${casadi_DIR}/../.." ABSOLUTE)
  endif()
endif()

# Older CasADi packages export an imported target named `casadi` instead of
# the newer namespaced `casadi::casadi` target expected by this workspace.
if(TARGET casadi AND NOT TARGET casadi::casadi)
  add_library(casadi::casadi INTERFACE IMPORTED)
  set_property(TARGET casadi::casadi PROPERTY INTERFACE_LINK_LIBRARIES casadi)
endif()

find_package(robot_dynamic_pin REQUIRED)







## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a exec_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a exec_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   std_msgs
# )

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a exec_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES polytope_ros_core
  CATKIN_DEPENDS geometry_msgs roscpp rospy sensor_msgs std_msgs std_srvs
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  /usr/include/eigen3
)

## Declare a C++ library
add_library(polytope_ros_core
  src/polytope.cpp
  src/rfp_comparison_algorithm.cpp
  src/rfp_comparison_costs.cpp
  src/nullspace_optimizer.cpp
  src/whole_body_optimizer.cpp
)

add_library(polytope_ros_casadi INTERFACE)
target_link_libraries(polytope_ros_casadi INTERFACE casadi::casadi)
if(_casadi_runtime_lib_dir)
  if(EXISTS "${_casadi_runtime_lib_dir}/libstdc++.so.6")
    target_link_libraries(
      polytope_ros_casadi
      INTERFACE
      "${_casadi_runtime_lib_dir}/libstdc++.so.6")
  elseif(EXISTS "${_casadi_runtime_lib_dir}/libstdc++.so")
    target_link_libraries(
      polytope_ros_casadi
      INTERFACE
      "${_casadi_runtime_lib_dir}/libstdc++.so")
  endif()
  if(EXISTS "${_casadi_runtime_lib_dir}/libgcc_s.so.1")
    target_link_libraries(
      polytope_ros_casadi
      INTERFACE
      "${_casadi_runtime_lib_dir}/libgcc_s.so.1")
  endif()
  target_link_options(
    polytope_ros_casadi
    INTERFACE
    "-Wl,-rpath,${_casadi_runtime_lib_dir}")
endif()

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
add_executable(polytope_node src/main.cpp)
add_executable(whole_body_demo_node src/whole_body_demo.cpp)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
# target_link_libraries(${PROJECT_NAME}_node
#   ${catkin_LIBRARIES}
# )

target_link_libraries(polytope_ros_core
  ${catkin_LIBRARIES}
  pinocchio::pinocchio
  polytope_ros_casadi
)

target_link_libraries(polytope_node
  ${catkin_LIBRARIES}
  pinocchio::pinocchio
  polytope_ros_core
)

target_link_libraries(whole_body_demo_node
  ${catkin_LIBRARIES}
  pinocchio::pinocchio
  polytope_ros_core
)


catkin_install_python(PROGRAMS
  scripts/polytope_plotter.py
  scripts/whole_body_demo_plotter.py
  scripts/whole_body_demo_live_plotter.py
  scripts/planner_metrics_plotter.py
  scripts/real_execution_plotter.py
  scripts/force_sweep_comparison_plotter.py
  scripts/experiment_data_recorder.py
  scripts/compare_metric_csvs.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# catkin_install_python(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
# install(TARGETS ${PROJECT_NAME}_node
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark libraries for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
# install(TARGETS ${PROJECT_NAME}
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_polytope_ros.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
