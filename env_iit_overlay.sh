#!/usr/bin/env bash

_moca_overlay_return() {
  return "$1" 2>/dev/null || exit "$1"
}

_moca_find_casadi_prefix() {
  if [ -n "${MOCA_CASADI_PREFIX:-}" ] && \
     [ -f "${MOCA_CASADI_PREFIX}/lib/cmake/casadi/casadi-config.cmake" ]; then
    printf '%s\n' "${MOCA_CASADI_PREFIX}"
    return 0
  fi

  if [ -n "${CASADI_PREFIX:-}" ] && \
     [ -f "${CASADI_PREFIX}/lib/cmake/casadi/casadi-config.cmake" ]; then
    printf '%s\n' "${CASADI_PREFIX}"
    return 0
  fi

  local system_prefix=""
  for system_prefix in /usr/local /usr; do
    if [ -f "${system_prefix}/lib/cmake/casadi/casadi-config.cmake" ]; then
      printf '%s\n' "${system_prefix}"
      return 0
    fi
  done

  return 1
}

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_space="${workspace_dir}/src"
iit_ws="${IIT_WS:-/home/${USER}/workspaces/catkin_ws}"

_moca_has_local_hrii_sources() {
  local required_paths=(
    "${source_space}/hrii_moca/hrii_moca_interface/package.xml"
    "${source_space}/hrii_robot_interface/hrii_mor_interface/package.xml"
    "${source_space}/hrii_robot_interface/hrii_ra_interface/package.xml"
    "${source_space}/hrii_robot_controllers/hrii_general_robot_controllers/package.xml"
    "${source_space}/hrii_utils/package.xml"
  )

  local path=""
  for path in "${required_paths[@]}"; do
    if [ ! -f "${path}" ]; then
      return 1
    fi
  done

  return 0
}

_moca_has_local_hrii_devel() {
  local required_paths=(
    "${workspace_dir}/devel/lib/Moca_controller/Moca_controller_node"
    "${workspace_dir}/devel/lib/hrii_gri_interface/gripper_srv_node"
    "${workspace_dir}/devel/lib/libhrii_ra_robot_interface.so"
    "${workspace_dir}/devel/lib/libhrii_franka_joint_interface_plugin.so"
    "${workspace_dir}/devel/lib/libMobileRobotROSInterface.so"
  )

  local path=""
  for path in "${required_paths[@]}"; do
    if [ ! -f "${path}" ]; then
      return 1
    fi
  done

  return 0
}

if [ ! -f /opt/ros/noetic/setup.bash ]; then
  echo "env_iit_overlay: missing /opt/ros/noetic/setup.bash" >&2
  _moca_overlay_return 1
fi
source /opt/ros/noetic/setup.bash

if ! _moca_has_local_hrii_devel; then
  if [ ! -f "${iit_ws}/devel/setup.bash" ]; then
    if _moca_has_local_hrii_sources; then
      echo "env_iit_overlay: bundled HRII sources found, but this workspace has not been rebuilt yet and ${iit_ws}/devel/setup.bash is also missing" >&2
    else
      echo "env_iit_overlay: missing bundled HRII sources and missing ${iit_ws}/devel/setup.bash" >&2
    fi
    _moca_overlay_return 1
  fi
  source "${iit_ws}/devel/setup.bash"
fi

if casadi_env_prefix="$(_moca_find_casadi_prefix)"; then
  export CASADI_PREFIX="${casadi_env_prefix}"
fi

if [ -f "${workspace_dir}/devel/setup.bash" ]; then
  export CATKIN_SETUP_UTIL_ARGS=--extend
  source "${workspace_dir}/devel/setup.bash"
  unset CATKIN_SETUP_UTIL_ARGS
fi

if _moca_has_local_hrii_devel; then
  echo "Sourced ROS Noetic + bundled HRII workspace + system CasADi + Moca_polytope overlay."
elif _moca_has_local_hrii_sources; then
  echo "Sourced ROS Noetic + external IIT underlay + bundled HRII sources waiting for rebuild + system CasADi + Moca_polytope overlay."
else
  echo "Sourced ROS Noetic + IIT underlay + system CasADi + Moca_polytope overlay (if built)."
fi
