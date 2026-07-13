# Moca_controller

<!-- AUTO-GENERATED ROS USAGE START -->

## ROS 使用说明

### 功能定位
MOCA 项目的主控制入口包，包含旧 TCP/IP 控制链、位置控制链、HRII Gazebo/硬件控制栈，以及对比实验 launch。

### 编译与环境
```bash
cd <your_catkin_workspace>
catkin_make --pkg Moca_controller
source devel/setup.bash
```

### Launch 文件

- `launch/moca_force_sweep_comparison.launch`: 启动力能力/操作能力扫描对比实验，并配套绘图节点。
  运行示例：`roslaunch Moca_controller moca_force_sweep_comparison.launch`
  常用参数：`trajectory_config_file:=$(find moca_trajectory_generator)/config/target_pose_generator_params.yaml`, `vlm_config_file:=$(find moca_vlm)/config/mass_estimator_params.yaml`, `plotter_config_file:=$(find polytope_ros)/config/force_sweep_comparison_plotter.yaml`, `planner_metrics_plotter_config_file:=$(find polytope_ros)/config/planner_metrics_plotter.yaml`, `launch_vlm_mass_estimator:=false`, `launch_planner_metrics_plotter:=true`, `launch_rviz:=true`, `rviz_config_file:=$(find Moca_controller)/rviz/moca_tf_tree.rviz`
  主要节点：`Moca_controller/Moca_controller_node/name=wb_cart_imp_controller`, `moca_trajectory_generator/target_pose_generator_node/name=target_pose_generator`, `moca_vlm/mass_estimator_node.py/name=moca_vlm_mass_estimator`, `polytope_ros/force_sweep_comparison_plotter.py/name=force_sweep_comparison_plotter`, `polytope_ros/planner_metrics_plotter.py/name=planner_metrics_plotter`, `rviz/rviz/name=moca_tf_rviz`
- `launch/moca_gazebo_stack.launch`: 启动迁移后的 HRII Gazebo 栈、目标桥接、VLM 质量估计和规划可视化。
  运行示例：`roslaunch Moca_controller moca_gazebo_stack.launch`
  常用参数：`interface_type:=SIMULATION`, `robot_model:=moca`, `arm_model:=franka`, `arm_id:=franka`, `robot_id:=moca_red`, `robot_brand:=moca`, `arm_transmission_type:=effort`, `launch_gripper:=franka_gripper`
  主要节点：`moca_trajectory_generator/hrii_target_command_bridge.py/name=hrii_target_command_bridge`, `moca_vlm/mass_estimator_node.py/name=moca_vlm_mass_estimator`, `moca_trajectory_generator/target_pose_generator_node/name=target_pose_generator`, `polytope_ros/planner_metrics_plotter.py/name=planner_metrics_plotter`
- `launch/moca_hardware_stack.launch`: 启动面向实体机器人/实验室硬件结构的 HRII 控制栈入口。
  运行示例：`roslaunch Moca_controller moca_hardware_stack.launch`
  常用参数：`interface_type:=HARDWARE`, `robot_model:=moca`, `arm_model:=franka`, `arm_id:=franka`, `robot_id:=moca_red`, `robot_brand:=moca`, `arm_transmission_type:=effort`, `launch_gripper:=franka_gripper`
  主要节点：`moca_trajectory_generator/hrii_target_command_bridge.py/name=hrii_target_command_bridge`, `moca_vlm/mass_estimator_node.py/name=moca_vlm_mass_estimator`, `moca_trajectory_generator/target_pose_generator_node/name=target_pose_generator`, `polytope_ros/planner_metrics_plotter.py/name=planner_metrics_plotter`
- `launch/moca_impedance_controller.launch`: 启动原 MOCA 阻抗控制器链路，适合旧 TCP/IP/MuJoCo 方向的控制实验。
  运行示例：`roslaunch Moca_controller moca_impedance_controller.launch`
  常用参数：`trajectory_config_file:=$(find moca_trajectory_generator)/config/target_pose_generator_params.yaml`, `vlm_config_file:=$(find moca_vlm)/config/mass_estimator_params.yaml`, `plotter_config_file:=$(find polytope_ros)/config/real_execution_plotter.yaml`, `planner_metrics_plotter_config_file:=$(find polytope_ros)/config/planner_metrics_plotter.yaml`, `launch_vlm_mass_estimator:=false`, `launch_planner_metrics_plotter:=true`, `launch_rviz:=true`, `rviz_config_file:=$(find Moca_controller)/rviz/moca_tf_tree.rviz`
  主要节点：`Moca_controller/Moca_controller_node/name=wb_cart_imp_controller`, `moca_trajectory_generator/target_pose_generator_node/name=target_pose_generator`, `moca_vlm/mass_estimator_node.py/name=moca_vlm_mass_estimator`, `polytope_ros/real_execution_plotter.py/name=real_execution_plotter`, `polytope_ros/planner_metrics_plotter.py/name=planner_metrics_plotter`, `rviz/rviz/name=moca_tf_rviz`
- `launch/moca_impedance_controller_rviz.launch`: 启动阻抗控制器相关 RViz 可视化。
  运行示例：`roslaunch Moca_controller moca_impedance_controller_rviz.launch`
  常用参数：`trajectory_config_file:=$(find moca_trajectory_generator)/config/target_pose_generator_params.yaml`, `plotter_config_file:=$(find polytope_ros)/config/real_execution_plotter.yaml`, `rviz_config_file:=$(find Moca_controller)/rviz/moca_tf_tree.rviz`, `trajectory_config_file:=$(arg trajectory_config_file)`, `plotter_config_file:=$(arg plotter_config_file)`
  主要节点：`rviz/rviz/name=moca_tf_rviz`
- `launch/moca_legacy_stack.launch`: 启动保留归档的旧控制/规划链路。
  运行示例：`roslaunch Moca_controller moca_legacy_stack.launch`
  常用参数：`trajectory_config_file:=$(find moca_trajectory_generator)/config/target_pose_generator_params.yaml`, `vlm_config_file:=$(find moca_vlm)/config/mass_estimator_params.yaml`, `plotter_config_file:=$(find polytope_ros)/config/real_execution_plotter.yaml`, `planner_metrics_plotter_config_file:=$(find polytope_ros)/config/planner_metrics_plotter.yaml`, `launch_vlm_mass_estimator:=false`, `launch_planner_metrics_plotter:=true`, `launch_rviz:=true`, `rviz_config_file:=$(find Moca_controller)/rviz/moca_tf_tree.rviz`
- `launch/moca_position_controller.launch`: 启动直接位置控制实验链路，用于对比阻抗控制效果。
  运行示例：`roslaunch Moca_controller moca_position_controller.launch`
  常用参数：`controller_config_file:=$(find Moca_controller)/config/position_controller_params.yaml`, `trajectory_config_file:=$(find moca_trajectory_generator)/config/target_pose_generator_params.yaml`, `plotter_config_file:=$(find polytope_ros)/config/real_execution_plotter.yaml`, `planner_metrics_plotter_config_file:=$(find polytope_ros)/config/planner_metrics_plotter.yaml`, `launch_planner_metrics_plotter:=true`, `launch_rviz:=true`, `rviz_config_file:=$(find Moca_controller)/rviz/moca_tf_tree.rviz`
  主要节点：`Moca_controller/Moca_position_controller_node/name=wb_cart_imp_controller`, `moca_trajectory_generator/target_pose_generator_node/name=target_pose_generator`, `polytope_ros/real_execution_plotter.py/name=real_execution_plotter`, `polytope_ros/planner_metrics_plotter.py/name=planner_metrics_plotter`, `rviz/rviz/name=moca_tf_rviz`
- `launch/moca_position_force_sweep_comparison.launch`: 启动位置控制版本的力能力扫描对比实验。
  运行示例：`roslaunch Moca_controller moca_position_force_sweep_comparison.launch`
  常用参数：`controller_config_file:=$(find Moca_controller)/config/position_controller_params.yaml`, `trajectory_config_file:=$(find moca_trajectory_generator)/config/target_pose_generator_params.yaml`, `plotter_config_file:=$(find polytope_ros)/config/force_sweep_comparison_plotter.yaml`, `planner_metrics_plotter_config_file:=$(find polytope_ros)/config/planner_metrics_plotter.yaml`, `launch_planner_metrics_plotter:=true`, `launch_rviz:=true`, `rviz_config_file:=$(find Moca_controller)/rviz/moca_tf_tree.rviz`
  主要节点：`Moca_controller/Moca_position_controller_node/name=wb_cart_imp_controller`, `moca_trajectory_generator/target_pose_generator_node/name=target_pose_generator`, `polytope_ros/force_sweep_comparison_plotter.py/name=force_sweep_comparison_plotter`, `polytope_ros/planner_metrics_plotter.py/name=planner_metrics_plotter`, `rviz/rviz/name=moca_tf_rviz`

### 常用检查命令
```bash
rosnode list
rostopic list
roslaunch Moca_controller <launch文件名>.launch
```

### 注意事项
- 运行前确认已经 `source devel/setup.bash`。
- 涉及 Gazebo、RealSense、实体机器人或外部 API 的 launch，需要确认对应硬件、驱动、权限和环境变量已经准备好。
- 本段为工作空间自动整理的使用说明；如果和实验室原始文档冲突，以当前 launch 参数和源码为准。

<!-- AUTO-GENERATED ROS USAGE END -->
