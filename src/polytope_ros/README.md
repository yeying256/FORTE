# polytope_ros

<!-- AUTO-GENERATED ROS USAGE START -->

## ROS 使用说明

### 功能定位
可操作度/力多面体与全身规划实验包，负责 polytope 计算、规划指标输出和可视化脚本。

### 编译与环境
```bash
cd <your_catkin_workspace>
catkin_make --pkg polytope_ros
source devel/setup.bash
```

### Launch 文件

- `launch/polytope_ros.launch`: 启动 polytope 计算与可视化节点。
  运行示例：`roslaunch polytope_ros polytope_ros.launch`
  常用参数：`urdf_path:=$(find polytope_ros)/../../urdf/my_robot_fix_wheels.urdf`, `ee_frame:=moca_franka_EE`, `vertices_csv_path:=$(find polytope_ros)/../../output/polytope_ros/polytope_vertices.csv`, `output_png_path:=$(find polytope_ros)/../../output/polytope_ros/polytope_plot.png`, `show_window:=true`
  主要节点：`polytope_ros/polytope_node/name=polytope_node`, `polytope_ros/polytope_plotter.py/name=polytope_plotter`
- `launch/real_execution_plotter.launch`: 启动真实/仿真执行数据绘图节点。
  运行示例：`roslaunch polytope_ros real_execution_plotter.launch`
  常用参数：`trajectory_config_file:=$(find moca_trajectory_generator)/config/target_pose_generator_params.yaml`, `plotter_config_file:=$(find polytope_ros)/config/real_execution_plotter.yaml`
  主要节点：`polytope_ros/real_execution_plotter.py/name=real_execution_plotter`
- `launch/whole_body_demo.launch`: 启动全身规划 demo，生成规划指标和可视化结果。
  运行示例：`roslaunch polytope_ros whole_body_demo.launch`
  常用参数：`urdf_path:=$(find polytope_ros)/../../urdf/my_robot_fix_wheels.urdf`, `base_frame:=moca_base_footprint`, `ee_frame:=moca_franka_EE`, `output_csv_path:=$(find polytope_ros)/../../output/polytope_ros/whole_body_demo.csv`, `horizon_sec:=10.0`, `trajectory_frequency_hz:=0.05`, `desired_force_magnitude:=70.0`, `amplitude_x:=0.04`
  主要节点：`polytope_ros/whole_body_demo_node/name=whole_body_demo_node`
- `launch/whole_body_demo_live.launch`: 启动在线/实时版本全身规划 demo。
  运行示例：`roslaunch polytope_ros whole_body_demo_live.launch`
  常用参数：`urdf_path:=$(find polytope_ros)/../../urdf/my_robot_fix_wheels.urdf`, `base_frame:=moca_base_footprint`, `ee_frame:=moca_franka_EE`, `output_csv_path:=$(find polytope_ros)/../../output/polytope_ros/whole_body_demo.csv`, `output_png_path:=$(find polytope_ros)/../../output/polytope_ros/whole_body_demo_live.png`, `horizon_sec:=10.0`, `trajectory_frequency_hz:=0.05`, `desired_force_magnitude:=70.0`
  主要节点：`polytope_ros/whole_body_demo_node/name=whole_body_demo_node`, `polytope_ros/whole_body_demo_live_plotter.py/name=whole_body_demo_live_plotter`

### 常用检查命令
```bash
rosnode list
rostopic list
roslaunch polytope_ros <launch文件名>.launch
```

### 注意事项
- 运行前确认已经 `source devel/setup.bash`。
- 涉及 Gazebo、RealSense、实体机器人或外部 API 的 launch，需要确认对应硬件、驱动、权限和环境变量已经准备好。
- 本段为工作空间自动整理的使用说明；如果和实验室原始文档冲突，以当前 launch 参数和源码为准。

<!-- AUTO-GENERATED ROS USAGE END -->
