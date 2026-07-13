# moca_trajectory_generator

<!-- AUTO-GENERATED ROS USAGE START -->

## ROS 使用说明

### 功能定位
MOCA 末端目标、关节目标和全身规划轨迹生成包；会向控制器发布目标位姿/关节状态，并可请求 VLM 质量估计。

### 编译与环境
```bash
cd <your_catkin_workspace>
catkin_make --pkg moca_trajectory_generator
source devel/setup.bash
```

### Launch 文件

- `launch/target_pose_generator.launch`: 单独启动目标/轨迹生成节点。
  运行示例：`roslaunch moca_trajectory_generator target_pose_generator.launch`
  常用参数：`config_file:=$(find moca_trajectory_generator)/config/target_pose_generator_params.yaml`, `launch_vlm_mass_estimator:=false`, `vlm_config_file:=$(find moca_vlm)/config/mass_estimator_params.yaml`
  主要节点：`moca_vlm/mass_estimator_node.py/name=moca_vlm_mass_estimator`, `moca_trajectory_generator/target_pose_generator_node/name=target_pose_generator`

### 常用检查命令
```bash
rosnode list
rostopic list
roslaunch moca_trajectory_generator <launch文件名>.launch
```

### 注意事项
- 运行前确认已经 `source devel/setup.bash`。
- 涉及 Gazebo、RealSense、实体机器人或外部 API 的 launch，需要确认对应硬件、驱动、权限和环境变量已经准备好。
- 本段为工作空间自动整理的使用说明；如果和实验室原始文档冲突，以当前 launch 参数和源码为准。

<!-- AUTO-GENERATED ROS USAGE END -->
