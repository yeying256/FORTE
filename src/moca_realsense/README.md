# moca_realsense

<!-- AUTO-GENERATED ROS USAGE START -->

## ROS 使用说明

### 功能定位
Intel RealSense D435i ROS 封装包，直接用 librealsense2 C++ API 发布 color/depth/point cloud。

### 编译与环境
```bash
cd <your_catkin_workspace>
catkin_make --pkg moca_realsense
source devel/setup.bash
```

### Launch 文件

- `launch/realsense_camera.launch`: 启动 RealSense D435i C++ 相机节点，并可自动打开 RViz 与 topic 测试节点。
  运行示例：`roslaunch moca_realsense realsense_camera.launch`
  常用参数：`serial_no:=`, `fps:=30`, `color_width:=640`, `color_height:=480`, `depth_width:=640`, `depth_height:=480`, `align_depth_to_color:=true`, `publish_pointcloud:=true`, `pointcloud_stride:=8`, `publish_every_n_frames:=2`, `wait_timeout_ms:=3000`
  需要降低 ROS 图像发布频率时，优先调 `publish_every_n_frames`，不要直接降低传感器 `fps`。
  主要节点：`moca_realsense/moca_realsense_node/name=moca_realsense`, `rviz/rviz/name=moca_realsense_rviz`, `moca_realsense/realsense_topic_test.py/name=moca_realsense_topic_test`

### 常用检查命令
```bash
rosnode list
rostopic list
roslaunch moca_realsense <launch文件名>.launch
```

### 注意事项
- 运行前确认已经 `source devel/setup.bash`。
- 涉及 Gazebo、RealSense、实体机器人或外部 API 的 launch，需要确认对应硬件、驱动、权限和环境变量已经准备好。
- 本段为工作空间自动整理的使用说明；如果和实验室原始文档冲突，以当前 launch 参数和源码为准。

<!-- AUTO-GENERATED ROS USAGE END -->
