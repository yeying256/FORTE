# robot_dynamic_pin

<!-- AUTO-GENERATED ROS USAGE START -->

## ROS 使用说明

### 功能定位
ROS 功能包，提供本工作空间中的相关节点、配置或资源。

### 编译与环境
```bash
cd <your_catkin_workspace>
catkin_make --pkg robot_dynamic_pin
source devel/setup.bash
```

### Launch 文件

- `launch/test.launch`: 启动该包的基础测试 launch。
  运行示例：`roslaunch robot_dynamic_pin test.launch`
  主要节点：`robot_dynamic_pin/test_pin_moca/name=test_pin_moca`

### 常用检查命令
```bash
rosnode list
rostopic list
roslaunch robot_dynamic_pin <launch文件名>.launch
```

### 注意事项
- 运行前确认已经 `source devel/setup.bash`。
- 涉及 Gazebo、RealSense、实体机器人或外部 API 的 launch，需要确认对应硬件、驱动、权限和环境变量已经准备好。
- 本段为工作空间自动整理的使用说明；如果和实验室原始文档冲突，以当前 launch 参数和源码为准。

<!-- AUTO-GENERATED ROS USAGE END -->
