# tcpip_polytope

<!-- AUTO-GENERATED ROS USAGE START -->

## ROS 使用说明

### 功能定位
ROS 功能包，提供本工作空间中的相关节点、配置或资源。

### 编译与环境
```bash
cd <your_catkin_workspace>
catkin_make --pkg tcpip_polytope
source devel/setup.bash
```

### Launch 文件

- `launch/tcpip_test.launch`: 启动 TCP/IP polytope 通信测试。
  运行示例：`roslaunch tcpip_polytope tcpip_test.launch`
  主要节点：`tcpip_polytope/tcpip_test/name=tcpip_test_node`

### 常用检查命令
```bash
rosnode list
rostopic list
roslaunch tcpip_polytope <launch文件名>.launch
```

### 注意事项
- 运行前确认已经 `source devel/setup.bash`。
- 涉及 Gazebo、RealSense、实体机器人或外部 API 的 launch，需要确认对应硬件、驱动、权限和环境变量已经准备好。
- 本段为工作空间自动整理的使用说明；如果和实验室原始文档冲突，以当前 launch 参数和源码为准。

<!-- AUTO-GENERATED ROS USAGE END -->
