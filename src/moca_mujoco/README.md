# moca_mujoco

这个包是给 MOCA 移动操作机器人准备的 MuJoCo 环境骨架，目标是尽量复现当前 `python/isaac/` 那条链路里的核心功能：

- TCP/IP 状态回传与控制指令协议
- 未连接前/未收到首条指令前不推进仿真
- `RESET_TO_HOME`
- 移动底盘 `VELOCITY/WRENCH`
- 机械臂 `POSITION/VELOCITY/TORQUE`
- 手指位置命令
- `[x, y, yaw] + 7 arm joints + finger` 状态反馈

默认会尝试用现有 URDF 自动生成一个带 freejoint 的 MuJoCo 场景文件，保存到：

- `src/moca_mujoco/generated/moca_scene.xml`

当前还支持在场景里直接生成一个可配置的“带提手重物”：

- 配置位置：`src/moca_mujoco/config/moca_mujoco.yaml`
- 关键参数：`mujoco.payload`
- 默认对象：前方地面上的箱体 + U 形提手，可用于抓取和抬升实验

外观方面，MuJoCo 现在会优先保留源 URDF 里的 `visual` mesh：

- 配置开关：`mujoco.use_source_visual_meshes`
- 默认值：`true`
- 若某些 `visual` 只提供 `dae` 且找不到 MuJoCo 兼容的 `stl/obj/msh` 替代，则会自动跳过该 visual，避免整张场景导入失败

使用：

```bash
source devel/setup.bash
roslaunch moca_mujoco moca_mujoco.launch
```

如果只想把源 URDF 单独转换成 MuJoCo XML，而不启动整个仿真环境，也可以直接运行：

```bash
PYTHONPATH=src/moca_mujoco/src \
python3 \
src/moca_mujoco/scripts/convert_urdf_to_mjcf.py \
  --config src/moca_mujoco/config/moca_mujoco.json \
  --urdf "$(rospack find polytope_ros)/urdf/my_robot_fix_wheels.urdf" \
  --output-xml "$(rospack find moca_mujoco)/generated/moca_scene.xml"
```

这个脚本会输出：

- `*_resolved.urdf`
- `*_compiled.xml`
- 最终可直接用于 MuJoCo 的场景 `*.xml`

注意：

- 当前这台机器上还没有安装 MuJoCo Python 包，所以这里做过代码级实现和 catkin 打包验证，但没有做真实运行验证。
- 如果 MuJoCo 没装，启动脚本会直接报出明确错误提示。

<!-- AUTO-GENERATED ROS USAGE START -->

## ROS 使用说明

### 功能定位
MOCA MuJoCo 仿真包，负责 MJCF 场景、相机和 ROS topic 输出。

### 编译与环境
```bash
cd <your_catkin_workspace>
catkin_make --pkg moca_mujoco
source devel/setup.bash
```

### Launch 文件

- `launch/moca_mujoco.launch`: 启动 MOCA MuJoCo 仿真环境及 ROS 数据桥接。
  运行示例：`roslaunch moca_mujoco moca_mujoco.launch`
  常用参数：`config_file:=$(find moca_mujoco)/config/moca_mujoco.json`
  主要节点：`moca_mujoco/moca_mujoco_main.py/name=moca_mujoco`

### 常用检查命令
```bash
rosnode list
rostopic list
roslaunch moca_mujoco <launch文件名>.launch
```

### 注意事项
- 运行前确认已经 `source devel/setup.bash`。
- 涉及 Gazebo、RealSense、实体机器人或外部 API 的 launch，需要确认对应硬件、驱动、权限和环境变量已经准备好。
- 本段为工作空间自动整理的使用说明；如果和实验室原始文档冲突，以当前 launch 参数和源码为准。

<!-- AUTO-GENERATED ROS USAGE END -->
