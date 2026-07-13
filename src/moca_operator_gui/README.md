# moca_operator_gui

MOCA 的桌面上位机界面。当前第一版实现了：

- 启动仿真控制程序
- 连接实体机器人
- 选择笛卡尔阻抗控制器或关节阻抗控制器
- 启动 VLM
- 启动规划器
- Reset 到原位置
- 关闭全部由上位机启动的进程
- 选择规划轨迹
- 正常提起
- 两次提起对比

## 启动

```bash
roslaunch moca_operator_gui operator_panel.launch
```

## 当前按钮行为

- `启动仿真控制程序`
  - 选择笛卡尔阻抗控制器时，调用 `roslaunch Moca_controller moca_gazebo_moca_controller.launch`
  - 选择关节阻抗控制器时，调用 `roslaunch moca_joint_impedance_controller moca_joint_impedance_controller.launch`
  - 默认关闭内嵌规划器和内嵌 VLM，由上位机单独控制
- `连接实体机器人`
  - 选择笛卡尔阻抗控制器时，调用 `roslaunch Moca_controller moca_hardware_moca_controller.launch`
  - 选择关节阻抗控制器时，调用 `roslaunch moca_joint_impedance_controller moca_joint_impedance_controller.launch interface_type:=HARDWARE`
  - 默认关闭内嵌规划器和内嵌 VLM，由上位机单独控制
- `启动VLM`
  - 调用 `roslaunch moca_vlm mass_estimator.launch`
- `启动规划器`
  - 调用 `roslaunch moca_trajectory_generator target_pose_generator.launch autostart_planning:=false`
  - 然后通过 `/target_pose_generator/start_planning` service 发送本次规划请求
- `Reset到原位置`
  - 向 `/reset_to_home` 发布 `std_msgs/Empty`
- `关闭全部`
  - 向所有由上位机拉起的 `roslaunch` 进程组发送退出信号

## 启动仿真时的注意事项

如果系统里已经有旧的 `gzserver` 或 `gzclient` 进程在运行，上位机会拒绝再启动第二套 Gazebo，
避免出现 `gzserver exit code 255`、模型堆叠或连到旧仿真实例的情况。

## 配置

界面使用的 launch 指令和规划 service 名称在：

`config/operator_panel.yaml`

后续如果要扩展更多页面、更多轨迹按钮或更多启动项，优先改这个配置文件和 `scripts/operator_panel_node.py`。
