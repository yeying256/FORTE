# 轨迹规划器发送给控制器的目标命令
# header + pose/twist/acceleration 描述末端期望轨迹状态
std_msgs/Header header
geometry_msgs/Pose pose
geometry_msgs/Twist twist
geometry_msgs/Accel acceleration
geometry_msgs/Vector3 desired_force
float64 finger_command

# 是否启用规划器生成的 whole-body 零空间位置目标
bool use_nullspace_joint_target

# 规划器给出的底盘平面目标，表达在世界坐标系 [base_x, base_y, base_yaw]
float64[3] base_planar_positions

# 机械臂 7 个关节的期望位置，顺序与控制器中的机械臂关节顺序一致
float64[7] arm_joint_positions
