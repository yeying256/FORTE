#pragma once

#include "robot_dynamic_pin/Moca_dynamic_pin.h"

#include <memory>
#include <string>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/Wrench.h>
#include <moca_trajectory_generator/TargetPoseCommand.h>
#include <realtime_tools/realtime_publisher.h>
#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Empty.h>
#include <std_msgs/Float64.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>

#include <hrii_utils/RateTrigger.h>
#include "hrii_utils/MessageDisplay.h"

namespace WBPositionController
{

class ControllerManager
{
public:
    ControllerManager() = default;
    ~ControllerManager();

    bool init(int argc, char **argv);
    bool starting();
    bool update();

private:
    double computePidCommand(
        double error,
        double& integral_state,
        double& previous_error,
        double kp,
        double ki,
        double kd,
        double output_limit) const;
    void resetPidStates();

    Eigen::Affine3d computeWorldToBaseTransform(
        const Eigen::Vector3d& mobile_base_state) const;
    Eigen::Matrix<double, 6, 3> computeMobileBaseJacobianInWorld(
        const Eigen::Affine3d& w_T_mobile_base,
        const Eigen::Affine3d& w_T_ee) const;
    Eigen::Matrix<double, 6, 10> computeWholeBodyJacobianInWorld(
        const Eigen::Vector3d& mobile_base_state,
        const Eigen::Matrix<double, 7, 1>& arm_q,
        Eigen::Affine3d* w_T_ee = nullptr) const;
    double computeWholeBodyManipulability(
        const Eigen::Vector3d& mobile_base_state,
        const Eigen::Matrix<double, 7, 1>& arm_q) const;
    Eigen::Matrix<double, 14, 1> buildGravityAdjustedArmTorqueLimits(
        const Eigen::Matrix<double, 7, 1>& arm_gravity) const;
    double computeDirectionalForceCapability(
        const Eigen::Matrix<double, 6, 7>& arm_jacobian_in_world_frame,
        const Eigen::Matrix<double, 7, 1>& arm_gravity) const;

    void targetPoseCallback(const moca_trajectory_generator::TargetPoseCommandConstPtr& msg);
    void resetToHomeCallback(const std_msgs::EmptyConstPtr& msg);
    void sendCommand(
        const Eigen::Vector3d& base_command,
        const Eigen::Matrix<double, 7, 1>& arm_command);

    ros::NodeHandle nh_;

    polytope_wx::tcpip_client tcpip_client_;
    std::unique_ptr<polytope_wx::Moca_dynamic_pin> moca_dynamic_pin_;

    Eigen::Vector3d mobile_base_position_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d mobile_base_velocity_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d mobile_base_target_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d mobile_base_position_error_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d base_velocity_command_world_{Eigen::Vector3d::Zero()};

    Eigen::Matrix<double, 7, 1> arm_q_target_{Eigen::Matrix<double, 7, 1>::Zero()};
    Eigen::Matrix<double, 7, 1> arm_dq_desired_{Eigen::Matrix<double, 7, 1>::Zero()};
    Eigen::Matrix<double, 7, 1> arm_tau_command_{Eigen::Matrix<double, 7, 1>::Zero()};

    Eigen::Matrix<double, 7, 1> arm_position_pid_kp_{Eigen::Matrix<double, 7, 1>::Zero()};
    Eigen::Matrix<double, 7, 1> arm_position_pid_ki_{Eigen::Matrix<double, 7, 1>::Zero()};
    Eigen::Matrix<double, 7, 1> arm_position_pid_kd_{Eigen::Matrix<double, 7, 1>::Zero()};
    Eigen::Matrix<double, 7, 1> arm_position_pid_output_limits_{
        Eigen::Matrix<double, 7, 1>::Constant(1.0)};
    Eigen::Matrix<double, 7, 1> arm_velocity_pid_kp_{Eigen::Matrix<double, 7, 1>::Zero()};
    Eigen::Matrix<double, 7, 1> arm_velocity_pid_ki_{Eigen::Matrix<double, 7, 1>::Zero()};
    Eigen::Matrix<double, 7, 1> arm_velocity_pid_kd_{Eigen::Matrix<double, 7, 1>::Zero()};
    Eigen::Matrix<double, 7, 1> arm_velocity_pid_output_limits_{
        Eigen::Matrix<double, 7, 1>::Constant(-1.0)};
    Eigen::Matrix<double, 7, 1> arm_position_pid_integral_{
        Eigen::Matrix<double, 7, 1>::Zero()};
    Eigen::Matrix<double, 7, 1> arm_position_pid_previous_error_{
        Eigen::Matrix<double, 7, 1>::Zero()};
    Eigen::Matrix<double, 7, 1> arm_velocity_pid_integral_{
        Eigen::Matrix<double, 7, 1>::Zero()};
    Eigen::Matrix<double, 7, 1> arm_velocity_pid_previous_error_{
        Eigen::Matrix<double, 7, 1>::Zero()};

    Eigen::Vector3d base_position_pid_kp_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d base_position_pid_ki_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d base_position_pid_kd_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d base_position_pid_integral_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d base_position_pid_previous_error_{Eigen::Vector3d::Zero()};

    Eigen::Affine3d w_T_ee_{Eigen::Affine3d::Identity()};
    Eigen::Affine3d w_T_mobile_base_{Eigen::Affine3d::Identity()};

    Eigen::Vector3d moca_position_d_{Eigen::Vector3d::Zero()};
    Eigen::Matrix3d moca_rotation_d_{Eigen::Matrix3d::Identity()};
    Eigen::Matrix<double, 6, 1> moca_twist_d_{Eigen::Matrix<double, 6, 1>::Zero()};
    Eigen::Vector3d desired_force_d_{Eigen::Vector3d::Zero()};
    bool whole_body_target_active_{false};

    Eigen::Matrix<double, 7, 1> arm_torque_limits_{
        Eigen::Matrix<double, 7, 1>::Constant(87.0)};
    Eigen::Matrix<double, 7, 1> arm_joint_lower_limits_{
        Eigen::Matrix<double, 7, 1>::Zero()};
    Eigen::Matrix<double, 7, 1> arm_joint_upper_limits_{
        Eigen::Matrix<double, 7, 1>::Zero()};
    Eigen::Vector3d directional_force_axis_world_{Eigen::Vector3d::UnitZ()};

    int publish_rate_hz_{30};
    double controller_dt_{0.001};
    double pid_anti_windup_coeff_{0.1};
    bool enable_arm_gravity_compensation_{true};
    bool enable_arm_coriolis_compensation_{false};
    double base_max_linear_velocity_{0.8};
    double base_max_angular_velocity_{0.8};
    std::string tcp_server_ip_{"127.0.0.1"};
    int tcp_server_port_{5005};
    int tcp_retry_delay_ms_{2000};
    std::string urdf_path_{"package://polytope_ros/urdf/my_robot_fix_wheels.urdf"};
    std::string base_frame_{"moca_base_footprint"};
    std::string ee_frame_{"moca_franka_EE"};
    std::string world_frame_{"odom"};
    std::string target_pose_frame_{"moca_target_pose"};
    std::string planned_base_frame_{"moca_controller_planned_nullspace_base"};
    std::string planned_ee_frame_{"moca_controller_planned_nullspace_ee"};
    std::string reset_to_home_topic_{"/reset_to_home"};
    double finger_command_{0.0};
    double nullspace_base_arrow_scale_{0.2};
    double base_cmd_velocity_arrow_scale_{2.0};

    ros::Subscriber target_pose_sub_;
    ros::Subscriber reset_to_home_sub_;
    ros::Publisher virtual_torques_pub_;
    ros::Publisher manipulability_pub_;
    ros::Publisher directional_force_capacity_pub_;
    ros::Publisher desired_force_pub_;
    ros::Publisher nullspace_task_acceleration_norm_pub_;
    ros::Publisher nullspace_base_arrow_pub_;
    ros::Publisher base_cmd_velocity_arrow_pub_;
    ros::Publisher planned_ee_target_position_error_arrow_pub_;
    ros::Publisher planned_ee_target_orientation_error_arrow_pub_;

    realtime_tools::RealtimePublisher<sensor_msgs::JointState> moca_joint_state_publisher_;
    realtime_tools::RealtimePublisher<sensor_msgs::JointState> arm_task_torque_publisher_;
    realtime_tools::RealtimePublisher<sensor_msgs::JointState> arm_nullspace_torque_publisher_;
    realtime_tools::RealtimePublisher<geometry_msgs::PoseStamped> moca_position_publisher_;
    realtime_tools::RealtimePublisher<geometry_msgs::PoseStamped> moca_target_pose_publisher_;
    realtime_tools::RealtimePublisher<geometry_msgs::TwistStamped> nullspace_task_acceleration_publisher_;
    realtime_tools::RealtimePublisher<geometry_msgs::TwistStamped> moca_velocity_publisher_;
    realtime_tools::RealtimePublisher<geometry_msgs::TwistStamped> base_command_velocity_publisher_;
    realtime_tools::RealtimePublisher<sensor_msgs::JointState> moca_target_state_publisher_;

    tf::TransformBroadcaster tf_broadcaster_;
    HRII_Utils::RateTrigger rate_trigger_;
    HRII_Utils::MessageDisplay display_;
};

} // namespace WBPositionController
