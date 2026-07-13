#pragma once

#include "robot_dynamic_pin/Moca_dynamic_pin.h"
#include "polytope_ros/nullspace_optimizer.h"

#include <memory>
#include <string>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Inertia.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/Wrench.h>
#include <hrii_robot_msgs/FrankaToolState.h>
#include <hrii_mor_interface/MobileRobotInterface.h>
#include <hrii_ra_interface/robot_interface/RoboticArmInterface.h>
#include <moca_trajectory_generator/TargetPoseCommand.h>
#include <moca_trajectory_generator/ForceCapabilitySettings.h>
#include <nav_msgs/Odometry.h>
#include <realtime_tools/realtime_publisher.h>
#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Empty.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Float64MultiArray.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>

#include <hrii_utils/RateTrigger.h>
#include "hrii_utils/MessageDisplay.h"

#include <matlogger2/matlogger2.h>
#include <matlogger2/utils/mat_appender.h>

#define LINEAR 1
#define ANGULAR 2

namespace WBCartesianImpedanceController
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
    Eigen::Matrix<double, 6, 3> computeMobileBaseJacobianInWorld(
        const Eigen::Affine3d& w_T_mobile_base,
        const Eigen::Affine3d& w_T_ee) const;
    Eigen::Affine3d computeWorldToBaseTransform(
        const Eigen::Vector3d& mobile_base_state) const;
    Eigen::Matrix<double, 6, 1> computeCartesianPoseError(
        const Eigen::Vector3d& desired_position,
        const Eigen::Matrix3d& desired_rotation,
        const Eigen::Vector3d& current_position,
        const Eigen::Matrix3d& current_rotation) const;
    Eigen::Matrix<double, 6, 10> computeWholeBodyJacobianInWorld(
        const Eigen::Vector3d& mobile_base_state,
        const Eigen::Matrix<double, 7, 1>& arm_q,
        Eigen::Affine3d* w_T_ee = nullptr) const;
    Eigen::Affine3d computeWorldToFrankaFlangeTransform(
        const Eigen::Affine3d& w_T_controller_ee) const;
    Eigen::Matrix<double, 6, 7> shiftArmJacobianToWorldPoint(
        const Eigen::Matrix<double, 6, 7>& arm_jacobian_in_world_frame,
        const Eigen::Vector3d& controller_ee_to_point_world) const;
    Eigen::Matrix<double, 7, 1> computeFrankaToolLoadGravityTorques(
        const Eigen::Affine3d& w_T_controller_ee,
        const Eigen::Matrix<double, 6, 7>& arm_jacobian_in_world_frame) const;
    double computeWholeBodyManipulability(
        const Eigen::Vector3d& mobile_base_state,
        const Eigen::Matrix<double, 7, 1>& arm_q) const;
    Eigen::Matrix3d computeWholeBodyManipulabilityEllipsoid(
        const Eigen::Vector3d& mobile_base_state,
        const Eigen::Matrix<double, 7, 1>& arm_q) const;
    Eigen::Matrix<double, 10, 1> computeWholeBodyManipulabilityGradient(
        const Eigen::Vector3d& mobile_base_state,
        const Eigen::Matrix<double, 7, 1>& arm_q) const;
    Eigen::Matrix<double, 14, 1> buildGravityAdjustedArmTorqueLimits(
        const Eigen::Matrix<double, 7, 1>& arm_gravity) const;
    double computeDirectionalForceCapability(
        const Eigen::Matrix<double, 6, 7>& arm_jacobian_in_world_frame,
        const Eigen::Matrix<double, 7, 1>& arm_gravity) const;
    Eigen::MatrixXd computeArmForcePolytopeVertices(
        const Eigen::Matrix<double, 6, 7>& arm_jacobian_in_world_frame,
        const Eigen::Matrix<double, 7, 1>& arm_gravity) const;

    Eigen::Matrix<double, 7, 1> saturateTorqueRate(
        const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
        const Eigen::Matrix<double, 7, 1>& tau_J_d);

    void targetPoseCallback(const moca_trajectory_generator::TargetPoseCommandConstPtr& msg);
    void forceCapabilitySettingsCallback(
        const moca_trajectory_generator::ForceCapabilitySettingsConstPtr& msg);
    void resetToHomeCallback(const std_msgs::EmptyConstPtr& msg);
    void mobileBaseInertiaCallback(const geometry_msgs::InertiaConstPtr& msg);
    void frankaToolStateCallback(const hrii_robot_msgs::FrankaToolStateConstPtr& msg);
    void checkDeadzone(double& value, int wrench_type);
    Eigen::Vector3d applyBaseCommandDeadband(
        const Eigen::Vector3d& base_command) const;
    Eigen::Matrix<double, 7, 1> computeArmJointLimitTorques(
        const Eigen::Matrix<double, 7, 1>& arm_q) const;
    void updateMobileBaseAdmittance(const Eigen::Vector3d& virtual_wrench);
    bool useHRIIBackend() const;
    bool initHRIIInterfaces();
    bool initializeOnlineNullspaceOptimizer();
    bool readHRIIRobotState(
        nav_msgs::Odometry* mobile_base_odom,
        Eigen::Matrix<double, 7, 1>* arm_q,
        Eigen::Matrix<double, 7, 1>* arm_dq);
    bool updatePinocchioFromArmState(
        const Eigen::Matrix<double, 7, 1>& arm_q,
        const Eigen::Matrix<double, 7, 1>& arm_dq);
    void sendCommand(const Eigen::Vector3d& base_command,
                     const Eigen::Matrix<double, 7, 1>& arm_command);

    ros::NodeHandle nh_;

    polytope_wx::tcpip_client tcpip_client_;
    std::unique_ptr<polytope_wx::Moca_dynamic_pin> moca_dynamic_pin_;
    HRII::RAInterface::RoboticArmInterface::Ptr arm_interface_;
    HRII::MORInterface::MobileRobotInterface::Ptr mobile_base_interface_;

    Eigen::Matrix<double, 10, 1> moca_q_;
    Eigen::Matrix<double, 10, 1> moca_q_d_nullspace_;
    Eigen::Matrix<double, 10, 1> online_nullspace_gradient_objective_{
        Eigen::Matrix<double, 10, 1>::Zero()};
    Eigen::Matrix<double, 10, 1> moca_dq_;
    Eigen::Matrix<double, 6, 10> previous_whole_body_jacobian_{
        Eigen::Matrix<double, 6, 10>::Zero()};
    bool has_previous_whole_body_jacobian_{false};

    Eigen::Vector3d mobile_base_position_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d mobile_base_velocity_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d mobile_base_vir_torques_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d cmd_vel{Eigen::Vector3d::Zero()};

    Eigen::Affine3d w_T_ee_{Eigen::Affine3d::Identity()};
    Eigen::Affine3d w_T_mobile_base_{Eigen::Affine3d::Identity()};

    Eigen::Vector3d moca_position_d_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d moca_position_d_target_{Eigen::Vector3d::Zero()};
    Eigen::Matrix3d moca_rotation_d_{Eigen::Matrix3d::Identity()};
    Eigen::Matrix3d moca_rotation_d_target_{Eigen::Matrix3d::Identity()};
    Eigen::Matrix<double, 6, 1> moca_twist_d_{Eigen::Matrix<double, 6, 1>::Zero()};
    Eigen::Matrix<double, 6, 1> moca_twist_d_target_{Eigen::Matrix<double, 6, 1>::Zero()};
    Eigen::Matrix<double, 6, 1> moca_acceleration_d_{
        Eigen::Matrix<double, 6, 1>::Zero()};
    Eigen::Matrix<double, 6, 1> moca_acceleration_d_target_{
        Eigen::Matrix<double, 6, 1>::Zero()};
    Eigen::Vector3d desired_force_d_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d desired_force_d_target_{Eigen::Vector3d::Zero()};

    Eigen::Matrix<double, 6, 1> external_wrench_{Eigen::Matrix<double, 6, 1>::Zero()};
    Eigen::Matrix<double, 3, 3> mobile_base_inertia_{Eigen::Matrix<double, 3, 3>::Zero()};
    Eigen::Matrix<double, 3, 3> mobile_base_admittance_damping_{Eigen::Matrix<double, 3, 3>::Zero()};
    Eigen::Matrix<double, 6, 6> cartesian_stiffness_{Eigen::Matrix<double, 6, 6>::Zero()};
    Eigen::Matrix<double, 6, 6> cartesian_stiffness_target_{Eigen::Matrix<double, 6, 6>::Zero()};
    Eigen::Matrix<double, 6, 6> cartesian_damping_{Eigen::Matrix<double, 6, 6>::Zero()};
    Eigen::Matrix<double, 6, 6> cartesian_damping_target_{Eigen::Matrix<double, 6, 6>::Zero()};
    Eigen::Matrix<double, 10, 10> nullspace_joint_stiffness_{Eigen::Matrix<double, 10, 10>::Zero()};
    Eigen::Matrix<double, 10, 10> nullspace_joint_stiffness_target_{Eigen::Matrix<double, 10, 10>::Zero()};
    Eigen::Matrix<double, 10, 10> nullspace_joint_damping_{Eigen::Matrix<double, 10, 10>::Zero()};
    Eigen::Matrix<double, 10, 10> nullspace_joint_damping_target_{Eigen::Matrix<double, 10, 10>::Zero()};
    Eigen::Matrix<double, 7, 1> previous_arm_tau_command_{Eigen::Matrix<double, 7, 1>::Zero()};
    Eigen::Matrix<double, 7, 1> arm_torque_limits_{Eigen::Matrix<double, 7, 1>::Constant(87.0)};
    Eigen::Matrix<double, 7, 1> arm_joint_lower_limits_{Eigen::Matrix<double, 7, 1>::Zero()};
    Eigen::Matrix<double, 7, 1> arm_joint_upper_limits_{Eigen::Matrix<double, 7, 1>::Zero()};
    Eigen::Matrix<double, 7, 1> arm_joint_limit_margins_{Eigen::Matrix<double, 7, 1>::Zero()};
    Eigen::Vector3d directional_force_axis_world_{Eigen::Vector3d::UnitZ()};
    Eigen::Matrix<double, 7, 1> current_arm_gravity_{
        Eigen::Matrix<double, 7, 1>::Zero()};

    int publish_rate_hz_{30};
    double controller_dt_{0.001};
    bool enable_matlogger_{false};
    bool enable_nullspace_optimization_{true};
    bool enable_jdot_compensation_{false};
    bool arm_only_nullspace_test_{false};
    double base_max_linear_velocity_{0.5};
    double base_max_angular_velocity_{1.0};
    double base_cmd_linear_velocity_deadband_{0.0};
    double base_cmd_angular_velocity_deadband_{0.0};
    double base_nullspace_linear_stiffness_{60.0};
    double base_nullspace_yaw_stiffness_{25.0};
    double joint_limit_margin_ratio_{0.10};
    double manipulability_gain_{30.0};
    double manipulability_gradient_step_{1e-4};
    double manipulability_epsilon_{1e-8};
    double manipulability_regularization_{1e-6};
    double manipulability_objective_max_norm_{20.0};
    std::string command_backend_{"tcpip"};
    std::string interface_type_{"HARDWARE"};
    std::string robot_interface_config_file_;
    std::string arm_transmission_type_{"EFFORT"};
    bool hrii_interface_adds_gravity_compensation_{true};
    bool enable_franka_tool_load_control_compensation_{false};
    bool controller_ee_frame_is_franka_flange_{true};
    bool hold_until_first_target_pose_{true};
    bool command_output_enabled_{true};
    std::string tcp_server_ip_{"127.0.0.1"};
    int tcp_server_port_{5005};
    int tcp_retry_delay_ms_{2000};
    std::string urdf_path_{"package://polytope_ros/urdf/my_robot_fix_wheels.urdf"};
    std::string base_frame_{"moca_base_footprint"};
    std::string ee_frame_{"moca_franka_EE"};
    std::string world_frame_{"odom"};
    std::string debug_current_base_frame_{"moca_controller_current_base"};
    std::string debug_current_ee_frame_{"moca_controller_current_ee"};
    std::string target_pose_frame_{"moca_target_pose"};
    std::string planned_base_frame_{"moca_controller_planned_nullspace_base"};
    std::string planned_ee_frame_{"moca_controller_planned_nullspace_ee"};
    std::string reset_to_home_topic_{"/reset_to_home"};
    std::string force_capability_settings_topic_{"/force_capability_settings"};
    std::string franka_tool_state_topic_{"/moca_white/franka_state_bridge/franka_tool_state"};
    double min_linear_wrench_{2.0};
    double max_linear_wrench_{100.0};
    double min_angular_wrench_{0.1};
    double max_angular_wrench_{50.0};
    double gravity_acceleration_mps2_{9.81};

    double mobile_base_yaw_{0.0};
    double w_r_{1.0};
    double w_f_{1.0};
    double w_r_yaw_{1.0};
    int J_weight_{0};

    double filter_params_{0.005};
    double nullspace_stiffness_target_{0.0};
    double delta_tau_max_{1.0};
    bool impedance_filtering_{false};
    double initial_transl_cartesian_stiffness_{500.0};
    double initial_rot_cartesian_stiffness_{80.0};
    double initial_nullspace_stiffness_{3.0};
    double joint_limit_torque_gain_{30.0};
    double finger_command_{0.0};
    double task_acceleration_feedforward_gain_{0.0};
    double online_nullspace_gradient_weight_{1.0};
    double nullspace_base_arrow_scale_{0.02};
    double base_cmd_velocity_arrow_scale_{0.5};
    bool enable_online_nullspace_optimizer_{true};
    double online_nullspace_optimizer_rate_hz_{20.0};
    std::string polytope_urdf_path_;
    std::string polytope_ee_frame_;
    std::string left_finger_joint_name_{"moca_franka_franka_gripper_finger_joint1"};
    std::string right_finger_joint_name_{"moca_franka_franka_gripper_finger_joint2"};
    int optimizer_max_iterations_per_waypoint_{60};
    double optimizer_pose_gain_{0.9};
    double optimizer_nullspace_step_size_{0.01};
    double optimizer_max_joint_update_norm_{0.08};
    double optimizer_pose_tolerance_{5e-4};
    double optimizer_capability_weight_{1.0};
    double optimizer_manipulability_weight_{1.0};
    double optimizer_joint_limit_weight_{1e-3};
    double optimizer_smoothness_weight_{5e-2};
    double optimizer_nominal_weight_{5e-2};
    double optimizer_capability_alpha_{0.8};
    double optimizer_capability_near_weight_{50.0};
    double optimizer_capability_over_weight_{500.0};
    double optimizer_capability_over4_weight_{5000.0};
    bool optimizer_constrain_orientation_{true};
    bool optimizer_verbose_{false};
    polytope_wx::PolytopeNullspaceOptimizer online_nullspace_optimizer_;
    bool online_nullspace_optimizer_initialized_{false};
    int online_optimizer_model_joint_count_{0};
    Eigen::VectorXd last_online_optimized_model_configuration_;
    bool has_last_online_optimized_model_configuration_{false};
    ros::Time last_online_nullspace_optimization_time_;
    bool external_whole_body_nullspace_target_active_{false};
    bool has_received_target_pose_{false};
    hrii_robot_msgs::FrankaToolState latest_franka_tool_state_;
    bool has_franka_tool_state_{false};

    ros::Subscriber target_pose_sub_;
    ros::Subscriber force_capability_settings_sub_;
    ros::Subscriber reset_to_home_sub_;
    ros::Subscriber mobile_base_inertia_sub_;
    ros::Subscriber franka_tool_state_sub_;
    ros::Publisher virtual_torques_pub_;
    ros::Publisher manipulability_pub_;
    ros::Publisher manipulability_ellipsoid_pub_;
    ros::Publisher manipulability_gradient_norm_pub_;
    ros::Publisher directional_force_capacity_pub_;
    ros::Publisher force_polytope_vertices_pub_;
    ros::Publisher desired_force_pub_;
    ros::Publisher nullspace_task_acceleration_norm_pub_;
    ros::Publisher nullspace_base_arrow_pub_;
    ros::Publisher base_cmd_velocity_arrow_pub_;
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
    XBot::MatLogger2::Ptr logger_;
    XBot::MatAppender::Ptr appender_;
    HRII_Utils::RateTrigger rate_trigger_;
    HRII_Utils::MessageDisplay display_;
};

} // namespace WBCartesianImpedanceController
