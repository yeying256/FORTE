#pragma once

#include <cmath>
#include <memory>
#include <string>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <geometry_msgs/Wrench.h>
#include <hrii_mor_interface/MobileRobotInterface.h>
#include <hrii_ra_interface/robot_interface/RoboticArmInterface.h>
#include <moca_trajectory_generator/ForceCapabilitySettings.h>
#include <moca_trajectory_generator/TargetPoseCommand.h>
#include <nav_msgs/Odometry.h>
#include <robot_dynamic_pin/Moca_dynamic_pin.h>
#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Empty.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Float64MultiArray.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>

namespace moca_joint_impedance_controller
{

class JointImpedanceController
{
public:
    JointImpedanceController() = default;
    ~JointImpedanceController();

    bool init(int argc, char** argv);
    bool starting();
    bool update();

private:
    bool initHRIIInterfaces();
    bool readRobotState(
        nav_msgs::Odometry* mobile_base_odom,
        Eigen::Matrix<double, 7, 1>* arm_q,
        Eigen::Matrix<double, 7, 1>* arm_dq);
    void targetPoseCallback(
        const moca_trajectory_generator::TargetPoseCommandConstPtr& msg);
    void forceCapabilitySettingsCallback(
        const moca_trajectory_generator::ForceCapabilitySettingsConstPtr& msg);
    void resetToHomeCallback(const std_msgs::EmptyConstPtr& msg);
    void updateBaseAdmittance(const Eigen::Vector3d& virtual_wrench);
    Eigen::Vector3d applyBaseCommandDeadband(
        const Eigen::Vector3d& base_command) const;
    Eigen::Matrix<double, 7, 1> saturateTorqueRate(
        const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
        const Eigen::Matrix<double, 7, 1>& tau_J_d) const;
    void sendCommand(
        const Eigen::Vector3d& base_command,
        const Eigen::Matrix<double, 7, 1>& arm_command);
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
    Eigen::Matrix3d computeWholeBodyManipulabilityEllipsoid(
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
    Eigen::Affine3d poseMsgToAffine(
        const geometry_msgs::Pose& pose) const;
    bool computePlannedEndEffectorTransform(Eigen::Affine3d* w_T_ee) const;
    void publishState(
        const nav_msgs::Odometry& mobile_base_odom,
        const Eigen::Matrix<double, 7, 1>& arm_q,
        const Eigen::Matrix<double, 7, 1>& arm_dq,
        const Eigen::Vector3d& base_virtual_wrench,
        const Eigen::Matrix<double, 7, 1>& arm_tau_command,
        const Eigen::Matrix<double, 7, 1>& arm_impedance_tau,
        const Eigen::Matrix<double, 7, 1>& arm_desired_force_tau,
        const Eigen::Matrix<double, 7, 1>& arm_gravity,
        const Eigen::Matrix<double, 7, 1>& arm_coriolis,
        double manipulability_measure,
        double directional_force_capacity,
        const Eigen::MatrixXd& force_polytope_vertices);

    ros::NodeHandle nh_;

    HRII::RAInterface::RoboticArmInterface::Ptr arm_interface_;
    HRII::MORInterface::MobileRobotInterface::Ptr mobile_base_interface_;
    std::unique_ptr<polytope_wx::Moca_dynamic_pin> moca_dynamic_pin_;

    Eigen::Matrix<double, 10, 1> q_{Eigen::Matrix<double, 10, 1>::Zero()};
    Eigen::Matrix<double, 10, 1> dq_{Eigen::Matrix<double, 10, 1>::Zero()};
    Eigen::Matrix<double, 10, 1> q_target_{Eigen::Matrix<double, 10, 1>::Zero()};
    Eigen::Matrix<double, 7, 1> arm_joint_stiffness_{
        Eigen::Matrix<double, 7, 1>::Constant(15.0)};
    Eigen::Matrix<double, 7, 1> arm_joint_damping_{
        Eigen::Matrix<double, 7, 1>::Constant(2.0 * std::sqrt(15.0))};
    Eigen::Vector3d base_planar_stiffness_{10.0, 10.0, 5.0};
    Eigen::Vector3d base_planar_damping_{2.0 * std::sqrt(10.0),
                                         2.0 * std::sqrt(10.0),
                                         2.0 * std::sqrt(5.0)};
    Eigen::Matrix<double, 7, 1> previous_arm_tau_command_{
        Eigen::Matrix<double, 7, 1>::Zero()};
    Eigen::Matrix<double, 7, 1> current_arm_gravity_{
        Eigen::Matrix<double, 7, 1>::Zero()};
    Eigen::Matrix<double, 7, 1> arm_torque_limits_{
        Eigen::Matrix<double, 7, 1>::Constant(87.0)};
    Eigen::Vector3d cmd_vel_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d desired_force_target_{Eigen::Vector3d::Zero()};
    Eigen::Matrix3d mobile_base_inertia_{Eigen::Matrix3d::Identity()};
    Eigen::Matrix3d mobile_base_admittance_damping_{Eigen::Matrix3d::Identity()};
    Eigen::Vector3d directional_force_axis_world_{Eigen::Vector3d::UnitZ()};
    Eigen::Affine3d w_T_mobile_base_{Eigen::Affine3d::Identity()};
    Eigen::Affine3d w_T_ee_{Eigen::Affine3d::Identity()};
    Eigen::Affine3d target_w_T_ee_display_{Eigen::Affine3d::Identity()};
    Eigen::Vector3d desired_force_display_{Eigen::Vector3d::Zero()};

    int publish_rate_hz_{30};
    double controller_dt_{0.001};
    double base_max_linear_velocity_{0.35};
    double base_max_angular_velocity_{0.6};
    double base_cmd_linear_velocity_deadband_{0.0};
    double base_cmd_angular_velocity_deadband_{0.0};
    double nullspace_base_arrow_scale_{0.02};
    double base_cmd_velocity_arrow_scale_{0.5};
    double manipulability_epsilon_{1e-8};
    double manipulability_regularization_{1e-6};
    double delta_tau_max_{1.0};
    double desired_force_torque_scale_{1.0};
    double desired_force_torque_limit_{30.0};
    bool command_output_enabled_{true};
    bool enable_desired_force_torque_feedforward_{true};
    bool hold_until_first_target_pose_{true};
    bool track_base_planar_target_{true};
    bool base_cmd_vel_in_base_frame_{true};
    bool hrii_interface_adds_gravity_compensation_{true};
    bool has_received_nullspace_target_{false};
    bool has_display_target_pose_{false};

    std::string interface_type_{"SIMULATION"};
    std::string robot_interface_config_file_;
    std::string arm_transmission_type_{"effort"};
    std::string urdf_path_{"package://polytope_ros/urdf/my_robot_fix_wheels.urdf"};
    std::string base_frame_{"moca_base_footprint"};
    std::string ee_frame_{"moca_franka_flange"};
    std::string world_frame_{"odom"};
    std::string debug_current_base_frame_{"moca_controller_current_base"};
    std::string debug_current_ee_frame_{"moca_controller_current_ee"};
    std::string target_pose_frame_{"moca_target_pose"};
    std::string planned_base_frame_{"moca_controller_planned_nullspace_base"};
    std::string planned_ee_frame_{"moca_controller_planned_nullspace_ee"};
    std::string reset_to_home_topic_{"/reset_to_home"};
    std::string force_capability_settings_topic_{"/force_capability_settings"};

    ros::Time last_publish_time_;

    ros::Subscriber target_pose_sub_;
    ros::Subscriber force_capability_settings_sub_;
    ros::Subscriber reset_to_home_sub_;
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
    ros::Publisher moca_state_pub_;
    ros::Publisher moca_target_state_pub_;
    ros::Publisher arm_gravity_torque_pub_;
    ros::Publisher arm_coriolis_torque_pub_;
    ros::Publisher arm_command_torque_pub_;
    ros::Publisher arm_task_torque_pub_;
    ros::Publisher arm_nullspace_torque_pub_;
    ros::Publisher moca_pose_pub_;
    ros::Publisher moca_pose_target_pub_;
    ros::Publisher nullspace_task_acceleration_pub_;
    ros::Publisher moca_velocity_pub_;
    ros::Publisher base_command_velocity_pub_;
    tf::TransformBroadcaster tf_broadcaster_;
};

} // namespace moca_joint_impedance_controller
