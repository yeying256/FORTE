#include "Moca_controller/Moca_position_controller.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <vector>

#include "eigen_conversions/eigen_msg.h"
#include <ros/package.h>

namespace
{

constexpr int kRobotArmNum = 7;
constexpr int kWholeBodyDim = 10;
constexpr const char* kForceCapabilitySettingsParamNs =
    "/force_capability_settings/arm_torque_limits";

const std::array<std::string, kWholeBodyDim> kWholeBodyJointNames = {
    "base_x",
    "base_y",
    "base_yaw",
    "moca_franka_joint1",
    "moca_franka_joint2",
    "moca_franka_joint3",
    "moca_franka_joint4",
    "moca_franka_joint5",
    "moca_franka_joint6",
    "moca_franka_joint7"
};

const std::array<std::string, kRobotArmNum> kArmJointNames = {
    "moca_franka_joint1",
    "moca_franka_joint2",
    "moca_franka_joint3",
    "moca_franka_joint4",
    "moca_franka_joint5",
    "moca_franka_joint6",
    "moca_franka_joint7"
};

std::string expandUserPath(const std::string& path)
{
    if (path.empty() || path[0] != '~')
    {
        return path;
    }

    const char* home = std::getenv("HOME");
    if (home == nullptr)
    {
        return path;
    }

    if (path.size() == 1)
    {
        return std::string(home);
    }
    if (path[1] == '/')
    {
        return std::string(home) + path.substr(1);
    }
    return path;
}

std::string resolveRosPath(const std::string& path)
{
    if (path.empty())
    {
        return path;
    }

    const std::string package_prefix = "package://";
    if (path.compare(0, package_prefix.size(), package_prefix) == 0)
    {
        const std::string resource = path.substr(package_prefix.size());
        const std::size_t slash = resource.find('/');
        const std::string package_name =
            slash == std::string::npos ? resource : resource.substr(0, slash);
        const std::string relative_path =
            slash == std::string::npos ? std::string() : resource.substr(slash + 1);
        const std::string package_path = ros::package::getPath(package_name);
        if (!package_path.empty())
        {
            return relative_path.empty() ? package_path : package_path + "/" + relative_path;
        }
        return path;
    }

    const std::string find_prefix = "$(find ";
    if (path.compare(0, find_prefix.size(), find_prefix) == 0)
    {
        const std::size_t close = path.find(')', find_prefix.size());
        if (close != std::string::npos)
        {
            const std::string package_name =
                path.substr(find_prefix.size(), close - find_prefix.size());
            std::string relative_path = path.substr(close + 1);
            if (!relative_path.empty() && relative_path[0] == '/')
            {
                relative_path.erase(0, 1);
            }
            const std::string package_path = ros::package::getPath(package_name);
            if (!package_path.empty())
            {
                return relative_path.empty() ? package_path : package_path + "/" + relative_path;
            }
        }
    }

    return expandUserPath(path);
}

double wrapAngle(const double angle)
{
    return std::atan2(std::sin(angle), std::cos(angle));
}

} // namespace

namespace WBPositionController
{

bool ControllerManager::init(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    nh_ = ros::NodeHandle("~");

    display_.setName(ros::this_node::getName());
    display_.info("PositionControllerManager: initialize TCP/IP controller.");

    nh_.param("publish_rate_hz", publish_rate_hz_, publish_rate_hz_);
    nh_.param("controller_dt", controller_dt_, controller_dt_);
    nh_.param("pid_anti_windup_coeff", pid_anti_windup_coeff_, pid_anti_windup_coeff_);
    nh_.param(
        "enable_arm_gravity_compensation",
        enable_arm_gravity_compensation_,
        enable_arm_gravity_compensation_);
    nh_.param(
        "enable_arm_coriolis_compensation",
        enable_arm_coriolis_compensation_,
        enable_arm_coriolis_compensation_);
    nh_.param("base_max_linear_velocity", base_max_linear_velocity_, base_max_linear_velocity_);
    nh_.param("base_max_angular_velocity", base_max_angular_velocity_, base_max_angular_velocity_);
    nh_.param("tcp_server_ip", tcp_server_ip_, tcp_server_ip_);
    nh_.param("tcp_server_port", tcp_server_port_, tcp_server_port_);
    nh_.param("tcp_retry_delay_ms", tcp_retry_delay_ms_, tcp_retry_delay_ms_);
    nh_.param("urdf_path", urdf_path_, urdf_path_);
    urdf_path_ = resolveRosPath(urdf_path_);
    nh_.param("base_frame", base_frame_, base_frame_);
    nh_.param("ee_frame", ee_frame_, ee_frame_);
    nh_.param("world_frame", world_frame_, world_frame_);
    nh_.param("target_pose_frame", target_pose_frame_, target_pose_frame_);
    nh_.param("planned_base_frame", planned_base_frame_, planned_base_frame_);
    nh_.param("planned_ee_frame", planned_ee_frame_, planned_ee_frame_);
    nh_.param("reset_to_home_topic", reset_to_home_topic_, reset_to_home_topic_);
    nh_.param("finger_command", finger_command_, finger_command_);
    nh_.param(
        "nullspace_base_arrow_scale",
        nullspace_base_arrow_scale_,
        nullspace_base_arrow_scale_);
    nh_.param(
        "base_cmd_velocity_arrow_scale",
        base_cmd_velocity_arrow_scale_,
        base_cmd_velocity_arrow_scale_);

    auto loadVector3Param = [this](
                                const std::string& param_name,
                                const std::array<double, 3>& defaults) {
        std::vector<double> values;
        if (!nh_.getParam(param_name, values) || values.size() != 3)
        {
            values.assign(defaults.begin(), defaults.end());
        }

        return Eigen::Vector3d(values[0], values[1], values[2]);
    };

    auto loadVector7Param = [this](
                                const std::string& param_name,
                                const std::array<double, 7>& defaults) {
        std::vector<double> values;
        if (!nh_.getParam(param_name, values) || values.size() != 7)
        {
            values.assign(defaults.begin(), defaults.end());
        }

        Eigen::Matrix<double, 7, 1> out = Eigen::Matrix<double, 7, 1>::Zero();
        for (int i = 0; i < kRobotArmNum; ++i)
        {
            out(i) = values[static_cast<std::size_t>(i)];
        }
        return out;
    };

    base_position_pid_kp_ =
        loadVector3Param("base_position_pid_kp", {1.5, 1.5, 2.0});
    base_position_pid_ki_ =
        loadVector3Param("base_position_pid_ki", {0.0, 0.0, 0.0});
    base_position_pid_kd_ =
        loadVector3Param("base_position_pid_kd", {0.10, 0.10, 0.20});

    arm_position_pid_kp_ =
        loadVector7Param("arm_position_pid_kp", {4.0, 4.0, 4.0, 4.0, 3.5, 3.0, 2.5});
    arm_position_pid_ki_ =
        loadVector7Param("arm_position_pid_ki", {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    arm_position_pid_kd_ =
        loadVector7Param("arm_position_pid_kd", {0.05, 0.05, 0.05, 0.05, 0.04, 0.03, 0.03});
    arm_position_pid_output_limits_ =
        loadVector7Param("arm_position_pid_output_limits", {1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8});

    arm_velocity_pid_kp_ =
        loadVector7Param("arm_velocity_pid_kp", {25.0, 25.0, 25.0, 25.0, 18.0, 14.0, 12.0});
    arm_velocity_pid_ki_ =
        loadVector7Param("arm_velocity_pid_ki", {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    arm_velocity_pid_kd_ =
        loadVector7Param("arm_velocity_pid_kd", {0.2, 0.2, 0.2, 0.2, 0.1, 0.08, 0.08});
    arm_velocity_pid_output_limits_ =
        loadVector7Param("arm_velocity_pid_output_limits", {-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0});

    directional_force_axis_world_ =
        loadVector3Param("directional_force_axis_world", {0.0, 0.0, 1.0});
    if (directional_force_axis_world_.norm() < 1e-9)
    {
        directional_force_axis_world_ = Eigen::Vector3d::UnitZ();
    }
    else
    {
        directional_force_axis_world_.normalize();
    }

    nh_.setParam("controller_started", false);

    target_pose_sub_ = nh_.subscribe(
        "/target_pose", 1, &ControllerManager::targetPoseCallback, this);
    reset_to_home_sub_ = nh_.subscribe(
        reset_to_home_topic_, 1, &ControllerManager::resetToHomeCallback, this);

    virtual_torques_pub_ = nh_.advertise<geometry_msgs::Wrench>("vir_torque", 1);
    manipulability_pub_ = nh_.advertise<std_msgs::Float64>("manipulability", 1);
    directional_force_capacity_pub_ =
        nh_.advertise<std_msgs::Float64>("directional_force_capacity", 1);
    desired_force_pub_ =
        nh_.advertise<geometry_msgs::Vector3Stamped>("desired_force", 1);
    nullspace_task_acceleration_norm_pub_ =
        nh_.advertise<std_msgs::Float64>("nullspace_task_acceleration_norm", 1);
    nullspace_base_arrow_pub_ =
        nh_.advertise<visualization_msgs::Marker>("nullspace_base_arrow", 1);
    base_cmd_velocity_arrow_pub_ =
        nh_.advertise<visualization_msgs::Marker>("base_cmd_velocity_arrow", 1);
    planned_ee_target_position_error_arrow_pub_ =
        nh_.advertise<visualization_msgs::Marker>(
            "planned_ee_target_position_error_arrow", 1);
    planned_ee_target_orientation_error_arrow_pub_ =
        nh_.advertise<visualization_msgs::Marker>(
            "planned_ee_target_orientation_error_arrow", 1);

    moca_joint_state_publisher_.init(nh_, "moca_state", 1);
    moca_joint_state_publisher_.msg_.name.assign(
        kWholeBodyJointNames.begin(), kWholeBodyJointNames.end());
    moca_joint_state_publisher_.msg_.position.resize(kWholeBodyDim, 0.0);
    moca_joint_state_publisher_.msg_.velocity.resize(kWholeBodyDim, 0.0);
    moca_joint_state_publisher_.msg_.effort.resize(kWholeBodyDim, 0.0);

    arm_task_torque_publisher_.init(nh_, "arm_task_torque", 1);
    arm_task_torque_publisher_.msg_.name.assign(
        kArmJointNames.begin(), kArmJointNames.end());
    arm_task_torque_publisher_.msg_.effort.resize(kRobotArmNum, 0.0);

    arm_nullspace_torque_publisher_.init(nh_, "arm_nullspace_torque", 1);
    arm_nullspace_torque_publisher_.msg_.name.assign(
        kArmJointNames.begin(), kArmJointNames.end());
    arm_nullspace_torque_publisher_.msg_.effort.resize(kRobotArmNum, 0.0);

    moca_position_publisher_.init(nh_, "moca_pose", 1);
    moca_target_pose_publisher_.init(nh_, "moca_pose_target", 1);
    nullspace_task_acceleration_publisher_.init(nh_, "nullspace_task_acceleration", 1);
    moca_velocity_publisher_.init(nh_, "moca_velocity", 1);
    base_command_velocity_publisher_.init(nh_, "base_cmd_velocity", 1);
    moca_target_state_publisher_.init(nh_, "moca_state_target", 1);
    moca_target_state_publisher_.msg_.name.assign(
        kWholeBodyJointNames.begin(), kWholeBodyJointNames.end());
    moca_target_state_publisher_.msg_.position.resize(kWholeBodyDim, 0.0);
    moca_target_state_publisher_.msg_.velocity.resize(kWholeBodyDim, 0.0);
    moca_target_state_publisher_.msg_.effort.resize(kWholeBodyDim, 0.0);

    rate_trigger_ = HRII_Utils::RateTrigger(publish_rate_hz_);

    if (!tcpip_client_.connect_with_retry(
            tcp_server_ip_, tcp_server_port_, -1, tcp_retry_delay_ms_))
    {
        display_.error(
            "PositionControllerManager: TCP/IP backend initialization aborted before a connection was established.");
        return false;
    }
    tcpip_client_.start_receive_thread();
    tcpip_client_.send_thread_start();

    moca_dynamic_pin_ = std::make_unique<polytope_wx::Moca_dynamic_pin>();
    if (!moca_dynamic_pin_->initFromUrdf(urdf_path_))
    {
        display_.error("Failed to initialize Pinocchio model.");
        return false;
    }

    const auto& pin_model = moca_dynamic_pin_->model();
    if (pin_model.lowerPositionLimit.size() >= kRobotArmNum &&
        pin_model.upperPositionLimit.size() >= kRobotArmNum)
    {
        arm_joint_lower_limits_ = pin_model.lowerPositionLimit.head<kRobotArmNum>();
        arm_joint_upper_limits_ = pin_model.upperPositionLimit.head<kRobotArmNum>();
    }
    if (pin_model.effortLimit.size() >= kRobotArmNum)
    {
        arm_torque_limits_ = pin_model.effortLimit.head<kRobotArmNum>();
        for (int i = 0; i < kRobotArmNum; ++i)
        {
            if (!std::isfinite(arm_torque_limits_(i)) || arm_torque_limits_(i) <= 0.0)
            {
                arm_torque_limits_(i) = 12.0;
            }
            else
            {
                arm_torque_limits_(i) = std::abs(arm_torque_limits_(i));
            }

        }
    }

    std::vector<double> configured_arm_torque_limits;
    if (ros::param::get(kForceCapabilitySettingsParamNs, configured_arm_torque_limits) &&
        configured_arm_torque_limits.size() == kRobotArmNum)
    {
        for (int i = 0; i < kRobotArmNum; ++i)
        {
            const double limit = configured_arm_torque_limits[static_cast<std::size_t>(i)];
            if (std::isfinite(limit) && limit > 0.0)
            {
                arm_torque_limits_(i) = std::abs(limit);
            }
        }
    }
    for (int i = 0; i < kRobotArmNum; ++i)
    {
        if (arm_velocity_pid_output_limits_(i) <= 0.0)
        {
            arm_velocity_pid_output_limits_(i) = arm_torque_limits_(i);
        }
    }

    resetPidStates();
    display_.info("PositionControllerManager: TCP/IP controller initialized.");
    return true;
}

bool ControllerManager::starting()
{
    const polytope_wx::RobotMessage robot_state_now = tcpip_client_.get_robot_state();
    if (!moca_dynamic_pin_->updateFromRobotMessage(robot_state_now))
    {
        display_.error("Failed to update Pinocchio state from TCP/IP.");
        return false;
    }

    const Eigen::Map<const Eigen::Matrix<double, 7, 1>> arm_q(robot_state_now.arm_positions);
    nav_msgs::Odometry mobile_base_odom = tcpip_client_.toOdometryMsg(robot_state_now);
    tf::poseMsgToEigen(mobile_base_odom.pose.pose, w_T_mobile_base_);

    const Eigen::Affine3d base_T_ee =
        moca_dynamic_pin_->getRelativeTransform(base_frame_, ee_frame_);
    w_T_ee_ = w_T_mobile_base_ * base_T_ee;

    const Eigen::Quaterniond mobile_base_orientation(
        mobile_base_odom.pose.pose.orientation.w,
        mobile_base_odom.pose.pose.orientation.x,
        mobile_base_odom.pose.pose.orientation.y,
        mobile_base_odom.pose.pose.orientation.z);
    const double mobile_base_yaw = std::atan2(
        2.0 * (mobile_base_orientation.w() * mobile_base_orientation.z() +
               mobile_base_orientation.x() * mobile_base_orientation.y()),
        1.0 - 2.0 * (mobile_base_orientation.y() * mobile_base_orientation.y() +
                     mobile_base_orientation.z() * mobile_base_orientation.z()));

    mobile_base_position_ << mobile_base_odom.pose.pose.position.x,
                             mobile_base_odom.pose.pose.position.y,
                             mobile_base_yaw;
    mobile_base_velocity_.setZero();
    mobile_base_target_ = mobile_base_position_;
    mobile_base_position_error_.setZero();
    base_velocity_command_world_.setZero();

    arm_q_target_ = arm_q;
    arm_dq_desired_.setZero();
    arm_tau_command_.setZero();

    moca_position_d_ = w_T_ee_.translation();
    moca_rotation_d_ = w_T_ee_.linear();
    moca_twist_d_.setZero();
    desired_force_d_.setZero();
    whole_body_target_active_ = false;

    resetPidStates();
    sendCommand(mobile_base_target_, arm_q_target_);

    nh_.setParam("controller_started", true);
    display_.info("PositionControllerManager: controller started.");
    return true;
}

bool ControllerManager::update()
{
    const polytope_wx::RobotMessage robot_state_now = tcpip_client_.get_robot_state();
    if (!moca_dynamic_pin_->updateFromRobotMessage(robot_state_now))
    {
        display_.error("Failed to update Pinocchio state during control loop.");
        return false;
    }

    const Eigen::Map<const Eigen::Matrix<double, 7, 1>> arm_q(robot_state_now.arm_positions);
    const Eigen::Map<const Eigen::Matrix<double, 7, 1>> arm_dq(robot_state_now.arm_velocities);

    nav_msgs::Odometry mobile_base_odom = tcpip_client_.toOdometryMsg(robot_state_now);
    tf::poseMsgToEigen(mobile_base_odom.pose.pose, w_T_mobile_base_);

    const Eigen::Quaterniond mobile_base_orientation(
        mobile_base_odom.pose.pose.orientation.w,
        mobile_base_odom.pose.pose.orientation.x,
        mobile_base_odom.pose.pose.orientation.y,
        mobile_base_odom.pose.pose.orientation.z);
    const double mobile_base_yaw = std::atan2(
        2.0 * (mobile_base_orientation.w() * mobile_base_orientation.z() +
               mobile_base_orientation.x() * mobile_base_orientation.y()),
        1.0 - 2.0 * (mobile_base_orientation.y() * mobile_base_orientation.y() +
                     mobile_base_orientation.z() * mobile_base_orientation.z()));

    mobile_base_position_ << mobile_base_odom.pose.pose.position.x,
                             mobile_base_odom.pose.pose.position.y,
                             mobile_base_yaw;
    mobile_base_velocity_ << mobile_base_odom.twist.twist.linear.x,
                             mobile_base_odom.twist.twist.linear.y,
                             mobile_base_odom.twist.twist.angular.z;

    const Eigen::Affine3d base_T_ee =
        moca_dynamic_pin_->getRelativeTransform(base_frame_, ee_frame_);
    w_T_ee_ = w_T_mobile_base_ * base_T_ee;

    const Eigen::Matrix<double, 6, 7> arm_jacobian_in_base_frame =
        moca_dynamic_pin_->getArmJacobian(ee_frame_);
    Eigen::Matrix<double, 6, 7> arm_jacobian_in_world_frame =
        Eigen::Matrix<double, 6, 7>::Zero();
    arm_jacobian_in_world_frame.topRows<3>() =
        w_T_mobile_base_.linear() * arm_jacobian_in_base_frame.topRows<3>();
    arm_jacobian_in_world_frame.bottomRows<3>() =
        w_T_mobile_base_.linear() * arm_jacobian_in_base_frame.bottomRows<3>();

    const Eigen::Matrix<double, 6, 10> whole_body_jacobian_in_world_frame =
        computeWholeBodyJacobianInWorld(mobile_base_position_, arm_q);
    Eigen::Matrix<double, 10, 1> whole_body_dq = Eigen::Matrix<double, 10, 1>::Zero();
    whole_body_dq.head<3>() = mobile_base_velocity_;
    whole_body_dq.tail<7>() = arm_dq;
    const Eigen::Matrix<double, 6, 1> moca_twist =
        whole_body_jacobian_in_world_frame * whole_body_dq;

    if (!whole_body_target_active_)
    {
        mobile_base_target_ = mobile_base_position_;
        arm_q_target_ = arm_q;
    }

    mobile_base_position_error_(0) = mobile_base_target_(0) - mobile_base_position_(0);
    mobile_base_position_error_(1) = mobile_base_target_(1) - mobile_base_position_(1);
    mobile_base_position_error_(2) = wrapAngle(mobile_base_target_(2) - mobile_base_position_(2));

    // Legacy dual-loop PID path is intentionally kept disabled here.
    // The position controller now forwards whole-body position targets directly
    // through the TCP/IP interface instead of generating torque/velocity commands.
    const double base_x_limit = base_max_linear_velocity_;
    const double base_y_limit = base_max_linear_velocity_;
    const double base_yaw_limit = base_max_angular_velocity_;
    base_velocity_command_world_(0) =
        std::max(
            -base_x_limit,
            std::min(
                base_x_limit,
                mobile_base_position_error_(0) / std::max(controller_dt_, 1e-6)));
    base_velocity_command_world_(1) =
        std::max(
            -base_y_limit,
            std::min(
                base_y_limit,
                mobile_base_position_error_(1) / std::max(controller_dt_, 1e-6)));
    base_velocity_command_world_(2) =
        std::max(
            -base_yaw_limit,
            std::min(
                base_yaw_limit,
                mobile_base_position_error_(2) / std::max(controller_dt_, 1e-6)));

    for (int i = 0; i < kRobotArmNum; ++i)
    {
        if (std::isfinite(arm_joint_lower_limits_(i)) && std::isfinite(arm_joint_upper_limits_(i)))
        {
            arm_q_target_(i) = std::min(
                std::max(arm_q_target_(i), arm_joint_lower_limits_(i)),
                arm_joint_upper_limits_(i));
        }

        const double q_error = arm_q_target_(i) - arm_q(i);
        const double dq_limit = std::abs(arm_position_pid_output_limits_(i));
        arm_dq_desired_(i) =
            dq_limit > 0.0
                ? std::max(
                      -dq_limit,
                      std::min(
                          dq_limit,
                          q_error / std::max(controller_dt_, 1e-6)))
                : q_error / std::max(controller_dt_, 1e-6);
    }

    const Eigen::Matrix<double, 7, 1> physical_arm_gravity =
        moca_dynamic_pin_->getArmGravityCompensation();
    arm_tau_command_.setZero();

    const double manipulability_measure =
        computeWholeBodyManipulability(mobile_base_position_, arm_q);
    const double directional_force_capacity =
        computeDirectionalForceCapability(
            arm_jacobian_in_world_frame,
            physical_arm_gravity);

    sendCommand(mobile_base_target_, arm_q_target_);

    if (rate_trigger_())
    {
        geometry_msgs::Wrench base_error_msg;
        base_error_msg.force.x = mobile_base_position_error_(0);
        base_error_msg.force.y = mobile_base_position_error_(1);
        base_error_msg.force.z = 0.0;
        base_error_msg.torque.x = 0.0;
        base_error_msg.torque.y = 0.0;
        base_error_msg.torque.z = mobile_base_position_error_(2);
        virtual_torques_pub_.publish(base_error_msg);

        std_msgs::Float64 manipulability_msg;
        manipulability_msg.data = manipulability_measure;
        manipulability_pub_.publish(manipulability_msg);

        std_msgs::Float64 directional_force_capacity_msg;
        directional_force_capacity_msg.data = directional_force_capacity;
        directional_force_capacity_pub_.publish(directional_force_capacity_msg);

        geometry_msgs::Vector3Stamped desired_force_msg;
        desired_force_msg.header.stamp = ros::Time::now();
        desired_force_msg.header.frame_id = world_frame_;
        desired_force_msg.vector.x = desired_force_d_.x();
        desired_force_msg.vector.y = desired_force_d_.y();
        desired_force_msg.vector.z = desired_force_d_.z();
        desired_force_pub_.publish(desired_force_msg);

        std_msgs::Float64 nullspace_acc_norm_msg;
        nullspace_acc_norm_msg.data = 0.0;
        nullspace_task_acceleration_norm_pub_.publish(nullspace_acc_norm_msg);

        visualization_msgs::Marker nullspace_base_arrow_msg;
        nullspace_base_arrow_msg.header.stamp = ros::Time::now();
        nullspace_base_arrow_msg.header.frame_id = world_frame_;
        nullspace_base_arrow_msg.ns = "position_base_error";
        nullspace_base_arrow_msg.id = 0;
        nullspace_base_arrow_msg.type = visualization_msgs::Marker::ARROW;
        nullspace_base_arrow_msg.action = visualization_msgs::Marker::ADD;
        nullspace_base_arrow_msg.pose.orientation.w = 1.0;
        nullspace_base_arrow_msg.scale.x = 0.03;
        nullspace_base_arrow_msg.scale.y = 0.06;
        nullspace_base_arrow_msg.scale.z = 0.10;
        nullspace_base_arrow_msg.color.r = 1.0f;
        nullspace_base_arrow_msg.color.g = 0.2f;
        nullspace_base_arrow_msg.color.b = 0.1f;
        nullspace_base_arrow_msg.color.a = 0.95f;
        nullspace_base_arrow_msg.lifetime =
            ros::Duration(2.0 / std::max(1, publish_rate_hz_));

        geometry_msgs::Point arrow_start;
        arrow_start.x = w_T_mobile_base_.translation().x();
        arrow_start.y = w_T_mobile_base_.translation().y();
        arrow_start.z = w_T_mobile_base_.translation().z();

        geometry_msgs::Point arrow_end;
        arrow_end.x =
            arrow_start.x +
            nullspace_base_arrow_scale_ * mobile_base_position_error_(0);
        arrow_end.y =
            arrow_start.y +
            nullspace_base_arrow_scale_ * mobile_base_position_error_(1);
        arrow_end.z = arrow_start.z;
        nullspace_base_arrow_msg.points.push_back(arrow_start);
        nullspace_base_arrow_msg.points.push_back(arrow_end);
        nullspace_base_arrow_pub_.publish(nullspace_base_arrow_msg);

        visualization_msgs::Marker base_cmd_velocity_arrow_msg;
        base_cmd_velocity_arrow_msg.header.stamp = ros::Time::now();
        base_cmd_velocity_arrow_msg.header.frame_id = world_frame_;
        base_cmd_velocity_arrow_msg.ns = "base_cmd_velocity";
        base_cmd_velocity_arrow_msg.id = 0;
        base_cmd_velocity_arrow_msg.type = visualization_msgs::Marker::ARROW;
        base_cmd_velocity_arrow_msg.action = visualization_msgs::Marker::ADD;
        base_cmd_velocity_arrow_msg.pose.orientation.w = 1.0;
        base_cmd_velocity_arrow_msg.scale.x = 0.03;
        base_cmd_velocity_arrow_msg.scale.y = 0.06;
        base_cmd_velocity_arrow_msg.scale.z = 0.10;
        base_cmd_velocity_arrow_msg.color.r = 0.1f;
        base_cmd_velocity_arrow_msg.color.g = 0.95f;
        base_cmd_velocity_arrow_msg.color.b = 0.2f;
        base_cmd_velocity_arrow_msg.color.a = 0.95f;
        base_cmd_velocity_arrow_msg.lifetime =
            ros::Duration(2.0 / std::max(1, publish_rate_hz_));

        geometry_msgs::Point cmd_arrow_start = arrow_start;
        cmd_arrow_start.z += 0.05;
        geometry_msgs::Point cmd_arrow_end;
        cmd_arrow_end.x =
            cmd_arrow_start.x +
            base_cmd_velocity_arrow_scale_ * base_velocity_command_world_(0);
        cmd_arrow_end.y =
            cmd_arrow_start.y +
            base_cmd_velocity_arrow_scale_ * base_velocity_command_world_(1);
        cmd_arrow_end.z = cmd_arrow_start.z;
        base_cmd_velocity_arrow_msg.points.push_back(cmd_arrow_start);
        base_cmd_velocity_arrow_msg.points.push_back(cmd_arrow_end);
        base_cmd_velocity_arrow_pub_.publish(base_cmd_velocity_arrow_msg);

        const Eigen::Quaterniond moca_orientation_msg(w_T_ee_.linear());
        const Eigen::Quaterniond moca_orientation_d_msg(moca_rotation_d_);

        if (moca_position_publisher_.trylock())
        {
            moca_position_publisher_.msg_.header.stamp = ros::Time::now();
            moca_position_publisher_.msg_.header.frame_id = world_frame_;
            moca_position_publisher_.msg_.pose.position.x = w_T_ee_.translation().x();
            moca_position_publisher_.msg_.pose.position.y = w_T_ee_.translation().y();
            moca_position_publisher_.msg_.pose.position.z = w_T_ee_.translation().z();
            moca_position_publisher_.msg_.pose.orientation.x = moca_orientation_msg.x();
            moca_position_publisher_.msg_.pose.orientation.y = moca_orientation_msg.y();
            moca_position_publisher_.msg_.pose.orientation.z = moca_orientation_msg.z();
            moca_position_publisher_.msg_.pose.orientation.w = moca_orientation_msg.w();
            moca_position_publisher_.unlockAndPublish();
        }

        if (moca_velocity_publisher_.trylock())
        {
            moca_velocity_publisher_.msg_.header.stamp = ros::Time::now();
            moca_velocity_publisher_.msg_.header.frame_id = world_frame_;
            moca_velocity_publisher_.msg_.twist.linear.x = moca_twist(0);
            moca_velocity_publisher_.msg_.twist.linear.y = moca_twist(1);
            moca_velocity_publisher_.msg_.twist.linear.z = moca_twist(2);
            moca_velocity_publisher_.msg_.twist.angular.x = moca_twist(3);
            moca_velocity_publisher_.msg_.twist.angular.y = moca_twist(4);
            moca_velocity_publisher_.msg_.twist.angular.z = moca_twist(5);
            moca_velocity_publisher_.unlockAndPublish();
        }

        if (base_command_velocity_publisher_.trylock())
        {
            base_command_velocity_publisher_.msg_.header.stamp = ros::Time::now();
            base_command_velocity_publisher_.msg_.header.frame_id = world_frame_;
            base_command_velocity_publisher_.msg_.twist.linear.x = base_velocity_command_world_(0);
            base_command_velocity_publisher_.msg_.twist.linear.y = base_velocity_command_world_(1);
            base_command_velocity_publisher_.msg_.twist.linear.z = 0.0;
            base_command_velocity_publisher_.msg_.twist.angular.x = 0.0;
            base_command_velocity_publisher_.msg_.twist.angular.y = 0.0;
            base_command_velocity_publisher_.msg_.twist.angular.z = base_velocity_command_world_(2);
            base_command_velocity_publisher_.unlockAndPublish();
        }

        if (moca_target_pose_publisher_.trylock())
        {
            moca_target_pose_publisher_.msg_.header.stamp = ros::Time::now();
            moca_target_pose_publisher_.msg_.header.frame_id = world_frame_;
            moca_target_pose_publisher_.msg_.pose.position.x = moca_position_d_(0);
            moca_target_pose_publisher_.msg_.pose.position.y = moca_position_d_(1);
            moca_target_pose_publisher_.msg_.pose.position.z = moca_position_d_(2);
            moca_target_pose_publisher_.msg_.pose.orientation.x = moca_orientation_d_msg.x();
            moca_target_pose_publisher_.msg_.pose.orientation.y = moca_orientation_d_msg.y();
            moca_target_pose_publisher_.msg_.pose.orientation.z = moca_orientation_d_msg.z();
            moca_target_pose_publisher_.msg_.pose.orientation.w = moca_orientation_d_msg.w();
            moca_target_pose_publisher_.unlockAndPublish();
        }

        if (nullspace_task_acceleration_publisher_.trylock())
        {
            nullspace_task_acceleration_publisher_.msg_.header.stamp = ros::Time::now();
            nullspace_task_acceleration_publisher_.msg_.header.frame_id = world_frame_;
            nullspace_task_acceleration_publisher_.msg_.twist.linear.x = 0.0;
            nullspace_task_acceleration_publisher_.msg_.twist.linear.y = 0.0;
            nullspace_task_acceleration_publisher_.msg_.twist.linear.z = 0.0;
            nullspace_task_acceleration_publisher_.msg_.twist.angular.x = 0.0;
            nullspace_task_acceleration_publisher_.msg_.twist.angular.y = 0.0;
            nullspace_task_acceleration_publisher_.msg_.twist.angular.z = 0.0;
            nullspace_task_acceleration_publisher_.unlockAndPublish();
        }

        if (moca_joint_state_publisher_.trylock())
        {
            moca_joint_state_publisher_.msg_.header.stamp = ros::Time::now();
            moca_joint_state_publisher_.msg_.position[0] = mobile_base_position_(0);
            moca_joint_state_publisher_.msg_.position[1] = mobile_base_position_(1);
            moca_joint_state_publisher_.msg_.position[2] = mobile_base_position_(2);
            moca_joint_state_publisher_.msg_.velocity[0] = mobile_base_velocity_(0);
            moca_joint_state_publisher_.msg_.velocity[1] = mobile_base_velocity_(1);
            moca_joint_state_publisher_.msg_.velocity[2] = mobile_base_velocity_(2);
            moca_joint_state_publisher_.msg_.effort[0] = 0.0;
            moca_joint_state_publisher_.msg_.effort[1] = 0.0;
            moca_joint_state_publisher_.msg_.effort[2] = 0.0;

            for (int i = 0; i < kRobotArmNum; ++i)
            {
                const int whole_body_index = 3 + i;
                moca_joint_state_publisher_.msg_.position[whole_body_index] = arm_q(i);
                moca_joint_state_publisher_.msg_.velocity[whole_body_index] = arm_dq(i);
                moca_joint_state_publisher_.msg_.effort[whole_body_index] = arm_tau_command_(i);
            }
            moca_joint_state_publisher_.unlockAndPublish();
        }

        if (arm_task_torque_publisher_.trylock())
        {
            arm_task_torque_publisher_.msg_.header.stamp = ros::Time::now();
            for (int i = 0; i < kRobotArmNum; ++i)
            {
                arm_task_torque_publisher_.msg_.effort[i] = arm_tau_command_(i);
            }
            arm_task_torque_publisher_.unlockAndPublish();
        }

        if (arm_nullspace_torque_publisher_.trylock())
        {
            arm_nullspace_torque_publisher_.msg_.header.stamp = ros::Time::now();
            for (int i = 0; i < kRobotArmNum; ++i)
            {
                arm_nullspace_torque_publisher_.msg_.effort[i] = 0.0;
            }
            arm_nullspace_torque_publisher_.unlockAndPublish();
        }

        if (moca_target_state_publisher_.trylock())
        {
            moca_target_state_publisher_.msg_.header.stamp = ros::Time::now();
            moca_target_state_publisher_.msg_.position[0] = mobile_base_target_(0);
            moca_target_state_publisher_.msg_.position[1] = mobile_base_target_(1);
            moca_target_state_publisher_.msg_.position[2] = mobile_base_target_(2);
            moca_target_state_publisher_.msg_.velocity[0] = base_velocity_command_world_(0);
            moca_target_state_publisher_.msg_.velocity[1] = base_velocity_command_world_(1);
            moca_target_state_publisher_.msg_.velocity[2] = base_velocity_command_world_(2);
            moca_target_state_publisher_.msg_.effort[0] = 0.0;
            moca_target_state_publisher_.msg_.effort[1] = 0.0;
            moca_target_state_publisher_.msg_.effort[2] = 0.0;

            for (int i = 0; i < kRobotArmNum; ++i)
            {
                const int whole_body_index = 3 + i;
                moca_target_state_publisher_.msg_.position[whole_body_index] = arm_q_target_(i);
                moca_target_state_publisher_.msg_.velocity[whole_body_index] = arm_dq_desired_(i);
                moca_target_state_publisher_.msg_.effort[whole_body_index] = 0.0;
            }
            moca_target_state_publisher_.unlockAndPublish();
        }

        const ros::Time tf_stamp = ros::Time::now();
        const Eigen::Affine3d base_T_ee_actual =
            w_T_mobile_base_.inverse() * w_T_ee_;

        const Eigen::Quaterniond world_q_base(w_T_mobile_base_.linear());
        tf::Transform world_T_base;
        world_T_base.setOrigin(
            tf::Vector3(
                w_T_mobile_base_.translation().x(),
                w_T_mobile_base_.translation().y(),
                w_T_mobile_base_.translation().z()));
        world_T_base.setRotation(
            tf::Quaternion(
                world_q_base.x(),
                world_q_base.y(),
                world_q_base.z(),
                world_q_base.w()));
        tf_broadcaster_.sendTransform(
            tf::StampedTransform(world_T_base, tf_stamp, world_frame_, base_frame_));

        const Eigen::Quaterniond base_q_ee_actual(base_T_ee_actual.linear());
        tf::Transform base_T_ee_tf;
        base_T_ee_tf.setOrigin(
            tf::Vector3(
                base_T_ee_actual.translation().x(),
                base_T_ee_actual.translation().y(),
                base_T_ee_actual.translation().z()));
        base_T_ee_tf.setRotation(
            tf::Quaternion(
                base_q_ee_actual.x(),
                base_q_ee_actual.y(),
                base_q_ee_actual.z(),
                base_q_ee_actual.w()));
        tf_broadcaster_.sendTransform(
            tf::StampedTransform(base_T_ee_tf, tf_stamp, base_frame_, ee_frame_));

        const Eigen::Quaterniond world_q_target(moca_rotation_d_);
        tf::Transform world_T_target;
        world_T_target.setOrigin(
            tf::Vector3(moca_position_d_(0), moca_position_d_(1), moca_position_d_(2)));
        world_T_target.setRotation(
            tf::Quaternion(
                world_q_target.x(),
                world_q_target.y(),
                world_q_target.z(),
                world_q_target.w()));
        tf_broadcaster_.sendTransform(
            tf::StampedTransform(
                world_T_target,
                tf_stamp,
                world_frame_,
                target_pose_frame_));

        Eigen::Affine3d planned_w_T_ee = Eigen::Affine3d::Identity();
        const Eigen::Affine3d planned_w_T_mobile_base =
            computeWorldToBaseTransform(mobile_base_target_);
        computeWholeBodyJacobianInWorld(
            mobile_base_target_,
            arm_q_target_,
            &planned_w_T_ee);
        const Eigen::Affine3d planned_base_T_ee =
            planned_w_T_mobile_base.inverse() * planned_w_T_ee;

        const Eigen::Quaterniond world_q_planned_base(
            planned_w_T_mobile_base.linear());
        tf::Transform world_T_planned_base;
        world_T_planned_base.setOrigin(
            tf::Vector3(
                planned_w_T_mobile_base.translation().x(),
                planned_w_T_mobile_base.translation().y(),
                planned_w_T_mobile_base.translation().z()));
        world_T_planned_base.setRotation(
            tf::Quaternion(
                world_q_planned_base.x(),
                world_q_planned_base.y(),
                world_q_planned_base.z(),
                world_q_planned_base.w()));
        tf_broadcaster_.sendTransform(
            tf::StampedTransform(
                world_T_planned_base,
                tf_stamp,
                world_frame_,
                planned_base_frame_));

        const Eigen::Quaterniond planned_base_q_ee(planned_base_T_ee.linear());
        tf::Transform planned_base_T_ee_tf;
        planned_base_T_ee_tf.setOrigin(
            tf::Vector3(
                planned_base_T_ee.translation().x(),
                planned_base_T_ee.translation().y(),
                planned_base_T_ee.translation().z()));
        planned_base_T_ee_tf.setRotation(
            tf::Quaternion(
                planned_base_q_ee.x(),
                planned_base_q_ee.y(),
                planned_base_q_ee.z(),
                planned_base_q_ee.w()));
        tf_broadcaster_.sendTransform(
            tf::StampedTransform(
                planned_base_T_ee_tf,
                tf_stamp,
                planned_base_frame_,
                planned_ee_frame_));

        visualization_msgs::Marker planned_ee_target_position_error_arrow_msg;
        planned_ee_target_position_error_arrow_msg.header.stamp = ros::Time::now();
        planned_ee_target_position_error_arrow_msg.header.frame_id = world_frame_;
        planned_ee_target_position_error_arrow_msg.ns =
            "planned_ee_target_position_error";
        planned_ee_target_position_error_arrow_msg.id = 0;
        planned_ee_target_position_error_arrow_msg.type =
            visualization_msgs::Marker::ARROW;
        planned_ee_target_position_error_arrow_msg.action =
            visualization_msgs::Marker::ADD;
        planned_ee_target_position_error_arrow_msg.pose.orientation.w = 1.0;
        planned_ee_target_position_error_arrow_msg.scale.x = 0.02;
        planned_ee_target_position_error_arrow_msg.scale.y = 0.04;
        planned_ee_target_position_error_arrow_msg.scale.z = 0.06;
        planned_ee_target_position_error_arrow_msg.color.r = 1.0f;
        planned_ee_target_position_error_arrow_msg.color.g = 0.8f;
        planned_ee_target_position_error_arrow_msg.color.b = 0.1f;
        planned_ee_target_position_error_arrow_msg.color.a = 0.95f;
        planned_ee_target_position_error_arrow_msg.lifetime =
            ros::Duration(2.0 / std::max(1, publish_rate_hz_));

        geometry_msgs::Point planned_error_arrow_start;
        planned_error_arrow_start.x = planned_w_T_ee.translation().x();
        planned_error_arrow_start.y = planned_w_T_ee.translation().y();
        planned_error_arrow_start.z = planned_w_T_ee.translation().z();

        geometry_msgs::Point planned_error_arrow_end;
        planned_error_arrow_end.x = moca_position_d_(0);
        planned_error_arrow_end.y = moca_position_d_(1);
        planned_error_arrow_end.z = moca_position_d_(2);

        planned_ee_target_position_error_arrow_msg.points.push_back(
            planned_error_arrow_start);
        planned_ee_target_position_error_arrow_msg.points.push_back(
            planned_error_arrow_end);
        planned_ee_target_position_error_arrow_pub_.publish(
            planned_ee_target_position_error_arrow_msg);

        const Eigen::Matrix3d planned_rotation_error_matrix =
            planned_w_T_ee.linear().transpose() * moca_rotation_d_;
        const Eigen::AngleAxisd planned_rotation_error_axis_angle(
            planned_rotation_error_matrix);
        const Eigen::Vector3d planned_rotation_error_world =
            planned_w_T_ee.linear() *
            (planned_rotation_error_axis_angle.axis() *
             planned_rotation_error_axis_angle.angle());

        visualization_msgs::Marker planned_ee_target_orientation_error_arrow_msg;
        planned_ee_target_orientation_error_arrow_msg.header.stamp = ros::Time::now();
        planned_ee_target_orientation_error_arrow_msg.header.frame_id = world_frame_;
        planned_ee_target_orientation_error_arrow_msg.ns =
            "planned_ee_target_orientation_error";
        planned_ee_target_orientation_error_arrow_msg.id = 0;
        planned_ee_target_orientation_error_arrow_msg.type =
            visualization_msgs::Marker::ARROW;
        planned_ee_target_orientation_error_arrow_msg.action =
            visualization_msgs::Marker::ADD;
        planned_ee_target_orientation_error_arrow_msg.pose.orientation.w = 1.0;
        planned_ee_target_orientation_error_arrow_msg.scale.x = 0.015;
        planned_ee_target_orientation_error_arrow_msg.scale.y = 0.03;
        planned_ee_target_orientation_error_arrow_msg.scale.z = 0.05;
        planned_ee_target_orientation_error_arrow_msg.color.r = 0.95f;
        planned_ee_target_orientation_error_arrow_msg.color.g = 0.2f;
        planned_ee_target_orientation_error_arrow_msg.color.b = 0.95f;
        planned_ee_target_orientation_error_arrow_msg.color.a = 0.95f;
        planned_ee_target_orientation_error_arrow_msg.lifetime =
            ros::Duration(2.0 / std::max(1, publish_rate_hz_));

        geometry_msgs::Point planned_orientation_arrow_start =
            planned_error_arrow_start;
        geometry_msgs::Point planned_orientation_arrow_end;
        planned_orientation_arrow_end.x =
            planned_orientation_arrow_start.x + planned_rotation_error_world.x();
        planned_orientation_arrow_end.y =
            planned_orientation_arrow_start.y + planned_rotation_error_world.y();
        planned_orientation_arrow_end.z =
            planned_orientation_arrow_start.z + planned_rotation_error_world.z();

        planned_ee_target_orientation_error_arrow_msg.points.push_back(
            planned_orientation_arrow_start);
        planned_ee_target_orientation_error_arrow_msg.points.push_back(
            planned_orientation_arrow_end);
        planned_ee_target_orientation_error_arrow_pub_.publish(
            planned_ee_target_orientation_error_arrow_msg);
    }

    return true;
}

double ControllerManager::computePidCommand(
    const double error,
    double& integral_state,
    double& previous_error,
    const double kp,
    const double ki,
    const double kd,
    const double output_limit) const
{
    const double derivative =
        controller_dt_ > 1e-9 ? (error - previous_error) / controller_dt_ : 0.0;
    const double unsaturated_output =
        kp * error + integral_state + kd * derivative;

    double output = unsaturated_output;
    if (output_limit > 0.0)
    {
        output = std::max(-output_limit, std::min(output_limit, output));
    }

    integral_state += ki * error * controller_dt_;
    integral_state += pid_anti_windup_coeff_ * (output - unsaturated_output);
    if (output_limit > 0.0)
    {
        integral_state = std::max(-output_limit, std::min(output_limit, integral_state));
    }

    previous_error = error;
    return output;
}

void ControllerManager::resetPidStates()
{
    base_position_pid_integral_.setZero();
    base_position_pid_previous_error_.setZero();
    arm_position_pid_integral_.setZero();
    arm_position_pid_previous_error_.setZero();
    arm_velocity_pid_integral_.setZero();
    arm_velocity_pid_previous_error_.setZero();
}

Eigen::Affine3d ControllerManager::computeWorldToBaseTransform(
    const Eigen::Vector3d& mobile_base_state) const
{
    Eigen::Affine3d w_T_mobile_base = Eigen::Affine3d::Identity();
    w_T_mobile_base.translation() <<
        mobile_base_state.x(),
        mobile_base_state.y(),
        0.0;
    w_T_mobile_base.linear() =
        Eigen::AngleAxisd(
            mobile_base_state.z(),
            Eigen::Vector3d::UnitZ()).toRotationMatrix();
    return w_T_mobile_base;
}

Eigen::Matrix<double, 6, 3> ControllerManager::computeMobileBaseJacobianInWorld(
    const Eigen::Affine3d& w_T_mobile_base,
    const Eigen::Affine3d& w_T_ee) const
{
    Eigen::Matrix<double, 6, 3> J_base = Eigen::Matrix<double, 6, 3>::Zero();

    const Eigen::Affine3d base_T_ee = w_T_mobile_base.inverse() * w_T_ee;
    const Eigen::Vector3d r = base_T_ee.translation();

    J_base(0, 0) = 1.0;
    J_base(1, 1) = 1.0;
    J_base(0, 2) = -r.y();
    J_base(1, 2) =  r.x();
    J_base(5, 2) = 1.0;

    const Eigen::Matrix3d R = w_T_mobile_base.rotation();
    Eigen::Matrix<double, 6, 6> Ad = Eigen::Matrix<double, 6, 6>::Zero();
    Ad.topLeftCorner<3, 3>() = R;
    Ad.bottomRightCorner<3, 3>() = R;

    return Ad * J_base;
}

Eigen::Matrix<double, 6, 10> ControllerManager::computeWholeBodyJacobianInWorld(
    const Eigen::Vector3d& mobile_base_state,
    const Eigen::Matrix<double, 7, 1>& arm_q,
    Eigen::Affine3d* w_T_ee) const
{
    const Eigen::Affine3d w_T_mobile_base =
        computeWorldToBaseTransform(mobile_base_state);
    const Eigen::Affine3d base_T_ee =
        moca_dynamic_pin_->getRelativeTransform(arm_q, base_frame_, ee_frame_);
    const Eigen::Affine3d current_w_T_ee = w_T_mobile_base * base_T_ee;

    if (w_T_ee != nullptr)
    {
        *w_T_ee = current_w_T_ee;
    }

    const Eigen::Matrix<double, 6, 7> arm_jacobian_in_base_frame =
        moca_dynamic_pin_->getArmJacobian(arm_q, ee_frame_);
    Eigen::Matrix<double, 6, 7> arm_jacobian_in_world_frame =
        Eigen::Matrix<double, 6, 7>::Zero();
    arm_jacobian_in_world_frame.topRows<3>() =
        w_T_mobile_base.linear() * arm_jacobian_in_base_frame.topRows<3>();
    arm_jacobian_in_world_frame.bottomRows<3>() =
        w_T_mobile_base.linear() * arm_jacobian_in_base_frame.bottomRows<3>();

    const Eigen::Matrix<double, 6, 3> mobile_base_jacobian_in_world_frame =
        computeMobileBaseJacobianInWorld(w_T_mobile_base, current_w_T_ee);

    Eigen::Matrix<double, 6, 10> whole_body_jacobian_in_world_frame;
    whole_body_jacobian_in_world_frame <<
        mobile_base_jacobian_in_world_frame,
        arm_jacobian_in_world_frame;
    return whole_body_jacobian_in_world_frame;
}

double ControllerManager::computeWholeBodyManipulability(
    const Eigen::Vector3d& mobile_base_state,
    const Eigen::Matrix<double, 7, 1>& arm_q) const
{
    constexpr double manipulability_regularization = 1e-6;
    constexpr double manipulability_epsilon = 1e-8;

    const Eigen::Matrix<double, 6, 10> jacobian =
        computeWholeBodyJacobianInWorld(mobile_base_state, arm_q);
    Eigen::Matrix<double, 3, 8> optimization_jacobian =
        Eigen::Matrix<double, 3, 8>::Zero();
    optimization_jacobian.col(0) = jacobian.topRows<3>().col(2);
    optimization_jacobian.rightCols<7>() = jacobian.topRows<3>().rightCols<7>();

    const Eigen::Matrix3d gram =
        optimization_jacobian * optimization_jacobian.transpose() +
        manipulability_regularization * Eigen::Matrix3d::Identity();

    return std::sqrt(std::max(manipulability_epsilon, gram.determinant()));
}

Eigen::Matrix<double, 14, 1> ControllerManager::buildGravityAdjustedArmTorqueLimits(
    const Eigen::Matrix<double, 7, 1>& arm_gravity) const
{
    Eigen::Matrix<double, 14, 1> adjusted_limits =
        Eigen::Matrix<double, 14, 1>::Zero();
    adjusted_limits.head<7>() = arm_torque_limits_ - arm_gravity;
    adjusted_limits.tail<7>() = -arm_torque_limits_ - arm_gravity;
    return adjusted_limits;
}

double ControllerManager::computeDirectionalForceCapability(
    const Eigen::Matrix<double, 6, 7>& arm_jacobian_in_world_frame,
    const Eigen::Matrix<double, 7, 1>& arm_gravity) const
{
    const Eigen::Matrix<double, 3, 7> translational_jacobian =
        arm_jacobian_in_world_frame.topRows<3>();
    const Eigen::Matrix<double, 7, 1> tau_direction =
        translational_jacobian.transpose() * directional_force_axis_world_;
    const double tau_direction_norm = tau_direction.norm();
    if (tau_direction_norm < 1e-9)
    {
        return 0.0;
    }

    const Eigen::Matrix<double, 7, 1> tau_direction_normalized =
        tau_direction / tau_direction_norm;
    Eigen::Matrix<double, 14, 14> block_matrix =
        Eigen::Matrix<double, 14, 14>::Zero();
    block_matrix.topLeftCorner<7, 7>() =
        tau_direction_normalized.asDiagonal();
    block_matrix.bottomRightCorner<7, 7>() =
        tau_direction_normalized.asDiagonal();

    const Eigen::Matrix<double, 14, 1> adjusted_tau_limits =
        buildGravityAdjustedArmTorqueLimits(arm_gravity);
    const Eigen::Matrix<double, 14, 1> directional_limits =
        (block_matrix + 1e-8 * Eigen::Matrix<double, 14, 14>::Identity())
            .fullPivLu()
            .solve(adjusted_tau_limits);

    double min_positive_limit = std::numeric_limits<double>::infinity();
    for (int i = 0; i < directional_limits.size(); ++i)
    {
        if (directional_limits(i) > 0.0)
        {
            min_positive_limit = std::min(min_positive_limit, directional_limits(i));
        }
    }
    if (!std::isfinite(min_positive_limit))
    {
        return 0.0;
    }

    const Eigen::Vector3d optimal_force =
        min_positive_limit *
        translational_jacobian.transpose()
            .completeOrthogonalDecomposition()
            .solve(tau_direction_normalized);
    if (!optimal_force.allFinite())
    {
        return 0.0;
    }

    return optimal_force.norm();
}

void ControllerManager::targetPoseCallback(
    const moca_trajectory_generator::TargetPoseCommandConstPtr& msg)
{
    moca_position_d_ << msg->pose.position.x,
                        msg->pose.position.y,
                        msg->pose.position.z;
    moca_twist_d_ << msg->twist.linear.x,
                     msg->twist.linear.y,
                     msg->twist.linear.z,
                     msg->twist.angular.x,
                     msg->twist.angular.y,
                     msg->twist.angular.z;
    desired_force_d_ << msg->desired_force.x,
                        msg->desired_force.y,
                        msg->desired_force.z;

    if (std::isfinite(msg->finger_command))
    {
        finger_command_ = msg->finger_command;
    }

    const Eigen::Quaterniond requested_orientation(
        msg->pose.orientation.w,
        msg->pose.orientation.x,
        msg->pose.orientation.y,
        msg->pose.orientation.z);
    if (requested_orientation.norm() > 1e-6)
    {
        moca_rotation_d_ = requested_orientation.normalized().toRotationMatrix();
    }

    if (!msg->use_nullspace_joint_target)
    {
        whole_body_target_active_ = false;
        return;
    }

    Eigen::Vector3d requested_base_target = Eigen::Vector3d::Zero();
    Eigen::Matrix<double, 7, 1> requested_arm_target =
        Eigen::Matrix<double, 7, 1>::Zero();

    for (int i = 0; i < 3; ++i)
    {
        if (!std::isfinite(msg->base_planar_positions[i]))
        {
            ROS_WARN_THROTTLE(
                1.0,
                "PositionControllerManager: received non-finite base target, ignoring this message.");
            return;
        }
        requested_base_target(i) = msg->base_planar_positions[i];
    }
    requested_base_target(2) = wrapAngle(requested_base_target(2));

    for (int i = 0; i < kRobotArmNum; ++i)
    {
        if (!std::isfinite(msg->arm_joint_positions[i]))
        {
            ROS_WARN_THROTTLE(
                1.0,
                "PositionControllerManager: received non-finite arm target, ignoring this message.");
            return;
        }

        requested_arm_target(i) = msg->arm_joint_positions[i];
        if (std::isfinite(arm_joint_lower_limits_(i)) && std::isfinite(arm_joint_upper_limits_(i)))
        {
            requested_arm_target(i) = std::min(
                std::max(requested_arm_target(i), arm_joint_lower_limits_(i)),
                arm_joint_upper_limits_(i));
        }
    }

    mobile_base_target_ = requested_base_target;
    arm_q_target_ = requested_arm_target;
    whole_body_target_active_ = true;
}

void ControllerManager::resetToHomeCallback(const std_msgs::EmptyConstPtr& msg)
{
    (void)msg;
    display_.info("PositionControllerManager: received reset-to-home request.");
    tcpip_client_.request_reset_to_home();
    whole_body_target_active_ = false;
    base_velocity_command_world_.setZero();
    arm_dq_desired_.setZero();
    arm_tau_command_.setZero();
    resetPidStates();
}

void ControllerManager::sendCommand(
    const Eigen::Vector3d& base_command,
    const Eigen::Matrix<double, 7, 1>& arm_command)
{
    std::vector<double> mobile_cmd(3, 0.0);
    std::vector<double> arm_cmd(7, 0.0);

    for (int i = 0; i < 3; ++i)
    {
        mobile_cmd[static_cast<std::size_t>(i)] = base_command(i);
    }
    for (int i = 0; i < 7; ++i)
    {
        arm_cmd[static_cast<std::size_t>(i)] = arm_command(i);
    }

    tcpip_client_.set_command(
        mobile_cmd,
        arm_cmd,
        finger_command_,
        polytope_wx::ArmCommandInterface::POSITION,
        polytope_wx::MobileCommandInterface::POSITION_WORLD);
    // Legacy impedance-style command path kept here for reference:
    // tcpip_client_.set_command(
    //     mobile_cmd,
    //     arm_cmd,
    //     finger_command_,
    //     polytope_wx::ArmCommandInterface::TORQUE,
    //     polytope_wx::MobileCommandInterface::VELOCITY);
}

ControllerManager::~ControllerManager()
{
    tcpip_client_.request_reset_to_home();
    ros::Duration(0.05).sleep();
    nh_.deleteParam("controller_started");
}

} // namespace WBPositionController
