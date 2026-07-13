/*
 * Author: Pietro Balatti, Yuqiang Wu
 *
 * Whole-body impedance controller for MOCA platform.
 * This version keeps the control computation, hardcodes the configuration,
 * and uses TCP/IP + Pinocchio only.
 */

#include "Moca_controller/Moca_controller.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <vector>
#include <Eigen/SVD>

#include "eigen_conversions/eigen_msg.h"
#include "hrii_utils/Math.h"
#include <ros/package.h>

namespace
{

constexpr int kRobotArmNum = 7;
constexpr int kWholeBodyDim = 10;

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

Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& v)
{
    Eigen::Matrix3d skew;
    skew << 0.0, -v.z(), v.y(),
            v.z(), 0.0, -v.x(),
            -v.y(), v.x(), 0.0;
    return skew;
}

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

std::string toLowerCopy(std::string value)
{
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool isHRIIManagedInterfaceType(const std::string& interface_type)
{
    const std::string normalized = toLowerCopy(interface_type);
    return normalized == "simulation" || normalized == "hardware";
}

constexpr const char* kForceCapabilitySettingsParamNs =
    "/force_capability_settings/arm_torque_limits";

} // namespace

namespace WBCartesianImpedanceController
{

bool ControllerManager::init(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    nh_ = ros::NodeHandle("~");

    display_.setName(ros::this_node::getName());
    display_.info("ControllerManager: initialize controller.");

    nh_.param("publish_rate_hz", publish_rate_hz_, publish_rate_hz_);
    nh_.param("controller_dt", controller_dt_, controller_dt_);
    nh_.param("enable_matlogger", enable_matlogger_, enable_matlogger_);
    nh_.param(
        "enable_nullspace_optimization",
        enable_nullspace_optimization_,
        enable_nullspace_optimization_);
    nh_.param(
        "enable_jdot_compensation",
        enable_jdot_compensation_,
        enable_jdot_compensation_);
    nh_.param(
        "arm_only_nullspace_test",
        arm_only_nullspace_test_,
        arm_only_nullspace_test_);
    nh_.param("base_max_linear_velocity", base_max_linear_velocity_, base_max_linear_velocity_);
    nh_.param("base_max_angular_velocity", base_max_angular_velocity_, base_max_angular_velocity_);
    nh_.param(
        "base_cmd_linear_velocity_deadband",
        base_cmd_linear_velocity_deadband_,
        base_cmd_linear_velocity_deadband_);
    nh_.param(
        "base_cmd_angular_velocity_deadband",
        base_cmd_angular_velocity_deadband_,
        base_cmd_angular_velocity_deadband_);
    base_cmd_linear_velocity_deadband_ =
        std::max(0.0, base_cmd_linear_velocity_deadband_);
    base_cmd_angular_velocity_deadband_ =
        std::max(0.0, base_cmd_angular_velocity_deadband_);
    nh_.param(
        "base_nullspace_linear_stiffness",
        base_nullspace_linear_stiffness_,
        base_nullspace_linear_stiffness_);
    nh_.param(
        "base_nullspace_yaw_stiffness",
        base_nullspace_yaw_stiffness_,
        base_nullspace_yaw_stiffness_);
    nh_.param("joint_limit_margin_ratio", joint_limit_margin_ratio_, joint_limit_margin_ratio_);
    nh_.param("manipulability_gain", manipulability_gain_, manipulability_gain_);
    nh_.param(
        "manipulability_gradient_step",
        manipulability_gradient_step_,
        manipulability_gradient_step_);
    nh_.param("manipulability_epsilon", manipulability_epsilon_, manipulability_epsilon_);
    nh_.param(
        "manipulability_regularization",
        manipulability_regularization_,
        manipulability_regularization_);
    nh_.param(
        "manipulability_objective_max_norm",
        manipulability_objective_max_norm_,
        manipulability_objective_max_norm_);
    nh_.param("command_backend", command_backend_, command_backend_);
    nh_.param("interface_type", interface_type_, interface_type_);
    nh_.param(
        "robot_interface_config_file",
        robot_interface_config_file_,
        robot_interface_config_file_);
    nh_.param(
        "arm_transmission_type",
        arm_transmission_type_,
        arm_transmission_type_);
    nh_.param(
        "hrii_interface_adds_gravity_compensation",
        hrii_interface_adds_gravity_compensation_,
        hrii_interface_adds_gravity_compensation_);
    nh_.param(
        "enable_franka_tool_load_control_compensation",
        enable_franka_tool_load_control_compensation_,
        enable_franka_tool_load_control_compensation_);
    nh_.param(
        "controller_ee_frame_is_franka_flange",
        controller_ee_frame_is_franka_flange_,
        controller_ee_frame_is_franka_flange_);
    nh_.param(
        "hold_until_first_target_pose",
        hold_until_first_target_pose_,
        hold_until_first_target_pose_);
    nh_.param(
        "command_output_enabled",
        command_output_enabled_,
        command_output_enabled_);
    if (!command_output_enabled_)
    {
        ROS_WARN("ControllerManager: command_output_enabled=false; dry-run mode is active and robot motion commands will not be sent.");
    }
    nh_.param("tcp_server_ip", tcp_server_ip_, tcp_server_ip_);
    nh_.param("tcp_server_port", tcp_server_port_, tcp_server_port_);
    nh_.param("tcp_retry_delay_ms", tcp_retry_delay_ms_, tcp_retry_delay_ms_);
    nh_.param("urdf_path", urdf_path_, urdf_path_);
    nh_.param("base_frame", base_frame_, base_frame_);
    nh_.param("ee_frame", ee_frame_, ee_frame_);
    nh_.param("world_frame", world_frame_, world_frame_);
    nh_.param(
        "gravity_acceleration_mps2",
        gravity_acceleration_mps2_,
        gravity_acceleration_mps2_);
    nh_.param(
        "debug_current_base_frame",
        debug_current_base_frame_,
        debug_current_base_frame_);
    nh_.param(
        "debug_current_ee_frame",
        debug_current_ee_frame_,
        debug_current_ee_frame_);
    nh_.param("target_pose_frame", target_pose_frame_, target_pose_frame_);
    nh_.param("planned_base_frame", planned_base_frame_, planned_base_frame_);
    nh_.param("planned_ee_frame", planned_ee_frame_, planned_ee_frame_);
    nh_.param("reset_to_home_topic", reset_to_home_topic_, reset_to_home_topic_);
    nh_.param(
        "force_capability_settings_topic",
        force_capability_settings_topic_,
        force_capability_settings_topic_);
    nh_.param(
        "franka_tool_state_topic",
        franka_tool_state_topic_,
        franka_tool_state_topic_);
    polytope_urdf_path_ = urdf_path_;
    polytope_ee_frame_ = ee_frame_;
    nh_.param(
        "enable_online_nullspace_optimizer",
        enable_online_nullspace_optimizer_,
        enable_online_nullspace_optimizer_);
    nh_.param(
        "online_nullspace_optimizer_rate_hz",
        online_nullspace_optimizer_rate_hz_,
        online_nullspace_optimizer_rate_hz_);
    nh_.param("polytope_urdf_path", polytope_urdf_path_, polytope_urdf_path_);
    nh_.param("polytope_ee_frame", polytope_ee_frame_, polytope_ee_frame_);
    urdf_path_ = resolveRosPath(urdf_path_);
    polytope_urdf_path_ = resolveRosPath(polytope_urdf_path_);
    robot_interface_config_file_ = resolveRosPath(robot_interface_config_file_);
    nh_.param(
        "left_finger_joint_name",
        left_finger_joint_name_,
        left_finger_joint_name_);
    nh_.param(
        "right_finger_joint_name",
        right_finger_joint_name_,
        right_finger_joint_name_);

    if (isHRIIManagedInterfaceType(interface_type_) && !useHRIIBackend())
    {
        display_.error(
            "ControllerManager: refusing to start the TCP/IP backend while interface_type=\"",
            interface_type_,
            "\". Gazebo and hardware launches must use command_backend=hrii.");
        return false;
    }

    nh_.param("w_r", w_r_, w_r_);
    nh_.param("w_f", w_f_, w_f_);
    nh_.param("w_r_yaw", w_r_yaw_, w_r_yaw_);
    nh_.param("J_weight", J_weight_, J_weight_);
    nh_.param("filter_params", filter_params_, filter_params_);
    nh_.param("delta_tau_max", delta_tau_max_, delta_tau_max_);
    nh_.param("impedance_filtering", impedance_filtering_, impedance_filtering_);
    nh_.param(
        "initial_transl_cartesian_stiffness",
        initial_transl_cartesian_stiffness_,
        initial_transl_cartesian_stiffness_);
    nh_.param(
        "initial_rot_cartesian_stiffness",
        initial_rot_cartesian_stiffness_,
        initial_rot_cartesian_stiffness_);
    nh_.param(
        "initial_nullspace_stiffness",
        initial_nullspace_stiffness_,
        initial_nullspace_stiffness_);
    nh_.param("joint_limit_torque_gain", joint_limit_torque_gain_, joint_limit_torque_gain_);
    nh_.param("finger_command", finger_command_, finger_command_);
    nh_.param(
        "task_acceleration_feedforward_gain",
        task_acceleration_feedforward_gain_,
        task_acceleration_feedforward_gain_);
    nh_.param(
        "online_nullspace_gradient_weight",
        online_nullspace_gradient_weight_,
        online_nullspace_gradient_weight_);
    nh_.param(
        "nullspace_base_arrow_scale",
        nullspace_base_arrow_scale_,
        nullspace_base_arrow_scale_);
    nh_.param(
        "base_cmd_velocity_arrow_scale",
        base_cmd_velocity_arrow_scale_,
        base_cmd_velocity_arrow_scale_);
    nh_.param("min_linear_wrench", min_linear_wrench_, min_linear_wrench_);
    nh_.param("max_linear_wrench", max_linear_wrench_, max_linear_wrench_);
    nh_.param("min_angular_wrench", min_angular_wrench_, min_angular_wrench_);
    nh_.param("max_angular_wrench", max_angular_wrench_, max_angular_wrench_);
    nh_.param(
        "optimizer_max_iterations_per_waypoint",
        optimizer_max_iterations_per_waypoint_,
        optimizer_max_iterations_per_waypoint_);
    nh_.param("optimizer_pose_gain", optimizer_pose_gain_, optimizer_pose_gain_);
    nh_.param(
        "optimizer_nullspace_step_size",
        optimizer_nullspace_step_size_,
        optimizer_nullspace_step_size_);
    nh_.param(
        "optimizer_max_joint_update_norm",
        optimizer_max_joint_update_norm_,
        optimizer_max_joint_update_norm_);
    nh_.param(
        "optimizer_pose_tolerance",
        optimizer_pose_tolerance_,
        optimizer_pose_tolerance_);
    nh_.param(
        "optimizer_capability_weight",
        optimizer_capability_weight_,
        optimizer_capability_weight_);
    nh_.param(
        "optimizer_manipulability_weight",
        optimizer_manipulability_weight_,
        optimizer_manipulability_weight_);
    nh_.param(
        "optimizer_joint_limit_weight",
        optimizer_joint_limit_weight_,
        optimizer_joint_limit_weight_);
    nh_.param(
        "optimizer_smoothness_weight",
        optimizer_smoothness_weight_,
        optimizer_smoothness_weight_);
    nh_.param(
        "optimizer_nominal_weight",
        optimizer_nominal_weight_,
        optimizer_nominal_weight_);
    nh_.param(
        "optimizer_capability_alpha",
        optimizer_capability_alpha_,
        optimizer_capability_alpha_);
    nh_.param(
        "optimizer_capability_near_weight",
        optimizer_capability_near_weight_,
        optimizer_capability_near_weight_);
    nh_.param(
        "optimizer_capability_over_weight",
        optimizer_capability_over_weight_,
        optimizer_capability_over_weight_);
    nh_.param(
        "optimizer_capability_over4_weight",
        optimizer_capability_over4_weight_,
        optimizer_capability_over4_weight_);
    nh_.param(
        "optimizer_constrain_orientation",
        optimizer_constrain_orientation_,
        optimizer_constrain_orientation_);
    nh_.param(
        "optimizer_verbose",
        optimizer_verbose_,
        optimizer_verbose_);

    auto loadVector3Param = [this](const std::string& param_name,
                                   const std::array<double, 3>& defaults) {
        std::vector<double> values;
        if (!nh_.getParam(param_name, values) || values.size() != 3)
        {
            values.assign(defaults.begin(), defaults.end());
        }

        return Eigen::Vector3d(values[0], values[1], values[2]);
    };

    nh_.setParam("controller_started", false);

    target_pose_sub_ = nh_.subscribe(
        "/target_pose", 1, &ControllerManager::targetPoseCallback, this);
    force_capability_settings_sub_ = nh_.subscribe(
        force_capability_settings_topic_,
        1,
        &ControllerManager::forceCapabilitySettingsCallback,
        this);
    reset_to_home_sub_ = nh_.subscribe(
        reset_to_home_topic_, 1, &ControllerManager::resetToHomeCallback, this);
    franka_tool_state_sub_ = nh_.subscribe(
        franka_tool_state_topic_,
        1,
        &ControllerManager::frankaToolStateCallback,
        this);
    if (useHRIIBackend())
    {
        ros::NodeHandle robot_ns_nh(ros::this_node::getNamespace());
        mobile_base_inertia_sub_ = robot_ns_nh.subscribe(
            "mobile_base_control/mass_matrix",
            1,
            &ControllerManager::mobileBaseInertiaCallback,
            this,
            ros::TransportHints().reliable().tcpNoDelay());
    }
    virtual_torques_pub_ = nh_.advertise<geometry_msgs::Wrench>("vir_torque", 1);
    manipulability_pub_ = nh_.advertise<std_msgs::Float64>("manipulability", 1);
    manipulability_ellipsoid_pub_ =
        nh_.advertise<std_msgs::Float64MultiArray>("manipulability_ellipsoid", 1);
    manipulability_gradient_norm_pub_ =
        nh_.advertise<std_msgs::Float64>("manipulability_gradient_norm", 1);
    directional_force_capacity_pub_ =
        nh_.advertise<std_msgs::Float64>("directional_force_capacity", 1);
    force_polytope_vertices_pub_ =
        nh_.advertise<std_msgs::Float64MultiArray>("force_polytope_vertices", 1);
    desired_force_pub_ =
        nh_.advertise<geometry_msgs::Vector3Stamped>("desired_force", 1);
    nullspace_task_acceleration_norm_pub_ =
        nh_.advertise<std_msgs::Float64>("nullspace_task_acceleration_norm", 1);
    nullspace_base_arrow_pub_ =
        nh_.advertise<visualization_msgs::Marker>("nullspace_base_arrow", 1);
    base_cmd_velocity_arrow_pub_ =
        nh_.advertise<visualization_msgs::Marker>("base_cmd_velocity_arrow", 1);

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
    std::string namespace_str = ros::this_node::getNamespace();
    if (!namespace_str.empty() && namespace_str.front() == '/')
    {
        namespace_str.erase(0, 1);
    }

    const char* home_path = std::getenv("HOME");
    const std::string log_root = home_path ? home_path : "/tmp";
    const std::string log_path =
        log_root + "/log/" + namespace_str + "_simple_wb_controller";

    if (enable_matlogger_)
    {
        logger_ = XBot::MatLogger2::MakeLogger(log_path);
        appender_ = XBot::MatAppender::MakeInstance();
        appender_->add_logger(logger_);
        appender_->start_flush_thread();
    }

    rate_trigger_ = HRII_Utils::RateTrigger(publish_rate_hz_);

    const Eigen::Vector3d mobile_base_inertia_diagonal =
        loadVector3Param("mobile_base_inertia_diagonal", {105.0, 105.0, 20.0});
    const Eigen::Vector3d mobile_base_admittance_damping_diagonal =
        loadVector3Param("mobile_base_admittance_damping_diagonal", {180.0, 180.0, 30.0});
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

    mobile_base_inertia_ = mobile_base_inertia_diagonal.asDiagonal();
    mobile_base_admittance_damping_ =
        mobile_base_admittance_damping_diagonal.asDiagonal();

    if (useHRIIBackend())
    {
        if (robot_interface_config_file_.empty())
        {
            display_.error(
                "ControllerManager: robot_interface_config_file is empty while command_backend=hrii.");
            return false;
        }

        if (!initHRIIInterfaces())
        {
            return false;
        }
    }
    else
    {
        if (!tcpip_client_.connect_with_retry(
                tcp_server_ip_, tcp_server_port_, -1, tcp_retry_delay_ms_))
        {
            display_.error(
                "ControllerManager: TCP/IP backend initialization aborted before a connection was established.");
            return false;
        }
        tcpip_client_.start_receive_thread();
        tcpip_client_.send_thread_start();
    }

    sendCommand(Eigen::Vector3d::Zero(), Eigen::Matrix<double, 7, 1>::Zero());

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
        arm_joint_limit_margins_ =
            joint_limit_margin_ratio_ * (arm_joint_upper_limits_ - arm_joint_lower_limits_);
    }
    if (pin_model.effortLimit.size() >= kRobotArmNum)
    {
        arm_torque_limits_ = pin_model.effortLimit.head<kRobotArmNum>();
        for (int i = 0; i < kRobotArmNum; ++i)
        {
            if (!std::isfinite(arm_torque_limits_(i)) || arm_torque_limits_(i) <= 0.0)
            {
                arm_torque_limits_(i) = 1.0;
            }
            else
            {
                arm_torque_limits_(i) = std::abs(arm_torque_limits_(i));
            }
        }
    }

    std::vector<double> configured_arm_torque_limits;
    if (ros::param::get(kForceCapabilitySettingsParamNs, configured_arm_torque_limits))
    {
        if (configured_arm_torque_limits.size() != kRobotArmNum)
        {
            display_.error(
                "ControllerManager: expected ",
                kRobotArmNum,
                " arm torque limits in ROS param namespace ",
                kForceCapabilitySettingsParamNs,
                ", but got ",
                static_cast<int>(configured_arm_torque_limits.size()),
                ".");
            return false;
        }

        for (int i = 0; i < kRobotArmNum; ++i)
        {
            const double torque_limit = configured_arm_torque_limits[i];
            if (!std::isfinite(torque_limit) || torque_limit <= 0.0)
            {
                display_.error(
                    "ControllerManager: invalid arm torque limit from ROS param at joint index ",
                    i,
                    ".");
                return false;
            }
            arm_torque_limits_(i) = std::abs(torque_limit);
        }
    }

    if (!initializeOnlineNullspaceOptimizer())
    {
        return false;
    }

    cartesian_stiffness_.setZero();
    cartesian_damping_.setZero();
    cartesian_stiffness_target_.setZero();
    cartesian_damping_target_.setZero();
    nullspace_joint_stiffness_.setZero();
    nullspace_joint_damping_.setZero();
    nullspace_joint_stiffness_target_.setZero();
    nullspace_joint_damping_target_.setZero();
    moca_q_.setZero();
    moca_dq_.setZero();
    moca_q_d_nullspace_.setZero();
    mobile_base_position_.setZero();
    mobile_base_velocity_.setZero();
    mobile_base_vir_torques_.setZero();
    external_wrench_.setZero();
    previous_arm_tau_command_.setZero();

    ros::Duration(0.1).sleep();

    display_.info(
        "ControllerManager: controller initialized with backend \"",
        command_backend_,
        "\".");
    return true;
}

bool ControllerManager::useHRIIBackend() const
{
    const std::string backend = toLowerCopy(command_backend_);
    return backend == "hrii" || backend == "ros" || backend == "interface";
}

bool ControllerManager::initHRIIInterfaces()
{
    arm_interface_ = HRII::RAInterface::RoboticArmInterface::GenerateRoboticArmInterface(
        robot_interface_config_file_,
        interface_type_,
        arm_transmission_type_);
    if (arm_interface_.get() == nullptr)
    {
        display_.error("ControllerManager: failed to create HRII arm interface.");
        return false;
    }
    if (!arm_interface_->init())
    {
        display_.error("ControllerManager: failed to initialize HRII arm interface.");
        return false;
    }

    mobile_base_interface_ = HRII::MORInterface::MobileRobotInterface::GenerateMobileRobotInterface(
        robot_interface_config_file_);
    if (mobile_base_interface_.get() == nullptr)
    {
        display_.error("ControllerManager: failed to create HRII mobile base interface.");
        return false;
    }
    if (!mobile_base_interface_->init())
    {
        display_.error("ControllerManager: failed to initialize HRII mobile base interface.");
        return false;
    }

    return true;
}

bool ControllerManager::initializeOnlineNullspaceOptimizer()
{
    if (!enable_online_nullspace_optimizer_)
    {
        online_nullspace_optimizer_initialized_ = false;
        online_optimizer_model_joint_count_ = 0;
        return true;
    }

    polytope_wx::NullspaceOptimizationConfig config;
    config.urdf_path = polytope_urdf_path_;
    config.ee_frame = polytope_ee_frame_;
    config.torque_limits = arm_torque_limits_.cast<double>();
    config.locked_joint_names = {
        left_finger_joint_name_,
        right_finger_joint_name_
    };
    config.max_iterations_per_waypoint = optimizer_max_iterations_per_waypoint_;
    config.pose_gain = optimizer_pose_gain_;
    config.nullspace_step_size = optimizer_nullspace_step_size_;
    config.max_joint_update_norm = optimizer_max_joint_update_norm_;
    config.pose_tolerance = optimizer_pose_tolerance_;
    config.capability_weight = optimizer_capability_weight_;
    config.manipulability_weight = optimizer_manipulability_weight_;
    config.joint_limit_weight = optimizer_joint_limit_weight_;
    config.smoothness_weight = optimizer_smoothness_weight_;
    config.nominal_weight = 0.0;
    config.capability_alpha = optimizer_capability_alpha_;
    config.capability_near_weight = optimizer_capability_near_weight_;
    config.capability_over_weight = optimizer_capability_over_weight_;
    config.capability_over4_weight = optimizer_capability_over4_weight_;
    config.constrain_orientation = optimizer_constrain_orientation_;
    config.verbose = optimizer_verbose_;

    if (!online_nullspace_optimizer_.initialize(config))
    {
        display_.error("Failed to initialize controller-side polytope optimizer.");
        online_nullspace_optimizer_initialized_ = false;
        return false;
    }

    online_optimizer_model_joint_count_ =
        static_cast<int>(online_nullspace_optimizer_.model().nq);
    if (online_optimizer_model_joint_count_ < kRobotArmNum)
    {
        display_.error("Controller-side polytope optimizer model has fewer than 7 joints.");
        online_nullspace_optimizer_initialized_ = false;
        return false;
    }

    online_nullspace_optimizer_initialized_ = true;
    display_.info(
        "ControllerManager: controller-side polytope nullspace optimizer initialized with current arm torque limits.");
    return true;
}

bool ControllerManager::updatePinocchioFromArmState(
    const Eigen::Matrix<double, 7, 1>& arm_q,
    const Eigen::Matrix<double, 7, 1>& arm_dq)
{
    polytope_wx::RobotMessage robot_state_now{};
    robot_state_now.timestamp = ros::Time::now().toSec();

    for (int i = 0; i < kRobotArmNum; ++i)
    {
        robot_state_now.arm_positions[i] = arm_q(i);
        robot_state_now.arm_velocities[i] = arm_dq(i);
    }

    return moca_dynamic_pin_->updateFromRobotMessage(robot_state_now);
}

bool ControllerManager::readHRIIRobotState(
    nav_msgs::Odometry* mobile_base_odom,
    Eigen::Matrix<double, 7, 1>* arm_q,
    Eigen::Matrix<double, 7, 1>* arm_dq)
{
    if (mobile_base_odom == nullptr || arm_q == nullptr || arm_dq == nullptr)
    {
        display_.error("ControllerManager: null output passed to readHRIIRobotState.");
        return false;
    }

    if (mobile_base_interface_.get() == nullptr || arm_interface_.get() == nullptr)
    {
        display_.error("ControllerManager: HRII interfaces are not initialized.");
        return false;
    }

    if (!mobile_base_interface_->updateRobotState())
    {
        display_.error("ControllerManager: mobile base interface update failed.");
        return false;
    }
    *mobile_base_odom = mobile_base_interface_->getOdometry();

    if (!arm_interface_->updateRobotState(*mobile_base_odom))
    {
        display_.error("ControllerManager: arm interface update failed.");
        return false;
    }

    const sensor_msgs::JointState::Ptr arm_joint_state = arm_interface_->getJointsStates();
    if (!arm_joint_state)
    {
        display_.error("ControllerManager: arm interface returned a null joint state.");
        return false;
    }
    if (arm_joint_state->position.size() < kRobotArmNum)
    {
        display_.error(
            "ControllerManager: arm joint state position size is ",
            arm_joint_state->position.size(),
            " but at least ",
            kRobotArmNum,
            " entries are required.");
        return false;
    }

    for (int i = 0; i < kRobotArmNum; ++i)
    {
        (*arm_q)(i) = arm_joint_state->position[static_cast<std::size_t>(i)];
    }

    arm_dq->setZero();
    if (arm_joint_state->velocity.size() >= kRobotArmNum)
    {
        for (int i = 0; i < kRobotArmNum; ++i)
        {
            (*arm_dq)(i) = arm_joint_state->velocity[static_cast<std::size_t>(i)];
        }
    }
    else
    {
        ROS_WARN_THROTTLE(
            2.0,
            "ControllerManager: arm joint velocity feedback is shorter than 7 entries, using zero velocities.");
    }

    if (!updatePinocchioFromArmState(*arm_q, *arm_dq))
    {
        display_.error("ControllerManager: failed to update Pinocchio from HRII arm state.");
        return false;
    }

    return true;
}

bool ControllerManager::starting()
{
    display_.info("ControllerManager: starting controller.");

    Eigen::Matrix<double, 7, 1> arm_q = Eigen::Matrix<double, 7, 1>::Zero();
    Eigen::Matrix<double, 7, 1> arm_dq = Eigen::Matrix<double, 7, 1>::Zero();
    nav_msgs::Odometry mobile_base_odom;

    if (useHRIIBackend())
    {
        if (!readHRIIRobotState(&mobile_base_odom, &arm_q, &arm_dq))
        {
            display_.error("Failed to read initial state from HRII interfaces.");
            return false;
        }
    }
    else
    {
        const polytope_wx::RobotMessage robot_state_now = tcpip_client_.get_robot_state();
        if (!moca_dynamic_pin_->updateFromRobotMessage(robot_state_now))
        {
            display_.error("Failed to update Pinocchio state from TCP/IP.");
            return false;
        }

        arm_q = Eigen::Map<const Eigen::Matrix<double, 7, 1>>(robot_state_now.arm_positions);
        arm_dq = Eigen::Map<const Eigen::Matrix<double, 7, 1>>(robot_state_now.arm_velocities);
        mobile_base_odom = tcpip_client_.toOdometryMsg(robot_state_now);
    }

    tf::poseMsgToEigen(mobile_base_odom.pose.pose, w_T_mobile_base_);
    if (useHRIIBackend())
    {
        w_T_ee_ = arm_interface_->getO_T_EE();
    }
    else
    {
        const Eigen::Affine3d base_T_ee =
            moca_dynamic_pin_->getRelativeTransform(base_frame_, ee_frame_);
        w_T_ee_ = w_T_mobile_base_ * base_T_ee;
    }


    moca_position_d_ = w_T_ee_.translation();
    moca_rotation_d_ = w_T_ee_.linear();
    moca_position_d_target_ = moca_position_d_;
    moca_rotation_d_target_ = moca_rotation_d_;
    moca_twist_d_.setZero();
    moca_twist_d_target_.setZero();
    moca_acceleration_d_.setZero();
    moca_acceleration_d_target_.setZero();
    desired_force_d_.setZero();
    desired_force_d_target_.setZero();

    cartesian_stiffness_target_.setZero();
    cartesian_damping_target_.setZero();
    cartesian_stiffness_target_.topLeftCorner<3, 3>() =
        initial_transl_cartesian_stiffness_ * Eigen::Matrix3d::Identity();
    cartesian_stiffness_target_.bottomRightCorner<3, 3>() =
        initial_rot_cartesian_stiffness_ * Eigen::Matrix3d::Identity();
    cartesian_damping_target_.topLeftCorner<3, 3>() =
        2.0 * 0.7 *
        std::sqrt(initial_transl_cartesian_stiffness_) *
        Eigen::Matrix3d::Identity();
    cartesian_damping_target_.bottomRightCorner<3, 3>() =
        2.0 * 0.7 *
        std::sqrt(initial_rot_cartesian_stiffness_) *
        Eigen::Matrix3d::Identity();

    cartesian_stiffness_ = cartesian_stiffness_target_;
    cartesian_damping_ = cartesian_damping_target_;

    nullspace_joint_stiffness_target_.setZero();
    nullspace_joint_damping_target_.setZero();
    nullspace_joint_stiffness_target_(0, 0) = base_nullspace_linear_stiffness_;
    nullspace_joint_stiffness_target_(1, 1) = base_nullspace_linear_stiffness_;
    nullspace_joint_stiffness_target_(2, 2) = base_nullspace_yaw_stiffness_;
    nullspace_joint_damping_target_(0, 0) = 2.0 * std::sqrt(base_nullspace_linear_stiffness_);
    nullspace_joint_damping_target_(1, 1) = 2.0 * std::sqrt(base_nullspace_linear_stiffness_);
    nullspace_joint_damping_target_(2, 2) = 2.0 * std::sqrt(base_nullspace_yaw_stiffness_);
    nullspace_joint_stiffness_target_.bottomRightCorner<7, 7>() =
        initial_nullspace_stiffness_ * Eigen::Matrix<double, 7, 7>::Identity();
    nullspace_joint_damping_target_.bottomRightCorner<7, 7>() =
        2.0 * std::sqrt(initial_nullspace_stiffness_) *
        Eigen::Matrix<double, 7, 7>::Identity();

    nullspace_joint_stiffness_ = nullspace_joint_stiffness_target_;
    nullspace_joint_damping_ = nullspace_joint_damping_target_;
    nullspace_stiffness_target_ = initial_nullspace_stiffness_;

    const Eigen::Quaterniond mobile_base_orientation(
        mobile_base_odom.pose.pose.orientation.w,
        mobile_base_odom.pose.pose.orientation.x,
        mobile_base_odom.pose.pose.orientation.y,
        mobile_base_odom.pose.pose.orientation.z);

    mobile_base_yaw_ = std::atan2(
        2.0 * (mobile_base_orientation.w() * mobile_base_orientation.z() +
               mobile_base_orientation.x() * mobile_base_orientation.y()),
        1.0 - 2.0 * (mobile_base_orientation.y() * mobile_base_orientation.y() +
                     mobile_base_orientation.z() * mobile_base_orientation.z()));
    mobile_base_position_ << mobile_base_odom.pose.pose.position.x,
                             mobile_base_odom.pose.pose.position.y,
                             mobile_base_yaw_;

    if (useHRIIBackend())
    {
        current_arm_gravity_ =
            arm_interface_->getGravityCompensation().head<kRobotArmNum>();
    }
    else
    {
        current_arm_gravity_ = moca_dynamic_pin_->getArmGravityCompensation();
    }
    mobile_base_velocity_.setZero();
    mobile_base_vir_torques_.setZero();
    if (useHRIIBackend())
    {
        const Eigen::MatrixXd whole_body_jacobian_hrii = arm_interface_->getJacobian();
        previous_whole_body_jacobian_.setZero();
        previous_whole_body_jacobian_.block<6, 2>(0, 0) =
            whole_body_jacobian_hrii.block(0, 0, 6, 2);
        previous_whole_body_jacobian_.col(2) =
            whole_body_jacobian_hrii.col(5);
        previous_whole_body_jacobian_.rightCols<kRobotArmNum>() =
            whole_body_jacobian_hrii.block(0, 6, 6, kRobotArmNum);
    }
    else
    {
        previous_whole_body_jacobian_ =
            computeWholeBodyJacobianInWorld(mobile_base_position_, arm_q);
    }
    has_previous_whole_body_jacobian_ = true;

    external_whole_body_nullspace_target_active_ = false;
    moca_q_d_nullspace_.head<3>() = mobile_base_position_;
    moca_q_d_nullspace_.tail<kRobotArmNum>() = arm_q;

    moca_q_.setZero();
    moca_dq_.setZero();
    cmd_vel.setZero();
    previous_arm_tau_command_ = useHRIIBackend()
        ? current_arm_gravity_
        : Eigen::Matrix<double, 7, 1>::Zero();
    last_online_nullspace_optimization_time_ = ros::Time(0);
    has_last_online_optimized_model_configuration_ = false;
    online_nullspace_gradient_objective_.setZero();
    has_received_target_pose_ = false;

    if (enable_online_nullspace_optimizer_ &&
        online_nullspace_optimizer_initialized_ &&
        !external_whole_body_nullspace_target_active_)
    {
        last_online_optimized_model_configuration_ =
            Eigen::VectorXd::Zero(online_optimizer_model_joint_count_);
        for (int i = 0; i < std::min(kRobotArmNum, online_optimizer_model_joint_count_); ++i)
        {
            last_online_optimized_model_configuration_(i) = arm_q(i);
        }
        has_last_online_optimized_model_configuration_ = true;
        moca_q_d_nullspace_.tail<kRobotArmNum>() = arm_q;
    }

    sendCommand(Eigen::Vector3d::Zero(), previous_arm_tau_command_);

    nh_.setParam("controller_started", true);
    display_.info("ControllerManager: controller started.");

    return true;
}

bool ControllerManager::update()
{
    Eigen::Matrix<double, 7, 1> arm_q = Eigen::Matrix<double, 7, 1>::Zero();
    Eigen::Matrix<double, 7, 1> arm_dq = Eigen::Matrix<double, 7, 1>::Zero();
    nav_msgs::Odometry mobile_base_odom;

    if (useHRIIBackend())
    {
        if (!readHRIIRobotState(&mobile_base_odom, &arm_q, &arm_dq))
        {
            display_.error("Failed to update state from HRII interfaces during control loop.");
            return false;
        }
    }
    else
    {
        const polytope_wx::RobotMessage robot_state_now = tcpip_client_.get_robot_state();
        if (!moca_dynamic_pin_->updateFromRobotMessage(robot_state_now))
        {
            display_.error("Failed to update Pinocchio state during control loop.");
            return false;
        }

        arm_q = Eigen::Map<const Eigen::Matrix<double, 7, 1>>(robot_state_now.arm_positions);
        arm_dq = Eigen::Map<const Eigen::Matrix<double, 7, 1>>(robot_state_now.arm_velocities);
        mobile_base_odom = tcpip_client_.toOdometryMsg(robot_state_now);
    }

    tf::poseMsgToEigen(mobile_base_odom.pose.pose, w_T_mobile_base_);

    const Eigen::Quaterniond mobile_base_orientation(
        mobile_base_odom.pose.pose.orientation.w,
        mobile_base_odom.pose.pose.orientation.x,
        mobile_base_odom.pose.pose.orientation.y,
        mobile_base_odom.pose.pose.orientation.z);

    mobile_base_yaw_ = std::atan2(
        2.0 * (mobile_base_orientation.w() * mobile_base_orientation.z() +
               mobile_base_orientation.x() * mobile_base_orientation.y()),
        1.0 - 2.0 * (mobile_base_orientation.y() * mobile_base_orientation.y() +
                     mobile_base_orientation.z() * mobile_base_orientation.z()));
    mobile_base_position_ << mobile_base_odom.pose.pose.position.x,
                             mobile_base_odom.pose.pose.position.y,
                             mobile_base_yaw_;

    if (useHRIIBackend())
    {
        const Eigen::Vector3d mobile_base_velocity_in_body_frame(
            mobile_base_odom.twist.twist.linear.x,
            mobile_base_odom.twist.twist.linear.y,
            mobile_base_odom.twist.twist.angular.z);
        mobile_base_velocity_ =
            mobile_base_orientation.toRotationMatrix() * mobile_base_velocity_in_body_frame;
    }
    else
    {
        // Isaac state feedback already reports the planar base velocity in world coordinates.
        // The whole-body controller also uses [x_dot, y_dot, yaw_dot] in world coordinates,
        // so we keep the feedback as-is and only rotate the commanded velocity before sending.
        mobile_base_velocity_ << mobile_base_odom.twist.twist.linear.x,
                                 mobile_base_odom.twist.twist.linear.y,
                                 mobile_base_odom.twist.twist.angular.z;
    }

    Eigen::Matrix<double, 7, 1> arm_coriolis =
        Eigen::Matrix<double, 7, 1>::Zero();
    Eigen::Matrix<double, 7, 1> arm_gravity =
        Eigen::Matrix<double, 7, 1>::Zero();
    Eigen::Matrix<double, 6, 7> arm_jacobian_in_world_frame =
        Eigen::Matrix<double, 6, 7>::Zero();
    Eigen::Matrix<double, 6, 3> mobile_base_jacobian_in_world_frame =
        Eigen::Matrix<double, 6, 3>::Zero();
    Eigen::Matrix<double, 7, 7> arm_inertia =
        Eigen::Matrix<double, 7, 7>::Zero();
    Eigen::MatrixXd whole_body_jacobian_hrii_raw;
    Eigen::MatrixXd arm_inertia_raw_debug;

    if (useHRIIBackend())
    {
        const Eigen::VectorXd arm_coriolis_raw = arm_interface_->getCoriolisCompensation();
        const Eigen::VectorXd arm_gravity_raw = arm_interface_->getGravityCompensation();
        const Eigen::MatrixXd arm_inertia_raw = arm_interface_->getInertiaMatrix();
        const Eigen::MatrixXd whole_body_jacobian_hrii = arm_interface_->getJacobian();
        // std::cout << "whole_body_jacobian_hrii_raw" << whole_body_jacobian_hrii << std::endl;
        whole_body_jacobian_hrii_raw = whole_body_jacobian_hrii;
        arm_inertia_raw_debug = arm_inertia_raw;

        arm_coriolis = arm_coriolis_raw.head<kRobotArmNum>();
        arm_gravity = arm_gravity_raw.head<kRobotArmNum>();
        current_arm_gravity_ = arm_gravity;

        w_T_ee_ = arm_interface_->getO_T_EE();
        arm_jacobian_in_world_frame =
            whole_body_jacobian_hrii.block<6, kRobotArmNum>(0, 6);
        mobile_base_jacobian_in_world_frame.block<6, 2>(0, 0) =
            whole_body_jacobian_hrii.block<6, 2>(0, 0);
        mobile_base_jacobian_in_world_frame.col(2) =
            whole_body_jacobian_hrii.block<6, 1>(0, 5);
        arm_inertia =
            arm_inertia_raw.block<kRobotArmNum, kRobotArmNum>(6, 6);
    }
    else
    {
        arm_coriolis = moca_dynamic_pin_->getArmCoriolisCompensation();
        arm_gravity = moca_dynamic_pin_->getArmGravityCompensation();
        current_arm_gravity_ = arm_gravity;

        const Eigen::Affine3d base_T_ee =
            moca_dynamic_pin_->getRelativeTransform(base_frame_, ee_frame_);
        w_T_ee_ = w_T_mobile_base_ * base_T_ee;

        const Eigen::Matrix<double, 6, 7> arm_jacobian_in_base_frame =
            moca_dynamic_pin_->getArmJacobian(ee_frame_);
        arm_jacobian_in_world_frame.topRows<3>() =
            w_T_mobile_base_.linear() * arm_jacobian_in_base_frame.topRows<3>();
        arm_jacobian_in_world_frame.bottomRows<3>() =
            w_T_mobile_base_.linear() * arm_jacobian_in_base_frame.bottomRows<3>();
        arm_inertia = moca_dynamic_pin_->getArmInertiaMatrix();


        // w_T_mobile_base_ 是机器人底座在世界坐标系中的位姿，w_T_ee_ 是末端执行器在世界坐标系中的位姿。computeMobileBaseJacobianInWorld函数计算了底座雅可比矩阵在世界坐标系中的表示。
        mobile_base_jacobian_in_world_frame =
            computeMobileBaseJacobianInWorld(w_T_mobile_base_, w_T_ee_);
    }

    const Eigen::Matrix<double, 7, 1> franka_tool_load_gravity_torques =
        computeFrankaToolLoadGravityTorques(w_T_ee_, arm_jacobian_in_world_frame);
    if (enable_franka_tool_load_control_compensation_ &&
        franka_tool_load_gravity_torques.allFinite())
    {
        arm_gravity += franka_tool_load_gravity_torques;
        current_arm_gravity_ = arm_gravity;
    }

    // std::cout << "arm_inertia: " << arm_inertia << std::endl;


    Eigen::Matrix<double, 6, 10> whole_body_jacobian_in_world_frame;
    whole_body_jacobian_in_world_frame <<
        mobile_base_jacobian_in_world_frame,
        arm_jacobian_in_world_frame;

    if (hold_until_first_target_pose_ && !has_received_target_pose_)
    {
        moca_twist_d_target_.setZero();
        moca_acceleration_d_target_.setZero();
        desired_force_d_target_.setZero();
        external_whole_body_nullspace_target_active_ = false;
        online_nullspace_gradient_objective_.setZero();
        moca_q_d_nullspace_.head<3>() = mobile_base_position_;
        moca_q_d_nullspace_.tail<kRobotArmNum>() = arm_q;
        mobile_base_vir_torques_.setZero();
        ROS_WARN_THROTTLE(
            2.0,
            "ControllerManager: holding the startup Cartesian pose until the first /target_pose command is received.");
    }

    const Eigen::Vector3d moca_position = w_T_ee_.translation();

    const Eigen::Matrix3d moca_rotation = w_T_ee_.linear();

    Eigen::Quaterniond moca_orientation_msg(moca_rotation);
    moca_orientation_msg.normalize();
    Eigen::Quaterniond moca_orientation_d_msg(moca_rotation_d_);
    moca_orientation_d_msg.normalize();

    // Promote the latest trajectory-generator command to the active target
    // before computing this cycle's Cartesian error and before publishing the
    // controller-side target pose/state topics.
    moca_position_d_ = moca_position_d_target_;
    moca_rotation_d_ = moca_rotation_d_target_;
    moca_twist_d_ = moca_twist_d_target_;
    moca_acceleration_d_ = moca_acceleration_d_target_;
    desired_force_d_ = desired_force_d_target_;

    if (enable_online_nullspace_optimizer_ &&
        online_nullspace_optimizer_initialized_ &&
        !external_whole_body_nullspace_target_active_)
    {
        const ros::Time now = ros::Time::now();
        const double optimization_period =
            1.0 / std::max(1e-3, online_nullspace_optimizer_rate_hz_);
        const bool should_run_online_optimization =
            last_online_nullspace_optimization_time_.isZero() ||
            (now - last_online_nullspace_optimization_time_).toSec() >= optimization_period;

        Eigen::VectorXd current_model_q =
            Eigen::VectorXd::Zero(online_optimizer_model_joint_count_);
        for (int i = 0; i < std::min(kRobotArmNum, online_optimizer_model_joint_count_); ++i)
        {
            current_model_q(i) = arm_q(i);
        }

        if (should_run_online_optimization)
        {
            const Eigen::Vector3d desired_force_in_base =
                w_T_mobile_base_.linear().transpose() * desired_force_d_;
            const Eigen::VectorXd* previous_q =
                has_last_online_optimized_model_configuration_
                    ? &last_online_optimized_model_configuration_
                    : nullptr;

            try
            {
                // Legacy controller-side path kept for reference:
                // const polytope_wx::SingleWaypointOptimizationResult optimization_result =
                //     online_nullspace_optimizer_.optimizeSingleWaypoint(
                //         seed_q,
                //         base_T_target,
                //         desired_force_in_base,
                //         current_model_q,
                //         previous_q);
                const polytope_wx::NullspaceGradientStepResult gradient_step_result =
                    online_nullspace_optimizer_.computeNullspaceGradientStep(
                        current_model_q,
                        desired_force_in_base,
                        current_model_q,
                        previous_q);

                if (gradient_step_result.success &&
                    gradient_step_result.objective_gradient.size() >= kRobotArmNum)
                {
                    online_nullspace_gradient_objective_.setZero();
                    online_nullspace_gradient_objective_.tail<kRobotArmNum>() =
                        -online_nullspace_gradient_weight_ *
                        gradient_step_result.objective_gradient.head<kRobotArmNum>();
                    moca_q_d_nullspace_.tail<kRobotArmNum>() = arm_q;
                    last_online_optimized_model_configuration_ =
                        current_model_q;
                    has_last_online_optimized_model_configuration_ = true;
                    last_online_nullspace_optimization_time_ = now;
                }
                else
                {
                    last_online_nullspace_optimization_time_ = now;
                    online_nullspace_gradient_objective_.setZero();
                    if (!has_last_online_optimized_model_configuration_)
                    {
                        moca_q_d_nullspace_.tail<kRobotArmNum>() = arm_q;
                    }
                    ROS_WARN_THROTTLE(
                        1.0,
                        "ControllerManager: online nullspace gradient step failed: %s",
                        gradient_step_result.message.c_str());
                }
            }
            catch (const std::exception& e)
            {
                last_online_nullspace_optimization_time_ = now;
                ROS_WARN_THROTTLE(
                    1.0,
                    "ControllerManager: online nullspace optimization failed: %s",
                    e.what());
                if (!has_last_online_optimized_model_configuration_)
                {
                    moca_q_d_nullspace_.tail<kRobotArmNum>() = arm_q;
                }
            }
        }
        else if (has_last_online_optimized_model_configuration_)
        {
            moca_q_d_nullspace_.tail<kRobotArmNum>() = arm_q;
        }
    }

    if (moca_position_d_.isZero(1e-3))
    {
        display_.error("Desired position is zero. Refusing to run control update.");
        return false;
    }

    const Eigen::Matrix<double, 6, 1> moca_error =
        computeCartesianPoseError(
            moca_position_d_,
            moca_rotation_d_,
            moca_position,
            moca_rotation);

    moca_q_ << mobile_base_position_, arm_q;
    moca_dq_ << mobile_base_velocity_, arm_dq;
    if (!external_whole_body_nullspace_target_active_)
    {
        moca_q_d_nullspace_.head<3>() = mobile_base_position_;
    }

    const Eigen::Matrix<double, 6, 1> moca_twist =
        whole_body_jacobian_in_world_frame * moca_dq_;
    Eigen::Matrix<double, 6, 1> whole_body_jacobian_dot_times_dq =
        Eigen::Matrix<double, 6, 1>::Zero();
    if (enable_jdot_compensation_ && has_previous_whole_body_jacobian_)
    {
        const double safe_dt = std::max(controller_dt_, 1e-6);
        const Eigen::Matrix<double, 6, 10> whole_body_jacobian_dot =
            (whole_body_jacobian_in_world_frame - previous_whole_body_jacobian_) / safe_dt;
        whole_body_jacobian_dot_times_dq = whole_body_jacobian_dot * moca_dq_;
    }
    // Reference-controller style: use desired-current for both pose and twist errors.
    const Eigen::Matrix<double, 6, 1> moca_twist_error =
        moca_twist_d_ - moca_twist;
    const Eigen::Matrix<double, 6, 1> moca_reference_acceleration =
        enable_jdot_compensation_
            ? (moca_acceleration_d_ - whole_body_jacobian_dot_times_dq)
            : moca_acceleration_d_;

    Eigen::Matrix<double, 10, 10> moca_inertia =
        Eigen::Matrix<double, 10, 10>::Zero();
    moca_inertia.topLeftCorner<3, 3>() = mobile_base_inertia_;
    moca_inertia.bottomRightCorner<7, 7>() = arm_inertia;

    Eigen::Matrix<double, 10, 1> moca_bias =
        Eigen::Matrix<double, 10, 1>::Zero();
    moca_bias.tail<7>() = arm_coriolis + arm_gravity;

    const Eigen::Matrix<double, 10, 10> identity_10 =
        Eigen::Matrix<double, 10, 10>::Identity();
    const Eigen::Matrix<double, 6, 6> identity_6 =
        Eigen::Matrix<double, 6, 6>::Identity();

    const Eigen::LDLT<Eigen::Matrix<double, 10, 10>> moca_inertia_ldlt(
        moca_inertia);
    const bool moca_inertia_invertible = moca_inertia.fullPivLu().isInvertible();
    const bool moca_inertia_ldlt_success =
        moca_inertia_ldlt.info() == Eigen::Success;
    const Eigen::Matrix<double, 10, 10> inertia_inv =
        moca_inertia_ldlt.solve(identity_10);

    Eigen::Matrix<double, 10, 1> weight_vector;
    weight_vector << w_r_, w_r_, w_r_yaw_,
                     Eigen::Matrix<double, 7, 1>::Constant(w_f_);

    const Eigen::Matrix<double, 10, 10> W_n = weight_vector.asDiagonal();
    const Eigen::Matrix<double, 10, 10> W_n_inv =
        weight_vector.cwiseInverse().asDiagonal();
    const Eigen::Matrix<double, 10, 10> W = inertia_inv * W_n;

    const Eigen::Matrix<double, 10, 6> inertia_inv_jacobian_transpose =
        moca_inertia_ldlt.solve(whole_body_jacobian_in_world_frame.transpose());
    const Eigen::Matrix<double, 6, 6> task_space_inertia_inverse =
        whole_body_jacobian_in_world_frame * inertia_inv_jacobian_transpose;
    const bool task_space_inertia_inverse_invertible =
        task_space_inertia_inverse.fullPivLu().isInvertible();

    const Eigen::Matrix<double, 6, 6> task_space_inertia_regularized =
        task_space_inertia_inverse + 1e-6 * identity_6;
    const Eigen::LDLT<Eigen::Matrix<double, 6, 6>> task_space_inertia_ldlt(
        task_space_inertia_regularized);
    const Eigen::Matrix<double, 6, 6> task_space_inertia =
        task_space_inertia_ldlt.solve(identity_6);

    const Eigen::Matrix<double, 10, 6> weighted_inertia_inv_jacobian_transpose =
        moca_inertia_ldlt.solve(W_n_inv *
                                whole_body_jacobian_in_world_frame.transpose());
    const Eigen::Matrix<double, 6, 6> operational_matrix =
        whole_body_jacobian_in_world_frame *
        weighted_inertia_inv_jacobian_transpose;
    const bool operational_matrix_invertible =
        operational_matrix.fullPivLu().isInvertible();
    const Eigen::LDLT<Eigen::Matrix<double, 6, 6>> operational_matrix_ldlt(
        operational_matrix);

    const Eigen::Matrix<double, 10, 6> moca_weighted_gen_inv_jacobian =
        W_n_inv * whole_body_jacobian_in_world_frame.transpose() *
        operational_matrix_ldlt.solve(task_space_inertia_inverse);

    const Eigen::Matrix<double, 10, 10> N_tor =
        identity_10 -
        W_n_inv * whole_body_jacobian_in_world_frame.transpose() *
            operational_matrix_ldlt.solve(
                whole_body_jacobian_in_world_frame * inertia_inv);

    const Eigen::Matrix<double, 6, 1> moca_task_wrench =
        // Reference-controller form:
        //   F = Lambda * (xddot_d - Jdot*qdot) + D * (xdot_d - xdot) + K * (x_d - x)
        // Legacy sign convention:
        //   F = Lambda * (xddot_d - Jdot*qdot) - D * (xdot - xdot_d) - K * (x - x_d)
        // They are mathematically equivalent; we keep the reference form explicitly.
        // task_acceleration_feedforward_gain_ *
        task_space_inertia * moca_reference_acceleration +
        // 速度误差
        cartesian_stiffness_ * moca_error 
        // 位置误差
        + cartesian_damping_ * moca_twist_error
        ;

    // In this controller the mobile base is executed through an admittance /
    // velocity layer rather than direct torque control, so the weighted
    // generalized-force mapping remains the more stable choice here.





    Eigen::Matrix<double, 10, 1> moca_tau_task =
        moca_weighted_gen_inv_jacobian * moca_task_wrench;
    // A direct J^T * F mapping was tried for debugging, but it tends to
    // over-excite this mixed base/arm execution chain:
    // Eigen::Matrix<double, 10, 1> moca_tau_task =
    //     whole_body_jacobian_in_world_frame.transpose() * moca_task_wrench;

    const Eigen::Matrix<double, 7, 1> arm_joint_limit_torques =
        computeArmJointLimitTorques(arm_q);
    Eigen::Matrix<double, 10, 1> joint_limit_torques =
        Eigen::Matrix<double, 10, 1>::Zero();
    joint_limit_torques.tail<7>() = arm_joint_limit_torques;

    Eigen::Matrix<double, 10, 1> manipulability_gradient =
        computeWholeBodyManipulabilityGradient(mobile_base_position_, arm_q);
    const double manipulability_gradient_norm = manipulability_gradient.norm();

    Eigen::Matrix<double, 10, 1> nullspace_state_error =
        moca_q_d_nullspace_ - moca_q_;
    nullspace_state_error(2) =
        std::atan2(std::sin(nullspace_state_error(2)), std::cos(nullspace_state_error(2)));

    const bool use_external_whole_body_nullspace_target =
        external_whole_body_nullspace_target_active_;
    Eigen::Matrix<double, 10, 1> nullspace_objective =
        enable_online_nullspace_optimizer_ && online_nullspace_optimizer_initialized_ &&
                !use_external_whole_body_nullspace_target
            ? online_nullspace_gradient_objective_
            : nullspace_joint_stiffness_ * nullspace_state_error
                  - nullspace_joint_damping_ * moca_dq_;
        // Legacy joint-space target shaping:
        // nullspace_joint_stiffness_ * (moca_q_d_nullspace_ - moca_q_)
        // - nullspace_joint_damping_ * moca_dq_
        // + joint_limit_torques;
    // The offline whole-body planner publishes [base_x, base_y, base_yaw]
    // directly in the world frame, which matches moca_q_.head<3>().
    // Only zero the base nullspace objective when we are not following an
    // external whole-body target and the controller is handling the base
    // implicitly through the main task / online arm-only nullspace shaping.
    if (!use_external_whole_body_nullspace_target)
    {
        // std::cout<<"Nullspace objective before zeroing base: " << nullspace_objective.transpose() << std::endl;
        nullspace_objective.head<3>().setZero();
    }

    Eigen::Matrix<double, 10, 1> moca_tau_nullspace =
        N_tor * nullspace_objective;
    // Legacy projector form kept for comparison:
    // const Eigen::Matrix<double, 10, 10> nullspace_projector =
    //     Eigen::Matrix<double, 10, 10>::Identity() -
    //     W.inverse() * J_bar *
    //         (J_bar.transpose() * W.inverse() * J_bar).inverse() *
    //         J_bar.transpose();
    // Eigen::Matrix<double, 10, 1> moca_tau_nullspace =
    //     nullspace_projector * nullspace_objective;


    // 零空间动态一致矩阵求解方法
    const Eigen::Matrix<double, 6, 6> lambda = (whole_body_jacobian_in_world_frame*inertia_inv*whole_body_jacobian_in_world_frame.transpose()).inverse();
    const Eigen::Matrix<double, 10, 10> I_w = Eigen::MatrixXd::Identity(10, 10) + W_n*inertia_inv*W_n*inertia_inv;
    const Eigen::Matrix<double, 10, 6> J_bar = inertia_inv * whole_body_jacobian_in_world_frame.transpose()*lambda;

    const Eigen::Matrix<double, 10, 10> nullspace_projector_wu =
    I_w.inverse()-I_w.inverse()*W*J_bar*(J_bar.transpose()*I_w.inverse()*W*J_bar).inverse()
    *J_bar.transpose()*I_w.inverse();

    Eigen::Matrix<double, 10, 1> moca_tau_nullspace_wu =
        nullspace_projector_wu *2* nullspace_objective;



    // const Eigen::Matrix<double, 10, 10> nullspace_projector_wu_2 =

    const Eigen::Matrix<double, 6, 1> nullspace_task_acceleration =
        whole_body_jacobian_in_world_frame * inertia_inv * moca_tau_nullspace;
    const double nullspace_task_acceleration_norm =
        nullspace_task_acceleration.norm();

    const double manipulability_measure =
        computeWholeBodyManipulability(mobile_base_position_, arm_q);
    const double directional_force_capacity =
        computeDirectionalForceCapability(arm_jacobian_in_world_frame, arm_gravity);

    // moca_tau_task.tail(7).setZero();
    // moca_bias.setZero();
    Eigen::Matrix<double, 10, 1> moca_tau_d =
         moca_bias 
        +
         moca_tau_task
        ;




    if(enable_nullspace_optimization_)
    {
        // moca_tau_d+=moca_tau_nullspace;
        moca_tau_d+=moca_tau_nullspace_wu;
    }
    // moca_tau_d.setZero();

    // moca_tau_d(8) = -100;
    // std::cout << "moca_tau_d: " << moca_tau_d.transpose() << std::endl;

    if (enable_matlogger_ && logger_)
    {
        logger_->add("tau_desired_not_saturated", moca_tau_d);
    }

    moca_tau_d.tail<7>() =
        saturateTorqueRate(moca_tau_d.tail<7>(), previous_arm_tau_command_);

    // checkDeadzone(moca_tau_d(0), LINEAR);
    // checkDeadzone(moca_tau_d(1), LINEAR);
    // checkDeadzone(moca_tau_d(2), ANGULAR);

    for (int i = 0; i < kRobotArmNum; ++i)
    {
        const double torque_limit = std::max(1.0, std::abs(arm_torque_limits_(i)));
        moca_tau_d(3 + i) =
            std::max(-torque_limit, std::min(torque_limit, moca_tau_d(3 + i)));
    }

    mobile_base_vir_torques_ = moca_tau_d.head<3>();
    updateMobileBaseAdmittance(mobile_base_vir_torques_);
    const Eigen::Vector3d output_cmd_vel = applyBaseCommandDeadband(cmd_vel);
    external_wrench_.setZero();

    if (rate_trigger_())
    {
        geometry_msgs::Wrench wrench_msg;
        wrench_msg.force.x = mobile_base_vir_torques_(0);
        wrench_msg.force.y = mobile_base_vir_torques_(1);
        wrench_msg.force.z = 0.0;
        wrench_msg.torque.x = 0.0;
        wrench_msg.torque.y = 0.0;
        wrench_msg.torque.z = mobile_base_vir_torques_(2);
        virtual_torques_pub_.publish(wrench_msg);

        std_msgs::Float64 manipulability_msg;
        manipulability_msg.data = manipulability_measure;
        manipulability_pub_.publish(manipulability_msg);

        const Eigen::Matrix3d manipulability_ellipsoid =
            computeWholeBodyManipulabilityEllipsoid(mobile_base_position_, arm_q);
        if (manipulability_ellipsoid.allFinite())
        {
            std_msgs::Float64MultiArray ellipsoid_msg;
            ellipsoid_msg.layout.dim.resize(2);
            ellipsoid_msg.layout.dim[0].label = "rows";
            ellipsoid_msg.layout.dim[0].size = 3;
            ellipsoid_msg.layout.dim[0].stride = 9;
            ellipsoid_msg.layout.dim[1].label = "cols";
            ellipsoid_msg.layout.dim[1].size = 3;
            ellipsoid_msg.layout.dim[1].stride = 3;
            ellipsoid_msg.data.reserve(9);
            for (int row = 0; row < 3; ++row)
            {
                for (int col = 0; col < 3; ++col)
                {
                    ellipsoid_msg.data.push_back(manipulability_ellipsoid(row, col));
                }
            }
            manipulability_ellipsoid_pub_.publish(ellipsoid_msg);
        }

        std_msgs::Float64 manipulability_gradient_norm_msg;
        manipulability_gradient_norm_msg.data = manipulability_gradient_norm;
        manipulability_gradient_norm_pub_.publish(manipulability_gradient_norm_msg);

        std_msgs::Float64 directional_force_capacity_msg;
        directional_force_capacity_msg.data = directional_force_capacity;
        directional_force_capacity_pub_.publish(directional_force_capacity_msg);

        const Eigen::MatrixXd force_polytope_vertices =
            computeArmForcePolytopeVertices(arm_jacobian_in_world_frame, arm_gravity);
        if (force_polytope_vertices.rows() > 0 &&
            force_polytope_vertices.cols() == 3 &&
            force_polytope_vertices.allFinite())
        {
            std_msgs::Float64MultiArray vertices_msg;
            vertices_msg.layout.dim.resize(2);
            vertices_msg.layout.dim[0].label = "vertices";
            vertices_msg.layout.dim[0].size =
                static_cast<unsigned int>(force_polytope_vertices.rows());
            vertices_msg.layout.dim[0].stride =
                static_cast<unsigned int>(force_polytope_vertices.rows() * 3);
            vertices_msg.layout.dim[1].label = "xyz";
            vertices_msg.layout.dim[1].size = 3;
            vertices_msg.layout.dim[1].stride = 3;
            vertices_msg.data.reserve(
                static_cast<std::size_t>(force_polytope_vertices.rows() * 3));
            for (int vertex_index = 0; vertex_index < force_polytope_vertices.rows();
                 ++vertex_index)
            {
                vertices_msg.data.push_back(force_polytope_vertices(vertex_index, 0));
                vertices_msg.data.push_back(force_polytope_vertices(vertex_index, 1));
                vertices_msg.data.push_back(force_polytope_vertices(vertex_index, 2));
            }
            force_polytope_vertices_pub_.publish(vertices_msg);
        }

        geometry_msgs::Vector3Stamped desired_force_msg;
        desired_force_msg.header.stamp = ros::Time::now();
        desired_force_msg.header.frame_id = world_frame_;
        desired_force_msg.vector.x = desired_force_d_.x();
        desired_force_msg.vector.y = desired_force_d_.y();
        desired_force_msg.vector.z = desired_force_d_.z();
        desired_force_pub_.publish(desired_force_msg);

        std_msgs::Float64 nullspace_task_acceleration_norm_msg;
        nullspace_task_acceleration_norm_msg.data = nullspace_task_acceleration_norm;
        nullspace_task_acceleration_norm_pub_.publish(
            nullspace_task_acceleration_norm_msg);

        visualization_msgs::Marker nullspace_base_arrow_msg;
        nullspace_base_arrow_msg.header.stamp = ros::Time::now();
        nullspace_base_arrow_msg.header.frame_id = world_frame_;
        nullspace_base_arrow_msg.ns = "nullspace";
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

        const Eigen::Vector3d nullspace_base_vector_in_world_frame(
            moca_tau_nullspace_wu(0),
            moca_tau_nullspace_wu(1),
            0.0);
// moca_tau_nullspace_wu
        geometry_msgs::Point arrow_end;
        arrow_end.x =
            arrow_start.x +
            nullspace_base_arrow_scale_ * nullspace_base_vector_in_world_frame.x();
        arrow_end.y =
            arrow_start.y +
            nullspace_base_arrow_scale_ * nullspace_base_vector_in_world_frame.y();
        // The third base component is yaw-related, not a spatial z direction.
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
            cmd_arrow_start.x + base_cmd_velocity_arrow_scale_ * output_cmd_vel.x();
        cmd_arrow_end.y =
            cmd_arrow_start.y + base_cmd_velocity_arrow_scale_ * output_cmd_vel.y();
        cmd_arrow_end.z = cmd_arrow_start.z;

        base_cmd_velocity_arrow_msg.points.push_back(cmd_arrow_start);
        base_cmd_velocity_arrow_msg.points.push_back(cmd_arrow_end);
        base_cmd_velocity_arrow_pub_.publish(base_cmd_velocity_arrow_msg);



        if (moca_position_publisher_.trylock())
        {
            moca_position_publisher_.msg_.header.stamp = ros::Time::now();
            moca_position_publisher_.msg_.header.frame_id = world_frame_;
            moca_position_publisher_.msg_.pose.position.x = moca_position(0);
            moca_position_publisher_.msg_.pose.position.y = moca_position(1);
            moca_position_publisher_.msg_.pose.position.z = moca_position(2);
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
            base_command_velocity_publisher_.msg_.twist.linear.x = output_cmd_vel(0);
            base_command_velocity_publisher_.msg_.twist.linear.y = output_cmd_vel(1);
            base_command_velocity_publisher_.msg_.twist.linear.z = 0.0;
            base_command_velocity_publisher_.msg_.twist.angular.x = 0.0;
            base_command_velocity_publisher_.msg_.twist.angular.y = 0.0;
            base_command_velocity_publisher_.msg_.twist.angular.z = output_cmd_vel(2);
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
            nullspace_task_acceleration_publisher_.msg_.twist.linear.x =
                nullspace_task_acceleration(0);
            nullspace_task_acceleration_publisher_.msg_.twist.linear.y =
                nullspace_task_acceleration(1);
            nullspace_task_acceleration_publisher_.msg_.twist.linear.z =
                nullspace_task_acceleration(2);
            nullspace_task_acceleration_publisher_.msg_.twist.angular.x =
                nullspace_task_acceleration(3);
            nullspace_task_acceleration_publisher_.msg_.twist.angular.y =
                nullspace_task_acceleration(4);
            nullspace_task_acceleration_publisher_.msg_.twist.angular.z =
                nullspace_task_acceleration(5);
            nullspace_task_acceleration_publisher_.unlockAndPublish();
        }



        if (moca_joint_state_publisher_.trylock())
        {
            moca_joint_state_publisher_.msg_.header.stamp = ros::Time::now();
            for (int i = 0; i < kWholeBodyDim; ++i)
            {
                moca_joint_state_publisher_.msg_.position[i] = moca_q_(i);
                moca_joint_state_publisher_.msg_.velocity[i] = moca_dq_(i);
                moca_joint_state_publisher_.msg_.effort[i] = moca_tau_d(i);
            }
            moca_joint_state_publisher_.unlockAndPublish();
        }

        if (arm_task_torque_publisher_.trylock())
        {
            arm_task_torque_publisher_.msg_.header.stamp = ros::Time::now();
            for (int i = 0; i < kRobotArmNum; ++i)
            {
                arm_task_torque_publisher_.msg_.effort[i] = moca_tau_task(3 + i);
            }
            arm_task_torque_publisher_.unlockAndPublish();
        }

        if (arm_nullspace_torque_publisher_.trylock())
        {
            arm_nullspace_torque_publisher_.msg_.header.stamp = ros::Time::now();
            for (int i = 0; i < kRobotArmNum; ++i)
            {
                arm_nullspace_torque_publisher_.msg_.effort[i] =
                    moca_tau_nullspace(3 + i);
            }
            arm_nullspace_torque_publisher_.unlockAndPublish();
        }

        if (moca_target_state_publisher_.trylock())
        {
            moca_target_state_publisher_.msg_.header.stamp = ros::Time::now();
            for (int i = 0; i < kWholeBodyDim; ++i)
            {
                moca_target_state_publisher_.msg_.position[i] = moca_q_d_nullspace_(i);
                moca_target_state_publisher_.msg_.velocity[i] = 0.0;
                moca_target_state_publisher_.msg_.effort[i] = 0.0;
            }
            moca_target_state_publisher_.unlockAndPublish();
        }

        const ros::Time tf_stamp = ros::Time::now();
        const Eigen::Affine3d base_T_ee = w_T_mobile_base_.inverse() * w_T_ee_;

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
            tf::StampedTransform(
                world_T_base,
                tf_stamp,
                world_frame_,
                debug_current_base_frame_));

        const Eigen::Quaterniond base_q_ee(base_T_ee.linear());
        tf::Transform base_T_ee_tf;
        base_T_ee_tf.setOrigin(
            tf::Vector3(
                base_T_ee.translation().x(),
                base_T_ee.translation().y(),
                base_T_ee.translation().z()));
        base_T_ee_tf.setRotation(
            tf::Quaternion(
                base_q_ee.x(),
                base_q_ee.y(),
                base_q_ee.z(),
                base_q_ee.w()));
        tf_broadcaster_.sendTransform(
            tf::StampedTransform(
                base_T_ee_tf,
                tf_stamp,
                debug_current_base_frame_,
                debug_current_ee_frame_));

        const Eigen::Quaterniond world_q_target(moca_rotation_d_);
        tf::Transform world_T_target;
        world_T_target.setOrigin(
            tf::Vector3(
                moca_position_d_(0),
                moca_position_d_(1),
                moca_position_d_(2)));
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

        if (moca_q_d_nullspace_.allFinite())
        {
            const Eigen::Affine3d planned_w_T_mobile_base =
                computeWorldToBaseTransform(moca_q_d_nullspace_.head<3>());
            Eigen::Affine3d planned_w_T_ee = Eigen::Affine3d::Identity();
            computeWholeBodyJacobianInWorld(
                moca_q_d_nullspace_.head<3>(),
                moca_q_d_nullspace_.tail<kRobotArmNum>(),
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
        }
    }

    const Eigen::Matrix<double, 7, 1> arm_tau_d = moca_tau_d.tail<7>();
    const Eigen::Vector3d interface_cmd_vel =
        useHRIIBackend()
            ? output_cmd_vel
            : mobile_base_orientation.toRotationMatrix().transpose() * output_cmd_vel;
    sendCommand(interface_cmd_vel, arm_tau_d);
    previous_arm_tau_command_ = arm_tau_d;

    if (enable_matlogger_ && logger_)
    {
        logger_->add("tau_desired_saturated", moca_tau_d);
    }

    if (impedance_filtering_)
    {
        cartesian_stiffness_ =
            filter_params_ * cartesian_stiffness_target_ +
            (1.0 - filter_params_) * cartesian_stiffness_;
        cartesian_damping_ =
            filter_params_ * cartesian_damping_target_ +
            (1.0 - filter_params_) * cartesian_damping_;
        nullspace_joint_stiffness_ =
            filter_params_ * nullspace_joint_stiffness_target_ +
            (1.0 - filter_params_) * nullspace_joint_stiffness_;
        nullspace_joint_damping_ =
            filter_params_ * nullspace_joint_damping_target_ +
            (1.0 - filter_params_) * nullspace_joint_damping_;
    }
    else
    {
        cartesian_stiffness_ = cartesian_stiffness_target_;
        cartesian_damping_ = cartesian_damping_target_;
        nullspace_joint_stiffness_ = nullspace_joint_stiffness_target_;
        nullspace_joint_damping_ = nullspace_joint_damping_target_;
    }

    previous_whole_body_jacobian_ = whole_body_jacobian_in_world_frame;
    has_previous_whole_body_jacobian_ = true;

    if (enable_matlogger_ && logger_)
    {
        logger_->add("tau_d", moca_tau_d);
        logger_->add("tau_bias", moca_bias);
        logger_->add("tau_gravity_arm", arm_gravity);
        logger_->add("tau_task", moca_tau_task);
        logger_->add("tau_joint_limits", arm_joint_limit_torques);
        logger_->add(
            "tau_manip_objective",
            manipulability_gain_ * manipulability_gradient);
        logger_->add("q", moca_q_);
        logger_->add("dq", moca_dq_);
        logger_->add("EE_position", moca_position);
        logger_->add("EE_orientation", moca_orientation_msg.coeffs());
        logger_->add("EE_position_d", moca_position_d_);
        logger_->add("EE_orientation_d", moca_orientation_d_msg.coeffs());
        logger_->add("EE_position_error", moca_error);
        logger_->add("EE_twist_d", moca_twist_d_);
        logger_->add("EE_acceleration_d", moca_acceleration_d_);
        logger_->add("EE_Jdot_dq", whole_body_jacobian_dot_times_dq);
        logger_->add("EE_reference_acceleration", moca_reference_acceleration);
        logger_->add("EE_twist_error", moca_twist_error);
        logger_->add("task_wrench_d", moca_task_wrench);
        logger_->add("task_space_inertia_diag", task_space_inertia.diagonal());
        logger_->add(
            "task_space_translational_mass_eq",
            task_space_inertia.topLeftCorner<3, 3>().trace() / 3.0);
        logger_->add("K", cartesian_stiffness_.diagonal());
        logger_->add("K_nullspace_joint", nullspace_joint_stiffness_.diagonal());
        logger_->add("arm_q_d_nullspace", moca_q_d_nullspace_.tail<7>());
        logger_->add(
            "online_nullspace_gradient_objective",
            online_nullspace_gradient_objective_.tail<7>());
        logger_->add("mp_position", mobile_base_position_);
        logger_->add("mp_orientation", mobile_base_orientation.coeffs());
        logger_->add("mp_cmd_vel", cmd_vel);
        logger_->add("mp_cmd_vel_output", output_cmd_vel);
        logger_->add("mp_cmd_vel_interface", interface_cmd_vel);
        logger_->add("J_w", J_weight_);
        logger_->add("twist", moca_twist);
        logger_->add("w_arm", w_f_);
        logger_->add("w_base", w_r_);
        logger_->add("w_base_yaw", w_r_yaw_);
        logger_->add("mobile_base_yaw", mobile_base_yaw_);
        logger_->add("mp_q_d", moca_q_d_nullspace_.head<3>());
        logger_->add("manipulability_gradient", manipulability_gradient);
        logger_->add("manipulability_measure", manipulability_measure);
        logger_->add("directional_force_capacity", directional_force_capacity);
        logger_->add(
            "tau_second_task",
            nullspace_objective.head<3>());
        logger_->add("tau_nullspace", moca_tau_nullspace.head<3>());
        logger_->add("external_wrench", external_wrench_);
        logger_->add("time", ros::Time::now().toSec());
    }

    return true;
}

Eigen::Matrix<double, 6, 3> 
ControllerManager::computeMobileBaseJacobianInWorld(
    const Eigen::Affine3d& w_T_mobile_base,
    const Eigen::Affine3d& w_T_ee) const
{
    Eigen::Matrix<double, 6, 3> J_base = Eigen::Matrix<double, 6, 3>::Zero();

    // ee position expressed in mobile-base frame
    const Eigen::Affine3d base_T_ee = w_T_mobile_base.inverse() * w_T_ee;
    const Eigen::Vector3d r = base_T_ee.translation();

    // Jacobian expressed in mobile-base frame
    J_base(0, 0) = 1.0;
    J_base(1, 1) = 1.0;
    J_base(0, 2) = -r.y();
    J_base(1, 2) =  r.x();
    J_base(5, 2) = 1.0;

    // rotate to world frame
    const Eigen::Matrix3d R = w_T_mobile_base.rotation();
    Eigen::Matrix<double, 6, 6> Ad = Eigen::Matrix<double, 6, 6>::Zero();
    Ad.topLeftCorner<3, 3>() = R;
    Ad.bottomRightCorner<3, 3>() = R;

    return Ad * J_base;
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
        Eigen::AngleAxisd(mobile_base_state.z(), Eigen::Vector3d::UnitZ()).toRotationMatrix();
    return w_T_mobile_base;
}

Eigen::Matrix<double, 6, 1> ControllerManager::computeCartesianPoseError(
    const Eigen::Vector3d& desired_position,
    const Eigen::Matrix3d& desired_rotation,
    const Eigen::Vector3d& current_position,
    const Eigen::Matrix3d& current_rotation) const
{
    Eigen::Matrix<double, 6, 1> error = Eigen::Matrix<double, 6, 1>::Zero();
    error.head<3>() = desired_position - current_position;


    // 检查这个旋转误差的计算是否正确。旋转误差通常可以通过以下方式计算：
    const Eigen::Matrix3d rotation_error =
        desired_rotation * current_rotation.transpose();
    const double cos_theta = std::max(
        -1.0,
        std::min(1.0, 0.5 * (rotation_error.trace() - 1.0)));
    const double theta = std::acos(cos_theta);

    if (theta < 1e-9)
    {
        error.tail<3>().setZero();
        return error;
    }

    Eigen::Vector3d axis_times_two_sin_theta;
    axis_times_two_sin_theta <<
        rotation_error(2, 1) - rotation_error(1, 2),
        rotation_error(0, 2) - rotation_error(2, 0),
        rotation_error(1, 0) - rotation_error(0, 1);

    const double sin_theta = std::sin(theta);
    if (std::abs(sin_theta) > 1e-6)
    {
        error.tail<3>() =
            (0.5 * theta / sin_theta) * axis_times_two_sin_theta;
    }
    else
    {
        const Eigen::AngleAxisd angle_axis(rotation_error);
        error.tail<3>() = angle_axis.axis() * angle_axis.angle();
    }

    return error;
}

Eigen::Matrix<double, 6, 10> ControllerManager::computeWholeBodyJacobianInWorld(
    const Eigen::Vector3d& mobile_base_state,
    const Eigen::Matrix<double, 7, 1>& arm_q,
    Eigen::Affine3d* w_T_ee) const
{
    const Eigen::Affine3d w_T_mobile_base = computeWorldToBaseTransform(mobile_base_state);
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

Eigen::Affine3d ControllerManager::computeWorldToFrankaFlangeTransform(
    const Eigen::Affine3d& w_T_controller_ee) const
{
    if (controller_ee_frame_is_franka_flange_ || !has_franka_tool_state_)
    {
        return w_T_controller_ee;
    }

    Eigen::Matrix4d flange_T_franka_ee_matrix = Eigen::Matrix4d::Identity();
    for (int col = 0; col < 4; ++col)
    {
        for (int row = 0; row < 4; ++row)
        {
            flange_T_franka_ee_matrix(row, col) =
                latest_franka_tool_state_.F_T_EE[static_cast<std::size_t>(col * 4 + row)];
        }
    }

    if (!flange_T_franka_ee_matrix.allFinite())
    {
        return w_T_controller_ee;
    }

    const Eigen::Affine3d flange_T_franka_ee(flange_T_franka_ee_matrix);
    return w_T_controller_ee * flange_T_franka_ee.inverse();
}

Eigen::Matrix<double, 6, 7> ControllerManager::shiftArmJacobianToWorldPoint(
    const Eigen::Matrix<double, 6, 7>& arm_jacobian_in_world_frame,
    const Eigen::Vector3d& controller_ee_to_point_world) const
{
    Eigen::Matrix<double, 6, 7> shifted_jacobian =
        arm_jacobian_in_world_frame;
    shifted_jacobian.topRows<3>() =
        arm_jacobian_in_world_frame.topRows<3>() -
        skewSymmetric(controller_ee_to_point_world) *
            arm_jacobian_in_world_frame.bottomRows<3>();
    return shifted_jacobian;
}

Eigen::Matrix<double, 7, 1> ControllerManager::computeFrankaToolLoadGravityTorques(
    const Eigen::Affine3d& w_T_controller_ee,
    const Eigen::Matrix<double, 6, 7>& arm_jacobian_in_world_frame) const
{
    Eigen::Matrix<double, 7, 1> torques =
        Eigen::Matrix<double, 7, 1>::Zero();
    if (!has_franka_tool_state_ ||
        !std::isfinite(latest_franka_tool_state_.m_total) ||
        latest_franka_tool_state_.m_total <= 0.0)
    {
        return torques;
    }

    Eigen::Vector3d flange_to_com_in_flange;
    for (int i = 0; i < 3; ++i)
    {
        flange_to_com_in_flange(i) =
            latest_franka_tool_state_.F_x_Ctotal[static_cast<std::size_t>(i)];
    }
    if (!flange_to_com_in_flange.allFinite())
    {
        return torques;
    }

    const Eigen::Affine3d w_T_flange =
        computeWorldToFrankaFlangeTransform(w_T_controller_ee);
    const Eigen::Vector3d w_p_com =
        w_T_flange.translation() + w_T_flange.linear() * flange_to_com_in_flange;
    const Eigen::Vector3d controller_ee_to_com_world =
        w_p_com - w_T_controller_ee.translation();
    const Eigen::Matrix<double, 6, 7> com_jacobian_in_world =
        shiftArmJacobianToWorldPoint(
            arm_jacobian_in_world_frame,
            controller_ee_to_com_world);

    const Eigen::Vector3d support_force_world(
        0.0,
        0.0,
        latest_franka_tool_state_.m_total * gravity_acceleration_mps2_);
    torques = com_jacobian_in_world.topRows<3>().transpose() *
              support_force_world;
    return torques;
}

double ControllerManager::computeWholeBodyManipulability(
    const Eigen::Vector3d& mobile_base_state,
    const Eigen::Matrix<double, 7, 1>& arm_q) const
{
    const Eigen::Matrix3d gram =
        computeWholeBodyManipulabilityEllipsoid(mobile_base_state, arm_q);
    return std::sqrt(std::max(manipulability_epsilon_, gram.determinant()));
}

Eigen::Matrix3d ControllerManager::computeWholeBodyManipulabilityEllipsoid(
    const Eigen::Vector3d& mobile_base_state,
    const Eigen::Matrix<double, 7, 1>& arm_q) const
{
    const Eigen::Matrix<double, 6, 10> jacobian =
        computeWholeBodyJacobianInWorld(mobile_base_state, arm_q);
    Eigen::Matrix<double, 3, 8> optimization_jacobian =
        Eigen::Matrix<double, 3, 8>::Zero();

    // The base x/y columns are always direct planar translations in world frame.
    // Including them makes the translational manipulability almost constant and
    // hides the useful variation from base yaw + arm posture.
    optimization_jacobian.col(0) = jacobian.topRows<3>().col(2);
    optimization_jacobian.rightCols<7>() = jacobian.topRows<3>().rightCols<7>();

    return optimization_jacobian * optimization_jacobian.transpose() +
        manipulability_regularization_ * Eigen::Matrix3d::Identity();
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

    const Eigen::VectorXd adjusted_tau_limits =
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

Eigen::MatrixXd ControllerManager::computeArmForcePolytopeVertices(
    const Eigen::Matrix<double, 6, 7>& arm_jacobian_in_world_frame,
    const Eigen::Matrix<double, 7, 1>& arm_gravity) const
{
    const Eigen::Matrix<double, 3, 7> translational_jacobian =
        arm_jacobian_in_world_frame.topRows<3>();
    const Eigen::VectorXd adjusted_tau_limits =
        buildGravityAdjustedArmTorqueLimits(arm_gravity);

    try
    {
        polytope_wx::polytope polytope_tool;
        return polytope_tool.compute_polytope_vertices(
            translational_jacobian,
            &adjusted_tau_limits);
    }
    catch (const std::exception& exc)
    {
        ROS_WARN_THROTTLE(
            2.0,
            "ControllerManager: failed to compute force polytope vertices: %s",
            exc.what());
    }
    return Eigen::MatrixXd(0, 3);
}

Eigen::Matrix<double, 10, 1> ControllerManager::computeWholeBodyManipulabilityGradient(
    const Eigen::Vector3d& mobile_base_state,
    const Eigen::Matrix<double, 7, 1>& arm_q) const
{
    Eigen::Matrix<double, 10, 1> gradient =
        Eigen::Matrix<double, 10, 1>::Zero();

    for (int i = 0; i < kWholeBodyDim; ++i)
    {
        Eigen::Matrix<double, 10, 1> q_plus = Eigen::Matrix<double, 10, 1>::Zero();
        Eigen::Matrix<double, 10, 1> q_minus = Eigen::Matrix<double, 10, 1>::Zero();
        q_plus << mobile_base_state, arm_q;
        q_minus << mobile_base_state, arm_q;

        q_plus(i) += manipulability_gradient_step_;
        q_minus(i) -= manipulability_gradient_step_;

        q_plus(2) = std::atan2(std::sin(q_plus(2)), std::cos(q_plus(2)));
        q_minus(2) = std::atan2(std::sin(q_minus(2)), std::cos(q_minus(2)));

        for (int joint_index = 0; joint_index < kRobotArmNum; ++joint_index)
        {
            const int whole_body_index = 3 + joint_index;
            q_plus(whole_body_index) = std::min(
                std::max(q_plus(whole_body_index), arm_joint_lower_limits_(joint_index)),
                arm_joint_upper_limits_(joint_index));
            q_minus(whole_body_index) = std::min(
                std::max(q_minus(whole_body_index), arm_joint_lower_limits_(joint_index)),
                arm_joint_upper_limits_(joint_index));
        }

        const double measure_plus = computeWholeBodyManipulability(
            q_plus.head<3>(),
            q_plus.tail<7>());
        const double measure_minus = computeWholeBodyManipulability(
            q_minus.head<3>(),
            q_minus.tail<7>());

        gradient(i) =
            (std::log(std::max(manipulability_epsilon_, measure_plus)) -
             std::log(std::max(manipulability_epsilon_, measure_minus))) /
            (2.0 * manipulability_gradient_step_);
    }

    return gradient;
}

Eigen::Matrix<double, 7, 1> ControllerManager::saturateTorqueRate(
    const Eigen::Matrix<double, 7, 1> &tau_d_calculated,
    const Eigen::Matrix<double, 7, 1> &tau_J_d)
{
    Eigen::Matrix<double, 7, 1> tau_d_saturated;
    for (int i = 0; i < 7; ++i)
    {
        const double difference = tau_d_calculated[i] - tau_J_d[i];
        tau_d_saturated[i] =
            tau_J_d[i] + std::max(std::min(difference, delta_tau_max_), -delta_tau_max_);
    }
    return tau_d_saturated;
}

void ControllerManager::checkDeadzone(double &value, int wrench_type)
{
    double min_wrench = 0.0;

    switch (wrench_type)
    {
    case LINEAR:
        min_wrench = min_linear_wrench_;
        break;
    case ANGULAR:
        min_wrench = min_angular_wrench_;
        break;
    default:
        return;
    }

    if (std::fabs(value) < min_wrench)
    {
        value = 0.0;
    }
}

Eigen::Vector3d ControllerManager::applyBaseCommandDeadband(
    const Eigen::Vector3d& base_command) const
{
    Eigen::Vector3d filtered_command = base_command;
    const double linear_deadband = std::max(0.0, base_cmd_linear_velocity_deadband_);
    const double angular_deadband = std::max(0.0, base_cmd_angular_velocity_deadband_);

    if (std::abs(filtered_command(0)) < linear_deadband)
    {
        filtered_command(0) = 0.0;
    }
    if (std::abs(filtered_command(1)) < linear_deadband)
    {
        filtered_command(1) = 0.0;
    }
    if (std::abs(filtered_command(2)) < angular_deadband)
    {
        filtered_command(2) = 0.0;
    }

    return filtered_command;
}

Eigen::Matrix<double, 7, 1> ControllerManager::computeArmJointLimitTorques(
    const Eigen::Matrix<double, 7, 1>& arm_q) const
{
    Eigen::Matrix<double, 7, 1> joint_limit_torques =
        Eigen::Matrix<double, 7, 1>::Zero();

    for (int i = 0; i < kRobotArmNum; ++i)
    {
        const double q_min = arm_joint_lower_limits_(i);
        const double q_max = arm_joint_upper_limits_(i);
        const double margin = arm_joint_limit_margins_(i);

        if (!std::isfinite(q_min) || !std::isfinite(q_max) || q_max <= q_min)
        {
            continue;
        }

        const double safe_lower = q_min + margin;
        const double safe_upper = q_max - margin;

        if (arm_q(i) < safe_lower)
        {
            joint_limit_torques(i) = joint_limit_torque_gain_ * (safe_lower - arm_q(i));
        }
        else if (arm_q(i) > safe_upper)
        {
            joint_limit_torques(i) = joint_limit_torque_gain_ * (safe_upper - arm_q(i));
        }
    }

    return joint_limit_torques;
}

void ControllerManager::targetPoseCallback(
    const moca_trajectory_generator::TargetPoseCommandConstPtr& msg)
{
    has_received_target_pose_ = true;
    moca_position_d_target_ << msg->pose.position.x,
                               msg->pose.position.y,
                               msg->pose.position.z;
    moca_twist_d_target_ << msg->twist.linear.x,
                             msg->twist.linear.y,
                             msg->twist.linear.z,
                             msg->twist.angular.x,
                             msg->twist.angular.y,
                             msg->twist.angular.z;
    moca_acceleration_d_target_ << msg->acceleration.linear.x,
                                    msg->acceleration.linear.y,
                                    msg->acceleration.linear.z,
                                    msg->acceleration.angular.x,
                                    msg->acceleration.angular.y,
                                    msg->acceleration.angular.z;
    desired_force_d_target_ << msg->desired_force.x,
                               msg->desired_force.y,
                               msg->desired_force.z;
    if (std::isfinite(msg->finger_command))
    {
        finger_command_ = msg->finger_command;
    }

    // std::cout << "TargetPoseCommand: " << moca_position_d_target_.transpose() << std::endl;
    const Eigen::Quaterniond requested_orientation(
        msg->pose.orientation.w,
        msg->pose.orientation.x,
        msg->pose.orientation.y,
        msg->pose.orientation.z);

    if (requested_orientation.norm() > 1e-6)
    {
        moca_rotation_d_target_ =
            requested_orientation.normalized().toRotationMatrix();
    }

    if (msg->use_nullspace_joint_target)
    {
        Eigen::Matrix<double, 3, 1> base_planar_target =
            Eigen::Matrix<double, 3, 1>::Zero();
        Eigen::Matrix<double, kRobotArmNum, 1> arm_joint_target =
            Eigen::Matrix<double, kRobotArmNum, 1>::Zero();

        for (int i = 0; i < 3; ++i)
        {
            const double base_target = msg->base_planar_positions[i];
            if (!std::isfinite(base_target))
            {
                ROS_WARN_THROTTLE(
                    1.0,
                    "ControllerManager: received non-finite base nullspace target, ignoring this message.");
                return;
            }

            base_planar_target(i) = base_target;
        }
        base_planar_target(2) =
            std::atan2(std::sin(base_planar_target(2)), std::cos(base_planar_target(2)));

        for (int i = 0; i < kRobotArmNum; ++i)
        {
            const double joint_target = msg->arm_joint_positions[i];
            if (!std::isfinite(joint_target))
            {
                ROS_WARN_THROTTLE(
                    1.0,
                    "ControllerManager: received non-finite nullspace joint target, ignoring this message.");
                return;
            }

            arm_joint_target(i) = joint_target;
        }
        // std::cout << "Received nullspace joint target: " << arm_joint_target.transpose() << std::endl;

        moca_q_d_nullspace_.head<3>() = base_planar_target;
        moca_q_d_nullspace_.tail<kRobotArmNum>() = arm_joint_target;
        online_nullspace_gradient_objective_.setZero();
        external_whole_body_nullspace_target_active_ = true;
    }
    else
    {
        external_whole_body_nullspace_target_active_ = false;
    }
}

void ControllerManager::forceCapabilitySettingsCallback(
    const moca_trajectory_generator::ForceCapabilitySettingsConstPtr& msg)
{
    if (msg->arm_torque_limits.size() != kRobotArmNum)
    {
        ROS_WARN_STREAM(
            "ControllerManager: received " << msg->arm_torque_limits.size()
            << " torque limits, expected " << kRobotArmNum << ". Ignoring update.");
        return;
    }

    Eigen::Matrix<double, kRobotArmNum, 1> updated_limits = arm_torque_limits_;
    for (int i = 0; i < kRobotArmNum; ++i)
    {
        const double torque_limit = msg->arm_torque_limits[i];
        if (!std::isfinite(torque_limit) || torque_limit <= 0.0)
        {
            ROS_WARN_STREAM(
                "ControllerManager: received invalid torque limit for joint index "
                << i << ". Ignoring update.");
            return;
        }
        updated_limits(i) = std::abs(torque_limit);
    }

    arm_torque_limits_ = updated_limits;
    ros::param::set(
        kForceCapabilitySettingsParamNs,
        std::vector<double>(
            arm_torque_limits_.data(),
            arm_torque_limits_.data() + arm_torque_limits_.size()));

    if (!initializeOnlineNullspaceOptimizer())
    {
        ROS_ERROR(
            "ControllerManager: failed to reinitialize the controller-side optimizer after a torque-limit update.");
    }

    ROS_INFO_STREAM(
        "ControllerManager: updated arm torque limits to "
        << arm_torque_limits_.transpose());
}

void ControllerManager::resetToHomeCallback(const std_msgs::EmptyConstPtr& msg)
{
    (void)msg;
    display_.info("ControllerManager: received reset-to-home request.");
    cmd_vel.setZero();
    previous_arm_tau_command_.setZero();
    desired_force_d_.setZero();
    desired_force_d_target_.setZero();
    moca_twist_d_.setZero();
    moca_twist_d_target_.setZero();
    moca_acceleration_d_.setZero();
    moca_acceleration_d_target_.setZero();
    mobile_base_vir_torques_.setZero();
    online_nullspace_gradient_objective_.setZero();
    has_received_target_pose_ = false;

    if (w_T_ee_.matrix().allFinite())
    {
        moca_position_d_ = w_T_ee_.translation();
        moca_rotation_d_ = w_T_ee_.linear();
        moca_position_d_target_ = moca_position_d_;
        moca_rotation_d_target_ = moca_rotation_d_;
    }

    if (moca_q_.allFinite())
    {
        moca_q_d_nullspace_ = moca_q_;
    }

    external_whole_body_nullspace_target_active_ = false;

    if (useHRIIBackend())
    {
        previous_arm_tau_command_ = current_arm_gravity_;
        sendCommand(Eigen::Vector3d::Zero(), current_arm_gravity_);
    }
    else
    {
        tcpip_client_.request_reset_to_home();
    }
}

void ControllerManager::mobileBaseInertiaCallback(
    const geometry_msgs::InertiaConstPtr& msg)
{
    mobile_base_inertia_ << msg->ixx, msg->ixy, msg->ixz,
                            msg->ixy, msg->iyy, msg->iyz,
                            msg->ixz, msg->iyz, msg->izz;
}

void ControllerManager::frankaToolStateCallback(
    const hrii_robot_msgs::FrankaToolStateConstPtr& msg)
{
    latest_franka_tool_state_ = *msg;
    has_franka_tool_state_ = true;
    ROS_INFO_STREAM_THROTTLE(
        5.0,
        "ControllerManager: received Franka tool/load state from '"
            << franka_tool_state_topic_
            << "' m_total=" << msg->m_total
            << " kg, m_load=" << msg->m_load
            << " kg.");
}

void ControllerManager::updateMobileBaseAdmittance(const Eigen::Vector3d& virtual_wrench)
{
    const Eigen::Vector3d cmd_acc =
        mobile_base_inertia_.ldlt().solve(
            virtual_wrench - mobile_base_admittance_damping_ * cmd_vel);

    cmd_vel += controller_dt_ * cmd_acc;
    cmd_vel(0) = std::max(-base_max_linear_velocity_, std::min(base_max_linear_velocity_, cmd_vel(0)));
    cmd_vel(1) = std::max(-base_max_linear_velocity_, std::min(base_max_linear_velocity_, cmd_vel(1)));
    cmd_vel(2) = std::max(-base_max_angular_velocity_, std::min(base_max_angular_velocity_, cmd_vel(2)));
}

void ControllerManager::sendCommand(const Eigen::Vector3d& base_command,
                                    const Eigen::Matrix<double, 7, 1>& arm_command)
{
    if (!command_output_enabled_)
    {
        return;
    }

    if (useHRIIBackend())
    {
        geometry_msgs::Twist cmd_vel_msg;
        cmd_vel_msg.linear.x = base_command(0);
        cmd_vel_msg.linear.y = base_command(1);
        cmd_vel_msg.linear.z = 0.0;
        cmd_vel_msg.angular.x = 0.0;
        cmd_vel_msg.angular.y = 0.0;
        cmd_vel_msg.angular.z = base_command(2);

        if (mobile_base_interface_)
        {
            mobile_base_interface_->setCmdVel(cmd_vel_msg);
        }

        if (arm_interface_)
        {
            Eigen::Matrix<double, 7, 1> interface_arm_command = arm_command;
            if (hrii_interface_adds_gravity_compensation_ &&
                toLowerCopy(arm_transmission_type_) == "effort")
            {
                interface_arm_command -= current_arm_gravity_;
            }
            arm_interface_->setJointCommands(interface_arm_command);
        }
        return;
    }

    std::vector<double> mobile_cmd(3, 0.0);
    std::vector<double> arm_cmd(7, 0.0);

    for (int i = 0; i < 3; ++i)
    {
        mobile_cmd[i] = base_command(i);
    }
    for (int i = 0; i < 7; ++i)
    {
        arm_cmd[i] = arm_command(i);
    }

    tcpip_client_.set_command(
        mobile_cmd,
        arm_cmd,
        finger_command_,
        polytope_wx::ArmCommandInterface::TORQUE,
        polytope_wx::MobileCommandInterface::VELOCITY);
}

ControllerManager::~ControllerManager()
{
    if (useHRIIBackend())
    {
        sendCommand(Eigen::Vector3d::Zero(), Eigen::Matrix<double, 7, 1>::Zero());
    }
    else
    {
        tcpip_client_.request_reset_to_home();
        ros::Duration(0.05).sleep();
    }
    nh_.deleteParam("controller_started");
}

} // namespace WBCartesianImpedanceController
