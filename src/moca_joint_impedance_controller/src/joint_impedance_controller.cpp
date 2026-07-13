#include "moca_joint_impedance_controller/joint_impedance_controller.h"

#include "polytope_ros/polytope.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <vector>

#include <ros/package.h>

namespace
{

constexpr int kRobotArmNum = 7;
constexpr int kBaseDim = 3;
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

constexpr const char* kForceCapabilitySettingsParamNs =
    "/force_capability_settings/arm_torque_limits";

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

double normalizeYaw(const double yaw)
{
    return std::atan2(std::sin(yaw), std::cos(yaw));
}

double yawFromQuaternion(const geometry_msgs::Quaternion& quaternion)
{
    return std::atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z));
}

Eigen::Vector3d loadVector3Param(
    const ros::NodeHandle& nh,
    const std::string& param_name,
    const Eigen::Vector3d& default_value)
{
    std::vector<double> values;
    if (!nh.getParam(param_name, values) || values.size() != 3)
    {
        return default_value;
    }

    Eigen::Vector3d result;
    for (int i = 0; i < 3; ++i)
    {
        result(i) = std::isfinite(values[static_cast<std::size_t>(i)])
            ? values[static_cast<std::size_t>(i)]
            : default_value(i);
    }
    return result;
}

Eigen::Matrix<double, kRobotArmNum, 1> loadVector7Param(
    const ros::NodeHandle& nh,
    const std::string& param_name,
    const Eigen::Matrix<double, kRobotArmNum, 1>& default_value)
{
    std::vector<double> values;
    if (!nh.getParam(param_name, values) || values.size() != kRobotArmNum)
    {
        return default_value;
    }

    Eigen::Matrix<double, kRobotArmNum, 1> result;
    for (int i = 0; i < kRobotArmNum; ++i)
    {
        result(i) = std::isfinite(values[static_cast<std::size_t>(i)])
            ? values[static_cast<std::size_t>(i)]
            : default_value(i);
    }
    return result;
}

bool readEigenVectorHead(
    const Eigen::VectorXd& source,
    Eigen::Matrix<double, kRobotArmNum, 1>* destination)
{
    if (destination == nullptr || source.size() < kRobotArmNum)
    {
        return false;
    }

    *destination = source.head<kRobotArmNum>();
    return destination->allFinite();
}

} // namespace

namespace moca_joint_impedance_controller
{

namespace
{

bool isFinitePoint(const geometry_msgs::Point& point)
{
    return std::isfinite(point.x) &&
           std::isfinite(point.y) &&
           std::isfinite(point.z);
}

bool hasFiniteMarkerPoints(const visualization_msgs::Marker& marker)
{
    for (const geometry_msgs::Point& point : marker.points)
    {
        if (!isFinitePoint(point))
        {
            return false;
        }
    }
    return true;
}

bool isFiniteQuaternion(const Eigen::Quaterniond& quaternion)
{
    return std::isfinite(quaternion.x()) &&
           std::isfinite(quaternion.y()) &&
           std::isfinite(quaternion.z()) &&
           std::isfinite(quaternion.w()) &&
           quaternion.norm() > 1e-9;
}

} // namespace

bool JointImpedanceController::init(int argc, char** argv)
{
    (void)argc;
    (void)argv;

    nh_ = ros::NodeHandle("~");

    nh_.param("publish_rate_hz", publish_rate_hz_, publish_rate_hz_);
    nh_.param("controller_dt", controller_dt_, controller_dt_);
    nh_.param("interface_type", interface_type_, interface_type_);
    nh_.param(
        "robot_interface_config_file",
        robot_interface_config_file_,
        robot_interface_config_file_);
    nh_.param("arm_transmission_type", arm_transmission_type_, arm_transmission_type_);
    nh_.param(
        "hrii_interface_adds_gravity_compensation",
        hrii_interface_adds_gravity_compensation_,
        hrii_interface_adds_gravity_compensation_);
    nh_.param("command_output_enabled", command_output_enabled_, command_output_enabled_);
    nh_.param(
        "hold_until_first_target_pose",
        hold_until_first_target_pose_,
        hold_until_first_target_pose_);
    nh_.param("track_base_planar_target", track_base_planar_target_, track_base_planar_target_);
    nh_.param("base_cmd_vel_in_base_frame", base_cmd_vel_in_base_frame_, base_cmd_vel_in_base_frame_);
    nh_.param("urdf_path", urdf_path_, urdf_path_);
    nh_.param("base_frame", base_frame_, base_frame_);
    nh_.param("ee_frame", ee_frame_, ee_frame_);
    nh_.param("world_frame", world_frame_, world_frame_);
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
    nh_.param("manipulability_epsilon", manipulability_epsilon_, manipulability_epsilon_);
    nh_.param(
        "manipulability_regularization",
        manipulability_regularization_,
        manipulability_regularization_);
    nh_.param(
        "nullspace_base_arrow_scale",
        nullspace_base_arrow_scale_,
        nullspace_base_arrow_scale_);
    nh_.param(
        "base_cmd_velocity_arrow_scale",
        base_cmd_velocity_arrow_scale_,
        base_cmd_velocity_arrow_scale_);
    nh_.param("delta_tau_max", delta_tau_max_, delta_tau_max_);
    nh_.param(
        "enable_desired_force_torque_feedforward",
        enable_desired_force_torque_feedforward_,
        enable_desired_force_torque_feedforward_);
    nh_.param(
        "desired_force_torque_scale",
        desired_force_torque_scale_,
        desired_force_torque_scale_);
    nh_.param(
        "desired_force_torque_limit",
        desired_force_torque_limit_,
        desired_force_torque_limit_);
    desired_force_torque_scale_ =
        std::max(0.0, desired_force_torque_scale_);
    desired_force_torque_limit_ =
        std::max(0.0, desired_force_torque_limit_);

    constexpr double kFallbackArmJointStiffness = 15.0;
    double damping_ratio = 1.0;
    nh_.param("joint_damping_ratio", damping_ratio, damping_ratio);

    const Eigen::Matrix<double, kRobotArmNum, 1> default_arm_stiffness =
        Eigen::Matrix<double, kRobotArmNum, 1>::Constant(
            std::max(0.0, kFallbackArmJointStiffness));
    arm_joint_stiffness_ =
        loadVector7Param(nh_, "arm_joint_stiffness", default_arm_stiffness);
    for (int i = 0; i < kRobotArmNum; ++i)
    {
        arm_joint_stiffness_(i) = std::max(0.0, arm_joint_stiffness_(i));
    }

    Eigen::Matrix<double, kRobotArmNum, 1> default_arm_damping;
    for (int i = 0; i < kRobotArmNum; ++i)
    {
        default_arm_damping(i) =
            2.0 * std::max(0.0, damping_ratio) *
            std::sqrt(std::max(0.0, arm_joint_stiffness_(i)));
    }
    arm_joint_damping_ =
        loadVector7Param(nh_, "arm_joint_damping", default_arm_damping);

    base_planar_stiffness_ = loadVector3Param(
        nh_,
        "base_planar_stiffness",
        Eigen::Vector3d(10.0, 10.0, 5.0));
    for (int i = 0; i < kBaseDim; ++i)
    {
        base_planar_stiffness_(i) = std::max(0.0, base_planar_stiffness_(i));
    }

    Eigen::Vector3d default_base_damping;
    for (int i = 0; i < kBaseDim; ++i)
    {
        default_base_damping(i) =
            2.0 * std::max(0.0, damping_ratio) *
            std::sqrt(std::max(0.0, base_planar_stiffness_(i)));
    }
    base_planar_damping_ =
        loadVector3Param(nh_, "base_planar_damping", default_base_damping);

    const Eigen::Vector3d mobile_base_inertia_diagonal =
        loadVector3Param(nh_, "mobile_base_inertia_diagonal", Eigen::Vector3d(105.0, 105.0, 20.0));
    const Eigen::Vector3d mobile_base_admittance_damping_diagonal =
        loadVector3Param(
            nh_,
            "mobile_base_admittance_damping_diagonal",
            Eigen::Vector3d(180.0, 180.0, 30.0));
    mobile_base_inertia_ = mobile_base_inertia_diagonal.cwiseMax(1e-6).asDiagonal();
    mobile_base_admittance_damping_ =
        mobile_base_admittance_damping_diagonal.cwiseMax(0.0).asDiagonal();
    directional_force_axis_world_ =
        loadVector3Param(nh_, "directional_force_axis_world", Eigen::Vector3d::UnitZ());
    if (directional_force_axis_world_.norm() < 1e-9)
    {
        directional_force_axis_world_ = Eigen::Vector3d::UnitZ();
    }
    else
    {
        directional_force_axis_world_.normalize();
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

    if (robot_interface_config_file_.empty())
    {
        ROS_ERROR("JointImpedanceController: robot_interface_config_file is empty.");
        return false;
    }
    if (!initHRIIInterfaces())
    {
        return false;
    }

    urdf_path_ = resolveRosPath(urdf_path_);
    moca_dynamic_pin_ = std::make_unique<polytope_wx::Moca_dynamic_pin>();
    if (!moca_dynamic_pin_->initFromUrdf(urdf_path_))
    {
        ROS_ERROR_STREAM(
            "JointImpedanceController: failed to initialize Pinocchio model from "
            << urdf_path_);
        return false;
    }

    target_pose_sub_ = nh_.subscribe(
        "/target_pose",
        1,
        &JointImpedanceController::targetPoseCallback,
        this);
    force_capability_settings_sub_ = nh_.subscribe(
        force_capability_settings_topic_,
        1,
        &JointImpedanceController::forceCapabilitySettingsCallback,
        this);
    reset_to_home_sub_ = nh_.subscribe(
        reset_to_home_topic_,
        1,
        &JointImpedanceController::resetToHomeCallback,
        this);

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
    moca_state_pub_ = nh_.advertise<sensor_msgs::JointState>("moca_state", 1);
    moca_target_state_pub_ = nh_.advertise<sensor_msgs::JointState>("moca_state_target", 1);
    arm_gravity_torque_pub_ =
        nh_.advertise<sensor_msgs::JointState>("arm_gravity_torque", 1);
    arm_coriolis_torque_pub_ =
        nh_.advertise<sensor_msgs::JointState>("arm_coriolis_torque", 1);
    arm_command_torque_pub_ =
        nh_.advertise<sensor_msgs::JointState>("arm_command_torque", 1);
    arm_task_torque_pub_ = nh_.advertise<sensor_msgs::JointState>("arm_task_torque", 1);
    arm_nullspace_torque_pub_ =
        nh_.advertise<sensor_msgs::JointState>("arm_nullspace_torque", 1);
    moca_pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("moca_pose", 1);
    moca_pose_target_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("moca_pose_target", 1);
    nullspace_task_acceleration_pub_ =
        nh_.advertise<geometry_msgs::TwistStamped>("nullspace_task_acceleration", 1);
    moca_velocity_pub_ = nh_.advertise<geometry_msgs::TwistStamped>("moca_velocity", 1);
    base_command_velocity_pub_ =
        nh_.advertise<geometry_msgs::TwistStamped>("base_cmd_velocity", 1);

    nh_.setParam("controller_started", false);

    ROS_INFO_STREAM(
        "JointImpedanceController: initialized. This controller tracks TargetPoseCommand "
        "nullspace joints and desired-force torque feedforward="
        << (enable_desired_force_torque_feedforward_ ? "enabled" : "disabled")
        << ", scale=" << desired_force_torque_scale_
        << ", per-joint limit=" << desired_force_torque_limit_ << " Nm.");
    return true;
}

bool JointImpedanceController::initHRIIInterfaces()
{
    arm_interface_ = HRII::RAInterface::RoboticArmInterface::GenerateRoboticArmInterface(
        robot_interface_config_file_,
        interface_type_,
        arm_transmission_type_);
    if (!arm_interface_ || !arm_interface_->init())
    {
        ROS_ERROR("JointImpedanceController: failed to initialize HRII arm interface.");
        return false;
    }

    mobile_base_interface_ =
        HRII::MORInterface::MobileRobotInterface::GenerateMobileRobotInterface(
            robot_interface_config_file_);
    if (!mobile_base_interface_ || !mobile_base_interface_->init())
    {
        ROS_ERROR("JointImpedanceController: failed to initialize HRII mobile base interface.");
        return false;
    }

    return true;
}

bool JointImpedanceController::readRobotState(
    nav_msgs::Odometry* mobile_base_odom,
    Eigen::Matrix<double, 7, 1>* arm_q,
    Eigen::Matrix<double, 7, 1>* arm_dq)
{
    if (mobile_base_odom == nullptr || arm_q == nullptr || arm_dq == nullptr)
    {
        ROS_ERROR("JointImpedanceController: null output passed to readRobotState.");
        return false;
    }
    if (!mobile_base_interface_ || !arm_interface_)
    {
        ROS_ERROR("JointImpedanceController: HRII interfaces are not initialized.");
        return false;
    }

    if (!mobile_base_interface_->updateRobotState())
    {
        ROS_ERROR("JointImpedanceController: mobile base interface update failed.");
        return false;
    }
    *mobile_base_odom = mobile_base_interface_->getOdometry();

    if (!arm_interface_->updateRobotState(*mobile_base_odom))
    {
        ROS_ERROR("JointImpedanceController: arm interface update failed.");
        return false;
    }

    const sensor_msgs::JointState::Ptr arm_joint_state = arm_interface_->getJointsStates();
    if (!arm_joint_state || arm_joint_state->position.size() < kRobotArmNum)
    {
        ROS_ERROR("JointImpedanceController: arm joint state is missing 7 positions.");
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
            "JointImpedanceController: arm velocity feedback has fewer than 7 entries; using zero velocity.");
    }

    return arm_q->allFinite() && arm_dq->allFinite();
}

bool JointImpedanceController::starting()
{
    Eigen::Matrix<double, 7, 1> arm_q = Eigen::Matrix<double, 7, 1>::Zero();
    Eigen::Matrix<double, 7, 1> arm_dq = Eigen::Matrix<double, 7, 1>::Zero();
    nav_msgs::Odometry mobile_base_odom;

    if (!readRobotState(&mobile_base_odom, &arm_q, &arm_dq))
    {
        return false;
    }

    const double base_yaw = yawFromQuaternion(mobile_base_odom.pose.pose.orientation);
    q_.head<3>() << mobile_base_odom.pose.pose.position.x,
                    mobile_base_odom.pose.pose.position.y,
                    base_yaw;
    q_.tail<kRobotArmNum>() = arm_q;
    dq_.setZero();
    q_target_ = q_;
    cmd_vel_.setZero();
    w_T_mobile_base_ = poseMsgToAffine(mobile_base_odom.pose.pose);
    w_T_ee_ = arm_interface_->getO_T_EE();
    if (w_T_ee_.matrix().allFinite())
    {
        target_w_T_ee_display_ = w_T_ee_;
        has_display_target_pose_ = true;
    }

    if (!readEigenVectorHead(arm_interface_->getGravityCompensation(), &current_arm_gravity_))
    {
        current_arm_gravity_.setZero();
        ROS_WARN("JointImpedanceController: gravity compensation vector is invalid at startup.");
    }
    previous_arm_tau_command_ = current_arm_gravity_;

    sendCommand(Eigen::Vector3d::Zero(), previous_arm_tau_command_);
    nh_.setParam("controller_started", true);

    ROS_INFO_STREAM(
        "JointImpedanceController: started. Initial nullspace target is current state: "
        << q_target_.transpose());
    return true;
}

bool JointImpedanceController::update()
{
    Eigen::Matrix<double, 7, 1> arm_q = Eigen::Matrix<double, 7, 1>::Zero();
    Eigen::Matrix<double, 7, 1> arm_dq = Eigen::Matrix<double, 7, 1>::Zero();
    nav_msgs::Odometry mobile_base_odom;

    if (!readRobotState(&mobile_base_odom, &arm_q, &arm_dq))
    {
        return false;
    }

    const double base_yaw = yawFromQuaternion(mobile_base_odom.pose.pose.orientation);
    q_.head<3>() << mobile_base_odom.pose.pose.position.x,
                    mobile_base_odom.pose.pose.position.y,
                    base_yaw;
    q_.tail<kRobotArmNum>() = arm_q;
    w_T_mobile_base_ = poseMsgToAffine(mobile_base_odom.pose.pose);

    const Eigen::Quaterniond base_orientation(
        mobile_base_odom.pose.pose.orientation.w,
        mobile_base_odom.pose.pose.orientation.x,
        mobile_base_odom.pose.pose.orientation.y,
        mobile_base_odom.pose.pose.orientation.z);
    const Eigen::Vector3d mobile_base_velocity_in_body_frame(
        mobile_base_odom.twist.twist.linear.x,
        mobile_base_odom.twist.twist.linear.y,
        mobile_base_odom.twist.twist.angular.z);
    dq_.head<3>() =
        base_orientation.toRotationMatrix() * mobile_base_velocity_in_body_frame;
    dq_.tail<kRobotArmNum>() = arm_dq;

    Eigen::Matrix<double, kRobotArmNum, 1> arm_gravity =
        Eigen::Matrix<double, kRobotArmNum, 1>::Zero();
    Eigen::Matrix<double, kRobotArmNum, 1> arm_coriolis =
        Eigen::Matrix<double, kRobotArmNum, 1>::Zero();
    if (!readEigenVectorHead(arm_interface_->getGravityCompensation(), &arm_gravity))
    {
        ROS_WARN_THROTTLE(
            2.0,
            "JointImpedanceController: invalid gravity vector; using zeros this cycle.");
    }
    if (!readEigenVectorHead(arm_interface_->getCoriolisCompensation(), &arm_coriolis))
    {
        ROS_WARN_THROTTLE(
            2.0,
            "JointImpedanceController: invalid Coriolis vector; using zeros this cycle.");
    }
    current_arm_gravity_ = arm_gravity;
    w_T_ee_ = arm_interface_->getO_T_EE();

    if (hold_until_first_target_pose_ && !has_received_nullspace_target_)
    {
        cmd_vel_.setZero();
        desired_force_target_.setZero();
        ROS_WARN_THROTTLE(
            2.0,
            "JointImpedanceController: holding startup joint target until the first /target_pose nullspace target.");
    }

    Eigen::Matrix<double, kWholeBodyDim, 1> q_error = q_target_ - q_;
    q_error(2) = normalizeYaw(q_error(2));

    Eigen::Vector3d base_virtual_wrench =
        base_planar_stiffness_.cwiseProduct(q_error.head<3>()) -
        base_planar_damping_.cwiseProduct(dq_.head<3>());
    if (!track_base_planar_target_)
    {
        base_virtual_wrench.setZero();
        cmd_vel_.setZero();
    }
    updateBaseAdmittance(base_virtual_wrench);

    Eigen::Matrix<double, kRobotArmNum, 1> arm_impedance_tau =
        arm_joint_stiffness_.cwiseProduct(q_error.tail<kRobotArmNum>()) -
        arm_joint_damping_.cwiseProduct(arm_dq);

    Eigen::Matrix<double, 6, 7> arm_jacobian_in_world_frame =
        Eigen::Matrix<double, 6, 7>::Zero();
    bool has_valid_arm_jacobian = false;
    const Eigen::MatrixXd arm_jacobian_raw = arm_interface_->getJacobian();
    if (arm_jacobian_raw.rows() >= 6 && arm_jacobian_raw.cols() >= kRobotArmNum)
    {
        arm_jacobian_in_world_frame =
            arm_jacobian_raw.block(0, arm_jacobian_raw.cols() - kRobotArmNum, 6, kRobotArmNum);
        has_valid_arm_jacobian = arm_jacobian_in_world_frame.allFinite();
    }
    if (!has_valid_arm_jacobian && moca_dynamic_pin_)
    {
        try
        {
            arm_jacobian_in_world_frame =
                computeWholeBodyJacobianInWorld(q_.head<3>(), arm_q).rightCols<kRobotArmNum>();
            has_valid_arm_jacobian = arm_jacobian_in_world_frame.allFinite();
        }
        catch (const std::exception& exc)
        {
            ROS_WARN_THROTTLE(
                2.0,
                "JointImpedanceController: failed to compute fallback arm Jacobian: %s",
                exc.what());
        }
    }

    Eigen::Matrix<double, kRobotArmNum, 1> arm_desired_force_tau =
        Eigen::Matrix<double, kRobotArmNum, 1>::Zero();
    if (enable_desired_force_torque_feedforward_ && has_valid_arm_jacobian)
    {
        arm_desired_force_tau =
            desired_force_torque_scale_ *
            arm_jacobian_in_world_frame.topRows<3>().transpose() *
            desired_force_target_;
        if (!arm_desired_force_tau.allFinite())
        {
            arm_desired_force_tau.setZero();
            ROS_WARN_THROTTLE(
                2.0,
                "JointImpedanceController: desired-force torque feedforward is non-finite; using zero.");
        }
        else if (desired_force_torque_limit_ > 0.0)
        {
            for (int i = 0; i < kRobotArmNum; ++i)
            {
                arm_desired_force_tau(i) =
                    std::max(
                        -desired_force_torque_limit_,
                        std::min(desired_force_torque_limit_, arm_desired_force_tau(i)));
            }
        }
    }

    Eigen::Matrix<double, kRobotArmNum, 1> arm_tau_command =
        arm_gravity + arm_coriolis + arm_impedance_tau + arm_desired_force_tau;

    arm_tau_command =
        saturateTorqueRate(arm_tau_command, previous_arm_tau_command_);
    for (int i = 0; i < kRobotArmNum; ++i)
    {
        const double torque_limit = std::max(1.0, std::abs(arm_torque_limits_(i)));
        arm_tau_command(i) =
            std::max(-torque_limit, std::min(torque_limit, arm_tau_command(i)));
    }

    const Eigen::Vector3d base_command_world = applyBaseCommandDeadband(cmd_vel_);
    const Eigen::Vector3d base_command =
        base_cmd_vel_in_base_frame_
            ? w_T_mobile_base_.linear().transpose() * base_command_world
            : base_command_world;

    double manipulability_measure = 0.0;
    Eigen::MatrixXd force_polytope_vertices;
    double directional_force_capacity = 0.0;
    try
    {
        manipulability_measure = computeWholeBodyManipulability(q_.head<3>(), arm_q);
        if (has_valid_arm_jacobian)
        {
            directional_force_capacity =
                computeDirectionalForceCapability(arm_jacobian_in_world_frame, arm_gravity);
            force_polytope_vertices =
                computeArmForcePolytopeVertices(arm_jacobian_in_world_frame, arm_gravity);
        }
    }
    catch (const std::exception& exc)
    {
        ROS_WARN_THROTTLE(
            2.0,
            "JointImpedanceController: failed to compute visualization metrics: %s",
            exc.what());
        manipulability_measure = 0.0;
        directional_force_capacity = 0.0;
        force_polytope_vertices.resize(0, 3);
    }

    publishState(
        mobile_base_odom,
        arm_q,
        arm_dq,
        base_virtual_wrench,
        arm_tau_command,
        arm_impedance_tau,
        arm_desired_force_tau,
        arm_gravity,
        arm_coriolis,
        manipulability_measure,
        directional_force_capacity,
        force_polytope_vertices);
    sendCommand(base_command, arm_tau_command);
    previous_arm_tau_command_ = arm_tau_command;

    return true;
}

void JointImpedanceController::targetPoseCallback(
    const moca_trajectory_generator::TargetPoseCommandConstPtr& msg)
{
    if (!msg->use_nullspace_joint_target)
    {
        ROS_WARN_THROTTLE(
            2.0,
            "JointImpedanceController: /target_pose has no nullspace joint target; Cartesian fields are intentionally ignored.");
        return;
    }

    Eigen::Matrix<double, kWholeBodyDim, 1> requested_target =
        Eigen::Matrix<double, kWholeBodyDim, 1>::Zero();

    for (int i = 0; i < kBaseDim; ++i)
    {
        const double value = msg->base_planar_positions[static_cast<std::size_t>(i)];
        if (!std::isfinite(value))
        {
            ROS_WARN("JointImpedanceController: received non-finite base nullspace target.");
            return;
        }
        requested_target(i) = value;
    }
    requested_target(2) = normalizeYaw(requested_target(2));

    for (int i = 0; i < kRobotArmNum; ++i)
    {
        const double value = msg->arm_joint_positions[static_cast<std::size_t>(i)];
        if (!std::isfinite(value))
        {
            ROS_WARN("JointImpedanceController: received non-finite arm nullspace target.");
            return;
        }
        requested_target(3 + i) = value;
    }

    q_target_ = requested_target;
    Eigen::Vector3d requested_desired_force(
        msg->desired_force.x,
        msg->desired_force.y,
        msg->desired_force.z);
    if (!requested_desired_force.allFinite())
    {
        ROS_WARN("JointImpedanceController: received non-finite desired force; using zero.");
        requested_desired_force.setZero();
    }
    desired_force_target_ = requested_desired_force;
    desired_force_display_ = requested_desired_force;
    const Eigen::Affine3d requested_w_T_ee = poseMsgToAffine(msg->pose);
    if (requested_w_T_ee.matrix().allFinite())
    {
        target_w_T_ee_display_ = requested_w_T_ee;
        has_display_target_pose_ = true;
    }
    has_received_nullspace_target_ = true;
}

void JointImpedanceController::forceCapabilitySettingsCallback(
    const moca_trajectory_generator::ForceCapabilitySettingsConstPtr& msg)
{
    if (msg->arm_torque_limits.size() != kRobotArmNum)
    {
        ROS_WARN_STREAM(
            "JointImpedanceController: received " << msg->arm_torque_limits.size()
            << " torque limits, expected " << kRobotArmNum << ". Ignoring update.");
        return;
    }

    Eigen::Matrix<double, kRobotArmNum, 1> updated_limits = arm_torque_limits_;
    for (int i = 0; i < kRobotArmNum; ++i)
    {
        const double torque_limit = msg->arm_torque_limits[static_cast<std::size_t>(i)];
        if (!std::isfinite(torque_limit) || torque_limit <= 0.0)
        {
            ROS_WARN_STREAM(
                "JointImpedanceController: invalid torque limit at joint index "
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

    ROS_INFO_STREAM(
        "JointImpedanceController: updated arm torque limits to "
        << arm_torque_limits_.transpose());
}

void JointImpedanceController::resetToHomeCallback(const std_msgs::EmptyConstPtr& msg)
{
    (void)msg;
    q_target_ = q_;
    cmd_vel_.setZero();
    desired_force_target_.setZero();
    desired_force_display_.setZero();
    has_received_nullspace_target_ = false;
    previous_arm_tau_command_ = current_arm_gravity_;
    sendCommand(Eigen::Vector3d::Zero(), current_arm_gravity_);
    ROS_INFO("JointImpedanceController: reset target to current joint state.");
}

void JointImpedanceController::updateBaseAdmittance(
    const Eigen::Vector3d& virtual_wrench)
{
    const Eigen::Vector3d cmd_acc =
        mobile_base_inertia_.ldlt().solve(
            virtual_wrench - mobile_base_admittance_damping_ * cmd_vel_);

    cmd_vel_ += controller_dt_ * cmd_acc;
    cmd_vel_(0) =
        std::max(-base_max_linear_velocity_, std::min(base_max_linear_velocity_, cmd_vel_(0)));
    cmd_vel_(1) =
        std::max(-base_max_linear_velocity_, std::min(base_max_linear_velocity_, cmd_vel_(1)));
    cmd_vel_(2) =
        std::max(-base_max_angular_velocity_, std::min(base_max_angular_velocity_, cmd_vel_(2)));
}

Eigen::Vector3d JointImpedanceController::applyBaseCommandDeadband(
    const Eigen::Vector3d& base_command) const
{
    Eigen::Vector3d filtered_command = base_command;
    if (std::abs(filtered_command(0)) < std::max(0.0, base_cmd_linear_velocity_deadband_))
    {
        filtered_command(0) = 0.0;
    }
    if (std::abs(filtered_command(1)) < std::max(0.0, base_cmd_linear_velocity_deadband_))
    {
        filtered_command(1) = 0.0;
    }
    if (std::abs(filtered_command(2)) < std::max(0.0, base_cmd_angular_velocity_deadband_))
    {
        filtered_command(2) = 0.0;
    }
    return filtered_command;
}

Eigen::Matrix<double, 7, 1> JointImpedanceController::saturateTorqueRate(
    const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
    const Eigen::Matrix<double, 7, 1>& tau_J_d) const
{
    Eigen::Matrix<double, 7, 1> tau_d_saturated;
    for (int i = 0; i < kRobotArmNum; ++i)
    {
        const double difference = tau_d_calculated(i) - tau_J_d(i);
        tau_d_saturated(i) =
            tau_J_d(i) +
            std::max(std::min(difference, delta_tau_max_), -delta_tau_max_);
    }
    return tau_d_saturated;
}

void JointImpedanceController::sendCommand(
    const Eigen::Vector3d& base_command,
    const Eigen::Matrix<double, 7, 1>& arm_command)
{
    if (!command_output_enabled_)
    {
        return;
    }

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
        Eigen::Matrix<double, kRobotArmNum, 1> interface_arm_command = arm_command;
        if (hrii_interface_adds_gravity_compensation_ &&
            toLowerCopy(arm_transmission_type_) == "effort")
        {
            interface_arm_command -= current_arm_gravity_;
        }
        arm_interface_->setJointCommands(interface_arm_command);
    }
}

Eigen::Affine3d JointImpedanceController::computeWorldToBaseTransform(
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

Eigen::Matrix<double, 6, 3> JointImpedanceController::computeMobileBaseJacobianInWorld(
    const Eigen::Affine3d& w_T_mobile_base,
    const Eigen::Affine3d& w_T_ee) const
{
    Eigen::Matrix<double, 6, 3> J_base = Eigen::Matrix<double, 6, 3>::Zero();
    const Eigen::Affine3d base_T_ee = w_T_mobile_base.inverse() * w_T_ee;
    const Eigen::Vector3d r = base_T_ee.translation();

    J_base(0, 0) = 1.0;
    J_base(1, 1) = 1.0;
    J_base(0, 2) = -r.y();
    J_base(1, 2) = r.x();
    J_base(5, 2) = 1.0;

    Eigen::Matrix<double, 6, 6> rotation_adjoint =
        Eigen::Matrix<double, 6, 6>::Zero();
    rotation_adjoint.topLeftCorner<3, 3>() = w_T_mobile_base.rotation();
    rotation_adjoint.bottomRightCorner<3, 3>() = w_T_mobile_base.rotation();
    return rotation_adjoint * J_base;
}

Eigen::Matrix<double, 6, 10> JointImpedanceController::computeWholeBodyJacobianInWorld(
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

    Eigen::Matrix<double, 6, 10> whole_body_jacobian_in_world_frame =
        Eigen::Matrix<double, 6, 10>::Zero();
    whole_body_jacobian_in_world_frame.leftCols<3>() =
        computeMobileBaseJacobianInWorld(w_T_mobile_base, current_w_T_ee);
    whole_body_jacobian_in_world_frame.rightCols<7>() =
        arm_jacobian_in_world_frame;
    return whole_body_jacobian_in_world_frame;
}

double JointImpedanceController::computeWholeBodyManipulability(
    const Eigen::Vector3d& mobile_base_state,
    const Eigen::Matrix<double, 7, 1>& arm_q) const
{
    const Eigen::Matrix3d gram =
        computeWholeBodyManipulabilityEllipsoid(mobile_base_state, arm_q);
    return std::sqrt(std::max(manipulability_epsilon_, gram.determinant()));
}

Eigen::Matrix3d JointImpedanceController::computeWholeBodyManipulabilityEllipsoid(
    const Eigen::Vector3d& mobile_base_state,
    const Eigen::Matrix<double, 7, 1>& arm_q) const
{
    const Eigen::Matrix<double, 6, 10> jacobian =
        computeWholeBodyJacobianInWorld(mobile_base_state, arm_q);
    Eigen::Matrix<double, 3, 8> optimization_jacobian =
        Eigen::Matrix<double, 3, 8>::Zero();

    optimization_jacobian.col(0) = jacobian.topRows<3>().col(2);
    optimization_jacobian.rightCols<7>() = jacobian.topRows<3>().rightCols<7>();

    return optimization_jacobian * optimization_jacobian.transpose() +
        manipulability_regularization_ * Eigen::Matrix3d::Identity();
}

Eigen::Matrix<double, 14, 1> JointImpedanceController::buildGravityAdjustedArmTorqueLimits(
    const Eigen::Matrix<double, 7, 1>& arm_gravity) const
{
    Eigen::Matrix<double, 14, 1> adjusted_limits =
        Eigen::Matrix<double, 14, 1>::Zero();
    adjusted_limits.head<7>() = arm_torque_limits_ - arm_gravity;
    adjusted_limits.tail<7>() = -arm_torque_limits_ - arm_gravity;
    return adjusted_limits;
}

double JointImpedanceController::computeDirectionalForceCapability(
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

Eigen::MatrixXd JointImpedanceController::computeArmForcePolytopeVertices(
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
            "JointImpedanceController: failed to compute force polytope vertices: %s",
            exc.what());
    }
    return Eigen::MatrixXd(0, 3);
}

Eigen::Affine3d JointImpedanceController::poseMsgToAffine(
    const geometry_msgs::Pose& pose) const
{
    Eigen::Affine3d transform = Eigen::Affine3d::Identity();
    transform.translation() << pose.position.x, pose.position.y, pose.position.z;
    Eigen::Quaterniond orientation(
        pose.orientation.w,
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z);
    if (orientation.norm() > 1e-9)
    {
        transform.linear() = orientation.normalized().toRotationMatrix();
    }
    return transform;
}

bool JointImpedanceController::computePlannedEndEffectorTransform(
    Eigen::Affine3d* w_T_ee) const
{
    if (w_T_ee == nullptr || !moca_dynamic_pin_ || !q_target_.allFinite())
    {
        return false;
    }

    const Eigen::Affine3d planned_w_T_mobile_base =
        computeWorldToBaseTransform(q_target_.head<3>());
    const Eigen::Matrix<double, kRobotArmNum, 1> arm_target =
        q_target_.tail<kRobotArmNum>();
    const Eigen::Affine3d planned_base_T_ee =
        moca_dynamic_pin_->getRelativeTransform(arm_target, base_frame_, ee_frame_);
    *w_T_ee = planned_w_T_mobile_base * planned_base_T_ee;
    return w_T_ee->matrix().allFinite();
}

void JointImpedanceController::publishState(
    const nav_msgs::Odometry& mobile_base_odom,
    const Eigen::Matrix<double, 7, 1>& arm_q,
    const Eigen::Matrix<double, 7, 1>& arm_dq,
    const Eigen::Vector3d& base_virtual_wrench,
    const Eigen::Matrix<double, 7, 1>& arm_tau_command,
    const Eigen::Matrix<double, 7, 1>& arm_impedance_tau,
    const Eigen::Matrix<double, 7, 1>& arm_desired_force_tau,
    const Eigen::Matrix<double, 7, 1>& arm_gravity,
    const Eigen::Matrix<double, 7, 1>& arm_coriolis,
    const double manipulability_measure,
    const double directional_force_capacity,
    const Eigen::MatrixXd& force_polytope_vertices)
{
    const ros::Time now = ros::Time::now();
    const double publish_period = 1.0 / std::max(1, publish_rate_hz_);
    if (!last_publish_time_.isZero() &&
        (now - last_publish_time_).toSec() < publish_period)
    {
        return;
    }
    last_publish_time_ = now;

    sensor_msgs::JointState state_msg;
    state_msg.header.stamp = now;
    state_msg.header.frame_id = world_frame_;
    state_msg.name.assign(kWholeBodyJointNames.begin(), kWholeBodyJointNames.end());
    state_msg.position.resize(kWholeBodyDim, 0.0);
    state_msg.velocity.resize(kWholeBodyDim, 0.0);
    state_msg.effort.resize(kWholeBodyDim, 0.0);
    for (int i = 0; i < kWholeBodyDim; ++i)
    {
        state_msg.position[static_cast<std::size_t>(i)] = q_(i);
        state_msg.velocity[static_cast<std::size_t>(i)] = dq_(i);
    }
    state_msg.effort[0] = base_virtual_wrench(0);
    state_msg.effort[1] = base_virtual_wrench(1);
    state_msg.effort[2] = base_virtual_wrench(2);
    for (int i = 0; i < kRobotArmNum; ++i)
    {
        state_msg.effort[static_cast<std::size_t>(3 + i)] = arm_tau_command(i);
    }
    moca_state_pub_.publish(state_msg);

    sensor_msgs::JointState target_msg;
    target_msg.header = state_msg.header;
    target_msg.name = state_msg.name;
    target_msg.position.resize(kWholeBodyDim, 0.0);
    target_msg.velocity.resize(kWholeBodyDim, 0.0);
    target_msg.effort.resize(kWholeBodyDim, 0.0);
    for (int i = 0; i < kWholeBodyDim; ++i)
    {
        target_msg.position[static_cast<std::size_t>(i)] = q_target_(i);
    }
    target_msg.effort[0] = base_virtual_wrench(0);
    target_msg.effort[1] = base_virtual_wrench(1);
    target_msg.effort[2] = base_virtual_wrench(2);
    for (int i = 0; i < kRobotArmNum; ++i)
    {
        target_msg.effort[static_cast<std::size_t>(3 + i)] = arm_impedance_tau(i);
    }
    moca_target_state_pub_.publish(target_msg);

    sensor_msgs::JointState arm_gravity_msg;
    arm_gravity_msg.header = state_msg.header;
    arm_gravity_msg.name.assign(kArmJointNames.begin(), kArmJointNames.end());
    arm_gravity_msg.effort.resize(kRobotArmNum, 0.0);

    sensor_msgs::JointState arm_coriolis_msg = arm_gravity_msg;
    sensor_msgs::JointState arm_command_msg = arm_gravity_msg;
    for (int i = 0; i < kRobotArmNum; ++i)
    {
        arm_gravity_msg.effort[static_cast<std::size_t>(i)] = arm_gravity(i);
        arm_coriolis_msg.effort[static_cast<std::size_t>(i)] = arm_coriolis(i);
        arm_command_msg.effort[static_cast<std::size_t>(i)] = arm_tau_command(i);
    }
    arm_gravity_torque_pub_.publish(arm_gravity_msg);
    arm_coriolis_torque_pub_.publish(arm_coriolis_msg);
    arm_command_torque_pub_.publish(arm_command_msg);

    std_msgs::Float64 manipulability_msg;
    manipulability_msg.data =
        std::isfinite(manipulability_measure) ? manipulability_measure : 0.0;
    manipulability_pub_.publish(manipulability_msg);

    const Eigen::Matrix3d manipulability_ellipsoid =
        computeWholeBodyManipulabilityEllipsoid(q_.head<3>(), arm_q);
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

    std_msgs::Float64 zero_float_msg;
    zero_float_msg.data = 0.0;
    manipulability_gradient_norm_pub_.publish(zero_float_msg);

    std_msgs::Float64 directional_force_capacity_msg;
    directional_force_capacity_msg.data =
        std::isfinite(directional_force_capacity) ? directional_force_capacity : 0.0;
    directional_force_capacity_pub_.publish(directional_force_capacity_msg);

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
        for (int vertex_index = 0; vertex_index < force_polytope_vertices.rows(); ++vertex_index)
        {
            vertices_msg.data.push_back(force_polytope_vertices(vertex_index, 0));
            vertices_msg.data.push_back(force_polytope_vertices(vertex_index, 1));
            vertices_msg.data.push_back(force_polytope_vertices(vertex_index, 2));
        }
        force_polytope_vertices_pub_.publish(vertices_msg);
    }

    nullspace_task_acceleration_norm_pub_.publish(zero_float_msg);

    geometry_msgs::Vector3Stamped desired_force_msg;
    desired_force_msg.header = state_msg.header;
    desired_force_msg.vector.x = desired_force_display_(0);
    desired_force_msg.vector.y = desired_force_display_(1);
    desired_force_msg.vector.z = desired_force_display_(2);
    desired_force_pub_.publish(desired_force_msg);

    sensor_msgs::JointState arm_task_msg;
    arm_task_msg.header = state_msg.header;
    arm_task_msg.name.assign(kArmJointNames.begin(), kArmJointNames.end());
    arm_task_msg.effort.assign(kRobotArmNum, 0.0);
    for (int i = 0; i < kRobotArmNum; ++i)
    {
        arm_task_msg.effort[static_cast<std::size_t>(i)] = arm_desired_force_tau(i);
    }
    arm_task_torque_pub_.publish(arm_task_msg);

    sensor_msgs::JointState arm_nullspace_msg = arm_task_msg;
    for (int i = 0; i < kRobotArmNum; ++i)
    {
        arm_nullspace_msg.effort[static_cast<std::size_t>(i)] = arm_impedance_tau(i);
    }
    arm_nullspace_torque_pub_.publish(arm_nullspace_msg);

    geometry_msgs::Wrench wrench_msg;
    wrench_msg.force.x = base_virtual_wrench(0);
    wrench_msg.force.y = base_virtual_wrench(1);
    wrench_msg.force.z = 0.0;
    wrench_msg.torque.x = 0.0;
    wrench_msg.torque.y = 0.0;
    wrench_msg.torque.z = base_virtual_wrench(2);
    virtual_torques_pub_.publish(wrench_msg);

    visualization_msgs::Marker nullspace_base_arrow_msg;
    nullspace_base_arrow_msg.header = state_msg.header;
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

    geometry_msgs::Point arrow_end;
    arrow_end.x = arrow_start.x + nullspace_base_arrow_scale_ * base_virtual_wrench(0);
    arrow_end.y = arrow_start.y + nullspace_base_arrow_scale_ * base_virtual_wrench(1);
    arrow_end.z = arrow_start.z;
    nullspace_base_arrow_msg.points.push_back(arrow_start);
    nullspace_base_arrow_msg.points.push_back(arrow_end);
    if (hasFiniteMarkerPoints(nullspace_base_arrow_msg))
    {
        nullspace_base_arrow_pub_.publish(nullspace_base_arrow_msg);
    }
    else
    {
        ROS_WARN_THROTTLE(
            1.0,
            "JointImpedanceController: skipped nullspace_base_arrow marker because it contains non-finite points.");
    }

    visualization_msgs::Marker base_cmd_velocity_arrow_msg;
    base_cmd_velocity_arrow_msg.header = state_msg.header;
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
    cmd_arrow_end.x = cmd_arrow_start.x + base_cmd_velocity_arrow_scale_ * cmd_vel_(0);
    cmd_arrow_end.y = cmd_arrow_start.y + base_cmd_velocity_arrow_scale_ * cmd_vel_(1);
    cmd_arrow_end.z = cmd_arrow_start.z;
    base_cmd_velocity_arrow_msg.points.push_back(cmd_arrow_start);
    base_cmd_velocity_arrow_msg.points.push_back(cmd_arrow_end);
    if (hasFiniteMarkerPoints(base_cmd_velocity_arrow_msg))
    {
        base_cmd_velocity_arrow_pub_.publish(base_cmd_velocity_arrow_msg);
    }
    else
    {
        ROS_WARN_THROTTLE(
            1.0,
            "JointImpedanceController: skipped base_cmd_velocity_arrow marker because it contains non-finite points.");
    }

    geometry_msgs::TwistStamped nullspace_task_acceleration_msg;
    nullspace_task_acceleration_msg.header = state_msg.header;
    nullspace_task_acceleration_pub_.publish(nullspace_task_acceleration_msg);

    geometry_msgs::TwistStamped moca_velocity_msg;
    moca_velocity_msg.header = state_msg.header;
    moca_velocity_pub_.publish(moca_velocity_msg);

    geometry_msgs::TwistStamped base_cmd_msg;
    base_cmd_msg.header = state_msg.header;
    base_cmd_msg.twist.linear.x = cmd_vel_(0);
    base_cmd_msg.twist.linear.y = cmd_vel_(1);
    base_cmd_msg.twist.angular.z = cmd_vel_(2);
    base_command_velocity_pub_.publish(base_cmd_msg);

    (void)arm_q;
    (void)arm_dq;
    if (arm_interface_)
    {
        if (w_T_ee_.matrix().allFinite())
        {
            Eigen::Quaterniond orientation(w_T_ee_.linear());
            orientation.normalize();

            geometry_msgs::PoseStamped pose_msg;
            pose_msg.header = state_msg.header;
            pose_msg.pose.position.x = w_T_ee_.translation().x();
            pose_msg.pose.position.y = w_T_ee_.translation().y();
            pose_msg.pose.position.z = w_T_ee_.translation().z();
            pose_msg.pose.orientation.x = orientation.x();
            pose_msg.pose.orientation.y = orientation.y();
            pose_msg.pose.orientation.z = orientation.z();
            pose_msg.pose.orientation.w = orientation.w();
            moca_pose_pub_.publish(pose_msg);

            const Eigen::Affine3d target_w_T_ee =
                has_display_target_pose_ ? target_w_T_ee_display_ : w_T_ee_;
            Eigen::Quaterniond target_orientation(target_w_T_ee.linear());
            target_orientation.normalize();
            geometry_msgs::PoseStamped target_pose_msg;
            target_pose_msg.header = state_msg.header;
            target_pose_msg.pose.position.x = target_w_T_ee.translation().x();
            target_pose_msg.pose.position.y = target_w_T_ee.translation().y();
            target_pose_msg.pose.position.z = target_w_T_ee.translation().z();
            target_pose_msg.pose.orientation.x = target_orientation.x();
            target_pose_msg.pose.orientation.y = target_orientation.y();
            target_pose_msg.pose.orientation.z = target_orientation.z();
            target_pose_msg.pose.orientation.w = target_orientation.w();
            moca_pose_target_pub_.publish(target_pose_msg);
        }
    }

    const ros::Time tf_stamp = state_msg.header.stamp;
    const Eigen::Affine3d base_T_ee =
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

    const Eigen::Affine3d target_w_T_ee =
        has_display_target_pose_ ? target_w_T_ee_display_ : w_T_ee_;
    const Eigen::Quaterniond world_q_target(target_w_T_ee.linear());
    tf::Transform world_T_target;
    world_T_target.setOrigin(
        tf::Vector3(
            target_w_T_ee.translation().x(),
            target_w_T_ee.translation().y(),
            target_w_T_ee.translation().z()));
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

    const Eigen::Affine3d planned_w_T_mobile_base =
        computeWorldToBaseTransform(q_target_.head<3>());
    const Eigen::Quaterniond world_q_planned_base(planned_w_T_mobile_base.linear());
    if (!planned_w_T_mobile_base.matrix().allFinite() ||
        !isFiniteQuaternion(world_q_planned_base))
    {
        ROS_WARN_THROTTLE(
            1.0,
            "JointImpedanceController: skipped planned base TF because it contains non-finite values.");
        return;
    }

    const Eigen::Quaterniond normalized_world_q_planned_base =
        world_q_planned_base.normalized();
    tf::Transform world_T_planned_base;
    world_T_planned_base.setOrigin(
        tf::Vector3(
            planned_w_T_mobile_base.translation().x(),
            planned_w_T_mobile_base.translation().y(),
            planned_w_T_mobile_base.translation().z()));
    world_T_planned_base.setRotation(
        tf::Quaternion(
            normalized_world_q_planned_base.x(),
            normalized_world_q_planned_base.y(),
            normalized_world_q_planned_base.z(),
            normalized_world_q_planned_base.w()));
    tf_broadcaster_.sendTransform(
        tf::StampedTransform(
            world_T_planned_base,
            tf_stamp,
            world_frame_,
            planned_base_frame_));

    Eigen::Affine3d planned_w_T_ee = Eigen::Affine3d::Identity();
    if (computePlannedEndEffectorTransform(&planned_w_T_ee))
    {
        const Eigen::Affine3d planned_base_T_ee =
            planned_w_T_mobile_base.inverse() * planned_w_T_ee;
        const Eigen::Quaterniond planned_base_q_ee(planned_base_T_ee.linear());
        if (!planned_base_T_ee.matrix().allFinite() ||
            !isFiniteQuaternion(planned_base_q_ee))
        {
            ROS_WARN_THROTTLE(
                1.0,
                "JointImpedanceController: skipped planned ee TF because it contains non-finite values.");
            return;
        }

        const Eigen::Quaterniond normalized_planned_base_q_ee =
            planned_base_q_ee.normalized();
        tf::Transform planned_base_T_ee_tf;
        planned_base_T_ee_tf.setOrigin(
            tf::Vector3(
                planned_base_T_ee.translation().x(),
                planned_base_T_ee.translation().y(),
                planned_base_T_ee.translation().z()));
        planned_base_T_ee_tf.setRotation(
            tf::Quaternion(
                normalized_planned_base_q_ee.x(),
                normalized_planned_base_q_ee.y(),
                normalized_planned_base_q_ee.z(),
                normalized_planned_base_q_ee.w()));
        tf_broadcaster_.sendTransform(
            tf::StampedTransform(
                planned_base_T_ee_tf,
                tf_stamp,
                planned_base_frame_,
                planned_ee_frame_));
    }

    (void)mobile_base_odom;
}

JointImpedanceController::~JointImpedanceController()
{
    sendCommand(Eigen::Vector3d::Zero(), Eigen::Matrix<double, 7, 1>::Zero());
    nh_.deleteParam("controller_started");
}

} // namespace moca_joint_impedance_controller
