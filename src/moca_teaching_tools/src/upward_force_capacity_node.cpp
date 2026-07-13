#include <pinocchio/fwd.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/QR>
#include <robot_dynamic_pin/Moca_dynamic_pin.h>

#include <hrii_robot_msgs/RobotState.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Float64.h>

namespace
{

constexpr int kArmJointCount = 7;
constexpr const char* kForceCapabilitySettingsParamNs =
    "/force_capability_settings/arm_torque_limits";

std::string resolveRosPath(const std::string& path)
{
    const std::string package_prefix = "package://";
    if (path.rfind(package_prefix, 0) == 0)
    {
        const std::string resource = path.substr(package_prefix.size());
        const std::size_t separator = resource.find('/');
        if (separator == std::string::npos)
        {
            return path;
        }
        const std::string package_name = resource.substr(0, separator);
        const std::string package_path = ros::package::getPath(package_name);
        if (package_path.empty())
        {
            return path;
        }
        return package_path + "/" + resource.substr(separator + 1);
    }
    return path;
}

Eigen::VectorXd buildGravityAdjustedArmTorqueLimits(
    const Eigen::Matrix<double, kArmJointCount, 1>& arm_torque_limits,
    const Eigen::Matrix<double, kArmJointCount, 1>& arm_gravity)
{
    Eigen::VectorXd adjusted_limits(2 * kArmJointCount);
    adjusted_limits.head(kArmJointCount) = arm_torque_limits - arm_gravity;
    adjusted_limits.tail(kArmJointCount) = -arm_torque_limits - arm_gravity;
    return adjusted_limits;
}

double computeDirectionalForceCapability(
    const Eigen::Matrix<double, 6, kArmJointCount>& arm_jacobian_in_world_frame,
    const Eigen::Matrix<double, kArmJointCount, 1>& arm_gravity,
    const Eigen::Matrix<double, kArmJointCount, 1>& arm_torque_limits,
    const Eigen::Vector3d& directional_force_axis_world)
{
    const Eigen::Matrix<double, 3, kArmJointCount> translational_jacobian =
        arm_jacobian_in_world_frame.topRows<3>();
    const Eigen::Matrix<double, kArmJointCount, 1> tau_direction =
        translational_jacobian.transpose() * directional_force_axis_world;
    const double tau_direction_norm = tau_direction.norm();
    if (tau_direction_norm < 1e-9)
    {
        return 0.0;
    }

    const Eigen::Matrix<double, kArmJointCount, 1> tau_direction_normalized =
        tau_direction / tau_direction_norm;
    Eigen::Matrix<double, 2 * kArmJointCount, 2 * kArmJointCount> block_matrix =
        Eigen::Matrix<double, 2 * kArmJointCount, 2 * kArmJointCount>::Zero();
    block_matrix.topLeftCorner<kArmJointCount, kArmJointCount>() =
        tau_direction_normalized.asDiagonal();
    block_matrix.bottomRightCorner<kArmJointCount, kArmJointCount>() =
        tau_direction_normalized.asDiagonal();

    const Eigen::VectorXd adjusted_tau_limits =
        buildGravityAdjustedArmTorqueLimits(arm_torque_limits, arm_gravity);
    const Eigen::Matrix<double, 2 * kArmJointCount, 1> directional_limits =
        (block_matrix +
         1e-8 * Eigen::Matrix<double, 2 * kArmJointCount, 2 * kArmJointCount>::Identity())
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

double computeArmTranslationalManipulability(
    const Eigen::Matrix<double, 6, kArmJointCount>& arm_jacobian_in_world_frame,
    const double regularization,
    const double epsilon)
{
    const Eigen::Matrix<double, 3, kArmJointCount> translational_jacobian =
        arm_jacobian_in_world_frame.topRows<3>();
    const Eigen::Matrix3d gram =
        translational_jacobian * translational_jacobian.transpose() +
        regularization * Eigen::Matrix3d::Identity();
    return std::sqrt(std::max(epsilon, gram.determinant()));
}

}  // namespace

class UpwardForceCapacityNode
{
public:
    UpwardForceCapacityNode()
        : pnh_("~")
    {
        std::string urdf_path;
        pnh_.param<std::string>(
            "urdf_path",
            urdf_path,
            "package://polytope_ros/urdf/my_robot_fix_wheels.urdf");
        pnh_.param<std::string>("ee_frame", ee_frame_, "moca_franka_EE");
        pnh_.param<std::string>("state_topic", state_topic_, "/wb_cart_imp_controller/moca_state");
        pnh_.param<std::string>(
            "robot_state_topic",
            robot_state_topic_,
            "/moca_white/franka_state_bridge/robot_state");
        loadRobotStateTopics();
        pnh_.param<std::string>(
            "capacity_topic",
            capacity_topic_,
            "/moca_teaching/upward_force_capacity");
        pnh_.param<std::string>(
            "manipulability_topic",
            manipulability_topic_,
            "/moca_teaching/manipulability");
        pnh_.param<double>(
            "manipulability_regularization",
            manipulability_regularization_,
            manipulability_regularization_);
        pnh_.param<double>(
            "manipulability_epsilon",
            manipulability_epsilon_,
            manipulability_epsilon_);

        directional_force_axis_world_ = Eigen::Vector3d::UnitZ();
        std::vector<double> axis_param;
        if (pnh_.getParam("directional_force_axis_world", axis_param) &&
            axis_param.size() == 3)
        {
            directional_force_axis_world_ <<
                axis_param[0],
                axis_param[1],
                axis_param[2];
            if (directional_force_axis_world_.norm() < 1e-9)
            {
                directional_force_axis_world_ = Eigen::Vector3d::UnitZ();
            }
            else
            {
                directional_force_axis_world_.normalize();
            }
        }

        arm_torque_limits_.setConstant(70.0);
        arm_torque_limits_.tail<3>().setConstant(12.0);
        refreshArmTorqueLimits();

        if (!model_.initFromUrdf(resolveRosPath(urdf_path), false))
        {
            throw std::runtime_error("failed to initialize Moca_dynamic_pin from URDF.");
        }

        capacity_pub_ = nh_.advertise<std_msgs::Float64>(capacity_topic_, 1, true);
        manipulability_pub_ =
            nh_.advertise<std_msgs::Float64>(manipulability_topic_, 1, true);
        state_sub_ = nh_.subscribe(
            state_topic_,
            1,
            &UpwardForceCapacityNode::stateCallback,
            this);
        for (const std::string& topic : robot_state_topics_)
        {
            robot_state_subs_.push_back(
                nh_.subscribe(
                    topic,
                    1,
                    &UpwardForceCapacityNode::robotStateCallback,
                    this));
        }

        std::string robot_state_topics_text;
        for (const std::string& topic : robot_state_topics_)
        {
            robot_state_topics_text += robot_state_topics_text.empty() ? topic : ", " + topic;
        }
        ROS_INFO_STREAM(
            "moca_teaching_tools: upward force capacity node monitoring '"
            << state_topic_ << "'"
            << (robot_state_topics_text.empty() ? "" : " and robot_state topics [" + robot_state_topics_text + "]")
            << ", publishing '" << capacity_topic_ << "' and '"
            << manipulability_topic_ << "'.");
    }

private:
    void loadRobotStateTopics()
    {
        robot_state_topics_.clear();
        std::vector<std::string> topics;
        if (pnh_.getParam("robot_state_topics", topics))
        {
            for (const std::string& topic : topics)
            {
                addRobotStateTopic(topic);
            }
        }
        addRobotStateTopic(robot_state_topic_);
    }

    void addRobotStateTopic(const std::string& topic)
    {
        if (topic.empty())
        {
            return;
        }
        if (std::find(robot_state_topics_.begin(), robot_state_topics_.end(), topic) ==
            robot_state_topics_.end())
        {
            robot_state_topics_.push_back(topic);
        }
    }

    void refreshArmTorqueLimits()
    {
        std::vector<double> limits;
        if (!ros::param::get(kForceCapabilitySettingsParamNs, limits))
        {
            return;
        }
        if (limits.size() != kArmJointCount)
        {
            ROS_WARN_STREAM_THROTTLE(
                2.0,
                "moca_teaching_tools: expected " << kArmJointCount
                << " arm torque limits at " << kForceCapabilitySettingsParamNs
                << ", got " << limits.size() << ".");
            return;
        }
        for (int i = 0; i < kArmJointCount; ++i)
        {
            if (std::isfinite(limits[i]) && limits[i] > 0.0)
            {
                arm_torque_limits_(i) = std::abs(limits[i]);
            }
        }
    }

    void stateCallback(const sensor_msgs::JointStateConstPtr& msg)
    {
        if (msg->position.size() < kArmJointCount)
        {
            ROS_WARN_THROTTLE(
                1.0,
                "moca_teaching_tools: received joint state with %zu positions, expected at least %d.",
                msg->position.size(),
                kArmJointCount);
            return;
        }

        refreshArmTorqueLimits();

        polytope_wx::RobotMessage robot_state{};
        const std::size_t arm_offset =
            msg->position.size() >= kArmJointCount + 3 ? msg->position.size() - kArmJointCount : 0;
        for (int i = 0; i < kArmJointCount; ++i)
        {
            robot_state.arm_positions[i] =
                msg->position[arm_offset + static_cast<std::size_t>(i)];
            if (msg->velocity.size() > arm_offset + static_cast<std::size_t>(i))
            {
                robot_state.arm_velocities[i] =
                    msg->velocity[arm_offset + static_cast<std::size_t>(i)];
            }
        }

        if (!model_.updateFromRobotMessage(robot_state))
        {
            ROS_WARN_THROTTLE(
                1.0,
                "moca_teaching_tools: failed to update dynamics model from joint state.");
            return;
        }

        try
        {
            const Eigen::Matrix<double, 6, kArmJointCount> arm_jacobian =
                model_.getArmJacobian(ee_frame_);
            const Eigen::Matrix<double, kArmJointCount, 1> arm_gravity =
                model_.getArmGravityCompensation();
            const double capacity = computeDirectionalForceCapability(
                arm_jacobian,
                arm_gravity,
                arm_torque_limits_,
                directional_force_axis_world_);
            const double manipulability = computeArmTranslationalManipulability(
                arm_jacobian,
                manipulability_regularization_,
                manipulability_epsilon_);

            std_msgs::Float64 msg_out;
            msg_out.data = std::isfinite(capacity) ? capacity : 0.0;
            capacity_pub_.publish(msg_out);

            std_msgs::Float64 manipulability_msg;
            manipulability_msg.data =
                std::isfinite(manipulability) ? manipulability : 0.0;
            manipulability_pub_.publish(manipulability_msg);
        }
        catch (const std::exception& exc)
        {
            ROS_WARN_THROTTLE(
                1.0,
                "moca_teaching_tools: failed to compute upward force capacity: %s",
                exc.what());
        }
    }

    void robotStateCallback(const hrii_robot_msgs::RobotStateConstPtr& msg)
    {
        const sensor_msgs::JointStateConstPtr joint_state(
            new sensor_msgs::JointState(msg->joints_states));
        stateCallback(joint_state);
    }

    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    ros::Subscriber state_sub_;
    std::vector<ros::Subscriber> robot_state_subs_;
    ros::Publisher capacity_pub_;
    ros::Publisher manipulability_pub_;
    polytope_wx::Moca_dynamic_pin model_;
    Eigen::Matrix<double, kArmJointCount, 1> arm_torque_limits_;
    Eigen::Vector3d directional_force_axis_world_{Eigen::Vector3d::UnitZ()};
    std::string state_topic_;
    std::string robot_state_topic_;
    std::vector<std::string> robot_state_topics_;
    std::string capacity_topic_;
    std::string manipulability_topic_;
    std::string ee_frame_;
    double manipulability_regularization_{1e-6};
    double manipulability_epsilon_{1e-8};
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "moca_teaching_upward_force_capacity");
    try
    {
        UpwardForceCapacityNode node;
        ros::spin();
    }
    catch (const std::exception& exc)
    {
        ROS_FATAL_STREAM("moca_teaching_tools: upward force capacity node failed: " << exc.what());
        return 1;
    }
    return 0;
}
