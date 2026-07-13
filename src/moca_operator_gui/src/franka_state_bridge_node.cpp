#include <array>
#include <algorithm>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <franka/exception.h>
#include <franka/robot.h>
#include <geometry_msgs/Wrench.h>
#include <hrii_robot_msgs/FrankaToolState.h>
#include <hrii_robot_msgs/RobotState.h>
#include <ros/ros.h>

namespace
{

std::vector<std::string> defaultJointNames()
{
    return {
        "moca_franka_joint1",
        "moca_franka_joint2",
        "moca_franka_joint3",
        "moca_franka_joint4",
        "moca_franka_joint5",
        "moca_franka_joint6",
        "moca_franka_joint7",
    };
}

geometry_msgs::Wrench toWrenchMsg(
    const std::array<double, 6>& wrench,
    const double sign)
{
    geometry_msgs::Wrench msg;
    msg.force.x = sign * wrench[0];
    msg.force.y = sign * wrench[1];
    msg.force.z = sign * wrench[2];
    msg.torque.x = sign * wrench[3];
    msg.torque.y = sign * wrench[4];
    msg.torque.z = sign * wrench[5];
    return msg;
}

geometry_msgs::Pose toPoseMsg(const std::array<double, 16>& transform)
{
    using Matrix4dColMajor =
        Eigen::Matrix<double, 4, 4, Eigen::ColMajor>;
    const Eigen::Map<const Matrix4dColMajor> matrix(transform.data());

    Eigen::Quaterniond orientation(matrix.block<3, 3>(0, 0));
    orientation.normalize();

    geometry_msgs::Pose pose;
    pose.position.x = matrix(0, 3);
    pose.position.y = matrix(1, 3);
    pose.position.z = matrix(2, 3);
    pose.orientation.x = orientation.x();
    pose.orientation.y = orientation.y();
    pose.orientation.z = orientation.z();
    pose.orientation.w = orientation.w();
    return pose;
}

hrii_robot_msgs::RobotState toRobotStateMsg(
    const franka::RobotState& state,
    const std::vector<std::string>& joint_names,
    const std::string& frame_id,
    const bool invert_external_wrench)
{
    hrii_robot_msgs::RobotState msg;
    msg.header.stamp = ros::Time::now();
    msg.header.frame_id = frame_id;

    msg.joints_states.header = msg.header;
    msg.joints_states.name = joint_names;
    msg.joints_states.position.assign(state.q.begin(), state.q.end());
    msg.joints_states.velocity.assign(state.dq.begin(), state.dq.end());
    msg.joints_states.effort.assign(state.tau_J.begin(), state.tau_J.end());

    const double sign = invert_external_wrench ? -1.0 : 1.0;
    msg.est_ext_wrench_base_frame =
        toWrenchMsg(state.O_F_ext_hat_K, sign);
    msg.est_ext_wrench_stiffness_frame =
        toWrenchMsg(state.K_F_ext_hat_K, sign);
    msg.est_ext_torques.assign(
        state.tau_ext_hat_filtered.begin(),
        state.tau_ext_hat_filtered.end());
    msg.O_T_EE = toPoseMsg(state.O_T_EE);
    return msg;
}

hrii_robot_msgs::FrankaToolState toFrankaToolStateMsg(
    const franka::RobotState& state,
    const std::string& frame_id)
{
    hrii_robot_msgs::FrankaToolState msg;
    msg.header.stamp = ros::Time::now();
    msg.header.frame_id = frame_id;

    msg.m_ee = state.m_ee;
    std::copy(state.F_x_Cee.begin(), state.F_x_Cee.end(), msg.F_x_Cee.begin());
    std::copy(state.I_ee.begin(), state.I_ee.end(), msg.I_ee.begin());

    msg.m_load = state.m_load;
    std::copy(state.F_x_Cload.begin(), state.F_x_Cload.end(), msg.F_x_Cload.begin());
    std::copy(state.I_load.begin(), state.I_load.end(), msg.I_load.begin());

    msg.m_total = state.m_total;
    std::copy(state.F_x_Ctotal.begin(), state.F_x_Ctotal.end(), msg.F_x_Ctotal.begin());
    std::copy(state.I_total.begin(), state.I_total.end(), msg.I_total.begin());

    std::copy(state.F_T_EE.begin(), state.F_T_EE.end(), msg.F_T_EE.begin());
    std::copy(state.NE_T_EE.begin(), state.NE_T_EE.end(), msg.NE_T_EE.begin());
    std::copy(state.EE_T_K.begin(), state.EE_T_K.end(), msg.EE_T_K.begin());
    return msg;
}

}  // namespace

int main(int argc, char** argv)
{
    ros::init(argc, argv, "franka_state_bridge_node");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");

    std::string robot_arm_ip;
    private_nh.param<std::string>(
        "robot_arm_ip",
        robot_arm_ip,
        "192.168.0.101");
    std::string frame_id;
    private_nh.param<std::string>("frame_id", frame_id, "franka_base");
    double publish_rate_hz = 100.0;
    private_nh.param<double>("publish_rate_hz", publish_rate_hz, publish_rate_hz);
    bool realtime_ignore = true;
    private_nh.param<bool>("realtime_ignore", realtime_ignore, realtime_ignore);
    bool invert_external_wrench = false;
    private_nh.param<bool>(
        "invert_external_wrench",
        invert_external_wrench,
        invert_external_wrench);

    std::vector<std::string> joint_names;
    private_nh.param<std::vector<std::string>>(
        "joint_names",
        joint_names,
        defaultJointNames());
    if (joint_names.size() != 7)
    {
        ROS_WARN_STREAM(
            "franka_state_bridge_node: joint_names has "
            << joint_names.size()
            << " entries; falling back to default 7 Franka joint names.");
        joint_names = defaultJointNames();
    }

    if (robot_arm_ip.empty())
    {
        ROS_ERROR("franka_state_bridge_node: robot_arm_ip is empty.");
        return 1;
    }

    ros::Publisher state_pub =
        private_nh.advertise<hrii_robot_msgs::RobotState>("robot_state", 1);
    ros::Publisher tool_state_pub =
        private_nh.advertise<hrii_robot_msgs::FrankaToolState>(
            "franka_tool_state",
            1,
            true);

    const double publish_period_sec =
        1.0 / std::max(1.0, publish_rate_hz);
    ros::WallTime last_publish_time = ros::WallTime(0.0);

    try
    {
        ROS_INFO_STREAM(
            "franka_state_bridge_node: connecting to Franka at "
            << robot_arm_ip
            << " in read-only mode.");
        franka::Robot robot(
            robot_arm_ip,
            realtime_ignore
                ? franka::RealtimeConfig::kIgnore
                : franka::RealtimeConfig::kEnforce);

        robot.read(
            [&](const franka::RobotState& robot_state) -> bool
            {
                if (!ros::ok())
                {
                    return false;
                }

                const ros::WallTime now = ros::WallTime::now();
                if ((now - last_publish_time).toSec() >= publish_period_sec)
                {
                    last_publish_time = now;
                    state_pub.publish(
                        toRobotStateMsg(
                            robot_state,
                            joint_names,
                            frame_id,
                            invert_external_wrench));
                    tool_state_pub.publish(
                        toFrankaToolStateMsg(
                            robot_state,
                            frame_id));
                }
                return true;
            });
    }
    catch (const franka::Exception& exception)
    {
        ROS_ERROR_STREAM(
            "franka_state_bridge_node: libfranka error: "
            << exception.what());
        return 1;
    }
    catch (const std::exception& exception)
    {
        ROS_ERROR_STREAM(
            "franka_state_bridge_node: error: "
            << exception.what());
        return 1;
    }

    ROS_INFO("franka_state_bridge_node: stopped.");
    return 0;
}
