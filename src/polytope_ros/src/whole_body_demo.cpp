#include <ros/ros.h>

#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <std_msgs/Float64MultiArray.h>

#include "polytope_ros/whole_body_optimizer.h"

namespace
{

constexpr int kBaseDof = 3;
constexpr int kArmDof = 7;

bool directoryExists(const std::string& path)
{
    if (path.empty())
    {
        return false;
    }

    struct stat info;
    if (stat(path.c_str(), &info) != 0)
    {
        return false;
    }

    return S_ISDIR(info.st_mode);
}

bool ensureDirectory(const std::string& path)
{
    if (path.empty() || directoryExists(path))
    {
        return true;
    }

    const std::size_t separator = path.find_last_of('/');
    if (separator != std::string::npos)
    {
        const std::string parent = path.substr(0, separator);
        if (!parent.empty() && !ensureDirectory(parent))
        {
            return false;
        }
    }

    if (::mkdir(path.c_str(), 0755) == 0 || errno == EEXIST)
    {
        return true;
    }

    return false;
}

bool ensureParentDirectoryForFile(const std::string& file_path)
{
    const std::size_t separator = file_path.find_last_of('/');
    if (separator == std::string::npos)
    {
        return true;
    }

    return ensureDirectory(file_path.substr(0, separator));
}

Eigen::Affine3d buildTargetPose(
    const Eigen::Affine3d& seed_pose,
    double time_sec,
    double frequency_hz,
    double amplitude_x,
    double amplitude_y,
    double amplitude_z)
{
    const double omega = 2.0 * M_PI * frequency_hz;

    Eigen::Affine3d target_pose = seed_pose;
    target_pose.translation() <<
        seed_pose.translation().x() + amplitude_x * std::sin(omega * time_sec),
        seed_pose.translation().y() + amplitude_y * (1.0 - std::cos(omega * time_sec)),
        seed_pose.translation().z() + amplitude_z * std::sin(0.5 * omega * time_sec);
    return target_pose;
}

} // namespace

int main(int argc, char* argv[])
{
    ros::init(argc, argv, "whole_body_demo");
    ros::NodeHandle nh("~");

    std::string urdf_path;
    std::string base_frame;
    std::string ee_frame;
    std::string output_csv_path;
    double horizon_sec;
    double trajectory_frequency_hz;
    double desired_force_magnitude;
    double amplitude_x;
    double amplitude_y;
    double amplitude_z;
    int waypoint_count;
    bool constrain_orientation;
    bool publish_replay;
    double replay_startup_delay_sec;
    double optimizer_capability_weight;
    double optimizer_manipulability_weight;
    double optimizer_joint_limit_weight;
    double optimizer_smoothness_weight;
    double optimizer_nominal_weight;
    double optimizer_base_smoothness_weight;
    double optimizer_base_nominal_weight;
    double optimizer_base_spectral_energy_weight;
    double optimizer_base_spectral_cutoff_ratio;
    double optimizer_base_spectral_power;
    double optimizer_base_spectral_metric_x;
    double optimizer_base_spectral_metric_y;
    double optimizer_base_spectral_metric_yaw;
    std::vector<double> initial_arm_joint_positions;

    nh.param<std::string>("urdf_path", urdf_path, "");
    nh.param<std::string>("base_frame", base_frame, "moca_base_footprint");
    nh.param<std::string>("ee_frame", ee_frame, "moca_franka_EE");
    nh.param<std::string>(
        "output_csv_path",
        output_csv_path,
        "/tmp/whole_body_demo.csv");
    nh.param<double>("horizon_sec", horizon_sec, 10.0);
    nh.param<double>("trajectory_frequency_hz", trajectory_frequency_hz, 0.05);
    nh.param<double>("desired_force_magnitude", desired_force_magnitude, 70.0);
    nh.param<double>("amplitude_x", amplitude_x, 0.04);
    nh.param<double>("amplitude_y", amplitude_y, 0.03);
    nh.param<double>("amplitude_z", amplitude_z, 0.0);
    nh.param<int>("waypoint_count", waypoint_count, 41);
    nh.param<bool>("constrain_orientation", constrain_orientation, true);
    nh.param<bool>("publish_replay", publish_replay, false);
    nh.param<double>("replay_startup_delay_sec", replay_startup_delay_sec, 1.0);
    nh.param<double>("optimizer_capability_weight", optimizer_capability_weight, 1.0);
    nh.param<double>("optimizer_manipulability_weight", optimizer_manipulability_weight, 0.0);
    nh.param<double>("optimizer_joint_limit_weight", optimizer_joint_limit_weight, 1e-3);
    nh.param<double>("optimizer_smoothness_weight", optimizer_smoothness_weight, 5e-2);
    nh.param<double>("optimizer_nominal_weight", optimizer_nominal_weight, 5e-2);
    nh.param<double>(
        "optimizer_base_smoothness_weight",
        optimizer_base_smoothness_weight,
        optimizer_smoothness_weight);
    nh.param<double>(
        "optimizer_base_nominal_weight",
        optimizer_base_nominal_weight,
        optimizer_nominal_weight);
    nh.param<double>(
        "optimizer_base_spectral_energy_weight",
        optimizer_base_spectral_energy_weight,
        0.0);
    nh.param<double>(
        "optimizer_base_spectral_cutoff_ratio",
        optimizer_base_spectral_cutoff_ratio,
        0.35);
    nh.param<double>(
        "optimizer_base_spectral_power",
        optimizer_base_spectral_power,
        2.0);
    nh.param<double>(
        "optimizer_base_spectral_metric_x",
        optimizer_base_spectral_metric_x,
        1.0);
    nh.param<double>(
        "optimizer_base_spectral_metric_y",
        optimizer_base_spectral_metric_y,
        1.0);
    nh.param<double>(
        "optimizer_base_spectral_metric_yaw",
        optimizer_base_spectral_metric_yaw,
        0.5);
    nh.getParam("initial_arm_joint_positions", initial_arm_joint_positions);

    if (urdf_path.empty())
    {
        ROS_ERROR("whole_body_demo_node: no urdf_path provided.");
        return -1;
    }

    polytope_wx::WholeBodyOptimizationConfig config;
    config.urdf_path = urdf_path;
    config.base_frame = base_frame;
    config.ee_frame = ee_frame;
    config.locked_joint_names = {
        "moca_franka_franka_gripper_finger_joint1",
        "moca_franka_franka_gripper_finger_joint2"
    };
    config.max_iterations_per_waypoint = 60;
    config.pose_gain = 0.9;
    config.nullspace_step_size = 0.01;
    config.max_state_update_norm = 0.08;
    config.pose_tolerance = 5e-4;
    config.capability_weight = optimizer_capability_weight;
    config.manipulability_weight = optimizer_manipulability_weight;
    config.joint_limit_weight = optimizer_joint_limit_weight;
    config.smoothness_weight = optimizer_smoothness_weight;
    // Keep the metric in the logs, but remove its effect from the objective.
    config.nominal_weight = 0.0;
    config.base_smoothness_weight = optimizer_base_smoothness_weight;
    // Keep the metric in the logs, but remove its effect from the objective.
    config.base_nominal_weight = 0.0;
    config.base_spectral_energy_weight = optimizer_base_spectral_energy_weight;
    config.base_spectral_energy_metric <<
        optimizer_base_spectral_metric_x,
        optimizer_base_spectral_metric_y,
        optimizer_base_spectral_metric_yaw;
    config.base_spectral_cutoff_ratio = optimizer_base_spectral_cutoff_ratio;
    config.base_spectral_power = optimizer_base_spectral_power;
    config.base_lower_limits << -0.25, -0.25, -0.75;
    config.base_upper_limits <<  0.25,  0.25,  0.75;
    config.constrain_orientation = constrain_orientation;
    config.verbose = false;

    polytope_wx::WholeBodyPolytopeOptimizer optimizer;
    if (!optimizer.initialize(config))
    {
        ROS_ERROR("whole_body_demo_node: failed to initialize whole-body optimizer.");
        return -1;
    }

    Eigen::VectorXd seed_state = optimizer.defaultState();
    seed_state.head(kBaseDof).setZero();
    if (initial_arm_joint_positions.size() == static_cast<std::size_t>(kArmDof))
    {
        for (int i = 0; i < kArmDof; ++i)
        {
            seed_state(kBaseDof + i) = initial_arm_joint_positions[i];
        }
    }
    else
    {
        const double fallback_arm[kArmDof] = {0.0, -0.4, 0.0, -1.8, 0.0, 1.4, 0.7};
        for (int i = 0; i < kArmDof; ++i)
        {
            seed_state(kBaseDof + i) = fallback_arm[i];
        }
    }

    const Eigen::Affine3d seed_pose = optimizer.computeEndEffectorPose(seed_state);

    polytope_wx::WholeBodyTrajectoryOptimizationInput input;
    input.target_poses.reserve(static_cast<std::size_t>(waypoint_count));
    input.desired_forces.reserve(static_cast<std::size_t>(waypoint_count));
    input.desired_force_radii.assign(static_cast<std::size_t>(waypoint_count), 0.0);
    input.nominal_state_trajectory.reserve(static_cast<std::size_t>(waypoint_count));
    input.waypoint_times_sec.reserve(static_cast<std::size_t>(waypoint_count));

    const double waypoint_dt =
        horizon_sec / std::max(1, waypoint_count - 1);
    for (int i = 0; i < waypoint_count; ++i)
    {
        const double time_sec = waypoint_dt * static_cast<double>(i);
        input.waypoint_times_sec.push_back(time_sec);
        input.target_poses.push_back(
            buildTargetPose(
                seed_pose,
                time_sec,
                trajectory_frequency_hz,
                amplitude_x,
                amplitude_y,
                amplitude_z));
        input.desired_forces.push_back(
            desired_force_magnitude * Eigen::Vector3d::UnitZ());
        input.nominal_state_trajectory.push_back(seed_state);
    }

    const polytope_wx::WholeBodyTrajectoryOptimizationResult result =
        optimizer.optimize(input);

    if (!ensureParentDirectoryForFile(output_csv_path))
    {
        ROS_ERROR_STREAM(
            "whole_body_demo_node: failed to create output directory for "
            << output_csv_path << ": " << std::strerror(errno));
        return -1;
    }

    std::ofstream csv_file(output_csv_path);
    if (!csv_file.is_open())
    {
        ROS_ERROR_STREAM("whole_body_demo_node: failed to open " << output_csv_path);
        return -1;
    }

    csv_file
        << "time_sec,"
        << "target_x,target_y,target_z,"
        << "optimized_ee_x,optimized_ee_y,optimized_ee_z,"
        << "base_x,base_y,base_yaw,"
        << "pose_error_norm,required_force,required_force_radius,force_capacity,force_ball_clearance,objective_value,"
        << "capability_cost,manipulability_measure,manipulability_cost,"
        << "joint_limit_cost,smoothness_cost,nominal_cost,"
        << "base_smoothness_cost,base_nominal_cost,base_spectral_energy_cost,"
        << "weighted_capability,weighted_manipulability,weighted_joint_limit,"
        << "weighted_smoothness,weighted_nominal,"
        << "weighted_base_smoothness,weighted_base_nominal,weighted_base_spectral_energy";
    for (int i = 0; i < kArmDof; ++i)
    {
        csv_file << ",arm_q" << (i + 1);
    }
    csv_file << "\n";

    double max_pose_error = 0.0;
    double max_base_translation = 0.0;
    double max_base_yaw = 0.0;

    for (std::size_t i = 0; i < result.optimized_state_trajectory.size(); ++i)
    {
        const double time_sec =
            i < input.waypoint_times_sec.size()
                ? input.waypoint_times_sec[i]
                : waypoint_dt * static_cast<double>(i);
        const Eigen::VectorXd& optimized_state = result.optimized_state_trajectory[i];
        const Eigen::Affine3d optimized_pose =
            optimizer.computeEndEffectorPose(optimized_state);
        const auto& metrics = result.waypoint_metrics[i];
        const double weighted_capability =
            config.capability_weight * metrics.capability_cost;
        const double weighted_manipulability =
            config.manipulability_weight * metrics.manipulability_cost;
        const double weighted_joint_limit =
            config.joint_limit_weight * metrics.joint_limit_cost;
        const double weighted_smoothness =
            config.smoothness_weight * metrics.smoothness_cost;
        const double weighted_nominal =
            config.nominal_weight * metrics.nominal_cost;
        const double weighted_base_smoothness =
            config.base_smoothness_weight * metrics.base_smoothness_cost;
        const double weighted_base_nominal =
            config.base_nominal_weight * metrics.base_nominal_cost;
        const double weighted_base_spectral_energy =
            config.base_spectral_energy_weight * result.base_spectral_energy_cost;

        csv_file
            << time_sec << ","
            << input.target_poses[i].translation().x() << ","
            << input.target_poses[i].translation().y() << ","
            << input.target_poses[i].translation().z() << ","
            << optimized_pose.translation().x() << ","
            << optimized_pose.translation().y() << ","
            << optimized_pose.translation().z() << ","
            << optimized_state(0) << ","
            << optimized_state(1) << ","
            << optimized_state(2) << ","
            << metrics.pose_error_norm << ","
            << metrics.required_force << ","
            << metrics.required_force_radius << ","
            << metrics.force_capacity << ","
            << metrics.force_ball_clearance << ","
            << metrics.objective_value << ","
            << metrics.capability_cost << ","
            << metrics.manipulability_measure << ","
            << metrics.manipulability_cost << ","
            << metrics.joint_limit_cost << ","
            << metrics.smoothness_cost << ","
            << metrics.nominal_cost << ","
            << metrics.base_smoothness_cost << ","
            << metrics.base_nominal_cost << ","
            << result.base_spectral_energy_cost << ","
            << weighted_capability << ","
            << weighted_manipulability << ","
            << weighted_joint_limit << ","
            << weighted_smoothness << ","
            << weighted_nominal << ","
            << weighted_base_smoothness << ","
            << weighted_base_nominal << ","
            << weighted_base_spectral_energy;

        for (int joint_index = 0; joint_index < kArmDof; ++joint_index)
        {
            csv_file << "," << optimized_state(kBaseDof + joint_index);
        }
        csv_file << "\n";

        max_pose_error = std::max(max_pose_error, metrics.pose_error_norm);
        max_base_translation = std::max(
            max_base_translation,
            std::sqrt(
                optimized_state(0) * optimized_state(0) +
                optimized_state(1) * optimized_state(1)));
        max_base_yaw = std::max(max_base_yaw, std::fabs(optimized_state(2)));
    }

    csv_file.close();

    ros::Publisher sample_publisher;
    if (publish_replay)
    {
        sample_publisher =
            nh.advertise<std_msgs::Float64MultiArray>("demo_sample", 4, false);
    }

    ROS_INFO_STREAM(
        "whole_body_demo_node: optimized " << result.optimized_state_trajectory.size()
        << " waypoints. success=" << (result.success ? "true" : "false")
        << ", max_pose_error=" << max_pose_error
        << ", max_base_translation=" << max_base_translation
        << ", max_base_yaw=" << max_base_yaw
        << ". CSV written to " << output_csv_path);

    if (publish_replay)
    {
        ROS_INFO_STREAM(
            "whole_body_demo_node: replaying optimized trajectory online on "
            << nh.resolveName("demo_sample"));
        ros::Duration(std::max(0.0, replay_startup_delay_sec)).sleep();

        for (std::size_t i = 0; i < result.optimized_state_trajectory.size() && ros::ok(); ++i)
        {
            const Eigen::VectorXd& optimized_state = result.optimized_state_trajectory[i];
            const Eigen::Affine3d optimized_pose =
                optimizer.computeEndEffectorPose(optimized_state);
            const auto& metrics = result.waypoint_metrics[i];
            const double weighted_capability =
                config.capability_weight * metrics.capability_cost;
            const double weighted_manipulability =
                config.manipulability_weight * metrics.manipulability_cost;
            const double weighted_joint_limit =
                config.joint_limit_weight * metrics.joint_limit_cost;
            const double weighted_smoothness =
                config.smoothness_weight * metrics.smoothness_cost;
            const double weighted_nominal =
                config.nominal_weight * metrics.nominal_cost;
            const double weighted_base_smoothness =
                config.base_smoothness_weight * metrics.base_smoothness_cost;
            const double weighted_base_nominal =
                config.base_nominal_weight * metrics.base_nominal_cost;

            std_msgs::Float64MultiArray sample_message;
            sample_message.data.reserve(1 + 3 + 3 + 3 + 17 + kArmDof);
            sample_message.data.push_back(waypoint_dt * static_cast<double>(i));
            sample_message.data.push_back(input.target_poses[i].translation().x());
            sample_message.data.push_back(input.target_poses[i].translation().y());
            sample_message.data.push_back(input.target_poses[i].translation().z());
            sample_message.data.push_back(optimized_pose.translation().x());
            sample_message.data.push_back(optimized_pose.translation().y());
            sample_message.data.push_back(optimized_pose.translation().z());
            sample_message.data.push_back(optimized_state(0));
            sample_message.data.push_back(optimized_state(1));
            sample_message.data.push_back(optimized_state(2));
            sample_message.data.push_back(metrics.pose_error_norm);
            sample_message.data.push_back(metrics.force_capacity);
            sample_message.data.push_back(metrics.objective_value);
            sample_message.data.push_back(metrics.required_force);
            sample_message.data.push_back(metrics.capability_cost);
            sample_message.data.push_back(metrics.manipulability_measure);
            sample_message.data.push_back(metrics.manipulability_cost);
            sample_message.data.push_back(metrics.joint_limit_cost);
            sample_message.data.push_back(metrics.smoothness_cost);
            sample_message.data.push_back(metrics.nominal_cost);
            sample_message.data.push_back(metrics.base_smoothness_cost);
            sample_message.data.push_back(metrics.base_nominal_cost);
            sample_message.data.push_back(weighted_capability);
            sample_message.data.push_back(weighted_manipulability);
            sample_message.data.push_back(weighted_joint_limit);
            sample_message.data.push_back(weighted_smoothness);
            sample_message.data.push_back(weighted_nominal);
            sample_message.data.push_back(weighted_base_smoothness);
            sample_message.data.push_back(weighted_base_nominal);
            for (int joint_index = 0; joint_index < kArmDof; ++joint_index)
            {
                sample_message.data.push_back(optimized_state(kBaseDof + joint_index));
            }

            sample_publisher.publish(sample_message);
            ros::spinOnce();

            if (i + 1 < result.optimized_state_trajectory.size())
            {
                ros::Duration(std::max(1e-3, waypoint_dt)).sleep();
            }
        }
    }

    return 0;
}
