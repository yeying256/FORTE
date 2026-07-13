#include <algorithm>
#include <array>
#include <cmath>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <geometry_msgs/PoseStamped.h>
#include <hrii_robot_msgs/FrankaToolState.h>
#include <moca_trajectory_generator/ForceCapabilitySettings.h>
#include <moca_trajectory_generator/StartPlanning.h>
#include <moca_vlm/MassEstimateRequest.h>
#include <moca_vlm/MassEstimateResult.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Empty.h>
#include <std_msgs/Int32.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/MarkerArray.h>
#include <XmlRpcValue.h>
#include <json/json.h>

#include <moca_trajectory_generator/TargetPoseCommand.h>
#include <polytope_ros/whole_body_optimizer.h>

namespace
{

constexpr int kArmJointCount = 7;
constexpr int kBaseJointCount = 3;
constexpr const char* kForceCapabilitySettingsParamNs =
    "/force_capability_settings/arm_torque_limits";

constexpr double kTrajectoryStartDelaySec = 1.0;
constexpr double kTrajectoryRampTimeSec = 2.0;
constexpr double kDefaultCircleFrequencyHz = 0.05;
constexpr double kDefaultCircleAmplitudeX = 0.04;
constexpr double kDefaultCircleAmplitudeY = 0.03;
constexpr double kDefaultCircleAmplitudeZ = 0.0;
constexpr double kDefaultTestForwardDistanceX = 1.2;
constexpr double kDefaultTestLiftDistanceZ = 0.08;
constexpr double kDefaultTestForwardDurationSec = 5.0;
constexpr double kDefaultTestLiftDurationSec = 5.0;

constexpr double kOptimizationHorizonSec = 10.0;
constexpr int kDefaultOptimizationWaypointCount = 41;
constexpr double kDefaultCircleDesiredForceMagnitude = 70.0;
constexpr double kDefaultLiftDesiredForceMagnitude = 70.0;
constexpr double kDefaultDesiredForceRadiusN = 0.0;
constexpr bool kDefaultEnableForceCapabilityOptimization = false;
constexpr bool kDefaultComparisonForceSweepEnabled = false;
constexpr double kDefaultComparisonFirstLiftForceMagnitude = 10.0;
constexpr double kDefaultComparisonSecondLiftForceMagnitude = 80.0;
constexpr double kDefaultComparisonReturnHomeDurationSec = 4.0;
constexpr double kDefaultComparisonHomeSettleDurationSec = 1.0;
constexpr double kDefaultComparisonGoalSettleDurationSec = 3.0;
constexpr double kDefaultDesiredForceRampTimeSec = 1.0;
constexpr bool kDefaultResetToHomeBeforeTrajectoryStart = true;
constexpr double kDefaultPreTrajectoryHomeSettleDurationSec = 1.0;
constexpr const char* kDefaultComparisonRunStateTopic = "/comparison_run_state";
constexpr const char* kDefaultResetToHomeTopic = "/reset_to_home";
constexpr const char* kDefaultVlmMassRequestTopic = "/vlm_mass_request";
constexpr const char* kDefaultVlmMassResultTopic = "/vlm_mass_result";
constexpr const char* kDefaultFrankaToolStateTopic =
    "/moca_white/franka_state_bridge/franka_tool_state";
constexpr const char* kDefaultStartPlanningService = "/target_pose_generator/start_planning";
constexpr const char* kDefaultTrajectoryMode = "payload_lift";
constexpr const char* kDefaultManualTargetPoseTopic = "/moca_manual_target_pose";
constexpr const char* kDefaultCollisionSphereMarkerTopic =
    "/target_pose_generator/collision_spheres";
constexpr const char* kReturnToHomeTrajectoryMode = "return_to_home";
constexpr const char* kTeachingReturnInitialTrajectoryMode = "teaching_return_initial";
constexpr const char* kTeachingWaypointsTrajectoryMode = "teaching_waypoints";
constexpr const char* kSinglePointOptimizationTrajectoryMode = "single_point_optimization";
constexpr const char* kDefaultTaskForceMode = "zero_force";
constexpr const char* kPolytopeUrdfPath = "package://polytope_ros/urdf/my_robot_fix_wheels.urdf";
constexpr const char* kPolytopeEeFrame = "moca_franka_EE";
constexpr const char* kLeftFingerJointName = "moca_franka_franka_gripper_finger_joint1";
constexpr const char* kRightFingerJointName = "moca_franka_franka_gripper_finger_joint2";
constexpr const char* kDefaultPlannerMetricsCsvPath =
    "package://polytope_ros/../../output/polytope_ros/planned_whole_body_metrics.csv";
constexpr double kDefaultPayloadBoxCenterX = 0.66;
constexpr double kDefaultPayloadBoxCenterY = 0.0;
constexpr double kDefaultPayloadBoxCenterZ = 0.12;
constexpr double kDefaultPayloadHandleTopZ = 0.34;
constexpr double kDefaultPayloadMoveAboveDurationSec = 5.0;
constexpr double kDefaultPayloadDescendDurationSec = 3.0;
constexpr double kDefaultPayloadCloseDurationSec = 1.5;
constexpr double kDefaultPayloadLiftDurationSec = 4.0;
constexpr double kDefaultPayloadPregraspHeight = 0.12;
constexpr double kDefaultPayloadGraspEeOffsetZ = 0.12;
constexpr double kDefaultPayloadLiftDistanceZ = 0.18;
constexpr double kDefaultFingerOpenCommand = 0.04;
constexpr double kDefaultFingerClosedCommand = 0.0;
constexpr double kDefaultManualTargetMoveDurationSec = 4.0;
constexpr double kDefaultSinglePointMoveDurationSec = 5.0;
constexpr double kDefaultHomeReturnDurationSec = 8.0;
constexpr double kDefaultTeachingWaypointSegmentDurationSec = 5.0;
constexpr const char* kDefaultHomeTeachingPointPath =
    "package://moca_trajectory_generator/teaching_point/home.json";
constexpr const char* kDefaultTeachingWaypointsPath =
    "package://moca_trajectory_generator/teaching_point/lift/way_points.json";

bool isFiniteQuaternion(const Eigen::Quaterniond& quaternion)
{
    return std::isfinite(quaternion.x()) &&
           std::isfinite(quaternion.y()) &&
           std::isfinite(quaternion.z()) &&
           std::isfinite(quaternion.w()) &&
           quaternion.squaredNorm() > 1e-12;
}

bool quaternionFromRotationMatrix(
    const Eigen::Matrix3d& rotation,
    Eigen::Quaterniond* quaternion)
{
    if (quaternion == nullptr || !rotation.allFinite())
    {
        return false;
    }

    Eigen::Quaterniond candidate(rotation);
    if (!isFiniteQuaternion(candidate))
    {
        return false;
    }

    candidate.normalize();
    if (!isFiniteQuaternion(candidate))
    {
        return false;
    }

    *quaternion = candidate;
    return true;
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

bool xmlRpcValueToDouble(XmlRpc::XmlRpcValue& value, double* output)
{
    if (output == nullptr)
    {
        return false;
    }

    if (value.getType() == XmlRpc::XmlRpcValue::TypeInt)
    {
        *output = static_cast<int>(value);
        return true;
    }
    if (value.getType() == XmlRpc::XmlRpcValue::TypeDouble)
    {
        *output = static_cast<double>(value);
        return true;
    }

    return false;
}

enum class ExperimentPhase
{
    kPrepareStart = -1,
    kTrajectory = 0,
    kReturnHome = 1,
    kSettleHome = 2,
    kComplete = 3,
};

struct CartesianTrajectorySample
{
    Eigen::Affine3d pose{Eigen::Affine3d::Identity()};
    Eigen::Matrix<double, 6, 1> twist{Eigen::Matrix<double, 6, 1>::Zero()};
    Eigen::Matrix<double, 6, 1> acceleration{Eigen::Matrix<double, 6, 1>::Zero()};
};

struct TeachingPoint
{
    Eigen::Matrix<double, kBaseJointCount + kArmJointCount, 1> whole_body_target{
        Eigen::Matrix<double, kBaseJointCount + kArmJointCount, 1>::Zero()};
    Eigen::Affine3d recorded_pose{Eigen::Affine3d::Identity()};
    bool has_recorded_pose{false};
};

struct OptimizerCostWeights
{
    double capability_weight{1.0};
    double manipulability_weight{1.0};
    double joint_limit_weight{1e-3};
    Eigen::VectorXd joint_limit_dof_weights{
        Eigen::VectorXd::Ones(kBaseJointCount + kArmJointCount)};
    double smoothness_weight{5e-2};
    double velocity_weight{0.0};
    double nominal_weight{0.0};
    double base_smoothness_weight{5e-2};
    double base_velocity_weight{0.0};
    double base_nominal_weight{0.0};
    double base_spectral_energy_weight{0.0};
    double collision_weight{0.0};
    double capability_near_weight{50.0};
    double capability_over_weight{500.0};
    double capability_over4_weight{5000.0};
};

class TargetPoseGenerator
{
public:
    using WholeBodyTarget = Eigen::Matrix<double, kBaseJointCount + kArmJointCount, 1>;

    TargetPoseGenerator()
    {
        pnh_.param("publish_rate_hz", publish_rate_hz_, publish_rate_hz_);
        pnh_.param(
            "use_nullspace_joint_target",
            use_nullspace_joint_target_,
            use_nullspace_joint_target_);
        pnh_.param(
            "log_optimization_info",
            log_optimization_info_,
            log_optimization_info_);
        pnh_.param(
            "replan_during_execution",
            replan_during_execution_,
            replan_during_execution_);
        pnh_.param<std::string>("seed_pose_topic", seed_pose_topic_, seed_pose_topic_);
        pnh_.param<std::string>(
            "seed_joint_state_topic", seed_joint_state_topic_, seed_joint_state_topic_);
        pnh_.param<std::string>(
            "target_pose_topic", target_pose_topic_, target_pose_topic_);
        pnh_.param<std::string>(
            "manual_target_pose_topic",
            manual_target_pose_topic_,
            manual_target_pose_topic_);
        pnh_.param(
            "publish_collision_sphere_markers",
            publish_collision_sphere_markers_,
            publish_collision_sphere_markers_);
        pnh_.param(
            "publish_realtime_collision_sphere_markers",
            publish_realtime_collision_sphere_markers_,
            publish_realtime_collision_sphere_markers_);
        pnh_.param<std::string>(
            "collision_sphere_marker_topic",
            collision_sphere_marker_topic_,
            collision_sphere_marker_topic_);
        pnh_.param<std::string>(
            "task_force_mode", task_force_mode_, task_force_mode_);
        pnh_.param(
            "gravity_acceleration_mps2",
            gravity_acceleration_mps2_,
            gravity_acceleration_mps2_);
        pnh_.param(
            "enable_vlm_mass_request",
            enable_vlm_mass_request_,
            enable_vlm_mass_request_);
        pnh_.param(
            "autostart_planning",
            autostart_planning_,
            autostart_planning_);
        pnh_.param<std::string>(
            "vlm_mass_request_topic",
            vlm_mass_request_topic_,
            vlm_mass_request_topic_);
        pnh_.param<std::string>(
            "vlm_mass_result_topic",
            vlm_mass_result_topic_,
            vlm_mass_result_topic_);
        pnh_.param<std::string>(
            "start_planning_service",
            start_planning_service_name_,
            start_planning_service_name_);
        pnh_.param<std::string>(
            "vlm_request_image_path",
            vlm_request_image_path_,
            vlm_request_image_path_);
        pnh_.param(
            "vlm_mass_request_timeout_sec",
            vlm_mass_request_timeout_sec_,
            vlm_mass_request_timeout_sec_);
        pnh_.param(
            "vlm_timeout_uses_default_payload_mass",
            vlm_timeout_uses_default_payload_mass_,
            vlm_timeout_uses_default_payload_mass_);
        pnh_.param(
            "vlm_mass_request_force_refresh",
            vlm_mass_request_force_refresh_,
            vlm_mass_request_force_refresh_);
        pnh_.param(
            "vlm_offline_mode",
            vlm_offline_mode_,
            vlm_offline_mode_);
        pnh_.param(
            "default_payload_mass_kg",
            default_payload_mass_kg_,
            default_payload_mass_kg_);
        pnh_.param<std::string>(
            "franka_tool_state_topic",
            franka_tool_state_topic_,
            franka_tool_state_topic_);
        pnh_.param(
            "enable_franka_tool_load_planning_compensation",
            enable_franka_tool_load_planning_compensation_,
            enable_franka_tool_load_planning_compensation_);
        pnh_.param<std::string>("world_frame", world_frame_, world_frame_);
        pnh_.param<std::string>(
            "planned_base_frame", planned_base_frame_, planned_base_frame_);
        pnh_.param<std::string>(
            "planned_ee_frame", planned_ee_frame_, planned_ee_frame_);
        pnh_.param(
            "export_planner_metrics",
            export_planner_metrics_,
            export_planner_metrics_);
        pnh_.param<std::string>(
            "planner_metrics_csv_path",
            planner_metrics_csv_path_,
            planner_metrics_csv_path_);
        pnh_.param<std::string>("trajectory_mode", trajectory_mode_, trajectory_mode_);
        pnh_.param(
            "manual_target_move_duration_sec",
            manual_target_move_duration_sec_,
            manual_target_move_duration_sec_);
        pnh_.param(
            "single_point_move_duration_sec",
            single_point_move_duration_sec_,
            single_point_move_duration_sec_);
        refreshSinglePointTargetParamsFromServer();
        pnh_.param<std::string>(
            "home_teaching_point_path",
            home_teaching_point_path_,
            home_teaching_point_path_);
        pnh_.param<std::string>(
            "teaching_waypoints_path",
            teaching_waypoints_path_,
            teaching_waypoints_path_);
        pnh_.param(
            "home_return_duration_sec",
            home_return_duration_sec_,
            home_return_duration_sec_);
        pnh_.param(
            "teaching_waypoint_segment_duration_sec",
            teaching_waypoint_segment_duration_sec_,
            teaching_waypoint_segment_duration_sec_);
        pnh_.param(
            "trajectory_start_delay_sec",
            trajectory_start_delay_sec_,
            trajectory_start_delay_sec_);
        pnh_.param(
            "trajectory_ramp_time_sec",
            trajectory_ramp_time_sec_,
            trajectory_ramp_time_sec_);
        pnh_.param(
            "optimization_horizon_sec",
            optimization_horizon_sec_,
            optimization_horizon_sec_);
        pnh_.param(
            "optimization_waypoint_count",
            optimization_waypoint_count_,
            optimization_waypoint_count_);
        pnh_.param("circle_frequency_hz", circle_frequency_hz_, circle_frequency_hz_);
        pnh_.param("circle_amplitude_x", circle_amplitude_x_, circle_amplitude_x_);
        pnh_.param("circle_amplitude_y", circle_amplitude_y_, circle_amplitude_y_);
        pnh_.param("circle_amplitude_z", circle_amplitude_z_, circle_amplitude_z_);
        pnh_.param(
            "circle_desired_force_magnitude",
            circle_desired_force_magnitude_,
            circle_desired_force_magnitude_);
        pnh_.param(
            "test_forward_distance_x",
            test_forward_distance_x_,
            test_forward_distance_x_);
        pnh_.param(
            "test_lift_distance_z",
            test_lift_distance_z_,
            test_lift_distance_z_);
        pnh_.param(
            "test_forward_duration_sec",
            test_forward_duration_sec_,
            test_forward_duration_sec_);
        pnh_.param(
            "test_lift_duration_sec",
            test_lift_duration_sec_,
            test_lift_duration_sec_);
        pnh_.param(
            "payload_box_center_x",
            payload_box_center_x_,
            payload_box_center_x_);
        pnh_.param(
            "payload_box_center_y",
            payload_box_center_y_,
            payload_box_center_y_);
        pnh_.param(
            "payload_box_center_z",
            payload_box_center_z_,
            payload_box_center_z_);
        pnh_.param(
            "payload_handle_top_z",
            payload_handle_top_z_,
            payload_handle_top_z_);
        pnh_.param(
            "payload_move_above_duration_sec",
            payload_move_above_duration_sec_,
            payload_move_above_duration_sec_);
        pnh_.param(
            "payload_descend_duration_sec",
            payload_descend_duration_sec_,
            payload_descend_duration_sec_);
        pnh_.param(
            "payload_close_duration_sec",
            payload_close_duration_sec_,
            payload_close_duration_sec_);
        pnh_.param(
            "payload_lift_duration_sec",
            payload_lift_duration_sec_,
            payload_lift_duration_sec_);
        pnh_.param(
            "payload_pregrasp_height",
            payload_pregrasp_height_,
            payload_pregrasp_height_);
        pnh_.param(
            "payload_grasp_ee_offset_z",
            payload_grasp_ee_offset_z_,
            payload_grasp_ee_offset_z_);
        pnh_.param(
            "payload_lift_distance_z",
            payload_lift_distance_z_,
            payload_lift_distance_z_);
        pnh_.param(
            "finger_open_command",
            finger_open_command_,
            finger_open_command_);
        pnh_.param(
            "finger_closed_command",
            finger_closed_command_,
            finger_closed_command_);
        pnh_.param(
            "lift_desired_force_magnitude",
            lift_desired_force_magnitude_,
            lift_desired_force_magnitude_);
        pnh_.param(
            "desired_force_radius_N",
            desired_force_radius_N_,
            desired_force_radius_N_);
        pnh_.param(
            "enable_force_capability_optimization",
            enable_force_capability_optimization_,
            enable_force_capability_optimization_);
        pnh_.param(
            "comparison_force_sweep_enabled",
            comparison_force_sweep_enabled_,
            comparison_force_sweep_enabled_);
        pnh_.param(
            "comparison_first_lift_force_magnitude",
            comparison_first_lift_force_magnitude_,
            comparison_first_lift_force_magnitude_);
        pnh_.param(
            "comparison_second_lift_force_magnitude",
            comparison_second_lift_force_magnitude_,
            comparison_second_lift_force_magnitude_);
        pnh_.param(
            "comparison_return_home_duration_sec",
            comparison_return_home_duration_sec_,
            comparison_return_home_duration_sec_);
        pnh_.param(
            "comparison_home_settle_duration_sec",
            comparison_home_settle_duration_sec_,
            comparison_home_settle_duration_sec_);
        pnh_.param(
            "comparison_goal_settle_duration_sec",
            comparison_goal_settle_duration_sec_,
            comparison_goal_settle_duration_sec_);
        pnh_.param(
            "desired_force_ramp_time_sec",
            desired_force_ramp_time_sec_,
            desired_force_ramp_time_sec_);
        pnh_.param(
            "reset_to_home_before_trajectory_start",
            reset_to_home_before_trajectory_start_,
            reset_to_home_before_trajectory_start_);
        pnh_.param(
            "pre_trajectory_home_settle_duration_sec",
            pre_trajectory_home_settle_duration_sec_,
            pre_trajectory_home_settle_duration_sec_);
        pnh_.param<std::string>(
            "comparison_run_state_topic",
            comparison_run_state_topic_,
            comparison_run_state_topic_);
        pnh_.param<std::string>(
            "reset_to_home_topic",
            reset_to_home_topic_,
            reset_to_home_topic_);
        pnh_.param<std::string>(
            "force_capability_settings_topic",
            force_capability_settings_topic_,
            force_capability_settings_topic_);
        pnh_.param<std::string>(
            "polytope_urdf_path", polytope_urdf_path_, polytope_urdf_path_);
        pnh_.param<std::string>(
            "polytope_ee_frame", polytope_ee_frame_, polytope_ee_frame_);
        pnh_.param<std::string>(
            "left_finger_joint_name", left_finger_joint_name_, left_finger_joint_name_);
        pnh_.param<std::string>(
            "right_finger_joint_name", right_finger_joint_name_, right_finger_joint_name_);
        pnh_.param(
            "optimizer_max_iterations_per_waypoint",
            optimizer_max_iterations_per_waypoint_,
            optimizer_max_iterations_per_waypoint_);
        pnh_.param(
            "optimizer_max_trajectory_iterations",
            optimizer_max_trajectory_iterations_,
            optimizer_max_trajectory_iterations_);
        pnh_.param("optimizer_pose_gain", optimizer_pose_gain_, optimizer_pose_gain_);
        pnh_.param(
            "optimizer_nullspace_step_size",
            optimizer_nullspace_step_size_,
            optimizer_nullspace_step_size_);
        pnh_.param(
            "optimizer_max_joint_update_norm",
            optimizer_max_joint_update_norm_,
            optimizer_max_joint_update_norm_);
        pnh_.param(
            "optimizer_pose_tolerance",
            optimizer_pose_tolerance_,
            optimizer_pose_tolerance_);
        pnh_.param(
            "optimizer_capability_weight",
            optimizer_capability_weight_,
            optimizer_capability_weight_);
        pnh_.param(
            "optimizer_manipulability_weight",
            optimizer_manipulability_weight_,
            optimizer_manipulability_weight_);
        pnh_.param(
            "optimizer_joint_limit_weight",
            optimizer_joint_limit_weight_,
            optimizer_joint_limit_weight_);
        optimizer_joint_limit_dof_weights_ =
            Eigen::VectorXd::Ones(kBaseJointCount + kArmJointCount);
        optimizer_joint_limit_dof_weights_.head(kBaseJointCount).setZero();
        std::vector<double> joint_limit_dof_weights;
        if (pnh_.getParam("optimizer_joint_limit_dof_weights", joint_limit_dof_weights))
        {
            if (joint_limit_dof_weights.size() == kBaseJointCount + kArmJointCount)
            {
                double max_positive_weight = 0.0;
                for (int i = 0; i < kBaseJointCount + kArmJointCount; ++i)
                {
                    const double raw_weight = joint_limit_dof_weights[i];
                    if (!std::isfinite(raw_weight))
                    {
                        ROS_WARN_STREAM(
                            "target_pose_generator: optimizer_joint_limit_dof_weights["
                            << i << "] is non-finite; using 0.0 for this entry.");
                        optimizer_joint_limit_dof_weights_(i) = 0.0;
                    }
                    else
                    {
                        optimizer_joint_limit_dof_weights_(i) = std::max(0.0, raw_weight);
                        max_positive_weight =
                            std::max(max_positive_weight, optimizer_joint_limit_dof_weights_(i));
                    }
                }
                if (max_positive_weight > 0.0)
                {
                    optimizer_joint_limit_dof_weights_ /= max_positive_weight;
                }
            }
            else
            {
                ROS_WARN_STREAM(
                    "target_pose_generator: optimizer_joint_limit_dof_weights expects "
                    << (kBaseJointCount + kArmJointCount) << " values, got "
                    << joint_limit_dof_weights.size() << "; using default [0,0,0,1,...,1].");
            }
        }
        pnh_.param(
            "optimizer_smoothness_weight",
            optimizer_smoothness_weight_,
            optimizer_smoothness_weight_);
        pnh_.param(
            "optimizer_velocity_weight",
            optimizer_velocity_weight_,
            optimizer_velocity_weight_);
        pnh_.param(
            "optimizer_base_velocity_weight",
            optimizer_base_velocity_weight_,
            optimizer_base_velocity_weight_);
        pnh_.param(
            "optimizer_nominal_weight",
            optimizer_nominal_weight_,
            optimizer_nominal_weight_);
        pnh_.param(
            "optimizer_base_spectral_energy_weight",
            optimizer_base_spectral_energy_weight_,
            optimizer_base_spectral_energy_weight_);
        pnh_.param(
            "optimizer_base_spectral_cutoff_ratio",
            optimizer_base_spectral_cutoff_ratio_,
            optimizer_base_spectral_cutoff_ratio_);
        pnh_.param(
            "optimizer_base_spectral_power",
            optimizer_base_spectral_power_,
            optimizer_base_spectral_power_);
        pnh_.param(
            "optimizer_base_spectral_metric_x",
            optimizer_base_spectral_metric_x_,
            optimizer_base_spectral_metric_x_);
        pnh_.param(
            "optimizer_base_spectral_metric_y",
            optimizer_base_spectral_metric_y_,
            optimizer_base_spectral_metric_y_);
        pnh_.param(
            "optimizer_base_spectral_metric_yaw",
            optimizer_base_spectral_metric_yaw_,
            optimizer_base_spectral_metric_yaw_);
        pnh_.param(
            "optimizer_collision_weight",
            optimizer_collision_weight_,
            optimizer_collision_weight_);
        pnh_.param(
            "optimizer_collision_safe_distance",
            optimizer_collision_safe_distance_,
            optimizer_collision_safe_distance_);
        loadCollisionSpheresParam(
            "optimizer_collision_basket_spheres",
            &optimizer_collision_basket_spheres_);
        loadCollisionSpheresParam(
            "optimizer_collision_body_spheres",
            &optimizer_collision_body_spheres_);
        pnh_.param(
            "optimizer_capability_alpha",
            optimizer_capability_alpha_,
            optimizer_capability_alpha_);
        pnh_.param(
            "optimizer_capability_exponent_limit_enabled",
            optimizer_capability_exponent_limit_enabled_,
            optimizer_capability_exponent_limit_enabled_);
        pnh_.param<std::string>(
            "optimizer_comparison_algorithm",
            optimizer_comparison_algorithm_,
            optimizer_comparison_algorithm_);
        std::vector<double> comparison_cone_axis_world;
        if (pnh_.getParam(
                "optimizer_comparison_cone_axis_world",
                comparison_cone_axis_world) &&
            comparison_cone_axis_world.size() == 3)
        {
            optimizer_comparison_cone_axis_world_ =
                Eigen::Vector3d(
                    comparison_cone_axis_world[0],
                    comparison_cone_axis_world[1],
                    comparison_cone_axis_world[2]);
        }
        pnh_.param(
            "optimizer_comparison_cone_half_angle_rad",
            optimizer_comparison_cone_half_angle_rad_,
            optimizer_comparison_cone_half_angle_rad_);
        pnh_.param(
            "optimizer_comparison_cone_ring_count",
            optimizer_comparison_cone_ring_count_,
            optimizer_comparison_cone_ring_count_);
        pnh_.param(
            "optimizer_comparison_cone_azimuth_count",
            optimizer_comparison_cone_azimuth_count_,
            optimizer_comparison_cone_azimuth_count_);
        pnh_.param(
            "optimizer_use_dynamic_residual_force_polytope",
            optimizer_use_dynamic_residual_force_polytope_,
            optimizer_use_dynamic_residual_force_polytope_);
        pnh_.param(
            "optimizer_dynamic_residual_diagnostics_enabled",
            optimizer_dynamic_residual_diagnostics_enabled_,
            optimizer_dynamic_residual_diagnostics_enabled_);
        pnh_.param<std::string>(
            "optimizer_dynamic_residual_diagnostics_directory",
            optimizer_dynamic_residual_diagnostics_directory_,
            optimizer_dynamic_residual_diagnostics_directory_);
        pnh_.param(
            "optimizer_dynamic_residual_diagnostics_max_records",
            optimizer_dynamic_residual_diagnostics_max_records_,
            optimizer_dynamic_residual_diagnostics_max_records_);
        pnh_.param(
            "optimizer_capability_near_weight",
            optimizer_capability_near_weight_,
            optimizer_capability_near_weight_);
        pnh_.param(
            "optimizer_capability_over_weight",
            optimizer_capability_over_weight_,
            optimizer_capability_over_weight_);
        pnh_.param(
            "optimizer_capability_over4_weight",
            optimizer_capability_over4_weight_,
            optimizer_capability_over4_weight_);
        pnh_.param(
            "optimizer_constrain_orientation",
            optimizer_constrain_orientation_,
            optimizer_constrain_orientation_);
        pnh_.param(
            "optimizer_verbose",
            optimizer_verbose_,
            optimizer_verbose_);
        loadAlgorithmCostWeightsParams();
        loadSinglePointOptimizerParams();

        optimization_waypoint_count_ = std::max(2, optimization_waypoint_count_);
        optimization_horizon_sec_ = std::max(1e-3, optimization_horizon_sec_);
        trajectory_ramp_time_sec_ = std::max(1e-6, trajectory_ramp_time_sec_);
        publish_rate_hz_ = std::max(1, publish_rate_hz_);
        vlm_request_image_path_ = resolveRosPath(vlm_request_image_path_);
        planner_metrics_csv_path_ = resolveRosPath(planner_metrics_csv_path_);
        polytope_urdf_path_ = resolveRosPath(polytope_urdf_path_);
        optimizer_dynamic_residual_diagnostics_directory_ =
            resolveRosPath(optimizer_dynamic_residual_diagnostics_directory_);

        target_pose_pub_ = nh_.advertise<moca_trajectory_generator::TargetPoseCommand>(
            target_pose_topic_, 1);
        collision_sphere_marker_pub_ =
            nh_.advertise<visualization_msgs::MarkerArray>(
                collision_sphere_marker_topic_,
                1);
        mass_request_pub_ = nh_.advertise<moca_vlm::MassEstimateRequest>(
            vlm_mass_request_topic_, 1);
        comparison_run_state_pub_ =
            nh_.advertise<std_msgs::Int32>(comparison_run_state_topic_, 1, true);
        reset_to_home_pub_ = nh_.advertise<std_msgs::Empty>(reset_to_home_topic_, 1, true);
        mass_result_sub_ = nh_.subscribe(
            vlm_mass_result_topic_, 1, &TargetPoseGenerator::vlmMassResultCallback, this);
        franka_tool_state_sub_ = nh_.subscribe(
            franka_tool_state_topic_,
            1,
            &TargetPoseGenerator::frankaToolStateCallback,
            this);
        seed_pose_sub_ = nh_.subscribe(
            seed_pose_topic_, 1, &TargetPoseGenerator::seedPoseCallback, this);
        seed_joint_state_sub_ = nh_.subscribe(
            seed_joint_state_topic_, 1, &TargetPoseGenerator::seedJointStateCallback, this);
        manual_target_pose_sub_ = nh_.subscribe(
            manual_target_pose_topic_, 1, &TargetPoseGenerator::manualTargetPoseCallback, this);
        force_capability_settings_sub_ = nh_.subscribe(
            force_capability_settings_topic_,
            1,
            &TargetPoseGenerator::forceCapabilitySettingsCallback,
            this);
        start_planning_service_ = nh_.advertiseService(
            start_planning_service_name_,
            &TargetPoseGenerator::startPlanningServiceCallback,
            this);

        std::vector<double> configured_arm_torque_limits;
        if (ros::param::get(kForceCapabilitySettingsParamNs, configured_arm_torque_limits))
        {
            if (configured_arm_torque_limits.size() == kArmJointCount)
            {
                optimizer_arm_torque_limits_override_ = Eigen::VectorXd::Zero(kArmJointCount);
                for (int i = 0; i < kArmJointCount; ++i)
                {
                    const double torque_limit = configured_arm_torque_limits[i];
                    if (!std::isfinite(torque_limit) || torque_limit <= 0.0)
                    {
                        ROS_ERROR_STREAM(
                            "target_pose_generator: invalid arm torque limit from ROS param at joint index "
                            << i << ".");
                        optimizer_arm_torque_limits_override_.resize(0);
                        break;
                    }
                    optimizer_arm_torque_limits_override_(i) = std::abs(torque_limit);
                }
                if (optimizer_arm_torque_limits_override_.size() == kArmJointCount)
                {
                    optimizer_reinitialization_requested_ = true;
                }
            }
            else
            {
                ROS_ERROR_STREAM(
                    "target_pose_generator: expected " << kArmJointCount
                    << " arm torque limits in ROS param namespace "
                    << kForceCapabilitySettingsParamNs << ", but got "
                    << configured_arm_torque_limits.size() << ".");
            }
        }

        estimated_payload_mass_kg_ = default_payload_mass_kg_;
        planning_start_requested_ = autostart_planning_;
        request_vlm_mass_for_pending_start_ =
            enable_vlm_mass_request_ && autostart_planning_;
    }

    void spin()
    {
        ros::Rate rate(publish_rate_hz_);
        while (ros::ok())
        {
            ros::spinOnce();
            maybeStartTrajectory();
            publishTargetCommand();
            publishRealtimeCollisionSphereMarkers(ros::Time::now());
            rate.sleep();
        }
    }

private:
    void seedPoseCallback(const geometry_msgs::PoseStampedConstPtr& msg)
    {
        seed_pose_ = *msg;
        seed_pose_.header.frame_id = world_frame_;
        has_seed_pose_ = true;
        if (!has_home_pose_)
        {
            home_pose_ = seed_pose_;
            has_home_pose_ = true;
        }
        maybeStartTrajectory();
    }

    void seedJointStateCallback(const sensor_msgs::JointStateConstPtr& msg)
    {
        if (msg->position.size() < kBaseJointCount + kArmJointCount)
        {
            ROS_WARN_THROTTLE(
                1.0,
                "target_pose_generator: received joint state with %zu positions, expected at least %d.",
                msg->position.size(),
                kBaseJointCount + kArmJointCount);
            return;
        }

        for (int i = 0; i < kBaseJointCount; ++i)
        {
            seed_base_planar_positions_[i] = msg->position[i];
        }

        for (int i = 0; i < kArmJointCount; ++i)
        {
            seed_arm_joint_positions_[i] = msg->position[kBaseJointCount + i];
        }

        has_seed_joint_positions_ = true;
        if (!has_home_joint_positions_)
        {
            home_base_planar_positions_ = seed_base_planar_positions_;
            home_arm_joint_positions_ = seed_arm_joint_positions_;
            has_home_joint_positions_ = true;
        }
        maybeStartTrajectory();
    }

    void manualTargetPoseCallback(const geometry_msgs::PoseStampedConstPtr& msg)
    {
        manual_target_pose_ = *msg;
        manual_target_pose_.header.frame_id = world_frame_;
        has_manual_target_pose_ = true;

        if (!trajectory_started_)
        {
            maybeStartTrajectory();
        }
    }

    bool readFixedSizeDoubleVectorParam(
        const std::string& param_name,
        int expected_size,
        std::vector<double>* values) const
    {
        if (values == nullptr)
        {
            return false;
        }

        std::vector<double> loaded_values;
        if (!pnh_.getParam(param_name, loaded_values))
        {
            return false;
        }

        if (static_cast<int>(loaded_values.size()) != expected_size)
        {
            ROS_WARN_STREAM(
                "target_pose_generator: " << param_name << " expects "
                << expected_size << " values, got " << loaded_values.size()
                << "; keeping the previous value.");
            return false;
        }

        for (int i = 0; i < expected_size; ++i)
        {
            if (!std::isfinite(loaded_values[static_cast<std::size_t>(i)]))
            {
                ROS_WARN_STREAM(
                    "target_pose_generator: " << param_name << "[" << i
                    << "] is non-finite; keeping the previous value.");
                return false;
            }
        }

        *values = loaded_values;
        return true;
    }

    static void normalizeOptimizerDofWeights(
        const std::string& param_name,
        const std::vector<double>& raw_weights,
        Eigen::VectorXd* weights)
    {
        if (weights == nullptr)
        {
            return;
        }
        if (raw_weights.size() != kBaseJointCount + kArmJointCount)
        {
            ROS_WARN_STREAM(
                "target_pose_generator: " << param_name << " expects "
                << (kBaseJointCount + kArmJointCount) << " values, got "
                << raw_weights.size() << "; keeping the inherited value.");
            return;
        }

        Eigen::VectorXd normalized =
            Eigen::VectorXd::Zero(kBaseJointCount + kArmJointCount);
        double max_positive_weight = 0.0;
        for (int i = 0; i < kBaseJointCount + kArmJointCount; ++i)
        {
            const double raw_weight = raw_weights[static_cast<std::size_t>(i)];
            if (!std::isfinite(raw_weight))
            {
                ROS_WARN_STREAM(
                    "target_pose_generator: " << param_name << "[" << i
                    << "] is non-finite; using 0.0 for this entry.");
                normalized(i) = 0.0;
            }
            else
            {
                normalized(i) = std::max(0.0, raw_weight);
                max_positive_weight = std::max(max_positive_weight, normalized(i));
            }
        }
        if (max_positive_weight > 0.0)
        {
            normalized /= max_positive_weight;
        }
        *weights = normalized;
    }

    OptimizerCostWeights buildDefaultOptimizerCostWeights() const
    {
        OptimizerCostWeights weights;
        weights.capability_weight = optimizer_capability_weight_;
        weights.manipulability_weight = optimizer_manipulability_weight_;
        weights.joint_limit_weight = optimizer_joint_limit_weight_;
        weights.joint_limit_dof_weights = optimizer_joint_limit_dof_weights_;
        weights.smoothness_weight = optimizer_smoothness_weight_;
        weights.velocity_weight = optimizer_velocity_weight_;
        weights.nominal_weight = optimizer_nominal_weight_;
        weights.base_smoothness_weight = optimizer_smoothness_weight_;
        weights.base_velocity_weight = optimizer_base_velocity_weight_;
        weights.base_nominal_weight = 0.0;
        weights.base_spectral_energy_weight = optimizer_base_spectral_energy_weight_;
        weights.collision_weight = optimizer_collision_weight_;
        weights.capability_near_weight = optimizer_capability_near_weight_;
        weights.capability_over_weight = optimizer_capability_over_weight_;
        weights.capability_over4_weight = optimizer_capability_over4_weight_;
        return weights;
    }

    OptimizerCostWeights buildSinglePointOptimizerCostWeights() const
    {
        OptimizerCostWeights weights;
        weights.capability_weight = single_point_optimizer_capability_weight_;
        weights.manipulability_weight = single_point_optimizer_manipulability_weight_;
        weights.joint_limit_weight = single_point_optimizer_joint_limit_weight_;
        weights.joint_limit_dof_weights = single_point_optimizer_joint_limit_dof_weights_;
        weights.smoothness_weight = single_point_optimizer_smoothness_weight_;
        weights.velocity_weight = single_point_optimizer_velocity_weight_;
        weights.nominal_weight = single_point_optimizer_nominal_weight_;
        weights.base_smoothness_weight = single_point_optimizer_smoothness_weight_;
        weights.base_velocity_weight = single_point_optimizer_base_velocity_weight_;
        weights.base_nominal_weight = 0.0;
        weights.base_spectral_energy_weight =
            single_point_optimizer_base_spectral_energy_weight_;
        weights.collision_weight = single_point_optimizer_collision_weight_;
        weights.capability_near_weight = single_point_optimizer_capability_near_weight_;
        weights.capability_over_weight = single_point_optimizer_capability_over_weight_;
        weights.capability_over4_weight = single_point_optimizer_capability_over4_weight_;
        return weights;
    }

    void loadAlgorithmCostWeightsParam(
        const std::string& algorithm_name,
        const OptimizerCostWeights& inherited_weights,
        OptimizerCostWeights* output)
    {
        if (output == nullptr)
        {
            return;
        }

        OptimizerCostWeights weights = inherited_weights;
        const std::string prefix = "optimizer_algorithm_weights/" + algorithm_name;
        pnh_.param(
            prefix + "/capability_weight",
            weights.capability_weight,
            weights.capability_weight);
        pnh_.param(
            prefix + "/manipulability_weight",
            weights.manipulability_weight,
            weights.manipulability_weight);
        pnh_.param(
            prefix + "/joint_limit_weight",
            weights.joint_limit_weight,
            weights.joint_limit_weight);

        std::vector<double> joint_limit_dof_weights;
        if (pnh_.getParam(prefix + "/joint_limit_dof_weights", joint_limit_dof_weights))
        {
            normalizeOptimizerDofWeights(
                prefix + "/joint_limit_dof_weights",
                joint_limit_dof_weights,
                &weights.joint_limit_dof_weights);
        }

        pnh_.param(
            prefix + "/smoothness_weight",
            weights.smoothness_weight,
            weights.smoothness_weight);
        pnh_.param(
            prefix + "/velocity_weight",
            weights.velocity_weight,
            weights.velocity_weight);
        pnh_.param(
            prefix + "/nominal_weight",
            weights.nominal_weight,
            weights.nominal_weight);
        pnh_.param(
            prefix + "/base_smoothness_weight",
            weights.base_smoothness_weight,
            weights.base_smoothness_weight);
        pnh_.param(
            prefix + "/base_velocity_weight",
            weights.base_velocity_weight,
            weights.base_velocity_weight);
        pnh_.param(
            prefix + "/base_nominal_weight",
            weights.base_nominal_weight,
            weights.base_nominal_weight);
        pnh_.param(
            prefix + "/base_spectral_energy_weight",
            weights.base_spectral_energy_weight,
            weights.base_spectral_energy_weight);
        pnh_.param(
            prefix + "/collision_weight",
            weights.collision_weight,
            weights.collision_weight);
        pnh_.param(
            prefix + "/capability_near_weight",
            weights.capability_near_weight,
            weights.capability_near_weight);
        pnh_.param(
            prefix + "/capability_over_weight",
            weights.capability_over_weight,
            weights.capability_over_weight);
        pnh_.param(
            prefix + "/capability_over4_weight",
            weights.capability_over4_weight,
            weights.capability_over4_weight);

        *output = weights;
    }

    void loadAlgorithmCostWeightsParams()
    {
        const OptimizerCostWeights inherited_weights =
            buildDefaultOptimizerCostWeights();
        optimizer_algorithm_cost_weights_.clear();

        const std::array<std::string, 6> algorithm_names{{
            "ours",
            "manipulability_only",
            "static_force_polytope",
            "residual_force_polytope",
            "cone_intersection_volume",
            "joint_quintic_interpolation"}};
        for (const std::string& algorithm_name : algorithm_names)
        {
            OptimizerCostWeights weights;
            loadAlgorithmCostWeightsParam(
                algorithm_name,
                inherited_weights,
                &weights);
            optimizer_algorithm_cost_weights_[algorithm_name] = weights;
        }
    }

    OptimizerCostWeights optimizerCostWeightsForAlgorithm(
        polytope_wx::RfpComparisonAlgorithm algorithm) const
    {
        const std::string algorithm_name =
            polytope_wx::rfpComparisonAlgorithmName(algorithm);
        const auto it = optimizer_algorithm_cost_weights_.find(algorithm_name);
        if (it != optimizer_algorithm_cost_weights_.end())
        {
            return it->second;
        }
        return buildDefaultOptimizerCostWeights();
    }

    bool isJointQuinticInterpolationAlgorithm() const
    {
        return polytope_wx::parseRfpComparisonAlgorithmOrOurs(
                   optimizer_comparison_algorithm_) ==
               polytope_wx::RfpComparisonAlgorithm::JointQuinticInterpolation;
    }

    void loadSinglePointOptimizerParams()
    {
        single_point_optimizer_max_iterations_per_waypoint_ =
            optimizer_max_iterations_per_waypoint_;
        single_point_optimizer_max_trajectory_iterations_ =
            optimizer_max_trajectory_iterations_;
        single_point_optimizer_pose_gain_ = optimizer_pose_gain_;
        single_point_optimizer_nullspace_step_size_ = optimizer_nullspace_step_size_;
        single_point_optimizer_max_joint_update_norm_ = optimizer_max_joint_update_norm_;
        single_point_optimizer_pose_tolerance_ = optimizer_pose_tolerance_;
        single_point_optimizer_capability_weight_ = optimizer_capability_weight_;
        single_point_optimizer_manipulability_weight_ = optimizer_manipulability_weight_;
        single_point_optimizer_joint_limit_weight_ = optimizer_joint_limit_weight_;
        single_point_optimizer_joint_limit_dof_weights_ = optimizer_joint_limit_dof_weights_;
        single_point_optimizer_smoothness_weight_ = optimizer_smoothness_weight_;
        single_point_optimizer_velocity_weight_ = optimizer_velocity_weight_;
        single_point_optimizer_base_velocity_weight_ = optimizer_base_velocity_weight_;
        single_point_optimizer_nominal_weight_ = optimizer_nominal_weight_;
        single_point_optimizer_base_spectral_energy_weight_ =
            optimizer_base_spectral_energy_weight_;
        single_point_optimizer_base_spectral_cutoff_ratio_ =
            optimizer_base_spectral_cutoff_ratio_;
        single_point_optimizer_base_spectral_power_ = optimizer_base_spectral_power_;
        single_point_optimizer_base_spectral_metric_x_ = optimizer_base_spectral_metric_x_;
        single_point_optimizer_base_spectral_metric_y_ = optimizer_base_spectral_metric_y_;
        single_point_optimizer_base_spectral_metric_yaw_ = optimizer_base_spectral_metric_yaw_;
        single_point_optimizer_collision_weight_ = optimizer_collision_weight_;
        single_point_optimizer_collision_safe_distance_ = optimizer_collision_safe_distance_;
        single_point_optimizer_collision_basket_spheres_ = optimizer_collision_basket_spheres_;
        single_point_optimizer_collision_body_spheres_ = optimizer_collision_body_spheres_;
        single_point_optimizer_capability_alpha_ = optimizer_capability_alpha_;
        single_point_optimizer_capability_exponent_limit_enabled_ =
            optimizer_capability_exponent_limit_enabled_;
        single_point_optimizer_use_dynamic_residual_force_polytope_ = false;
        single_point_optimizer_capability_near_weight_ = optimizer_capability_near_weight_;
        single_point_optimizer_capability_over_weight_ = optimizer_capability_over_weight_;
        single_point_optimizer_capability_over4_weight_ = optimizer_capability_over4_weight_;
        single_point_optimizer_constrain_orientation_ = optimizer_constrain_orientation_;
        single_point_optimizer_verbose_ = optimizer_verbose_;

        pnh_.param(
            "single_point_optimizer_max_iterations_per_waypoint",
            single_point_optimizer_max_iterations_per_waypoint_,
            single_point_optimizer_max_iterations_per_waypoint_);
        pnh_.param(
            "single_point_optimizer_max_trajectory_iterations",
            single_point_optimizer_max_trajectory_iterations_,
            single_point_optimizer_max_trajectory_iterations_);
        pnh_.param(
            "single_point_optimizer_pose_gain",
            single_point_optimizer_pose_gain_,
            single_point_optimizer_pose_gain_);
        pnh_.param(
            "single_point_optimizer_nullspace_step_size",
            single_point_optimizer_nullspace_step_size_,
            single_point_optimizer_nullspace_step_size_);
        pnh_.param(
            "single_point_optimizer_max_joint_update_norm",
            single_point_optimizer_max_joint_update_norm_,
            single_point_optimizer_max_joint_update_norm_);
        pnh_.param(
            "single_point_optimizer_pose_tolerance",
            single_point_optimizer_pose_tolerance_,
            single_point_optimizer_pose_tolerance_);
        pnh_.param(
            "single_point_optimizer_capability_weight",
            single_point_optimizer_capability_weight_,
            single_point_optimizer_capability_weight_);
        pnh_.param(
            "single_point_optimizer_manipulability_weight",
            single_point_optimizer_manipulability_weight_,
            single_point_optimizer_manipulability_weight_);
        pnh_.param(
            "single_point_optimizer_joint_limit_weight",
            single_point_optimizer_joint_limit_weight_,
            single_point_optimizer_joint_limit_weight_);

        std::vector<double> single_point_joint_limit_dof_weights;
        if (pnh_.getParam(
                "single_point_optimizer_joint_limit_dof_weights",
                single_point_joint_limit_dof_weights))
        {
            normalizeOptimizerDofWeights(
                "single_point_optimizer_joint_limit_dof_weights",
                single_point_joint_limit_dof_weights,
                &single_point_optimizer_joint_limit_dof_weights_);
        }

        pnh_.param(
            "single_point_optimizer_smoothness_weight",
            single_point_optimizer_smoothness_weight_,
            single_point_optimizer_smoothness_weight_);
        pnh_.param(
            "single_point_optimizer_velocity_weight",
            single_point_optimizer_velocity_weight_,
            single_point_optimizer_velocity_weight_);
        pnh_.param(
            "single_point_optimizer_base_velocity_weight",
            single_point_optimizer_base_velocity_weight_,
            single_point_optimizer_base_velocity_weight_);
        pnh_.param(
            "single_point_optimizer_nominal_weight",
            single_point_optimizer_nominal_weight_,
            single_point_optimizer_nominal_weight_);
        pnh_.param(
            "single_point_optimizer_base_spectral_energy_weight",
            single_point_optimizer_base_spectral_energy_weight_,
            single_point_optimizer_base_spectral_energy_weight_);
        pnh_.param(
            "single_point_optimizer_base_spectral_cutoff_ratio",
            single_point_optimizer_base_spectral_cutoff_ratio_,
            single_point_optimizer_base_spectral_cutoff_ratio_);
        pnh_.param(
            "single_point_optimizer_base_spectral_power",
            single_point_optimizer_base_spectral_power_,
            single_point_optimizer_base_spectral_power_);
        pnh_.param(
            "single_point_optimizer_base_spectral_metric_x",
            single_point_optimizer_base_spectral_metric_x_,
            single_point_optimizer_base_spectral_metric_x_);
        pnh_.param(
            "single_point_optimizer_base_spectral_metric_y",
            single_point_optimizer_base_spectral_metric_y_,
            single_point_optimizer_base_spectral_metric_y_);
        pnh_.param(
            "single_point_optimizer_base_spectral_metric_yaw",
            single_point_optimizer_base_spectral_metric_yaw_,
            single_point_optimizer_base_spectral_metric_yaw_);
        pnh_.param(
            "single_point_optimizer_collision_weight",
            single_point_optimizer_collision_weight_,
            single_point_optimizer_collision_weight_);
        pnh_.param(
            "single_point_optimizer_collision_safe_distance",
            single_point_optimizer_collision_safe_distance_,
            single_point_optimizer_collision_safe_distance_);
        if (pnh_.hasParam("single_point_optimizer_collision_basket_spheres"))
        {
            loadCollisionSpheresParam(
                "single_point_optimizer_collision_basket_spheres",
                &single_point_optimizer_collision_basket_spheres_);
        }
        if (pnh_.hasParam("single_point_optimizer_collision_body_spheres"))
        {
            loadCollisionSpheresParam(
                "single_point_optimizer_collision_body_spheres",
                &single_point_optimizer_collision_body_spheres_);
        }
        pnh_.param(
            "single_point_optimizer_capability_alpha",
            single_point_optimizer_capability_alpha_,
            single_point_optimizer_capability_alpha_);
        pnh_.param(
            "single_point_optimizer_capability_exponent_limit_enabled",
            single_point_optimizer_capability_exponent_limit_enabled_,
            single_point_optimizer_capability_exponent_limit_enabled_);
        pnh_.param(
            "single_point_optimizer_use_dynamic_residual_force_polytope",
            single_point_optimizer_use_dynamic_residual_force_polytope_,
            single_point_optimizer_use_dynamic_residual_force_polytope_);
        pnh_.param(
            "single_point_optimizer_capability_near_weight",
            single_point_optimizer_capability_near_weight_,
            single_point_optimizer_capability_near_weight_);
        pnh_.param(
            "single_point_optimizer_capability_over_weight",
            single_point_optimizer_capability_over_weight_,
            single_point_optimizer_capability_over_weight_);
        pnh_.param(
            "single_point_optimizer_capability_over4_weight",
            single_point_optimizer_capability_over4_weight_,
            single_point_optimizer_capability_over4_weight_);
        pnh_.param(
            "single_point_optimizer_constrain_orientation",
            single_point_optimizer_constrain_orientation_,
            single_point_optimizer_constrain_orientation_);
        pnh_.param(
            "single_point_optimizer_verbose",
            single_point_optimizer_verbose_,
            single_point_optimizer_verbose_);
    }

    void refreshSinglePointTargetParamsFromServer()
    {
        pnh_.param(
            "single_point_move_duration_sec",
            single_point_move_duration_sec_,
            single_point_move_duration_sec_);
        single_point_move_duration_sec_ =
            std::max(1e-6, single_point_move_duration_sec_);

        std::vector<double> position_values;
        if (readFixedSizeDoubleVectorParam(
                "single_point_target_position",
                3,
                &position_values))
        {
            single_point_target_position_ <<
                position_values[0],
                position_values[1],
                position_values[2];
        }

        std::vector<double> orientation_values;
        if (readFixedSizeDoubleVectorParam(
                "single_point_target_orientation_xyzw",
                4,
                &orientation_values))
        {
            const Eigen::Quaterniond target_orientation(
                orientation_values[3],
                orientation_values[0],
                orientation_values[1],
                orientation_values[2]);
            if (isFiniteQuaternion(target_orientation))
            {
                single_point_target_orientation_ = target_orientation.normalized();
            }
            else
            {
                ROS_WARN(
                    "target_pose_generator: single_point_target_orientation_xyzw is invalid; "
                    "keeping the previous orientation.");
            }
        }

        single_point_has_target_whole_body_ = false;
        std::vector<double> whole_body_values;
        if (readFixedSizeDoubleVectorParam(
                "single_point_target_whole_body",
                kBaseJointCount + kArmJointCount,
                &whole_body_values))
        {
            for (int i = 0; i < kBaseJointCount + kArmJointCount; ++i)
            {
                single_point_target_whole_body_(i) =
                    whole_body_values[static_cast<std::size_t>(i)];
            }
            single_point_has_target_whole_body_ = true;
        }
    }

    void forceCapabilitySettingsCallback(
        const moca_trajectory_generator::ForceCapabilitySettingsConstPtr& msg)
    {
        if (msg->arm_torque_limits.size() != kArmJointCount)
        {
            ROS_WARN_STREAM(
                "target_pose_generator: received " << msg->arm_torque_limits.size()
                << " torque limits, expected " << kArmJointCount << ". Ignoring update.");
            return;
        }

        Eigen::VectorXd updated_limits = Eigen::VectorXd::Zero(kArmJointCount);
        for (int i = 0; i < kArmJointCount; ++i)
        {
            const double torque_limit = msg->arm_torque_limits[i];
            if (!std::isfinite(torque_limit) || torque_limit <= 0.0)
            {
                ROS_WARN_STREAM(
                    "target_pose_generator: received invalid torque limit for joint index "
                    << i << ". Ignoring update.");
                return;
            }
            updated_limits(i) = std::abs(torque_limit);
        }

        optimizer_arm_torque_limits_override_ = updated_limits;
        optimizer_reinitialization_requested_ = true;
        ros::param::set(
            kForceCapabilitySettingsParamNs,
            std::vector<double>(
                optimizer_arm_torque_limits_override_.data(),
                optimizer_arm_torque_limits_override_.data() +
                    optimizer_arm_torque_limits_override_.size()));

        if (!initializeOptimizer())
        {
            ROS_ERROR(
                "target_pose_generator: failed to reinitialize whole-body optimizer after a torque-limit update.");
            return;
        }

        ROS_INFO_STREAM(
            "target_pose_generator: updated arm torque limits to "
            << optimizer_arm_torque_limits_override_.transpose());
    }

    bool jsonArrayToVector(
        const Json::Value& value,
        std::size_t expected_size,
        std::vector<double>* output,
        const std::string& field_name) const
    {
        if (output == nullptr || !value.isArray() || value.size() != expected_size)
        {
            ROS_WARN_STREAM(
                "target_pose_generator: teaching point field '" << field_name
                << "' must be an array of size " << expected_size << ".");
            return false;
        }

        output->clear();
        output->reserve(expected_size);
        for (Json::ArrayIndex i = 0; i < value.size(); ++i)
        {
            if (!value[i].isNumeric())
            {
                ROS_WARN_STREAM(
                    "target_pose_generator: teaching point field '" << field_name
                    << "' contains a non-numeric value at index " << i << ".");
                return false;
            }
            output->push_back(value[i].asDouble());
        }
        return true;
    }

    bool readTeachingJson(const std::string& path, Json::Value* root) const
    {
        if (root == nullptr)
        {
            return false;
        }

        const std::string resolved_path = resolveRosPath(path);
        std::ifstream input(resolved_path);
        if (!input)
        {
            ROS_WARN_STREAM(
                "target_pose_generator: failed to open teaching JSON '"
                << resolved_path << "'.");
            return false;
        }

        Json::CharReaderBuilder builder;
        std::string errors;
        if (!Json::parseFromStream(builder, input, root, &errors))
        {
            ROS_WARN_STREAM(
                "target_pose_generator: failed to parse teaching JSON '"
                << resolved_path << "': " << errors);
            return false;
        }
        return true;
    }

    bool parseTeachingPoint(const Json::Value& point, TeachingPoint* output) const
    {
        if (output == nullptr || !point.isObject())
        {
            return false;
        }

        std::vector<double> arm_positions;
        if (!jsonArrayToVector(
                point["arm_joint_positions"],
                kArmJointCount,
                &arm_positions,
                "arm_joint_positions"))
        {
            return false;
        }

        output->whole_body_target.setZero();
        const Json::Value& planar = point["odom"]["planar"];
        if (planar.isObject())
        {
            output->whole_body_target(0) = planar.get("x", 0.0).asDouble();
            output->whole_body_target(1) = planar.get("y", 0.0).asDouble();
            output->whole_body_target(2) = planar.get("yaw", 0.0).asDouble();
        }
        for (int i = 0; i < kArmJointCount; ++i)
        {
            output->whole_body_target(kBaseJointCount + i) =
                arm_positions[static_cast<std::size_t>(i)];
        }

        const Json::Value& pose_value = point["ee_pose"]["pose"];
        if (pose_value.isObject())
        {
            const Json::Value& position = pose_value["position"];
            const Json::Value& orientation = pose_value["orientation"];
            if (position.isObject() && orientation.isObject())
            {
                output->recorded_pose.translation() <<
                    position.get("x", 0.0).asDouble(),
                    position.get("y", 0.0).asDouble(),
                    position.get("z", 0.0).asDouble();
                const Eigen::Quaterniond q(
                    orientation.get("w", 1.0).asDouble(),
                    orientation.get("x", 0.0).asDouble(),
                    orientation.get("y", 0.0).asDouble(),
                    orientation.get("z", 0.0).asDouble());
                output->recorded_pose.linear() =
                    q.normalized().toRotationMatrix();
                output->has_recorded_pose = true;
            }
        }
        return true;
    }

    bool loadTeachingPoints(
        const std::string& path,
        std::vector<TeachingPoint>* points) const
    {
        if (points == nullptr)
        {
            return false;
        }

        Json::Value root;
        if (!readTeachingJson(path, &root))
        {
            return false;
        }

        const Json::Value& point_array = root["points"];
        if (!point_array.isArray() || point_array.empty())
        {
            ROS_WARN_STREAM(
                "target_pose_generator: teaching JSON '" << resolveRosPath(path)
                << "' does not contain a non-empty points array.");
            return false;
        }

        points->clear();
        points->reserve(point_array.size());
        for (Json::ArrayIndex i = 0; i < point_array.size(); ++i)
        {
            TeachingPoint parsed_point;
            if (!parseTeachingPoint(point_array[i], &parsed_point))
            {
                ROS_WARN_STREAM(
                    "target_pose_generator: failed to parse teaching point "
                    << i << " from " << resolveRosPath(path) << ".");
                return false;
            }
            points->push_back(parsed_point);
        }
        return true;
    }

    void refreshTeachingTrajectoryParamsFromServer()
    {
        pnh_.param<std::string>(
            "home_teaching_point_path",
            home_teaching_point_path_,
            home_teaching_point_path_);
        pnh_.param<std::string>(
            "teaching_waypoints_path",
            teaching_waypoints_path_,
            teaching_waypoints_path_);
        pnh_.param(
            "home_return_duration_sec",
            home_return_duration_sec_,
            home_return_duration_sec_);
        pnh_.param(
            "teaching_waypoint_segment_duration_sec",
            teaching_waypoint_segment_duration_sec_,
            teaching_waypoint_segment_duration_sec_);
    }

    void refreshOptimizerComparisonParamsFromServer()
    {
        std::string requested_algorithm = optimizer_comparison_algorithm_;
        bool requested_dynamic_residual = optimizer_use_dynamic_residual_force_polytope_;
        pnh_.param<std::string>(
            "optimizer_comparison_algorithm",
            requested_algorithm,
            requested_algorithm);
        Eigen::Vector3d requested_cone_axis_eigen =
            optimizer_comparison_cone_axis_world_;
        std::vector<double> requested_cone_axis;
        if (pnh_.getParam("optimizer_comparison_cone_axis_world", requested_cone_axis) &&
            requested_cone_axis.size() == 3)
        {
            requested_cone_axis_eigen =
                Eigen::Vector3d(
                    requested_cone_axis[0],
                    requested_cone_axis[1],
                    requested_cone_axis[2]);
        }
        double requested_cone_half_angle =
            optimizer_comparison_cone_half_angle_rad_;
        int requested_cone_ring_count =
            optimizer_comparison_cone_ring_count_;
        int requested_cone_azimuth_count =
            optimizer_comparison_cone_azimuth_count_;
        bool requested_diagnostics_enabled =
            optimizer_dynamic_residual_diagnostics_enabled_;
        std::string requested_diagnostics_directory =
            optimizer_dynamic_residual_diagnostics_directory_;
        int requested_diagnostics_max_records =
            optimizer_dynamic_residual_diagnostics_max_records_;
        pnh_.param(
            "optimizer_comparison_cone_half_angle_rad",
            requested_cone_half_angle,
            requested_cone_half_angle);
        pnh_.param(
            "optimizer_comparison_cone_ring_count",
            requested_cone_ring_count,
            requested_cone_ring_count);
        pnh_.param(
            "optimizer_comparison_cone_azimuth_count",
            requested_cone_azimuth_count,
            requested_cone_azimuth_count);
        pnh_.param(
            "optimizer_use_dynamic_residual_force_polytope",
            requested_dynamic_residual,
            requested_dynamic_residual);
        pnh_.param(
            "optimizer_dynamic_residual_diagnostics_enabled",
            requested_diagnostics_enabled,
            requested_diagnostics_enabled);
        pnh_.param<std::string>(
            "optimizer_dynamic_residual_diagnostics_directory",
            requested_diagnostics_directory,
            requested_diagnostics_directory);
        pnh_.param(
            "optimizer_dynamic_residual_diagnostics_max_records",
            requested_diagnostics_max_records,
            requested_diagnostics_max_records);
        requested_diagnostics_directory =
            resolveRosPath(requested_diagnostics_directory);

        const std::string normalized_algorithm =
            polytope_wx::rfpComparisonAlgorithmName(
                polytope_wx::parseRfpComparisonAlgorithmOrOurs(
                    requested_algorithm));
        if (normalized_algorithm != optimizer_comparison_algorithm_ ||
            requested_dynamic_residual != optimizer_use_dynamic_residual_force_polytope_ ||
            (requested_cone_axis_eigen - optimizer_comparison_cone_axis_world_).norm() > 1e-12 ||
            std::abs(requested_cone_half_angle - optimizer_comparison_cone_half_angle_rad_) > 1e-12 ||
            requested_cone_ring_count != optimizer_comparison_cone_ring_count_ ||
            requested_cone_azimuth_count != optimizer_comparison_cone_azimuth_count_ ||
            requested_diagnostics_enabled != optimizer_dynamic_residual_diagnostics_enabled_ ||
            requested_diagnostics_directory != optimizer_dynamic_residual_diagnostics_directory_ ||
            requested_diagnostics_max_records != optimizer_dynamic_residual_diagnostics_max_records_)
        {
            optimizer_comparison_algorithm_ = normalized_algorithm;
            optimizer_use_dynamic_residual_force_polytope_ = requested_dynamic_residual;
            optimizer_comparison_cone_axis_world_ = requested_cone_axis_eigen;
            optimizer_comparison_cone_half_angle_rad_ = requested_cone_half_angle;
            optimizer_comparison_cone_ring_count_ = requested_cone_ring_count;
            optimizer_comparison_cone_azimuth_count_ = requested_cone_azimuth_count;
            optimizer_dynamic_residual_diagnostics_enabled_ =
                requested_diagnostics_enabled;
            optimizer_dynamic_residual_diagnostics_directory_ =
                requested_diagnostics_directory;
            optimizer_dynamic_residual_diagnostics_max_records_ =
                requested_diagnostics_max_records;
            optimizer_reinitialization_requested_ = true;
        }
    }

    void maybeStartTrajectory()
    {
        if (trajectory_started_ || !planning_start_requested_)
        {
            return;
        }

        if (!has_seed_pose_ || !has_seed_joint_positions_)
        {
            return;
        }

        if (isManualTargetPoseMode() && !has_manual_target_pose_)
        {
            return;
        }

        if (isReturnToHomeMode() && (!has_home_pose_ || !has_home_joint_positions_))
        {
            return;
        }

        if (enableVlmMassRequest() && !has_estimated_payload_mass_)
        {
            if (maybeRequestVlmMassEstimate())
            {
                return;
            }
        }

        if (!isJointQuinticInterpolationAlgorithm() &&
            comparison_force_sweep_enabled_ &&
            use_nullspace_joint_target_ &&
            !precomputeComparisonPlans())
        {
            ROS_WARN(
                "target_pose_generator: failed to precompute comparison trajectories; "
                "falling back to on-demand planning.");
        }

        trajectory_start_time_ = ros::Time::now();
        trajectory_started_ = true;
        planning_start_requested_ = false;
        phase_start_time_ = trajectory_start_time_;
        experiment_phase_ = ExperimentPhase::kTrajectory;
        comparison_run_index_ = 0;
        pre_trajectory_reset_completed_ = false;
        trajectory_seed_pose_ = seed_pose_;
        trajectory_seed_base_planar_positions_ = seed_base_planar_positions_;
        trajectory_seed_arm_joint_positions_ = seed_arm_joint_positions_;
        if (isManualTargetPoseMode())
        {
            trajectory_manual_target_pose_ = manual_target_pose_;
        }

        ROS_INFO_STREAM(
            "target_pose_generator: starting trajectory mode '" << trajectory_mode_ << "'.");
        startTrajectoryRun();
    }

    bool enableVlmMassRequest() const
    {
        return enable_vlm_mass_request_ && request_vlm_mass_for_pending_start_;
    }

    bool isSupportedTrajectoryMode(const std::string& trajectory_mode) const
    {
        return trajectory_mode == "payload_lift" ||
               trajectory_mode == "circle" ||
               trajectory_mode == "test_lift" ||
               trajectory_mode == "manual_target_pose" ||
               trajectory_mode == kSinglePointOptimizationTrajectoryMode ||
               trajectory_mode == kReturnToHomeTrajectoryMode ||
               trajectory_mode == kTeachingReturnInitialTrajectoryMode ||
               trajectory_mode == kTeachingWaypointsTrajectoryMode;
    }

    bool shouldRequestVlmMassForTrajectoryMode(const std::string& trajectory_mode) const
    {
        return trajectory_mode != kReturnToHomeTrajectoryMode &&
               trajectory_mode != kTeachingReturnInitialTrajectoryMode;
    }

    void resetPlanningStateForNewRequest(bool clear_cached_mass)
    {
        trajectory_started_ = false;
        plan_ready_ = false;
        pre_trajectory_reset_completed_ = false;
        reset_to_home_requested_ = false;
        waiting_for_vlm_mass_ = false;
        active_vlm_request_id_.clear();
        pending_vlm_request_ids_.clear();
        last_vlm_mass_error_.clear();
        last_plan_failure_reason_.clear();
        planned_waypoint_times_.clear();
        planned_whole_body_targets_.clear();
        planned_waypoint_poses_.clear();
        last_optimized_model_configuration_.resize(0);
        comparison_run_index_ = 0;
        experiment_phase_ = ExperimentPhase::kTrajectory;
        trajectory_start_time_ = ros::Time(0);
        phase_start_time_ = ros::Time(0);
        return_start_pose_affine_ = Eigen::Affine3d::Identity();
        return_start_whole_body_target_.setZero();
        comparison_plan_ready_.fill(false);

        for (std::size_t i = 0; i < comparison_planned_waypoint_times_.size(); ++i)
        {
            comparison_planned_waypoint_times_[i].clear();
            comparison_planned_whole_body_targets_[i].clear();
            comparison_last_optimized_model_configuration_[i].resize(0);
        }

        if (clear_cached_mass)
        {
            estimated_payload_mass_kg_ = default_payload_mass_kg_;
            has_estimated_payload_mass_ = false;
            last_vlm_result_json_path_.clear();
            last_vlm_model_.clear();
        }
    }

    bool startPlanningServiceCallback(
        moca_trajectory_generator::StartPlanning::Request& req,
        moca_trajectory_generator::StartPlanning::Response& res)
    {
        const std::string requested_mode =
            req.trajectory_mode.empty() ? trajectory_mode_ : req.trajectory_mode;
        refreshTeachingTrajectoryParamsFromServer();
        refreshVlmRuntimeParamsFromServer();
        refreshOptimizerComparisonParamsFromServer();
        if (!isSupportedTrajectoryMode(requested_mode))
        {
            res.success = false;
            res.message =
                "Unsupported trajectory_mode '" + requested_mode +
                "'. Expected one of: payload_lift, circle, test_lift, manual_target_pose, single_point_optimization, return_to_home, teaching_return_initial, teaching_waypoints.";
            ROS_WARN_STREAM("target_pose_generator: " << res.message);
            return true;
        }

        if (req.comparison_force_sweep_enabled && requested_mode != "payload_lift")
        {
            res.success = false;
            res.message =
                "Two-run comparison is currently only supported for trajectory_mode='payload_lift'.";
            ROS_WARN_STREAM("target_pose_generator: " << res.message);
            return true;
        }

        trajectory_mode_ = requested_mode;
        if (requested_mode == kSinglePointOptimizationTrajectoryMode)
        {
            refreshSinglePointTargetParamsFromServer();
        }
        comparison_force_sweep_enabled_ = req.comparison_force_sweep_enabled;
        const bool force_vlm_mass_request =
            shouldRequestVlmMassForTrajectoryMode(requested_mode);
        request_vlm_mass_for_pending_start_ =
            enable_vlm_mass_request_ && force_vlm_mass_request;
        if (force_vlm_mass_request && !req.request_vlm_mass_estimate)
        {
            ROS_INFO_STREAM(
                "target_pose_generator: forcing VLM mass request for trajectory_mode='"
                << requested_mode << "'.");
        }
        if (force_vlm_mass_request && !enable_vlm_mass_request_)
        {
            ROS_WARN(
                "target_pose_generator: this trajectory mode requires a VLM mass request, "
                "but VLM mass requests are disabled by parameter; reusing the stored/default mass.");
        }

        resetPlanningStateForNewRequest(request_vlm_mass_for_pending_start_);
        planning_start_requested_ = true;

        std::ostringstream message_stream;
        message_stream
            << "accepted planning request: trajectory_mode='" << trajectory_mode_ << "'"
            << ", comparison_force_sweep_enabled="
            << (comparison_force_sweep_enabled_ ? "true" : "false")
            << ", optimizer_comparison_algorithm='"
            << optimizer_comparison_algorithm_ << "'"
            << ", request_vlm_mass_estimate="
            << (request_vlm_mass_for_pending_start_ ? "true" : "false");
        if (!has_seed_pose_ || !has_seed_joint_positions_)
        {
            message_stream << "; waiting for current seed pose/joint state before starting.";
        }
        else if (requested_mode == "manual_target_pose" && !has_manual_target_pose_)
        {
            message_stream << "; waiting for the current manual target pose before starting.";
        }
        else if (requested_mode == kReturnToHomeTrajectoryMode &&
                 (!has_home_pose_ || !has_home_joint_positions_))
        {
            message_stream << "; waiting for the initial home pose/state snapshot before starting.";
        }
        else if (request_vlm_mass_for_pending_start_)
        {
            message_stream << "; waiting for a fresh VLM mass estimate before planning.";
        }
        else
        {
            message_stream << "; starting immediately with the current seed state.";
        }

        res.success = true;
        res.message = message_stream.str();
        ROS_INFO_STREAM("target_pose_generator: " << res.message);

        maybeStartTrajectory();
        return true;
    }

    void vlmMassResultCallback(const moca_vlm::MassEstimateResultConstPtr& msg)
    {
        if (!enableVlmMassRequest())
        {
            return;
        }

        const bool is_pending_request =
            pending_vlm_request_ids_.find(msg->request_id) != pending_vlm_request_ids_.end();
        if (active_vlm_request_id_.empty() || !is_pending_request)
        {
            return;
        }

        pending_vlm_request_ids_.erase(msg->request_id);
        last_vlm_result_json_path_ = msg->result_json_path;
        last_vlm_model_ = msg->model;

        if (!msg->success)
        {
            last_vlm_mass_error_ = msg->error_message;
            ROS_WARN_STREAM(
                "target_pose_generator: VLM mass request '" << msg->request_id
                << "' failed: " << msg->error_message);
            if (pending_vlm_request_ids_.empty())
            {
                useDefaultPayloadMassAfterVlmFailure(
                    "VLM mass request '" + msg->request_id + "' failed: " +
                    msg->error_message);
                maybeStartTrajectory();
            }
            return;
        }

        waiting_for_vlm_mass_ = false;
        active_vlm_request_id_.clear();
        pending_vlm_request_ids_.clear();
        estimated_payload_mass_kg_ = msg->mass_kg;
        has_estimated_payload_mass_ = true;
        last_vlm_mass_error_.clear();

        ROS_INFO_STREAM(
            "target_pose_generator: received VLM mass estimate "
            << estimated_payload_mass_kg_ << " kg from model '"
            << last_vlm_model_ << "' (json="
            << last_vlm_result_json_path_ << ").");

        maybeStartTrajectory();
    }

    void frankaToolStateCallback(const hrii_robot_msgs::FrankaToolStateConstPtr& msg)
    {
        latest_franka_tool_state_ = *msg;
        has_franka_tool_state_ = true;
        ROS_INFO_STREAM_THROTTLE(
            5.0,
            "target_pose_generator: received Franka tool/load state from '"
                << franka_tool_state_topic_
                << "' m_total=" << msg->m_total
                << " kg, m_load=" << msg->m_load
                << " kg.");
    }

    bool maybeRequestVlmMassEstimate()
    {
        const ros::Time now = ros::Time::now();
        if (waiting_for_vlm_mass_)
        {
            const double elapsed_sec = (now - vlm_mass_request_time_).toSec();
            if (elapsed_sec < vlm_mass_request_timeout_sec_)
            {
                ROS_WARN_THROTTLE(
                    1.0,
                    "target_pose_generator: waiting for VLM mass estimate before starting offline planning.");
                return true;
            }

            if (!vlm_timeout_uses_default_payload_mass_)
            {
                ROS_WARN_THROTTLE(
                    2.0,
                    "target_pose_generator: VLM mass request has been pending for %.3f s, exceeding timeout %.3f s; keeping planner blocked because vlm_timeout_uses_default_payload_mass=false.",
                    elapsed_sec,
                    vlm_mass_request_timeout_sec_);
                return true;
            }

            useDefaultPayloadMassAfterVlmFailure(
                "VLM mass request timed out after " + std::to_string(elapsed_sec) + " s");
            return false;
        }

        return requestVlmMassEstimate();
    }

    bool requestVlmMassEstimate()
    {
        refreshVlmRuntimeParamsFromServer();

        if (mass_request_pub_.getNumSubscribers() == 0)
        {
            useDefaultPayloadMassAfterVlmFailure(
                "no subscribers on VLM mass request topic '" + vlm_mass_request_topic_ + "'");
            return false;
        }

        moca_vlm::MassEstimateRequest request;
        request.header.stamp = ros::Time::now();
        request.header.frame_id = world_frame_;

        std::ostringstream request_stream;
        request_stream << "vlm_mass_request_" << request.header.stamp.toNSec();
        request.request_id = request_stream.str();
        request.image_path = vlm_request_image_path_;
        request.force_refresh = vlm_mass_request_force_refresh_;

        active_vlm_request_id_ = request.request_id;
        pending_vlm_request_ids_.insert(request.request_id);
        waiting_for_vlm_mass_ = true;
        vlm_mass_request_time_ = request.header.stamp;
        last_vlm_mass_error_.clear();

        mass_request_pub_.publish(request);
        ROS_INFO_STREAM(
            "target_pose_generator: requested VLM mass estimate with request id '"
            << active_vlm_request_id_ << "' using image path '"
            << vlm_request_image_path_ << "'.");
        return true;
    }

    void useDefaultPayloadMassAfterVlmFailure(const std::string& reason)
    {
        waiting_for_vlm_mass_ = false;
        active_vlm_request_id_.clear();
        pending_vlm_request_ids_.clear();
        request_vlm_mass_for_pending_start_ = false;
        estimated_payload_mass_kg_ = default_payload_mass_kg_;
        has_estimated_payload_mass_ = true;
        last_vlm_mass_error_ = reason;
        ROS_WARN_STREAM(
            "target_pose_generator: " << reason
            << "; using default payload mass "
            << default_payload_mass_kg_ << " kg for this planning request.");
    }

    bool initializeOptimizer()
    {
        const bool single_point_optimizer_mode = isSinglePointOptimizationMode();
        if (optimizer_initialized_ &&
            !optimizer_reinitialization_requested_ &&
            optimizer_initialized_for_single_point_mode_ == single_point_optimizer_mode)
        {
            return true;
        }

        polytope_wx::WholeBodyOptimizationConfig config;
        config.urdf_path = polytope_urdf_path_;
        config.base_frame = "moca_base_footprint";
        config.ee_frame = polytope_ee_frame_;
        config.locked_joint_names = {
            left_finger_joint_name_,
            right_finger_joint_name_
        };
        if (optimizer_arm_torque_limits_override_.size() == kArmJointCount)
        {
            config.arm_torque_limits = optimizer_arm_torque_limits_override_;
        }
        config.max_iterations_per_waypoint =
            single_point_optimizer_mode
                ? single_point_optimizer_max_iterations_per_waypoint_
                : optimizer_max_iterations_per_waypoint_;
        config.max_trajectory_iterations =
            single_point_optimizer_mode
                ? single_point_optimizer_max_trajectory_iterations_
                : optimizer_max_trajectory_iterations_;
        config.pose_gain =
            single_point_optimizer_mode ? single_point_optimizer_pose_gain_ : optimizer_pose_gain_;
        config.nullspace_step_size =
            single_point_optimizer_mode
                ? single_point_optimizer_nullspace_step_size_
                : optimizer_nullspace_step_size_;
        config.max_state_update_norm =
            single_point_optimizer_mode
                ? single_point_optimizer_max_joint_update_norm_
                : optimizer_max_joint_update_norm_;
        config.pose_tolerance =
            single_point_optimizer_mode
                ? single_point_optimizer_pose_tolerance_
                : optimizer_pose_tolerance_;
        const polytope_wx::RfpComparisonAlgorithm comparison_algorithm =
            polytope_wx::parseRfpComparisonAlgorithmOrOurs(
                optimizer_comparison_algorithm_);
        const OptimizerCostWeights active_cost_weights =
            single_point_optimizer_mode
                ? buildSinglePointOptimizerCostWeights()
                : optimizerCostWeightsForAlgorithm(comparison_algorithm);

        config.capability_weight = active_cost_weights.capability_weight;
        config.manipulability_weight = active_cost_weights.manipulability_weight;
        config.joint_limit_weight = active_cost_weights.joint_limit_weight;
        config.joint_limit_dof_weights =
            active_cost_weights.joint_limit_dof_weights;
        config.smoothness_weight = active_cost_weights.smoothness_weight;
        config.velocity_weight = active_cost_weights.velocity_weight;
        config.nominal_weight = active_cost_weights.nominal_weight;
        config.base_smoothness_weight =
            active_cost_weights.base_smoothness_weight;
        config.base_velocity_weight = active_cost_weights.base_velocity_weight;
        config.base_nominal_weight = active_cost_weights.base_nominal_weight;
        config.base_spectral_energy_weight =
            active_cost_weights.base_spectral_energy_weight;
        config.base_spectral_energy_metric <<
            (single_point_optimizer_mode
                 ? single_point_optimizer_base_spectral_metric_x_
                 : optimizer_base_spectral_metric_x_),
            (single_point_optimizer_mode
                 ? single_point_optimizer_base_spectral_metric_y_
                 : optimizer_base_spectral_metric_y_),
            (single_point_optimizer_mode
                 ? single_point_optimizer_base_spectral_metric_yaw_
                 : optimizer_base_spectral_metric_yaw_);
        config.base_spectral_cutoff_ratio =
            single_point_optimizer_mode
                ? single_point_optimizer_base_spectral_cutoff_ratio_
                : optimizer_base_spectral_cutoff_ratio_;
        config.base_spectral_power =
            single_point_optimizer_mode
                ? single_point_optimizer_base_spectral_power_
                : optimizer_base_spectral_power_;
        config.collision_weight = active_cost_weights.collision_weight;
        config.collision_safe_distance =
            single_point_optimizer_mode
                ? single_point_optimizer_collision_safe_distance_
                : optimizer_collision_safe_distance_;
        config.collision_basket_spheres =
            single_point_optimizer_mode
                ? single_point_optimizer_collision_basket_spheres_
                : optimizer_collision_basket_spheres_;
        config.collision_body_spheres =
            single_point_optimizer_mode
                ? single_point_optimizer_collision_body_spheres_
                : optimizer_collision_body_spheres_;
        config.capability_alpha =
            single_point_optimizer_mode
                ? single_point_optimizer_capability_alpha_
                : optimizer_capability_alpha_;
        config.capability_exponent_limit_enabled =
            single_point_optimizer_mode
                ? single_point_optimizer_capability_exponent_limit_enabled_
                : optimizer_capability_exponent_limit_enabled_;
        config.comparison_algorithm = comparison_algorithm;
        config.comparison_cone_axis_world = optimizer_comparison_cone_axis_world_;
        config.comparison_cone_half_angle_rad =
            optimizer_comparison_cone_half_angle_rad_;
        config.comparison_cone_ring_count =
            optimizer_comparison_cone_ring_count_;
        config.comparison_cone_azimuth_count =
            optimizer_comparison_cone_azimuth_count_;
        config.use_dynamic_residual_force_polytope =
            single_point_optimizer_mode
                ? single_point_optimizer_use_dynamic_residual_force_polytope_
                : optimizer_use_dynamic_residual_force_polytope_;
        config.dynamic_residual_diagnostics_enabled =
            !single_point_optimizer_mode &&
            optimizer_dynamic_residual_diagnostics_enabled_;
        config.dynamic_residual_diagnostics_directory =
            optimizer_dynamic_residual_diagnostics_directory_;
        config.dynamic_residual_diagnostics_max_records =
            optimizer_dynamic_residual_diagnostics_max_records_;
        config.capability_near_weight =
            active_cost_weights.capability_near_weight;
        config.capability_over_weight =
            active_cost_weights.capability_over_weight;
        config.capability_over4_weight =
            active_cost_weights.capability_over4_weight;
        config.constrain_orientation =
            single_point_optimizer_mode
                ? single_point_optimizer_constrain_orientation_
                : optimizer_constrain_orientation_;
        config.verbose =
            single_point_optimizer_mode ? single_point_optimizer_verbose_ : optimizer_verbose_;

        if (!whole_body_optimizer_.initialize(config))
        {
            return false;
        }

        optimizer_state_dim_ = whole_body_optimizer_.stateDim();
        if (optimizer_state_dim_ < kBaseJointCount + kArmJointCount)
        {
            ROS_ERROR(
                "target_pose_generator: optimizer state dimension is %d, but at least %d are required.",
                optimizer_state_dim_,
                kArmJointCount);
            return false;
        }

        optimizer_initialized_ = true;
        optimizer_initialized_for_single_point_mode_ = single_point_optimizer_mode;
        optimizer_reinitialization_requested_ = false;
        ROS_INFO_STREAM(
            "target_pose_generator: whole-body polytope optimizer initialized for frame " <<
            polytope_ee_frame_ << " with state dim = " << optimizer_state_dim_
            << (single_point_optimizer_mode ? " (single-point params)." : " (trajectory params)."));
        return true;
    }

    Eigen::VectorXd buildNominalStateConfiguration() const
    {
        Eigen::VectorXd nominal_state = Eigen::VectorXd::Zero(optimizer_state_dim_);

        for (int i = 0; i < kBaseJointCount; ++i)
        {
            nominal_state(i) = trajectory_seed_base_planar_positions_[i];
        }

        for (int i = 0; i < std::min(kArmJointCount, optimizer_state_dim_ - kBaseJointCount); ++i)
        {
            nominal_state(kBaseJointCount + i) = trajectory_seed_arm_joint_positions_[i];
        }

        return nominal_state;
    }

    void logPlannedWholeBodyTrajectorySegment() const
    {
        if (!log_optimization_info_)
        {
            return;
        }

        if (planned_waypoint_times_.empty() || planned_whole_body_targets_.empty())
        {
            ROS_WARN("target_pose_generator: no optimized whole-body trajectory is available to print.");
            return;
        }

        std::ostringstream stream;
        stream << std::fixed << std::setprecision(4);
        stream << "target_pose_generator: optimized whole-body trajectory";

        for (std::size_t waypoint_index = 0;
             waypoint_index < planned_whole_body_targets_.size() &&
             waypoint_index < planned_waypoint_times_.size();
             ++waypoint_index)
        {
            stream << "\n  wp[" << waypoint_index << "]"
                   << " t=" << planned_waypoint_times_[waypoint_index] << "s"
                   << " base=["
                   << planned_whole_body_targets_[waypoint_index](0) << ", "
                   << planned_whole_body_targets_[waypoint_index](1) << ", "
                   << planned_whole_body_targets_[waypoint_index](2) << "]"
                   << " arm=[";

            for (int joint_index = 0; joint_index < kArmJointCount; ++joint_index)
            {
                stream << planned_whole_body_targets_[waypoint_index](kBaseJointCount + joint_index);
                if (joint_index + 1 < kArmJointCount)
                {
                    stream << ", ";
                }
            }

            stream << "]";
        }

        ROS_INFO_STREAM(stream.str());
    }

    bool writePlannerMetricsCsv(
        const std::string& csv_path,
        const polytope_wx::WholeBodyTrajectoryOptimizationInput& input,
        const polytope_wx::WholeBodyTrajectoryOptimizationResult& result) const
    {
        if (csv_path.empty())
        {
            ROS_WARN("target_pose_generator: planner metrics CSV path is empty; skipping export.");
            return false;
        }

        const std::size_t waypoint_count =
            std::min(
                input.target_poses.size(),
                std::min(
                    result.optimized_state_trajectory.size(),
                    result.waypoint_metrics.size()));
        if (waypoint_count == 0)
        {
            ROS_WARN("target_pose_generator: planner metrics export skipped because no waypoints are available.");
            return false;
        }

        if (!ensureParentDirectoryForFile(csv_path))
        {
            ROS_WARN_STREAM(
                "target_pose_generator: failed to create planner metrics output directory for "
                << csv_path << ": " << std::strerror(errno));
            return false;
        }

        std::ofstream csv_file(csv_path);
        if (!csv_file.is_open())
        {
            ROS_WARN_STREAM(
                "target_pose_generator: failed to open planner metrics CSV " << csv_path);
            return false;
        }

        const auto& config = whole_body_optimizer_.config();
        const double weighted_base_spectral_energy =
            config.base_spectral_energy_weight * result.base_spectral_energy_cost;
        std::vector<polytope_wx::WholeBodyWaypointOptimizationMetrics> nominal_metrics;
        nominal_metrics.reserve(waypoint_count);
        const double nominal_base_spectral_energy =
            whole_body_optimizer_.computeTrajectoryBaseSpectralEnergyCost(
                input.nominal_state_trajectory,
                input.waypoint_times_sec);

        for (std::size_t i = 0; i < waypoint_count; ++i)
        {
            const Eigen::VectorXd* previous_nominal_state =
                i > 0 ? &input.nominal_state_trajectory[i - 1] : nullptr;
            const double previous_dt_sec =
                i > 0 && i < input.waypoint_times_sec.size()
                    ? std::max(1e-9, input.waypoint_times_sec[i] - input.waypoint_times_sec[i - 1])
                    : 1.0;
            nominal_metrics.push_back(
                whole_body_optimizer_.evaluateWaypointMetrics(
                    input.nominal_state_trajectory[i],
                    input.target_poses[i],
                    input.desired_forces[i],
                    i < input.desired_force_radii.size() ? input.desired_force_radii[i] : 0.0,
                    input.nominal_state_trajectory[i],
                    previous_nominal_state,
                    previous_dt_sec));
        }

        csv_file
            << "time_sec,"
            << "target_x,target_y,target_z,"
            << "optimized_ee_x,optimized_ee_y,optimized_ee_z,"
            << "base_x,base_y,base_yaw,"
            << "before_pose_error_norm,before_required_force,before_required_force_radius,before_force_capacity,before_force_ball_clearance,before_objective_value,"
            << "before_capability_cost,before_manipulability_measure,before_manipulability_cost,"
            << "before_joint_limit_cost,before_smoothness_cost,before_velocity_cost,before_nominal_cost,"
            << "before_base_smoothness_cost,before_base_velocity_cost,before_base_nominal_cost,before_base_spectral_energy_cost,"
            << "before_collision_cost,before_min_collision_clearance,"
            << "pose_error_norm,required_force,required_force_radius,force_capacity,force_ball_clearance,objective_value,"
            << "capability_cost,manipulability_measure,manipulability_cost,"
            << "joint_limit_cost,smoothness_cost,velocity_cost,nominal_cost,"
            << "base_smoothness_cost,base_velocity_cost,base_nominal_cost,base_spectral_energy_cost,"
            << "collision_cost,min_collision_clearance,"
            << "weighted_capability,weighted_manipulability,weighted_joint_limit,"
            << "weighted_smoothness,weighted_velocity,weighted_nominal,"
            << "weighted_base_smoothness,weighted_base_velocity,weighted_base_nominal,"
            << "weighted_collision,weighted_base_spectral_energy";
        for (int i = 0; i < kArmJointCount; ++i)
        {
            csv_file << ",arm_q" << (i + 1);
        }
        csv_file << "\n";

        for (std::size_t i = 0; i < waypoint_count; ++i)
        {
            const double time_sec =
                i < input.waypoint_times_sec.size()
                    ? input.waypoint_times_sec[i]
                    : static_cast<double>(i);
            const Eigen::VectorXd& optimized_state = result.optimized_state_trajectory[i];
            const Eigen::Affine3d optimized_pose =
                whole_body_optimizer_.computeEndEffectorPose(optimized_state);
            const auto& metrics = result.waypoint_metrics[i];
            const auto& before_metrics = nominal_metrics[i];
            const double weighted_capability =
                config.capability_weight * metrics.capability_cost;
            const double weighted_manipulability =
                config.manipulability_weight * metrics.manipulability_cost;
            const double weighted_joint_limit =
                config.joint_limit_weight * metrics.joint_limit_cost;
            const double weighted_smoothness =
                config.smoothness_weight * metrics.smoothness_cost;
            const double weighted_velocity =
                config.velocity_weight * metrics.velocity_cost;
            const double weighted_nominal =
                config.nominal_weight * metrics.nominal_cost;
            const double weighted_base_smoothness =
                config.base_smoothness_weight * metrics.base_smoothness_cost;
            const double weighted_base_velocity =
                config.base_velocity_weight * metrics.base_velocity_cost;
            const double weighted_base_nominal =
                config.base_nominal_weight * metrics.base_nominal_cost;
            const double weighted_collision =
                config.collision_weight * metrics.collision_cost;

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
                << before_metrics.pose_error_norm << ","
                << before_metrics.required_force << ","
                << before_metrics.required_force_radius << ","
                << before_metrics.force_capacity << ","
                << before_metrics.force_ball_clearance << ","
                << before_metrics.objective_value << ","
                << before_metrics.capability_cost << ","
                << before_metrics.manipulability_measure << ","
                << before_metrics.manipulability_cost << ","
                << before_metrics.joint_limit_cost << ","
                << before_metrics.smoothness_cost << ","
                << before_metrics.velocity_cost << ","
                << before_metrics.nominal_cost << ","
                << before_metrics.base_smoothness_cost << ","
                << before_metrics.base_velocity_cost << ","
                << before_metrics.base_nominal_cost << ","
                << nominal_base_spectral_energy << ","
                << before_metrics.collision_cost << ","
                << before_metrics.min_collision_clearance << ","
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
                << metrics.velocity_cost << ","
                << metrics.nominal_cost << ","
                << metrics.base_smoothness_cost << ","
                << metrics.base_velocity_cost << ","
                << metrics.base_nominal_cost << ","
                << result.base_spectral_energy_cost << ","
                << metrics.collision_cost << ","
                << metrics.min_collision_clearance << ","
                << weighted_capability << ","
                << weighted_manipulability << ","
                << weighted_joint_limit << ","
                << weighted_smoothness << ","
                << weighted_velocity << ","
                << weighted_nominal << ","
                << weighted_base_smoothness << ","
                << weighted_base_velocity << ","
                << weighted_base_nominal << ","
                << weighted_collision << ","
                << weighted_base_spectral_energy;

            for (int joint_index = 0; joint_index < kArmJointCount; ++joint_index)
            {
                csv_file << "," << optimized_state(kBaseJointCount + joint_index);
            }
            csv_file << "\n";
        }

        csv_file.close();
        return true;
    }

    void maybeExportPlannerMetricsCsv(
        const polytope_wx::WholeBodyTrajectoryOptimizationInput& input,
        const polytope_wx::WholeBodyTrajectoryOptimizationResult& result) const
    {
        if (!export_planner_metrics_)
        {
            return;
        }

        const bool wrote_primary =
            writePlannerMetricsCsv(planner_metrics_csv_path_, input, result);
        if (wrote_primary)
        {
            ROS_INFO_STREAM(
                "target_pose_generator: planner metrics CSV written to " <<
                planner_metrics_csv_path_);
        }

        if (!comparison_force_sweep_enabled_)
        {
            return;
        }

        const std::string run_specific_path = appendSuffixBeforeExtension(
            planner_metrics_csv_path_,
            "_run" + std::to_string(comparison_run_index_ + 1));
        if (run_specific_path == planner_metrics_csv_path_)
        {
            return;
        }

        if (writePlannerMetricsCsv(run_specific_path, input, result))
        {
            ROS_INFO_STREAM(
                "target_pose_generator: planner metrics CSV written to " <<
                run_specific_path);
        }
    }

    static double clampUnit(double value)
    {
        return std::max(0.0, std::min(1.0, value));
    }

    static double smoothStep(double value)
    {
        const double clamped = clampUnit(value);
        return clamped * clamped * (3.0 - 2.0 * clamped);
    }

    static double smoothStepFirstDerivative(double value)
    {
        if (value <= 0.0 || value >= 1.0)
        {
            return 0.0;
        }

        return 6.0 * value * (1.0 - value);
    }

    static double smoothStepSecondDerivative(double value)
    {
        if (value <= 0.0 || value >= 1.0)
        {
            return 0.0;
        }

        return 6.0 - 12.0 * value;
    }

    static bool ensureParentDirectoryForFile(const std::string& file_path)
    {
        const std::size_t separator = file_path.find_last_of('/');
        if (separator == std::string::npos)
        {
            return true;
        }

        const std::string directory = file_path.substr(0, separator);
        if (directory.empty())
        {
            return true;
        }

        std::size_t pos = 0;
        while (pos < directory.size())
        {
            pos = directory.find_first_not_of('/', pos);
            if (pos == std::string::npos)
            {
                break;
            }

            const std::size_t next_separator = directory.find('/', pos);
            const std::string partial_directory =
                directory.substr(
                    0,
                    next_separator == std::string::npos ? directory.size() : next_separator);

            if (::mkdir(partial_directory.c_str(), 0777) != 0 && errno != EEXIST)
            {
                return false;
            }

            if (next_separator == std::string::npos)
            {
                break;
            }

            pos = next_separator + 1;
        }

        return true;
    }

    static std::string appendSuffixBeforeExtension(
        const std::string& file_path,
        const std::string& suffix)
    {
        const std::size_t extension_pos = file_path.find_last_of('.');
        const std::size_t separator_pos = file_path.find_last_of('/');
        if (extension_pos == std::string::npos ||
            (separator_pos != std::string::npos && extension_pos < separator_pos))
        {
            return file_path + suffix;
        }

        return file_path.substr(0, extension_pos) +
               suffix +
               file_path.substr(extension_pos);
    }

    static bool parseCollisionSphereParam(
        const std::string& param_name,
        int index,
        XmlRpc::XmlRpcValue sphere_value,
        polytope_wx::CollisionSphereSpec* sphere)
    {
        if (sphere == nullptr)
        {
            return false;
        }
        if (sphere_value.getType() != XmlRpc::XmlRpcValue::TypeStruct)
        {
            ROS_WARN_STREAM(
                "target_pose_generator: " << param_name << "[" << index
                << "] must be a struct with frame/center/radius.");
            return false;
        }
        if (!sphere_value.hasMember("frame") ||
            !sphere_value.hasMember("center") ||
            !sphere_value.hasMember("radius"))
        {
            ROS_WARN_STREAM(
                "target_pose_generator: " << param_name << "[" << index
                << "] is missing frame, center, or radius.");
            return false;
        }

        XmlRpc::XmlRpcValue frame_value = sphere_value["frame"];
        if (frame_value.getType() != XmlRpc::XmlRpcValue::TypeString)
        {
            ROS_WARN_STREAM(
                "target_pose_generator: " << param_name << "[" << index
                << "].frame must be a string.");
            return false;
        }

        XmlRpc::XmlRpcValue center_value = sphere_value["center"];
        if (center_value.getType() != XmlRpc::XmlRpcValue::TypeArray ||
            center_value.size() != 3)
        {
            ROS_WARN_STREAM(
                "target_pose_generator: " << param_name << "[" << index
                << "].center must contain exactly 3 numeric values.");
            return false;
        }

        Eigen::Vector3d center = Eigen::Vector3d::Zero();
        for (int axis = 0; axis < 3; ++axis)
        {
            XmlRpc::XmlRpcValue coordinate_value = center_value[axis];
            double coordinate = 0.0;
            if (!xmlRpcValueToDouble(coordinate_value, &coordinate) ||
                !std::isfinite(coordinate))
            {
                ROS_WARN_STREAM(
                    "target_pose_generator: " << param_name << "[" << index
                    << "].center[" << axis << "] must be finite.");
                return false;
            }
            center(axis) = coordinate;
        }

        XmlRpc::XmlRpcValue radius_value = sphere_value["radius"];
        double radius = 0.0;
        if (!xmlRpcValueToDouble(radius_value, &radius) ||
            !std::isfinite(radius) ||
            radius <= 0.0)
        {
            ROS_WARN_STREAM(
                "target_pose_generator: " << param_name << "[" << index
                << "].radius must be a positive finite value.");
            return false;
        }

        sphere->frame = static_cast<std::string>(frame_value);
        sphere->center = center;
        sphere->radius = radius;
        return true;
    }

    bool loadCollisionSpheresParam(
        const std::string& param_name,
        std::vector<polytope_wx::CollisionSphereSpec>* spheres)
    {
        if (spheres == nullptr)
        {
            return false;
        }
        spheres->clear();

        XmlRpc::XmlRpcValue sphere_values;
        if (!pnh_.getParam(param_name, sphere_values))
        {
            return true;
        }
        if (sphere_values.getType() != XmlRpc::XmlRpcValue::TypeArray)
        {
            ROS_WARN_STREAM(
                "target_pose_generator: " << param_name
                << " must be a list of sphere structs.");
            return false;
        }

        bool all_valid = true;
        for (int i = 0; i < sphere_values.size(); ++i)
        {
            XmlRpc::XmlRpcValue sphere_value = sphere_values[i];
            polytope_wx::CollisionSphereSpec sphere;
            if (parseCollisionSphereParam(param_name, i, sphere_value, &sphere))
            {
                spheres->push_back(sphere);
            }
            else
            {
                all_valid = false;
            }
        }

        if (!spheres->empty())
        {
            ROS_INFO_STREAM(
                "target_pose_generator: loaded " << spheres->size()
                << " collision spheres from " << param_name << ".");
        }
        return all_valid;
    }

    static double quinticTimeScaling(double value)
    {
        const double clamped = clampUnit(value);
        const double c2 = clamped * clamped;
        const double c3 = c2 * clamped;
        const double c4 = c3 * clamped;
        const double c5 = c4 * clamped;
        return 10.0 * c3 - 15.0 * c4 + 6.0 * c5;
    }

    static double quinticTimeScalingFirstDerivative(double value)
    {
        if (value <= 0.0 || value >= 1.0)
        {
            return 0.0;
        }

        const double v2 = value * value;
        const double v3 = v2 * value;
        const double v4 = v3 * value;
        return 30.0 * v2 - 60.0 * v3 + 30.0 * v4;
    }

    static double quinticTimeScalingSecondDerivative(double value)
    {
        if (value <= 0.0 || value >= 1.0)
        {
            return 0.0;
        }

        const double v2 = value * value;
        const double v3 = v2 * value;
        return 60.0 * value - 180.0 * v2 + 120.0 * v3;
    }

    double computeActiveTime(double elapsed_time) const
    {
        return std::max(0.0, elapsed_time - trajectory_start_delay_sec_);
    }

    bool isCircleMode() const
    {
        return trajectory_mode_ == "circle";
    }

    bool isPayloadLiftMode() const
    {
        return trajectory_mode_ == "payload_lift";
    }

    bool isManualTargetPoseMode() const
    {
        return trajectory_mode_ == "manual_target_pose";
    }

    bool isSinglePointOptimizationMode() const
    {
        return trajectory_mode_ == kSinglePointOptimizationTrajectoryMode;
    }

    bool isReturnToHomeMode() const
    {
        return trajectory_mode_ == kReturnToHomeTrajectoryMode;
    }

    bool isTeachingReturnInitialMode() const
    {
        return trajectory_mode_ == kTeachingReturnInitialTrajectoryMode;
    }

    bool isTeachingWaypointsMode() const
    {
        return trajectory_mode_ == kTeachingWaypointsTrajectoryMode;
    }

    bool isTeachingFileTrajectoryMode() const
    {
        return isReturnToHomeMode() ||
               isTeachingReturnInitialMode() ||
               isTeachingWaypointsMode();
    }

    Eigen::Vector3d buildPayloadHandleWorldPosition() const
    {
        return Eigen::Vector3d(
            payload_box_center_x_,
            payload_box_center_y_,
            payload_handle_top_z_);
    }

    Eigen::Affine3d buildPayloadPregraspPoseAffine() const
    {
        Eigen::Affine3d pose = Eigen::Affine3d::Identity();
        pose.linear() = buildDownwardOrientation().toRotationMatrix();
        pose.translation() =
            buildPayloadHandleWorldPosition() +
            Eigen::Vector3d(0.0, 0.0, payload_grasp_ee_offset_z_ + payload_pregrasp_height_);
        return pose;
    }

    Eigen::Affine3d buildPayloadGraspPoseAffine() const
    {
        Eigen::Affine3d pose = Eigen::Affine3d::Identity();
        pose.linear() = buildDownwardOrientation().toRotationMatrix();
        pose.translation() =
            buildPayloadHandleWorldPosition() +
            Eigen::Vector3d(0.0, 0.0, payload_grasp_ee_offset_z_);
        return pose;
    }

    Eigen::Affine3d buildPayloadLiftPoseAffine() const
    {
        Eigen::Affine3d pose = buildPayloadGraspPoseAffine();
        pose.translation().z() += payload_lift_distance_z_;
        return pose;
    }

    Eigen::Affine3d buildManualTargetGoalPoseAffine() const
    {
        Eigen::Affine3d pose = Eigen::Affine3d::Identity();
        pose.translation() <<
            trajectory_manual_target_pose_.pose.position.x,
            trajectory_manual_target_pose_.pose.position.y,
            trajectory_manual_target_pose_.pose.position.z;

        const Eigen::Quaterniond target_orientation(
            trajectory_manual_target_pose_.pose.orientation.w,
            trajectory_manual_target_pose_.pose.orientation.x,
            trajectory_manual_target_pose_.pose.orientation.y,
            trajectory_manual_target_pose_.pose.orientation.z);
        pose.linear() = target_orientation.normalized().toRotationMatrix();
        return pose;
    }

    Eigen::Affine3d buildStoredHomePoseAffine() const
    {
        Eigen::Affine3d pose = Eigen::Affine3d::Identity();
        pose.translation() <<
            home_pose_.pose.position.x,
            home_pose_.pose.position.y,
            home_pose_.pose.position.z;

        const Eigen::Quaterniond home_orientation(
            home_pose_.pose.orientation.w,
            home_pose_.pose.orientation.x,
            home_pose_.pose.orientation.y,
            home_pose_.pose.orientation.z);
        pose.linear() = home_orientation.normalized().toRotationMatrix();
        return pose;
    }

    static void computeQuinticSegmentKinematics(
        const Eigen::Vector3d& start,
        const Eigen::Vector3d& goal,
        double elapsed_time,
        double duration,
        Eigen::Vector3d* position,
        Eigen::Vector3d* velocity,
        Eigen::Vector3d* acceleration)
    {
        const double clamped_duration = std::max(1e-6, duration);
        const double s = elapsed_time / clamped_duration;
        const double alpha = quinticTimeScaling(s);
        const double alpha_dot =
            quinticTimeScalingFirstDerivative(s) / clamped_duration;
        const double alpha_ddot =
            quinticTimeScalingSecondDerivative(s) /
            (clamped_duration * clamped_duration);
        const Eigen::Vector3d delta = goal - start;

        if (position != nullptr)
        {
            *position = start + alpha * delta;
        }
        if (velocity != nullptr)
        {
            *velocity = alpha_dot * delta;
        }
        if (acceleration != nullptr)
        {
            *acceleration = alpha_ddot * delta;
        }
    }

    double currentLiftDesiredForceMagnitude() const
    {
        return liftForceMagnitudeForRunIndex(comparison_run_index_);
    }

    double liftForceMagnitudeForRunIndex(int run_index) const
    {
        if (!comparison_force_sweep_enabled_)
        {
            return lift_desired_force_magnitude_;
        }

        return run_index <= 0 ? comparison_first_lift_force_magnitude_
                              : comparison_second_lift_force_magnitude_;
    }

    bool ensureOptimizerReady()
    {
        if (!use_nullspace_joint_target_)
        {
            ROS_INFO_ONCE(
                "target_pose_generator: external nullspace optimizer is disabled; "
                "publishing pose-only targets and letting the controller handle manipulability shaping.");
            return false;
        }

        if (!optimizer_initialized_ && !initializeOptimizer())
        {
            ROS_WARN(
                "target_pose_generator: polytope optimizer initialization failed, "
                "using seed joint targets only.");
            return false;
        }

        return optimizer_initialized_;
    }

    Eigen::Affine3d computeTeachingTargetPose(
        const WholeBodyTarget& target,
        const TeachingPoint* fallback_point) const
    {
        if (optimizer_initialized_)
        {
            try
            {
                return whole_body_optimizer_.computeEndEffectorPose(
                    buildOptimizerStateFromWholeBodyTarget(target));
            }
            catch (const std::exception& e)
            {
                ROS_WARN_STREAM_THROTTLE(
                    1.0,
                    "target_pose_generator: FK from teaching joint target failed: "
                    << e.what() << "; falling back to recorded ee_pose if available.");
            }
        }

        if (fallback_point != nullptr && fallback_point->has_recorded_pose)
        {
            return fallback_point->recorded_pose;
        }
        return buildSeedPoseAffine();
    }

    bool prepareTeachingFileTrajectory()
    {
        std::vector<TeachingPoint> file_points;
        if (isReturnToHomeMode())
        {
            if (!loadTeachingPoints(home_teaching_point_path_, &file_points))
            {
                return false;
            }
            if (file_points.size() > 1)
            {
                file_points.resize(1);
            }
        }
        else
        {
            if (!loadTeachingPoints(teaching_waypoints_path_, &file_points))
            {
                return false;
            }
            if (isTeachingReturnInitialMode() && file_points.size() > 1)
            {
                file_points.resize(1);
            }
        }

        ensureOptimizerReady();

        planned_waypoint_times_.clear();
        planned_whole_body_targets_.clear();
        planned_waypoint_poses_.clear();

        planned_waypoint_times_.push_back(trajectory_start_delay_sec_);
        planned_whole_body_targets_.push_back(fallbackWholeBodyTarget());
        planned_waypoint_poses_.push_back(buildSeedPoseAffine());

        const double segment_duration = std::max(
            1e-6,
            isReturnToHomeMode()
                ? home_return_duration_sec_
                : teaching_waypoint_segment_duration_sec_);
        double waypoint_time = trajectory_start_delay_sec_;
        for (std::size_t i = 0; i < file_points.size(); ++i)
        {
            waypoint_time += segment_duration;
            planned_waypoint_times_.push_back(waypoint_time);
            planned_whole_body_targets_.push_back(file_points[i].whole_body_target);
            planned_waypoint_poses_.push_back(
                computeTeachingTargetPose(file_points[i].whole_body_target, &file_points[i]));
        }

        plan_ready_ = true;
        ROS_INFO_STREAM(
            "target_pose_generator: loaded " << file_points.size()
            << " teaching waypoint(s) for mode '" << trajectory_mode_
            << "' from "
            << resolveRosPath(isReturnToHomeMode()
                                  ? home_teaching_point_path_
                                  : teaching_waypoints_path_)
            << ".");
        return true;
    }

    bool prepareSinglePointJointQuinticTrajectory()
    {
        if (!single_point_has_target_whole_body_)
        {
            last_plan_failure_reason_ =
                "joint_quintic_interpolation requires a selected single-point whole-body target.";
            ROS_WARN_STREAM("target_pose_generator: " << last_plan_failure_reason_);
            return false;
        }

        planned_waypoint_times_.clear();
        planned_whole_body_targets_.clear();
        planned_waypoint_poses_.clear();

        planned_waypoint_times_.push_back(single_point_move_duration_sec_);
        planned_whole_body_targets_.push_back(single_point_target_whole_body_);
        planned_waypoint_poses_.push_back(buildSinglePointGoalPoseAffine());
        last_optimized_model_configuration_ =
            buildOptimizerStateFromWholeBodyTarget(single_point_target_whole_body_);
        plan_ready_ = true;
        last_plan_failure_reason_.clear();

        ROS_INFO(
            "target_pose_generator: joint_quintic_interpolation selected for single-point mode; "
            "using selected whole-body target directly without optimizer.");
        return true;
    }

    void startTrajectoryRun()
    {
        trajectory_start_time_ = ros::Time::now();
        phase_start_time_ = trajectory_start_time_;
        experiment_phase_ = ExperimentPhase::kTrajectory;
        trajectory_start_desired_force_ =
            has_last_published_desired_force_
                ? last_published_desired_force_
                : Eigen::Vector3d::Zero();
        has_trajectory_start_desired_force_ = true;

        if (has_estimated_payload_mass_)
        {
            ROS_INFO_STREAM(
                "target_pose_generator: starting planning with stored VLM mass estimate "
                << estimated_payload_mass_kg_ << " kg.");
        }

        if (isReturnToHomeMode() || isTeachingReturnInitialMode())
        {
            if (!prepareTeachingFileTrajectory())
            {
                plan_ready_ = false;
                planned_waypoint_times_.clear();
                planned_whole_body_targets_.clear();
                planned_waypoint_poses_.clear();
            }
            return;
        }

        if (isTeachingWaypointsMode() && !prepareTeachingFileTrajectory())
        {
            plan_ready_ = false;
            planned_waypoint_times_.clear();
            planned_whole_body_targets_.clear();
            planned_waypoint_poses_.clear();
            return;
        }

        if (isTeachingWaypointsMode() && isJointQuinticInterpolationAlgorithm())
        {
            last_plan_failure_reason_.clear();
            ROS_INFO(
                "target_pose_generator: joint_quintic_interpolation selected; "
                "using teaching whole-body waypoints directly without optimizer.");
            return;
        }

        if (isSinglePointOptimizationMode() && isJointQuinticInterpolationAlgorithm())
        {
            if (!prepareSinglePointJointQuinticTrajectory())
            {
                plan_ready_ = false;
                planned_waypoint_times_.clear();
                planned_whole_body_targets_.clear();
                planned_waypoint_poses_.clear();
            }
            return;
        }

        if (comparison_force_sweep_enabled_ && comparison_plan_ready_[comparison_run_index_])
        {
            planned_waypoint_times_ = comparison_planned_waypoint_times_[comparison_run_index_];
            planned_whole_body_targets_ =
                comparison_planned_whole_body_targets_[comparison_run_index_];
            last_optimized_model_configuration_ =
                comparison_last_optimized_model_configuration_[comparison_run_index_];
            plan_ready_ = true;

            ROS_INFO_STREAM(
                "target_pose_generator: starting comparison run " << (comparison_run_index_ + 1)
                << " with cached lift force "
                << liftForceMagnitudeForRunIndex(comparison_run_index_) << " N.");
            if (shouldResetToHomeBeforeTrajectoryStart())
            {
                experiment_phase_ = ExperimentPhase::kPrepareStart;
                reset_to_home_requested_ = false;
                phase_start_time_ = ros::Time::now();
            }
            return;
        }

        if (!ensureOptimizerReady())
        {
            plan_ready_ = false;
            planned_waypoint_times_.clear();
            planned_whole_body_targets_.clear();
            return;
        }

        const Eigen::VectorXd nominal_seed = buildNominalStateConfiguration();
        if (!planOptimizedJointTrajectorySegment(
                0.0,
                nominal_seed,
                currentLiftDesiredForceMagnitude()))
        {
            ROS_WARN(
                "target_pose_generator: failed to build optimized joint trajectory, "
                "using seed joint targets only.");
            plan_ready_ = false;
            planned_waypoint_times_.clear();
            planned_whole_body_targets_.clear();
        }
        else if (shouldResetToHomeBeforeTrajectoryStart())
        {
            experiment_phase_ = ExperimentPhase::kPrepareStart;
            reset_to_home_requested_ = false;
            phase_start_time_ = ros::Time::now();
        }
        else
        {
            trajectory_start_time_ = ros::Time::now();
            phase_start_time_ = trajectory_start_time_;
            ROS_INFO(
                "target_pose_generator: optimized plan is ready; resetting trajectory clock before publishing commands.");
        }
    }

    bool precomputeComparisonPlans()
    {
        if (!comparison_force_sweep_enabled_)
        {
            return true;
        }

        if (!ensureOptimizerReady())
        {
            return false;
        }

        const Eigen::VectorXd nominal_seed = buildNominalStateConfiguration();
        const int original_run_index = comparison_run_index_;

        for (int run_index = 0; run_index < 2; ++run_index)
        {
            comparison_run_index_ = run_index;

            if (!planOptimizedJointTrajectorySegment(
                    0.0,
                    nominal_seed,
                    liftForceMagnitudeForRunIndex(run_index)))
            {
                comparison_run_index_ = original_run_index;
                comparison_plan_ready_[run_index] = false;
                return false;
            }

            comparison_planned_waypoint_times_[run_index] = planned_waypoint_times_;
            comparison_planned_whole_body_targets_[run_index] = planned_whole_body_targets_;
            comparison_last_optimized_model_configuration_[run_index] =
                last_optimized_model_configuration_;
            comparison_plan_ready_[run_index] = true;
        }

        comparison_run_index_ = original_run_index;
        return true;
    }

    double computeTrajectoryDurationSec() const
    {
        if (isCircleMode())
        {
            return trajectory_start_delay_sec_ +
                   std::max(1e-6, 1.0 / std::max(1e-6, circle_frequency_hz_));
        }

        if (isPayloadLiftMode())
        {
            return trajectory_start_delay_sec_ +
                   payload_move_above_duration_sec_ +
                   payload_descend_duration_sec_ +
                   payload_close_duration_sec_ +
                   payload_lift_duration_sec_;
        }

        if (isManualTargetPoseMode())
        {
            return trajectory_start_delay_sec_ +
                   manual_target_move_duration_sec_;
        }

        if (isSinglePointOptimizationMode())
        {
            return trajectory_start_delay_sec_ +
                   single_point_move_duration_sec_;
        }

        if (isReturnToHomeMode())
        {
            return trajectory_start_delay_sec_ +
                   home_return_duration_sec_;
        }

        if (isTeachingReturnInitialMode() || isTeachingWaypointsMode())
        {
            const std::size_t segment_count =
                planned_waypoint_times_.size() > 1
                    ? planned_waypoint_times_.size() - 1
                    : 1;
            return trajectory_start_delay_sec_ +
                   teaching_waypoint_segment_duration_sec_ *
                       static_cast<double>(segment_count);
        }

        return trajectory_start_delay_sec_ +
               test_forward_duration_sec_ +
               test_lift_duration_sec_;
    }

    Eigen::Quaterniond buildDownwardOrientation() const
    {
        const Eigen::Quaterniond seed_orientation(
            seed_pose_.pose.orientation.w,
            seed_pose_.pose.orientation.x,
            seed_pose_.pose.orientation.y,
            seed_pose_.pose.orientation.z);

        Eigen::Vector3d x_axis = seed_orientation.normalized().toRotationMatrix().col(0);
        x_axis.z() = 0.0;
        if (x_axis.norm() < 1e-6)
        {
            x_axis = Eigen::Vector3d::UnitX();
        }
        x_axis.normalize();

        const Eigen::Vector3d z_axis = -Eigen::Vector3d::UnitZ();
        Eigen::Vector3d y_axis = z_axis.cross(x_axis);
        if (y_axis.norm() < 1e-6)
        {
            x_axis = Eigen::Vector3d::UnitY();
            y_axis = z_axis.cross(x_axis);
        }
        y_axis.normalize();
        x_axis = y_axis.cross(z_axis).normalized();

        Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
        rotation.col(0) = x_axis;
        rotation.col(1) = y_axis;
        rotation.col(2) = z_axis;
        return Eigen::Quaterniond(rotation);
    }

    Eigen::Quaterniond interpolateSeedToDownwardOrientation(
        double alpha) const
    {
        const Eigen::Quaterniond seed_orientation(
            seed_pose_.pose.orientation.w,
            seed_pose_.pose.orientation.x,
            seed_pose_.pose.orientation.y,
            seed_pose_.pose.orientation.z);

        return seed_orientation.normalized().slerp(
            clampUnit(alpha),
            buildDownwardOrientation()).normalized();
    }

    Eigen::Affine3d buildCircleTargetPoseAffine(double elapsed_time) const
    {
        const double active_time = computeActiveTime(elapsed_time);
        const double ramp = std::min(1.0, active_time / trajectory_ramp_time_sec_);
        const double omega = 2.0 * M_PI * circle_frequency_hz_;

        const Eigen::Quaterniond seed_orientation(
            seed_pose_.pose.orientation.w,
            seed_pose_.pose.orientation.x,
            seed_pose_.pose.orientation.y,
            seed_pose_.pose.orientation.z);

        Eigen::Affine3d target_pose = Eigen::Affine3d::Identity();
        target_pose.linear() = seed_orientation.normalized().toRotationMatrix();
        target_pose.translation() << seed_pose_.pose.position.x +
                                         circle_amplitude_x_ * ramp * std::sin(omega * active_time),
                                     seed_pose_.pose.position.y +
                                         circle_amplitude_y_ * ramp * (1.0 - std::cos(omega * active_time)),
                                     seed_pose_.pose.position.z +
                                         circle_amplitude_z_ * ramp * std::sin(0.5 * omega * active_time);
        return target_pose;
    }

    Eigen::Affine3d buildTestLiftTargetPoseAffine(double elapsed_time) const
    {
        const double active_time = computeActiveTime(elapsed_time);
        const double x_progress =
            quinticTimeScaling(active_time / std::max(1e-6, test_forward_duration_sec_));
        const double z_progress =
            quinticTimeScaling(
                (active_time - test_forward_duration_sec_) /
                std::max(1e-6, test_lift_duration_sec_));

        Eigen::Affine3d target_pose = Eigen::Affine3d::Identity();
        target_pose.linear() =
            interpolateSeedToDownwardOrientation(x_progress).toRotationMatrix();
        target_pose.translation() << seed_pose_.pose.position.x + test_forward_distance_x_ * x_progress,
                                     seed_pose_.pose.position.y,
                                     seed_pose_.pose.position.z + test_lift_distance_z_ * z_progress;
        return target_pose;
    }

    /**
     * @brief Build the target pose affine for the payload lift trajectory at the given elapsed time.
     * 
     * @param elapsed_time 
     * @return Eigen::Affine3d 
     */
    Eigen::Affine3d buildPayloadLiftTargetPoseAffine(double elapsed_time) const
    {
        const double active_time = computeActiveTime(elapsed_time);
        const Eigen::Affine3d seed_pose_affine = buildSeedPoseAffine();
        const Eigen::Affine3d pregrasp_pose_affine = buildPayloadPregraspPoseAffine();
        const Eigen::Affine3d grasp_pose_affine = buildPayloadGraspPoseAffine();
        const Eigen::Affine3d lift_pose_affine = buildPayloadLiftPoseAffine();

        Eigen::Affine3d target_pose = Eigen::Affine3d::Identity();
        target_pose.linear() = buildDownwardOrientation().toRotationMatrix();

        if (active_time <= payload_move_above_duration_sec_)
        {
            const double orientation_progress =
                quinticTimeScaling(
                    active_time /
                    std::max(1e-6, payload_move_above_duration_sec_));
            target_pose.linear() =
                interpolateSeedToDownwardOrientation(
                    orientation_progress).toRotationMatrix();
            Eigen::Vector3d target_position = target_pose.translation();
            computeQuinticSegmentKinematics(
                seed_pose_affine.translation(),
                pregrasp_pose_affine.translation(),
                active_time,
                payload_move_above_duration_sec_,
                &target_position,
                nullptr,
                nullptr);
            target_pose.translation() = target_position;
            return target_pose;
        }

        const double descend_phase_time =
            active_time - payload_move_above_duration_sec_;
        if (descend_phase_time <= payload_descend_duration_sec_)
        {
            Eigen::Vector3d target_position = target_pose.translation();
            computeQuinticSegmentKinematics(
                pregrasp_pose_affine.translation(),
                grasp_pose_affine.translation(),
                descend_phase_time,
                payload_descend_duration_sec_,
                &target_position,
                nullptr,
                nullptr);
            target_pose.translation() = target_position;
            return target_pose;
        }

        const double close_phase_time =
            descend_phase_time - payload_descend_duration_sec_;
        if (close_phase_time <= payload_close_duration_sec_)
        {
            target_pose.translation() = grasp_pose_affine.translation();
            return target_pose;
        }

        const double lift_phase_time =
            close_phase_time - payload_close_duration_sec_;
        if (lift_phase_time <= payload_lift_duration_sec_)
        {
            Eigen::Vector3d target_position = target_pose.translation();
            computeQuinticSegmentKinematics(
                grasp_pose_affine.translation(),
                lift_pose_affine.translation(),
                lift_phase_time,
                payload_lift_duration_sec_,
                &target_position,
                nullptr,
                nullptr);
            target_pose.translation() = target_position;
            return target_pose;
        }

        target_pose.translation() = lift_pose_affine.translation();
        return target_pose;
    }

    Eigen::Affine3d buildManualTargetPoseAffine(double elapsed_time) const
    {
        const Eigen::Affine3d seed_pose_affine = buildSeedPoseAffine();
        const Eigen::Affine3d goal_pose_affine = buildManualTargetGoalPoseAffine();
        const double active_time = computeActiveTime(elapsed_time);
        const double duration = std::max(1e-6, manual_target_move_duration_sec_);
        const double alpha = quinticTimeScaling(active_time / duration);

        const Eigen::Quaterniond seed_orientation(seed_pose_affine.linear());
        const Eigen::Quaterniond goal_orientation(goal_pose_affine.linear());

        Eigen::Affine3d target_pose = Eigen::Affine3d::Identity();
        target_pose.linear() =
            seed_orientation.slerp(alpha, goal_orientation).normalized().toRotationMatrix();
        target_pose.translation() =
            (1.0 - alpha) * seed_pose_affine.translation() +
            alpha * goal_pose_affine.translation();
        return target_pose;
    }

    /**
     * @brief 得到单点目标末端位姿
     * 
     * @return Eigen::Affine3d 
     */
    Eigen::Affine3d buildSinglePointGoalPoseAffine() const
    {
        if (single_point_has_target_whole_body_ && optimizer_initialized_)
        {
            try
            {
                // 计算末端位姿
                return whole_body_optimizer_.computeEndEffectorPose(
                    buildOptimizerStateFromWholeBodyTarget(
                        single_point_target_whole_body_));
            }
            catch (const std::exception& e)
            {
                ROS_WARN_STREAM_THROTTLE(
                    1.0,
                    "target_pose_generator: FK from selected single-point target failed: "
                    << e.what() << "; falling back to single_point_target_position.");
            }
        }
        // 把成员函数中的目标位置和姿态信息组合成一个仿射变换矩阵返回
        Eigen::Affine3d pose = Eigen::Affine3d::Identity();
        pose.translation() = single_point_target_position_;
        pose.linear() = single_point_target_orientation_.normalized().toRotationMatrix();
        return pose;
    }

    Eigen::Affine3d buildSinglePointTargetPoseAffine(double elapsed_time) const
    {
        const Eigen::Affine3d seed_pose_affine = buildSeedPoseAffine();
        const Eigen::Affine3d goal_pose_affine = buildSinglePointGoalPoseAffine();
        const double active_time = computeActiveTime(elapsed_time);
        const double duration = std::max(1e-6, single_point_move_duration_sec_);
        const double alpha = quinticTimeScaling(active_time / duration);

        const Eigen::Quaterniond seed_orientation(seed_pose_affine.linear());
        const Eigen::Quaterniond goal_orientation(goal_pose_affine.linear());

        Eigen::Affine3d target_pose = Eigen::Affine3d::Identity();
        target_pose.linear() =
            seed_orientation.slerp(alpha, goal_orientation).normalized().toRotationMatrix();
        target_pose.translation() =
            (1.0 - alpha) * seed_pose_affine.translation() +
            alpha * goal_pose_affine.translation();
        return target_pose;
    }

    Eigen::Affine3d buildReturnToHomePoseAffine(double elapsed_time) const
    {
        const Eigen::Affine3d seed_pose_affine = buildSeedPoseAffine();
        const Eigen::Affine3d home_pose_affine = buildStoredHomePoseAffine();
        const double active_time = computeActiveTime(elapsed_time);
        const double duration = std::max(1e-6, comparison_return_home_duration_sec_);
        const double alpha = quinticTimeScaling(active_time / duration);

        const Eigen::Quaterniond seed_orientation(seed_pose_affine.linear());
        const Eigen::Quaterniond home_orientation(home_pose_affine.linear());

        Eigen::Affine3d target_pose = Eigen::Affine3d::Identity();
        target_pose.linear() =
            seed_orientation.slerp(alpha, home_orientation).normalized().toRotationMatrix();
        target_pose.translation() =
            (1.0 - alpha) * seed_pose_affine.translation() +
            alpha * home_pose_affine.translation();
        return target_pose;
    }

    Eigen::Affine3d interpolateTeachingPose(double elapsed_time) const
    {
        if (planned_waypoint_times_.empty() || planned_waypoint_poses_.empty())
        {
            return buildSeedPoseAffine();
        }

        if (elapsed_time <= planned_waypoint_times_.front())
        {
            return planned_waypoint_poses_.front();
        }
        if (elapsed_time >= planned_waypoint_times_.back())
        {
            return planned_waypoint_poses_.back();
        }

        const auto upper_it = std::upper_bound(
            planned_waypoint_times_.begin(),
            planned_waypoint_times_.end(),
            elapsed_time);
        const std::size_t upper_index =
            static_cast<std::size_t>(std::distance(planned_waypoint_times_.begin(), upper_it));
        const std::size_t lower_index = upper_index - 1;
        const double t0 = planned_waypoint_times_[lower_index];
        const double t1 = planned_waypoint_times_[upper_index];
        const double alpha = quinticTimeScaling(
            (elapsed_time - t0) / std::max(1e-9, t1 - t0));

        const Eigen::Quaterniond q0(planned_waypoint_poses_[lower_index].linear());
        const Eigen::Quaterniond q1(planned_waypoint_poses_[upper_index].linear());
        Eigen::Affine3d pose = Eigen::Affine3d::Identity();
        pose.linear() = q0.slerp(alpha, q1).normalized().toRotationMatrix();
        pose.translation() =
            (1.0 - alpha) * planned_waypoint_poses_[lower_index].translation() +
            alpha * planned_waypoint_poses_[upper_index].translation();
        return pose;
    }

    /**
     * @brief Build the target pose affine for the current trajectory mode at the given elapsed time.
     * 
     * @param elapsed_time 
     * @return Eigen::Affine3d 
     */
    Eigen::Affine3d buildTargetPoseAffine(double elapsed_time) const
    {
        if (isCircleMode())
        {
            return buildCircleTargetPoseAffine(elapsed_time);
        }

        if (isPayloadLiftMode())
        {
            return buildPayloadLiftTargetPoseAffine(elapsed_time);
        }

        if (isManualTargetPoseMode())
        {
            return buildManualTargetPoseAffine(elapsed_time);
        }

        if (isSinglePointOptimizationMode())
        {
            return buildSinglePointTargetPoseAffine(elapsed_time);
        }

        if (isReturnToHomeMode())
        {
            if (!planned_waypoint_poses_.empty())
            {
                return interpolateTeachingPose(elapsed_time);
            }
            return buildReturnToHomePoseAffine(elapsed_time);
        }

        if (isTeachingReturnInitialMode() || isTeachingWaypointsMode())
        {
            return interpolateTeachingPose(elapsed_time);
        }

        return buildTestLiftTargetPoseAffine(elapsed_time);
    }

    Eigen::Matrix<double, 6, 1> buildCircleTargetTwist(double elapsed_time) const
    {
        Eigen::Matrix<double, 6, 1> twist = Eigen::Matrix<double, 6, 1>::Zero();
        if (elapsed_time >= computeTrajectoryDurationSec())
        {
            return twist;
        }

        const double active_time = computeActiveTime(elapsed_time);
        const double ramp_duration = std::max(1e-6, trajectory_ramp_time_sec_);
        const double ramp = std::min(1.0, active_time / ramp_duration);
        const double ramp_dot = active_time < ramp_duration ? 1.0 / ramp_duration : 0.0;
        const double omega = 2.0 * M_PI * circle_frequency_hz_;

        const double sin_omega_t = std::sin(omega * active_time);
        const double cos_omega_t = std::cos(omega * active_time);
        const double sin_half_omega_t = std::sin(0.5 * omega * active_time);
        const double cos_half_omega_t = std::cos(0.5 * omega * active_time);

        twist(0) =
            circle_amplitude_x_ *
            (ramp_dot * sin_omega_t + ramp * omega * cos_omega_t);
        twist(1) =
            circle_amplitude_y_ *
            (ramp_dot * (1.0 - cos_omega_t) + ramp * omega * sin_omega_t);
        twist(2) =
            circle_amplitude_z_ *
            (ramp_dot * sin_half_omega_t + 0.5 * ramp * omega * cos_half_omega_t);

        return twist;
    }

    Eigen::Matrix<double, 6, 1> buildCircleTargetAcceleration(double elapsed_time) const
    {
        Eigen::Matrix<double, 6, 1> acceleration =
            Eigen::Matrix<double, 6, 1>::Zero();
        if (elapsed_time >= computeTrajectoryDurationSec())
        {
            return acceleration;
        }

        const double active_time = computeActiveTime(elapsed_time);
        const double ramp_duration = std::max(1e-6, trajectory_ramp_time_sec_);
        const double ramp = std::min(1.0, active_time / ramp_duration);
        const double ramp_dot = active_time < ramp_duration ? 1.0 / ramp_duration : 0.0;
        const double omega = 2.0 * M_PI * circle_frequency_hz_;

        const double sin_omega_t = std::sin(omega * active_time);
        const double cos_omega_t = std::cos(omega * active_time);
        const double sin_half_omega_t = std::sin(0.5 * omega * active_time);
        const double cos_half_omega_t = std::cos(0.5 * omega * active_time);

        acceleration(0) =
            circle_amplitude_x_ *
            (2.0 * ramp_dot * omega * cos_omega_t -
             ramp * omega * omega * sin_omega_t);
        acceleration(1) =
            circle_amplitude_y_ *
            (2.0 * ramp_dot * omega * sin_omega_t +
             ramp * omega * omega * cos_omega_t);
        acceleration(2) =
            circle_amplitude_z_ *
            (ramp_dot * omega * cos_half_omega_t -
             0.25 * ramp * omega * omega * sin_half_omega_t);

        return acceleration;
    }

    Eigen::Matrix<double, 6, 1> buildTestLiftTargetTwist(double elapsed_time) const
    {
        Eigen::Matrix<double, 6, 1> twist = Eigen::Matrix<double, 6, 1>::Zero();
        if (elapsed_time >= computeTrajectoryDurationSec())
        {
            return twist;
        }

        const double active_time = computeActiveTime(elapsed_time);
        const double forward_duration = std::max(1e-6, test_forward_duration_sec_);
        const double lift_duration = std::max(1e-6, test_lift_duration_sec_);

        const double x_unit = active_time / forward_duration;
        const double z_unit =
            (active_time - test_forward_duration_sec_) / lift_duration;

        twist(0) =
            test_forward_distance_x_ *
            quinticTimeScalingFirstDerivative(x_unit) / forward_duration;
        twist(2) =
            test_lift_distance_z_ *
            quinticTimeScalingFirstDerivative(z_unit) / lift_duration;

        return twist;
    }

    Eigen::Matrix<double, 6, 1> buildTestLiftTargetAcceleration(double elapsed_time) const
    {
        Eigen::Matrix<double, 6, 1> acceleration =
            Eigen::Matrix<double, 6, 1>::Zero();
        if (elapsed_time >= computeTrajectoryDurationSec())
        {
            return acceleration;
        }

        const double active_time = computeActiveTime(elapsed_time);
        const double forward_duration = std::max(1e-6, test_forward_duration_sec_);
        const double lift_duration = std::max(1e-6, test_lift_duration_sec_);

        const double x_unit = active_time / forward_duration;
        const double z_unit =
            (active_time - test_forward_duration_sec_) / lift_duration;

        acceleration(0) =
            test_forward_distance_x_ *
            quinticTimeScalingSecondDerivative(x_unit) /
            (forward_duration * forward_duration);
        acceleration(2) =
            test_lift_distance_z_ *
            quinticTimeScalingSecondDerivative(z_unit) /
            (lift_duration * lift_duration);

        return acceleration;
    }

    Eigen::Matrix<double, 6, 1> buildPayloadLiftTargetTwist(double elapsed_time) const
    {
        Eigen::Matrix<double, 6, 1> twist = Eigen::Matrix<double, 6, 1>::Zero();
        if (elapsed_time >= computeTrajectoryDurationSec())
        {
            return twist;
        }

        const double active_time = computeActiveTime(elapsed_time);
        const Eigen::Affine3d seed_pose_affine = buildSeedPoseAffine();
        const Eigen::Affine3d pregrasp_pose_affine = buildPayloadPregraspPoseAffine();
        const Eigen::Affine3d grasp_pose_affine = buildPayloadGraspPoseAffine();
        const Eigen::Affine3d lift_pose_affine = buildPayloadLiftPoseAffine();

        Eigen::Vector3d velocity = Eigen::Vector3d::Zero();

        if (active_time <= payload_move_above_duration_sec_)
        {
            computeQuinticSegmentKinematics(
                seed_pose_affine.translation(),
                pregrasp_pose_affine.translation(),
                active_time,
                payload_move_above_duration_sec_,
                nullptr,
                &velocity,
                nullptr);
        }
        else
        {
            const double descend_phase_time =
                active_time - payload_move_above_duration_sec_;
            if (descend_phase_time <= payload_descend_duration_sec_)
            {
                computeQuinticSegmentKinematics(
                    pregrasp_pose_affine.translation(),
                    grasp_pose_affine.translation(),
                    descend_phase_time,
                    payload_descend_duration_sec_,
                    nullptr,
                    &velocity,
                    nullptr);
            }
            else
            {
                const double close_phase_time =
                    descend_phase_time - payload_descend_duration_sec_;
                if (close_phase_time > payload_close_duration_sec_)
                {
                    const double lift_phase_time =
                        close_phase_time - payload_close_duration_sec_;
                    if (lift_phase_time <= payload_lift_duration_sec_)
                    {
                        computeQuinticSegmentKinematics(
                            grasp_pose_affine.translation(),
                            lift_pose_affine.translation(),
                            lift_phase_time,
                            payload_lift_duration_sec_,
                            nullptr,
                            &velocity,
                            nullptr);
                    }
                }
            }
        }

        twist.head<3>() = velocity;
        return twist;
    }

    Eigen::Matrix<double, 6, 1> buildPayloadLiftTargetAcceleration(double elapsed_time) const
    {
        Eigen::Matrix<double, 6, 1> acceleration =
            Eigen::Matrix<double, 6, 1>::Zero();
        if (elapsed_time >= computeTrajectoryDurationSec())
        {
            return acceleration;
        }

        const double active_time = computeActiveTime(elapsed_time);
        const Eigen::Affine3d seed_pose_affine = buildSeedPoseAffine();
        const Eigen::Affine3d pregrasp_pose_affine = buildPayloadPregraspPoseAffine();
        const Eigen::Affine3d grasp_pose_affine = buildPayloadGraspPoseAffine();
        const Eigen::Affine3d lift_pose_affine = buildPayloadLiftPoseAffine();

        Eigen::Vector3d linear_acceleration = Eigen::Vector3d::Zero();

        if (active_time <= payload_move_above_duration_sec_)
        {
            computeQuinticSegmentKinematics(
                seed_pose_affine.translation(),
                pregrasp_pose_affine.translation(),
                active_time,
                payload_move_above_duration_sec_,
                nullptr,
                nullptr,
                &linear_acceleration);
        }
        else
        {
            const double descend_phase_time =
                active_time - payload_move_above_duration_sec_;
            if (descend_phase_time <= payload_descend_duration_sec_)
            {
                computeQuinticSegmentKinematics(
                    pregrasp_pose_affine.translation(),
                    grasp_pose_affine.translation(),
                    descend_phase_time,
                    payload_descend_duration_sec_,
                    nullptr,
                    nullptr,
                    &linear_acceleration);
            }
            else
            {
                const double close_phase_time =
                    descend_phase_time - payload_descend_duration_sec_;
                if (close_phase_time > payload_close_duration_sec_)
                {
                    const double lift_phase_time =
                        close_phase_time - payload_close_duration_sec_;
                    if (lift_phase_time <= payload_lift_duration_sec_)
                    {
                        computeQuinticSegmentKinematics(
                            grasp_pose_affine.translation(),
                            lift_pose_affine.translation(),
                            lift_phase_time,
                            payload_lift_duration_sec_,
                            nullptr,
                            nullptr,
                            &linear_acceleration);
                    }
                }
            }
        }

        acceleration.head<3>() = linear_acceleration;
        return acceleration;
    }

    Eigen::Matrix<double, 6, 1> buildManualTargetTwist(double elapsed_time) const
    {
        Eigen::Matrix<double, 6, 1> twist = Eigen::Matrix<double, 6, 1>::Zero();
        if (elapsed_time >= computeTrajectoryDurationSec())
        {
            return twist;
        }

        Eigen::Vector3d linear_velocity = Eigen::Vector3d::Zero();
        computeQuinticSegmentKinematics(
            buildSeedPoseAffine().translation(),
            buildManualTargetGoalPoseAffine().translation(),
            computeActiveTime(elapsed_time),
            manual_target_move_duration_sec_,
            nullptr,
            &linear_velocity,
            nullptr);
        twist.head<3>() = linear_velocity;
        return twist;
    }

    Eigen::Matrix<double, 6, 1> buildManualTargetAcceleration(double elapsed_time) const
    {
        Eigen::Matrix<double, 6, 1> acceleration =
            Eigen::Matrix<double, 6, 1>::Zero();
        if (elapsed_time >= computeTrajectoryDurationSec())
        {
            return acceleration;
        }

        Eigen::Vector3d linear_acceleration = Eigen::Vector3d::Zero();
        computeQuinticSegmentKinematics(
            buildSeedPoseAffine().translation(),
            buildManualTargetGoalPoseAffine().translation(),
            computeActiveTime(elapsed_time),
            manual_target_move_duration_sec_,
            nullptr,
            nullptr,
            &linear_acceleration);
        acceleration.head<3>() = linear_acceleration;
        return acceleration;
    }

    Eigen::Matrix<double, 6, 1> buildSinglePointTargetTwist(double elapsed_time) const
    {
        Eigen::Matrix<double, 6, 1> twist = Eigen::Matrix<double, 6, 1>::Zero();
        if (elapsed_time >= computeTrajectoryDurationSec())
        {
            return twist;
        }

        Eigen::Vector3d linear_velocity = Eigen::Vector3d::Zero();
        computeQuinticSegmentKinematics(
            buildSeedPoseAffine().translation(),
            buildSinglePointGoalPoseAffine().translation(),
            computeActiveTime(elapsed_time),
            single_point_move_duration_sec_,
            nullptr,
            &linear_velocity,
            nullptr);
        twist.head<3>() = linear_velocity;
        return twist;
    }

    Eigen::Matrix<double, 6, 1> buildSinglePointTargetAcceleration(double elapsed_time) const
    {
        Eigen::Matrix<double, 6, 1> acceleration =
            Eigen::Matrix<double, 6, 1>::Zero();
        if (elapsed_time >= computeTrajectoryDurationSec())
        {
            return acceleration;
        }

        Eigen::Vector3d linear_acceleration = Eigen::Vector3d::Zero();
        computeQuinticSegmentKinematics(
            buildSeedPoseAffine().translation(),
            buildSinglePointGoalPoseAffine().translation(),
            computeActiveTime(elapsed_time),
            single_point_move_duration_sec_,
            nullptr,
            nullptr,
            &linear_acceleration);
        acceleration.head<3>() = linear_acceleration;
        return acceleration;
    }

    Eigen::Matrix<double, 6, 1> buildReturnToHomeTwist(double elapsed_time) const
    {
        Eigen::Matrix<double, 6, 1> twist = Eigen::Matrix<double, 6, 1>::Zero();
        if (elapsed_time >= computeTrajectoryDurationSec())
        {
            return twist;
        }

        Eigen::Vector3d linear_velocity = Eigen::Vector3d::Zero();
        computeQuinticSegmentKinematics(
            buildSeedPoseAffine().translation(),
            buildStoredHomePoseAffine().translation(),
            computeActiveTime(elapsed_time),
            comparison_return_home_duration_sec_,
            nullptr,
            &linear_velocity,
            nullptr);
        twist.head<3>() = linear_velocity;
        return twist;
    }

    Eigen::Matrix<double, 6, 1> buildReturnToHomeAcceleration(double elapsed_time) const
    {
        Eigen::Matrix<double, 6, 1> acceleration =
            Eigen::Matrix<double, 6, 1>::Zero();
        if (elapsed_time >= computeTrajectoryDurationSec())
        {
            return acceleration;
        }

        Eigen::Vector3d linear_acceleration = Eigen::Vector3d::Zero();
        computeQuinticSegmentKinematics(
            buildSeedPoseAffine().translation(),
            buildStoredHomePoseAffine().translation(),
            computeActiveTime(elapsed_time),
            comparison_return_home_duration_sec_,
            nullptr,
            nullptr,
            &linear_acceleration);
        acceleration.head<3>() = linear_acceleration;
        return acceleration;
    }

    CartesianTrajectorySample buildStaticTrajectorySample(
        const Eigen::Affine3d& pose) const
    {
        CartesianTrajectorySample sample;
        sample.pose = pose;
        sample.twist.setZero();
        sample.acceleration.setZero();
        return sample;
    }

    CartesianTrajectorySample buildTargetTrajectorySample(double elapsed_time) const
    {
        if (isTeachingFileTrajectoryMode())
        {
            CartesianTrajectorySample sample;
            sample.pose = buildTargetPoseAffine(elapsed_time);
            return sample;
        }

        if (elapsed_time >= computeTrajectoryDurationSec())
        {
            return buildStaticTrajectorySample(
                buildTargetPoseAffine(computeTrajectoryDurationSec()));
        }

        CartesianTrajectorySample sample;
        sample.pose = buildTargetPoseAffine(elapsed_time);

        if (isCircleMode())
        {
            sample.twist = buildCircleTargetTwist(elapsed_time);
            sample.acceleration = buildCircleTargetAcceleration(elapsed_time);
        }
        else if (isPayloadLiftMode())
        {
            sample.twist = buildPayloadLiftTargetTwist(elapsed_time);
            sample.acceleration = buildPayloadLiftTargetAcceleration(elapsed_time);
        }
        else if (isManualTargetPoseMode())
        {
            sample.twist = buildManualTargetTwist(elapsed_time);
            sample.acceleration = buildManualTargetAcceleration(elapsed_time);
        }
        else if (isSinglePointOptimizationMode())
        {
            sample.twist = buildSinglePointTargetTwist(elapsed_time);
            sample.acceleration = buildSinglePointTargetAcceleration(elapsed_time);
        }
        else if (isReturnToHomeMode())
        {
            sample.twist = buildReturnToHomeTwist(elapsed_time);
            sample.acceleration = buildReturnToHomeAcceleration(elapsed_time);
        }
        else
        {
            sample.twist = buildTestLiftTargetTwist(elapsed_time);
            sample.acceleration = buildTestLiftTargetAcceleration(elapsed_time);
        }

        return sample;
    }

    geometry_msgs::PoseStamped buildTargetPoseMessage(double elapsed_time) const
    {
        const Eigen::Affine3d target_pose = buildTargetPoseAffine(elapsed_time);
        return buildPoseMessageFromAffine(target_pose);
    }

    geometry_msgs::PoseStamped buildPoseMessageFromAffine(
        const Eigen::Affine3d& target_pose) const
    {
        geometry_msgs::PoseStamped target_pose_msg;
        target_pose_msg.header.stamp = ros::Time::now();
        target_pose_msg.header.frame_id = world_frame_;
        if (!target_pose.matrix().allFinite())
        {
            ROS_WARN_THROTTLE(
                1.0,
                "target_pose_generator: target pose contains non-finite values; publishing identity pose instead.");
            target_pose_msg.pose.orientation.w = 1.0;
            return target_pose_msg;
        }

        target_pose_msg.pose.position.x = target_pose.translation().x();
        target_pose_msg.pose.position.y = target_pose.translation().y();
        target_pose_msg.pose.position.z = target_pose.translation().z();

        Eigen::Quaterniond target_orientation;
        if (!quaternionFromRotationMatrix(target_pose.linear(), &target_orientation))
        {
            ROS_WARN_THROTTLE(
                1.0,
                "target_pose_generator: target pose has an invalid orientation; publishing identity orientation instead.");
            target_orientation = Eigen::Quaterniond::Identity();
        }
        target_pose_msg.pose.orientation.x = target_orientation.x();
        target_pose_msg.pose.orientation.y = target_orientation.y();
        target_pose_msg.pose.orientation.z = target_orientation.z();
        target_pose_msg.pose.orientation.w = target_orientation.w();

        return target_pose_msg;
    }

    Eigen::Affine3d buildSeedPoseAffine() const
    {
        Eigen::Affine3d seed_pose_affine = Eigen::Affine3d::Identity();
        seed_pose_affine.translation() <<
            trajectory_seed_pose_.pose.position.x,
            trajectory_seed_pose_.pose.position.y,
            trajectory_seed_pose_.pose.position.z;

        const Eigen::Quaterniond seed_orientation(
            trajectory_seed_pose_.pose.orientation.w,
            trajectory_seed_pose_.pose.orientation.x,
            trajectory_seed_pose_.pose.orientation.y,
            trajectory_seed_pose_.pose.orientation.z);
        seed_pose_affine.linear() = seed_orientation.normalized().toRotationMatrix();
        return seed_pose_affine;
    }

    Eigen::Affine3d buildHomePoseAffine() const
    {
        return buildTargetPoseAffine(0.0);
    }

    bool shouldHoldStaticTargetUntilPlanReady() const
    {
        if (isReturnToHomeMode())
        {
            return false;
        }
        return use_nullspace_joint_target_ && !plan_ready_;
    }

    bool shouldPublishExternalWholeBodyTarget() const
    {
        if (isTeachingFileTrajectoryMode())
        {
            return use_nullspace_joint_target_;
        }
        return use_nullspace_joint_target_ && plan_ready_;
    }

    bool shouldResetToHomeBeforeTrajectoryStart() const
    {
        return reset_to_home_before_trajectory_start_ &&
               !isReturnToHomeMode() &&
               !isTeachingReturnInitialMode() &&
               !isTeachingWaypointsMode() &&
               !isManualTargetPoseMode() &&
               !isSinglePointOptimizationMode() &&
               use_nullspace_joint_target_ &&
               plan_ready_ &&
               !pre_trajectory_reset_completed_;
    }

    void publishComparisonRunState(int state)
    {
        std_msgs::Int32 msg;
        msg.data = state;
        comparison_run_state_pub_.publish(msg);
    }

    void publishResetToHomeRequest()
    {
        std_msgs::Empty msg;
        reset_to_home_pub_.publish(msg);
    }

    void publishPrepareStartCommand()
    {
        const ros::Time now = ros::Time::now();

        if (!reset_to_home_requested_)
        {
            publishResetToHomeRequest();
            reset_to_home_requested_ = true;
            phase_start_time_ = now;
            ROS_INFO(
                "target_pose_generator: requested reset-to-home before starting the planned trajectory.");
        }

        if (comparison_force_sweep_enabled_)
        {
            publishComparisonRunState(-1);
        }

        publishCommand(
            buildStaticTrajectorySample(buildSeedPoseAffine()),
            buildHomeWholeBodyTarget(),
            Eigen::Vector3d::Zero(),
            finger_open_command_);

        const double phase_elapsed =
            std::max(0.0, (now - phase_start_time_).toSec());
        if (phase_elapsed >= pre_trajectory_home_settle_duration_sec_)
        {
            trajectory_start_time_ = now;
            phase_start_time_ = now;
            experiment_phase_ = ExperimentPhase::kTrajectory;
            reset_to_home_requested_ = false;
            pre_trajectory_reset_completed_ = true;
            ROS_INFO(
                "target_pose_generator: reset-to-home settle finished, starting planned trajectory execution.");
        }
    }

    /** Reset to home pose  */
    geometry_msgs::PoseStamped buildReturnHomePoseMessage(double phase_elapsed_time) const
    {
        const Eigen::Affine3d home_pose = buildHomePoseAffine();
        const double alpha = quinticTimeScaling(
            phase_elapsed_time / std::max(1e-6, comparison_return_home_duration_sec_));

        Eigen::Affine3d blended_pose = Eigen::Affine3d::Identity();
        blended_pose.translation() =
            (1.0 - alpha) * return_start_pose_affine_.translation() +
            alpha * home_pose.translation();

        const Eigen::Quaterniond start_orientation(return_start_pose_affine_.linear());
        const Eigen::Quaterniond home_orientation(home_pose.linear());
        blended_pose.linear() =
            start_orientation.slerp(alpha, home_orientation).normalized().toRotationMatrix();

        return buildPoseMessageFromAffine(blended_pose);
    }

    /**
     * @brief Builds a desired force trajectory for the arm.
     *
     * @param[in] start_pose The start pose of the arm.
     * @param[in] end_pose The end pose of the arm.
     * @param[in] alpha The alpha parameter for blending between the start and end poses.
     * @param[in] gravity_adjusted Whether to adjust the desired forces for gravity.
     * @return The desired force trajectory.
     * 
     * @param waypoint_times 
     * @param target_poses 
     * @param lift_force_magnitude 
     * @return std::vector<Eigen::Vector3d> 
     */
    std::vector<Eigen::Vector3d> buildDesiredForceTrajectory(
        const std::vector<double>& waypoint_times,
        const std::vector<Eigen::Affine3d>& target_poses,
        double lift_force_magnitude) const
    {
        (void)target_poses;
        (void)lift_force_magnitude;

        std::vector<Eigen::Vector3d> desired_forces;
        desired_forces.reserve(waypoint_times.size());

        for (double waypoint_time : waypoint_times)
        {
            const CartesianTrajectorySample target_sample =
                buildTargetTrajectorySample(waypoint_time);
            desired_forces.emplace_back(
                buildDesiredForceAtTime(waypoint_time, target_sample));
        }

        return desired_forces;
    }

    double computeDesiredForceScale(double command_time) const
    {
        if (!enable_force_capability_optimization_)
        {
            return 0.0;
        }

        const double ramp_duration =
            std::max(1e-6, desired_force_ramp_time_sec_);
        const double active_time = computeActiveTime(command_time);

        if (isCircleMode())
        {
            return quinticTimeScaling(active_time / ramp_duration);
        }

        if (isManualTargetPoseMode() || isSinglePointOptimizationMode())
        {
            return 1.0;
        }

        if (isPayloadLiftMode())
        {
            const double lift_phase_start =
                payload_move_above_duration_sec_ +
                payload_descend_duration_sec_ +
                payload_close_duration_sec_;
            if (active_time <= lift_phase_start)
            {
                return 0.0;
            }

            const double lift_force_ramp_duration =
                std::min(ramp_duration, std::max(1e-6, payload_lift_duration_sec_));
            return quinticTimeScaling(
                (active_time - lift_phase_start) / lift_force_ramp_duration);
        }

        if (isTeachingWaypointsMode())
        {
            return quinticTimeScaling(active_time / ramp_duration);
        }

        if (active_time <= test_forward_duration_sec_)
        {
            return 0.0;
        }

        const double lift_ramp_duration =
            std::min(ramp_duration, std::max(1e-6, test_lift_duration_sec_));
        return quinticTimeScaling(
            (active_time - test_forward_duration_sec_) / lift_ramp_duration);
    }

    bool isZeroForceMode() const
    {
        return task_force_mode_ == "zero_force";
    }

    bool isMassBasedLoadForceMode() const
    {
        return task_force_mode_ == "mass_based_load";
    }

    bool isLegacyFixedForceMode() const
    {
        return task_force_mode_ == "legacy_fixed";
    }

    double currentPayloadMassKg() const
    {
        if (has_estimated_payload_mass_)
        {
            return estimated_payload_mass_kg_;
        }
        return default_payload_mass_kg_;
    }

    Eigen::Vector3d gravityVectorWorld() const
    {
        return Eigen::Vector3d(0.0, 0.0, -gravity_acceleration_mps2_);
    }

    double configuredFrankaToolLoadMassKg() const
    {
        if (!has_franka_tool_state_ ||
            !std::isfinite(latest_franka_tool_state_.m_total) ||
            latest_franka_tool_state_.m_total <= 0.0)
        {
            return 0.0;
        }
        return latest_franka_tool_state_.m_total;
    }

    void refreshVlmRuntimeParamsFromServer()
    {
        std::string runtime_vlm_request_image_path = vlm_request_image_path_;
        pnh_.param<std::string>(
            "vlm_request_image_path",
            runtime_vlm_request_image_path,
            runtime_vlm_request_image_path);
        pnh_.param(
            "vlm_mass_request_force_refresh",
            vlm_mass_request_force_refresh_,
            vlm_mass_request_force_refresh_);
        pnh_.param(
            "vlm_mass_request_timeout_sec",
            vlm_mass_request_timeout_sec_,
            vlm_mass_request_timeout_sec_);
        pnh_.param(
            "vlm_timeout_uses_default_payload_mass",
            vlm_timeout_uses_default_payload_mass_,
            vlm_timeout_uses_default_payload_mass_);
        pnh_.param(
            "vlm_offline_mode",
            vlm_offline_mode_,
            vlm_offline_mode_);
        vlm_request_image_path_ = resolveRosPath(runtime_vlm_request_image_path);
    }

    Eigen::Vector3d buildFrankaToolLoadRequiredForce(
        const CartesianTrajectorySample& target_sample) const
    {
        if (!enable_franka_tool_load_planning_compensation_ || vlm_offline_mode_)
        {
            return Eigen::Vector3d::Zero();
        }

        const double tool_load_mass_kg = configuredFrankaToolLoadMassKg();
        if (tool_load_mass_kg <= 0.0)
        {
            return Eigen::Vector3d::Zero();
        }

        Eigen::Vector3d desired_linear_acceleration = Eigen::Vector3d::Zero();
        if (!isSinglePointOptimizationMode())
        {
            desired_linear_acceleration = target_sample.acceleration.head<3>();
        }
        return tool_load_mass_kg *
               (desired_linear_acceleration - gravityVectorWorld());
    }

    Eigen::Vector3d buildLegacyFixedDesiredForceAtTime(double command_time) const
    {
        const double desired_force_scale = computeDesiredForceScale(command_time);
        Eigen::Vector3d target_force = Eigen::Vector3d::Zero();
        if (isCircleMode())
        {
            target_force =
                circle_desired_force_magnitude_ * Eigen::Vector3d::UnitZ();
        }
        else
        {
            target_force =
                currentLiftDesiredForceMagnitude() * Eigen::Vector3d::UnitZ();
        }

        (void)command_time;
        return desired_force_scale * target_force;
    }

    Eigen::Vector3d buildMassBasedDesiredForceAtTime(
        double command_time,
        const CartesianTrajectorySample& target_sample) const
    {
        const double desired_force_scale = computeDesiredForceScale(command_time);
        const double payload_mass_kg = currentPayloadMassKg();
        if (payload_mass_kg <= 0.0)
        {
            ROS_WARN_THROTTLE(
                1.0,
                "target_pose_generator: task_force_mode is mass_based_load but the current payload mass is %.6f kg; publishing zero desired force.",
                payload_mass_kg);
            (void)command_time;
            (void)desired_force_scale;
            return Eigen::Vector3d::Zero();
        }

        Eigen::Vector3d desired_linear_acceleration = Eigen::Vector3d::Zero();
        if (!isSinglePointOptimizationMode())
        {
            desired_linear_acceleration = target_sample.acceleration.head<3>();
        }
        const Eigen::Vector3d target_force =
            payload_mass_kg *
            (desired_linear_acceleration - gravityVectorWorld());
        (void)command_time;
        return desired_force_scale * target_force;
    }

    Eigen::Vector3d buildUnscaledTaskDesiredForceTarget(
        const CartesianTrajectorySample& target_sample) const
    {
        if (isMassBasedLoadForceMode())
        {
            const double payload_mass_kg = currentPayloadMassKg();
            if (payload_mass_kg <= 0.0)
            {
                return Eigen::Vector3d::Zero();
            }

            Eigen::Vector3d desired_linear_acceleration = Eigen::Vector3d::Zero();
            if (!isSinglePointOptimizationMode())
            {
                desired_linear_acceleration = target_sample.acceleration.head<3>();
            }
            return payload_mass_kg *
                   (desired_linear_acceleration - gravityVectorWorld());
        }

        if (isCircleMode())
        {
            return circle_desired_force_magnitude_ * Eigen::Vector3d::UnitZ();
        }

        return currentLiftDesiredForceMagnitude() * Eigen::Vector3d::UnitZ();
    }

    Eigen::Vector3d blendDesiredForceTarget(
        double command_time,
        const Eigen::Vector3d& target_force,
        double desired_force_scale) const
    {
        if (!isTeachingWaypointsMode())
        {
            return desired_force_scale * target_force;
        }

        const double alpha = std::max(0.0, std::min(1.0, desired_force_scale));
        const Eigen::Vector3d start_force =
            has_trajectory_start_desired_force_
                ? trajectory_start_desired_force_
                : Eigen::Vector3d::Zero();
        (void)command_time;
        return (1.0 - alpha) * start_force + alpha * target_force;
    }

    Eigen::Vector3d buildDesiredForceAtTime(
        double command_time,
        const CartesianTrajectorySample& target_sample) const
    {
        if (isReturnToHomeMode())
        {
            return Eigen::Vector3d::Zero();
        }

        if (isZeroForceMode())
        {
            return Eigen::Vector3d::Zero();
        }

        if (isMassBasedLoadForceMode())
        {
            if (isTeachingWaypointsMode())
            {
                return blendDesiredForceTarget(
                    command_time,
                    buildUnscaledTaskDesiredForceTarget(target_sample) +
                        buildFrankaToolLoadRequiredForce(target_sample),
                    computeDesiredForceScale(command_time));
            }
            return buildMassBasedDesiredForceAtTime(command_time, target_sample) +
                   buildFrankaToolLoadRequiredForce(target_sample);
        }

        if (isTeachingWaypointsMode())
        {
            return blendDesiredForceTarget(
                command_time,
                buildUnscaledTaskDesiredForceTarget(target_sample) +
                    buildFrankaToolLoadRequiredForce(target_sample),
                computeDesiredForceScale(command_time));
        }

        return buildLegacyFixedDesiredForceAtTime(command_time) +
               buildFrankaToolLoadRequiredForce(target_sample);
    }

    Eigen::Vector3d buildControlDesiredForceAtTime(
        double command_time,
        const CartesianTrajectorySample& target_sample) const
    {
        if (vlm_offline_mode_)
        {
            return Eigen::Vector3d::Zero();
        }
        return buildDesiredForceAtTime(command_time, target_sample);
    }

    double buildFingerCommandAtTime(double command_time) const
    {
        if (!isPayloadLiftMode())
        {
            return finger_open_command_;
        }

        const double active_time = computeActiveTime(command_time);
        const double close_phase_start =
            payload_move_above_duration_sec_ + payload_descend_duration_sec_;
        if (active_time < close_phase_start)
        {
            return finger_open_command_;
        }

        const double close_phase_progress = quinticTimeScaling(
            (active_time - close_phase_start) /
            std::max(1e-6, payload_close_duration_sec_));
        return (1.0 - close_phase_progress) * finger_open_command_ +
               close_phase_progress * finger_closed_command_;
    }

    /**
     * @brief 生成一段 whole-body optimized trajectory（移动底盘 + 机械臂）的轨迹规划
     * 
     * @param segment_start_time 当前时刻
     * @param nominal_seed 
     * @param lift_force_magnitude 期望 lifting force
     * @return true 
     * @return false 
     */
    bool planOptimizedJointTrajectorySegment(
        double segment_start_time,
        const Eigen::VectorXd& nominal_seed,
        double lift_force_magnitude)
    {
        if (!optimizer_initialized_)
        {
            last_plan_failure_reason_ =
                "offline whole-body optimizer is not initialized.";
            return false;
        }

        try
        {
            // 先判断一下是否是单点优化
            const bool single_point_mode = isSinglePointOptimizationMode();
            
            const double planning_duration =
                single_point_mode
                    ? std::max(1e-3, single_point_move_duration_sec_)
                    : replan_during_execution_
                          ? optimization_horizon_sec_
                          : std::max(1e-3, computeTrajectoryDurationSec() - segment_start_time);
            // 如果是单点优化模式，那么规划的 waypoint 就只有 1 个，否则根据 optimization_waypoint_count_ 来规划多个 waypoint
            const int planning_waypoint_count =
                single_point_mode
                    ? 1
                    : optimization_waypoint_count_;
            // 根据规划的总时长和 waypoint 数量来计算每个 waypoint 之间的时间间隔
            const double waypoint_dt =
                planning_waypoint_count > 1
                    ? planning_duration /
                          static_cast<double>(planning_waypoint_count - 1)
                    : 0.0;

            polytope_wx::WholeBodyTrajectoryOptimizationInput input;
            // 是不是把第一个点固定住，如果是单点优化模式就不固定住第一个点，否则固定住第一个点
            input.pin_first_state = !single_point_mode;
            // 如果是单点优化模式的话，优化器的 cost term 只包含单点的 pose cost term，
            // 不包含 trajectory 的 smoothness cost term 和 waypoint pose cost term；如果不是单点优化模式的话，优化器的 cost term 包含 trajectory 的 smoothness cost term 和 waypoint pose cost term
            input.single_point_cost_terms_only = single_point_mode;
            
            // 开辟容量
            input.target_poses.reserve(planning_waypoint_count);
            input.desired_forces.reserve(planning_waypoint_count);
            input.desired_force_radii.assign(
                static_cast<std::size_t>(planning_waypoint_count),
                std::max(0.0, desired_force_radius_N_));
            input.nominal_state_trajectory.reserve(planning_waypoint_count);
            input.waypoint_times_sec.reserve(planning_waypoint_count);

            std::vector<double> waypoint_times;
            waypoint_times.reserve(planning_waypoint_count);

            for (int i = 0; i < planning_waypoint_count; ++i)
            {
                const double waypoint_time = segment_start_time + waypoint_dt * static_cast<double>(i);
                waypoint_times.push_back(waypoint_time);
                input.waypoint_times_sec.push_back(waypoint_time);
                if (single_point_mode)
                {
                    // 把“单点优化的目标末端位姿”放进优化器输入里。
                    input.target_poses.push_back(buildSinglePointGoalPoseAffine());
                }
                else
                {
                    input.target_poses.push_back(buildTargetPoseAffine(waypoint_time));
                }
                // 是否是示教模式下
                if (isTeachingWaypointsMode() && plan_ready_ &&
                    !planned_waypoint_times_.empty() &&
                    !planned_whole_body_targets_.empty())
                {
                    input.nominal_state_trajectory.push_back(
                        buildOptimizerStateFromWholeBodyTarget(
                            interpolateTeachingWholeBodyTarget(waypoint_time)));
                }
                else if (single_point_mode && single_point_has_target_whole_body_)
                {
                    input.nominal_state_trajectory.push_back(
                        buildOptimizerStateFromWholeBodyTarget(
                            single_point_target_whole_body_));
                }
                else
                {
                    input.nominal_state_trajectory.push_back(nominal_seed);
                }
            }

            if (single_point_mode)
            {
                const CartesianTrajectorySample static_goal_sample =
                    buildStaticTrajectorySample(buildSinglePointGoalPoseAffine());
                input.desired_forces.push_back(
                    buildDesiredForceAtTime(
                        segment_start_time + planning_duration,
                        static_goal_sample));
            }
            else
            {
                input.desired_forces = buildDesiredForceTrajectory(
                    waypoint_times,
                    input.target_poses,
                    lift_force_magnitude);
            }

            const polytope_wx::WholeBodyTrajectoryOptimizationResult result =
                whole_body_optimizer_.optimize(input);
            if (result.optimized_state_trajectory.empty())
            {
                last_plan_failure_reason_ =
                    "optimizer returned an empty whole-body trajectory.";
                ROS_ERROR("target_pose_generator: optimizer returned an empty whole-body trajectory.");
                return false;
            }

            double max_pose_error = 0.0;
            double min_force_capacity = std::numeric_limits<double>::infinity();
            for (const auto& metrics : result.waypoint_metrics)
            {
                max_pose_error = std::max(max_pose_error, metrics.pose_error_norm);
                min_force_capacity = std::min(min_force_capacity, metrics.force_capacity);
            }

            maybeExportPlannerMetricsCsv(input, result);

            if (!result.success)
            {
                last_plan_failure_reason_ = result.message;
                ROS_WARN_STREAM(
                    "target_pose_generator: rejecting whole-body plan because the optimizer "
                    "did not satisfy the end-effector constraints tightly enough. " <<
                    result.message <<
                    ", lift_force=" << lift_force_magnitude <<
                    ", max_pose_error=" << max_pose_error <<
                    ", min_force_capacity=" << min_force_capacity);
                return false;
            }

            planned_waypoint_times_ = std::move(waypoint_times);
            planned_whole_body_targets_.clear();
            planned_whole_body_targets_.reserve(result.optimized_state_trajectory.size());
            planned_waypoint_poses_.clear();
            planned_waypoint_poses_.reserve(result.optimized_state_trajectory.size());

            for (const Eigen::VectorXd& optimized_state : result.optimized_state_trajectory)
            {
                if (optimized_state.size() < kBaseJointCount + kArmJointCount)
                {
                    std::ostringstream failure_stream;
                    failure_stream
                        << "optimized state vector size " << optimized_state.size()
                        << " is smaller than the required whole-body target size "
                        << (kBaseJointCount + kArmJointCount) << ".";
                    last_plan_failure_reason_ = failure_stream.str();
                    ROS_ERROR(
                        "target_pose_generator: optimized state vector size %ld is smaller than whole-body target size %d.",
                        optimized_state.size(),
                        kBaseJointCount + kArmJointCount);
                    return false;
                }

                WholeBodyTarget whole_body_target = WholeBodyTarget::Zero();
                whole_body_target.head<kBaseJointCount>() =
                    optimized_state.head<kBaseJointCount>();
                whole_body_target.tail<kArmJointCount>() =
                    optimized_state.segment(kBaseJointCount, kArmJointCount);
                planned_whole_body_targets_.push_back(whole_body_target);

                const Eigen::Affine3d optimized_pose =
                    whole_body_optimizer_.computeEndEffectorPose(optimized_state);
                if (!optimized_pose.matrix().allFinite())
                {
                    last_plan_failure_reason_ =
                        "optimized whole-body trajectory produced a non-finite end-effector pose.";
                    ROS_ERROR(
                        "target_pose_generator: optimized whole-body trajectory produced a non-finite end-effector pose.");
                    return false;
                }
                planned_waypoint_poses_.push_back(optimized_pose);
            }

            last_optimized_model_configuration_ = result.optimized_state_trajectory.back();
            plan_ready_ = true;
            last_plan_failure_reason_.clear();

            if (log_optimization_info_)
            {
                ROS_INFO_STREAM(
                    "target_pose_generator: planned optimized whole-body segment [" <<
                    planned_waypoint_times_.front() << ", " <<
                    planned_waypoint_times_.back() << "] s with " <<
                    planned_waypoint_times_.size() << " waypoints. success=" <<
                    (result.success ? "true" : "false") <<
                    ", lift_force=" << lift_force_magnitude <<
                    ", max_pose_error=" << max_pose_error <<
                    ", min_force_capacity=" << min_force_capacity);
            }
            logPlannedWholeBodyTrajectorySegment();
            return true;
        }
        catch (const std::exception& e)
        {
            last_plan_failure_reason_ =
                std::string("polytope optimization failed: ") + e.what();
            ROS_ERROR_STREAM(
                "target_pose_generator: polytope optimization failed: " << e.what());
            return false;
        }
    }

    bool maybeRetryOfflinePlanPreparation()
    {
        if (plan_ready_ || !use_nullspace_joint_target_ || !optimizer_initialized_)
        {
            return false;
        }

        const ros::Time now = ros::Time::now();
        if (!last_plan_retry_time_.isZero() &&
            (now - last_plan_retry_time_).toSec() < offline_plan_retry_delay_sec_)
        {
            return false;
        }
        last_plan_retry_time_ = now;

        const Eigen::VectorXd nominal_seed = buildNominalStateConfiguration();
        const bool planned =
            planOptimizedJointTrajectorySegment(
                0.0,
                nominal_seed,
                currentLiftDesiredForceMagnitude());
        if (planned)
        {
            trajectory_start_time_ = now;
            phase_start_time_ = now;
            ROS_INFO("target_pose_generator: recovered an offline whole-body plan while holding the seed pose and restarted the trajectory clock.");
        }
        return planned;
    }

    std::string describeOfflinePlanHoldState() const
    {
        if (last_plan_failure_reason_.empty())
        {
            return "planner has not produced an acceptable offline whole-body plan yet";
        }

        return std::string("last planning attempt failed: ") + last_plan_failure_reason_;
    }

    void maybePlanNextSegment(double elapsed_time)
    {
        if (!replan_during_execution_ ||
            !use_nullspace_joint_target_ ||
            !optimizer_initialized_ || !plan_ready_ || planned_waypoint_times_.empty())
        {
            return;
        }

        if (elapsed_time < planned_waypoint_times_.back())
        {
            return;
        }

        if (!planOptimizedJointTrajectorySegment(
                planned_waypoint_times_.back(),
                last_optimized_model_configuration_,
                currentLiftDesiredForceMagnitude()))
        {
            ROS_WARN("target_pose_generator: keeping the last optimized nullspace target because replanning failed.");
        }
    }

    WholeBodyTarget fallbackWholeBodyTarget() const
    {
        WholeBodyTarget whole_body_target = WholeBodyTarget::Zero();
        for (int i = 0; i < kBaseJointCount; ++i)
        {
            whole_body_target(i) = trajectory_seed_base_planar_positions_[i];
        }
        for (int i = 0; i < kArmJointCount; ++i)
        {
            whole_body_target(kBaseJointCount + i) = trajectory_seed_arm_joint_positions_[i];
        }
        return whole_body_target;
    }

    bool buildRealtimeWholeBodyTarget(WholeBodyTarget* whole_body_target) const
    {
        if (whole_body_target == nullptr || !has_seed_joint_positions_)
        {
            return false;
        }

        whole_body_target->setZero();
        for (int i = 0; i < kBaseJointCount; ++i)
        {
            (*whole_body_target)(i) = seed_base_planar_positions_[i];
        }
        for (int i = 0; i < kArmJointCount; ++i)
        {
            (*whole_body_target)(kBaseJointCount + i) = seed_arm_joint_positions_[i];
        }
        return true;
    }

    WholeBodyTarget buildStoredHomeWholeBodyTarget() const
    {
        WholeBodyTarget whole_body_target = WholeBodyTarget::Zero();
        for (int i = 0; i < kBaseJointCount; ++i)
        {
            whole_body_target(i) = home_base_planar_positions_[i];
        }
        for (int i = 0; i < kArmJointCount; ++i)
        {
            whole_body_target(kBaseJointCount + i) = home_arm_joint_positions_[i];
        }
        return whole_body_target;
    }

    WholeBodyTarget buildReturnToStartupWholeBodyTarget(double elapsed_time) const
    {
        const double active_time = computeActiveTime(elapsed_time);
        const double duration = std::max(1e-6, comparison_return_home_duration_sec_);
        const double alpha = quinticTimeScaling(active_time / duration);

        const WholeBodyTarget start_target = fallbackWholeBodyTarget();
        const WholeBodyTarget home_target = buildStoredHomeWholeBodyTarget();
        WholeBodyTarget blended_target =
            (1.0 - alpha) * start_target +
            alpha * home_target;

        const double start_yaw = start_target(2);
        const double home_yaw = home_target(2);
        const double yaw_delta =
            std::atan2(std::sin(home_yaw - start_yaw), std::cos(home_yaw - start_yaw));
        blended_target(2) = std::atan2(
            std::sin(start_yaw + alpha * yaw_delta),
            std::cos(start_yaw + alpha * yaw_delta));
        return blended_target;
    }

    WholeBodyTarget interpolateWholeBodyTarget(
        double elapsed_time) const
    {
        if (!plan_ready_ || planned_waypoint_times_.empty() ||
            planned_whole_body_targets_.empty())
        {
            return fallbackWholeBodyTarget();
        }

        if (isSinglePointOptimizationMode() &&
            planned_whole_body_targets_.size() == 1)
        {
            const WholeBodyTarget start_target = fallbackWholeBodyTarget();
            const WholeBodyTarget goal_target = planned_whole_body_targets_.front();
            const double duration = std::max(1e-6, single_point_move_duration_sec_);
            const double alpha =
                quinticTimeScaling(std::max(0.0, std::min(1.0, elapsed_time / duration)));

            WholeBodyTarget blended_target =
                (1.0 - alpha) * start_target + alpha * goal_target;
            const double start_yaw = start_target(2);
            const double goal_yaw = goal_target(2);
            const double yaw_delta =
                std::atan2(std::sin(goal_yaw - start_yaw), std::cos(goal_yaw - start_yaw));
            blended_target(2) = std::atan2(
                std::sin(start_yaw + alpha * yaw_delta),
                std::cos(start_yaw + alpha * yaw_delta));
            return blended_target;
        }

        if (elapsed_time <= planned_waypoint_times_.front())
        {
            return planned_whole_body_targets_.front();
        }

        if (elapsed_time >= planned_waypoint_times_.back())
        {
            return planned_whole_body_targets_.back();
        }

        const auto upper_it = std::upper_bound(
            planned_waypoint_times_.begin(),
            planned_waypoint_times_.end(),
            elapsed_time);
        const std::size_t upper_index =
            static_cast<std::size_t>(std::distance(planned_waypoint_times_.begin(), upper_it));
        const std::size_t lower_index = upper_index - 1;

        const double t0 = planned_waypoint_times_[lower_index];
        const double t1 = planned_waypoint_times_[upper_index];
        const double alpha =
            (elapsed_time - t0) / std::max(1e-9, t1 - t0);

        WholeBodyTarget blended_target =
            (1.0 - alpha) * planned_whole_body_targets_[lower_index] +
            alpha * planned_whole_body_targets_[upper_index];
        const double lower_yaw = planned_whole_body_targets_[lower_index](2);
        const double upper_yaw = planned_whole_body_targets_[upper_index](2);
        const double yaw_delta =
            std::atan2(std::sin(upper_yaw - lower_yaw), std::cos(upper_yaw - lower_yaw));
        blended_target(2) = std::atan2(
            std::sin(lower_yaw + alpha * yaw_delta),
            std::cos(lower_yaw + alpha * yaw_delta));
        return blended_target;
    }

    WholeBodyTarget interpolateTeachingWholeBodyTarget(
        double elapsed_time) const
    {
        if (!plan_ready_ || planned_waypoint_times_.empty() ||
            planned_whole_body_targets_.empty())
        {
            return fallbackWholeBodyTarget();
        }

        if (elapsed_time <= planned_waypoint_times_.front())
        {
            return planned_whole_body_targets_.front();
        }

        if (elapsed_time >= planned_waypoint_times_.back())
        {
            return planned_whole_body_targets_.back();
        }

        const auto upper_it = std::upper_bound(
            planned_waypoint_times_.begin(),
            planned_waypoint_times_.end(),
            elapsed_time);
        const std::size_t upper_index =
            static_cast<std::size_t>(std::distance(planned_waypoint_times_.begin(), upper_it));
        const std::size_t lower_index = upper_index - 1;
        const double t0 = planned_waypoint_times_[lower_index];
        const double t1 = planned_waypoint_times_[upper_index];
        const double alpha = quinticTimeScaling(
            (elapsed_time - t0) / std::max(1e-9, t1 - t0));

        WholeBodyTarget blended_target =
            (1.0 - alpha) * planned_whole_body_targets_[lower_index] +
            alpha * planned_whole_body_targets_[upper_index];
        const double lower_yaw = planned_whole_body_targets_[lower_index](2);
        const double upper_yaw = planned_whole_body_targets_[upper_index](2);
        const double yaw_delta =
            std::atan2(std::sin(upper_yaw - lower_yaw), std::cos(upper_yaw - lower_yaw));
        blended_target(2) = std::atan2(
            std::sin(lower_yaw + alpha * yaw_delta),
            std::cos(lower_yaw + alpha * yaw_delta));
        return blended_target;
    }

    WholeBodyTarget buildHomeWholeBodyTarget() const
    {
        return fallbackWholeBodyTarget();
    }

    WholeBodyTarget buildReturnHomeWholeBodyTarget(
        double phase_elapsed_time) const
    {
        const double alpha = quinticTimeScaling(
            phase_elapsed_time / std::max(1e-6, comparison_return_home_duration_sec_));
        WholeBodyTarget blended_target =
            (1.0 - alpha) * return_start_whole_body_target_ +
            alpha * buildHomeWholeBodyTarget();
        const double start_yaw = return_start_whole_body_target_(2);
        const double home_yaw = buildHomeWholeBodyTarget()(2);
        const double yaw_delta =
            std::atan2(std::sin(home_yaw - start_yaw), std::cos(home_yaw - start_yaw));
        blended_target(2) = std::atan2(
            std::sin(start_yaw + alpha * yaw_delta),
            std::cos(start_yaw + alpha * yaw_delta));
        return blended_target;
    }

    Eigen::VectorXd buildOptimizerStateFromWholeBodyTarget(
        const WholeBodyTarget& whole_body_target) const
    {
        Eigen::VectorXd optimizer_state = Eigen::VectorXd::Zero(
            std::max(optimizer_state_dim_, kBaseJointCount + kArmJointCount));
        optimizer_state.head<kBaseJointCount>() = whole_body_target.head<kBaseJointCount>();
        optimizer_state.segment(kBaseJointCount, kArmJointCount) =
            whole_body_target.tail<kArmJointCount>();
        return optimizer_state;
    }

    void publishPlannedWholeBodyTf(
        const WholeBodyTarget& whole_body_target,
        const ros::Time& stamp)
    {
        if (!last_planned_whole_body_tf_stamp_.isZero() &&
            stamp <= last_planned_whole_body_tf_stamp_)
        {
            return;
        }

        if (!optimizer_initialized_ || !whole_body_target.allFinite() ||
            !std::isfinite(whole_body_target(2)))
        {
            ROS_WARN_THROTTLE(
                1.0,
                "target_pose_generator: skipped planned nullspace TF because the whole-body target is not finite.");
            return;
        }

        tf::Transform world_T_planned_base;
        world_T_planned_base.setOrigin(
            tf::Vector3(whole_body_target(0), whole_body_target(1), 0.0));
        tf::Quaternion planned_base_quaternion;
        const double planned_base_yaw =
            std::atan2(std::sin(whole_body_target(2)), std::cos(whole_body_target(2)));
        planned_base_quaternion.setRPY(0.0, 0.0, planned_base_yaw);
        if (!std::isfinite(planned_base_quaternion.x()) ||
            !std::isfinite(planned_base_quaternion.y()) ||
            !std::isfinite(planned_base_quaternion.z()) ||
            !std::isfinite(planned_base_quaternion.w()) ||
            planned_base_quaternion.length2() <= 1e-12)
        {
            ROS_WARN_THROTTLE(
                1.0,
                "target_pose_generator: skipped planned nullspace base TF because the yaw produced an invalid quaternion.");
            return;
        }
        planned_base_quaternion.normalize();
        world_T_planned_base.setRotation(planned_base_quaternion);
        tf_broadcaster_.sendTransform(
            tf::StampedTransform(
                world_T_planned_base,
                stamp,
                world_frame_,
                planned_base_frame_));

        Eigen::Affine3d world_T_planned_ee = Eigen::Affine3d::Identity();
        try
        {
            world_T_planned_ee =
                whole_body_optimizer_.computeEndEffectorPose(
                    buildOptimizerStateFromWholeBodyTarget(whole_body_target));
        }
        catch (const std::exception& e)
        {
            ROS_WARN_THROTTLE(
                1.0,
                "target_pose_generator: skipped planned nullspace EE TF because FK failed: %s",
                e.what());
            return;
        }

        Eigen::Quaterniond planned_ee_quaternion;
        if (!world_T_planned_ee.matrix().allFinite() ||
            !quaternionFromRotationMatrix(
                world_T_planned_ee.linear(),
                &planned_ee_quaternion))
        {
            ROS_WARN_THROTTLE(
                1.0,
                "target_pose_generator: skipped planned nullspace EE TF because FK produced an invalid transform.");
            return;
        }

        tf::Transform world_T_planned_ee_tf;
        world_T_planned_ee_tf.setOrigin(
            tf::Vector3(
                world_T_planned_ee.translation().x(),
                world_T_planned_ee.translation().y(),
                world_T_planned_ee.translation().z()));
        world_T_planned_ee_tf.setRotation(
            tf::Quaternion(
                planned_ee_quaternion.x(),
                planned_ee_quaternion.y(),
                planned_ee_quaternion.z(),
                planned_ee_quaternion.w()));
        tf_broadcaster_.sendTransform(
            tf::StampedTransform(
                world_T_planned_ee_tf,
                stamp,
                world_frame_,
                planned_ee_frame_));
        last_planned_whole_body_tf_stamp_ = stamp;
    }

    visualization_msgs::Marker makeCollisionSphereMarker(
        const std::string& marker_namespace,
        int marker_id,
        const Eigen::Vector3d& center,
        double radius,
        float red,
        float green,
        float blue,
        const ros::Time& stamp) const
    {
        visualization_msgs::Marker marker;
        marker.header.frame_id = world_frame_;
        marker.header.stamp = stamp;
        marker.ns = marker_namespace;
        marker.id = marker_id;
        marker.type = visualization_msgs::Marker::SPHERE;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.position.x = center.x();
        marker.pose.position.y = center.y();
        marker.pose.position.z = center.z();
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 2.0 * radius;
        marker.scale.y = 2.0 * radius;
        marker.scale.z = 2.0 * radius;
        marker.color.r = red;
        marker.color.g = green;
        marker.color.b = blue;
        marker.color.a = 0.35f;
        marker.lifetime = ros::Duration(0.35);
        return marker;
    }

    void appendCollisionSphereMarkers(
        visualization_msgs::MarkerArray* marker_array,
        const std::string& marker_namespace,
        const std::vector<polytope_wx::CollisionSphereSpec>& spheres,
        const std::vector<Eigen::Vector3d>& centers,
        float red,
        float green,
        float blue,
        const ros::Time& stamp) const
    {
        if (marker_array == nullptr)
        {
            return;
        }

        const std::size_t sphere_count = std::min(spheres.size(), centers.size());
        for (std::size_t i = 0; i < sphere_count; ++i)
        {
            if (!centers[i].allFinite() ||
                !std::isfinite(spheres[i].radius) ||
                spheres[i].radius <= 0.0)
            {
                ROS_WARN_THROTTLE(
                    1.0,
                    "target_pose_generator: skipped a collision sphere marker because its center or radius is invalid.");
                continue;
            }
            marker_array->markers.push_back(
                makeCollisionSphereMarker(
                    marker_namespace,
                    static_cast<int>(i),
                    centers[i],
                    spheres[i].radius,
                    red,
                    green,
                    blue,
                    stamp));
        }
    }

    void publishCollisionSphereMarkers(
        const WholeBodyTarget& whole_body_target,
        const ros::Time& stamp)
    {
        if (!publish_collision_sphere_markers_)
        {
            return;
        }

        visualization_msgs::MarkerArray marker_array;
        visualization_msgs::Marker delete_all_marker;
        delete_all_marker.header.frame_id = world_frame_;
        delete_all_marker.header.stamp = stamp;
        delete_all_marker.action = visualization_msgs::Marker::DELETEALL;
        delete_all_marker.pose.orientation.w = 1.0;
        marker_array.markers.push_back(delete_all_marker);

        if (!optimizer_initialized_ || optimizer_reinitialization_requested_)
        {
            if (!initializeOptimizer())
            {
                collision_sphere_marker_pub_.publish(marker_array);
                return;
            }
        }

        if (optimizer_collision_basket_spheres_.empty() &&
            optimizer_collision_body_spheres_.empty())
        {
            collision_sphere_marker_pub_.publish(marker_array);
            return;
        }

        try
        {
            const Eigen::VectorXd optimizer_state =
                buildOptimizerStateFromWholeBodyTarget(whole_body_target);
            const std::vector<Eigen::Vector3d> basket_centers =
                whole_body_optimizer_.computeCollisionSphereCentersWorld(
                    optimizer_state,
                    optimizer_collision_basket_spheres_);
            const std::vector<Eigen::Vector3d> body_centers =
                whole_body_optimizer_.computeCollisionSphereCentersWorld(
                    optimizer_state,
                    optimizer_collision_body_spheres_);

            appendCollisionSphereMarkers(
                &marker_array,
                "basket_collision_spheres",
                optimizer_collision_basket_spheres_,
                basket_centers,
                0.1f,
                0.45f,
                1.0f,
                stamp);
            appendCollisionSphereMarkers(
                &marker_array,
                "body_collision_spheres",
                optimizer_collision_body_spheres_,
                body_centers,
                0.1f,
                0.9f,
                0.25f,
                stamp);

            WholeBodyTarget realtime_whole_body_target;
            if (publish_realtime_collision_sphere_markers_ &&
                buildRealtimeWholeBodyTarget(&realtime_whole_body_target))
            {
                appendRealtimeCollisionSphereMarkers(
                    &marker_array,
                    realtime_whole_body_target,
                    stamp);
            }
        }
        catch (const std::exception& e)
        {
            ROS_WARN_THROTTLE(
                2.0,
                "target_pose_generator: failed to publish collision sphere markers: %s",
                e.what());
        }

        collision_sphere_marker_pub_.publish(marker_array);
    }

    void appendRealtimeCollisionSphereMarkers(
        visualization_msgs::MarkerArray* marker_array,
        const WholeBodyTarget& realtime_whole_body_target,
        const ros::Time& stamp)
    {
        if (marker_array == nullptr ||
            (optimizer_collision_basket_spheres_.empty() &&
             optimizer_collision_body_spheres_.empty()))
        {
            return;
        }

        const Eigen::VectorXd optimizer_state =
            buildOptimizerStateFromWholeBodyTarget(realtime_whole_body_target);
        const std::vector<Eigen::Vector3d> realtime_basket_centers =
            whole_body_optimizer_.computeCollisionSphereCentersWorld(
                optimizer_state,
                optimizer_collision_basket_spheres_);
        const std::vector<Eigen::Vector3d> realtime_body_centers =
            whole_body_optimizer_.computeCollisionSphereCentersWorld(
                optimizer_state,
                optimizer_collision_body_spheres_);

        appendCollisionSphereMarkers(
            marker_array,
            "current_basket_collision_spheres",
            optimizer_collision_basket_spheres_,
            realtime_basket_centers,
            1.0f,
            0.55f,
            0.05f,
            stamp);
        appendCollisionSphereMarkers(
            marker_array,
            "current_body_collision_spheres",
            optimizer_collision_body_spheres_,
            realtime_body_centers,
            0.95f,
            0.15f,
            0.85f,
            stamp);
    }

    void publishRealtimeCollisionSphereMarkers(const ros::Time& stamp)
    {
        if (!publish_collision_sphere_markers_ ||
            !publish_realtime_collision_sphere_markers_)
        {
            return;
        }

        if (!optimizer_initialized_ || optimizer_reinitialization_requested_)
        {
            if (!initializeOptimizer())
            {
                return;
            }
        }

        WholeBodyTarget realtime_whole_body_target;
        if (!buildRealtimeWholeBodyTarget(&realtime_whole_body_target))
        {
            return;
        }

        try
        {
            visualization_msgs::MarkerArray marker_array;
            appendRealtimeCollisionSphereMarkers(
                &marker_array,
                realtime_whole_body_target,
                stamp);
            if (!marker_array.markers.empty())
            {
                collision_sphere_marker_pub_.publish(marker_array);
            }
        }
        catch (const std::exception& e)
        {
            ROS_WARN_THROTTLE(
                2.0,
                "target_pose_generator: failed to publish realtime collision sphere markers: %s",
                e.what());
        }
    }

    void publishCommand(
        const CartesianTrajectorySample& target_sample,
        const WholeBodyTarget& whole_body_target,
        const Eigen::Vector3d& desired_force,
        double finger_command)
    {
        if (!target_sample.pose.matrix().allFinite() ||
            !target_sample.twist.allFinite() ||
            !target_sample.acceleration.allFinite() ||
            !whole_body_target.allFinite())
        {
            ROS_WARN_THROTTLE(
                1.0,
                "target_pose_generator: skipped /target_pose command because the planned command contains non-finite values.");
            return;
        }

        Eigen::Vector3d sanitized_desired_force = desired_force;
        if (!sanitized_desired_force.allFinite())
        {
            ROS_WARN_THROTTLE(
                1.0,
                "target_pose_generator: desired force contains non-finite values; publishing zero desired force for this cycle.");
            sanitized_desired_force.setZero();
        }

        moca_trajectory_generator::TargetPoseCommand target_command;
        const geometry_msgs::PoseStamped target_pose =
            buildPoseMessageFromAffine(target_sample.pose);
        target_command.header = target_pose.header;
        target_command.pose = target_pose.pose;
        target_command.twist.linear.x = target_sample.twist(0);
        target_command.twist.linear.y = target_sample.twist(1);
        target_command.twist.linear.z = target_sample.twist(2);
        target_command.twist.angular.x = target_sample.twist(3);
        target_command.twist.angular.y = target_sample.twist(4);
        target_command.twist.angular.z = target_sample.twist(5);
        target_command.acceleration.linear.x = target_sample.acceleration(0);
        target_command.acceleration.linear.y = target_sample.acceleration(1);
        target_command.acceleration.linear.z = target_sample.acceleration(2);
        target_command.acceleration.angular.x = target_sample.acceleration(3);
        target_command.acceleration.angular.y = target_sample.acceleration(4);
        target_command.acceleration.angular.z = target_sample.acceleration(5);
        target_command.desired_force.x = sanitized_desired_force.x();
        target_command.desired_force.y = sanitized_desired_force.y();
        target_command.desired_force.z = sanitized_desired_force.z();
        target_command.finger_command = finger_command;
        target_command.use_nullspace_joint_target =
            shouldPublishExternalWholeBodyTarget();

        for (int i = 0; i < kBaseJointCount; ++i)
        {
            target_command.base_planar_positions[i] = whole_body_target(i);
        }

        for (int i = 0; i < kArmJointCount; ++i)
        {
            target_command.arm_joint_positions[i] =
                whole_body_target(kBaseJointCount + i);
        }

        target_pose_pub_.publish(target_command);
        last_published_desired_force_ = sanitized_desired_force;
        has_last_published_desired_force_ = true;
        publishPlannedWholeBodyTf(whole_body_target, target_command.header.stamp);
        publishCollisionSphereMarkers(whole_body_target, target_command.header.stamp);
    }

    void publishSingleRunTargetCommand()
    {
        const double elapsed_time =
            std::max(0.0, (ros::Time::now() - trajectory_start_time_).toSec());

        if (shouldHoldStaticTargetUntilPlanReady())
        {
            maybeRetryOfflinePlanPreparation();
            ROS_WARN_THROTTLE(
                1.0,
                "target_pose_generator: holding the end-effector target at the seed pose because the offline whole-body plan is not ready; %s.",
                describeOfflinePlanHoldState().c_str());
            publishCommand(
                buildStaticTrajectorySample(buildSeedPoseAffine()),
                fallbackWholeBodyTarget(),
                Eigen::Vector3d::Zero(),
                finger_open_command_);
            return;
        }

        maybePlanNextSegment(elapsed_time);
        const double command_time =
            std::min(elapsed_time, computeTrajectoryDurationSec());

        if (isReturnToHomeMode())
        {
            publishCommand(
                buildTargetTrajectorySample(command_time),
                plan_ready_
                    ? interpolateTeachingWholeBodyTarget(command_time)
                    : buildReturnToStartupWholeBodyTarget(command_time),
                Eigen::Vector3d::Zero(),
                finger_open_command_);
            return;
        }

        if (isTeachingReturnInitialMode())
        {
            publishCommand(
                buildTargetTrajectorySample(command_time),
                interpolateTeachingWholeBodyTarget(command_time),
                Eigen::Vector3d::Zero(),
                finger_open_command_);
            return;
        }

        if (isTeachingWaypointsMode())
        {
            const CartesianTrajectorySample target_sample =
                buildTargetTrajectorySample(command_time);
            publishCommand(
                target_sample,
                interpolateTeachingWholeBodyTarget(command_time),
                buildControlDesiredForceAtTime(command_time, target_sample),
                finger_open_command_);
            return;
        }

        const CartesianTrajectorySample target_sample =
            buildTargetTrajectorySample(command_time);
        publishCommand(
            target_sample,
            interpolateWholeBodyTarget(command_time),
            buildControlDesiredForceAtTime(command_time, target_sample),
            buildFingerCommandAtTime(command_time));
    }

    void publishComparisonTargetCommand()
    {
        const ros::Time now = ros::Time::now();

        if (experiment_phase_ == ExperimentPhase::kTrajectory)
        {
            publishComparisonRunState(comparison_run_index_ + 1);
            const double elapsed_time =
                std::max(0.0, (now - trajectory_start_time_).toSec());

            if (shouldHoldStaticTargetUntilPlanReady())
            {
                maybeRetryOfflinePlanPreparation();
                ROS_WARN_THROTTLE(
                    1.0,
                    "target_pose_generator: holding the end-effector target at the seed pose because the offline whole-body comparison plan is not ready; %s.",
                    describeOfflinePlanHoldState().c_str());
                publishCommand(
                    buildStaticTrajectorySample(buildSeedPoseAffine()),
                    fallbackWholeBodyTarget(),
                    Eigen::Vector3d::Zero(),
                    finger_open_command_);
                return;
            }

            maybePlanNextSegment(elapsed_time);
            const double command_time =
                std::min(elapsed_time, computeTrajectoryDurationSec());

            const CartesianTrajectorySample target_sample =
                buildTargetTrajectorySample(command_time);
            publishCommand(
                target_sample,
                interpolateWholeBodyTarget(command_time),
                buildControlDesiredForceAtTime(command_time, target_sample),
                buildFingerCommandAtTime(command_time));

            if (elapsed_time >=
                    computeTrajectoryDurationSec() + comparison_goal_settle_duration_sec_ &&
                comparison_run_index_ == 0)
            {
                return_start_pose_affine_ =
                    buildTargetPoseAffine(computeTrajectoryDurationSec());
                return_start_whole_body_target_ =
                    interpolateWholeBodyTarget(computeTrajectoryDurationSec());
                experiment_phase_ = ExperimentPhase::kReturnHome;
                reset_to_home_requested_ = false;
                phase_start_time_ = now;
            }
            else if (elapsed_time >=
                     computeTrajectoryDurationSec() + comparison_goal_settle_duration_sec_)
            {
                experiment_phase_ = ExperimentPhase::kComplete;
                phase_start_time_ = now;
            }
            return;
        }

        if (experiment_phase_ == ExperimentPhase::kReturnHome)
        {
            publishComparisonRunState(-1);
            if (!reset_to_home_requested_)
            {
                publishResetToHomeRequest();
                reset_to_home_requested_ = true;
                ROS_INFO(
                    "target_pose_generator: requested Isaac reset-to-home between comparison runs.");
                phase_start_time_ = now;
            }

            publishCommand(
                buildStaticTrajectorySample(buildHomePoseAffine()),
                buildHomeWholeBodyTarget(),
                Eigen::Vector3d::Zero(),
                finger_open_command_);
            experiment_phase_ = ExperimentPhase::kSettleHome;
            return;
        }

        if (experiment_phase_ == ExperimentPhase::kSettleHome)
        {
            publishComparisonRunState(-1);
            publishCommand(
                buildStaticTrajectorySample(buildHomePoseAffine()),
                buildHomeWholeBodyTarget(),
                Eigen::Vector3d::Zero(),
                finger_open_command_);

            const double phase_elapsed =
                std::max(0.0, (now - phase_start_time_).toSec());
            if (phase_elapsed >= comparison_home_settle_duration_sec_)
            {
                ++comparison_run_index_;
                reset_to_home_requested_ = false;
                startTrajectoryRun();
            }
            return;
        }

        publishComparisonRunState(3);
        publishCommand(
            buildTargetTrajectorySample(computeTrajectoryDurationSec()),
            interpolateWholeBodyTarget(computeTrajectoryDurationSec()),
            Eigen::Vector3d::Zero(),
            buildFingerCommandAtTime(computeTrajectoryDurationSec()));
    }

    void publishTargetCommand()
    {
        if (!trajectory_started_)
        {
            return;
        }

        if (experiment_phase_ == ExperimentPhase::kPrepareStart)
        {
            publishPrepareStartCommand();
            return;
        }

        if (comparison_force_sweep_enabled_)
        {
            publishComparisonTargetCommand();
            return;
        }

        publishSingleRunTargetCommand();
    }

    ros::NodeHandle nh_;
    ros::NodeHandle pnh_{"~"};
    ros::Publisher target_pose_pub_;
    ros::Publisher collision_sphere_marker_pub_;
    ros::Publisher mass_request_pub_;
    ros::Publisher comparison_run_state_pub_;
    ros::Publisher reset_to_home_pub_;
    ros::Subscriber seed_pose_sub_;
    ros::Subscriber seed_joint_state_sub_;
    ros::Subscriber manual_target_pose_sub_;
    ros::Subscriber force_capability_settings_sub_;
    ros::Subscriber mass_result_sub_;
    ros::Subscriber franka_tool_state_sub_;
    ros::ServiceServer start_planning_service_;

    polytope_wx::WholeBodyPolytopeOptimizer whole_body_optimizer_;

    geometry_msgs::PoseStamped seed_pose_;
    geometry_msgs::PoseStamped manual_target_pose_;
    geometry_msgs::PoseStamped trajectory_seed_pose_;
    geometry_msgs::PoseStamped trajectory_manual_target_pose_;
    geometry_msgs::PoseStamped home_pose_;
    std::array<double, kBaseJointCount> seed_base_planar_positions_{};
    std::array<double, kArmJointCount> seed_arm_joint_positions_{};
    std::array<double, kBaseJointCount> trajectory_seed_base_planar_positions_{};
    std::array<double, kArmJointCount> trajectory_seed_arm_joint_positions_{};
    std::array<double, kBaseJointCount> home_base_planar_positions_{};
    std::array<double, kArmJointCount> home_arm_joint_positions_{};

    int publish_rate_hz_{100};
    bool use_nullspace_joint_target_{true};
    bool log_optimization_info_{false};
    bool replan_during_execution_{false};
    std::string seed_pose_topic_{"/wb_cart_imp_controller/moca_pose"};
    std::string seed_joint_state_topic_{"/wb_cart_imp_controller/moca_state"};
    std::string target_pose_topic_{"/target_pose"};
    std::string manual_target_pose_topic_{kDefaultManualTargetPoseTopic};
    bool publish_collision_sphere_markers_{true};
    bool publish_realtime_collision_sphere_markers_{true};
    std::string collision_sphere_marker_topic_{kDefaultCollisionSphereMarkerTopic};
    std::string task_force_mode_{kDefaultTaskForceMode};
    double gravity_acceleration_mps2_{9.81};
    bool enable_vlm_mass_request_{false};
    bool autostart_planning_{true};
    std::string vlm_mass_request_topic_{kDefaultVlmMassRequestTopic};
    std::string vlm_mass_result_topic_{kDefaultVlmMassResultTopic};
    std::string start_planning_service_name_{kDefaultStartPlanningService};
    std::string vlm_request_image_path_;
    double vlm_mass_request_timeout_sec_{30.0};
    bool vlm_mass_request_force_refresh_{false};
    bool vlm_timeout_uses_default_payload_mass_{false};
    bool vlm_offline_mode_{false};
    double default_payload_mass_kg_{0.0};
    std::string franka_tool_state_topic_{kDefaultFrankaToolStateTopic};
    bool enable_franka_tool_load_planning_compensation_{true};
    std::string comparison_run_state_topic_{kDefaultComparisonRunStateTopic};
    std::string reset_to_home_topic_{kDefaultResetToHomeTopic};
    std::string force_capability_settings_topic_{"/force_capability_settings"};
    std::string world_frame_{"odom"};
    std::string planned_base_frame_{"moca_planned_nullspace_base"};
    std::string planned_ee_frame_{"moca_planned_nullspace_ee"};
    ros::Time last_planned_whole_body_tf_stamp_;
    bool export_planner_metrics_{true};
    std::string planner_metrics_csv_path_{kDefaultPlannerMetricsCsvPath};
    std::string trajectory_mode_{kDefaultTrajectoryMode};
    double manual_target_move_duration_sec_{kDefaultManualTargetMoveDurationSec};
    double single_point_move_duration_sec_{kDefaultSinglePointMoveDurationSec};
    Eigen::Vector3d single_point_target_position_{0.55, 0.0, 1.10};
    Eigen::Quaterniond single_point_target_orientation_{1.0, 0.0, 0.0, 0.0};
    WholeBodyTarget single_point_target_whole_body_ = WholeBodyTarget::Zero();
    bool single_point_has_target_whole_body_{false};
    std::string home_teaching_point_path_{kDefaultHomeTeachingPointPath};
    std::string teaching_waypoints_path_{kDefaultTeachingWaypointsPath};
    double home_return_duration_sec_{kDefaultHomeReturnDurationSec};
    double teaching_waypoint_segment_duration_sec_{
        kDefaultTeachingWaypointSegmentDurationSec};
    double trajectory_start_delay_sec_{kTrajectoryStartDelaySec};
    double trajectory_ramp_time_sec_{kTrajectoryRampTimeSec};
    double optimization_horizon_sec_{kOptimizationHorizonSec};
    int optimization_waypoint_count_{kDefaultOptimizationWaypointCount};
    double circle_frequency_hz_{kDefaultCircleFrequencyHz};
    double circle_amplitude_x_{kDefaultCircleAmplitudeX};
    double circle_amplitude_y_{kDefaultCircleAmplitudeY};
    double circle_amplitude_z_{kDefaultCircleAmplitudeZ};
    double circle_desired_force_magnitude_{kDefaultCircleDesiredForceMagnitude};
    double test_forward_distance_x_{kDefaultTestForwardDistanceX};
    double test_lift_distance_z_{kDefaultTestLiftDistanceZ};
    double test_forward_duration_sec_{kDefaultTestForwardDurationSec};
    double test_lift_duration_sec_{kDefaultTestLiftDurationSec};
    double payload_box_center_x_{kDefaultPayloadBoxCenterX};
    double payload_box_center_y_{kDefaultPayloadBoxCenterY};
    double payload_box_center_z_{kDefaultPayloadBoxCenterZ};
    double payload_handle_top_z_{kDefaultPayloadHandleTopZ};
    double payload_move_above_duration_sec_{kDefaultPayloadMoveAboveDurationSec};
    double payload_descend_duration_sec_{kDefaultPayloadDescendDurationSec};
    double payload_close_duration_sec_{kDefaultPayloadCloseDurationSec};
    double payload_lift_duration_sec_{kDefaultPayloadLiftDurationSec};
    double payload_pregrasp_height_{kDefaultPayloadPregraspHeight};
    double payload_grasp_ee_offset_z_{kDefaultPayloadGraspEeOffsetZ};
    double payload_lift_distance_z_{kDefaultPayloadLiftDistanceZ};
    double finger_open_command_{kDefaultFingerOpenCommand};
    double finger_closed_command_{kDefaultFingerClosedCommand};
    double lift_desired_force_magnitude_{kDefaultLiftDesiredForceMagnitude};
    double desired_force_radius_N_{kDefaultDesiredForceRadiusN};
    bool enable_force_capability_optimization_{kDefaultEnableForceCapabilityOptimization};
    bool comparison_force_sweep_enabled_{kDefaultComparisonForceSweepEnabled};
    double comparison_first_lift_force_magnitude_{kDefaultComparisonFirstLiftForceMagnitude};
    double comparison_second_lift_force_magnitude_{kDefaultComparisonSecondLiftForceMagnitude};
    double comparison_return_home_duration_sec_{kDefaultComparisonReturnHomeDurationSec};
    double comparison_home_settle_duration_sec_{kDefaultComparisonHomeSettleDurationSec};
    double comparison_goal_settle_duration_sec_{kDefaultComparisonGoalSettleDurationSec};
    double desired_force_ramp_time_sec_{kDefaultDesiredForceRampTimeSec};
    bool reset_to_home_before_trajectory_start_{kDefaultResetToHomeBeforeTrajectoryStart};
    double pre_trajectory_home_settle_duration_sec_{kDefaultPreTrajectoryHomeSettleDurationSec};
    std::string polytope_urdf_path_{kPolytopeUrdfPath};
    std::string polytope_ee_frame_{kPolytopeEeFrame};
    std::string left_finger_joint_name_{kLeftFingerJointName};
    std::string right_finger_joint_name_{kRightFingerJointName};
    int optimizer_max_iterations_per_waypoint_{60};
    int optimizer_max_trajectory_iterations_{30};
    double optimizer_pose_gain_{0.9};
    double optimizer_nullspace_step_size_{0.01};
    double optimizer_max_joint_update_norm_{0.08};
    double optimizer_pose_tolerance_{5e-4};
    double optimizer_capability_weight_{1.0};
    double optimizer_manipulability_weight_{1.0};
    double optimizer_joint_limit_weight_{1e-3};
    Eigen::VectorXd optimizer_joint_limit_dof_weights_;
    double optimizer_smoothness_weight_{5e-2};
    double optimizer_velocity_weight_{0.0};
    double optimizer_base_velocity_weight_{0.0};
    double optimizer_nominal_weight_{5e-2};
    double optimizer_base_spectral_energy_weight_{0.0};
    double optimizer_base_spectral_cutoff_ratio_{0.35};
    double optimizer_base_spectral_power_{2.0};
    double optimizer_base_spectral_metric_x_{1.0};
    double optimizer_base_spectral_metric_y_{1.0};
    double optimizer_base_spectral_metric_yaw_{0.5};
    double optimizer_collision_weight_{0.0};
    double optimizer_collision_safe_distance_{0.08};
    std::vector<polytope_wx::CollisionSphereSpec> optimizer_collision_basket_spheres_;
    std::vector<polytope_wx::CollisionSphereSpec> optimizer_collision_body_spheres_;
    double optimizer_capability_alpha_{0.8};
    bool optimizer_capability_exponent_limit_enabled_{true};
    std::string optimizer_comparison_algorithm_{"ours"};
    Eigen::Vector3d optimizer_comparison_cone_axis_world_{Eigen::Vector3d::UnitZ()};
    double optimizer_comparison_cone_half_angle_rad_{0.35};
    int optimizer_comparison_cone_ring_count_{3};
    int optimizer_comparison_cone_azimuth_count_{8};
    bool optimizer_use_dynamic_residual_force_polytope_{true};
    bool optimizer_dynamic_residual_diagnostics_enabled_{false};
    std::string optimizer_dynamic_residual_diagnostics_directory_{
        "$(find polytope_ros)/../../output/polytope_ros/debug"};
    int optimizer_dynamic_residual_diagnostics_max_records_{20};
    std::map<std::string, OptimizerCostWeights> optimizer_algorithm_cost_weights_;
    double optimizer_capability_near_weight_{50.0};
    double optimizer_capability_over_weight_{500.0};
    double optimizer_capability_over4_weight_{5000.0};
    bool optimizer_constrain_orientation_{true};
    bool optimizer_verbose_{false};
    int single_point_optimizer_max_iterations_per_waypoint_{60};
    int single_point_optimizer_max_trajectory_iterations_{30};
    double single_point_optimizer_pose_gain_{0.9};
    double single_point_optimizer_nullspace_step_size_{0.01};
    double single_point_optimizer_max_joint_update_norm_{0.08};
    double single_point_optimizer_pose_tolerance_{5e-4};
    double single_point_optimizer_capability_weight_{1.0};
    double single_point_optimizer_manipulability_weight_{1.0};
    double single_point_optimizer_joint_limit_weight_{1e-3};
    Eigen::VectorXd single_point_optimizer_joint_limit_dof_weights_;
    double single_point_optimizer_smoothness_weight_{5e-2};
    double single_point_optimizer_velocity_weight_{0.0};
    double single_point_optimizer_base_velocity_weight_{0.0};
    double single_point_optimizer_nominal_weight_{5e-2};
    double single_point_optimizer_base_spectral_energy_weight_{0.0};
    double single_point_optimizer_base_spectral_cutoff_ratio_{0.35};
    double single_point_optimizer_base_spectral_power_{2.0};
    double single_point_optimizer_base_spectral_metric_x_{1.0};
    double single_point_optimizer_base_spectral_metric_y_{1.0};
    double single_point_optimizer_base_spectral_metric_yaw_{0.5};
    double single_point_optimizer_collision_weight_{0.0};
    double single_point_optimizer_collision_safe_distance_{0.08};
    std::vector<polytope_wx::CollisionSphereSpec>
        single_point_optimizer_collision_basket_spheres_;
    std::vector<polytope_wx::CollisionSphereSpec>
        single_point_optimizer_collision_body_spheres_;
    double single_point_optimizer_capability_alpha_{0.8};
    bool single_point_optimizer_capability_exponent_limit_enabled_{true};
    bool single_point_optimizer_use_dynamic_residual_force_polytope_{false};
    double single_point_optimizer_capability_near_weight_{50.0};
    double single_point_optimizer_capability_over_weight_{500.0};
    double single_point_optimizer_capability_over4_weight_{5000.0};
    bool single_point_optimizer_constrain_orientation_{true};
    bool single_point_optimizer_verbose_{false};

    std::vector<double> planned_waypoint_times_;
    std::vector<WholeBodyTarget> planned_whole_body_targets_;
    std::vector<Eigen::Affine3d> planned_waypoint_poses_;
    Eigen::VectorXd last_optimized_model_configuration_;
    std::array<std::vector<double>, 2> comparison_planned_waypoint_times_;
    std::array<std::vector<WholeBodyTarget>, 2>
        comparison_planned_whole_body_targets_;
    std::array<Eigen::VectorXd, 2> comparison_last_optimized_model_configuration_;
    std::array<bool, 2> comparison_plan_ready_{{false, false}};

    ros::Time trajectory_start_time_;
    ros::Time phase_start_time_;
    ExperimentPhase experiment_phase_{ExperimentPhase::kTrajectory};
    int comparison_run_index_{0};
    bool reset_to_home_requested_{false};
    Eigen::Affine3d return_start_pose_affine_{Eigen::Affine3d::Identity()};
    WholeBodyTarget return_start_whole_body_target_ = WholeBodyTarget::Zero();

    int optimizer_state_dim_{0};
    bool has_seed_pose_{false};
    bool has_seed_joint_positions_{false};
    bool has_manual_target_pose_{false};
    bool has_home_pose_{false};
    bool has_home_joint_positions_{false};
    bool trajectory_started_{false};
    bool planning_start_requested_{false};
    bool optimizer_initialized_{false};
    bool optimizer_initialized_for_single_point_mode_{false};
    bool optimizer_reinitialization_requested_{false};
    bool plan_ready_{false};
    bool pre_trajectory_reset_completed_{false};
    double estimated_payload_mass_kg_{0.0};
    bool has_estimated_payload_mass_{false};
    Eigen::Vector3d last_published_desired_force_{Eigen::Vector3d::Zero()};
    bool has_last_published_desired_force_{false};
    Eigen::Vector3d trajectory_start_desired_force_{Eigen::Vector3d::Zero()};
    bool has_trajectory_start_desired_force_{false};
    hrii_robot_msgs::FrankaToolState latest_franka_tool_state_;
    bool has_franka_tool_state_{false};
    bool request_vlm_mass_for_pending_start_{false};
    bool waiting_for_vlm_mass_{false};
    ros::Time vlm_mass_request_time_;
    std::string active_vlm_request_id_;
    std::set<std::string> pending_vlm_request_ids_;
    std::string last_vlm_result_json_path_;
    std::string last_vlm_model_;
    std::string last_vlm_mass_error_;
    std::string last_plan_failure_reason_;
    ros::Time last_plan_retry_time_;
    double offline_plan_retry_delay_sec_{1.0};
    Eigen::VectorXd optimizer_arm_torque_limits_override_;
    tf::TransformBroadcaster tf_broadcaster_;
};

} // namespace

int main(int argc, char** argv)
{
    ros::init(argc, argv, "target_pose_generator");

    TargetPoseGenerator generator;
    generator.spin();

    return 0;
}
