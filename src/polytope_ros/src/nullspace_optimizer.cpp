#include "polytope_ros/nullspace_optimizer.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/spatial/explog.hpp>

namespace
{

constexpr double kEpsilon = 1e-9;

} // namespace

namespace polytope_wx
{

PolytopeNullspaceOptimizer::PolytopeNullspaceOptimizer() = default;

PolytopeNullspaceOptimizer::~PolytopeNullspaceOptimizer() = default;

bool PolytopeNullspaceOptimizer::initialize(const NullspaceOptimizationConfig& config)
{
    try
    {
        if (config.urdf_path.empty())
        {
            throw std::runtime_error("URDF path is empty.");
        }

        if (config.ee_frame.empty())
        {
            throw std::runtime_error("End-effector frame is empty.");
        }

        pinocchio::Model full_model;
        pinocchio::urdf::buildModel(config.urdf_path, full_model);

        if (!config.locked_joint_names.empty())
        {
            std::vector<pinocchio::JointIndex> joints_to_lock;
            joints_to_lock.reserve(config.locked_joint_names.size());

            for (const std::string& joint_name : config.locked_joint_names)
            {
                if (!full_model.existJointName(joint_name))
                {
                    throw std::runtime_error("Joint to lock not found in model: " + joint_name);
                }

                joints_to_lock.push_back(full_model.getJointId(joint_name));
            }

            const Eigen::VectorXd reference_configuration = pinocchio::neutral(full_model);
            model_ = pinocchio::buildReducedModel(
                full_model,
                joints_to_lock,
                reference_configuration);
        }
        else
        {
            model_ = full_model;
        }

        data_ = std::make_unique<pinocchio::Data>(model_);

        if (model_.nq != model_.nv)
        {
            throw std::runtime_error(
                "Only nq == nv models are supported by this standalone optimizer.");
        }

        ee_frame_id_ = model_.getFrameId(config.ee_frame);
        if (ee_frame_id_ == static_cast<pinocchio::FrameIndex>(model_.nframes))
        {
            throw std::runtime_error("End-effector frame not found: " + config.ee_frame);
        }

        config_ = config;

        if (config.torque_limits.size() == model_.nv)
        {
            torque_limits_ = config.torque_limits;
        }
        else
        {
            torque_limits_ = model_.effortLimit;
        }

        for (int i = 0; i < torque_limits_.size(); ++i)
        {
            if (!std::isfinite(torque_limits_(i)) || torque_limits_(i) <= 0.0)
            {
                torque_limits_(i) = std::max(1.0, std::abs(torque_limits_(i)));
            }
        }

        lower_position_limits_ = model_.lowerPositionLimit;
        upper_position_limits_ = model_.upperPositionLimit;
        joint_limit_margins_ =
            config_.joint_limit_margin_ratio * (upper_position_limits_ - lower_position_limits_);

        polytope_tool_.set_torque_limit(torque_limits_);
        initialized_ = true;
        return true;
    }
    catch (const std::exception& e)
    {
        initialized_ = false;
        data_.reset();
        std::cerr << "[PolytopeNullspaceOptimizer] initialize failed: "
                  << e.what() << std::endl;
        return false;
    }
}

bool PolytopeNullspaceOptimizer::isInitialized() const
{
    return initialized_;
}

const NullspaceOptimizationConfig& PolytopeNullspaceOptimizer::config() const
{
    if (!initialized_)
    {
        throw std::runtime_error("Optimizer not initialized.");
    }
    return config_;
}

const pinocchio::Model& PolytopeNullspaceOptimizer::model() const
{
    if (!initialized_)
    {
        throw std::runtime_error("Optimizer not initialized.");
    }
    return model_;
}

Eigen::VectorXd PolytopeNullspaceOptimizer::defaultConfiguration() const
{
    if (!initialized_)
    {
        throw std::runtime_error("Optimizer not initialized.");
    }

    Eigen::VectorXd q = Eigen::VectorXd::Zero(model_.nq);

    for (int i = 0; i < q.size(); ++i)
    {
        const double lower = lower_position_limits_(i);
        const double upper = upper_position_limits_(i);

        if (std::isfinite(lower) && std::isfinite(upper) && lower < upper)
        {
            if (0.0 < lower || 0.0 > upper)
            {
                q(i) = 0.5 * (lower + upper);
            }
        }
    }

    return clampToJointLimits(q);
}

Eigen::MatrixXd PolytopeNullspaceOptimizer::computeTranslationalForcePolytope(
    const Eigen::VectorXd& q,
    bool gravity_adjusted) const
{
    if (!initialized_)
    {
        throw std::runtime_error("Optimizer not initialized.");
    }

    if (q.size() != model_.nq)
    {
        throw std::runtime_error("computeTranslationalForcePolytope: q dimension mismatch.");
    }

    const Eigen::VectorXd clamped_q = clampToJointLimits(q);
    const Eigen::MatrixXd translational_jacobian = computeTaskJacobian(clamped_q).topRows(3);

    if (gravity_adjusted)
    {
        const Eigen::VectorXd adjusted_limits =
            buildGravityAdjustedTorqueLimits(clamped_q);
        return polytope_tool_.compute_polytope_vertices(translational_jacobian, &adjusted_limits);
    }

    return polytope_tool_.compute_polytope_vertices(translational_jacobian);
}

Eigen::Affine3d
PolytopeNullspaceOptimizer::computeForwardKinematics(const Eigen::VectorXd& q) const
{
    if (!initialized_ || !data_)
    {
        throw std::runtime_error("Optimizer not initialized.");
    }

    pinocchio::forwardKinematics(model_, *data_, q);
    pinocchio::updateFramePlacements(model_, *data_);

    return Eigen::Affine3d(data_->oMf[ee_frame_id_].toHomogeneousMatrix());
}

Eigen::MatrixXd
PolytopeNullspaceOptimizer::computeTaskJacobian(const Eigen::VectorXd& q) const
{
    if (!initialized_ || !data_)
    {
        throw std::runtime_error("Optimizer not initialized.");
    }

    pinocchio::computeJointJacobians(model_, *data_, q);
    pinocchio::updateFramePlacements(model_, *data_);

    Eigen::MatrixXd jacobian(6, model_.nv);
    jacobian.setZero();

    pinocchio::getFrameJacobian(
        model_,
        *data_,
        ee_frame_id_,
        pinocchio::LOCAL_WORLD_ALIGNED,
        jacobian);

    if (config_.constrain_orientation)
    {
        return jacobian;
    }

    return jacobian.topRows(3);
}

Eigen::Matrix<double, 6, 1>
PolytopeNullspaceOptimizer::computePoseError(const Eigen::VectorXd& q,
                                             const Eigen::Affine3d& target_pose) const
{
    computeForwardKinematics(q);

    const pinocchio::SE3 current_pose = data_->oMf[ee_frame_id_];
    const pinocchio::SE3 desired_pose(target_pose.linear(), target_pose.translation());
    const pinocchio::Motion error_motion = pinocchio::log6(current_pose.inverse() * desired_pose);

    Eigen::Matrix<double, 6, 1> pose_error = error_motion.toVector();
    if (!config_.constrain_orientation)
    {
        pose_error.tail<3>().setZero();
    }

    return pose_error;
}

Eigen::VectorXd
PolytopeNullspaceOptimizer::buildGravityAdjustedTorqueLimits(const Eigen::VectorXd& q) const
{
    if (!initialized_ || !data_)
    {
        throw std::runtime_error("Optimizer not initialized.");
    }

    const Eigen::VectorXd zero_velocity = Eigen::VectorXd::Zero(model_.nv);
    pinocchio::forwardKinematics(model_, *data_, q, zero_velocity);

    const Eigen::VectorXd gravity = pinocchio::computeGeneralizedGravity(model_, *data_, q);

    Eigen::VectorXd adjusted_limits(2 * model_.nv);
    adjusted_limits.head(model_.nv) = torque_limits_ - gravity;
    adjusted_limits.tail(model_.nv) = -torque_limits_ - gravity;

    return adjusted_limits;
}

Eigen::VectorXd
PolytopeNullspaceOptimizer::clampToJointLimits(const Eigen::VectorXd& q) const
{
    Eigen::VectorXd clamped = q;

    for (int i = 0; i < clamped.size(); ++i)
    {
        const double lower = lower_position_limits_(i);
        const double upper = upper_position_limits_(i);

        if (!std::isfinite(lower) || !std::isfinite(upper) || lower >= upper)
        {
            continue;
        }

        clamped(i) = std::min(std::max(clamped(i), lower + 1e-6), upper - 1e-6);
    }

    return clamped;
}

double PolytopeNullspaceOptimizer::computeManipulability(const Eigen::VectorXd& q) const
{
    const Eigen::MatrixXd translational_jacobian = computeTaskJacobian(q).topRows(3);
    const Eigen::Matrix3d gram =
        translational_jacobian * translational_jacobian.transpose() +
        config_.manipulability_regularization * Eigen::Matrix3d::Identity();

    return std::sqrt(
        std::max(config_.manipulability_epsilon, gram.determinant()));
}

PolytopeNullspaceOptimizer::CostBreakdown
PolytopeNullspaceOptimizer::evaluateCost(const Eigen::VectorXd& q,
                                         const Eigen::Vector3d& desired_force,
                                         const Eigen::VectorXd& nominal_q,
                                         const Eigen::VectorXd* previous_q) const
{
    CostBreakdown breakdown;

    const double required_force = desired_force.norm();
    breakdown.required_force = required_force;

    if (required_force > kEpsilon)
    {
        const Eigen::MatrixXd translational_jacobian =
            computeTaskJacobian(q).topRows(3);
        const Eigen::VectorXd adjusted_limits =
            buildGravityAdjustedTorqueLimits(q);

        breakdown.force_capacity = polytope_tool_.compute_force_capacity(
            translational_jacobian,
            desired_force,
            adjusted_limits);
        const double force_ball_clearance =
            polytope_tool_.compute_force_ball_signed_clearance(
                translational_jacobian,
                desired_force,
                0.0,
                adjusted_limits,
                config_.capability_alpha);
        breakdown.capability_cost =
            polytope_tool_.compute_clearance_based_capability_cost(
                force_ball_clearance,
                required_force);
    }

    breakdown.manipulability_measure = computeManipulability(q);
    breakdown.manipulability_cost =
        -std::log(std::max(
            config_.manipulability_epsilon,
            breakdown.manipulability_measure));

    for (int i = 0; i < q.size(); ++i)
    {
        const double lower = lower_position_limits_(i);
        const double upper = upper_position_limits_(i);
        const double margin = joint_limit_margins_(i);

        if (!std::isfinite(lower) || !std::isfinite(upper) || lower >= upper)
        {
            continue;
        }

        if (margin <= 0.0)
        {
            continue;
        }

        const double lower_guard = lower + margin;
        const double upper_guard = upper - margin;
        const double eps = 1e-6;

        if (q(i) < lower_guard)
        {
            const double distance = lower_guard - q(i);
            breakdown.joint_limit_cost += 1.0 / (distance * distance + eps);
        }

        if (q(i) > upper_guard)
        {
            const double distance = q(i) - upper_guard;
            breakdown.joint_limit_cost += 1.0 / (distance * distance + eps);
        }
    }

    if (previous_q != nullptr && previous_q->size() == q.size())
    {
        breakdown.smoothness_cost = 0.5 * (q - *previous_q).squaredNorm();
    }

    breakdown.nominal_cost = 0.5 * (q - nominal_q).squaredNorm();

    breakdown.objective_value =
        config_.capability_weight * breakdown.capability_cost +
        config_.manipulability_weight * breakdown.manipulability_cost +
        config_.joint_limit_weight * breakdown.joint_limit_cost +
        config_.smoothness_weight * breakdown.smoothness_cost +
        config_.nominal_weight * breakdown.nominal_cost;

    return breakdown;
}

Eigen::VectorXd
PolytopeNullspaceOptimizer::computeObjectiveGradient(
    const Eigen::VectorXd& q,
    const Eigen::Vector3d& desired_force,
    const Eigen::VectorXd& nominal_q,
    const Eigen::VectorXd* previous_q) const
{
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(model_.nv);

    for (int i = 0; i < model_.nv; ++i)
    {
        Eigen::VectorXd tangent_plus = Eigen::VectorXd::Zero(model_.nv);
        Eigen::VectorXd tangent_minus = Eigen::VectorXd::Zero(model_.nv);
        tangent_plus(i) = config_.finite_difference_step;
        tangent_minus(i) = -config_.finite_difference_step;

        const Eigen::VectorXd q_plus =
            clampToJointLimits(pinocchio::integrate(model_, q, tangent_plus));
        const Eigen::VectorXd q_minus =
            clampToJointLimits(pinocchio::integrate(model_, q, tangent_minus));

        const double value_plus =
            evaluateCost(q_plus, desired_force, nominal_q, previous_q).objective_value;
        const double value_minus =
            evaluateCost(q_minus, desired_force, nominal_q, previous_q).objective_value;

        gradient(i) =
            (value_plus - value_minus) / (2.0 * config_.finite_difference_step);
    }

    return gradient;
}

WaypointOptimizationMetrics
PolytopeNullspaceOptimizer::optimizeWaypoint(
    const Eigen::VectorXd& seed_q,
    const Eigen::Affine3d& target_pose,
    const Eigen::Vector3d& desired_force,
    const Eigen::VectorXd& nominal_q,
    const Eigen::VectorXd* previous_q,
    Eigen::VectorXd& optimized_q) const
{
    optimized_q = clampToJointLimits(seed_q);

    WaypointOptimizationMetrics metrics;

    for (int iteration = 0; iteration < config_.max_iterations_per_waypoint; ++iteration)
    {
        const Eigen::Matrix<double, 6, 1> pose_error =
            computePoseError(optimized_q, target_pose);
        const Eigen::MatrixXd task_jacobian = computeTaskJacobian(optimized_q);
        const Eigen::VectorXd task_error =
            config_.constrain_orientation ? pose_error : pose_error.head(3);

        const Eigen::MatrixXd task_jacobian_pinv =
            task_jacobian.completeOrthogonalDecomposition().pseudoInverse();
        const Eigen::MatrixXd nullspace_projector =
            Eigen::MatrixXd::Identity(model_.nv, model_.nv) -
            task_jacobian_pinv * task_jacobian;

        const Eigen::VectorXd objective_gradient =
            computeObjectiveGradient(optimized_q, desired_force, nominal_q, previous_q);

        Eigen::VectorXd delta_q =
            config_.pose_gain * (task_jacobian_pinv * task_error) -
            config_.nullspace_step_size * (nullspace_projector * objective_gradient);

        const double delta_norm = delta_q.norm();
        if (delta_norm > config_.max_joint_update_norm && delta_norm > kEpsilon)
        {
            delta_q *= config_.max_joint_update_norm / delta_norm;
        }

        optimized_q = clampToJointLimits(pinocchio::integrate(model_, optimized_q, delta_q));

        const double pose_error_norm = task_error.norm();
        metrics.iterations = iteration + 1;

        if (pose_error_norm < config_.pose_tolerance &&
            (nullspace_projector * objective_gradient).norm() < 1e-4)
        {
            break;
        }
    }

    const CostBreakdown breakdown =
        evaluateCost(optimized_q, desired_force, nominal_q, previous_q);
    const Eigen::Matrix<double, 6, 1> final_pose_error =
        computePoseError(optimized_q, target_pose);

    metrics.objective_value = breakdown.objective_value;
    metrics.force_capacity = breakdown.force_capacity;
    metrics.required_force = breakdown.required_force;
    metrics.capability_cost = breakdown.capability_cost;
    metrics.manipulability_measure = breakdown.manipulability_measure;
    metrics.manipulability_cost = breakdown.manipulability_cost;
    metrics.joint_limit_cost = breakdown.joint_limit_cost;
    metrics.smoothness_cost = breakdown.smoothness_cost;
    metrics.nominal_cost = breakdown.nominal_cost;
    metrics.pose_error_norm =
        (config_.constrain_orientation ? final_pose_error : final_pose_error.head(3)).norm();

    return metrics;
}

TrajectoryOptimizationResult
PolytopeNullspaceOptimizer::optimize(const TrajectoryOptimizationInput& input) const
{
    if (!initialized_)
    {
        throw std::runtime_error("Optimizer not initialized.");
    }

    if (input.target_poses.empty())
    {
        throw std::runtime_error("Target pose trajectory is empty.");
    }

    const std::size_t horizon = input.target_poses.size();

    if (input.desired_forces.size() != horizon)
    {
        throw std::runtime_error("Desired force trajectory size mismatch.");
    }

    if (input.nominal_joint_trajectory.size() != horizon)
    {
        throw std::runtime_error("Nominal joint trajectory size mismatch.");
    }

    TrajectoryOptimizationResult result;
    result.optimized_joint_trajectory.reserve(horizon);
    result.waypoint_metrics.reserve(horizon);

    bool all_waypoints_valid = true;

    for (std::size_t k = 0; k < horizon; ++k)
    {
        const Eigen::VectorXd& nominal_q = input.nominal_joint_trajectory[k];
        if (nominal_q.size() != model_.nq)
        {
            throw std::runtime_error("Nominal joint vector dimension mismatch.");
        }

        const Eigen::VectorXd& seed_q =
            k > 0 ? result.optimized_joint_trajectory.back() : nominal_q;
        Eigen::VectorXd optimized_q = seed_q;
        const Eigen::VectorXd* previous_q =
            k > 0 ? &result.optimized_joint_trajectory.back() : nullptr;

        const WaypointOptimizationMetrics metrics = optimizeWaypoint(
            seed_q,
            input.target_poses[k],
            input.desired_forces[k],
            nominal_q,
            previous_q,
            optimized_q);

        if (metrics.pose_error_norm > 10.0 * config_.pose_tolerance)
        {
            all_waypoints_valid = false;
        }

        if (config_.verbose)
        {
            std::cout << "[PolytopeNullspaceOptimizer] waypoint " << k
                      << " | pose_error=" << metrics.pose_error_norm
                      << " | force_capacity=" << metrics.force_capacity
                      << " | manipulability=" << metrics.manipulability_measure
                      << " | objective=" << metrics.objective_value
                      << std::endl;
        }

        result.optimized_joint_trajectory.push_back(optimized_q);
        result.waypoint_metrics.push_back(metrics);
    }

    result.success = all_waypoints_valid;

    std::ostringstream message_stream;
    message_stream << "Optimized " << horizon
                   << " waypoints with polytope-based nullspace shaping.";
    if (!all_waypoints_valid)
    {
        message_stream << " Some waypoints ended with pose error above tolerance.";
    }
    result.message = message_stream.str();

    return result;
}

SingleWaypointOptimizationResult
PolytopeNullspaceOptimizer::optimizeSingleWaypoint(
    const Eigen::VectorXd& seed_q,
    const Eigen::Affine3d& target_pose,
    const Eigen::Vector3d& desired_force,
    const Eigen::VectorXd& nominal_q,
    const Eigen::VectorXd* previous_q) const
{
    if (!initialized_)
    {
        throw std::runtime_error("Optimizer not initialized.");
    }

    if (seed_q.size() != model_.nq)
    {
        throw std::runtime_error("Seed joint vector dimension mismatch.");
    }

    if (nominal_q.size() != model_.nq)
    {
        throw std::runtime_error("Nominal joint vector dimension mismatch.");
    }

    if (previous_q != nullptr && previous_q->size() != model_.nq)
    {
        throw std::runtime_error("Previous joint vector dimension mismatch.");
    }

    SingleWaypointOptimizationResult result;
    result.optimized_q = seed_q;
    result.metrics = optimizeWaypoint(
        seed_q,
        target_pose,
        desired_force,
        nominal_q,
        previous_q,
        result.optimized_q);
    result.success = result.metrics.pose_error_norm <= 10.0 * config_.pose_tolerance;

    std::ostringstream message_stream;
    message_stream << "Optimized one waypoint with polytope-based nullspace shaping.";
    if (!result.success)
    {
        message_stream << " Final pose error is above tolerance.";
    }
    result.message = message_stream.str();

    return result;
}

NullspaceGradientStepResult
PolytopeNullspaceOptimizer::computeNullspaceGradientStep(
    const Eigen::VectorXd& current_q,
    const Eigen::Vector3d& desired_force,
    const Eigen::VectorXd& nominal_q,
    const Eigen::VectorXd* previous_q) const
{
    if (!initialized_)
    {
        throw std::runtime_error("Optimizer not initialized.");
    }

    if (current_q.size() != model_.nq)
    {
        throw std::runtime_error("Current joint vector dimension mismatch.");
    }

    if (nominal_q.size() != model_.nq)
    {
        throw std::runtime_error("Nominal joint vector dimension mismatch.");
    }

    if (previous_q != nullptr && previous_q->size() != model_.nq)
    {
        throw std::runtime_error("Previous joint vector dimension mismatch.");
    }

    NullspaceGradientStepResult result;
    const Eigen::VectorXd clamped_q = clampToJointLimits(current_q);

    result.objective_gradient =
        computeObjectiveGradient(clamped_q, desired_force, nominal_q, previous_q);
    result.projected_gradient = Eigen::VectorXd::Zero(model_.nv);
    result.delta_q = Eigen::VectorXd::Zero(model_.nv);
    result.target_q = clamped_q;

    const CostBreakdown breakdown =
        evaluateCost(clamped_q, desired_force, nominal_q, previous_q);
    result.metrics.objective_value = breakdown.objective_value;
    result.metrics.force_capacity = breakdown.force_capacity;
    result.metrics.required_force = breakdown.required_force;
    result.metrics.capability_cost = breakdown.capability_cost;
    result.metrics.manipulability_measure = breakdown.manipulability_measure;
    result.metrics.manipulability_cost = breakdown.manipulability_cost;
    result.metrics.joint_limit_cost = breakdown.joint_limit_cost;
    result.metrics.smoothness_cost = breakdown.smoothness_cost;
    result.metrics.nominal_cost = breakdown.nominal_cost;
    result.metrics.pose_error_norm = 0.0;
    result.metrics.iterations = 1;

    result.success =
        result.objective_gradient.allFinite() &&
        result.target_q.allFinite();

    std::ostringstream message_stream;
    message_stream << "Computed one nullspace objective gradient.";
    if (!result.success)
    {
        message_stream << " Gradient result contains non-finite values.";
    }
    result.message = message_stream.str();

    return result;
}

} // namespace polytope_wx
