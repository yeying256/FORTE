#include "polytope_ros/whole_body_optimizer.h"
#include "polytope_ros/rfp_comparison_costs.h"

#include <casadi/casadi.hpp>

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/types.h>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/autodiff/casadi-algo.hpp>
#include <pinocchio/autodiff/casadi.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/spatial/explog.hpp>

namespace
{

constexpr double kEpsilon = 1e-9;
constexpr int kBasePlanarDof = 3;
constexpr double kCapabilityExponentScale = 3.0;
constexpr double kCapabilityMaxExponent = 40.0;
constexpr double kStrictWaypointPoseValidationMultiplier = 2.0;

using CasadiScalar = casadi::SX;
using CasadiModel = pinocchio::ModelTpl<CasadiScalar>;
using CasadiData = pinocchio::DataTpl<CasadiScalar>;
using CasadiVectorXs = Eigen::Matrix<CasadiScalar, Eigen::Dynamic, 1>;
using CasadiMatrixXs = Eigen::Matrix<CasadiScalar, Eigen::Dynamic, Eigen::Dynamic>;
using CasadiVector3 = Eigen::Matrix<CasadiScalar, 3, 1>;
using CasadiVector6 = Eigen::Matrix<CasadiScalar, 6, 1>;
using CasadiMatrix3 = Eigen::Matrix<CasadiScalar, 3, 3>;
using CasadiMatrix6x3 = Eigen::Matrix<CasadiScalar, 6, 3>;
using CasadiSE3 = pinocchio::SE3Tpl<CasadiScalar>;

CasadiScalar sx(double value)
{
    return CasadiScalar(value);
}

double computeBoundedClearanceCapabilityCost(
    double signed_clearance,
    double normalization_scale,
    bool exponent_limit_enabled,
    double eps = 1e-6)
{
    const double safe_scale = std::max(std::abs(normalization_scale), eps);
    const double normalized_violation =
        std::max(0.0, -signed_clearance / safe_scale);
    const double exponent = kCapabilityExponentScale * normalized_violation;
    return std::expm1(
        exponent_limit_enabled
            ? std::min(kCapabilityMaxExponent, exponent)
            : exponent);
}

bool ensureDirectoryExists(const std::string& directory)
{
    if (directory.empty())
    {
        return false;
    }

    std::string current;
    for (std::size_t i = 0; i < directory.size(); ++i)
    {
        current.push_back(directory[i]);
        if (directory[i] != '/' && i + 1 != directory.size())
        {
            continue;
        }
        if (current.empty() || current == "/")
        {
            continue;
        }
        struct stat info;
        if (stat(current.c_str(), &info) == 0)
        {
            if ((info.st_mode & S_IFDIR) == 0)
            {
                return false;
            }
            continue;
        }
        if (mkdir(current.c_str(), 0755) != 0)
        {
            return false;
        }
    }
    return true;
}

std::string timestampForFilename()
{
    const std::time_t now = std::time(nullptr);
    std::tm time_info;
    localtime_r(&now, &time_info);
    std::ostringstream stream;
    stream << std::put_time(&time_info, "%Y%m%d_%H%M%S");
    return stream.str();
}

void writeVectorCsv(std::ostream& stream, const Eigen::VectorXd& vector)
{
    for (int i = 0; i < vector.size(); ++i)
    {
        if (i > 0)
        {
            stream << ';';
        }
        stream << vector(i);
    }
}

void writeMatrixCsv(std::ostream& stream, const Eigen::MatrixXd& matrix)
{
    for (int row = 0; row < matrix.rows(); ++row)
    {
        for (int col = 0; col < matrix.cols(); ++col)
        {
            if (row > 0 || col > 0)
            {
                stream << ';';
            }
            stream << matrix(row, col);
        }
    }
}

template <typename Derived>
casadi::SX toCasadiVector(const Eigen::MatrixBase<Derived>& eigen_vector)
{
    casadi::SX out = casadi::SX::zeros(eigen_vector.rows(), eigen_vector.cols());
    for (int i = 0; i < eigen_vector.rows(); ++i)
    {
        for (int j = 0; j < eigen_vector.cols(); ++j)
        {
            out(i, j) = eigen_vector(i, j);
        }
    }
    return out;
}

CasadiVectorXs constantCasadiVector(const Eigen::VectorXd& values)
{
    CasadiVectorXs out(values.size());
    for (int i = 0; i < values.size(); ++i)
    {
        out(i) = sx(values(i));
    }
    return out;
}

CasadiVector3 constantCasadiVector3(const Eigen::Vector3d& values)
{
    CasadiVector3 out;
    for (int i = 0; i < 3; ++i)
    {
        out(i) = sx(values(i));
    }
    return out;
}

CasadiMatrix3 constantCasadiMatrix3(const Eigen::Matrix3d& values)
{
    CasadiMatrix3 out;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            out(i, j) = sx(values(i, j));
        }
    }
    return out;
}

CasadiVectorXs extractSymbolicState(
    const casadi::SX& X,
    int state_dim,
    std::size_t waypoint_index)
{
    CasadiVectorXs state(state_dim);
    const int offset = static_cast<int>(waypoint_index) * state_dim;
    for (int i = 0; i < state_dim; ++i)
    {
        state(i) = X(offset + i);
    }
    return state;
}

CasadiScalar squaredNorm(const CasadiVectorXs& vector)
{
    CasadiScalar sum = sx(0.0);
    for (int i = 0; i < vector.size(); ++i)
    {
        sum += vector(i) * vector(i);
    }
    return sum;
}

CasadiScalar squaredNorm3(const CasadiVector3& vector)
{
    CasadiScalar sum = sx(0.0);
    for (int i = 0; i < 3; ++i)
    {
        sum += vector(i) * vector(i);
    }
    return sum;
}

CasadiScalar weightedSquaredNorm(
    const CasadiVector3& vector,
    const Eigen::Vector3d& metric)
{
    CasadiScalar sum = sx(0.0);
    for (int i = 0; i < 3; ++i)
    {
        sum += sx(metric(i)) * vector(i) * vector(i);
    }
    return sum;
}

CasadiScalar determinant3x3(const CasadiMatrix3& matrix)
{
    return
        matrix(0, 0) * (matrix(1, 1) * matrix(2, 2) - matrix(1, 2) * matrix(2, 1)) -
        matrix(0, 1) * (matrix(1, 0) * matrix(2, 2) - matrix(1, 2) * matrix(2, 0)) +
        matrix(0, 2) * (matrix(1, 0) * matrix(2, 1) - matrix(1, 1) * matrix(2, 0));
}

double clampUnitInterval(double value)
{
    return std::max(0.0, std::min(1.0, value));
}

Eigen::VectorXd computeOrthonormalDct(const Eigen::VectorXd& signal)
{
    const int sample_count = signal.size();
    if (sample_count <= 0)
    {
        return Eigen::VectorXd();
    }

    Eigen::VectorXd coefficients = Eigen::VectorXd::Zero(sample_count);
    const double inv_sample_count = 1.0 / static_cast<double>(sample_count);

    for (int k = 0; k < sample_count; ++k)
    {
        double coefficient = 0.0;
        for (int n = 0; n < sample_count; ++n)
        {
            coefficient +=
                signal(n) *
                std::cos(
                    M_PI *
                    (static_cast<double>(n) + 0.5) *
                    static_cast<double>(k) *
                    inv_sample_count);
        }

        const double scale =
            k == 0
                ? std::sqrt(inv_sample_count)
                : std::sqrt(2.0 * inv_sample_count);
        coefficients(k) = scale * coefficient;
    }

    return coefficients;
}

double estimateAverageWaypointDtSec(
    const std::vector<double>& waypoint_times_sec,
    int sample_count)
{
    if (sample_count < 2 || waypoint_times_sec.size() != static_cast<std::size_t>(sample_count))
    {
        return 1.0;
    }

    double positive_dt_sum = 0.0;
    int positive_dt_count = 0;
    for (int i = 1; i < sample_count; ++i)
    {
        const double dt = waypoint_times_sec[static_cast<std::size_t>(i)] -
                          waypoint_times_sec[static_cast<std::size_t>(i - 1)];
        if (dt > kEpsilon)
        {
            positive_dt_sum += dt;
            ++positive_dt_count;
        }
    }

    if (positive_dt_count <= 0)
    {
        return 1.0;
    }

    return positive_dt_sum / static_cast<double>(positive_dt_count);
}

double waypointPreviousDtSec(
    const std::vector<double>& waypoint_times_sec,
    std::size_t waypoint_index)
{
    if (waypoint_index == 0 ||
        waypoint_index >= waypoint_times_sec.size())
    {
        return 1.0;
    }

    const double dt =
        waypoint_times_sec[waypoint_index] -
        waypoint_times_sec[waypoint_index - 1];
    if (!std::isfinite(dt) || dt <= kEpsilon)
    {
        return 1.0;
    }
    return dt;
}

std::vector<Eigen::VectorXd> unpackDecisionTrajectory(
    const std::vector<double>& flat_values,
    int state_dim,
    std::size_t horizon)
{
    if (flat_values.size() != static_cast<std::size_t>(state_dim) * horizon)
    {
        throw std::runtime_error("CasADi decision vector size mismatch.");
    }

    std::vector<Eigen::VectorXd> trajectory;
    trajectory.reserve(horizon);
    for (std::size_t k = 0; k < horizon; ++k)
    {
        Eigen::VectorXd state(state_dim);
        for (int i = 0; i < state_dim; ++i)
        {
            state(i) = flat_values[k * static_cast<std::size_t>(state_dim) + i];
        }
        trajectory.push_back(state);
    }

    return trajectory;
}

casadi::DM denseColumn(const std::vector<double>& values)
{
    casadi::DM column = casadi::DM::zeros(static_cast<casadi_int>(values.size()), 1);
    std::copy(values.begin(), values.end(), column.nonzeros().begin());
    return column;
}

casadi::DM denseColumn(const Eigen::VectorXd& values)
{
    casadi::DM column = casadi::DM::zeros(values.size(), 1);
    std::copy(values.data(), values.data() + values.size(), column.nonzeros().begin());
    return column;
}

} // namespace

namespace polytope_wx
{

WholeBodyOptimizationConfig::WholeBodyOptimizationConfig()
    : base_frame("moca_base_footprint"),
      base_smoothness_metric(Eigen::Vector3d(1.0, 1.0, 0.5)),
      base_nominal_metric(Eigen::Vector3d(1.0, 1.0, 0.5)),
      base_spectral_energy_metric(Eigen::Vector3d(1.0, 1.0, 0.5)),
      comparison_cone_axis_world(Eigen::Vector3d::UnitZ()),
      base_lower_limits(Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN())),
      base_upper_limits(Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN()))
{
    joint_limit_dof_weights = Eigen::VectorXd::Ones(10);
    joint_limit_dof_weights.head(3).setZero();
}

class WholeBodyTrajectoryObjectiveCallback : public casadi::Callback
{
public:
    WholeBodyTrajectoryObjectiveCallback(
        const std::string& name,
        const WholeBodyPolytopeOptimizer& optimizer,
        const WholeBodyTrajectoryOptimizationInput& input)
        : optimizer_(optimizer),
          input_(input),
          horizon_(input.target_poses.size()),
          decision_dim_(optimizer.state_dim_ * static_cast<int>(input.target_poses.size()))
    {
        casadi::Dict opts;
        opts["enable_fd"] = true;
        opts["fd_method"] = "forward";
        construct(name, opts);
    }

    casadi_int get_n_in() override
    {
        return 1;
    }

    casadi_int get_n_out() override
    {
        return 1;
    }

    casadi::Sparsity get_sparsity_in(casadi_int i) override
    {
        if (i != 0)
        {
            throw std::runtime_error("WholeBodyTrajectoryObjectiveCallback: invalid input index.");
        }
        return casadi::Sparsity::dense(decision_dim_, 1);
    }

    casadi::Sparsity get_sparsity_out(casadi_int i) override
    {
        if (i != 0)
        {
            throw std::runtime_error("WholeBodyTrajectoryObjectiveCallback: invalid output index.");
        }
        return casadi::Sparsity::dense(1, 1);
    }

    std::vector<casadi::DM> eval(const std::vector<casadi::DM>& arg) const override
    {
        const std::vector<Eigen::VectorXd> trajectory =
            unpackDecisionTrajectory(
                arg.at(0).get_nonzeros<double>(),
                optimizer_.state_dim_,
                horizon_);

        casadi::DM objective = casadi::DM::zeros(1, 1);
        objective.nonzeros().front() =
            optimizer_.evaluateTrajectoryObjective(trajectory, input_);
        return {objective};
    }

private:
    const WholeBodyPolytopeOptimizer& optimizer_;
    WholeBodyTrajectoryOptimizationInput input_;
    std::size_t horizon_{0};
    int decision_dim_{0};
};

class WholeBodyTrajectoryConstraintCallback : public casadi::Callback
{
public:
    WholeBodyTrajectoryConstraintCallback(
        const std::string& name,
        const WholeBodyPolytopeOptimizer& optimizer,
        const WholeBodyTrajectoryOptimizationInput& input)
        : optimizer_(optimizer),
          input_(input),
          horizon_(input.target_poses.size()),
          task_dim_(optimizer.config_.constrain_orientation ? 6 : 3),
          decision_dim_(optimizer.state_dim_ * static_cast<int>(input.target_poses.size())),
          constraint_dim_(
              static_cast<int>(input.target_poses.size()) * task_dim_ +
              optimizer.state_dim_)
    {
        casadi::Dict opts;
        opts["enable_fd"] = true;
        opts["fd_method"] = "forward";
        construct(name, opts);
    }

    casadi_int get_n_in() override
    {
        return 1;
    }

    casadi_int get_n_out() override
    {
        return 1;
    }

    casadi::Sparsity get_sparsity_in(casadi_int i) override
    {
        if (i != 0)
        {
            throw std::runtime_error("WholeBodyTrajectoryConstraintCallback: invalid input index.");
        }
        return casadi::Sparsity::dense(decision_dim_, 1);
    }

    casadi::Sparsity get_sparsity_out(casadi_int i) override
    {
        if (i != 0)
        {
            throw std::runtime_error("WholeBodyTrajectoryConstraintCallback: invalid output index.");
        }
        return casadi::Sparsity::dense(constraint_dim_, 1);
    }

    std::vector<casadi::DM> eval(const std::vector<casadi::DM>& arg) const override
    {
        const std::vector<Eigen::VectorXd> trajectory =
            unpackDecisionTrajectory(
                arg.at(0).get_nonzeros<double>(),
                optimizer_.state_dim_,
                horizon_);

        casadi::DM constraints = casadi::DM::zeros(constraint_dim_, 1);
        std::vector<double>& values = constraints.nonzeros();

        int constraint_index = 0;
        for (std::size_t k = 0; k < horizon_; ++k)
        {
            const Eigen::VectorXd clamped_state =
                optimizer_.clampToStateLimits(trajectory[k]);
            const Eigen::Matrix<double, 6, 1> pose_error =
                optimizer_.computePoseError(clamped_state, input_.target_poses[k]);

            if (optimizer_.config_.constrain_orientation)
            {
                for (int i = 0; i < 6; ++i)
                {
                    values[constraint_index++] = pose_error(i);
                }
            }
            else
            {
                for (int i = 0; i < 3; ++i)
                {
                    values[constraint_index++] = pose_error(i);
                }
            }
        }

        const Eigen::VectorXd first_state =
            optimizer_.clampToStateLimits(trajectory.front());
        const Eigen::VectorXd first_nominal =
            optimizer_.clampToStateLimits(input_.nominal_state_trajectory.front());
        const Eigen::Vector3d first_base_error =
            optimizer_.baseDifference(
                optimizer_.baseState(first_state),
                optimizer_.baseState(first_nominal));
        const Eigen::VectorXd first_arm_error =
            optimizer_.armState(first_state) -
            optimizer_.armState(first_nominal);

        for (int i = 0; i < kBasePlanarDof; ++i)
        {
            values[constraint_index++] = first_base_error(i);
        }
        for (int i = 0; i < first_arm_error.size(); ++i)
        {
            values[constraint_index++] = first_arm_error(i);
        }

        return {constraints};
    }

private:
    const WholeBodyPolytopeOptimizer& optimizer_;
    WholeBodyTrajectoryOptimizationInput input_;
    std::size_t horizon_{0};
    int task_dim_{0};
    int decision_dim_{0};
    int constraint_dim_{0};
};

WholeBodyPolytopeOptimizer::WholeBodyPolytopeOptimizer() = default;

WholeBodyPolytopeOptimizer::~WholeBodyPolytopeOptimizer() = default;

bool WholeBodyPolytopeOptimizer::initialize(const WholeBodyOptimizationConfig& config)
{
    try
    {
        if (config.urdf_path.empty())
        {
            throw std::runtime_error("URDF path is empty.");
        }

        if (config.base_frame.empty())
        {
            throw std::runtime_error("Base frame is empty.");
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
            arm_model_ = pinocchio::buildReducedModel(
                full_model,
                joints_to_lock,
                reference_configuration);
        }
        else
        {
            arm_model_ = full_model;
        }

        if (arm_model_.nq != arm_model_.nv)
        {
            throw std::runtime_error(
                "Only nq == nv models are supported by this whole-body optimizer.");
        }

        data_ = std::make_unique<pinocchio::Data>(arm_model_);

        base_frame_id_ = arm_model_.getFrameId(config.base_frame);
        if (base_frame_id_ == static_cast<pinocchio::FrameIndex>(arm_model_.nframes))
        {
            throw std::runtime_error("Base frame not found: " + config.base_frame);
        }

        ee_frame_id_ = arm_model_.getFrameId(config.ee_frame);
        if (ee_frame_id_ == static_cast<pinocchio::FrameIndex>(arm_model_.nframes))
        {
            throw std::runtime_error("End-effector frame not found: " + config.ee_frame);
        }

        state_dim_ = kBasePlanarDof + static_cast<int>(arm_model_.nq);
        config_ = config;

        if (config_.joint_limit_dof_weights.size() == 0)
        {
            config_.joint_limit_dof_weights = Eigen::VectorXd::Ones(state_dim_);
            config_.joint_limit_dof_weights.head(kBasePlanarDof).setZero();
        }
        else if (config_.joint_limit_dof_weights.size() != state_dim_)
        {
            throw std::runtime_error(
                "joint_limit_dof_weights size mismatch: expected " +
                std::to_string(state_dim_) + ", got " +
                std::to_string(config_.joint_limit_dof_weights.size()));
        }

        double max_positive_joint_limit_weight = 0.0;
        for (int i = 0; i < config_.joint_limit_dof_weights.size(); ++i)
        {
            const double weight = config_.joint_limit_dof_weights(i);
            if (!std::isfinite(weight))
            {
                throw std::runtime_error(
                    "joint_limit_dof_weights contains a non-finite value at index " +
                    std::to_string(i));
            }
            config_.joint_limit_dof_weights(i) = std::max(0.0, weight);
            max_positive_joint_limit_weight =
                std::max(max_positive_joint_limit_weight, config_.joint_limit_dof_weights(i));
        }
        if (max_positive_joint_limit_weight > 0.0)
        {
            config_.joint_limit_dof_weights /= max_positive_joint_limit_weight;
        }

        config_.velocity_weight = std::max(0.0, config_.velocity_weight);
        config_.base_velocity_weight = std::max(0.0, config_.base_velocity_weight);
        config_.collision_weight = std::max(0.0, config_.collision_weight);
        config_.collision_safe_distance =
            std::max(0.0, config_.collision_safe_distance);
        basket_collision_spheres_.clear();
        body_collision_spheres_.clear();
        if (config_.collision_weight > 0.0)
        {
            basket_collision_spheres_ =
                resolveCollisionSpheres(
                    config_.collision_basket_spheres,
                    "collision_basket_spheres");
            body_collision_spheres_ =
                resolveCollisionSpheres(
                    config_.collision_body_spheres,
                    "collision_body_spheres");
            if (basket_collision_spheres_.empty() || body_collision_spheres_.empty())
            {
                throw std::runtime_error(
                    "collision_weight is positive, but collision sphere sets are incomplete.");
            }
        }

        if (config.arm_torque_limits.size() == arm_model_.nv)
        {
            arm_torque_limits_ = config.arm_torque_limits;
        }
        else
        {
            arm_torque_limits_ = arm_model_.effortLimit;
        }

        for (int i = 0; i < arm_torque_limits_.size(); ++i)
        {
            if (!std::isfinite(arm_torque_limits_(i)) || arm_torque_limits_(i) <= 0.0)
            {
                arm_torque_limits_(i) = std::max(1.0, std::abs(arm_torque_limits_(i)));
            }
        }

        lower_arm_limits_ = arm_model_.lowerPositionLimit;
        upper_arm_limits_ = arm_model_.upperPositionLimit;
        arm_joint_limit_margins_ =
            config_.joint_limit_margin_ratio * (upper_arm_limits_ - lower_arm_limits_);

        polytope_tool_.set_torque_limit(arm_torque_limits_);
        initialized_ = true;
        return true;
    }
    catch (const std::exception& e)
    {
        initialized_ = false;
        data_.reset();
        basket_collision_spheres_.clear();
        body_collision_spheres_.clear();
        std::cerr << "[WholeBodyPolytopeOptimizer] initialize failed: "
                  << e.what() << std::endl;
        return false;
    }
}

bool WholeBodyPolytopeOptimizer::isInitialized() const
{
    return initialized_;
}

const WholeBodyOptimizationConfig& WholeBodyPolytopeOptimizer::config() const
{
    if (!initialized_)
    {
        throw std::runtime_error("Whole-body optimizer not initialized.");
    }
    return config_;
}

const pinocchio::Model& WholeBodyPolytopeOptimizer::armModel() const
{
    if (!initialized_)
    {
        throw std::runtime_error("Whole-body optimizer not initialized.");
    }
    return arm_model_;
}

int WholeBodyPolytopeOptimizer::armDof() const
{
    if (!initialized_)
    {
        throw std::runtime_error("Whole-body optimizer not initialized.");
    }
    return static_cast<int>(arm_model_.nq);
}

int WholeBodyPolytopeOptimizer::stateDim() const
{
    if (!initialized_)
    {
        throw std::runtime_error("Whole-body optimizer not initialized.");
    }
    return state_dim_;
}

Eigen::VectorXd WholeBodyPolytopeOptimizer::defaultState() const
{
    if (!initialized_)
    {
        throw std::runtime_error("Whole-body optimizer not initialized.");
    }

    Eigen::VectorXd state = Eigen::VectorXd::Zero(state_dim_);
    Eigen::VectorXd arm_q = Eigen::VectorXd::Zero(arm_model_.nq);

    for (int i = 0; i < arm_q.size(); ++i)
    {
        const double lower = lower_arm_limits_(i);
        const double upper = upper_arm_limits_(i);

        if (std::isfinite(lower) && std::isfinite(upper) && lower < upper)
        {
            if (0.0 < lower || 0.0 > upper)
            {
                arm_q(i) = 0.5 * (lower + upper);
            }
        }
    }

    state.tail(arm_model_.nq) = arm_q;
    return clampToStateLimits(state);
}

Eigen::Affine3d WholeBodyPolytopeOptimizer::computeEndEffectorPose(
    const Eigen::VectorXd& state) const
{
    if (!initialized_)
    {
        throw std::runtime_error("Whole-body optimizer not initialized.");
    }

    if (state.size() != state_dim_)
    {
        throw std::runtime_error("computeEndEffectorPose: state dimension mismatch.");
    }

    return computeForwardKinematics(clampToStateLimits(state));
}

double WholeBodyPolytopeOptimizer::resolveDesiredForceRadius(
    const WholeBodyTrajectoryOptimizationInput& input,
    std::size_t waypoint_index) const
{
    if (input.desired_force_radii.empty())
    {
        return 0.0;
    }

    if (waypoint_index >= input.desired_force_radii.size())
    {
        throw std::runtime_error("Desired force radius trajectory size mismatch.");
    }

    return std::max(0.0, input.desired_force_radii[waypoint_index]);
}

WholeBodyWaypointOptimizationMetrics WholeBodyPolytopeOptimizer::evaluateWaypointMetrics(
    const Eigen::VectorXd& state,
    const Eigen::Affine3d& target_pose,
    const Eigen::Vector3d& desired_force,
    double desired_force_radius,
    const Eigen::VectorXd& nominal_state,
    const Eigen::VectorXd* previous_state,
    double previous_dt_sec) const
{
    if (!initialized_)
    {
        throw std::runtime_error("Whole-body optimizer not initialized.");
    }

    const CostBreakdown breakdown =
        evaluateCost(
            state,
            desired_force,
            desired_force_radius,
            nominal_state,
            previous_state,
            previous_dt_sec);

    WholeBodyWaypointOptimizationMetrics metrics;
    metrics.objective_value = breakdown.objective_value;
    metrics.force_capacity = breakdown.force_capacity;
    metrics.required_force = breakdown.required_force;
    metrics.required_force_radius = breakdown.required_force_radius;
    metrics.force_ball_clearance = breakdown.force_ball_clearance;
    metrics.capability_cost = breakdown.capability_cost;
    metrics.manipulability_measure = breakdown.manipulability_measure;
    metrics.manipulability_cost = breakdown.manipulability_cost;
    metrics.joint_limit_cost = breakdown.joint_limit_cost;
    metrics.smoothness_cost = breakdown.smoothness_cost;
    metrics.velocity_cost = breakdown.velocity_cost;
    metrics.nominal_cost = breakdown.nominal_cost;
    metrics.base_smoothness_cost = breakdown.base_smoothness_cost;
    metrics.base_velocity_cost = breakdown.base_velocity_cost;
    metrics.base_nominal_cost = breakdown.base_nominal_cost;
    metrics.collision_cost = breakdown.collision_cost;
    metrics.min_collision_clearance = breakdown.min_collision_clearance;
    metrics.pose_error_norm = computePoseError(state, target_pose).norm();
    return metrics;
}

double WholeBodyPolytopeOptimizer::computeTrajectoryBaseSpectralEnergyCost(
    const std::vector<Eigen::VectorXd>& trajectory,
    const std::vector<double>& waypoint_times_sec) const
{
    if (!initialized_)
    {
        throw std::runtime_error("Whole-body optimizer not initialized.");
    }

    return computeBaseSpectralEnergyCost(trajectory, waypoint_times_sec);
}

std::vector<Eigen::Vector3d>
WholeBodyPolytopeOptimizer::computeCollisionSphereCentersWorld(
    const Eigen::VectorXd& state,
    const std::vector<CollisionSphereSpec>& spheres) const
{
    if (!initialized_)
    {
        throw std::runtime_error("Whole-body optimizer not initialized.");
    }
    if (state.size() != state_dim_)
    {
        throw std::runtime_error("computeCollisionSphereCentersWorld: state dimension mismatch.");
    }

    const std::vector<ResolvedCollisionSphere> resolved_spheres =
        resolveCollisionSpheres(spheres, "collision_visualization_spheres");
    const Eigen::VectorXd clamped_state = clampToStateLimits(state);

    std::vector<Eigen::Vector3d> centers;
    centers.reserve(resolved_spheres.size());
    for (const ResolvedCollisionSphere& sphere : resolved_spheres)
    {
        centers.push_back(computeCollisionSphereCenterWorld(clamped_state, sphere));
    }
    return centers;
}

Eigen::MatrixXd WholeBodyPolytopeOptimizer::computeArmTranslationalForcePolytope(
    const Eigen::VectorXd& state,
    bool gravity_adjusted) const
{
    if (!initialized_)
    {
        throw std::runtime_error("Whole-body optimizer not initialized.");
    }

    if (state.size() != state_dim_)
    {
        throw std::runtime_error("computeArmTranslationalForcePolytope: state dimension mismatch.");
    }

    const Eigen::VectorXd clamped_state = clampToStateLimits(state);
    const Eigen::MatrixXd translational_jacobian =
        computeArmJacobianWorld(clamped_state).topRows(3);

    if (gravity_adjusted)
    {
        const Eigen::VectorXd adjusted_limits =
            buildGravityAdjustedTorqueLimits(clamped_state);
        return polytope_tool_.compute_polytope_vertices(
            translational_jacobian,
            &adjusted_limits);
    }

    return polytope_tool_.compute_polytope_vertices(translational_jacobian);
}

Eigen::Vector3d WholeBodyPolytopeOptimizer::baseState(const Eigen::VectorXd& state) const
{
    if (state.size() != state_dim_)
    {
        throw std::runtime_error("baseState: state dimension mismatch.");
    }

    return state.head<3>();
}

Eigen::VectorXd WholeBodyPolytopeOptimizer::armState(const Eigen::VectorXd& state) const
{
    if (state.size() != state_dim_)
    {
        throw std::runtime_error("armState: state dimension mismatch.");
    }

    return state.tail(arm_model_.nq);
}

Eigen::Vector3d WholeBodyPolytopeOptimizer::baseDifference(
    const Eigen::Vector3d& lhs,
    const Eigen::Vector3d& rhs) const
{
    Eigen::Vector3d diff = lhs - rhs;
    diff(2) = wrapAngle(diff(2));
    return diff;
}

double WholeBodyPolytopeOptimizer::wrapAngle(double angle) const
{
    while (angle > M_PI)
    {
        angle -= 2.0 * M_PI;
    }
    while (angle < -M_PI)
    {
        angle += 2.0 * M_PI;
    }
    return angle;
}

Eigen::VectorXd WholeBodyPolytopeOptimizer::clampToStateLimits(const Eigen::VectorXd& state) const
{
    Eigen::VectorXd clamped = state;
    clamped(2) = wrapAngle(clamped(2));

    for (int i = 0; i < kBasePlanarDof; ++i)
    {
        const double lower = config_.base_lower_limits(i);
        const double upper = config_.base_upper_limits(i);

        if (std::isfinite(lower) && std::isfinite(upper) && lower < upper)
        {
            clamped(i) = std::min(std::max(clamped(i), lower), upper);
        }
    }

    for (int i = 0; i < arm_model_.nq; ++i)
    {
        const double lower = lower_arm_limits_(i);
        const double upper = upper_arm_limits_(i);

        if (!std::isfinite(lower) || !std::isfinite(upper) || lower >= upper)
        {
            continue;
        }

        clamped(kBasePlanarDof + i) =
            std::min(
                std::max(clamped(kBasePlanarDof + i), lower + 1e-6),
                upper - 1e-6);
    }

    return clamped;
}

Eigen::VectorXd WholeBodyPolytopeOptimizer::integrateState(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& delta) const
{
    if (state.size() != state_dim_ || delta.size() != state_dim_)
    {
        throw std::runtime_error("integrateState: dimension mismatch.");
    }

    Eigen::VectorXd integrated = state;
    integrated.head<2>() += delta.head<2>();
    integrated(2) = wrapAngle(integrated(2) + delta(2));
    integrated.tail(arm_model_.nq) = pinocchio::integrate(
        arm_model_,
        state.tail(arm_model_.nq),
        delta.tail(arm_model_.nv));

    return clampToStateLimits(integrated);
}

Eigen::Affine3d WholeBodyPolytopeOptimizer::computeWorldToBase(
    const Eigen::VectorXd& state) const
{
    const Eigen::Vector3d base_state = baseState(state);
    Eigen::Affine3d w_T_base = Eigen::Affine3d::Identity();
    w_T_base.linear() = Eigen::AngleAxisd(base_state(2), Eigen::Vector3d::UnitZ()).toRotationMatrix();
    w_T_base.translation() << base_state(0), base_state(1), 0.0;
    return w_T_base;
}

Eigen::Affine3d WholeBodyPolytopeOptimizer::computeBaseToEe(
    const Eigen::VectorXd& state) const
{
    if (!initialized_ || !data_)
    {
        throw std::runtime_error("Whole-body optimizer not initialized.");
    }

    const Eigen::VectorXd arm_q = armState(state);
    pinocchio::forwardKinematics(arm_model_, *data_, arm_q);
    pinocchio::updateFramePlacements(arm_model_, *data_);

    const pinocchio::SE3& root_T_base = data_->oMf[base_frame_id_];
    const pinocchio::SE3& root_T_ee = data_->oMf[ee_frame_id_];
    const pinocchio::SE3 base_T_ee = root_T_base.inverse() * root_T_ee;

    return Eigen::Affine3d(base_T_ee.toHomogeneousMatrix());
}

Eigen::Affine3d WholeBodyPolytopeOptimizer::computeForwardKinematics(
    const Eigen::VectorXd& state) const
{
    return computeWorldToBase(state) * computeBaseToEe(state);
}

std::string WholeBodyPolytopeOptimizer::resolveCollisionFrameName(
    const std::string& frame) const
{
    if (frame == "base" || frame == "BASE")
    {
        return config_.base_frame;
    }
    if (frame == "ee" || frame == "EE" || frame == "end_effector")
    {
        return config_.ee_frame;
    }
    return frame;
}

std::vector<WholeBodyPolytopeOptimizer::ResolvedCollisionSphere>
WholeBodyPolytopeOptimizer::resolveCollisionSpheres(
    const std::vector<CollisionSphereSpec>& specs,
    const std::string& label) const
{
    std::vector<ResolvedCollisionSphere> resolved_spheres;
    resolved_spheres.reserve(specs.size());

    for (std::size_t i = 0; i < specs.size(); ++i)
    {
        const CollisionSphereSpec& spec = specs[i];
        const std::string frame_name = resolveCollisionFrameName(spec.frame);
        if (frame_name.empty())
        {
            throw std::runtime_error(label + "[" + std::to_string(i) + "] has an empty frame.");
        }
        if (!spec.center.allFinite())
        {
            throw std::runtime_error(label + "[" + std::to_string(i) + "] center is non-finite.");
        }
        if (!std::isfinite(spec.radius) || spec.radius <= 0.0)
        {
            throw std::runtime_error(label + "[" + std::to_string(i) + "] radius must be positive.");
        }

        const pinocchio::FrameIndex frame_id = arm_model_.getFrameId(frame_name);
        if (frame_id == static_cast<pinocchio::FrameIndex>(arm_model_.nframes))
        {
            throw std::runtime_error(
                label + "[" + std::to_string(i) + "] frame not found: " + frame_name);
        }

        ResolvedCollisionSphere sphere;
        sphere.frame = frame_name;
        sphere.frame_id = frame_id;
        sphere.center = spec.center;
        sphere.radius = spec.radius;
        resolved_spheres.push_back(sphere);
    }

    return resolved_spheres;
}

Eigen::Vector3d WholeBodyPolytopeOptimizer::computeCollisionSphereCenterWorld(
    const Eigen::VectorXd& state,
    const ResolvedCollisionSphere& sphere) const
{
    if (!initialized_ || !data_)
    {
        throw std::runtime_error("Whole-body optimizer not initialized.");
    }

    const Eigen::VectorXd arm_q = armState(state);
    pinocchio::forwardKinematics(arm_model_, *data_, arm_q);
    pinocchio::updateFramePlacements(arm_model_, *data_);

    const pinocchio::SE3& root_T_base = data_->oMf[base_frame_id_];
    const pinocchio::SE3& root_T_sphere_frame = data_->oMf[sphere.frame_id];
    const pinocchio::SE3 base_T_sphere_frame =
        root_T_base.inverse() * root_T_sphere_frame;
    const Eigen::Affine3d w_T_sphere_frame =
        computeWorldToBase(state) *
        Eigen::Affine3d(base_T_sphere_frame.toHomogeneousMatrix());

    return w_T_sphere_frame.translation() +
           w_T_sphere_frame.linear() * sphere.center;
}

double WholeBodyPolytopeOptimizer::evaluateCollisionCost(
    const Eigen::VectorXd& state,
    double* min_clearance) const
{
    if (min_clearance != nullptr)
    {
        *min_clearance = 0.0;
    }

    if (basket_collision_spheres_.empty() || body_collision_spheres_.empty())
    {
        return 0.0;
    }

    double min_clearance_value = std::numeric_limits<double>::infinity();
    double collision_cost = 0.0;
    for (const ResolvedCollisionSphere& basket_sphere : basket_collision_spheres_)
    {
        const Eigen::Vector3d basket_center =
            computeCollisionSphereCenterWorld(state, basket_sphere);
        for (const ResolvedCollisionSphere& body_sphere : body_collision_spheres_)
        {
            const Eigen::Vector3d body_center =
                computeCollisionSphereCenterWorld(state, body_sphere);
            const double clearance =
                (basket_center - body_center).norm() -
                basket_sphere.radius -
                body_sphere.radius;
            min_clearance_value = std::min(min_clearance_value, clearance);

            const double violation = config_.collision_safe_distance - clearance;
            if (violation > 0.0)
            {
                collision_cost += violation * violation;
            }
        }
    }

    if (min_clearance != nullptr && std::isfinite(min_clearance_value))
    {
        *min_clearance = min_clearance_value;
    }
    return collision_cost;
}

Eigen::Matrix<double, 6, 3> WholeBodyPolytopeOptimizer::computeBaseJacobianWorld(
    const Eigen::Affine3d& w_T_base,
    const Eigen::Affine3d& w_T_ee) const
{
    Eigen::Matrix<double, 6, 3> jacobian = Eigen::Matrix<double, 6, 3>::Zero();
    const Eigen::Vector3d offset = w_T_ee.translation() - w_T_base.translation();

    jacobian(0, 0) = 1.0;
    jacobian(1, 1) = 1.0;
    jacobian(0, 2) = -offset.y();
    jacobian(1, 2) = offset.x();
    jacobian(5, 2) = 1.0;

    return jacobian;
}

Eigen::MatrixXd WholeBodyPolytopeOptimizer::computeArmJacobianWorld(
    const Eigen::VectorXd& state) const
{
    if (!initialized_ || !data_)
    {
        throw std::runtime_error("Whole-body optimizer not initialized.");
    }

    const Eigen::VectorXd arm_q = armState(state);
    Eigen::MatrixXd arm_jacobian(6, arm_model_.nv);
    arm_jacobian.setZero();

    pinocchio::computeFrameJacobian(
        arm_model_,
        *data_,
        arm_q,
        ee_frame_id_,
        pinocchio::LOCAL_WORLD_ALIGNED,
        arm_jacobian);

    const Eigen::Matrix3d rotation_world_from_base = computeWorldToBase(state).linear();
    arm_jacobian.topRows(3) =
        rotation_world_from_base * arm_jacobian.topRows(3);
    arm_jacobian.bottomRows(3) =
        rotation_world_from_base * arm_jacobian.bottomRows(3);

    return arm_jacobian;
}

double WholeBodyPolytopeOptimizer::computeArmManipulability(
    const Eigen::VectorXd& state) const
{
    const Eigen::Affine3d w_T_base = computeWorldToBase(state);
    const Eigen::Affine3d w_T_ee = computeForwardKinematics(state);
    const Eigen::Matrix<double, 6, 3> base_jacobian =
        computeBaseJacobianWorld(w_T_base, w_T_ee);
    const Eigen::MatrixXd arm_jacobian = computeArmJacobianWorld(state);
    Eigen::MatrixXd whole_body_jacobian(6, state_dim_);
    whole_body_jacobian << base_jacobian, arm_jacobian;

    Eigen::MatrixXd optimization_jacobian(3, 1 + arm_model_.nv);
    optimization_jacobian.setZero();

    // Match the controller-side manipulability definition: keep base yaw and
    // arm joints, but omit base x/y because they make the translational
    // manipulability nearly constant and hide posture-related variation.
    optimization_jacobian.col(0) = whole_body_jacobian.topRows(3).col(2);
    optimization_jacobian.rightCols(arm_model_.nv) =
        whole_body_jacobian.topRows(3).rightCols(arm_model_.nv);

    // Legacy arm-only translational manipulability kept here for reference:
    // const Eigen::MatrixXd optimization_jacobian =
    //     computeArmJacobianWorld(state).topRows(3);

    const Eigen::Matrix3d gram =
        optimization_jacobian * optimization_jacobian.transpose() +
        config_.manipulability_regularization * Eigen::Matrix3d::Identity();

    return std::sqrt(
        std::max(config_.manipulability_epsilon, gram.determinant()));
}

Eigen::MatrixXd WholeBodyPolytopeOptimizer::computeTaskJacobian(
    const Eigen::VectorXd& state) const
{
    const Eigen::Affine3d w_T_base = computeWorldToBase(state);
    const Eigen::Affine3d w_T_ee = computeForwardKinematics(state);

    const Eigen::Matrix<double, 6, 3> base_jacobian =
        computeBaseJacobianWorld(w_T_base, w_T_ee);
    const Eigen::MatrixXd arm_jacobian = computeArmJacobianWorld(state);

    Eigen::MatrixXd whole_body_jacobian(6, state_dim_);
    whole_body_jacobian << base_jacobian, arm_jacobian;

    if (config_.constrain_orientation)
    {
        return whole_body_jacobian;
    }

    return whole_body_jacobian.topRows(3);
}

Eigen::Matrix<double, 6, 1> WholeBodyPolytopeOptimizer::computePoseError(
    const Eigen::VectorXd& state,
    const Eigen::Affine3d& target_pose) const
{
    const Eigen::Affine3d current_pose = computeForwardKinematics(state);

    const pinocchio::SE3 current_se3(current_pose.linear(), current_pose.translation());
    const pinocchio::SE3 target_se3(target_pose.linear(), target_pose.translation());
    const pinocchio::Motion error_motion = pinocchio::log6(current_se3.inverse() * target_se3);

    Eigen::Matrix<double, 6, 1> pose_error = error_motion.toVector();
    if (!config_.constrain_orientation)
    {
        pose_error.tail<3>().setZero();
    }

    return pose_error;
}

Eigen::VectorXd WholeBodyPolytopeOptimizer::buildGravityAdjustedTorqueLimits(
    const Eigen::VectorXd& state) const
{
    if (!initialized_ || !data_)
    {
        throw std::runtime_error("Whole-body optimizer not initialized.");
    }

    const Eigen::VectorXd arm_q = armState(state);
    const Eigen::VectorXd gravity =
        pinocchio::computeGeneralizedGravity(arm_model_, *data_, arm_q);

    Eigen::VectorXd adjusted_limits(2 * arm_model_.nv);
    adjusted_limits.head(arm_model_.nv) = arm_torque_limits_ - gravity;
    adjusted_limits.tail(arm_model_.nv) = -arm_torque_limits_ - gravity;
    return adjusted_limits;
}

Eigen::VectorXd WholeBodyPolytopeOptimizer::buildDynamicResidualTorqueLimits(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd* previous_state,
    const Eigen::VectorXd* previous_previous_state,
    double previous_dt_sec,
    double previous_previous_dt_sec) const
{
    if (!rfpComparisonShouldUseDynamicResidualForcePolytope(
            config_.comparison_algorithm,
            config_.use_dynamic_residual_force_polytope) ||
        previous_state == nullptr)
    {
        return buildGravityAdjustedTorqueLimits(state);
    }

    if (!initialized_ || !data_)
    {
        throw std::runtime_error("Whole-body optimizer not initialized.");
    }

    const double dt = std::max(kEpsilon, previous_dt_sec);
    const double prev_dt = std::max(kEpsilon, previous_previous_dt_sec);

    const Eigen::VectorXd arm_q = armState(state);
    const Eigen::VectorXd previous_arm_q = armState(*previous_state);
    const Eigen::VectorXd arm_dq = (arm_q - previous_arm_q) / dt;

    Eigen::VectorXd previous_arm_dq = Eigen::VectorXd::Zero(arm_model_.nv);
    if (previous_previous_state != nullptr)
    {
        const Eigen::VectorXd previous_previous_arm_q = armState(*previous_previous_state);
        previous_arm_dq = (previous_arm_q - previous_previous_arm_q) / prev_dt;
    }

    const Eigen::VectorXd arm_ddq = (arm_dq - previous_arm_dq) / dt;
    const Eigen::VectorXd nominal_torque =
        pinocchio::rnea(arm_model_, *data_, arm_q, arm_dq, arm_ddq);

    Eigen::VectorXd adjusted_limits(2 * arm_model_.nv);
    adjusted_limits.head(arm_model_.nv) = arm_torque_limits_ - nominal_torque;
    adjusted_limits.tail(arm_model_.nv) = -arm_torque_limits_ - nominal_torque;
    return adjusted_limits;
}

WholeBodyPolytopeOptimizer::CostBreakdown
WholeBodyPolytopeOptimizer::evaluateCost(
    const Eigen::VectorXd& state,
    const Eigen::Vector3d& desired_force,
    double desired_force_radius,
    const Eigen::VectorXd& nominal_state,
    const Eigen::VectorXd* previous_state,
    double previous_dt_sec,
    const Eigen::VectorXd* previous_previous_state,
    double previous_previous_dt_sec) const
{
    CostBreakdown breakdown;

    const Eigen::VectorXd clamped_state = clampToStateLimits(state);
    const Eigen::Vector3d arm_force = desired_force;
    const double clamped_force_radius = std::max(0.0, desired_force_radius);
    breakdown.required_force = arm_force.norm();
    breakdown.required_force_radius = clamped_force_radius;
    const RfpComparisonAlgorithmPolicy comparison_policy =
        rfpComparisonAlgorithmPolicy(config_.comparison_algorithm);

    if (comparison_policy.include_capability &&
        (breakdown.required_force > kEpsilon ||
         clamped_force_radius > kEpsilon ||
         comparison_policy.capability_objective != RfpCapabilityObjective::Clearance))
    {
        const Eigen::MatrixXd translational_arm_jacobian =
            computeArmJacobianWorld(clamped_state).topRows(3);
        const Eigen::VectorXd adjusted_limits =
            buildDynamicResidualTorqueLimits(
                clamped_state,
                previous_state,
                previous_previous_state,
                previous_dt_sec,
                previous_previous_dt_sec);

        if (comparison_policy.capability_objective ==
            RfpCapabilityObjective::CenteredBallRadius)
        {
            const double radius =
                computeCenteredForceBallRadius(
                    translational_arm_jacobian,
                    adjusted_limits,
                    config_.capability_alpha);
            breakdown.force_capacity = radius;
            breakdown.force_ball_clearance = radius;
            breakdown.capability_cost = positiveQuantityMaximizationCost(radius);
        }
        else if (comparison_policy.capability_objective ==
                 RfpCapabilityObjective::ConeIntersectionVolume)
        {
            RfpConeSamplingConfig cone_config;
            cone_config.axis_world =
                breakdown.required_force > kEpsilon
                    ? arm_force.normalized()
                    : config_.comparison_cone_axis_world;
            cone_config.half_angle_rad = config_.comparison_cone_half_angle_rad;
            cone_config.ring_count = config_.comparison_cone_ring_count;
            cone_config.azimuth_count = config_.comparison_cone_azimuth_count;
            const double volume_proxy =
                computeConeIntersectionVolumeProxy(
                    translational_arm_jacobian,
                    adjusted_limits,
                    cone_config,
                    config_.capability_alpha);
            breakdown.force_capacity = std::cbrt(std::max(0.0, volume_proxy));
            breakdown.force_ball_clearance = volume_proxy;
            breakdown.capability_cost =
                positiveQuantityMaximizationCost(volume_proxy);
        }
        else
        {
            breakdown.force_capacity = polytope_tool_.compute_force_capacity(
                translational_arm_jacobian,
                arm_force,
                adjusted_limits);
            breakdown.force_ball_clearance =
                polytope_tool_.compute_force_ball_signed_clearance(
                    translational_arm_jacobian,
                    arm_force,
                    clamped_force_radius,
                    adjusted_limits,
                    config_.capability_alpha);
            breakdown.capability_cost =
                computeBoundedClearanceCapabilityCost(
                    breakdown.force_ball_clearance,
                    breakdown.required_force + clamped_force_radius,
                    config_.capability_exponent_limit_enabled);
        }
    }

    breakdown.manipulability_measure = computeArmManipulability(clamped_state);
    const double safe_manipulability =
        std::max(
            config_.manipulability_epsilon,
            breakdown.manipulability_measure);
    breakdown.manipulability_cost = 1.0 / safe_manipulability;

    // Legacy manipulability costs kept here for reference:
    // breakdown.manipulability_cost =
    //     1.0 / (safe_manipulability * safe_manipulability);
    // breakdown.manipulability_cost =
    //     -std::log(std::max(
    //         config_.manipulability_epsilon,
    //         breakdown.manipulability_measure));

    const Eigen::Vector3d current_base = baseState(clamped_state);
    const Eigen::VectorXd current_arm = armState(clamped_state);
    const Eigen::Vector3d nominal_base = baseState(nominal_state);
    const Eigen::VectorXd nominal_arm = armState(nominal_state);
    const Eigen::Vector3d base_nominal_diff = baseDifference(current_base, nominal_base);

    for (int i = 0; i < kBasePlanarDof; ++i)
    {
        const double weight = config_.joint_limit_dof_weights(i);
        if (weight <= 0.0)
        {
            continue;
        }
        breakdown.joint_limit_cost += weight * base_nominal_diff(i) * base_nominal_diff(i);
    }

    const Eigen::VectorXd arm_q = current_arm;
    for (int i = 0; i < arm_q.size(); ++i)
    {
        const double joint_weight =
            config_.joint_limit_dof_weights(kBasePlanarDof + i);
        if (joint_weight <= 0.0)
        {
            continue;
        }

        const double lower = lower_arm_limits_(i);
        const double upper = upper_arm_limits_(i);
        const double margin = arm_joint_limit_margins_(i);

        if (!std::isfinite(lower) || !std::isfinite(upper) || lower >= upper || margin <= 0.0)
        {
            continue;
        }

        const double center = 0.5 * (lower + upper);
        const double half_range = 0.5 * (upper - lower);
        const double safe_half_range =
            std::max(1e-6, half_range - margin);
        const double normalized_offset =
            (arm_q(i) - center) / safe_half_range;

        // Use a midpoint-centered cost so the joint-limit term is smallest
        // near the middle of each joint's allowable range.
        breakdown.joint_limit_cost +=
            joint_weight * normalized_offset * normalized_offset;

        const double distance_from_center = std::abs(arm_q(i) - center);
        if (distance_from_center > safe_half_range)
        {
            const double normalized_excess =
                (distance_from_center - safe_half_range) /
                std::max(1e-6, margin);
            breakdown.joint_limit_cost +=
                joint_weight *
                normalized_excess * normalized_excess *
                normalized_excess * normalized_excess;
        }

        // Legacy guard-band barrier kept here for reference:
        // const double lower_guard = lower + margin;
        // const double upper_guard = upper - margin;
        // const double eps = 1e-6;
        // if (arm_q(i) < lower_guard)
        // {
        //     const double distance = lower_guard - arm_q(i);
        //     breakdown.joint_limit_cost += 1.0 / (distance * distance + eps);
        // }
        // if (arm_q(i) > upper_guard)
        // {
        //     const double distance = arm_q(i) - upper_guard;
        //     breakdown.joint_limit_cost += 1.0 / (distance * distance + eps);
        // }
    }

    breakdown.base_nominal_cost =
        0.5 * base_nominal_diff.transpose() *
        config_.base_nominal_metric.asDiagonal() *
        base_nominal_diff;
    breakdown.nominal_cost = 0.5 * (current_arm - nominal_arm).squaredNorm();

    if (previous_state != nullptr && previous_state->size() == state_dim_)
    {
        const Eigen::Vector3d previous_base = baseState(*previous_state);
        const Eigen::VectorXd previous_arm = armState(*previous_state);
        const Eigen::Vector3d base_smoothness_diff =
            baseDifference(current_base, previous_base);
        const double safe_dt = std::max(kEpsilon, previous_dt_sec);

        breakdown.base_smoothness_cost =
            0.5 * base_smoothness_diff.transpose() *
            config_.base_smoothness_metric.asDiagonal() *
            base_smoothness_diff;
        breakdown.smoothness_cost = 0.5 * (current_arm - previous_arm).squaredNorm();
        breakdown.base_velocity_cost =
            breakdown.base_smoothness_cost / (safe_dt * safe_dt);
        breakdown.velocity_cost =
            breakdown.smoothness_cost / (safe_dt * safe_dt);
    }

    breakdown.collision_cost =
        evaluateCollisionCost(clamped_state, &breakdown.min_collision_clearance);

    breakdown.objective_value = 0.0;
    if (comparison_policy.include_capability)
    {
        breakdown.objective_value +=
            config_.capability_weight * breakdown.capability_cost;
    }
    if (comparison_policy.include_manipulability)
    {
        breakdown.objective_value +=
            config_.manipulability_weight * breakdown.manipulability_cost;
    }
    if (comparison_policy.include_joint_limit)
    {
        breakdown.objective_value +=
            config_.joint_limit_weight * breakdown.joint_limit_cost;
    }
    if (comparison_policy.include_trajectory_regularization)
    {
        breakdown.objective_value +=
            config_.smoothness_weight * breakdown.smoothness_cost +
            config_.velocity_weight * breakdown.velocity_cost +
            config_.nominal_weight * breakdown.nominal_cost +
            config_.base_smoothness_weight * breakdown.base_smoothness_cost +
            config_.base_velocity_weight * breakdown.base_velocity_cost +
            config_.base_nominal_weight * breakdown.base_nominal_cost +
            config_.collision_weight * breakdown.collision_cost;
    }

    return breakdown;
}

Eigen::VectorXd WholeBodyPolytopeOptimizer::computeObjectiveGradient(
    const Eigen::VectorXd& state,
    const Eigen::Vector3d& desired_force,
    double desired_force_radius,
    const Eigen::VectorXd& nominal_state,
    const Eigen::VectorXd* previous_state) const
{
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(state_dim_);

    for (int i = 0; i < state_dim_; ++i)
    {
        Eigen::VectorXd tangent_plus = Eigen::VectorXd::Zero(state_dim_);
        Eigen::VectorXd tangent_minus = Eigen::VectorXd::Zero(state_dim_);
        tangent_plus(i) = config_.finite_difference_step;
        tangent_minus(i) = -config_.finite_difference_step;

        const Eigen::VectorXd state_plus =
            integrateState(state, tangent_plus);
        const Eigen::VectorXd state_minus =
            integrateState(state, tangent_minus);

        const double value_plus =
            evaluateCost(
                state_plus,
                desired_force,
                desired_force_radius,
                nominal_state,
                previous_state).objective_value;
        const double value_minus =
            evaluateCost(
                state_minus,
                desired_force,
                desired_force_radius,
                nominal_state,
                previous_state).objective_value;

        gradient(i) =
            (value_plus - value_minus) / (2.0 * config_.finite_difference_step);
    }

    return gradient;
}

double WholeBodyPolytopeOptimizer::computeBaseSpectralEnergyCost(
    const std::vector<Eigen::VectorXd>& trajectory,
    const std::vector<double>& waypoint_times_sec) const
{
    if (trajectory.size() < 2 || config_.base_spectral_energy_weight <= 0.0)
    {
        return 0.0;
    }

    const int sample_count = static_cast<int>(trajectory.size());
    const double average_dt_sec =
        estimateAverageWaypointDtSec(waypoint_times_sec, sample_count);
    const double sample_rate_hz = 1.0 / std::max(kEpsilon, average_dt_sec);
    const double nyquist_hz = 0.5 * sample_rate_hz;
    const double cutoff_ratio = clampUnitInterval(config_.base_spectral_cutoff_ratio);
    const double cutoff_hz = cutoff_ratio * nyquist_hz;
    const double spectral_power = std::max(0.0, config_.base_spectral_power);

    Eigen::VectorXd base_x = Eigen::VectorXd::Zero(sample_count);
    Eigen::VectorXd base_y = Eigen::VectorXd::Zero(sample_count);
    Eigen::VectorXd base_yaw = Eigen::VectorXd::Zero(sample_count);

    double unwrapped_yaw = baseState(clampToStateLimits(trajectory.front()))(2);
    for (int k = 0; k < sample_count; ++k)
    {
        const Eigen::Vector3d current_base =
            baseState(clampToStateLimits(trajectory[static_cast<std::size_t>(k)]));

        base_x(k) = current_base(0);
        base_y(k) = current_base(1);

        if (k == 0)
        {
            base_yaw(k) = unwrapped_yaw;
        }
        else
        {
            const Eigen::Vector3d previous_base =
                baseState(clampToStateLimits(trajectory[static_cast<std::size_t>(k - 1)]));
            unwrapped_yaw += wrapAngle(current_base(2) - previous_base(2));
            base_yaw(k) = unwrapped_yaw;
        }
    }

    const std::array<Eigen::VectorXd, 3> base_signals{{base_x, base_y, base_yaw}};
    double spectral_cost = 0.0;

    for (int axis = 0; axis < kBasePlanarDof; ++axis)
    {
        const double axis_metric = config_.base_spectral_energy_metric(axis);
        if (axis_metric <= 0.0)
        {
            continue;
        }

        const Eigen::VectorXd coefficients = computeOrthonormalDct(base_signals[axis]);
        const int highest_bin = std::max(1, sample_count - 1);

        for (int k = 1; k < sample_count; ++k)
        {
            const double frequency_hz =
                static_cast<double>(k) *
                nyquist_hz /
                static_cast<double>(highest_bin);
            if (frequency_hz <= cutoff_hz)
            {
                continue;
            }

            const double normalized_tail_frequency =
                (frequency_hz - cutoff_hz) /
                std::max(kEpsilon, nyquist_hz - cutoff_hz);
            const double spectral_weight =
                std::pow(normalized_tail_frequency, spectral_power);
            spectral_cost +=
                axis_metric * spectral_weight * coefficients(k) * coefficients(k);
        }
    }

    return 0.5 * spectral_cost;
}

void WholeBodyPolytopeOptimizer::writeDynamicResidualDiagnostics(
    const WholeBodyTrajectoryOptimizationInput& input,
    const std::vector<Eigen::VectorXd>& trajectory,
    const std::string& label) const
{
    if (!config_.dynamic_residual_diagnostics_enabled ||
        trajectory.empty() ||
        dynamic_residual_diagnostics_record_count_ >=
            std::max(0, config_.dynamic_residual_diagnostics_max_records))
    {
        return;
    }

    const RfpComparisonAlgorithmPolicy policy =
        rfpComparisonAlgorithmPolicy(config_.comparison_algorithm);
    if (!policy.include_capability ||
        !rfpComparisonShouldUseDynamicResidualForcePolytope(
            config_.comparison_algorithm,
            config_.use_dynamic_residual_force_polytope))
    {
        return;
    }

    if (!ensureDirectoryExists(config_.dynamic_residual_diagnostics_directory))
    {
        std::cerr
            << "[WholeBodyPolytopeOptimizer] failed to create dynamic residual diagnostics directory: "
            << config_.dynamic_residual_diagnostics_directory << std::endl;
        return;
    }

    const int record_index = dynamic_residual_diagnostics_record_count_++;
    std::ostringstream path;
    path << config_.dynamic_residual_diagnostics_directory
         << "/dynamic_residual_"
         << timestampForFilename()
         << "_"
         << label
         << "_"
         << record_index
         << ".csv";

    std::ofstream file(path.str());
    if (!file.is_open())
    {
        std::cerr
            << "[WholeBodyPolytopeOptimizer] failed to open dynamic residual diagnostics file: "
            << path.str() << std::endl;
        return;
    }

    file << std::setprecision(17);
    file
        << "label,waypoint,previous_dt,previous_previous_dt,"
        << "state_all_finite,jacobian_all_finite,nominal_torque_all_finite,"
        << "adjusted_limits_all_finite,metrics_all_finite,"
        << "desired_force_norm,desired_force_radius,force_capacity,force_ball_clearance,"
        << "capability_cost,required_force,required_force_radius,"
        << "state,previous_state,previous_previous_state,arm_q,arm_dq,arm_ddq,"
        << "nominal_torque,adjusted_upper_limits,adjusted_lower_limits,"
        << "translational_arm_jacobian\n";

    for (std::size_t k = 0; k < trajectory.size(); ++k)
    {
        if (k >= input.desired_forces.size() ||
            k >= input.nominal_state_trajectory.size())
        {
            break;
        }

        const Eigen::VectorXd state = clampToStateLimits(trajectory[k]);
        const Eigen::VectorXd* previous_state =
            k == 0 ? nullptr : &trajectory[k - 1];
        const Eigen::VectorXd* previous_previous_state =
            k < 2 ? nullptr : &trajectory[k - 2];
        const double previous_dt_sec =
            waypointPreviousDtSec(input.waypoint_times_sec, k);
        const double previous_previous_dt_sec =
            k < 2
                ? previous_dt_sec
                : waypointPreviousDtSec(input.waypoint_times_sec, k - 1);
        const double safe_dt = std::max(kEpsilon, previous_dt_sec);
        const double safe_previous_dt =
            std::max(kEpsilon, previous_previous_dt_sec);

        const Eigen::VectorXd arm_q = armState(state);
        const Eigen::VectorXd previous_arm_q =
            previous_state == nullptr
                ? arm_q
                : armState(clampToStateLimits(*previous_state));
        Eigen::VectorXd arm_dq = Eigen::VectorXd::Zero(arm_model_.nv);
        if (previous_state != nullptr)
        {
            arm_dq = (arm_q - previous_arm_q) / safe_dt;
        }

        Eigen::VectorXd previous_arm_dq =
            Eigen::VectorXd::Zero(arm_model_.nv);
        if (previous_state != nullptr && previous_previous_state != nullptr)
        {
            const Eigen::VectorXd previous_previous_arm_q =
                armState(clampToStateLimits(*previous_previous_state));
            previous_arm_dq =
                (previous_arm_q - previous_previous_arm_q) / safe_previous_dt;
        }
        Eigen::VectorXd arm_ddq = Eigen::VectorXd::Zero(arm_model_.nv);
        if (previous_state != nullptr)
        {
            arm_ddq = (arm_dq - previous_arm_dq) / safe_dt;
        }

        const Eigen::VectorXd nominal_torque =
            pinocchio::rnea(arm_model_, *data_, arm_q, arm_dq, arm_ddq);
        const Eigen::VectorXd adjusted_limits =
            buildDynamicResidualTorqueLimits(
                state,
                previous_state,
                previous_previous_state,
                previous_dt_sec,
                previous_previous_dt_sec);
        const Eigen::MatrixXd translational_arm_jacobian =
            computeArmJacobianWorld(state).topRows(3);
        const CostBreakdown breakdown =
            evaluateCost(
                state,
                input.desired_forces[k],
                resolveDesiredForceRadius(input, k),
                input.nominal_state_trajectory[k],
                previous_state,
                previous_dt_sec,
                previous_previous_state,
                previous_previous_dt_sec);

        const bool metrics_all_finite =
            std::isfinite(breakdown.force_capacity) &&
            std::isfinite(breakdown.force_ball_clearance) &&
            std::isfinite(breakdown.capability_cost) &&
            std::isfinite(breakdown.required_force) &&
            std::isfinite(breakdown.required_force_radius);

        file
            << label << ","
            << k << ","
            << previous_dt_sec << ","
            << previous_previous_dt_sec << ","
            << (state.allFinite() ? 1 : 0) << ","
            << (translational_arm_jacobian.allFinite() ? 1 : 0) << ","
            << (nominal_torque.allFinite() ? 1 : 0) << ","
            << (adjusted_limits.allFinite() ? 1 : 0) << ","
            << (metrics_all_finite ? 1 : 0) << ","
            << input.desired_forces[k].norm() << ","
            << resolveDesiredForceRadius(input, k) << ","
            << breakdown.force_capacity << ","
            << breakdown.force_ball_clearance << ","
            << breakdown.capability_cost << ","
            << breakdown.required_force << ","
            << breakdown.required_force_radius << ",";

        writeVectorCsv(file, state);
        file << ",";
        if (previous_state == nullptr)
        {
            writeVectorCsv(file, Eigen::VectorXd::Zero(state_dim_));
        }
        else
        {
            writeVectorCsv(file, clampToStateLimits(*previous_state));
        }
        file << ",";
        if (previous_previous_state == nullptr)
        {
            writeVectorCsv(file, Eigen::VectorXd::Zero(state_dim_));
        }
        else
        {
            writeVectorCsv(file, clampToStateLimits(*previous_previous_state));
        }
        file << ",";
        writeVectorCsv(file, arm_q);
        file << ",";
        writeVectorCsv(file, arm_dq);
        file << ",";
        writeVectorCsv(file, arm_ddq);
        file << ",";
        writeVectorCsv(file, nominal_torque);
        file << ",";
        writeVectorCsv(file, adjusted_limits.head(arm_model_.nv));
        file << ",";
        writeVectorCsv(file, adjusted_limits.tail(arm_model_.nv));
        file << ",";
        writeMatrixCsv(file, translational_arm_jacobian);
        file << "\n";
    }

    std::cout
        << "[WholeBodyPolytopeOptimizer] dynamic residual diagnostics written to "
        << path.str() << std::endl;
}

double WholeBodyPolytopeOptimizer::evaluateTrajectoryObjective(
    const std::vector<Eigen::VectorXd>& trajectory,
    const WholeBodyTrajectoryOptimizationInput& input) const
{
    if (trajectory.size() != input.target_poses.size() ||
        trajectory.size() != input.desired_forces.size() ||
        trajectory.size() != input.nominal_state_trajectory.size())
    {
        throw std::runtime_error("evaluateTrajectoryObjective: trajectory size mismatch.");
    }
    if (!input.waypoint_times_sec.empty() &&
        trajectory.size() != input.waypoint_times_sec.size())
    {
        throw std::runtime_error("evaluateTrajectoryObjective: waypoint time size mismatch.");
    }
    if (!input.desired_force_radii.empty() &&
        trajectory.size() != input.desired_force_radii.size())
    {
        throw std::runtime_error("evaluateTrajectoryObjective: force radius size mismatch.");
    }

    double objective_value = 0.0;
    for (std::size_t k = 0; k < trajectory.size(); ++k)
    {
        const Eigen::VectorXd* previous_state =
            k == 0 ? nullptr : &trajectory[k - 1];
        const Eigen::VectorXd* previous_previous_state =
            k < 2 ? nullptr : &trajectory[k - 2];
        const double previous_dt_sec =
            waypointPreviousDtSec(input.waypoint_times_sec, k);
        const double previous_previous_dt_sec =
            k < 2 ? previous_dt_sec : waypointPreviousDtSec(input.waypoint_times_sec, k - 1);
        const CostBreakdown breakdown = evaluateCost(
            trajectory[k],
            input.desired_forces[k],
            resolveDesiredForceRadius(input, k),
            input.nominal_state_trajectory[k],
            previous_state,
            previous_dt_sec,
            previous_previous_state,
            previous_previous_dt_sec);
        objective_value += breakdown.objective_value;
    }

    const RfpComparisonAlgorithmPolicy policy =
        rfpComparisonAlgorithmPolicy(config_.comparison_algorithm);
    if (policy.include_trajectory_regularization &&
        !input.single_point_cost_terms_only)
    {
        objective_value +=
            config_.base_spectral_energy_weight *
            computeBaseSpectralEnergyCost(trajectory, input.waypoint_times_sec);
    }

    return objective_value;
}

std::vector<Eigen::VectorXd> WholeBodyPolytopeOptimizer::computeTrajectoryObjectiveGradient(
    const std::vector<Eigen::VectorXd>& trajectory,
    const WholeBodyTrajectoryOptimizationInput& input) const
{
    const std::size_t horizon = trajectory.size();
    std::vector<Eigen::VectorXd> trajectory_gradient(
        horizon,
        Eigen::VectorXd::Zero(state_dim_));

    for (std::size_t k = 0; k < horizon; ++k)
    {
        for (int i = 0; i < state_dim_; ++i)
        {
            Eigen::VectorXd tangent_plus = Eigen::VectorXd::Zero(state_dim_);
            Eigen::VectorXd tangent_minus = Eigen::VectorXd::Zero(state_dim_);
            tangent_plus(i) = config_.finite_difference_step;
            tangent_minus(i) = -config_.finite_difference_step;

            const Eigen::VectorXd state_plus =
                integrateState(trajectory[k], tangent_plus);
            const Eigen::VectorXd state_minus =
                integrateState(trajectory[k], tangent_minus);
            const Eigen::VectorXd* previous_state =
                k == 0 ? nullptr : &trajectory[k - 1];
            const double previous_dt_sec =
                waypointPreviousDtSec(input.waypoint_times_sec, k);

            double value_plus =
                evaluateCost(
                    state_plus,
                    input.desired_forces[k],
                    resolveDesiredForceRadius(input, k),
                    input.nominal_state_trajectory[k],
                    previous_state,
                    previous_dt_sec).objective_value;
            double value_minus =
                evaluateCost(
                    state_minus,
                    input.desired_forces[k],
                    resolveDesiredForceRadius(input, k),
                    input.nominal_state_trajectory[k],
                    previous_state,
                    previous_dt_sec).objective_value;

            if (k + 1 < horizon)
            {
                const double next_dt_sec =
                    waypointPreviousDtSec(input.waypoint_times_sec, k + 1);
                value_plus +=
                    evaluateCost(
                        trajectory[k + 1],
                        input.desired_forces[k + 1],
                        resolveDesiredForceRadius(input, k + 1),
                        input.nominal_state_trajectory[k + 1],
                        &state_plus,
                        next_dt_sec).objective_value;
                value_minus +=
                    evaluateCost(
                        trajectory[k + 1],
                        input.desired_forces[k + 1],
                        resolveDesiredForceRadius(input, k + 1),
                        input.nominal_state_trajectory[k + 1],
                        &state_minus,
                        next_dt_sec).objective_value;
            }

            trajectory_gradient[k](i) =
                (value_plus - value_minus) / (2.0 * config_.finite_difference_step);
        }
    }

    return trajectory_gradient;
}

WholeBodyWaypointOptimizationMetrics WholeBodyPolytopeOptimizer::optimizeWaypoint(
    const Eigen::VectorXd& seed_state,
    const Eigen::Affine3d& target_pose,
    const Eigen::Vector3d& desired_force,
    double desired_force_radius,
    const Eigen::VectorXd& nominal_state,
    const Eigen::VectorXd* previous_state,
    Eigen::VectorXd& optimized_state) const
{
    optimized_state = clampToStateLimits(seed_state);
    WholeBodyWaypointOptimizationMetrics metrics;

    for (int iteration = 0; iteration < config_.max_iterations_per_waypoint; ++iteration)
    {
        const Eigen::Matrix<double, 6, 1> pose_error =
            computePoseError(optimized_state, target_pose);
        const Eigen::MatrixXd task_jacobian = computeTaskJacobian(optimized_state);
        const Eigen::VectorXd task_error =
            config_.constrain_orientation ? pose_error : pose_error.head(3);

        const Eigen::MatrixXd task_jacobian_pinv =
            task_jacobian.completeOrthogonalDecomposition().pseudoInverse();
        const Eigen::MatrixXd nullspace_projector =
            Eigen::MatrixXd::Identity(state_dim_, state_dim_) -
            task_jacobian_pinv * task_jacobian;

        const Eigen::VectorXd objective_gradient =
            computeObjectiveGradient(
                optimized_state,
                desired_force,
                desired_force_radius,
                nominal_state,
                previous_state);

        Eigen::VectorXd delta_state =
            config_.pose_gain * (task_jacobian_pinv * task_error) -
            config_.nullspace_step_size * (nullspace_projector * objective_gradient);

        const double delta_norm = delta_state.norm();
        if (delta_norm > config_.max_state_update_norm && delta_norm > kEpsilon)
        {
            delta_state *= config_.max_state_update_norm / delta_norm;
        }

        optimized_state = integrateState(optimized_state, delta_state);
        metrics.iterations = iteration + 1;

        if (task_error.norm() < config_.pose_tolerance &&
            (nullspace_projector * objective_gradient).norm() < 1e-4)
        {
            break;
        }
    }

    const CostBreakdown breakdown =
        evaluateCost(
            optimized_state,
            desired_force,
            desired_force_radius,
            nominal_state,
            previous_state);
    const Eigen::Matrix<double, 6, 1> final_pose_error =
        computePoseError(optimized_state, target_pose);

    metrics.objective_value = breakdown.objective_value;
    metrics.force_capacity = breakdown.force_capacity;
    metrics.required_force = breakdown.required_force;
    metrics.required_force_radius = breakdown.required_force_radius;
    metrics.force_ball_clearance = breakdown.force_ball_clearance;
    metrics.capability_cost = breakdown.capability_cost;
    metrics.manipulability_measure = breakdown.manipulability_measure;
    metrics.manipulability_cost = breakdown.manipulability_cost;
    metrics.joint_limit_cost = breakdown.joint_limit_cost;
    metrics.smoothness_cost = breakdown.smoothness_cost;
    metrics.velocity_cost = breakdown.velocity_cost;
    metrics.nominal_cost = breakdown.nominal_cost;
    metrics.base_smoothness_cost = breakdown.base_smoothness_cost;
    metrics.base_velocity_cost = breakdown.base_velocity_cost;
    metrics.base_nominal_cost = breakdown.base_nominal_cost;
    metrics.collision_cost = breakdown.collision_cost;
    metrics.min_collision_clearance = breakdown.min_collision_clearance;
    metrics.pose_error_norm =
        (config_.constrain_orientation ? final_pose_error : final_pose_error.head(3)).norm();

    return metrics;
}

/**
 * @brief Optimize the whole body trajectory.
 *
 * @param input The input to the optimization.
 * @return WholeBodyTrajectoryOptimizationResult The result of the optimization.
 */
WholeBodyTrajectoryOptimizationResult WholeBodyPolytopeOptimizer::optimize(
    const WholeBodyTrajectoryOptimizationInput& input) const
{
    if (!initialized_)
    {
        throw std::runtime_error("Whole-body optimizer not initialized.");
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

    if (input.nominal_state_trajectory.size() != horizon)
    {
        throw std::runtime_error("Nominal whole-body trajectory size mismatch.");
    }
    if (!input.desired_force_radii.empty() && input.desired_force_radii.size() != horizon)
    {
        throw std::runtime_error("Desired force radius trajectory size mismatch.");
    }
    if (!input.waypoint_times_sec.empty() && input.waypoint_times_sec.size() != horizon)
    {
        throw std::runtime_error("Waypoint time trajectory size mismatch.");
    }

    std::vector<Eigen::VectorXd> initial_trajectory;
    initial_trajectory.reserve(horizon);

    for (std::size_t k = 0; k < horizon; ++k)
    {
        if (input.nominal_state_trajectory[k].size() != state_dim_)
        {
            std::ostringstream error_stream;
            error_stream
                << "Nominal whole-body state size mismatch at waypoint " << k
                << ": expected " << state_dim_
                << ", got " << input.nominal_state_trajectory[k].size() << ".";
            throw std::runtime_error(error_stream.str());
        }

        initial_trajectory.push_back(
            clampToStateLimits(input.nominal_state_trajectory[k]));
    }
    writeDynamicResidualDiagnostics(input, initial_trajectory, "initial");

    if (!casadi::has_nlpsol("ipopt"))
    {
        casadi::load_nlpsol("ipopt");
    }

    const int decision_dim = static_cast<int>(horizon) * state_dim_;
    const int task_dim = config_.constrain_orientation ? 6 : 3;
    const int constraint_dim =
        static_cast<int>(horizon) * task_dim +
        (input.pin_first_state ? state_dim_ : 0);
    const CasadiModel arm_model_casadi = arm_model_.template cast<CasadiScalar>();

    auto baseStateSymbolic =
        [&](const CasadiVectorXs& state) -> CasadiVector3
        {
            CasadiVector3 base_state;
            for (int i = 0; i < kBasePlanarDof; ++i)
            {
                base_state(i) = state(i);
            }
            return base_state;
        };

    auto armStateSymbolic =
        [&](const CasadiVectorXs& state) -> CasadiVectorXs
        {
            CasadiVectorXs arm_q(arm_model_.nq);
            for (int i = 0; i < arm_model_.nq; ++i)
            {
                arm_q(i) = state(kBasePlanarDof + i);
            }
            return arm_q;
        };

    auto wrapAngleSymbolic =
        [&](const CasadiScalar& angle) -> CasadiScalar
        {
            return atan2(sin(angle), cos(angle));
        };

    auto baseDifferenceSymbolic =
        [&](const CasadiVector3& lhs,
            const CasadiVector3& rhs) -> CasadiVector3
        {
            CasadiVector3 diff;
            diff(0) = lhs(0) - rhs(0);
            diff(1) = lhs(1) - rhs(1);
            diff(2) = wrapAngleSymbolic(lhs(2) - rhs(2));
            return diff;
        };

    auto computeWorldToBaseSymbolic =
        [&](const CasadiVectorXs& state) -> CasadiSE3
        {
            const CasadiScalar yaw = state(2);
            CasadiMatrix3 rotation;
            rotation(0, 0) = cos(yaw);
            rotation(0, 1) = -sin(yaw);
            rotation(0, 2) = sx(0.0);
            rotation(1, 0) = sin(yaw);
            rotation(1, 1) = cos(yaw);
            rotation(1, 2) = sx(0.0);
            rotation(2, 0) = sx(0.0);
            rotation(2, 1) = sx(0.0);
            rotation(2, 2) = sx(1.0);

            CasadiVector3 translation;
            translation(0) = state(0);
            translation(1) = state(1);
            translation(2) = sx(0.0);
            return CasadiSE3(rotation, translation);
        };

    auto computeBaseToEeSymbolic =
        [&](const CasadiVectorXs& state) -> CasadiSE3
        {
            CasadiData data(arm_model_casadi);
            const CasadiVectorXs arm_q = armStateSymbolic(state);
            pinocchio::forwardKinematics(arm_model_casadi, data, arm_q);
            pinocchio::updateFramePlacements(arm_model_casadi, data);

            const CasadiSE3& root_T_base = data.oMf[base_frame_id_];
            const CasadiSE3& root_T_ee = data.oMf[ee_frame_id_];
            return root_T_base.inverse() * root_T_ee;
        };

    auto computeForwardKinematicsSymbolic =
        [&](const CasadiVectorXs& state) -> CasadiSE3
        {
            return computeWorldToBaseSymbolic(state) * computeBaseToEeSymbolic(state);
        };

    auto computeCollisionSphereCenterWorldSymbolic =
        [&](const CasadiVectorXs& state,
            const ResolvedCollisionSphere& sphere) -> CasadiVector3
        {
            CasadiData data(arm_model_casadi);
            const CasadiVectorXs arm_q = armStateSymbolic(state);
            pinocchio::forwardKinematics(arm_model_casadi, data, arm_q);
            pinocchio::updateFramePlacements(arm_model_casadi, data);

            const CasadiSE3& root_T_base = data.oMf[base_frame_id_];
            const CasadiSE3& root_T_sphere_frame = data.oMf[sphere.frame_id];
            const CasadiSE3 base_T_sphere_frame =
                root_T_base.inverse() * root_T_sphere_frame;
            const CasadiSE3 w_T_sphere_frame =
                computeWorldToBaseSymbolic(state) * base_T_sphere_frame;
            return w_T_sphere_frame.translation() +
                   w_T_sphere_frame.rotation() *
                       constantCasadiVector3(sphere.center);
        };

    auto computeCollisionCostSymbolic =
        [&](const CasadiVectorXs& state) -> CasadiScalar
        {
            if (basket_collision_spheres_.empty() || body_collision_spheres_.empty())
            {
                return sx(0.0);
            }

            CasadiScalar collision_cost = sx(0.0);
            for (const ResolvedCollisionSphere& basket_sphere : basket_collision_spheres_)
            {
                const CasadiVector3 basket_center =
                    computeCollisionSphereCenterWorldSymbolic(state, basket_sphere);
                for (const ResolvedCollisionSphere& body_sphere : body_collision_spheres_)
                {
                    const CasadiVector3 body_center =
                        computeCollisionSphereCenterWorldSymbolic(state, body_sphere);
                    const CasadiVector3 diff = basket_center - body_center;
                    const CasadiScalar distance =
                        sqrt(squaredNorm3(diff) + sx(1e-12));
                    const CasadiScalar clearance =
                        distance -
                        sx(basket_sphere.radius + body_sphere.radius);
                    const CasadiScalar violation =
                        sx(config_.collision_safe_distance) - clearance;
                    collision_cost +=
                        if_else(
                            violation > sx(0.0),
                            violation * violation,
                            sx(0.0));
                }
            }
            return collision_cost;
        };

    auto computeBaseJacobianWorldSymbolic =
        [&](const CasadiSE3& w_T_base,
            const CasadiSE3& w_T_ee) -> CasadiMatrix6x3
        {
            CasadiMatrix6x3 jacobian;
            jacobian.setConstant(sx(0.0));

            const CasadiVector3 offset = w_T_ee.translation() - w_T_base.translation();
            jacobian(0, 0) = sx(1.0);
            jacobian(1, 1) = sx(1.0);
            jacobian(0, 2) = -offset(1);
            jacobian(1, 2) = offset(0);
            jacobian(5, 2) = sx(1.0);
            return jacobian;
        };

    auto computeArmJacobianWorldSymbolic =
        [&](const CasadiVectorXs& state) -> CasadiMatrixXs
        {
            CasadiData data(arm_model_casadi);
            const CasadiVectorXs arm_q = armStateSymbolic(state);

            CasadiMatrixXs arm_jacobian(6, arm_model_casadi.nv);
            arm_jacobian.setConstant(sx(0.0));

            pinocchio::computeFrameJacobian(
                arm_model_casadi,
                data,
                arm_q,
                ee_frame_id_,
                pinocchio::LOCAL_WORLD_ALIGNED,
                arm_jacobian);

            const CasadiMatrix3 rotation_world_from_base =
                computeWorldToBaseSymbolic(state).rotation();
            arm_jacobian.topRows(3) =
                rotation_world_from_base * arm_jacobian.topRows(3);
            arm_jacobian.bottomRows(3) =
                rotation_world_from_base * arm_jacobian.bottomRows(3);
            return arm_jacobian;
        };

    auto computeTaskJacobianSymbolic =
        [&](const CasadiVectorXs& state) -> CasadiMatrixXs
        {
            const CasadiSE3 w_T_base = computeWorldToBaseSymbolic(state);
            const CasadiSE3 w_T_ee = computeForwardKinematicsSymbolic(state);
            const CasadiMatrix6x3 base_jacobian =
                computeBaseJacobianWorldSymbolic(w_T_base, w_T_ee);
            const CasadiMatrixXs arm_jacobian =
                computeArmJacobianWorldSymbolic(state);

            CasadiMatrixXs whole_body_jacobian(6, state_dim_);
            whole_body_jacobian.leftCols(kBasePlanarDof) = base_jacobian;
            whole_body_jacobian.rightCols(arm_model_.nv) = arm_jacobian;

            if (config_.constrain_orientation)
            {
                return whole_body_jacobian;
            }

            return whole_body_jacobian.topRows(3);
        };

    auto computePoseErrorSymbolic =
        [&](const CasadiVectorXs& state,
            const Eigen::Affine3d& target_pose) -> CasadiVector6
        {
            const CasadiSE3 current_pose = computeForwardKinematicsSymbolic(state);
            const CasadiSE3 target_se3(
                constantCasadiMatrix3(target_pose.linear()),
                constantCasadiVector3(target_pose.translation()));

            const auto error_motion =
                pinocchio::log6(current_pose.inverse() * target_se3);
            CasadiVector6 pose_error = error_motion.toVector();

            if (!config_.constrain_orientation)
            {
                pose_error.tail<3>().setConstant(sx(0.0));
            }

            return pose_error;
        };

    auto buildGravityAdjustedTorqueLimitsSymbolic =
        [&](const CasadiVectorXs& state) -> CasadiVectorXs
        {
            CasadiData data(arm_model_casadi);
            const CasadiVectorXs arm_q = armStateSymbolic(state);
            const CasadiVectorXs gravity =
                pinocchio::computeGeneralizedGravity(arm_model_casadi, data, arm_q);

            CasadiVectorXs adjusted_limits(2 * arm_model_.nv);
            for (int i = 0; i < arm_model_.nv; ++i)
            {
                adjusted_limits(i) = sx(arm_torque_limits_(i)) - gravity(i);
                adjusted_limits(arm_model_.nv + i) =
                    -sx(arm_torque_limits_(i)) - gravity(i);
            }
            return adjusted_limits;
        };

    auto buildDynamicResidualTorqueLimitsSymbolic =
        [&](const CasadiVectorXs& state,
            const CasadiVectorXs* previous_state,
            const CasadiVectorXs* previous_previous_state,
            double previous_dt_sec,
            double previous_previous_dt_sec) -> CasadiVectorXs
        {
            if (!rfpComparisonShouldUseDynamicResidualForcePolytope(
                    config_.comparison_algorithm,
                    config_.use_dynamic_residual_force_polytope) ||
                previous_state == nullptr)
            {
                return buildGravityAdjustedTorqueLimitsSymbolic(state);
            }

            CasadiData data(arm_model_casadi);
            const CasadiVectorXs arm_q = armStateSymbolic(state);
            const CasadiVectorXs previous_arm_q = armStateSymbolic(*previous_state);
            const CasadiScalar dt = sx(std::max(kEpsilon, previous_dt_sec));
            const CasadiScalar prev_dt =
                sx(std::max(kEpsilon, previous_previous_dt_sec));
            const CasadiVectorXs arm_dq = (arm_q - previous_arm_q) / dt;

            CasadiVectorXs previous_arm_dq(arm_model_casadi.nv);
            previous_arm_dq.setConstant(sx(0.0));
            if (previous_previous_state != nullptr)
            {
                const CasadiVectorXs previous_previous_arm_q =
                    armStateSymbolic(*previous_previous_state);
                previous_arm_dq =
                    (previous_arm_q - previous_previous_arm_q) / prev_dt;
            }

            const CasadiVectorXs arm_ddq = (arm_dq - previous_arm_dq) / dt;
            const CasadiVectorXs nominal_torque =
                pinocchio::rnea(
                    arm_model_casadi,
                    data,
                    arm_q,
                    arm_dq,
                    arm_ddq);

            CasadiVectorXs adjusted_limits(2 * arm_model_.nv);
            for (int i = 0; i < arm_model_.nv; ++i)
            {
                adjusted_limits(i) =
                    sx(arm_torque_limits_(i)) - nominal_torque(i);
                adjusted_limits(arm_model_.nv + i) =
                    -sx(arm_torque_limits_(i)) - nominal_torque(i);
            }
            return adjusted_limits;
        };

    auto computeResidualForceCapacityCostSymbolic =
        [&](const CasadiScalar& signed_clearance,
            const CasadiScalar& normalization_scale) -> CasadiScalar
        {
            const CasadiScalar eps = sx(1e-6);
            const CasadiScalar normalized_clearance =
                signed_clearance / fmax(fabs(normalization_scale), eps);
            const CasadiScalar normalized_violation =
                fmax(sx(0.0), -normalized_clearance);
            const CasadiScalar exponent =
                sx(kCapabilityExponentScale) * normalized_violation;
            if (config_.capability_exponent_limit_enabled)
            {
                return exp(fmin(sx(kCapabilityMaxExponent), exponent)) - sx(1.0);
            }
            return exp(exponent) - sx(1.0);
        };

    auto computeForceBallSignedClearanceSymbolic =
        [&](const CasadiMatrixXs& translational_arm_jacobian,
            const CasadiVector3& force_center,
            double force_radius,
            const CasadiVectorXs& tau_lim) -> CasadiScalar
        {
            const CasadiScalar eps = sx(1e-9);
            const CasadiScalar clamped_force_radius = sx(std::max(0.0, force_radius));
            CasadiScalar min_clearance = sx(1e12);

            for (int i = 0; i < translational_arm_jacobian.cols(); ++i)
            {
                CasadiVector3 facet_normal = translational_arm_jacobian.col(i);
                CasadiScalar facet_normal_squared_norm = sx(0.0);
                for (int row = 0; row < 3; ++row)
                {
                    facet_normal_squared_norm += facet_normal(row) * facet_normal(row);
                }
                const CasadiScalar facet_normal_norm =
                    sqrt(fmax(facet_normal_squared_norm, eps * eps));
                const CasadiScalar safe_facet_normal_norm =
                    fmax(facet_normal_norm, eps);
                CasadiScalar projection = sx(0.0);
                for (int row = 0; row < 3; ++row)
                {
                    projection += facet_normal(row) * force_center(row);
                }

                const CasadiScalar scaled_upper_bound =
                    sx(config_.capability_alpha) * tau_lim(i);
                const CasadiScalar scaled_lower_bound =
                    sx(config_.capability_alpha) *
                    (-tau_lim(translational_arm_jacobian.cols() + i));
                const CasadiScalar upper_clearance =
                    (scaled_upper_bound - projection) / safe_facet_normal_norm -
                    clamped_force_radius;
                const CasadiScalar lower_clearance =
                    (scaled_lower_bound + projection) / safe_facet_normal_norm -
                    clamped_force_radius;

                min_clearance =
                    fmin(
                        min_clearance,
                        if_else(
                            facet_normal_norm > eps,
                            upper_clearance,
                            sx(1e12)));
                min_clearance =
                    fmin(
                        min_clearance,
                        if_else(
                            facet_normal_norm > eps,
                            lower_clearance,
                            sx(1e12)));
            }

            return if_else(min_clearance < sx(1e11), min_clearance, sx(0.0));
        };

    auto computeForceCapacitySymbolic =
        [&](const CasadiMatrixXs& translational_arm_jacobian,
            const CasadiVector3& force,
            const CasadiVectorXs& tau_lim) -> CasadiScalar
        {
            const casadi::SX jacobian = toCasadiVector(translational_arm_jacobian);
            const casadi::SX force_vector = toCasadiVector(force);
            const casadi::SX tau_limits = toCasadiVector(tau_lim);

            const casadi::SX v_tau = mtimes(jacobian.T(), force_vector);
            const CasadiScalar magnitude = norm_2(v_tau);
            const CasadiScalar safe_magnitude = fmax(magnitude, sx(1e-9));
            const casadi::SX v_tau_n = v_tau / safe_magnitude;

            casadi::SX block_matrix =
                casadi::SX::zeros(2 * translational_arm_jacobian.cols(),
                                  2 * translational_arm_jacobian.cols());
            for (int i = 0; i < translational_arm_jacobian.cols(); ++i)
            {
                block_matrix(i, i) = v_tau_n(i);
                block_matrix(translational_arm_jacobian.cols() + i,
                             translational_arm_jacobian.cols() + i) = v_tau_n(i);
            }

            casadi::SX system_matrix = block_matrix;
            for (int i = 0; i < system_matrix.size1(); ++i)
            {
                system_matrix(i, i) = system_matrix(i, i) + sx(1e-8);
            }

            const casadi::SX d_tau = solve(system_matrix, tau_limits);
            casadi::SX positive_d_tau = d_tau;
            for (int i = 0; i < d_tau.size1(); ++i)
            {
                positive_d_tau(i) =
                    if_else(d_tau(i) > sx(0.0), d_tau(i), sx(1e12));
            }

            const CasadiScalar d_tau_min = mmin(positive_d_tau);
            const casadi::SX opt_force_max =
                d_tau_min * mtimes(pinv(jacobian.T()), v_tau_n);
            return if_else(magnitude > sx(1e-9), norm_2(opt_force_max), sx(0.0));
        };

    auto computeManipulabilitySymbolic =
        [&](const CasadiVectorXs& state) -> CasadiScalar
        {
            const CasadiSE3 w_T_base = computeWorldToBaseSymbolic(state);
            const CasadiSE3 w_T_ee = computeForwardKinematicsSymbolic(state);
            const CasadiMatrix6x3 base_jacobian =
                computeBaseJacobianWorldSymbolic(w_T_base, w_T_ee);
            const CasadiMatrixXs arm_jacobian =
                computeArmJacobianWorldSymbolic(state);

            CasadiMatrixXs whole_body_jacobian(6, state_dim_);
            whole_body_jacobian.leftCols(kBasePlanarDof) = base_jacobian;
            whole_body_jacobian.rightCols(arm_model_.nv) = arm_jacobian;

            CasadiMatrixXs optimization_jacobian(3, 1 + arm_model_.nv);
            optimization_jacobian.setConstant(sx(0.0));
            optimization_jacobian.col(0) = whole_body_jacobian.topRows(3).col(2);
            optimization_jacobian.rightCols(arm_model_.nv) =
                whole_body_jacobian.topRows(3).rightCols(arm_model_.nv);

            CasadiMatrix3 gram;
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    CasadiScalar entry = sx(0.0);
                    for (int k = 0; k < optimization_jacobian.cols(); ++k)
                    {
                        entry += optimization_jacobian(i, k) * optimization_jacobian(j, k);
                    }
                    if (i == j)
                    {
                        entry += sx(config_.manipulability_regularization);
                    }
                    gram(i, j) = entry;
                }
            }

            return sqrt(
                fmax(
                    sx(config_.manipulability_epsilon),
                    determinant3x3(gram)));
        };

    auto evaluateCostSymbolic =
        [&](const CasadiVectorXs& state,
            const Eigen::Vector3d& desired_force,
            double desired_force_radius,
            const CasadiVectorXs& nominal_state,
            const CasadiVectorXs* previous_state,
            double previous_dt_sec,
            const CasadiVectorXs* previous_previous_state,
            double previous_previous_dt_sec) -> CasadiScalar
        {
            const CasadiVectorXs clamped_nominal_state = nominal_state;
            const CasadiVector3 desired_force_symbolic =
                constantCasadiVector3(desired_force);
            const CasadiScalar required_force =
                sqrt(squaredNorm(desired_force_symbolic));
            const CasadiScalar force_radius = sx(std::max(0.0, desired_force_radius));

            CasadiScalar capability_cost = sx(0.0);
            const RfpComparisonAlgorithmPolicy comparison_policy =
                rfpComparisonAlgorithmPolicy(config_.comparison_algorithm);
            if (comparison_policy.include_capability &&
                (desired_force.norm() > kEpsilon ||
                 desired_force_radius > kEpsilon ||
                 comparison_policy.capability_objective != RfpCapabilityObjective::Clearance))
            {
                const CasadiMatrixXs translational_arm_jacobian =
                    computeArmJacobianWorldSymbolic(state).topRows(3);
                const CasadiVectorXs adjusted_limits =
                    buildDynamicResidualTorqueLimitsSymbolic(
                        state,
                        previous_state,
                        previous_previous_state,
                        previous_dt_sec,
                        previous_previous_dt_sec);
                if (comparison_policy.capability_objective ==
                    RfpCapabilityObjective::CenteredBallRadius)
                {
                    const CasadiScalar radius =
                        computeCenteredForceBallRadiusSymbolic(
                            translational_arm_jacobian,
                            adjusted_limits,
                            config_.capability_alpha);
                    capability_cost =
                        positiveQuantityMaximizationCostSymbolic(radius);
                }
                else if (comparison_policy.capability_objective ==
                         RfpCapabilityObjective::ConeIntersectionVolume)
                {
                    RfpConeSamplingConfig cone_config;
                    cone_config.axis_world =
                        desired_force.norm() > kEpsilon
                            ? desired_force.normalized()
                            : config_.comparison_cone_axis_world;
                    cone_config.half_angle_rad = config_.comparison_cone_half_angle_rad;
                    cone_config.ring_count = config_.comparison_cone_ring_count;
                    cone_config.azimuth_count = config_.comparison_cone_azimuth_count;
                    const CasadiScalar volume_proxy =
                        computeConeIntersectionVolumeProxySymbolic(
                            translational_arm_jacobian,
                            adjusted_limits,
                            cone_config,
                            config_.capability_alpha);
                    capability_cost =
                        positiveQuantityMaximizationCostSymbolic(volume_proxy);
                }
                else
                {
                    const CasadiScalar signed_clearance =
                        computeForceBallSignedClearanceSymbolic(
                            translational_arm_jacobian,
                            desired_force_symbolic,
                            desired_force_radius,
                            adjusted_limits);
                    capability_cost =
                        computeResidualForceCapacityCostSymbolic(
                            signed_clearance,
                            required_force + force_radius);
                }
            }

            const CasadiScalar manipulability_measure =
                computeManipulabilitySymbolic(state);
            const CasadiScalar safe_manipulability =
                fmax(
                    sx(config_.manipulability_epsilon),
                    manipulability_measure);
            const CasadiScalar manipulability_cost =
                sx(1.0) / safe_manipulability;

            // Legacy manipulability costs kept here for reference:
            // const CasadiScalar manipulability_cost =
            //     sx(1.0) / (safe_manipulability * safe_manipulability);
            // const CasadiScalar manipulability_cost =
            //     -log(
            //         fmax(
            //             sx(config_.manipulability_epsilon),
            //             manipulability_measure));

            const CasadiVectorXs current_arm = armStateSymbolic(state);
            CasadiScalar joint_limit_cost = sx(0.0);
            const CasadiVector3 current_base = baseStateSymbolic(state);
            const CasadiVector3 nominal_base = baseStateSymbolic(clamped_nominal_state);
            const CasadiVector3 base_nominal_diff =
                baseDifferenceSymbolic(current_base, nominal_base);

            for (int i = 0; i < kBasePlanarDof; ++i)
            {
                const double weight = config_.joint_limit_dof_weights(i);
                if (weight <= 0.0)
                {
                    continue;
                }
                joint_limit_cost +=
                    sx(weight) * base_nominal_diff(i) * base_nominal_diff(i);
            }

            for (int i = 0; i < current_arm.size(); ++i)
            {
                const double joint_weight =
                    config_.joint_limit_dof_weights(kBasePlanarDof + i);
                if (joint_weight <= 0.0)
                {
                    continue;
                }

                const double lower = lower_arm_limits_(i);
                const double upper = upper_arm_limits_(i);
                const double margin = arm_joint_limit_margins_(i);
                if (!std::isfinite(lower) || !std::isfinite(upper) || lower >= upper || margin <= 0.0)
                {
                    continue;
                }

                const double center = 0.5 * (lower + upper);
                const double half_range = 0.5 * (upper - lower);
                const double safe_half_range =
                    std::max(1e-6, half_range - margin);
                const CasadiScalar normalized_offset =
                    (current_arm(i) - sx(center)) / sx(safe_half_range);
                joint_limit_cost +=
                    sx(joint_weight) * normalized_offset * normalized_offset;

                const CasadiScalar distance_from_center =
                    fabs(current_arm(i) - sx(center));
                const CasadiScalar normalized_excess =
                    (distance_from_center - sx(safe_half_range)) /
                    sx(std::max(1e-6, margin));
                joint_limit_cost +=
                    if_else(
                        distance_from_center > sx(safe_half_range),
                        sx(joint_weight) *
                            normalized_excess * normalized_excess *
                            normalized_excess * normalized_excess,
                        sx(0.0));
            }

            const CasadiScalar base_nominal_cost =
                sx(0.5) * weightedSquaredNorm(
                    base_nominal_diff,
                    config_.base_nominal_metric);
            const CasadiScalar nominal_cost =
                sx(0.5) * squaredNorm(current_arm - armStateSymbolic(clamped_nominal_state));

            CasadiScalar base_smoothness_cost = sx(0.0);
            CasadiScalar smoothness_cost = sx(0.0);
            CasadiScalar base_velocity_cost = sx(0.0);
            CasadiScalar velocity_cost = sx(0.0);
            if (previous_state != nullptr)
            {
                const CasadiVector3 previous_base = baseStateSymbolic(*previous_state);
                const CasadiVector3 base_smoothness_diff =
                    baseDifferenceSymbolic(current_base, previous_base);
                const double safe_dt = std::max(kEpsilon, previous_dt_sec);
                base_smoothness_cost =
                    sx(0.5) * weightedSquaredNorm(
                        base_smoothness_diff,
                        config_.base_smoothness_metric);
                smoothness_cost =
                    sx(0.5) *
                    squaredNorm(current_arm - armStateSymbolic(*previous_state));
                base_velocity_cost =
                    base_smoothness_cost / sx(safe_dt * safe_dt);
                velocity_cost =
                    smoothness_cost / sx(safe_dt * safe_dt);
            }

            const CasadiScalar collision_cost =
                computeCollisionCostSymbolic(state);

            const RfpComparisonAlgorithmPolicy policy =
                rfpComparisonAlgorithmPolicy(config_.comparison_algorithm);
            CasadiScalar objective = sx(0.0);
            if (policy.include_capability)
            {
                objective += sx(config_.capability_weight) * capability_cost;
            }
            if (policy.include_manipulability)
            {
                objective += sx(config_.manipulability_weight) * manipulability_cost;
            }
            if (policy.include_joint_limit)
            {
                objective += sx(config_.joint_limit_weight) * joint_limit_cost;
            }
            if (policy.include_trajectory_regularization &&
                !input.single_point_cost_terms_only)
            {
                objective +=
                    sx(config_.smoothness_weight) * smoothness_cost +
                    sx(config_.velocity_weight) * velocity_cost +
                    sx(config_.nominal_weight) * nominal_cost +
                    sx(config_.base_smoothness_weight) * base_smoothness_cost +
                    sx(config_.base_velocity_weight) * base_velocity_cost +
                    sx(config_.base_nominal_weight) * base_nominal_cost +
                    sx(config_.collision_weight) * collision_cost;
            }
            return objective;
        };

    auto computeBaseSpectralEnergyCostSymbolic =
        [&](const std::vector<CasadiVectorXs>& trajectory) -> CasadiScalar
        {
            if (trajectory.size() < 2 || config_.base_spectral_energy_weight <= 0.0)
            {
                return sx(0.0);
            }

            const int sample_count = static_cast<int>(trajectory.size());
            const double average_dt_sec =
                estimateAverageWaypointDtSec(input.waypoint_times_sec, sample_count);
            const double sample_rate_hz = 1.0 / std::max(kEpsilon, average_dt_sec);
            const double nyquist_hz = 0.5 * sample_rate_hz;
            const double cutoff_ratio = clampUnitInterval(config_.base_spectral_cutoff_ratio);
            const double cutoff_hz = cutoff_ratio * nyquist_hz;
            const double spectral_power = std::max(0.0, config_.base_spectral_power);

            std::vector<CasadiScalar> base_x(sample_count, sx(0.0));
            std::vector<CasadiScalar> base_y(sample_count, sx(0.0));
            std::vector<CasadiScalar> base_yaw(sample_count, sx(0.0));

            CasadiScalar unwrapped_yaw =
                baseStateSymbolic(trajectory.front())(2);
            for (int k = 0; k < sample_count; ++k)
            {
                const CasadiVector3 current_base =
                    baseStateSymbolic(trajectory[static_cast<std::size_t>(k)]);
                base_x[k] = current_base(0);
                base_y[k] = current_base(1);

                if (k == 0)
                {
                    base_yaw[k] = unwrapped_yaw;
                }
                else
                {
                    const CasadiVector3 previous_base =
                        baseStateSymbolic(trajectory[static_cast<std::size_t>(k - 1)]);
                    unwrapped_yaw +=
                        wrapAngleSymbolic(current_base(2) - previous_base(2));
                    base_yaw[k] = unwrapped_yaw;
                }
            }

            const std::array<std::vector<CasadiScalar>, 3> base_signals{
                {base_x, base_y, base_yaw}};
            CasadiScalar spectral_cost = sx(0.0);
            const int highest_bin = std::max(1, sample_count - 1);

            for (int axis = 0; axis < kBasePlanarDof; ++axis)
            {
                const double axis_metric = config_.base_spectral_energy_metric(axis);
                if (axis_metric <= 0.0)
                {
                    continue;
                }

                for (int k = 1; k < sample_count; ++k)
                {
                    const double frequency_hz =
                        static_cast<double>(k) * nyquist_hz /
                        static_cast<double>(highest_bin);
                    if (frequency_hz <= cutoff_hz)
                    {
                        continue;
                    }

                    CasadiScalar coefficient = sx(0.0);
                    const double inv_sample_count =
                        1.0 / static_cast<double>(sample_count);
                    for (int n = 0; n < sample_count; ++n)
                    {
                        coefficient +=
                            sx(std::cos(
                                M_PI *
                                (static_cast<double>(n) + 0.5) *
                                static_cast<double>(k) *
                                inv_sample_count)) *
                            base_signals[axis][n];
                    }

                    const double scale =
                        k == 0
                            ? std::sqrt(inv_sample_count)
                            : std::sqrt(2.0 * inv_sample_count);
                    coefficient *= sx(scale);

                    const double normalized_tail_frequency =
                        (frequency_hz - cutoff_hz) /
                        std::max(kEpsilon, nyquist_hz - cutoff_hz);
                    const double spectral_weight =
                        std::pow(normalized_tail_frequency, spectral_power);

                    spectral_cost +=
                        sx(axis_metric * spectral_weight) *
                        coefficient * coefficient;
                }
            }

            return sx(0.5) * spectral_cost;
        };

    std::ostringstream callback_suffix;
    callback_suffix << reinterpret_cast<std::uintptr_t>(this) << "_" << horizon;

    casadi::SX X = casadi::SX::sym("X", decision_dim, 1);
    std::vector<CasadiVectorXs> symbolic_trajectory;
    symbolic_trajectory.reserve(horizon);
    for (std::size_t k = 0; k < horizon; ++k)
    {
        symbolic_trajectory.push_back(
            extractSymbolicState(X, state_dim_, k));
    }

    std::vector<CasadiVectorXs> symbolic_nominal_trajectory;
    symbolic_nominal_trajectory.reserve(horizon);
    for (std::size_t k = 0; k < horizon; ++k)
    {
        symbolic_nominal_trajectory.push_back(
            constantCasadiVector(initial_trajectory[k]));
    }

    CasadiScalar objective = sx(0.0);
    for (std::size_t k = 0; k < horizon; ++k)
    {
        const CasadiVectorXs* previous_state =
            k == 0 ? nullptr : &symbolic_trajectory[k - 1];
        const CasadiVectorXs* previous_previous_state =
            k < 2 ? nullptr : &symbolic_trajectory[k - 2];
        const double previous_dt_sec =
            waypointPreviousDtSec(input.waypoint_times_sec, k);
        const double previous_previous_dt_sec =
            k < 2 ? previous_dt_sec : waypointPreviousDtSec(input.waypoint_times_sec, k - 1);
        objective += evaluateCostSymbolic(
            symbolic_trajectory[k],
            input.desired_forces[k],
            resolveDesiredForceRadius(input, k),
            symbolic_nominal_trajectory[k],
            previous_state,
            previous_dt_sec,
            previous_previous_state,
            previous_previous_dt_sec);
    }

    const RfpComparisonAlgorithmPolicy objective_policy =
        rfpComparisonAlgorithmPolicy(config_.comparison_algorithm);
    if (objective_policy.include_trajectory_regularization &&
        !input.single_point_cost_terms_only)
    {
        objective +=
            sx(config_.base_spectral_energy_weight) *
            computeBaseSpectralEnergyCostSymbolic(symbolic_trajectory);
    }

    std::vector<casadi::SX> constraint_values;
    constraint_values.reserve(constraint_dim);
    for (std::size_t k = 0; k < horizon; ++k)
    {
        const CasadiVector6 pose_error =
            computePoseErrorSymbolic(symbolic_trajectory[k], input.target_poses[k]);
        const int pose_constraint_dim = config_.constrain_orientation ? 6 : 3;
        for (int i = 0; i < pose_constraint_dim; ++i)
        {
            constraint_values.push_back(pose_error(i));
        }
    }

    if (input.pin_first_state)
    {
        const CasadiVector3 first_base_error =
            baseDifferenceSymbolic(
                baseStateSymbolic(symbolic_trajectory.front()),
                baseStateSymbolic(symbolic_nominal_trajectory.front()));
        for (int i = 0; i < kBasePlanarDof; ++i)
        {
            constraint_values.push_back(first_base_error(i));
        }

        const CasadiVectorXs first_arm_error =
            armStateSymbolic(symbolic_trajectory.front()) -
            armStateSymbolic(symbolic_nominal_trajectory.front());
        for (int i = 0; i < first_arm_error.size(); ++i)
        {
            constraint_values.push_back(first_arm_error(i));
        }
    }

    const casadi::SX constraints = casadi::SX::vertcat(constraint_values);

    casadi::SXDict nlp;
    nlp["x"] = X;
    nlp["f"] = objective;
    nlp["g"] = constraints;

    casadi::Dict solver_options;
    solver_options["ipopt.print_level"] = config_.verbose ? 5 : 0;
    solver_options["print_time"] = config_.verbose;
    solver_options["ipopt.sb"] = "yes";
    solver_options["ipopt.max_iter"] =
        std::max(10, config_.max_trajectory_iterations);
    solver_options["ipopt.hessian_approximation"] = "limited-memory";
    solver_options["ipopt.mu_strategy"] = "adaptive";
    solver_options["ipopt.tol"] = config_.pose_tolerance;
    solver_options["ipopt.acceptable_tol"] = config_.pose_tolerance;
    solver_options["ipopt.constr_viol_tol"] = config_.pose_tolerance;
    solver_options["ipopt.acceptable_constr_viol_tol"] = config_.pose_tolerance;
    solver_options["ipopt.acceptable_iter"] = 3;
    solver_options["ipopt.acceptable_obj_change_tol"] = 1e-3;
    solver_options["ipopt.max_cpu_time"] =
        std::max(1.0, 0.05 * static_cast<double>(horizon) * config_.max_trajectory_iterations);

    const casadi::Function solver =
        casadi::nlpsol(
            "whole_body_trajectory_solver_" + callback_suffix.str(),
            "ipopt",
            nlp,
            solver_options);

    std::vector<double> x0_values;
    x0_values.reserve(decision_dim);
    std::vector<double> lower_bounds(decision_dim, -std::numeric_limits<double>::infinity());
    std::vector<double> upper_bounds(decision_dim, std::numeric_limits<double>::infinity());
    for (std::size_t k = 0; k < horizon; ++k)
    {
        const Eigen::VectorXd clamped_nominal =
            clampToStateLimits(input.nominal_state_trajectory[k]);
        for (int i = 0; i < state_dim_; ++i)
        {
            x0_values.push_back(clamped_nominal(i));
        }

        for (int i = 0; i < kBasePlanarDof; ++i)
        {
            const int index = static_cast<int>(k) * state_dim_ + i;
            const double lower = config_.base_lower_limits(i);
            const double upper = config_.base_upper_limits(i);
            if (std::isfinite(lower))
            {
                lower_bounds[index] = lower;
            }
            if (std::isfinite(upper))
            {
                upper_bounds[index] = upper;
            }
        }

        for (int i = 0; i < arm_model_.nq; ++i)
        {
            const int index = static_cast<int>(k) * state_dim_ + kBasePlanarDof + i;
            if (std::isfinite(lower_arm_limits_(i)))
            {
                lower_bounds[index] = lower_arm_limits_(i) + 1e-6;
            }
            if (std::isfinite(upper_arm_limits_(i)))
            {
                upper_bounds[index] = upper_arm_limits_(i) - 1e-6;
            }
        }
    }

    casadi::DMDict solver_input;
    solver_input["x0"] = denseColumn(x0_values);
    solver_input["lbx"] = denseColumn(lower_bounds);
    solver_input["ubx"] = denseColumn(upper_bounds);
    solver_input["lbg"] = casadi::DM::zeros(constraint_dim, 1);
    solver_input["ubg"] = casadi::DM::zeros(constraint_dim, 1);

    const casadi::DMDict solver_output = solver(solver_input);
    const std::vector<Eigen::VectorXd> optimized_trajectory =
        unpackDecisionTrajectory(
            solver_output.at("x").get_nonzeros<double>(),
            state_dim_,
            horizon);
    writeDynamicResidualDiagnostics(input, optimized_trajectory, "optimized");

    const casadi::Dict solver_stats = solver.stats();
    int trajectory_iterations = 0;
    bool solver_success = true;
    std::string solver_return_status = "unknown";
    auto iter_count_it = solver_stats.find("iter_count");
    if (iter_count_it != solver_stats.end())
    {
        trajectory_iterations = static_cast<int>(iter_count_it->second);
    }
    auto success_it = solver_stats.find("success");
    if (success_it != solver_stats.end())
    {
        solver_success = static_cast<bool>(success_it->second);
    }
    auto return_status_it = solver_stats.find("return_status");
    if (return_status_it != solver_stats.end())
    {
        std::ostringstream return_status_stream;
        return_status_stream << return_status_it->second;
        solver_return_status = return_status_stream.str();
    }

    WholeBodyTrajectoryOptimizationResult result;
    result.trajectory_iterations = trajectory_iterations;
    result.optimized_state_trajectory = optimized_trajectory;
    result.base_spectral_energy_cost =
        computeBaseSpectralEnergyCost(
            result.optimized_state_trajectory,
            input.waypoint_times_sec);
    result.waypoint_metrics.reserve(horizon);

    bool all_waypoints_valid = true;
    bool all_waypoints_finite = true;
    double max_pose_error = 0.0;
    for (std::size_t k = 0; k < horizon; ++k)
    {
        const Eigen::VectorXd* previous_state =
            k == 0 ? nullptr : &result.optimized_state_trajectory[k - 1];
        const Eigen::VectorXd* previous_previous_state =
            k < 2 ? nullptr : &result.optimized_state_trajectory[k - 2];
        const double previous_dt_sec =
            waypointPreviousDtSec(input.waypoint_times_sec, k);
        const double previous_previous_dt_sec =
            k < 2 ? previous_dt_sec : waypointPreviousDtSec(input.waypoint_times_sec, k - 1);
        const CostBreakdown breakdown =
            evaluateCost(
                result.optimized_state_trajectory[k],
                input.desired_forces[k],
                resolveDesiredForceRadius(input, k),
                input.nominal_state_trajectory[k],
                previous_state,
                previous_dt_sec,
                previous_previous_state,
                previous_previous_dt_sec);
        const Eigen::Matrix<double, 6, 1> final_pose_error =
            computePoseError(result.optimized_state_trajectory[k], input.target_poses[k]);

        WholeBodyWaypointOptimizationMetrics metrics;
        metrics.objective_value =
            breakdown.objective_value;
        metrics.force_capacity = breakdown.force_capacity;
        metrics.required_force = breakdown.required_force;
        metrics.required_force_radius = breakdown.required_force_radius;
        metrics.force_ball_clearance = breakdown.force_ball_clearance;
        metrics.capability_cost = breakdown.capability_cost;
        metrics.manipulability_measure = breakdown.manipulability_measure;
        metrics.manipulability_cost = breakdown.manipulability_cost;
        metrics.joint_limit_cost = breakdown.joint_limit_cost;
        metrics.smoothness_cost = breakdown.smoothness_cost;
        metrics.velocity_cost = breakdown.velocity_cost;
        metrics.nominal_cost = breakdown.nominal_cost;
        metrics.base_smoothness_cost = breakdown.base_smoothness_cost;
        metrics.base_velocity_cost = breakdown.base_velocity_cost;
        metrics.base_nominal_cost = breakdown.base_nominal_cost;
        metrics.collision_cost = breakdown.collision_cost;
        metrics.min_collision_clearance = breakdown.min_collision_clearance;
        metrics.pose_error_norm =
            (config_.constrain_orientation ? final_pose_error : final_pose_error.head(3)).norm();
        metrics.iterations = trajectory_iterations;
        result.waypoint_metrics.push_back(metrics);

        const bool waypoint_finite =
            result.optimized_state_trajectory[k].allFinite() &&
            std::isfinite(metrics.objective_value) &&
            std::isfinite(metrics.pose_error_norm) &&
            std::isfinite(metrics.force_capacity);
        all_waypoints_finite = all_waypoints_finite && waypoint_finite;
        const bool validation_waypoint = input.pin_first_state ? k > 0 : true;
        if (validation_waypoint)
        {
            max_pose_error = std::max(max_pose_error, metrics.pose_error_norm);

            const bool waypoint_valid =
                metrics.pose_error_norm <
                kStrictWaypointPoseValidationMultiplier * config_.pose_tolerance;
            all_waypoints_valid = all_waypoints_valid && waypoint_valid;
        }

        if (config_.verbose)
        {
            std::cout
                << "[WholeBodyPolytopeOptimizer] waypoint " << k
                << " trajectory_iterations=" << trajectory_iterations
                << " pose_error=" << metrics.pose_error_norm
                << " force_capacity=" << metrics.force_capacity
                << " objective=" << metrics.objective_value
                << " base_spectral_energy=" << result.base_spectral_energy_cost
                << std::endl;
        }
    }

    const bool trajectory_quality_valid =
        all_waypoints_valid && all_waypoints_finite;
    result.success = trajectory_quality_valid;
    result.message =
        result.success
            ? "Whole-body trajectory optimization completed successfully."
            : "Whole-body trajectory optimization completed with solver or waypoint errors.";
    if (!result.success)
    {
        std::ostringstream error_stream;
        error_stream
            << "Whole-body trajectory optimization rejected. solver_success="
            << (solver_success ? "true" : "false")
            << ", return_status=" << solver_return_status
            << ", max_pose_error=" << max_pose_error
            << ", allowed_pose_error="
            << (kStrictWaypointPoseValidationMultiplier * config_.pose_tolerance);
        result.message = error_stream.str();
    }
    else if (!solver_success)
    {
        std::ostringstream warning_stream;
        warning_stream
            << "Whole-body trajectory optimization accepted with solver_success=false"
            << ", return_status=" << solver_return_status
            << ", max_pose_error=" << max_pose_error;
        result.message = warning_stream.str();
    }

    return result;
}

} // namespace polytope_wx
