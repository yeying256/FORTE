#include "polytope_ros/rfp_comparison_costs.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace polytope_wx
{
namespace
{

Eigen::Vector3d safeNormalizedAxis(const Eigen::Vector3d& axis_world)
{
    if (!axis_world.allFinite() || axis_world.norm() < 1e-9)
    {
        return Eigen::Vector3d::UnitZ();
    }
    return axis_world.normalized();
}

double rayCapacityAlongDirection(
    const Eigen::MatrixXd& translational_jacobian,
    const Eigen::VectorXd& torque_limits,
    const Eigen::Vector3d& direction,
    double alpha,
    double eps)
{
    const int dof = translational_jacobian.cols();
    double capacity = std::numeric_limits<double>::infinity();
    for (int i = 0; i < dof; ++i)
    {
        const Eigen::Vector3d facet_normal = translational_jacobian.col(i);
        const double projection = facet_normal.dot(direction);
        const double upper_bound = alpha * torque_limits(i);
        const double lower_bound = alpha * (-torque_limits(dof + i));

        if (projection > eps)
        {
            capacity = std::min(capacity, upper_bound / projection);
        }
        if (projection < -eps)
        {
            capacity = std::min(capacity, lower_bound / (-projection));
        }
    }

    if (!std::isfinite(capacity))
    {
        return 0.0;
    }
    return std::max(0.0, capacity);
}

casadi::SX rayCapacityAlongDirectionSymbolic(
    const Eigen::Matrix<casadi::SX, Eigen::Dynamic, Eigen::Dynamic>& translational_jacobian,
    const Eigen::Matrix<casadi::SX, Eigen::Dynamic, 1>& torque_limits,
    const Eigen::Vector3d& direction,
    double alpha,
    double eps)
{
    const int dof = translational_jacobian.cols();
    casadi::SX capacity = casadi::SX(1e12);
    for (int i = 0; i < dof; ++i)
    {
        casadi::SX projection = casadi::SX(0.0);
        for (int row = 0; row < 3; ++row)
        {
            projection += translational_jacobian(row, i) * direction(row);
        }

        const casadi::SX upper_capacity =
            casadi::SX(alpha) * torque_limits(i) /
            fmax(projection, casadi::SX(eps));
        const casadi::SX lower_capacity =
            casadi::SX(alpha) * (-torque_limits(dof + i)) /
            fmax(-projection, casadi::SX(eps));

        capacity = fmin(
            capacity,
            if_else(
                projection > casadi::SX(eps),
                upper_capacity,
                casadi::SX(1e12)));
        capacity = fmin(
            capacity,
            if_else(
                projection < casadi::SX(-eps),
                lower_capacity,
                casadi::SX(1e12)));
    }

    return fmax(
        casadi::SX(0.0),
        if_else(capacity < casadi::SX(1e11), capacity, casadi::SX(0.0)));
}

}  // namespace

double computeCenteredForceBallRadius(
    const Eigen::MatrixXd& translational_jacobian,
    const Eigen::VectorXd& torque_limits,
    double alpha,
    double eps)
{
    const int dof = translational_jacobian.cols();
    if (translational_jacobian.rows() != 3 ||
        torque_limits.size() != 2 * dof ||
        dof <= 0)
    {
        return 0.0;
    }

    double radius = std::numeric_limits<double>::infinity();
    for (int i = 0; i < dof; ++i)
    {
        const Eigen::Vector3d facet_normal = translational_jacobian.col(i);
        const double normal_norm = facet_normal.norm();
        if (normal_norm <= eps)
        {
            continue;
        }

        const double upper_bound = alpha * torque_limits(i);
        const double lower_bound = alpha * (-torque_limits(dof + i));
        radius = std::min(radius, upper_bound / normal_norm);
        radius = std::min(radius, lower_bound / normal_norm);
    }

    if (!std::isfinite(radius))
    {
        return 0.0;
    }
    return std::max(0.0, radius);
}

casadi::SX computeCenteredForceBallRadiusSymbolic(
    const Eigen::Matrix<casadi::SX, Eigen::Dynamic, Eigen::Dynamic>& translational_jacobian,
    const Eigen::Matrix<casadi::SX, Eigen::Dynamic, 1>& torque_limits,
    double alpha,
    double eps)
{
    const int dof = translational_jacobian.cols();
    casadi::SX radius = casadi::SX(1e12);
    for (int i = 0; i < dof; ++i)
    {
        casadi::SX normal_squared = casadi::SX(0.0);
        for (int row = 0; row < 3; ++row)
        {
            normal_squared +=
                translational_jacobian(row, i) * translational_jacobian(row, i);
        }
        const casadi::SX normal_norm =
            sqrt(fmax(normal_squared, casadi::SX(eps * eps)));
        const casadi::SX upper_radius =
            casadi::SX(alpha) * torque_limits(i) / normal_norm;
        const casadi::SX lower_radius =
            casadi::SX(alpha) * (-torque_limits(dof + i)) / normal_norm;

        radius = fmin(
            radius,
            if_else(
                normal_norm > casadi::SX(eps),
                upper_radius,
                casadi::SX(1e12)));
        radius = fmin(
            radius,
            if_else(
                normal_norm > casadi::SX(eps),
                lower_radius,
                casadi::SX(1e12)));
    }

    return fmax(
        casadi::SX(0.0),
        if_else(radius < casadi::SX(1e11), radius, casadi::SX(0.0)));
}

std::vector<Eigen::Vector3d> buildConeSampleDirections(
    const Eigen::Vector3d& axis_world,
    double half_angle_rad,
    int ring_count,
    int azimuth_count)
{
    const Eigen::Vector3d axis = safeNormalizedAxis(axis_world);
    const double clamped_half_angle =
        std::max(0.0, std::min(M_PI, half_angle_rad));
    const int safe_ring_count = std::max(1, ring_count);
    const int safe_azimuth_count = std::max(3, azimuth_count);

    Eigen::Vector3d reference = Eigen::Vector3d::UnitX();
    if (std::abs(axis.dot(reference)) > 0.9)
    {
        reference = Eigen::Vector3d::UnitY();
    }
    const Eigen::Vector3d tangent_a = axis.cross(reference).normalized();
    const Eigen::Vector3d tangent_b = axis.cross(tangent_a).normalized();

    std::vector<Eigen::Vector3d> directions;
    directions.reserve(
        1 + static_cast<std::size_t>(safe_ring_count) *
                static_cast<std::size_t>(safe_azimuth_count));
    directions.push_back(axis);

    for (int ring = 1; ring <= safe_ring_count; ++ring)
    {
        const double theta =
            clamped_half_angle *
            static_cast<double>(ring) /
            static_cast<double>(safe_ring_count);
        for (int sample = 0; sample < safe_azimuth_count; ++sample)
        {
            const double phi =
                2.0 * M_PI *
                static_cast<double>(sample) /
                static_cast<double>(safe_azimuth_count);
            Eigen::Vector3d direction =
                std::cos(theta) * axis +
                std::sin(theta) *
                    (std::cos(phi) * tangent_a + std::sin(phi) * tangent_b);
            directions.push_back(direction.normalized());
        }
    }
    return directions;
}

double computeConeIntersectionVolumeProxy(
    const Eigen::MatrixXd& translational_jacobian,
    const Eigen::VectorXd& torque_limits,
    const RfpConeSamplingConfig& cone_config,
    double alpha)
{
    const std::vector<Eigen::Vector3d> directions =
        buildConeSampleDirections(
            cone_config.axis_world,
            cone_config.half_angle_rad,
            cone_config.ring_count,
            cone_config.azimuth_count);
    if (directions.empty())
    {
        return 0.0;
    }

    double radial_cubic_sum = 0.0;
    for (const Eigen::Vector3d& direction : directions)
    {
        const double capacity =
            rayCapacityAlongDirection(
                translational_jacobian,
                torque_limits,
                direction,
                alpha,
                cone_config.eps);
        radial_cubic_sum += capacity * capacity * capacity;
    }

    const double solid_angle =
        2.0 * M_PI * (1.0 - std::cos(
            std::max(0.0, std::min(M_PI, cone_config.half_angle_rad))));
    return solid_angle * radial_cubic_sum /
           (3.0 * static_cast<double>(directions.size()));
}

casadi::SX computeConeIntersectionVolumeProxySymbolic(
    const Eigen::Matrix<casadi::SX, Eigen::Dynamic, Eigen::Dynamic>& translational_jacobian,
    const Eigen::Matrix<casadi::SX, Eigen::Dynamic, 1>& torque_limits,
    const RfpConeSamplingConfig& cone_config,
    double alpha)
{
    const std::vector<Eigen::Vector3d> directions =
        buildConeSampleDirections(
            cone_config.axis_world,
            cone_config.half_angle_rad,
            cone_config.ring_count,
            cone_config.azimuth_count);
    if (directions.empty())
    {
        return casadi::SX(0.0);
    }

    casadi::SX radial_cubic_sum = casadi::SX(0.0);
    for (const Eigen::Vector3d& direction : directions)
    {
        const casadi::SX capacity =
            rayCapacityAlongDirectionSymbolic(
                translational_jacobian,
                torque_limits,
                direction,
                alpha,
                cone_config.eps);
        radial_cubic_sum += capacity * capacity * capacity;
    }

    const double solid_angle =
        2.0 * M_PI * (1.0 - std::cos(
            std::max(0.0, std::min(M_PI, cone_config.half_angle_rad))));
    return casadi::SX(solid_angle / (3.0 * static_cast<double>(directions.size()))) *
           radial_cubic_sum;
}

double positiveQuantityMaximizationCost(double value, double eps)
{
    return 1.0 / std::max(std::abs(value), eps);
}

casadi::SX positiveQuantityMaximizationCostSymbolic(
    const casadi::SX& value,
    double eps)
{
    return casadi::SX(1.0) / fmax(fabs(value), casadi::SX(eps));
}

}  // namespace polytope_wx
