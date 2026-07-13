#pragma once

#include <Eigen/Dense>
#include <casadi/casadi.hpp>

#include <vector>

namespace polytope_wx
{

struct RfpConeSamplingConfig
{
    Eigen::Vector3d axis_world{Eigen::Vector3d::UnitZ()};
    double half_angle_rad{0.35};
    int ring_count{3};
    int azimuth_count{8};
    double eps{1e-9};
};

double computeCenteredForceBallRadius(
    const Eigen::MatrixXd& translational_jacobian,
    const Eigen::VectorXd& torque_limits,
    double alpha = 1.0,
    double eps = 1e-9);

casadi::SX computeCenteredForceBallRadiusSymbolic(
    const Eigen::Matrix<casadi::SX, Eigen::Dynamic, Eigen::Dynamic>& translational_jacobian,
    const Eigen::Matrix<casadi::SX, Eigen::Dynamic, 1>& torque_limits,
    double alpha = 1.0,
    double eps = 1e-9);

std::vector<Eigen::Vector3d> buildConeSampleDirections(
    const Eigen::Vector3d& axis_world,
    double half_angle_rad,
    int ring_count,
    int azimuth_count);

double computeConeIntersectionVolumeProxy(
    const Eigen::MatrixXd& translational_jacobian,
    const Eigen::VectorXd& torque_limits,
    const RfpConeSamplingConfig& cone_config,
    double alpha = 1.0);

casadi::SX computeConeIntersectionVolumeProxySymbolic(
    const Eigen::Matrix<casadi::SX, Eigen::Dynamic, Eigen::Dynamic>& translational_jacobian,
    const Eigen::Matrix<casadi::SX, Eigen::Dynamic, 1>& torque_limits,
    const RfpConeSamplingConfig& cone_config,
    double alpha = 1.0);

double positiveQuantityMaximizationCost(double value, double eps = 1e-6);

casadi::SX positiveQuantityMaximizationCostSymbolic(
    const casadi::SX& value,
    double eps = 1e-6);

}  // namespace polytope_wx
