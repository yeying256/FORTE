#pragma once

#include <string>

namespace polytope_wx
{

enum class RfpComparisonAlgorithm
{
    Ours,
    ManipulabilityOnly,
    StaticForcePolytope,
    ResidualForcePolytope,
    ConeIntersectionVolume,
    JointQuinticInterpolation
};

enum class RfpCapabilityObjective
{
    Clearance,
    CenteredBallRadius,
    ConeIntersectionVolume
};

struct RfpComparisonAlgorithmPolicy
{
    bool include_capability{true};
    bool include_manipulability{true};
    bool include_joint_limit{true};
    bool include_trajectory_regularization{true};
    bool force_dynamic_residual{false};
    bool force_static_residual{false};
    RfpCapabilityObjective capability_objective{RfpCapabilityObjective::Clearance};
};

RfpComparisonAlgorithm parseRfpComparisonAlgorithmOrOurs(
    const std::string& algorithm_name);

std::string rfpComparisonAlgorithmName(RfpComparisonAlgorithm algorithm);

RfpComparisonAlgorithmPolicy rfpComparisonAlgorithmPolicy(
    RfpComparisonAlgorithm algorithm);

bool rfpComparisonShouldUseDynamicResidualForcePolytope(
    RfpComparisonAlgorithm algorithm,
    bool configured_dynamic_residual);

}  // namespace polytope_wx
