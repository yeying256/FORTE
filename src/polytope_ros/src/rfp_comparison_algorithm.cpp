#include "polytope_ros/rfp_comparison_algorithm.h"

#include <algorithm>
#include <cctype>

namespace polytope_wx
{
namespace
{

std::string normalizeAlgorithmName(std::string value)
{
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });

    for (char& ch : value)
    {
        if (ch == '-' || ch == ' ')
        {
            ch = '_';
        }
    }
    return value;
}

}  // namespace

RfpComparisonAlgorithm parseRfpComparisonAlgorithmOrOurs(
    const std::string& algorithm_name)
{
    const std::string normalized = normalizeAlgorithmName(algorithm_name);
    if (normalized == "ours" || normalized == "our_method")
    {
        return RfpComparisonAlgorithm::Ours;
    }
    if (normalized == "manipulability" ||
        normalized == "manipulability_only" ||
        normalized == "gc")
    {
        return RfpComparisonAlgorithm::ManipulabilityOnly;
    }
    if (normalized == "static_force_polytope" ||
        normalized == "force_polytope" ||
        normalized == "gd")
    {
        return RfpComparisonAlgorithm::StaticForcePolytope;
    }
    if (normalized == "residual_force_polytope" ||
        normalized == "dynamic_residual_force_polytope" ||
        normalized == "rfp" ||
        normalized == "ge")
    {
        return RfpComparisonAlgorithm::ResidualForcePolytope;
    }
    if (normalized == "cone_intersection_volume" ||
        normalized == "residual_cone_intersection_volume" ||
        normalized == "g_f" ||
        normalized == "gf")
    {
        return RfpComparisonAlgorithm::ConeIntersectionVolume;
    }
    if (normalized == "joint_quintic_interpolation" ||
        normalized == "quintic_joint_interpolation" ||
        normalized == "direct_joint_interpolation" ||
        normalized == "no_optimization" ||
        normalized == "no_optimizer")
    {
        return RfpComparisonAlgorithm::JointQuinticInterpolation;
    }
    return RfpComparisonAlgorithm::Ours;
}

std::string rfpComparisonAlgorithmName(RfpComparisonAlgorithm algorithm)
{
    switch (algorithm)
    {
        case RfpComparisonAlgorithm::ManipulabilityOnly:
            return "manipulability_only";
        case RfpComparisonAlgorithm::StaticForcePolytope:
            return "static_force_polytope";
        case RfpComparisonAlgorithm::ResidualForcePolytope:
            return "residual_force_polytope";
        case RfpComparisonAlgorithm::ConeIntersectionVolume:
            return "cone_intersection_volume";
        case RfpComparisonAlgorithm::JointQuinticInterpolation:
            return "joint_quintic_interpolation";
        case RfpComparisonAlgorithm::Ours:
        default:
            return "ours";
    }
}

RfpComparisonAlgorithmPolicy rfpComparisonAlgorithmPolicy(
    RfpComparisonAlgorithm algorithm)
{
    RfpComparisonAlgorithmPolicy policy;
    switch (algorithm)
    {
        case RfpComparisonAlgorithm::ManipulabilityOnly:
            policy.include_capability = false;
            policy.include_manipulability = true;
            policy.include_joint_limit = true;
            policy.include_trajectory_regularization = false;
            policy.force_static_residual = true;
            break;
        case RfpComparisonAlgorithm::StaticForcePolytope:
            policy.include_capability = true;
            policy.include_manipulability = false;
            policy.include_joint_limit = true;
            policy.include_trajectory_regularization = true;
            policy.force_static_residual = true;
            policy.capability_objective = RfpCapabilityObjective::CenteredBallRadius;
            break;
        case RfpComparisonAlgorithm::ResidualForcePolytope:
            policy.include_capability = true;
            policy.include_manipulability = false;
            policy.include_joint_limit = true;
            policy.include_trajectory_regularization = true;
            policy.force_dynamic_residual = true;
            policy.capability_objective = RfpCapabilityObjective::CenteredBallRadius;
            break;
        case RfpComparisonAlgorithm::ConeIntersectionVolume:
            policy.include_capability = true;
            policy.include_manipulability = false;
            policy.include_joint_limit = true;
            policy.include_trajectory_regularization = true;
            policy.force_dynamic_residual = true;
            policy.capability_objective = RfpCapabilityObjective::ConeIntersectionVolume;
            break;
        case RfpComparisonAlgorithm::JointQuinticInterpolation:
            policy.include_capability = false;
            policy.include_manipulability = false;
            policy.include_joint_limit = false;
            policy.include_trajectory_regularization = false;
            policy.force_static_residual = true;
            break;
        case RfpComparisonAlgorithm::Ours:
        default:
            break;
    }
    return policy;
}

bool rfpComparisonShouldUseDynamicResidualForcePolytope(
    RfpComparisonAlgorithm algorithm,
    bool configured_dynamic_residual)
{
    const RfpComparisonAlgorithmPolicy policy =
        rfpComparisonAlgorithmPolicy(algorithm);
    if (policy.force_dynamic_residual)
    {
        return true;
    }
    if (policy.force_static_residual)
    {
        return false;
    }
    return configured_dynamic_residual;
}

}  // namespace polytope_wx
