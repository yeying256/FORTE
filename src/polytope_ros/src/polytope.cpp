#include "polytope_ros/polytope.h"

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <limits>

namespace polytope_wx
{

polytope::polytope()
    : data_(model_)
{
}

polytope::polytope(const std::string& urdf_path)
    : data_(model_)
{
    load_urdf(urdf_path);
}

polytope::polytope(const Eigen::VectorXd& lim_tor)
    : data_(model_),
      lim_tor_(lim_tor)
{
}

polytope::polytope(const std::string& urdf_path, const Eigen::VectorXd& lim_tor)
    : data_(model_),
      lim_tor_(lim_tor)
{
    load_urdf(urdf_path);
}

void polytope::print_full_model_info() const
{
    std::cout << "\n================ Joint Info ================\n";

    for (size_t i = 0; i < model_.joints.size(); ++i)
    {
        std::cout << "Joint ID: " << i
                  << " | Name: " << model_.names[i]
                  << " | Type: " << model_.joints[i].shortname()
                  << std::endl;
    }

    std::cout << "\n================ Frame Info ================\n";

    for (size_t i = 0; i < model_.frames.size(); ++i)
    {
        const auto &frame = model_.frames[i];

        std::cout << "Frame ID: " << i
                  << " | Name: " << frame.name
                  << " | Parent Joint: " << frame.parent
                  << std::endl;
    }

    std::cout << "\n================ Frame -> Joint Mapping ================\n";

    for (size_t i = 0; i < model_.frames.size(); ++i)
    {
        const auto &frame = model_.frames[i];

        std::cout << frame.name
                  << " -> joint: "
                  << model_.names[frame.parent]
                  << std::endl;
    }

    std::cout << "\n================ Limits ================\n";

    std::cout << "Torque limits:\n"
              << model_.effortLimit.transpose() << std::endl;

    std::cout << "Velocity limits:\n"
              << model_.velocityLimit.transpose() << std::endl;

    std::cout << "Position lower:\n"
              << model_.lowerPositionLimit.transpose() << std::endl;

    std::cout << "Position upper:\n"
              << model_.upperPositionLimit.transpose() << std::endl;

    std::cout << "=========================================================\n";
}
void polytope::print_model_info() const
{
    std::cout << "================ Robot Model Info ================" << std::endl;

    std::cout << "nq (configuration dim): " << model_.nq << std::endl;
    std::cout << "nv (velocity dim): " << model_.nv << std::endl;
    std::cout << "njoints: " << model_.njoints << std::endl;
    std::cout << "nframes: " << model_.nframes << std::endl;

    std::cout << "\n========== Joint Names ==========" << std::endl;
    for (size_t i = 0; i < model_.names.size(); ++i)
    {
        std::cout << i << " : " << model_.names[i] << std::endl;
    }

    std::cout << "\n========== Frame Names ==========" << std::endl;
    for (size_t i = 0; i < model_.frames.size(); ++i)
    {
        std::cout << i << " : " << model_.frames[i].name << std::endl;
    }

    std::cout << "\n========== Torque Limits ==========" << std::endl;
    std::cout << model_.effortLimit.transpose() << std::endl;

    std::cout << "\n========== Velocity Limits ==========" << std::endl;
    std::cout << model_.velocityLimit.transpose() << std::endl;

    std::cout << "\n========== Position Limits ==========" << std::endl;
    std::cout << "lower: " << model_.lowerPositionLimit.transpose() << std::endl;
    std::cout << "upper: " << model_.upperPositionLimit.transpose() << std::endl;

    std::cout << "==================================================" << std::endl;
}

polytope::~polytope()
{
}

void polytope::load_urdf(const std::string& urdf_path)
{
    pinocchio::urdf::buildModel(urdf_path, model_);
    data_ = pinocchio::Data(model_);
    // lim_tor_ = model_.effortLimit;

    // print_model_info();
    print_full_model_info();
}

void polytope::set_torque_limit(const Eigen::VectorXd& lim_tor)
{
    lim_tor_ = lim_tor;
}

void polytope::generate_combinations(
    int n,
    int k,
    int start,
    std::vector<int>& current,
    std::vector<std::vector<int>>& result) const
{
    if ((int)current.size() == k)
    {
        result.push_back(current);
        return;
    }

    for (int i = start; i < n; ++i)
    {
        current.push_back(i);
        generate_combinations(n, k, i + 1, current, result);
        current.pop_back();
    }
}

std::map<std::vector<int>, Eigen::MatrixXd>
polytope::remove_arbitrary_columns(const Eigen::MatrixXd& matrix, int m) const
{
    int n_cols = matrix.cols();

    if (m >= n_cols)
    {
        throw std::runtime_error("remove_arbitrary_columns: m must be smaller than number of columns");
    }

    std::vector<std::vector<int>> cols_combinations;
    std::vector<int> current;
    generate_combinations(n_cols, m, 0, current, cols_combinations);

    std::map<std::vector<int>, Eigen::MatrixXd> results;

    for (const auto& cols_to_remove : cols_combinations)
    {
        std::vector<int> keep_indices;
        for (int i = 0; i < n_cols; ++i)
        {
            if (std::find(cols_to_remove.begin(), cols_to_remove.end(), i) == cols_to_remove.end())
            {
                keep_indices.push_back(i);
            }
        }

        Eigen::MatrixXd result_matrix(matrix.rows(), keep_indices.size());
        for (int j = 0; j < (int)keep_indices.size(); ++j)
        {
            result_matrix.col(j) = matrix.col(keep_indices[j]);
        }

        results[cols_to_remove] = result_matrix;
    }

    return results;
}

bool polytope::row_is_finite(const Eigen::RowVectorXd& row) const
{
    for (int i = 0; i < row.size(); ++i)
    {
        if (!std::isfinite(row(i)))
        {
            return false;
        }
    }
    return true;
}

Eigen::MatrixXd polytope::unique_rows(
    const Eigen::MatrixXd& mat,
    double tol) const
{
    if (mat.rows() == 0)
    {
        return mat;
    }

    std::vector<Eigen::RowVectorXd> rows;

    for (int i = 0; i < mat.rows(); ++i)
    {
        Eigen::RowVectorXd r = mat.row(i);

        if (!row_is_finite(r))
        {
            continue;
        }

        bool duplicated = false;
        for (const auto& rr : rows)
        {
            if ((r - rr).norm() < tol)
            {
                duplicated = true;
                break;
            }
        }

        if (!duplicated)
        {
            rows.push_back(r);
        }
    }

    Eigen::MatrixXd out(rows.size(), mat.cols());
    for (int i = 0; i < (int)rows.size(); ++i)
    {
        out.row(i) = rows[i];
    }

    return out;
}

Eigen::MatrixXd polytope::compute_polytope_vertices(
    const Eigen::MatrixXd& J,
    const Eigen::VectorXd* tor_lim) const
{
    if (lim_tor_.size() == 0 && tor_lim == nullptr)
    {
        throw std::runtime_error("compute_polytope_vertices: torque limit is empty");
    }

    const int n = J.cols();
    const int m = J.rows();

    Eigen::VectorXd b;
    if (tor_lim == nullptr)
    {
        Eigen::VectorXd tau_residual = lim_tor_;
        for (int i = 0; i < tau_residual.size(); ++i)
        {
            tau_residual(i) = std::max(tau_residual(i), 0.1);
        }

        b.resize(2 * n);
        b.head(n) = tau_residual;
        b.tail(n) = -tau_residual;
    }
    else
    {
        b = *tor_lim;
    }

    Eigen::MatrixXd JT = J.transpose();

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2 * n, 2 * n + m);

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        J,
        Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::MatrixXd V = svd.matrixV();
    Eigen::MatrixXd V1 = V.leftCols(m);

    A.block(0, 0, n, m) = V1;
    A.block(n, 0, n, m) = V1;

    A.block(0, m, 2 * n, 2 * n) = Eigen::MatrixXd::Identity(2 * n, 2 * n);
    A.block(0, m + n, 2 * n, n) = -A.block(0, m + n, 2 * n, n);

    Eigen::MatrixXd A_end = A.block(0, m, 2 * n, 2 * n);
    auto A_end_map = remove_arbitrary_columns(A_end, m);

    std::vector<Eigen::VectorXd> results_list;

    for (const auto& item : A_end_map)
    {
        const Eigen::MatrixXd& result_matrix = item.second;

        Eigen::MatrixXd temp(2 * n, 2 * n);
        temp << A.block(0, 0, 2 * n, m), result_matrix;

        Eigen::FullPivLU<Eigen::MatrixXd> lu(temp);
        if (lu.rank() != 2 * n)
        {
            continue;
        }

        Eigen::VectorXd x = temp.fullPivLu().solve(b);
        Eigen::VectorXd x_first_m = x.head(m);

        bool valid = true;
        Eigen::VectorXd slack = x.tail(2 * n - m);
        for (int i = 0; i < slack.size(); ++i)
        {
            if (slack(i) < -1e-9)
            {
                valid = false;
                break;
            }
        }

        if (!valid)
        {
            continue;
        }

        Eigen::VectorXd tor = V1 * x_first_m;

        Eigen::MatrixXd JTinv = JT.completeOrthogonalDecomposition().pseudoInverse();
        Eigen::VectorXd F_car = JTinv * tor;

        results_list.push_back(F_car);
    }

    if (results_list.empty())
    {
        std::cout << "result list is empty" << std::endl;
        return Eigen::MatrixXd(0, m);
    }

    Eigen::MatrixXd points(results_list.size(), results_list[0].size());
    for (int i = 0; i < (int)results_list.size(); ++i)
    {
        points.row(i) = results_list[i].transpose();
    }

    points = unique_rows(points);

    return points;
}

Eigen::MatrixXd polytope::compute_polytope_vertices_g(
    const Eigen::VectorXd& gravity_vector,
    const Eigen::MatrixXd& J,
    const Eigen::VectorXd* tor_lim) const
{
    const int n = J.cols();

    Eigen::VectorXd adjusted_tor_lim;
    if (tor_lim == nullptr)
    {
        adjusted_tor_lim.resize(2 * n);
        adjusted_tor_lim.head(n) = lim_tor_ - gravity_vector;
        adjusted_tor_lim.tail(n) = -lim_tor_ - gravity_vector;
    }
    else
    {
        adjusted_tor_lim = *tor_lim;
    }

    return compute_polytope_vertices(J, &adjusted_tor_lim);
}

double polytope::compute_force_capacity(
    const Eigen::MatrixXd& J,
    const Eigen::VectorXd& force,
    const Eigen::VectorXd& tau_lim) const
{
    if (J.cols() <= 0 || J.rows() <= 0)
    {
        throw std::runtime_error("compute_force_capacity: invalid Jacobian size.");
    }

    if (force.size() != J.rows())
    {
        throw std::runtime_error("compute_force_capacity: force dimension mismatch.");
    }

    const int n = J.cols();
    if (tau_lim.size() != 2 * n)
    {
        throw std::runtime_error("compute_force_capacity: tau_lim dimension mismatch.");
    }

    const Eigen::VectorXd v_tau = J.transpose() * force;
    const double magnitude = v_tau.norm();
    if (magnitude < 1e-9)
    {
        return 0.0;
    }

    const Eigen::VectorXd v_tau_n = v_tau / magnitude;

    Eigen::MatrixXd block_matrix = Eigen::MatrixXd::Zero(2 * n, 2 * n);
    block_matrix.topLeftCorner(n, n) = v_tau_n.asDiagonal();
    block_matrix.bottomRightCorner(n, n) = v_tau_n.asDiagonal();

    const Eigen::VectorXd d_tau =
        (block_matrix + 1e-8 * Eigen::MatrixXd::Identity(2 * n, 2 * n)).fullPivLu().solve(tau_lim);

    double d_tau_min = std::numeric_limits<double>::infinity();
    for (int i = 0; i < d_tau.size(); ++i)
    {
        if (d_tau(i) > 0.0)
        {
            d_tau_min = std::min(d_tau_min, d_tau(i));
        }
    }

    if (!std::isfinite(d_tau_min))
    {
        return 0.0;
    }

    const Eigen::MatrixXd Jt_pinv =
        J.transpose().completeOrthogonalDecomposition().pseudoInverse();
    const Eigen::VectorXd opt_force_max = d_tau_min * Jt_pinv * v_tau_n;

    if (!opt_force_max.allFinite())
    {
        return 0.0;
    }

    return opt_force_max.norm();
}

double polytope::compute_force_ball_signed_clearance(
    const Eigen::MatrixXd& J,
    const Eigen::VectorXd& force_center,
    double force_radius,
    const Eigen::VectorXd& tau_lim,
    double alpha,
    double eps) const
{
    if (J.cols() <= 0 || J.rows() <= 0)
    {
        throw std::runtime_error(
            "compute_force_ball_signed_clearance: invalid Jacobian size.");
    }

    if (force_center.size() != J.rows())
    {
        throw std::runtime_error(
            "compute_force_ball_signed_clearance: force dimension mismatch.");
    }

    const int n = J.cols();
    if (tau_lim.size() != 2 * n)
    {
        throw std::runtime_error(
            "compute_force_ball_signed_clearance: tau_lim dimension mismatch.");
    }

    const double clamped_radius = std::max(0.0, force_radius);
    double min_clearance = std::numeric_limits<double>::infinity();

    for (int i = 0; i < n; ++i)
    {
        const Eigen::VectorXd facet_normal = J.col(i);
        const double facet_normal_norm = facet_normal.norm();
        if (facet_normal_norm <= eps)
        {
            continue;
        }

        const double scaled_upper_bound = alpha * tau_lim(i);
        const double scaled_lower_bound = alpha * (-tau_lim(n + i));
        const double projection = facet_normal.dot(force_center);

        const double upper_clearance =
            (scaled_upper_bound - projection) / facet_normal_norm - clamped_radius;
        const double lower_clearance =
            (scaled_lower_bound + projection) / facet_normal_norm - clamped_radius;

        min_clearance = std::min(min_clearance, upper_clearance);
        min_clearance = std::min(min_clearance, lower_clearance);
    }

    if (!std::isfinite(min_clearance))
    {
        return 0.0;
    }

    return min_clearance;
}

double polytope::compute_clearance_based_capability_cost(
    double signed_clearance,
    double normalization_scale,
    double eps) const
{
    constexpr double kCapabilityExponentScale = 3.0;

    const double safe_scale = std::max(std::abs(normalization_scale), eps);
    const double normalized_violation =
        std::max(0.0, -signed_clearance / safe_scale);
    return std::expm1(kCapabilityExponentScale * normalized_violation);
}

double polytope::compute_residual_force_capacity_cost(
    double d_car_max,
    double f_d,
    double alpha,
    double w_near,
    double w_over,
    double w_over4,
    double eps) const
{
    constexpr double kCapabilityExponentScale = 3.0;

    if (f_d <= eps)
    {
        return 0.0;
    }

    const double signed_clearance = alpha * d_car_max - f_d;
    const double normalized_clearance =
        signed_clearance / std::max(std::abs(f_d), eps);
    const double normalized_violation = std::max(0.0, -normalized_clearance);

    // Keep an explicit alpha-sized safe region where the capability cost is
    // exactly zero, then switch to an exponential penalty once the requested
    // force exceeds that safe capability margin.
    (void)w_near;
    (void)w_over;
    (void)w_over4;
    return std::expm1(kCapabilityExponentScale * normalized_violation);

    // Legacy smooth distance / quadratic / always-on exponential capability
    // costs kept here for reference:
    // return normalized_violation * normalized_violation;
    //
    // constexpr double kCapabilitySoftplusSharpness = 6.0;
    // const double scaled_violation = -normalized_clearance;
    // const double x = kCapabilitySoftplusSharpness * scaled_violation;
    // const double softplus =
    //     (std::max(x, 0.0) + std::log1p(std::exp(-std::abs(x)))) /
    //     kCapabilitySoftplusSharpness;
    // return softplus * softplus;
    //
    // const double effective_capacity = std::max(alpha * d_car_max, eps);
    // const double exponent =
    //     -(effective_capacity - f_d) / std::max(std::abs(f_d), eps);
    // return std::exp(kCapabilityExponentScale * exponent);
    //
    // const double ratio = f_d / (d_car_max + eps);
    // if (ratio <= alpha)
    // {
    //     return 0.0;
    // }
    // if (ratio <= 1.0)
    // {
    //     return w_near * (ratio - alpha) * (ratio - alpha);
    // }
    // return w_near * (1.0 - alpha) * (1.0 - alpha) +
    //        w_over * (ratio - 1.0) * (ratio - 1.0) +
    //        w_over4 * std::pow(ratio - 1.0, 4);
}

}  // namespace polytope_wx
