#pragma once

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>

#include <Eigen/Dense>

#include <map>
#include <string>
#include <vector>

namespace polytope_wx
{

/**
 * @brief Polytope 相关工具类
 *
 * 这个类主要负责两类工作：
 * 1. 根据 Jacobian 和力矩限制计算力多面体顶点
 * 2. 根据给定受力方向计算可输出力能力以及对应代价
 */
class polytope
{
public:
    /// 默认构造
    polytope();
    /// 用 URDF 路径构造
    explicit polytope(const std::string& urdf_path);
    /// 只用力矩限制构造
    explicit polytope(const Eigen::VectorXd& lim_tor);
    /// 同时用 URDF 和力矩限制构造
    polytope(const std::string& urdf_path, const Eigen::VectorXd& lim_tor);
    /// 默认析构
    ~polytope();

    /// 打印模型的基础信息
    void print_model_info() const;
    /// 打印模型的完整 joint/frame/limit 信息
    void print_full_model_info() const;
    /// 从 URDF 加载 Pinocchio 模型
    void load_urdf(const std::string& urdf_path);
    /// 设置力矩限制向量
    void set_torque_limit(const Eigen::VectorXd& lim_tor);

    /// 枚举移除矩阵中任意 m 列后的所有结果
    std::map<std::vector<int>, Eigen::MatrixXd>
    remove_arbitrary_columns(const Eigen::MatrixXd& matrix, int m) const;

    /// 根据 Jacobian 和力矩限制计算力多面体顶点
    Eigen::MatrixXd compute_polytope_vertices(
        const Eigen::MatrixXd& J,
        const Eigen::VectorXd* tor_lim = nullptr) const;

    /// 在考虑重力项后的力矩边界下计算力多面体顶点
    Eigen::MatrixXd compute_polytope_vertices_g(
        const Eigen::VectorXd& gravity_vector,
        const Eigen::MatrixXd& J,
        const Eigen::VectorXd* tor_lim = nullptr) const;

    /// 计算沿给定受力方向的最大可输出力能力
    double compute_force_capacity(
        const Eigen::MatrixXd& J,
        const Eigen::VectorXd& force,
        const Eigen::VectorXd& tau_lim) const;

    /// 计算力球到 alpha-缩放力 polytope 各边面的最小有符号距离
    double compute_force_ball_signed_clearance(
        const Eigen::MatrixXd& J,
        const Eigen::VectorXd& force_center,
        double force_radius,
        const Eigen::VectorXd& tau_lim,
        double alpha = 1.0,
        double eps = 1e-9) const;

    /// 根据有符号距离构造 capability 代价；安全区内严格为 0，越界后指数增大
    double compute_clearance_based_capability_cost(
        double signed_clearance,
        double normalization_scale,
        double eps = 1e-6) const;

    /// 根据力能力安全距离构造一个平滑代价，越接近或越穿出极限代价越大
    double compute_residual_force_capacity_cost(
        double d_car_max,
        double f_d,
        double alpha = 0.8,
        double w_near = 50.0,
        double w_over = 500.0,
        double w_over4 = 5000.0,
        double eps = 1e-6) const;

private:
    /// 内部 Pinocchio 模型
    pinocchio::Model model_;
    /// 内部 Pinocchio 数据
    pinocchio::Data data_;

    /// 默认使用的力矩限制
    Eigen::VectorXd lim_tor_;

private:
    /// 递归生成组合索引
    void generate_combinations(
        int n,
        int k,
        int start,
        std::vector<int>& current,
        std::vector<std::vector<int>>& result) const;

    /// 对矩阵按行去重
    Eigen::MatrixXd unique_rows(
        const Eigen::MatrixXd& mat,
        double tol = 1e-9) const;

    /// 检查一行数据是否全是有限值
    bool row_is_finite(const Eigen::RowVectorXd& row) const;
};

}  // namespace polytope_wx
