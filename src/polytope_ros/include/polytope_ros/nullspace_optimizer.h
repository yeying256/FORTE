#pragma once

#include "polytope_ros/polytope.h"

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>

namespace polytope_wx
{

/**
 * @brief 零空间优化器的配置参数
 *
 * 这组参数控制：
 * 1. 机器人模型和末端 frame 的加载
 * 2. 每个轨迹点的迭代次数、步长和容差
 * 3. polytope 能力项、可操作度项、关节限位项、平滑项和名义轨迹偏离项的权重
 */
struct NullspaceOptimizationConfig
{
    /// 机器人 URDF 路径
    std::string urdf_path;
    /// 末端执行器 frame 名称
    std::string ee_frame;

    /// 关节力矩上限；如果不填，则默认使用 URDF/model 中的 effort limit
    Eigen::VectorXd torque_limits;
    /// 需要在 reduced model 中锁定的关节名称，例如夹爪手指
    std::vector<std::string> locked_joint_names;

    /// 每个轨迹点最多做多少次局部优化迭代
    int max_iterations_per_waypoint{40};
    /// 任务空间纠偏步长
    double pose_gain{0.8};
    /// 零空间目标下降步长
    double nullspace_step_size{0.05};
    /// 单次关节更新的最大范数，防止步长过大
    double max_joint_update_norm{0.15};
    /// 数值梯度计算时使用的扰动步长
    double finite_difference_step{1e-4};
    /// 末端位姿误差收敛阈值
    double pose_tolerance{1e-4};

    /// polytope 力能力项的权重
    double capability_weight{1.0};
    /// 可操作度代价权重；越大越偏向高 manipulability 位形
    double manipulability_weight{0.0};
    /// 关节限位代价权重
    double joint_limit_weight{1e-3};
    /// 轨迹平滑项权重
    double smoothness_weight{1e-2};
    /// 偏离名义关节轨迹的代价权重
    double nominal_weight{1e-2};
    /// 关节限位安全边界比例
    double joint_limit_margin_ratio{0.05};

    /// 力能力代价的安全比例阈值
    double capability_alpha{0.4};
    /// 接近能力边界时的二次惩罚权重
    double capability_near_weight{50.0};
    /// 超出能力边界时的二次惩罚权重
    double capability_over_weight{500.0};
    /// 超出能力边界时的四次惩罚权重
    double capability_over4_weight{5000.0};
    /// manipulability Gram 矩阵的数值正则项
    double manipulability_regularization{1e-6};
    /// manipulability 数值下界，避免 log(0)
    double manipulability_epsilon{1e-8};

    /// 是否把姿态也作为严格任务约束；false 时只约束位置
    bool constrain_orientation{true};
    /// 是否打印每个轨迹点的调试信息
    bool verbose{false};
};

/**
 * @brief 轨迹优化器的输入
 *
 * 三个序列长度必须一致：
 * - target_poses: 已知末端轨迹
 * - desired_forces: 每个轨迹点对应的期望受力方向/大小
 * - nominal_joint_trajectory: 初始参考关节轨迹
 */
struct TrajectoryOptimizationInput
{
    /// 末端目标位姿轨迹
    std::vector<Eigen::Affine3d> target_poses;
    /// 对应的期望受力向量轨迹
    std::vector<Eigen::Vector3d> desired_forces;
    /// 名义关节轨迹，可理解为零空间优化前的参考解
    std::vector<Eigen::VectorXd> nominal_joint_trajectory;
};

/**
 * @brief 单个轨迹点优化后的诊断信息
 */
struct WaypointOptimizationMetrics
{
    /// 总目标函数值
    double objective_value{0.0};
    /// 沿目标受力方向可输出的最大力能力
    double force_capacity{0.0};
    /// 当前任务真正要求的力大小
    double required_force{0.0};
    /// polytope 力能力代价
    double capability_cost{0.0};
    /// 当前位形的 manipulability 数值
    double manipulability_measure{0.0};
    /// manipulability 代价
    double manipulability_cost{0.0};
    /// 关节限位代价
    double joint_limit_cost{0.0};
    /// 平滑项代价
    double smoothness_cost{0.0};
    /// 偏离名义轨迹的代价
    double nominal_cost{0.0};
    /// 最终末端误差范数
    double pose_error_norm{0.0};
    /// 该轨迹点实际用了多少次迭代
    int iterations{0};
};

/**
 * @brief 整条轨迹优化的输出结果
 */
struct TrajectoryOptimizationResult
{
    /// 是否整体优化成功
    bool success{false};
    /// 简要结果描述
    std::string message;
    /// 优化后的关节轨迹
    std::vector<Eigen::VectorXd> optimized_joint_trajectory;
    /// 每个轨迹点的诊断指标
    std::vector<WaypointOptimizationMetrics> waypoint_metrics;
};

/**
 * @brief 单个轨迹点在线优化的输出结果
 */
struct SingleWaypointOptimizationResult
{
    /// 是否成功收敛到可接受解
    bool success{false};
    /// 简要结果描述
    std::string message;
    /// 优化后的关节解
    Eigen::VectorXd optimized_q;
    /// 单点优化诊断指标
    WaypointOptimizationMetrics metrics;
};

/**
 * @brief 在线零空间梯度步进的输出结果
 *
 * 不做单点全局/局部位姿优化，只在当前关节位形上：
 * 1. 评估与 polytope 相关的目标函数
 * 2. 数值计算目标函数梯度
 * 3. 返回可直接用于 nullspace torque shaping 的目标函数梯度
 */
struct NullspaceGradientStepResult
{
    /// 是否成功计算出有限的梯度步进
    bool success{false};
    /// 简要结果描述
    std::string message;
    /// 原始目标函数梯度
    Eigen::VectorXd objective_gradient;
    /// 预留字段；当前不再在优化器内部做额外零空间投影
    Eigen::VectorXd projected_gradient;
    /// 预留字段；当前不再在优化器内部生成关节步进
    Eigen::VectorXd delta_q;
    /// 预留字段；当前保持为输入关节位形
    Eigen::VectorXd target_q;
    /// 步进后对应的代价指标
    WaypointOptimizationMetrics metrics;
};

/**
 * @brief 基于 polytope 指标的零空间轨迹优化器
 *
 * 工作流程：
 * 1. 用 Jacobian 伪逆保证末端轨迹尽量跟踪目标
 * 2. 在零空间内用 polytope 相关目标做梯度下降
 * 3. 同时兼顾关节限位、轨迹平滑和名义轨迹偏离
 */
class PolytopeNullspaceOptimizer
{
public:
    /// 默认构造；真正初始化在 initialize() 里完成
    PolytopeNullspaceOptimizer();
    /// 默认析构
    ~PolytopeNullspaceOptimizer();

    /// 初始化优化器：加载 URDF、解析 frame、准备关节限制和力矩限制
    bool initialize(const NullspaceOptimizationConfig& config);
    /// 查询当前对象是否已经成功初始化
    bool isInitialized() const;

    /// 获取当前使用的配置
    const NullspaceOptimizationConfig& config() const;
    /// 只读访问内部 Pinocchio 模型
    const pinocchio::Model& model() const;
    /// 生成一组通用的默认关节位形：优先取 0，超限时取上下限中点
    Eigen::VectorXd defaultConfiguration() const;
    /// 计算给定关节位形下的平移力 polytope 顶点
    Eigen::MatrixXd computeTranslationalForcePolytope(
        const Eigen::VectorXd& q,
        bool gravity_adjusted = true) const;

    /// 对整条已知末端轨迹做 polytope 零空间优化
    TrajectoryOptimizationResult optimize(
        const TrajectoryOptimizationInput& input) const;
    /// 对单个末端目标位姿在线做一次 polytope 零空间优化
    SingleWaypointOptimizationResult optimizeSingleWaypoint(
        const Eigen::VectorXd& seed_q,
        const Eigen::Affine3d& target_pose,
        const Eigen::Vector3d& desired_force,
        const Eigen::VectorXd& nominal_q,
        const Eigen::VectorXd* previous_q = nullptr) const;
    /// 直接在当前关节位形上计算一次数值目标函数梯度
    NullspaceGradientStepResult computeNullspaceGradientStep(
        const Eigen::VectorXd& current_q,
        const Eigen::Vector3d& desired_force,
        const Eigen::VectorXd& nominal_q,
        const Eigen::VectorXd* previous_q = nullptr) const;

private:
    /// 内部使用的代价拆分，便于分别统计每一项
    struct CostBreakdown
    {
        double objective_value{0.0};
        double force_capacity{0.0};
        double required_force{0.0};
        double capability_cost{0.0};
        double manipulability_measure{0.0};
        double manipulability_cost{0.0};
        double joint_limit_cost{0.0};
        double smoothness_cost{0.0};
        double nominal_cost{0.0};
    };

    /// 计算给定关节位形下的末端正向运动学
    Eigen::Affine3d computeForwardKinematics(const Eigen::VectorXd& q) const;
    /// 计算任务 Jacobian；可选 6 维位姿任务或 3 维位置任务
    Eigen::MatrixXd computeTaskJacobian(const Eigen::VectorXd& q) const;
    /// 计算当前末端到目标末端之间的 SE(3) 误差
    Eigen::Matrix<double, 6, 1> computePoseError(
        const Eigen::VectorXd& q,
        const Eigen::Affine3d& target_pose) const;
    /// 计算考虑重力补偿后的剩余力矩上下界
    Eigen::VectorXd buildGravityAdjustedTorqueLimits(const Eigen::VectorXd& q) const;
    /// 把关节位置裁剪到合法范围内
    Eigen::VectorXd clampToJointLimits(const Eigen::VectorXd& q) const;
    /// 计算当前关节位形下的平移 manipulability 指标
    double computeManipulability(const Eigen::VectorXd& q) const;

    /// 评估某个关节解对应的总代价及其分项
    CostBreakdown evaluateCost(
        const Eigen::VectorXd& q,
        const Eigen::Vector3d& desired_force,
        const Eigen::VectorXd& nominal_q,
        const Eigen::VectorXd* previous_q) const;
    /// 用中心差分数值计算总代价对关节的梯度
    Eigen::VectorXd computeObjectiveGradient(
        const Eigen::VectorXd& q,
        const Eigen::Vector3d& desired_force,
        const Eigen::VectorXd& nominal_q,
        const Eigen::VectorXd* previous_q) const;
    /// 优化单个轨迹点：任务空间纠偏 + 零空间 polytope 优化
    WaypointOptimizationMetrics optimizeWaypoint(
        const Eigen::VectorXd& seed_q,
        const Eigen::Affine3d& target_pose,
        const Eigen::Vector3d& desired_force,
        const Eigen::VectorXd& nominal_q,
        const Eigen::VectorXd* previous_q,
        Eigen::VectorXd& optimized_q) const;

    /// Pinocchio 模型
    pinocchio::Model model_;
    /// Pinocchio 数据缓存；声明成 mutable 以便 const 接口内部更新运动学
    mutable std::unique_ptr<pinocchio::Data> data_;
    /// 末端 frame 的索引
    pinocchio::FrameIndex ee_frame_id_{0};

    /// 当前配置副本
    NullspaceOptimizationConfig config_;
    /// 最终实际使用的力矩限制
    Eigen::VectorXd torque_limits_;
    /// 关节下限
    Eigen::VectorXd lower_position_limits_;
    /// 关节上限
    Eigen::VectorXd upper_position_limits_;
    /// 关节安全边界
    Eigen::VectorXd joint_limit_margins_;

    /// polytope 相关工具类
    polytope polytope_tool_;
    /// 初始化状态标记
    bool initialized_{false};
};

} // namespace polytope_wx
