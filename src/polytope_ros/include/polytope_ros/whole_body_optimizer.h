#pragma once

#include "polytope_ros/polytope.h"
#include "polytope_ros/rfp_comparison_algorithm.h"

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

namespace polytope_wx
{

class WholeBodyTrajectoryObjectiveCallback;
class WholeBodyTrajectoryConstraintCallback;

struct CollisionSphereSpec
{
    std::string frame;
    Eigen::Vector3d center;
    double radius{0.0};

    CollisionSphereSpec()
        : center(Eigen::Vector3d::Zero())
    {
    }
};

/**
 * @brief Whole-body polytope 优化器的配置
 *
 * 这一版优化器把变量扩成：
 * [base_x, base_y, base_yaw, arm_q]
 *
 * 其中：
 * - 末端位姿任务通过 whole-body Jacobian 保证
 * - polytope 力能力项只作用在机械臂 7 轴上
 * - 底盘自由度主要帮助机械臂腾挪姿态、降低关节极限风险
 */
struct WholeBodyOptimizationConfig
{
    WholeBodyOptimizationConfig();

    /// 机器人 URDF 路径
    std::string urdf_path;
    /// 作为平面基座参考的 frame，默认使用 base footprint
    std::string base_frame;
    /// 末端执行器 frame 名称
    std::string ee_frame;

    /// 机械臂力矩上限；如果不填，则优先使用 URDF/model 中的 effort limit
    Eigen::VectorXd arm_torque_limits;
    /// 需要在 reduced model 中锁定的关节名称，例如夹爪手指
    std::vector<std::string> locked_joint_names;

    /// 每个轨迹点最多迭代次数
    int max_iterations_per_waypoint{40};
    /// 整条轨迹联合优化的最大外层迭代次数
    int max_trajectory_iterations{20};
    /// 任务空间纠偏步长
    double pose_gain{0.8};
    /// 零空间目标下降步长
    double nullspace_step_size{0.05};
    /// 单次状态更新的最大范数
    double max_state_update_norm{0.15};
    /// 数值梯度使用的扰动步长
    double finite_difference_step{1e-4};
    /// 末端位姿误差收敛阈值
    double pose_tolerance{1e-4};

    /// polytope 力能力项权重
    double capability_weight{1.0};
    /// manipulability 项权重（当前使用底盘 yaw + 机械臂的 whole-body 平移 manipulability）
    double manipulability_weight{0.0};
    /// manipulability 的数值稳定项
    double manipulability_epsilon{1e-8};
    double manipulability_regularization{1e-6};
    /// 机械臂关节限位项权重
    double joint_limit_weight{1e-3};
    /// 10 维逐自由度 joint-limit / base-anchor 权重
    /// [base_x, base_y, base_yaw, arm_q1 ... arm_q7]
    /// 前 3 维用于“远离名义底盘位置”的惩罚，后 7 维用于“远离关节中值”的惩罚
    Eigen::VectorXd joint_limit_dof_weights;
    /// 机械臂平滑项权重
    double smoothness_weight{1e-2};
    /// 机械臂速度项权重；按相邻 waypoint 的关节速度 dq/dt 惩罚
    double velocity_weight{0.0};
    /// 机械臂偏离名义轨迹项权重
    double nominal_weight{1e-2};
    /// 关节限位安全边界比例
    double joint_limit_margin_ratio{0.05};

    /// 底盘平滑项权重
    double base_smoothness_weight{5e-2};
    /// 底盘速度项权重；按 [vx, vy, yaw_rate] 惩罚
    double base_velocity_weight{0.0};
    /// 底盘偏离名义轨迹项权重
    double base_nominal_weight{5e-2};
    /// 底盘平滑项度量，对应 [x, y, yaw]
    Eigen::Vector3d base_smoothness_metric;
    /// 底盘名义项度量，对应 [x, y, yaw]
    Eigen::Vector3d base_nominal_metric;
    /// 底盘频谱能量项权重；设为 0 可关闭
    double base_spectral_energy_weight{0.0};
    /// 底盘频谱能量项度量，对应 [x, y, yaw]
    Eigen::Vector3d base_spectral_energy_metric;
    /// 归一化频率截止比，小于该比例的低频不惩罚
    double base_spectral_cutoff_ratio{0.35};
    /// 频谱权重随频率升高的幂次
    double base_spectral_power{2.0};

    /// 球碰撞代价权重；0 表示关闭
    double collision_weight{0.0};
    /// 球与球之间希望保留的最小安全距离
    double collision_safe_distance{0.05};
    /// 附着在末端/篮子上的球，center 表达在各自 frame 下
    std::vector<CollisionSphereSpec> collision_basket_spheres;
    /// 附着在机器人本体上的球，center 表达在各自 frame 下
    std::vector<CollisionSphereSpec> collision_body_spheres;

    /// 底盘平面状态上下界；非有限值表示不限制
    Eigen::Vector3d base_lower_limits;
    Eigen::Vector3d base_upper_limits;

    /// 力能力代价的安全比例阈值
    double capability_alpha{0.8};
    /// true 时限制 clearance exponential cost 的指数，false 时恢复原始无界 exp 行为
    bool capability_exponent_limit_enabled{true};
    /// 论文对比实验使用的目标函数模式；ours 表示使用当前完整方法
    RfpComparisonAlgorithm comparison_algorithm{RfpComparisonAlgorithm::Ours};
    /// true 时用轨迹差分速度/加速度扣除 RNEA 名义力矩，形成动态 residual force polytope
    bool use_dynamic_residual_force_polytope{false};
    /// gF cone-intersection-volume 近似使用的期望外力方向；desired_force 非零时优先使用 desired_force
    Eigen::Vector3d comparison_cone_axis_world;
    /// gF 圆锥半角，单位 rad
    double comparison_cone_half_angle_rad{0.35};
    /// gF 圆锥采样环数；越大越接近体积积分，但优化更慢
    int comparison_cone_ring_count{3};
    /// gF 每圈方位采样数
    int comparison_cone_azimuth_count{8};
    /// 接近能力边界时的二次惩罚权重
    double capability_near_weight{50.0};
    /// 超出能力边界时的二次惩罚权重
    double capability_over_weight{500.0};
    /// 超出能力边界时的四次惩罚权重
    double capability_over4_weight{5000.0};
    /// true 时保存 dynamic residual force polytope 的逐 waypoint 诊断数据
    bool dynamic_residual_diagnostics_enabled{false};
    /// 诊断 CSV 输出目录
    std::string dynamic_residual_diagnostics_directory;
    /// 每次进程最多写多少组诊断文件，避免长时间实验刷满磁盘
    int dynamic_residual_diagnostics_max_records{20};

    /// 是否把姿态也作为严格任务约束；false 时只约束位置
    bool constrain_orientation{true};
    /// 是否打印调试信息
    bool verbose{false};
};

/**
 * @brief Whole-body 轨迹优化输入
 */
struct WholeBodyTrajectoryOptimizationInput
{
    /// 末端目标位姿轨迹，表达在世界坐标系
    std::vector<Eigen::Affine3d> target_poses;
    /// 每个轨迹点的期望受力向量，表达在世界坐标系
    std::vector<Eigen::Vector3d> desired_forces;
    /// 每个轨迹点对应的期望力球半径；为空时默认按 0 处理
    std::vector<double> desired_force_radii;
    /// 名义 whole-body 轨迹，排列为 [base_x, base_y, base_yaw, arm_q]
    std::vector<Eigen::VectorXd> nominal_state_trajectory;
    /// 轨迹点时间戳，单位秒；若为空，则回退到单位时间间隔
    std::vector<double> waypoint_times_sec;
    /// 轨迹优化时固定第一个 whole-body 状态；单点构型优化应关闭
    bool pin_first_state{true};
    /// 单点构型优化只使用 force capability / manipulability / joint-limit 三类 cost
    bool single_point_cost_terms_only{false};
};

/**
 * @brief 单个轨迹点优化后的诊断信息
 */
struct WholeBodyWaypointOptimizationMetrics
{
    double objective_value{0.0};
    double force_capacity{0.0};
    double required_force{0.0};
    double required_force_radius{0.0};
    double force_ball_clearance{0.0};
    double capability_cost{0.0};
    double manipulability_measure{0.0};
    double manipulability_cost{0.0};
    double joint_limit_cost{0.0};
    double smoothness_cost{0.0};
    double velocity_cost{0.0};
    double nominal_cost{0.0};
    double base_smoothness_cost{0.0};
    double base_velocity_cost{0.0};
    double base_nominal_cost{0.0};
    double collision_cost{0.0};
    double min_collision_clearance{0.0};
    double pose_error_norm{0.0};
    int iterations{0};
};

/**
 * @brief Whole-body 轨迹优化结果
 */
struct WholeBodyTrajectoryOptimizationResult
{
    bool success{false};
    std::string message;
    int trajectory_iterations{0};
    double base_spectral_energy_cost{0.0};
    /// 优化后的 whole-body 状态轨迹，排列为 [base_x, base_y, base_yaw, arm_q]
    std::vector<Eigen::VectorXd> optimized_state_trajectory;
    std::vector<WholeBodyWaypointOptimizationMetrics> waypoint_metrics;
};

/**
 * @brief 基于 polytope 的 whole-body 零空间轨迹优化器
 *
 * 这版优化器不把底盘当作力矩 polytope 的参与者，而是：
 * - 用底盘 3 自由度帮助末端位姿可达性与姿态腾挪
 * - 用机械臂 7 轴承担真正的力能力优化
 */
class WholeBodyPolytopeOptimizer
{
public:
    WholeBodyPolytopeOptimizer();
    ~WholeBodyPolytopeOptimizer();

    bool initialize(const WholeBodyOptimizationConfig& config);
    bool isInitialized() const;

    const WholeBodyOptimizationConfig& config() const;
    const pinocchio::Model& armModel() const;

    int armDof() const;
    int stateDim() const;

    /// 默认状态：底盘置零，机械臂优先取 0，超限时取中点
    Eigen::VectorXd defaultState() const;
    /// 计算某个 whole-body 状态下的末端位姿
    Eigen::Affine3d computeEndEffectorPose(const Eigen::VectorXd& state) const;
    /// 计算某个 whole-body 状态对应的单点 cost / force / manipulability 指标
    WholeBodyWaypointOptimizationMetrics evaluateWaypointMetrics(
        const Eigen::VectorXd& state,
        const Eigen::Affine3d& target_pose,
        const Eigen::Vector3d& desired_force,
        double desired_force_radius,
        const Eigen::VectorXd& nominal_state,
        const Eigen::VectorXd* previous_state,
        double previous_dt_sec = 1.0) const;
    /// 计算一整条 whole-body 轨迹的底盘频谱能量 cost
    double computeTrajectoryBaseSpectralEnergyCost(
        const std::vector<Eigen::VectorXd>& trajectory,
        const std::vector<double>& waypoint_times_sec) const;
    /// 计算一组碰撞球球心在世界坐标系下的位置；用于 RViz 可视化和调试
    std::vector<Eigen::Vector3d> computeCollisionSphereCentersWorld(
        const Eigen::VectorXd& state,
        const std::vector<CollisionSphereSpec>& spheres) const;

    /// 计算某个 whole-body 状态下，仅由机械臂 7 轴产生的平移力 polytope
    Eigen::MatrixXd computeArmTranslationalForcePolytope(
        const Eigen::VectorXd& state,
        bool gravity_adjusted = true) const;

    WholeBodyTrajectoryOptimizationResult optimize(
        const WholeBodyTrajectoryOptimizationInput& input) const;

private:
    friend class WholeBodyTrajectoryObjectiveCallback;
    friend class WholeBodyTrajectoryConstraintCallback;

    struct CostBreakdown
    {
        double objective_value{0.0};
        double force_capacity{0.0};
        double required_force{0.0};
        double required_force_radius{0.0};
        double force_ball_clearance{0.0};
        double capability_cost{0.0};
        double manipulability_measure{0.0};
        double manipulability_cost{0.0};
        double joint_limit_cost{0.0};
        double smoothness_cost{0.0};
        double velocity_cost{0.0};
        double nominal_cost{0.0};
        double base_smoothness_cost{0.0};
        double base_velocity_cost{0.0};
        double base_nominal_cost{0.0};
        double collision_cost{0.0};
        double min_collision_clearance{0.0};
    };

    struct ResolvedCollisionSphere
    {
        std::string frame;
        pinocchio::FrameIndex frame_id{0};
        Eigen::Vector3d center{Eigen::Vector3d::Zero()};
        double radius{0.0};
    };

    Eigen::Vector3d baseState(const Eigen::VectorXd& state) const;
    Eigen::VectorXd armState(const Eigen::VectorXd& state) const;
    Eigen::Vector3d baseDifference(
        const Eigen::Vector3d& lhs,
        const Eigen::Vector3d& rhs) const;

    double wrapAngle(double angle) const;
    /**
     * @brief 使各个状态限制在有效范围内，例如底盘位置限制、机械臂关节限制等
     * 
     * @param state 
     * @return Eigen::VectorXd 
     */
    Eigen::VectorXd clampToStateLimits(const Eigen::VectorXd& state) const;
    Eigen::VectorXd integrateState(
        const Eigen::VectorXd& state,
        const Eigen::VectorXd& delta) const;

    Eigen::Affine3d computeWorldToBase(const Eigen::VectorXd& state) const;
    Eigen::Affine3d computeBaseToEe(const Eigen::VectorXd& state) const;
    Eigen::Affine3d computeForwardKinematics(const Eigen::VectorXd& state) const;
    std::string resolveCollisionFrameName(const std::string& frame) const;
    std::vector<ResolvedCollisionSphere> resolveCollisionSpheres(
        const std::vector<CollisionSphereSpec>& specs,
        const std::string& label) const;
    Eigen::Vector3d computeCollisionSphereCenterWorld(
        const Eigen::VectorXd& state,
        const ResolvedCollisionSphere& sphere) const;
    double evaluateCollisionCost(
        const Eigen::VectorXd& state,
        double* min_clearance) const;

    Eigen::Matrix<double, 6, 3> computeBaseJacobianWorld(
        const Eigen::Affine3d& w_T_base,
        const Eigen::Affine3d& w_T_ee) const;
    Eigen::MatrixXd computeArmJacobianWorld(const Eigen::VectorXd& state) const;
    double computeArmManipulability(const Eigen::VectorXd& state) const;
    Eigen::MatrixXd computeTaskJacobian(const Eigen::VectorXd& state) const;
    Eigen::Matrix<double, 6, 1> computePoseError(
        const Eigen::VectorXd& state,
        const Eigen::Affine3d& target_pose) const;

    Eigen::VectorXd buildGravityAdjustedTorqueLimits(const Eigen::VectorXd& state) const;
    Eigen::VectorXd buildDynamicResidualTorqueLimits(
        const Eigen::VectorXd& state,
        const Eigen::VectorXd* previous_state,
        const Eigen::VectorXd* previous_previous_state,
        double previous_dt_sec,
        double previous_previous_dt_sec) const;

    CostBreakdown evaluateCost(
        const Eigen::VectorXd& state,
        const Eigen::Vector3d& desired_force,
        double desired_force_radius,
        const Eigen::VectorXd& nominal_state,
        const Eigen::VectorXd* previous_state,
        double previous_dt_sec = 1.0,
        const Eigen::VectorXd* previous_previous_state = nullptr,
        double previous_previous_dt_sec = 1.0) const;
    Eigen::VectorXd computeObjectiveGradient(
        const Eigen::VectorXd& state,
        const Eigen::Vector3d& desired_force,
        double desired_force_radius,
        const Eigen::VectorXd& nominal_state,
        const Eigen::VectorXd* previous_state) const;
    double computeBaseSpectralEnergyCost(
        const std::vector<Eigen::VectorXd>& trajectory,
        const std::vector<double>& waypoint_times_sec) const;
    void writeDynamicResidualDiagnostics(
        const WholeBodyTrajectoryOptimizationInput& input,
        const std::vector<Eigen::VectorXd>& trajectory,
        const std::string& label) const;
    double evaluateTrajectoryObjective(
        const std::vector<Eigen::VectorXd>& trajectory,
        const WholeBodyTrajectoryOptimizationInput& input) const;
    std::vector<Eigen::VectorXd> computeTrajectoryObjectiveGradient(
        const std::vector<Eigen::VectorXd>& trajectory,
        const WholeBodyTrajectoryOptimizationInput& input) const;
    WholeBodyWaypointOptimizationMetrics optimizeWaypoint(
        const Eigen::VectorXd& seed_state,
        const Eigen::Affine3d& target_pose,
        const Eigen::Vector3d& desired_force,
        double desired_force_radius,
        const Eigen::VectorXd& nominal_state,
        const Eigen::VectorXd* previous_state,
        Eigen::VectorXd& optimized_state) const;

    double resolveDesiredForceRadius(
        const WholeBodyTrajectoryOptimizationInput& input,
        std::size_t waypoint_index) const;

    pinocchio::Model arm_model_;
    mutable std::unique_ptr<pinocchio::Data> data_;
    pinocchio::FrameIndex base_frame_id_{0};
    pinocchio::FrameIndex ee_frame_id_{0};

    WholeBodyOptimizationConfig config_;
    Eigen::VectorXd arm_torque_limits_;
    Eigen::VectorXd lower_arm_limits_;
    Eigen::VectorXd upper_arm_limits_;
    Eigen::VectorXd arm_joint_limit_margins_;
    std::vector<ResolvedCollisionSphere> basket_collision_spheres_;
    std::vector<ResolvedCollisionSphere> body_collision_spheres_;

    polytope polytope_tool_;
    int state_dim_{0};
    bool initialized_{false};
    mutable int dynamic_residual_diagnostics_record_count_{0};
};

} // namespace polytope_wx
