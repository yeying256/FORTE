#pragma once

#include <memory>
#include <string>

#include <pinocchio/macros.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/joint/joint-free-flyer.hpp>
#include <pinocchio/spatial/se3.hpp>

#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/jacobian.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "tcpip_polytope/tcpip.h"

namespace polytope_wx {

class Moca_dynamic_pin
{
private:
    pinocchio::Model model_;
    std::unique_ptr<pinocchio::Data> data_;

    std::string urdf_path_;
    bool initialized_{false};

    Eigen::VectorXd q_;
    Eigen::VectorXd dq_;

public:
    Moca_dynamic_pin();
    explicit Moca_dynamic_pin(const std::string& urdf_path, bool use_free_flyer = false);
    ~Moca_dynamic_pin();

    /**
     * @brief 通过 RobotMessage 来更新机械臂的数据
     * 
     * @param robot_state_now 
     * @return true 
     * @return false 
     */
    bool updateFromRobotMessage(const RobotMessage& robot_state_now);

    /**
     * @brief 不用use_free_flyer，因为这样会很复杂，会让它出现很多问题
     * 
     * @param urdf_path 
     * @param use_free_flyer 
     * @return true 
     * @return false 
     */
    bool initFromUrdf(const std::string& urdf_path, bool use_free_flyer = false);
    /**
     * @brief 返回是否已经初始化成功
     * 
     * @return true 
     * @return false 
     */
    bool isInitialized() const;

    /**
     * @brief 返回model
     * 
     * @return const pinocchio::Model& 
     */
    const pinocchio::Model& model() const;
    pinocchio::Model& model();

    /**
     * @brief 返回data
     * 
     * @return const pinocchio::Data& 
     */
    const pinocchio::Data& data() const;
    pinocchio::Data& data();

    /**
     * @brief 获取urdf路径
     * 
     * @return std::string 
     */
    std::string urdfPath() const;


     /**
     * @brief 正向运动学：计算某个 frame 的位姿
     * @param q 关节位置，维度必须等于 model_.nq
     * @param frame_name 目标 frame 名字
     * @return Eigen::Affine3d
     */
    Eigen::Affine3d forwardKinematics(const Eigen::VectorXd& q,
                                      const std::string& frame_name);

    /**
     * @brief 正向运动学：通过 frame id
     */
    Eigen::Affine3d forwardKinematics(const Eigen::VectorXd& q,
                                      pinocchio::FrameIndex frame_id);

    /**
     * @brief 获取 frame id
     */
    pinocchio::FrameIndex getFrameId(const std::string& frame_name) const;


    /**
     * @brief 返回当前状态下的 coriolis/centrifugal compensation
     *        不包含 gravity
     */
    Eigen::VectorXd getCoriolisCompensation() const;

    /**
     * @brief 只返回机械臂前7维
     */
    Eigen::Matrix<double, 7, 1> getArmCoriolisCompensation() const;

    /**
     * @brief 返回当前状态下的 gravity compensation
     */
    Eigen::VectorXd getGravityCompensation() const;

    /**
     * @brief 只返回机械臂前7维重力补偿
     */
    Eigen::Matrix<double, 7, 1> getArmGravityCompensation() const;

    Eigen::VectorXd getCoriolisCompensation(const Eigen::VectorXd& q,
                                        const Eigen::VectorXd& dq) const;

    const Eigen::VectorXd& getQ() const;

    const Eigen::VectorXd& getDQ() const;

    int nq() const;
    int nv() const;
    int njoints() const;

    /**
     * @brief 获取当前机械臂前7维关节位置
     */
    Eigen::Matrix<double, 7, 1> getArmQ() const;

    /**
     * @brief 获取当前机械臂前7维关节速度
     */
    Eigen::Matrix<double, 7, 1> getArmDQ() const;

    /**
     * @brief 获取当前缓存状态下某个 frame 的位姿
     */
    Eigen::Affine3d getFramePose(const std::string& frame_name) const;

    /**
     * @brief 获取当前缓存状态下某个 frame 的位姿
     */
    Eigen::Affine3d getFramePose(pinocchio::FrameIndex frame_id) const;

    /**
     * @brief 获取当前机械臂前7维惯性矩阵
     */
    Eigen::Matrix<double, 7, 7> getArmInertiaMatrix() const;

    /**
     * @brief 获取当前机械臂末端 6x7 Jacobian
     */
    Eigen::Matrix<double, 6, 7> getArmJacobian(const std::string& frame_name) const;

    /**
     * @brief 在给定机械臂 7 维关节位形下获取末端 6x7 Jacobian
     *
     * Jacobian 采用 LOCAL_WORLD_ALIGNED 表达；如果模型里还有夹爪等关节，
     * 这里会自动把非前 7 维关节置 0。
     */
    Eigen::Matrix<double, 6, 7> getArmJacobian(
        const Eigen::Matrix<double, 7, 1>& arm_q,
        const std::string& frame_name) const;

    /**
     * @brief 获取当前机械臂末端 6x7 Jacobian
     */
    // Eigen::Matrix<double, 6, 7> getArmJacobian(pinocchio::FrameIndex frame_id) const;

    /**
     * @brief 获取 frame_a 到 frame_b 的相对位姿 a_T_b
     */
    Eigen::Affine3d getRelativeTransform(const std::string& base_frame,
                                        const std::string& target_frame) const;

    /**
     * @brief 在给定机械臂 7 维关节位形下，计算 frame_a 到 frame_b 的相对位姿 a_T_b
     *
     * 如果模型里还有夹爪等额外关节，会自动把非前 7 维关节置 0。
     */
    Eigen::Affine3d getRelativeTransform(
        const Eigen::Matrix<double, 7, 1>& arm_q,
        const std::string& base_frame,
        const std::string& target_frame) const;

    /**
     * @brief 获取机械臂 Jacobian，并表达在 base_frame 坐标系下
     * @return 6x7
     */
    Eigen::Matrix<double, 6, 7> getArmJacobianInBaseFrame(
        const std::string& base_frame = "moca_base_footprint",
        const std::string& ee_frame   = "moca_franka_EE") const;

    /**
     * @brief 获取移动底盘 Jacobian，并表达在 base_frame 坐标系下
     * @return 6x3
     */
    Eigen::Matrix<double, 6, 3> getMobileBaseJacobianInBaseFrame(
        const std::string& base_frame = "moca_base_footprint",
        const std::string& ee_frame   = "moca_franka_EE") const;

    /**
     * @brief 获取移动操作机器人整体 Jacobian，并表达在 base_frame 坐标系下
     *        排列为 [J_base(6x3) | J_arm(6x7)]
     * @return 6x10
     */
    Eigen::Matrix<double, 6, 10> getWholeBodyJacobianInBaseFrame(
        const std::string& base_frame = "moca_base_footprint",
        const std::string& ee_frame   = "moca_franka_EE") const;

};

} // namespace polytope_wx
