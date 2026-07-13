#include "robot_dynamic_pin/Moca_dynamic_pin.h"

#include <iostream>
#include <stdexcept>

#include <pinocchio/algorithm/crba.hpp>

namespace polytope_wx {

Moca_dynamic_pin::Moca_dynamic_pin() = default;

Moca_dynamic_pin::Moca_dynamic_pin(const std::string& urdf_path, bool use_free_flyer)
{
    initFromUrdf(urdf_path, use_free_flyer);
}

Moca_dynamic_pin::~Moca_dynamic_pin() = default;

bool Moca_dynamic_pin::initFromUrdf(const std::string& urdf_path, bool use_free_flyer)
{
    try
    {
        urdf_path_ = urdf_path;

        if (urdf_path_.empty())
        {
            throw std::runtime_error("URDF path is empty.");
        }

        if (use_free_flyer)
        {
            pinocchio::urdf::buildModel(
                urdf_path_,
                pinocchio::JointModelFreeFlyer(),
                model_);
        }
        else
        {
            pinocchio::urdf::buildModel(urdf_path_, model_);
        }

        data_ = std::make_unique<pinocchio::Data>(model_);

        // 初始化当前状态缓存
        q_ = Eigen::VectorXd::Zero(model_.nq);
        dq_ = Eigen::VectorXd::Zero(model_.nv);

        initialized_ = true;

        std::cout << "\n========== Pinocchio Model Info ==========\n";

        std::cout << "Model name: " << model_.name << std::endl;
        std::cout << "nq: " << model_.nq << std::endl;
        std::cout << "nv: " << model_.nv << std::endl;
        std::cout << "njoints: " << model_.njoints << std::endl;
        std::cout << "nframes: " << model_.nframes << std::endl;

        std::cout << "\n----------- JOINTS -----------\n";

        for (pinocchio::JointIndex i = 0; i < model_.njoints; ++i)
        {
            const auto& joint = model_.joints[i];

            std::cout << "Joint ID: " << i
                      << " | name: " << model_.names[i]
                      << " | parent: " << model_.parents[i]
                      << " | nq: " << joint.nq()
                      << " | nv: " << joint.nv()
                      << std::endl;
        }

        std::cout << "\n----------- FRAMES -----------\n";

        for (pinocchio::FrameIndex i = 0; i < model_.nframes; ++i)
        {
            const auto& frame = model_.frames[i];
            std::cout << "Frame ID: " << i
                      << " | name: " << frame.name
                      << " | parent joint: " << frame.parentJoint
                      << " | type: ";
            switch(frame.type)
            {
                case pinocchio::FrameType::JOINT:
                    std::cout << "JOINT";
                    break;
                case pinocchio::FrameType::FIXED_JOINT:
                    std::cout << "FIXED_JOINT";
                    break;
                case pinocchio::FrameType::BODY:
                    std::cout << "BODY";
                    break;
                case pinocchio::FrameType::OP_FRAME:
                    std::cout << "OP_FRAME";
                    break;
                case pinocchio::FrameType::SENSOR:
                    std::cout << "SENSOR";
                    break;
                default:
                    std::cout << "UNKNOWN";
            }
            std::cout << std::endl;
        }
        std::cout << "===========================================\n\n";

        return true;
    }
    catch (const std::exception& e)
    {
        initialized_ = false;
        data_.reset();
        q_.resize(0);
        dq_.resize(0);

        std::cerr << "[Moca_dynamic_pin] Failed to initialize from URDF: "
                  << e.what() << std::endl;
        return false;
    }
}

bool Moca_dynamic_pin::isInitialized() const
{
    return initialized_;
}

const pinocchio::Model& Moca_dynamic_pin::model() const
{
    if (!initialized_)
    {
        throw std::runtime_error("Pinocchio model is not initialized.");
    }
    return model_;
}

pinocchio::Model& Moca_dynamic_pin::model()
{
    if (!initialized_)
    {
        throw std::runtime_error("Pinocchio model is not initialized.");
    }
    return model_;
}

const pinocchio::Data& Moca_dynamic_pin::data() const
{
    if (!initialized_ || !data_)
    {
        throw std::runtime_error("Pinocchio data is not initialized.");
    }
    return *data_;
}

pinocchio::Data& Moca_dynamic_pin::data()
{
    if (!initialized_ || !data_)
    {
        throw std::runtime_error("Pinocchio data is not initialized.");
    }
    return *data_;
}

std::string Moca_dynamic_pin::urdfPath() const
{
    return urdf_path_;
}

pinocchio::FrameIndex
Moca_dynamic_pin::getFrameId(const std::string& frame_name) const
{
    if (!initialized_)
    {
        throw std::runtime_error("Pinocchio model not initialized.");
    }

    const pinocchio::FrameIndex frame_id = model_.getFrameId(frame_name);

    if (frame_id == static_cast<pinocchio::FrameIndex>(model_.nframes))
    {
        throw std::runtime_error("Frame not found: " + frame_name);
    }

    return frame_id;
}

Eigen::Affine3d
Moca_dynamic_pin::forwardKinematics(const Eigen::VectorXd& q,
                                    pinocchio::FrameIndex frame_id)
{
    if (!initialized_)
    {
        throw std::runtime_error("Pinocchio model not initialized.");
    }

    if (q.size() != model_.nq)
    {
        throw std::runtime_error("Joint dimension mismatch in forwardKinematics.");
    }

    if (frame_id >= model_.nframes)
    {
        throw std::runtime_error("Frame index out of range.");
    }

    pinocchio::forwardKinematics(model_, *data_, q);
    pinocchio::updateFramePlacements(model_, *data_);

    const pinocchio::SE3& oMf = data_->oMf[frame_id];
    return Eigen::Affine3d(oMf.toHomogeneousMatrix());
}

Eigen::Affine3d
Moca_dynamic_pin::forwardKinematics(const Eigen::VectorXd& q,
                                    const std::string& frame_name)
{
    return forwardKinematics(q, getFrameId(frame_name));
}

bool Moca_dynamic_pin::updateFromRobotMessage(const RobotMessage& robot_state_now)
{
    if (!initialized_)
    {
        std::cerr << "[Moca_dynamic_pin] Model not initialized!" << std::endl;
        return false;
    }

    try
    {
        q_.setZero();
        dq_.setZero();

        // 这里只更新机械臂前 7 维
        // 如果 URDF 后面还有 gripper，自然保持为 0
        const int arm_dim = std::min<int>(7, std::min<int>(model_.nq, model_.nv));

        for (int i = 0; i < arm_dim; ++i)
        {
            q_(i) = robot_state_now.arm_positions[i];
            dq_(i) = robot_state_now.arm_velocities[i];
        }

        pinocchio::forwardKinematics(model_, *data_, q_, dq_);
        pinocchio::updateFramePlacements(model_, *data_);
        pinocchio::computeJointJacobians(model_, *data_);
        pinocchio::crba(model_, *data_, q_);
        pinocchio::computeGeneralizedGravity(model_, *data_, q_);

        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[Moca_dynamic_pin] updateFromRobotMessage failed: "
                  << e.what() << std::endl;
        return false;
    }
}

Eigen::VectorXd Moca_dynamic_pin::getCoriolisCompensation() const
{
    if (!initialized_)
    {
        throw std::runtime_error("[Moca_dynamic_pin] getCoriolisCompensation: model not initialized.");
    }

    if (!data_)
    {
        throw std::runtime_error("[Moca_dynamic_pin] getCoriolisCompensation: data is null.");
    }

    auto& data_ref = *const_cast<pinocchio::Data*>(data_.get());

    // 非线性项 = 重力 + 科氏/离心
    const Eigen::VectorXd nle = pinocchio::nonLinearEffects(model_, data_ref, q_, dq_);

    // 重力项
    const Eigen::VectorXd g = pinocchio::computeGeneralizedGravity(model_, data_ref, q_);

    // 返回科氏/离心项
    return nle - g;
}

Eigen::Matrix<double, 7, 1> Moca_dynamic_pin::getArmCoriolisCompensation() const
{
    const Eigen::VectorXd coriolis_all = getCoriolisCompensation();

    if (coriolis_all.size() < 7)
    {
        throw std::runtime_error("[Moca_dynamic_pin] getArmCoriolisCompensation: coriolis vector size < 7.");
    }

    return coriolis_all.head<7>();
}

Eigen::VectorXd Moca_dynamic_pin::getGravityCompensation() const
{
    if (!initialized_)
    {
        throw std::runtime_error("[Moca_dynamic_pin] getGravityCompensation: model not initialized.");
    }

    if (!data_)
    {
        throw std::runtime_error("[Moca_dynamic_pin] getGravityCompensation: data is null.");
    }

    auto& data_ref = *const_cast<pinocchio::Data*>(data_.get());
    return pinocchio::computeGeneralizedGravity(model_, data_ref, q_);
}

Eigen::Matrix<double, 7, 1> Moca_dynamic_pin::getArmGravityCompensation() const
{
    const Eigen::VectorXd gravity_all = getGravityCompensation();

    if (gravity_all.size() < 7)
    {
        throw std::runtime_error("[Moca_dynamic_pin] getArmGravityCompensation: gravity vector size < 7.");
    }

    return gravity_all.head<7>();
}

// ===== 如果你头文件里加了这些接口，就把下面也保留 =====

const Eigen::VectorXd& Moca_dynamic_pin::getQ() const
{
    if (!initialized_)
    {
        throw std::runtime_error("[Moca_dynamic_pin] getQ: model not initialized.");
    }
    return q_;
}

const Eigen::VectorXd& Moca_dynamic_pin::getDQ() const
{
    if (!initialized_)
    {
        throw std::runtime_error("[Moca_dynamic_pin] getDQ: model not initialized.");
    }
    return dq_;
}

int Moca_dynamic_pin::nq() const
{
    if (!initialized_)
    {
        throw std::runtime_error("[Moca_dynamic_pin] nq: model not initialized.");
    }
    return model_.nq;
}

int Moca_dynamic_pin::nv() const
{
    if (!initialized_)
    {
        throw std::runtime_error("[Moca_dynamic_pin] nv: model not initialized.");
    }
    return model_.nv;
}

int Moca_dynamic_pin::njoints() const
{
    if (!initialized_)
    {
        throw std::runtime_error("[Moca_dynamic_pin] njoints: model not initialized.");
    }
    return model_.njoints;
}

Eigen::VectorXd
Moca_dynamic_pin::getCoriolisCompensation(const Eigen::VectorXd& q,
                                          const Eigen::VectorXd& dq) const
{
    if (!initialized_)
    {
        throw std::runtime_error("[Moca_dynamic_pin] getCoriolisCompensation(q,dq): model not initialized.");
    }

    if (!data_)
    {
        throw std::runtime_error("[Moca_dynamic_pin] getCoriolisCompensation(q,dq): data is null.");
    }

    if (q.size() != model_.nq)
    {
        throw std::runtime_error("[Moca_dynamic_pin] getCoriolisCompensation(q,dq): q dimension mismatch.");
    }

    if (dq.size() != model_.nv)
    {
        throw std::runtime_error("[Moca_dynamic_pin] getCoriolisCompensation(q,dq): dq dimension mismatch.");
    }

    auto& data_ref = *const_cast<pinocchio::Data*>(data_.get());

    const Eigen::VectorXd nle = pinocchio::nonLinearEffects(model_, data_ref, q, dq);
    const Eigen::VectorXd g   = pinocchio::computeGeneralizedGravity(model_, data_ref, q);

    return nle - g;
}

Eigen::Affine3d
Moca_dynamic_pin::getRelativeTransform(const std::string& base_frame,
                                       const std::string& target_frame) const
{
    if (!initialized_)
    {
        throw std::runtime_error("[Moca_dynamic_pin] getRelativeTransform: model not initialized.");
    }

    auto& data_ref = *const_cast<pinocchio::Data*>(data_.get());

    pinocchio::forwardKinematics(model_, data_ref, q_, dq_);
    pinocchio::updateFramePlacements(model_, data_ref);

    const pinocchio::FrameIndex base_id   = getFrameId(base_frame);
    const pinocchio::FrameIndex target_id = getFrameId(target_frame);

    const pinocchio::SE3& w_T_base   = data_ref.oMf[base_id];
    const pinocchio::SE3& w_T_target = data_ref.oMf[target_id];

    const pinocchio::SE3 base_T_target = w_T_base.inverse() * w_T_target;

    return Eigen::Affine3d(base_T_target.toHomogeneousMatrix());
}

Eigen::Affine3d
Moca_dynamic_pin::getRelativeTransform(const Eigen::Matrix<double, 7, 1>& arm_q,
                                       const std::string& base_frame,
                                       const std::string& target_frame) const
{
    if (!initialized_)
    {
        throw std::runtime_error("[Moca_dynamic_pin] getRelativeTransform(q): model not initialized.");
    }

    if (!data_)
    {
        throw std::runtime_error("[Moca_dynamic_pin] getRelativeTransform(q): data is null.");
    }

    Eigen::VectorXd q_full = Eigen::VectorXd::Zero(model_.nq);
    const int arm_dim = std::min<int>(7, model_.nq);
    q_full.head(arm_dim) = arm_q.head(arm_dim);

    auto& data_ref = *const_cast<pinocchio::Data*>(data_.get());

    pinocchio::forwardKinematics(model_, data_ref, q_full);
    pinocchio::updateFramePlacements(model_, data_ref);

    const pinocchio::FrameIndex base_id = getFrameId(base_frame);
    const pinocchio::FrameIndex target_id = getFrameId(target_frame);

    const pinocchio::SE3& w_T_base = data_ref.oMf[base_id];
    const pinocchio::SE3& w_T_target = data_ref.oMf[target_id];

    const pinocchio::SE3 base_T_target = w_T_base.inverse() * w_T_target;
    return Eigen::Affine3d(base_T_target.toHomogeneousMatrix());
}


Eigen::Matrix<double, 6, 7>
Moca_dynamic_pin::getArmJacobianInBaseFrame(const std::string& base_frame,
                                            const std::string& ee_frame) const
{
    if (!initialized_)
    {
        throw std::runtime_error("[Moca_dynamic_pin] getArmJacobianInBaseFrame: model not initialized.");
    }

    auto& data_ref = *const_cast<pinocchio::Data*>(data_.get());

    const pinocchio::FrameIndex base_id = getFrameId(base_frame);
    const pinocchio::FrameIndex ee_id   = getFrameId(ee_frame);

    pinocchio::computeJointJacobians(model_, data_ref, q_);
    pinocchio::updateFramePlacements(model_, data_ref);

    Eigen::MatrixXd J_world(6, model_.nv);
    J_world.setZero();

    pinocchio::getFrameJacobian(
        model_,
        data_ref,
        ee_id,
        pinocchio::ReferenceFrame::WORLD,
        J_world);

    const pinocchio::SE3& w_T_base = data_ref.oMf[base_id];
    const pinocchio::SE3  base_T_w = w_T_base.inverse();

    Eigen::Matrix<double, 6, 6> X_base_world = base_T_w.toActionMatrix();
    Eigen::MatrixXd J_base = X_base_world * J_world;

    if (J_base.cols() < 7)
    {
        throw std::runtime_error("[Moca_dynamic_pin] getArmJacobianInBaseFrame: nv < 7.");
    }

    return J_base.leftCols<7>();
}


Eigen::Matrix<double, 6, 3>
Moca_dynamic_pin::getMobileBaseJacobianInBaseFrame(const std::string& base_frame,
                                                   const std::string& ee_frame) const
{
    const Eigen::Affine3d base_T_ee = getRelativeTransform(base_frame, ee_frame);

    Eigen::Matrix<double, 6, 3> J_base;
    J_base.setZero();

    const double x = base_T_ee.translation()(0);
    const double y = base_T_ee.translation()(1);

    // base x
    J_base(0, 0) = 1.0;

    // base y
    J_base(1, 1) = 1.0;

    // base yaw
    J_base(0, 2) = -y;
    J_base(1, 2) =  x;
    J_base(5, 2) = 1.0;

    return J_base;
}

Eigen::Matrix<double, 6, 10>
Moca_dynamic_pin::getWholeBodyJacobianInBaseFrame(const std::string& base_frame,
                                                  const std::string& ee_frame) const
{
    Eigen::Matrix<double, 6, 10> J_all;
    J_all.setZero();

    const Eigen::Matrix<double, 6, 3> J_base =
        getMobileBaseJacobianInBaseFrame(base_frame, ee_frame);

    const Eigen::Matrix<double, 6, 7> J_arm =
        getArmJacobianInBaseFrame(base_frame, ee_frame);

    J_all << J_base, J_arm;
    return J_all;
}

Eigen::Matrix<double, 6, 7>
Moca_dynamic_pin::getArmJacobian(const std::string& frame_name) const
{
    if (!initialized_)
    {
        throw std::runtime_error("[Moca_dynamic_pin] getArmJacobian: model not initialized.");
    }

    if (!data_)
    {
        throw std::runtime_error("[Moca_dynamic_pin] getArmJacobian: data is null.");
    }

    const pinocchio::FrameIndex frame_id = getFrameId(frame_name);

    auto& data_ref = *const_cast<pinocchio::Data*>(data_.get());

    Eigen::Matrix<double, 6, Eigen::Dynamic> J(6, model_.nv);
    J.setZero();

    pinocchio::computeFrameJacobian(
        model_,
        data_ref,
        q_,
        frame_id,
        pinocchio::LOCAL_WORLD_ALIGNED,
        J
    );

    // std::cout << "J: " << J << std::endl;

    if (model_.nv < 7)
    {
        throw std::runtime_error("[Moca_dynamic_pin] getArmJacobian: nv < 7.");
    }

    return J.leftCols<7>();
}

Eigen::Matrix<double, 6, 7>
Moca_dynamic_pin::getArmJacobian(const Eigen::Matrix<double, 7, 1>& arm_q,
                                 const std::string& frame_name) const
{
    if (!initialized_)
    {
        throw std::runtime_error("[Moca_dynamic_pin] getArmJacobian(q): model not initialized.");
    }

    if (!data_)
    {
        throw std::runtime_error("[Moca_dynamic_pin] getArmJacobian(q): data is null.");
    }

    const pinocchio::FrameIndex frame_id = getFrameId(frame_name);

    Eigen::VectorXd q_full = Eigen::VectorXd::Zero(model_.nq);
    const int arm_dim = std::min<int>(7, model_.nq);
    q_full.head(arm_dim) = arm_q.head(arm_dim);

    auto& data_ref = *const_cast<pinocchio::Data*>(data_.get());

    Eigen::Matrix<double, 6, Eigen::Dynamic> J(6, model_.nv);
    J.setZero();

    pinocchio::computeFrameJacobian(
        model_,
        data_ref,
        q_full,
        frame_id,
        pinocchio::LOCAL_WORLD_ALIGNED,
        J);

    if (model_.nv < 7)
    {
        throw std::runtime_error("[Moca_dynamic_pin] getArmJacobian(q): nv < 7.");
    }

    return J.leftCols<7>();
}

Eigen::Matrix<double, 7, 7>
Moca_dynamic_pin::getArmInertiaMatrix() const
{
    if (!initialized_)
    {
        throw std::runtime_error("[Moca_dynamic_pin] getArmInertiaMatrix: model not initialized.");
    }

    if (!data_)
    {
        throw std::runtime_error("[Moca_dynamic_pin] getArmInertiaMatrix: data is null.");
    }

    auto& data_ref = *const_cast<pinocchio::Data*>(data_.get());

    // 计算机器人在当前关节位置 q_ 下的 关节空间惯性矩阵（mass matrix）。
    pinocchio::crba(model_, data_ref, q_);

    Eigen::MatrixXd M = data_ref.M;
    M.triangularView<Eigen::StrictlyLower>() =
        M.transpose().triangularView<Eigen::StrictlyLower>();

    if (M.rows() < 7 || M.cols() < 7)
    {
        throw std::runtime_error("[Moca_dynamic_pin] getArmInertiaMatrix: inertia matrix size < 7.");
    }

    return M.topLeftCorner<7, 7>();
}



} // namespace polytope_wx
