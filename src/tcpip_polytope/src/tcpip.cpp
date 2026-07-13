#include "tcpip_polytope/tcpip.h" // 假设你把上面的定义放在这个头文件里
#include <iostream>
#include <string>
#include <cstring>
#include <thread>
#include <chrono>
#include <vector>
#include <cstdint>
#include <cmath>
// Linux 专用头文件
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/select.h>


namespace polytope_wx {


// --- 构造函数 ---

tcpip_client::tcpip_client() : sock_(-1), server_port_(0), is_connected_(false) {
    std::memset(&robot_message_, 0, sizeof(robot_message_));
    std::memset(&robot_command_, 0, sizeof(robot_command_));
    sock_ = socket(AF_INET, SOCK_STREAM, 0);
    if (sock_ < 0) {
        std::cerr << "[Error] Failed to create socket: " << strerror(errno) << std::endl;
    }
}


// --- 析构函数 ---

tcpip_client::~tcpip_client() {
    stop_requested_ = true;
    disconnect();
    if (recv_thread_.joinable()) {
        recv_thread_.join();
    }
    if (send_thread_.joinable()) {
        send_thread_.join();
    }
}


// --- 连接服务器 ---

void tcpip_client::connect(std::string ip, int port) {

    if (sock_ < 0) {
        std::cerr << "[Error] Invalid socket." << std::endl;
        return;
    }

    struct sockaddr_in server_addr;
    std::memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);

    if (inet_pton(AF_INET, ip.c_str(), &server_addr.sin_addr) <= 0) {
        std::cerr << "[Error] Invalid address / Address not supported: " << ip << std::endl;
        return;
    }
    if (::connect(sock_, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "[Error] Connection failed to " << ip << ":" << port 
                  << " - " << strerror(errno) << std::endl;
        is_connected_ = false;
        return;
    }
    server_ip_ = ip;
    server_port_ = port;
    is_connected_ = true;
    std::cout << "[Info] Connected to " << ip << ":" << port << std::endl;

}


// --- 断开连接 ---

void tcpip_client::disconnect() {
    stop_requested_ = true;
    if (sock_ >= 0) {
        shutdown(sock_, SHUT_RDWR);
        close(sock_);
        sock_ = -1;
    }
    is_connected_ = false;
    std::cout << "[Info] Disconnected." << std::endl;
}



// --- 【核心】接收二进制数据 (通用版) ---





bool tcpip_client::receive_data(void* buffer, size_t max_size, int timeout_ms) {

    if (!is_connected_ || sock_ < 0 || buffer == nullptr) {
        return false;
    }

    size_t total_received = 0;
    auto* byte_buffer = static_cast<char*>(buffer);

    while (total_received < max_size) {
        fd_set read_fds;
        FD_ZERO(&read_fds);
        FD_SET(sock_, &read_fds);

        struct timeval tv;
        tv.tv_sec = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;

        const int activity = select(sock_ + 1, &read_fds, nullptr, nullptr, &tv);
        if (activity <= 0) {
            return false;
        }

        const ssize_t n_bytes = recv(sock_, byte_buffer + total_received, max_size - total_received, 0);

        if (n_bytes < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                return false;
            }
            disconnect();
            return false;
        }
        if (n_bytes == 0) {
            disconnect();
            return false;
        }

        total_received += static_cast<size_t>(n_bytes);
    }

    return true;
}


void tcpip_client::receive_thread()
{
    ros::Rate rate(1000);
    while (ros::ok() && !stop_requested_) {

        if (this->receive_data(this->buffer_, sizeof(RobotMessage), 100)) {
        // int ret = this->receive_data(this->buffer_,100,100);

        RobotMessage incoming_message;
        std::memcpy(&incoming_message, this->buffer_, sizeof(RobotMessage));

        this->mutex_.lock();
        this->robot_message_ = incoming_message;
        this->mutex_.unlock();

        // std::cout << "=== 收到新数据 ===" << std::endl;
        // std::cout << "MsgType: " << this->robot_message_.msg_type << std::endl;
        // std::cout << "Timestamp: " << this->robot_message_.timestamp << std::endl;
        // std::cout << "Reserved: " << this->robot_message_.reserved << std::endl;
        // std::cout << "MobilePos: [";
        // for(int i=0; i<3; ++i) std::cout << this->robot_message_.mobile_positions[i] << (i==2?"":", ");
        // std::cout << "]" << std::endl;
        // std::cout << "ArmPos:    [";
        // for(int i=0; i<7; ++i) std::cout << this->robot_message_.arm_positions[i] << (i==6?"":", ");
        // std::cout << "]" << std::endl;
        // std::cout << "MobileVel: [";
        // for(int i=0; i<3; ++i) std::cout << this->robot_message_.mobile_velocities[i] << (i==2?"":", ");
        // std::cout << "]" << std::endl;
        // std::cout << "ArmVel:    [";
        // for(int i=0; i<7; ++i) std::cout << this->robot_message_.arm_velocities[i] << (i==6?"":", ");
        // std::cout << "]" << std::endl;
        // std::cout << "FiggerPos: " << this->robot_message_.figger_position << std::endl;
        // std::cout << "==================" << std::endl << std::endl;
        }
        else
        {
            std::cout << "没有收到数据" << std::endl;
        }
        // usleep(1000);
        rate.sleep();
    }
}

void tcpip_client::start_receive_thread()
{
    if (recv_thread_.joinable()) {
        return;
    }
    this->recv_thread_ = std::thread(&tcpip_client::receive_thread, this);
}




bool tcpip_client::connect_with_retry(std::string ip, int port, int max_retries, int retry_delay_ms) {

    int attempt = 0;
    stop_requested_ = false;
    std::cout << "[Info] Starting connection to " << ip << ":" << port << "..." << std::endl;

    while (ros::ok() && !stop_requested_) {
        // 1. 尝试连接
        // 注意：这里的 connect 是你之前写的那个同步阻塞版本的 connect
        // 如果之前的 connect 内部没有超时处理，建议给它加一个 socket 超时，否则这里会卡死在 connect 调用上
        connect(ip, port); 

        // 2. 检查是否连接成功
        if (is_connected_) {
            std::cout << "[Success] Connected successfully after " << (attempt + 1) << " attempt(s)." << std::endl;
            return true;
        }

        // 3. 如果失败，检查重试次数
        attempt++;
        // max_retries = -1 表示无限重试
        if (max_retries != -1 && attempt > max_retries) {
            std::cerr << "[Error] Connection failed after " << max_retries << " attempts. Giving up." << std::endl;
            return false;
        }
        // 4. 检查是否被外部强制断开 (例如用户调用了 disconnect())

        // 为了防止死循环，我们在休眠前检查一下标志位 (虽然 connect 失败时 is_connected_ 已经是 false，

        // 但我们可以增加一个 stop_flag 或者检查 sock_ 是否被故意关闭)

        // 简单起见，这里我们依赖用户如果在另一个线程调用 disconnect，需要配合原子变量或互斥锁。

        // 对于单线程简单场景，直接休眠即可。

        std::cout << "[Warning] Connection attempt " << attempt << " failed. Retrying in " 
                  << retry_delay_ms << " ms... (Press Ctrl+C to stop)" << std::endl;
        // 5. 休眠等待，同时响应 ROS shutdown / 显式 stop
        const int sleep_step_ms = 50;
        int waited_ms = 0;
        while (waited_ms < retry_delay_ms && ros::ok() && !stop_requested_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_step_ms));
            waited_ms += sleep_step_ms;
        }
    }

    std::cout << "[Info] Stopping connection retry loop." << std::endl;
    return false;
}


bool tcpip_client::send_command(const RobotCommandStruct& cmd)
{
    if (!is_connected_ || sock_ < 0)
        return false;

    size_t total_sent = 0;
    const char* data_ptr = reinterpret_cast<const char*>(&cmd);
    size_t data_size = sizeof(RobotCommandStruct);

    while (total_sent < data_size)
    {
        ssize_t n = send(sock_, data_ptr + total_sent, data_size - total_sent, 0);

        if (n < 0)
        {
            std::cerr << "[Error] send failed: " << strerror(errno) << std::endl;
            disconnect();
            return false;
        }

        total_sent += n;
    }

    return true;
}

void tcpip_client::send_thread()
{
    ros::Rate rate(1000); // 1000Hz

    while (ros::ok() && !stop_requested_)
    {
        RobotCommandStruct cmd;


        command_mutex_.lock();
        cmd = robot_command_;
        command_mutex_.unlock();
        cmd.timestamp = ros::Time::now().toSec();

        if (!send_command(cmd))
        {
            std::cout << "Send failed" << std::endl;
        }

        rate.sleep();
        // usleep(1000);

    }
}

void tcpip_client::send_thread_start()
{
    if (send_thread_.joinable()) {
        return;
    }

    this->send_thread_ = std::thread(&tcpip_client::send_thread, this);
}


void tcpip_client::set_command(
    const std::vector<double>& mobile,
    const std::vector<double>& arm,
    double finger,
    ArmCommandInterface arm_interface,
    MobileCommandInterface mobile_interface)
{
    command_mutex_.lock();

    robot_command_.Command_type = static_cast<int>(CommandType::CONTROL_CMD);
    robot_command_.reserved = encodeCommandInterfaces(arm_interface, mobile_interface);

    for (int i = 0; i < 3; i++)
        robot_command_.mobile_command[i] = mobile[i];

    for (int i = 0; i < 7; i++)
        robot_command_.arm_command[i] = arm[i];

    robot_command_.figger_position = finger;

    // // ===== 打印指令 =====
    // std::cout << "----- Robot Command -----" << std::endl;

    // std::cout << "Mobile: ";
    // for (int i = 0; i < 3; i++)
    //     std::cout << mobile[i] << " ";
    // std::cout << std::endl;

    // std::cout << "Arm: ";
    // for (int i = 0; i < 7; i++)
    //     std::cout << arm[i] << " ";
    // std::cout << std::endl;

    // std::cout << "Finger: " << finger << std::endl;

    // std::cout << "-------------------------" << std::endl;

    command_mutex_.unlock();

}

void tcpip_client::request_reset_to_home()
{
    RobotCommandStruct reset_cmd;
    std::memset(&reset_cmd, 0, sizeof(reset_cmd));
    reset_cmd.Command_type = static_cast<int>(CommandType::RESET_TO_HOME);
    reset_cmd.timestamp = ros::Time::now().toSec();
    reset_cmd.reserved = encodeCommandInterfaces(
        ArmCommandInterface::NONE,
        MobileCommandInterface::NONE);

    {
        std::lock_guard<std::mutex> lock(command_mutex_);
        robot_command_ = reset_cmd;
    }

    if (!is_connected_)
    {
        return;
    }

    // Send a few times so Isaac can reliably receive the reset request before
    // the controller process disconnects.
    for (int i = 0; i < 3; ++i)
    {
        reset_cmd.timestamp = ros::Time::now().toSec();
        send_command(reset_cmd);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

/**
 * @brief 外部接口返回机器人状态
 * 
 * @return RobotMessage 
 */
RobotMessage tcpip_client::get_robot_state()
{
    std::lock_guard<std::mutex> lock(mutex_);
    return robot_message_;
}

nav_msgs::Odometry
tcpip_client::toOdometryMsg(const RobotMessage& robot_state_now)
{
    nav_msgs::Odometry mobile_base_odom;

    mobile_base_odom.header.stamp = ros::Time::now();
    mobile_base_odom.header.frame_id = "odom";
    mobile_base_odom.child_frame_id = "base_link";

    mobile_base_odom.pose.pose.position.x = robot_state_now.mobile_positions[0];
    mobile_base_odom.pose.pose.position.y = robot_state_now.mobile_positions[1];
    mobile_base_odom.pose.pose.position.z = 0.0;

    double yaw = robot_state_now.mobile_positions[2];
    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, yaw);

    mobile_base_odom.pose.pose.orientation.x = q.x();
    mobile_base_odom.pose.pose.orientation.y = q.y();
    mobile_base_odom.pose.pose.orientation.z = q.z();
    mobile_base_odom.pose.pose.orientation.w = q.w();

    mobile_base_odom.twist.twist.linear.x  = robot_state_now.mobile_velocities[0];
    mobile_base_odom.twist.twist.linear.y  = robot_state_now.mobile_velocities[1];
    mobile_base_odom.twist.twist.linear.z  = 0.0;
    mobile_base_odom.twist.twist.angular.x = 0.0;
    mobile_base_odom.twist.twist.angular.y = 0.0;
    mobile_base_odom.twist.twist.angular.z = robot_state_now.mobile_velocities[2];

    return mobile_base_odom;
}

sensor_msgs::JointState
tcpip_client::toArmJointStateMsg(const RobotMessage& robot_state_now)
{
    sensor_msgs::JointState arm_joint_state;

    arm_joint_state.header.stamp = ros::Time::now();
    arm_joint_state.header.frame_id = "";

    arm_joint_state.name.resize(7);
    arm_joint_state.position.resize(7);
    arm_joint_state.velocity.resize(7);
    arm_joint_state.effort.resize(7, 0.0);

    // 这里用你 MOCA Franka 的关节名字
    arm_joint_state.name[0] = "moca_franka_joint1";
    arm_joint_state.name[1] = "moca_franka_joint2";
    arm_joint_state.name[2] = "moca_franka_joint3";
    arm_joint_state.name[3] = "moca_franka_joint4";
    arm_joint_state.name[4] = "moca_franka_joint5";
    arm_joint_state.name[5] = "moca_franka_joint6";
    arm_joint_state.name[6] = "moca_franka_joint7";

    for (int i = 0; i < 7; ++i)
    {
        arm_joint_state.position[i] = robot_state_now.arm_positions[i];
        arm_joint_state.velocity[i] = robot_state_now.arm_velocities[i];
    }

    return arm_joint_state;
}

sensor_msgs::JointState
tcpip_client::getArmJointStateMsg()
{
    std::lock_guard<std::mutex> lock(mutex_);
    return toArmJointStateMsg(robot_message_);
}



// --- 兼容旧接口的 receive (返回字符串，不推荐用于结构体) ---


} // namespace polytope_wx
