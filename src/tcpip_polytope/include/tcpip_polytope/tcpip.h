#pragma once

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

#include "yaml-cpp/yaml.h"
#include <mutex>
#include <atomic>

#include <ros/ros.h>

#include <thread>

#include <nav_msgs/Odometry.h>
#include <tf2/LinearMath/Quaternion.h>


#include <sensor_msgs/JointState.h>

struct Config {
    std::string server_ip;
    int server_port;
    int timeout_ms;
    bool enable_logging;
    std::vector<double> joint_limits;
};



namespace polytope_wx
{

enum class CommandType : int
{
    HEARTBEAT = 1,
    STATE_UPDATE = 2,
    CONTROL_CMD = 3,
    ERROR_REPORT = 4,
    RESET_TO_HOME = 5
};

enum class MobileCommandInterface : int
{
    NONE = 0,
    VELOCITY = 1,
    WRENCH = 2,
    POSITION_WORLD = 3
};

enum class ArmCommandInterface : int
{
    NONE = 0,
    POSITION = 1,
    VELOCITY = 2,
    TORQUE = 3
};

inline int encodeCommandInterfaces(const ArmCommandInterface arm_interface,
                                   const MobileCommandInterface mobile_interface)
{
    return (static_cast<int>(mobile_interface) << 8) |
           (static_cast<int>(arm_interface) & 0xFF);
}

inline ArmCommandInterface decodeArmCommandInterface(const int encoded_interfaces)
{
    return static_cast<ArmCommandInterface>(encoded_interfaces & 0xFF);
}

inline MobileCommandInterface decodeMobileCommandInterface(const int encoded_interfaces)
{
    return static_cast<MobileCommandInterface>((encoded_interfaces >> 8) & 0xFF);
}

// 只有网络二进制协议结构体需要 1 字节对齐。
#pragma pack(push, 1)
struct RobotMessage {
    int msg_type;           // ctypes.c_int (4 bytes)
    double timestamp;       // ctypes.c_double (8 bytes)
    int reserved;           // ctypes.c_int (4 bytes)
    double mobile_positions[3];  // ctypes.c_double * 3 (24 bytes)
    double arm_positions[7];     // ctypes.c_double * 7 (56 bytes)
    double mobile_velocities[3]; // ctypes.c_double * 3 (24 bytes)
    double arm_velocities[7];    // ctypes.c_double * 7 (56 bytes)
    double figger_position;      // ctypes.c_double (8 bytes)
};

struct RobotCommandStruct
{
    int Command_type;
    double timestamp;
    int reserved;

    double mobile_command[3];
    double arm_command[7];

    double figger_position;
};
#pragma pack(pop)



class tcpip_client
{

    public:
    // 构造函数
    tcpip_client();

    // tcpip_client();

    // 析构函数
    ~tcpip_client();

    /**
     * @brief 连接服务器
     * 
     * @param ip 服务器IP地址
     * @param port 服务器端口号 
     */
    void connect(std::string ip, int port);

    /**
     * @brief 断线重连
     * 
     * @param ip 
     * @param port 
     * @param max_retries -1是无限制重连
     * @param retry_delay_ms 
     * @return true 
     * @return false 
     */
    bool connect_with_retry(std::string ip, int port, int max_retries = -1, int retry_delay_ms = 2000);
    
    /**
     * @brief 断开连接
     * 
     */
    void disconnect();

    /**

    * @brief 从 socket 接收指定长度的二进制数据
    * @param buffer 接收缓冲区指针
    * @param size 需要接收的字节数
    * @param timeout_ms 超时时间 (毫秒)，默认 1000ms
    * @return true 如果成功接收了完整的 size 字节，false 如果超时或断开
    */
    bool receive_data(void* buffer, size_t size, int timeout_ms);
    /**
     * @brief 发送函数
     * 
     * @param cmd 
     * @return true 
     * @return false 
     */
    bool send_command(const RobotCommandStruct& cmd);

    void send_thread();
    /**
     * @brief 发送线程启动
     * 
     */
    void send_thread_start();


    void set_command(
    const std::vector<double>& mobile,
    const std::vector<double>& arm,
    double finger,
    ArmCommandInterface arm_interface = ArmCommandInterface::VELOCITY,
    MobileCommandInterface mobile_interface = MobileCommandInterface::VELOCITY);

    void request_reset_to_home();

    void receive_thread();

    /**
     * @brief 启动接收线程
     * 
     */
    void start_receive_thread();

    RobotMessage get_robot_state();

    /**
     * @brief 转换三个数值到和odom
     * 
     * @param robot_state_now 
     * @return nav_msgs::Odometry 
     */
    nav_msgs::Odometry toOdometryMsg(const RobotMessage& robot_state_now);


    /**
     * @brief 将 RobotMessage 转成机械臂 JointState
     * 
     * @param robot_state_now
     * @return sensor_msgs::JointState
     */
    sensor_msgs::JointState toArmJointStateMsg(const RobotMessage& robot_state_now);

    /**
     * @brief 直接返回当前缓存状态对应的机械臂 JointState
     * 
     * @return sensor_msgs::JointState
     */
    sensor_msgs::JointState getArmJointStateMsg();


    private:
    int sock_;
    std::string server_ip_;
    int server_port_;
    bool is_connected_;
    // 反馈的数据
    RobotMessage robot_message_;
    Config config_;
    char buffer_[sizeof(RobotMessage)];
    std::mutex mutex_;

    RobotCommandStruct robot_command_;
    std::mutex command_mutex_;
    std::thread recv_thread_;
    std::thread send_thread_;
    std::atomic<bool> stop_requested_{false};

};


    
} // namespace polytope_wx
