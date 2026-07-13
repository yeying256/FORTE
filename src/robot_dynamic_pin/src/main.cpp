#include "robot_dynamic_pin/Moca_dynamic_pin.h"

#include <iostream>
#include <string>

#include <ros/ros.h>

int main(int argc, char * argv[])
{
    ros::init(argc, argv, "pinoccin");

    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    std::string urdf_path;
    pnh.param<std::string>("urdf_path", urdf_path, "");
    if (urdf_path.empty())
    {
        ROS_ERROR("robot_dynamic_pin test node requires the private param '~urdf_path'.");
        return 1;
    }

    polytope_wx::Moca_dynamic_pin pin_dy(urdf_path, false);
    std::cout << "Pinocchio C++ version is working!" << std::endl;

    ros::spin();
    ros::shutdown();
    return 0;
}
