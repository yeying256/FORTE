#include "Moca_controller/Moca_position_controller.h"

#include <ros/ros.h>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "wb_cart_position_controller");
    ros::NodeHandle nh("~");

    int controller_frequency_hz = 1000;
    nh.param("controller_frequency_hz", controller_frequency_hz, controller_frequency_hz);

    WBPositionController::ControllerManager controller_manager;

    if (!controller_manager.init(argc, argv))
    {
        ROS_ERROR("PositionControllerManager: initialization failed.");
        return -1;
    }

    if (!controller_manager.starting())
    {
        ROS_ERROR("PositionControllerManager: starting phase failed.");
        return -1;
    }

    const double controller_period =
        1.0 / static_cast<double>(controller_frequency_hz);
    ros::Rate loop_rate(controller_frequency_hz);

    bool update_state = true;
    while (ros::ok() && update_state)
    {
        ros::spinOnce();

        const double starting_time = ros::Time::now().toSec();
        update_state = controller_manager.update();
        const double elapsed_time = ros::Time::now().toSec() - starting_time;

        if (elapsed_time > controller_period)
        {
            ROS_WARN_STREAM_ONCE(
                "PositionControllerManager: the controller period is set to " <<
                controller_period << " s, but the previous update took " <<
                elapsed_time << " s.");
        }
        loop_rate.sleep();
    }

    ROS_WARN("PositionControllerManager: ROS stopped controller execution.");
    return 0;
}
