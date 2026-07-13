#include <ros/ros.h>

#include "tcpip_polytope/tcpip.h"


int main(int argc, char * argv[])
{
    ros::init(argc, argv, "tcp_client_test");
    // auto /* node_name */ = /* namespace_name::ClassName */();
    ros::NodeHandle nh;

    polytope_wx::tcpip_client client;
    if (!client.connect_with_retry("127.0.0.1", 5005, -1, 2000)) {
        return 0;
    }

    char buffer[1024];

    client.start_receive_thread();

    client.set_command(
        {0.2, 0.3, 0.0},
        {0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        0.1
    );

    client.send_thread_start();


    // while (ros::ok())
    // {
    //     client.send_thread();
    // }
    // client.connect("127.0.0.1", 5005);
    
    ros::spin();
    ros::shutdown();
    return 0;
}
