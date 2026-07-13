#include <ros/ros.h>

#include <cerrno>
#include <cstring>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

#include <pinocchio/fwd.hpp>
#include "polytope_ros/polytope.h"
#include "polytope_ros/nullspace_optimizer.h"

namespace
{

bool directoryExists(const std::string& path)
{
    if (path.empty())
    {
        return false;
    }

    struct stat info;
    if (stat(path.c_str(), &info) != 0)
    {
        return false;
    }

    return S_ISDIR(info.st_mode);
}

bool ensureDirectory(const std::string& path)
{
    if (path.empty() || directoryExists(path))
    {
        return true;
    }

    const std::size_t separator = path.find_last_of('/');
    if (separator != std::string::npos)
    {
        const std::string parent = path.substr(0, separator);
        if (!parent.empty() && !ensureDirectory(parent))
        {
            return false;
        }
    }

    if (::mkdir(path.c_str(), 0755) == 0 || errno == EEXIST)
    {
        return true;
    }

    return false;
}

bool ensureParentDirectoryForFile(const std::string& file_path)
{
    const std::size_t separator = file_path.find_last_of('/');
    if (separator == std::string::npos)
    {
        return true;
    }

    return ensureDirectory(file_path.substr(0, separator));
}

} // namespace

int main(int argc, char * argv[])
{
    ros::init(argc, argv, "polytope");
    ros::NodeHandle nh("~");

    std::string urdf_path;
    std::string ee_frame;
    std::string vertices_csv_path;
    bool gravity_adjusted = true;
    std::vector<double> joint_positions_param;
    nh.param<std::string>("urdf_path", urdf_path, "");
    nh.param<std::string>("ee_frame", ee_frame, "");
    nh.param<std::string>("vertices_csv_path", vertices_csv_path, "/tmp/polytope_vertices.csv");
    nh.param<bool>("gravity_adjusted", gravity_adjusted, true);
    nh.getParam("joint_positions", joint_positions_param);

    if (urdf_path.empty())
    {
        ROS_WARN("polytope_node: no urdf_path provided. Library smoke test only.");
        return 0;
    }

    polytope_wx::NullspaceOptimizationConfig config;
    config.urdf_path = urdf_path;
    config.ee_frame = ee_frame.empty() ? "moca_franka_EE" : ee_frame;
    config.verbose = true;

    polytope_wx::PolytopeNullspaceOptimizer optimizer;
    if (!optimizer.initialize(config))
    {
        ROS_ERROR("polytope_node: failed to initialize nullspace optimizer.");
        return -1;
    }

    ROS_INFO_STREAM(
        "polytope_node: standalone polytope nullspace optimizer initialized for frame "
        << config.ee_frame << " with nq = " << optimizer.model().nq << ".");

    Eigen::VectorXd q = optimizer.defaultConfiguration();
    if (!joint_positions_param.empty())
    {
        if (static_cast<int>(joint_positions_param.size()) != q.size())
        {
            ROS_WARN_STREAM(
                "polytope_node: joint_positions size mismatch. Expected "
                << q.size() << ", got " << joint_positions_param.size()
                << ". Falling back to default configuration.");
        }
        else
        {
            for (int i = 0; i < q.size(); ++i)
            {
                q(i) = joint_positions_param[i];
            }
        }
    }

    const Eigen::MatrixXd vertices =
        optimizer.computeTranslationalForcePolytope(q, gravity_adjusted);

    if (vertices.rows() == 0)
    {
        ROS_WARN("polytope_node: polytope computation returned no vertices.");
        return 0;
    }

    if (!ensureParentDirectoryForFile(vertices_csv_path))
    {
        ROS_ERROR_STREAM(
            "polytope_node: failed to create output directory for "
            << vertices_csv_path << ": " << std::strerror(errno));
        return -1;
    }

    std::ofstream csv_file(vertices_csv_path);
    if (!csv_file.is_open())
    {
        ROS_ERROR_STREAM("polytope_node: failed to open " << vertices_csv_path);
        return -1;
    }

    csv_file << "fx,fy,fz\n";
    for (int i = 0; i < vertices.rows(); ++i)
    {
        csv_file << vertices(i, 0) << "," << vertices(i, 1) << "," << vertices(i, 2) << "\n";
    }
    csv_file.close();

    ROS_INFO_STREAM(
        "polytope_node: wrote " << vertices.rows()
        << " polytope vertices to " << vertices_csv_path << ".");

    return 0;
}
