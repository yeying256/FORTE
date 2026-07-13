#include <cmath>
#include <cstdint>
#include <cctype>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <geometry_msgs/TransformStamped.h>
#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include <moca_realsense/CaptureImage.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/static_transform_broadcaster.h>

namespace
{
sensor_msgs::CameraInfo cameraInfoFromIntrinsics(const rs2_intrinsics& intr,
                                                 const std::string& frame_id,
                                                 const ros::Time& stamp)
{
  sensor_msgs::CameraInfo info;
  info.header.stamp = stamp;
  info.header.frame_id = frame_id;
  info.width = static_cast<uint32_t>(intr.width);
  info.height = static_cast<uint32_t>(intr.height);
  info.distortion_model = "plumb_bob";
  info.D.resize(5, 0.0);
  for (size_t i = 0; i < info.D.size() && i < 5; ++i)
  {
    info.D[i] = intr.coeffs[i];
  }

  info.K[0] = intr.fx;
  info.K[2] = intr.ppx;
  info.K[4] = intr.fy;
  info.K[5] = intr.ppy;
  info.K[8] = 1.0;

  info.R[0] = 1.0;
  info.R[4] = 1.0;
  info.R[8] = 1.0;

  info.P[0] = intr.fx;
  info.P[2] = intr.ppx;
  info.P[5] = intr.fy;
  info.P[6] = intr.ppy;
  info.P[10] = 1.0;
  return info;
}

geometry_msgs::TransformStamped opticalTransform(const std::string& parent,
                                                 const std::string& child)
{
  geometry_msgs::TransformStamped transform;
  transform.header.stamp = ros::Time::now();
  transform.header.frame_id = parent;
  transform.child_frame_id = child;
  transform.transform.translation.x = 0.0;
  transform.transform.translation.y = 0.0;
  transform.transform.translation.z = 0.0;

  tf2::Quaternion q;
  q.setRPY(-M_PI / 2.0, 0.0, -M_PI / 2.0);
  transform.transform.rotation.x = q.x();
  transform.transform.rotation.y = q.y();
  transform.transform.rotation.z = q.z();
  transform.transform.rotation.w = q.w();
  return transform;
}

std::string upperAscii(std::string value)
{
  for (char& c : value)
  {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  }
  return value;
}

rs2_format parseColorFormat(const std::string& value)
{
  const std::string format = upperAscii(value);
  if (format == "RGB8")
  {
    return RS2_FORMAT_RGB8;
  }
  if (format == "BGR8")
  {
    return RS2_FORMAT_BGR8;
  }
  if (format == "YUYV")
  {
    return RS2_FORMAT_YUYV;
  }
  if (format == "ANY")
  {
    return RS2_FORMAT_ANY;
  }
  ROS_WARN_STREAM("moca_realsense: unsupported color_format='" << value
                  << "', falling back to RGB8.");
  return RS2_FORMAT_RGB8;
}

std::string colorFormatName(rs2_format format)
{
  switch (format)
  {
    case RS2_FORMAT_RGB8:
      return "RGB8";
    case RS2_FORMAT_BGR8:
      return "BGR8";
    case RS2_FORMAT_YUYV:
      return "YUYV";
    case RS2_FORMAT_ANY:
      return "ANY";
    default:
      return "UNKNOWN";
  }
}
}  // namespace

class MocaRealSenseNode
{
 public:
  MocaRealSenseNode()
      : nh_(),
        pnh_("~"),
        align_to_color_(RS2_STREAM_COLOR)
  {
    pnh_.param<std::string>("serial_no", serial_no_, "");
    pnh_.param<std::string>("frame_id", frame_id_, "moca_realsense_link");
    pnh_.param<std::string>("color_frame_id", color_frame_id_, "moca_realsense_color_optical_frame");
    pnh_.param<std::string>("depth_frame_id", depth_frame_id_, "moca_realsense_depth_optical_frame");
    pnh_.param("color_width", color_width_, 640);
    pnh_.param("color_height", color_height_, 480);
    pnh_.param("depth_width", depth_width_, 640);
    pnh_.param("depth_height", depth_height_, 480);
    pnh_.param("fps", fps_, 30);
    pnh_.param<std::string>("color_format", color_format_name_, "RGB8");
    color_format_ = parseColorFormat(color_format_name_);
    pnh_.param("enable_color", enable_color_, true);
    pnh_.param("enable_depth", enable_depth_, true);
    pnh_.param("align_depth_to_color", align_depth_to_color_, true);
    pnh_.param("publish_color_topic", publish_color_topic_, true);
    pnh_.param("publish_depth_topic", publish_depth_topic_, true);
    pnh_.param("publish_pointcloud", publish_pointcloud_, true);
    pnh_.param("pointcloud_stride", pointcloud_stride_, 4);
    pnh_.param("publish_every_n_frames", publish_every_n_frames_, 1);
    pnh_.param("wait_timeout_ms", wait_timeout_ms_, 1000);
    pnh_.param("publish_tf", publish_tf_, true);
    pnh_.param("show_color_window", show_color_window_, false);
    pnh_.param<std::string>("color_window_name", color_window_name_, "MOCA RealSense");
    pnh_.param("color_window_scale", color_window_scale_, 1.0);
    pnh_.param("capture_jpeg_quality", capture_jpeg_quality_, 90);

    if (pointcloud_stride_ < 1)
    {
      pointcloud_stride_ = 1;
    }
    if (publish_every_n_frames_ < 1)
    {
      publish_every_n_frames_ = 1;
    }
    if (wait_timeout_ms_ < 100)
    {
      wait_timeout_ms_ = 100;
    }
    if (color_window_scale_ <= 0.0)
    {
      color_window_scale_ = 1.0;
    }
    if (capture_jpeg_quality_ < 1)
    {
      capture_jpeg_quality_ = 1;
    }
    if (capture_jpeg_quality_ > 100)
    {
      capture_jpeg_quality_ = 100;
    }

    color_pub_ = nh_.advertise<sensor_msgs::Image>("color/image_raw", 1);
    color_info_pub_ = nh_.advertise<sensor_msgs::CameraInfo>("color/camera_info", 1);
    depth_pub_ = nh_.advertise<sensor_msgs::Image>("depth/image_raw", 1);
    depth_info_pub_ = nh_.advertise<sensor_msgs::CameraInfo>("depth/camera_info", 1);
    pointcloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("depth/color/points", 1);
    capture_service_ = pnh_.advertiseService(
        "capture_image", &MocaRealSenseNode::captureImageService, this);
  }

  void start()
  {
    rs2::config cfg;
    if (!serial_no_.empty())
    {
      cfg.enable_device(serial_no_);
    }
    if (enable_color_)
    {
      cfg.enable_stream(RS2_STREAM_COLOR, color_width_, color_height_, color_format_, fps_);
    }
    if (enable_depth_)
    {
      cfg.enable_stream(RS2_STREAM_DEPTH, depth_width_, depth_height_, RS2_FORMAT_Z16, fps_);
    }

    profile_ = pipeline_.start(cfg);

    if (publish_tf_)
    {
      static_tf_broadcaster_.sendTransform(opticalTransform(frame_id_, color_frame_id_));
      static_tf_broadcaster_.sendTransform(opticalTransform(frame_id_, depth_frame_id_));
    }

    ROS_INFO_STREAM("moca_realsense: streaming RealSense topics. color="
                    << enable_color_ << ", depth=" << enable_depth_
                    << ", publish_color_topic=" << publish_color_topic_
                    << ", publish_depth_topic=" << publish_depth_topic_
                    << ", pointcloud=" << publish_pointcloud_
                    << ", show_color_window=" << show_color_window_
                    << ", fps=" << fps_
                    << ", color_format=" << colorFormatName(color_format_)
                    << ", publish_every_n_frames=" << publish_every_n_frames_
                    << ", pointcloud_stride=" << pointcloud_stride_
                    << ", wait_timeout_ms=" << wait_timeout_ms_
                    << ", serial='" << (serial_no_.empty() ? "auto" : serial_no_) << "'");
  }

  void spin()
  {
    ros::Rate idle_rate(100);
    while (ros::ok())
    {
      try
      {
        rs2::frameset frames = pipeline_.wait_for_frames(wait_timeout_ms_);
        if (align_depth_to_color_ && enable_color_ && enable_depth_)
        {
          frames = align_to_color_.process(frames);
        }

        ++frame_count_;
        if ((frame_count_ - 1) % publish_every_n_frames_ != 0)
        {
          ros::spinOnce();
          continue;
        }

        const ros::Time stamp = ros::Time::now();
        rs2::video_frame color_frame = frames.get_color_frame();
        rs2::depth_frame depth_frame = frames.get_depth_frame();

        if (enable_color_ && color_frame)
        {
          handleColor(color_frame, stamp);
        }
        if (enable_depth_ && depth_frame)
        {
          if (publish_depth_topic_)
          {
            publishDepth(depth_frame, stamp);
          }
          if (publish_pointcloud_)
          {
            publishPointCloud(depth_frame, stamp);
          }
        }
      }
      catch (const rs2::error& e)
      {
        ROS_WARN_STREAM_THROTTLE(2.0, "moca_realsense: RealSense error: " << e.what());
        idle_rate.sleep();
      }

      ros::spinOnce();
      if (show_color_window_)
      {
        cv::waitKey(1);
      }
    }
  }

 private:
  bool captureImageService(moca_realsense::CaptureImage::Request&,
                           moca_realsense::CaptureImage::Response& response)
  {
    cv::Mat rgb;
    ros::Time stamp;
    {
      std::lock_guard<std::mutex> lock(latest_color_mutex_);
      if (latest_color_rgb_.empty())
      {
        response.success = false;
        response.error_message = "No color frame has been received yet.";
        return true;
      }
      rgb = latest_color_rgb_.clone();
      stamp = latest_color_stamp_;
    }

    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    std::vector<uint8_t> encoded;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, capture_jpeg_quality_};
    if (!cv::imencode(".jpg", bgr, encoded, params))
    {
      response.success = false;
      response.error_message = "Failed to encode latest color frame as JPEG.";
      return true;
    }

    response.image.header.stamp = stamp;
    response.image.header.frame_id = color_frame_id_;
    response.image.format = "jpeg";
    response.image.data = std::move(encoded);
    response.success = true;
    response.error_message.clear();
    return true;
  }

  void handleColor(const rs2::video_frame& frame, const ros::Time& stamp)
  {
    cv::Mat rgb_copy = colorFrameToRgb(frame);
    if (rgb_copy.empty())
    {
      ROS_WARN_THROTTLE(2.0, "moca_realsense: received unsupported color frame format.");
      return;
    }

    {
      std::lock_guard<std::mutex> lock(latest_color_mutex_);
      latest_color_rgb_ = rgb_copy;
      latest_color_stamp_ = stamp;
    }

    if (show_color_window_)
    {
      cv::Mat bgr;
      cv::cvtColor(rgb_copy, bgr, cv::COLOR_RGB2BGR);
      if (std::abs(color_window_scale_ - 1.0) > 1e-6)
      {
        cv::Mat resized;
        cv::resize(bgr, resized, cv::Size(), color_window_scale_, color_window_scale_);
        cv::imshow(color_window_name_, resized);
      }
      else
      {
        cv::imshow(color_window_name_, bgr);
      }
    }

    if (publish_color_topic_)
    {
      publishColor(frame, rgb_copy, stamp);
    }
  }

  cv::Mat colorFrameToRgb(const rs2::video_frame& frame)
  {
    const auto* data = reinterpret_cast<const uint8_t*>(frame.get_data());
    const int height = frame.get_height();
    const int width = frame.get_width();
    const size_t stride = static_cast<size_t>(frame.get_stride_in_bytes());
    const rs2_format format = frame.get_profile().format();

    if (format == RS2_FORMAT_RGB8)
    {
      return cv::Mat(height, width, CV_8UC3, const_cast<uint8_t*>(data), stride).clone();
    }
    if (format == RS2_FORMAT_BGR8)
    {
      cv::Mat bgr(height, width, CV_8UC3, const_cast<uint8_t*>(data), stride);
      cv::Mat rgb;
      cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
      return rgb;
    }
    if (format == RS2_FORMAT_YUYV)
    {
      cv::Mat yuyv(height, width, CV_8UC2, const_cast<uint8_t*>(data), stride);
      cv::Mat rgb;
      cv::cvtColor(yuyv, rgb, cv::COLOR_YUV2RGB_YUYV);
      return rgb;
    }
    return cv::Mat();
  }

  void publishColor(const rs2::video_frame& frame, const cv::Mat& rgb, const ros::Time& stamp)
  {
    sensor_msgs::Image msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = color_frame_id_;
    msg.height = static_cast<uint32_t>(rgb.rows);
    msg.width = static_cast<uint32_t>(rgb.cols);
    msg.encoding = "rgb8";
    msg.is_bigendian = false;
    msg.step = msg.width * 3;
    msg.data.assign(rgb.datastart, rgb.dataend);
    color_pub_.publish(msg);

    const auto intr = frame.get_profile().as<rs2::video_stream_profile>().get_intrinsics();
    color_info_pub_.publish(cameraInfoFromIntrinsics(intr, color_frame_id_, stamp));
  }

  void publishDepth(const rs2::depth_frame& frame, const ros::Time& stamp)
  {
    const std::string depth_frame_id = align_depth_to_color_ && enable_color_ ? color_frame_id_ : depth_frame_id_;
    sensor_msgs::Image msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = depth_frame_id;
    msg.height = static_cast<uint32_t>(frame.get_height());
    msg.width = static_cast<uint32_t>(frame.get_width());
    msg.encoding = "16UC1";
    msg.is_bigendian = false;
    msg.step = msg.width * sizeof(uint16_t);
    const auto* data = reinterpret_cast<const uint8_t*>(frame.get_data());
    msg.data.assign(data, data + msg.step * msg.height);
    depth_pub_.publish(msg);

    const auto intr = frame.get_profile().as<rs2::video_stream_profile>().get_intrinsics();
    depth_info_pub_.publish(cameraInfoFromIntrinsics(intr, depth_frame_id, stamp));
  }

  void publishPointCloud(const rs2::depth_frame& depth_frame, const ros::Time& stamp)
  {
    const auto intr = depth_frame.get_profile().as<rs2::video_stream_profile>().get_intrinsics();
    const int out_width = (depth_frame.get_width() + pointcloud_stride_ - 1) / pointcloud_stride_;
    const int out_height = (depth_frame.get_height() + pointcloud_stride_ - 1) / pointcloud_stride_;
    const std::string cloud_frame_id = align_depth_to_color_ && enable_color_ ? color_frame_id_ : depth_frame_id_;

    sensor_msgs::PointCloud2 cloud;
    cloud.header.stamp = stamp;
    cloud.header.frame_id = cloud_frame_id;
    cloud.height = static_cast<uint32_t>(out_height);
    cloud.width = static_cast<uint32_t>(out_width);
    cloud.is_dense = false;

    sensor_msgs::PointCloud2Modifier modifier(cloud);
    modifier.setPointCloud2FieldsByString(1, "xyz");
    modifier.resize(out_width * out_height);

    sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(cloud, "z");
    const float nan = std::numeric_limits<float>::quiet_NaN();

    for (int y = 0; y < depth_frame.get_height(); y += pointcloud_stride_)
    {
      for (int x = 0; x < depth_frame.get_width(); x += pointcloud_stride_)
      {
        const float depth_m = depth_frame.get_distance(x, y);
        if (depth_m > 0.0f && std::isfinite(depth_m))
        {
          float pixel[2] = {static_cast<float>(x), static_cast<float>(y)};
          float point[3] = {0.0f, 0.0f, 0.0f};
          rs2_deproject_pixel_to_point(point, &intr, pixel, depth_m);
          *iter_x = point[0];
          *iter_y = point[1];
          *iter_z = point[2];
        }
        else
        {
          *iter_x = nan;
          *iter_y = nan;
          *iter_z = nan;
        }
        ++iter_x;
        ++iter_y;
        ++iter_z;
      }
    }

    pointcloud_pub_.publish(cloud);
  }

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::Publisher color_pub_;
  ros::Publisher color_info_pub_;
  ros::Publisher depth_pub_;
  ros::Publisher depth_info_pub_;
  ros::Publisher pointcloud_pub_;
  ros::ServiceServer capture_service_;
  tf2_ros::StaticTransformBroadcaster static_tf_broadcaster_;

  rs2::pipeline pipeline_;
  rs2::pipeline_profile profile_;
  rs2::align align_to_color_;

  std::string serial_no_;
  std::string frame_id_;
  std::string color_frame_id_;
  std::string depth_frame_id_;
  int color_width_{640};
  int color_height_{480};
  int depth_width_{640};
  int depth_height_{480};
  int fps_{30};
  std::string color_format_name_{"RGB8"};
  rs2_format color_format_{RS2_FORMAT_RGB8};
  int pointcloud_stride_{4};
  int publish_every_n_frames_{1};
  int wait_timeout_ms_{1000};
  int capture_jpeg_quality_{90};
  uint64_t frame_count_{0};
  bool enable_color_{true};
  bool enable_depth_{true};
  bool align_depth_to_color_{true};
  bool publish_color_topic_{true};
  bool publish_depth_topic_{true};
  bool publish_pointcloud_{true};
  bool publish_tf_{true};
  bool show_color_window_{false};
  double color_window_scale_{1.0};
  std::string color_window_name_{"MOCA RealSense"};
  std::mutex latest_color_mutex_;
  cv::Mat latest_color_rgb_;
  ros::Time latest_color_stamp_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "moca_realsense_node");
  try
  {
    MocaRealSenseNode node;
    node.start();
    node.spin();
  }
  catch (const rs2::error& e)
  {
    ROS_FATAL_STREAM("moca_realsense: failed to start RealSense pipeline: " << e.what());
    return 1;
  }
  catch (const std::exception& e)
  {
    ROS_FATAL_STREAM("moca_realsense: failed: " << e.what());
    return 1;
  }
  return 0;
}
