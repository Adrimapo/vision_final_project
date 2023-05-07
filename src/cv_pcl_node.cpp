/*
Autor: Adrián Madinabeitia Portanova
Partes implementadas:
- Detección de pelota en 2D y proyección 3D
- Detección de pelota en 3D y proyección 2D
- Proyección líneas
- Funcionalidad extra:
  - Proyección de la pelota de 3D a 2D teniendo en cuenta el radio calculado en 3D y
    dibujarlo sobre la imagen (medio).
  - Aplicar K-means en 3D (avanzado).
  
*/

#include <chrono>
#include <functional>
#include <memory>
#include <string.h>
#include <stdlib.h>
#include <thread>
#include <iostream>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <image_transport/image_transport.hpp>
#include <image_geometry/pinhole_camera_model.h>
#include <Eigen/Dense>

#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/sample_consensus/ransac.h>

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2/exceptions.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"

#include "cv_bridge/cv_bridge.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "std_msgs/msg/u_int16_multi_array.hpp"
#include "std_msgs/msg/int32_multi_array.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"

#include <opencv2/objdetect.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core.hpp>

#include <pcl/ml/kmeans.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/distances.h>


// Window parameters
#define WINDOW_NAME "PRACTICA_FINAL"

// ** Trackbar params
#define LIMIT_1 2
#define LIMIT_2 8
#define LIMIT_3 10
#define INIT_VAL_1 0
#define INIT_VAL_2 0
#define INIT_VAL_3 1

// 2D parameters
#define CIRCLE_RAD 4
#define THICKNESS 2
#define CV_CIRCLE_RADIUS 4 
#define TEXT_DIST 7

// 3D parameters
#define MIN_PCL 400   // Min pcl particles in a ball

// Distance lines parameters
#define R_WIDHT 1     // Width of distnace points
#define MIN_DIST 3
#define MAX_DIST 8


pcl::PointCloud<pcl::PointXYZRGB> pcl_processing(const pcl::PointCloud<pcl::PointXYZRGB> in_pointcloud, 
  int mode, int dist, int k_fil, std::vector<std::vector<float>> centers2d, 
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr center_publisher);

cv::Mat image_processing(cv::Mat in_image, cv::Mat depth_image, int mode, int dist, std::vector<cv::Vec3f> centers3d,
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr center_publisher,  std::vector<cv::Rect> people);

Eigen::Matrix<float, 3, 4> T_matrix_pcl;

cv::Mat T_matrix_cv;
image_geometry::PinholeCameraModel camera_model;

//int pinks[6];



std::vector<cv::Rect> detect_people(cv::Mat img)
{
  cv::HOGDescriptor hog;
  hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
  std::vector<cv::Rect> found, found_filtered;

  hog.detectMultiScale(img, found, 0, cv::Size(8, 8), cv::Size(20, 20), 1.05, 2); 

  size_t i, j;
  for (i = 0; i < found.size(); i++) {

    cv::Rect r = found[i];

    for (j = 0; j < found.size(); j++)
      if (j != i && (r & found[j]) == r)
        break;
    
    if (j == found.size())
      found_filtered.push_back(r);
  }
  
  return found_filtered;
}



class pclNode : public rclcpp::Node
{
  public:
    pclNode()
    : Node("pcl_subscriber")
    {
      auto qos = rclcpp::QoS( rclcpp::QoSInitialization( RMW_QOS_POLICY_HISTORY_KEEP_LAST, 5 ));
      qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);

      subscription_3d_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "head_front_camera/depth_registered/points", qos, std::bind(&pclNode::topic_callback_3d, this, std::placeholders::_1));

      publisher_3d_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "pcl_points_filtered", qos);

      trace_bar_sub_ = this->create_subscription<std_msgs::msg::Int32MultiArray>(
        "/cv_tracebar", qos, std::bind(&pclNode::trace_bar_callback, this, std::placeholders::_1));

      center_2d_sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
        "/cv_centers", qos, std::bind(&pclNode::centers_2d_callback, this, std::placeholders::_1));

      pub_center_3d_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(
        "pcl_centers", qos);

      tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
      tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

      timer_ = this->create_wall_timer(std::chrono::milliseconds(50), std::bind(&pclNode::on_timer, this));
    }

  private:

    static Eigen::Matrix<float, 3, 4> convert_to_matrix(geometry_msgs::msg::TransformStamped t) 
    {
      const auto& quat = t.transform.rotation;
      const auto& translation = t.transform.translation;

      // Quaternions to rotation matrix
      Eigen::Quaterniond q_vect(quat.w, quat.x, quat.y, quat.z);
      Eigen::Matrix3d r_mat = q_vect.toRotationMatrix();

      // Translation and rotation matrix
      Eigen::Matrix<float, 3, 4> T;

      T <<  r_mat(0, 0), r_mat(0, 1), r_mat(0, 2), translation.x,
            r_mat(1, 0), r_mat(1, 1), r_mat(1, 2), translation.y,
            r_mat(2, 0), r_mat(2, 1), r_mat(2, 2), translation.z;

      return T;
    }

    void on_timer() 
    {
      geometry_msgs::msg::TransformStamped t;

      try {
        t = tf_buffer_->lookupTransform("head_front_camera_rgb_optical_frame", "base_footprint", tf2::TimePointZero);

        } catch (const tf2::TransformException & ex) {
          RCLCPP_INFO(this->get_logger(), "Could not transform");
        }

        T_matrix_pcl = convert_to_matrix(t);
    }

    void info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) const
    {
      // Gets the K matrix
      camera_model.fromCameraInfo(msg);
    }

    void trace_bar_callback(const std_msgs::msg::Int32MultiArray msg) const
    {
      mode_ = static_cast<int>(msg.data[0]);
      dist_ = static_cast<int>(msg.data[1]);
      k_fil_ = static_cast<int>(msg.data[2]);
    }

    void topic_callback_3d(const sensor_msgs::msg::PointCloud2::SharedPtr msg) const
    {    
      
      // Convert to PCL data type
      pcl::PointCloud<pcl::PointXYZRGB> point_cloud;
      pcl::fromROSMsg(*msg, point_cloud);     

      // PCL Processing
      pcl::PointCloud<pcl::PointXYZRGB> pcl_pointcloud = pcl_processing(point_cloud, mode_, dist_, k_fil_, centers2d_, pub_center_3d_);
      
      // Convert to ROS data type
      sensor_msgs::msg::PointCloud2 output;
      pcl::toROSMsg(pcl_pointcloud, output);
      output.header = msg->header;

      // Publish the data
      publisher_3d_ -> publish(output);
    }

    void centers_2d_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg) const
    {
      centers2d_.clear();
      std::vector<float> subvector;

      for (int i = 1 ; i <= msg->data[0]*3 ; i++) {
        subvector.push_back(msg->data[i]);

        if (subvector.size() == 3) {
          centers2d_.push_back(subvector);
          subvector.clear();
        }
      }
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_3d_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_3d_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
    rclcpp::Subscription<std_msgs::msg::Int32MultiArray>::SharedPtr trace_bar_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr center_2d_sub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pub_center_3d_;

    // Tfs
    rclcpp::TimerBase::SharedPtr timer_{nullptr};
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};

    mutable int mode_;
    mutable int dist_;
    mutable int k_fil_;
    mutable std::vector<std::vector<float>> centers2d_;
};




class cvNode : public rclcpp::Node
{

  public:

    cvNode()
    : Node("opencv_node")
    {
      auto qos = rclcpp::QoS( rclcpp::QoSInitialization( RMW_QOS_POLICY_HISTORY_KEEP_LAST, 5 ));
      qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
   
      subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/head_front_camera/rgb/image_raw", qos, std::bind(&cvNode::topic_callback, this, std::placeholders::_1));

      raw_img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/head_front_camera/depth_registered/image_raw", qos, std::bind(&cvNode::deph_img_callback, this, std::placeholders::_1));

      cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      "/head_front_camera/rgb/camera_info", qos, std::bind(&cvNode::info_callback, this, std::placeholders::_1));

      /* trace_bar_sub_ = this->create_subscription<std_msgs::msg::UInt16MultiArray>(
      "/trace_bar", qos, std::bind(&cvNode::trace_bar_callback, this, std::placeholders::_1)); */

      publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
      "cv_image", qos);

      mode_pub_ = this->create_publisher<std_msgs::msg::Int32MultiArray>(
        "cv_tracebar", qos);
      
      cv_center_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(
        "cv_centers", qos);
      
      center_3d_sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
        "/pcl_centers", qos, std::bind(&cvNode::centers_3d_callback, this, std::placeholders::_1));


      tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
      tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

      timer_ = this->create_wall_timer(std::chrono::seconds(1), std::bind(&cvNode::on_timer, this));
    }

    static cv::Mat convert_to_matrix(geometry_msgs::msg::TransformStamped t) {
      const auto& quat = t.transform.rotation;
      const auto& translation = t.transform.translation;

      // Quaternions to rotation matrix
      Eigen::Quaterniond q_vect(quat.w, quat.x, quat.y, quat.z);
      Eigen::Matrix3d r_mat = q_vect.toRotationMatrix();

      // Translation and rotation matrix
      cv::Matx<double, 3, 4> T(
        r_mat(0, 0), r_mat(0, 1), r_mat(0, 2), translation.x,
        r_mat(1, 0), r_mat(1, 1), r_mat(1, 2), translation.y,
        r_mat(2, 0), r_mat(2, 1), r_mat(2, 2), translation.z);

      return cv::Mat(T);
    }

  private:
    void on_timer() 
    {
      geometry_msgs::msg::TransformStamped t;

      try {
        t = tf_buffer_->lookupTransform(camera_model.tfFrame(), "base_footprint", tf2::TimePointZero);

        } catch (const tf2::TransformException & ex) {
          RCLCPP_INFO(this->get_logger(), "Could not transform ");
        }

        T_matrix_cv = convert_to_matrix(t);
    }

    void info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) const
    {
      // Gets the K matrix
      camera_model.fromCameraInfo(msg);
    }

    // Gets the depth image from the topic
    void deph_img_callback(const sensor_msgs::msg::Image::SharedPtr msg) const
    {  
      cv_bridge::CvImageConstPtr cv_ptr;
      try
      {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
      }
      catch (const cv_bridge::Exception& e)
      {
        return;
      }
      depth_image_ = cv_ptr->image;
      cv::waitKey(1);
    }

    void topic_callback(const sensor_msgs::msg::Image::SharedPtr msg) const
    {     
      // Convert ROS Image to CV Image
      cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
      cv::Mat image_raw =  cv_ptr->image;

      // Sets the window
      if (first_exe_) {

        cv::namedWindow(WINDOW_NAME); 
  
        cv::createTrackbar(TEXT[0], WINDOW_NAME, nullptr, LIMIT_1, 0);
        cv::setTrackbarPos(TEXT[0], WINDOW_NAME, INIT_VAL_1);

        cv::createTrackbar(TEXT[1], WINDOW_NAME, nullptr, LIMIT_2, 0);
        cv::setTrackbarPos(TEXT[1], WINDOW_NAME, INIT_VAL_2);

        cv::createTrackbar(TEXT[2], WINDOW_NAME, nullptr, LIMIT_3, 0);
        cv::setTrackbarPos(TEXT[2], WINDOW_NAME, INIT_VAL_3);
        first_exe_ = false;
      }

      // Gets the trackbar parameters
      int mode = cv::getTrackbarPos(TEXT[0], WINDOW_NAME); 
      int dist = cv::getTrackbarPos(TEXT[1], WINDOW_NAME); 
      int k_val = cv::getTrackbarPos(TEXT[2], WINDOW_NAME); 

      std::vector<cv::Rect> people;
      if (mode == 1) {
        people = detect_people(image_raw);
        if (people.empty()) {
          mode = 0;
        }
      }

      // Publish the tracebar data
      std_msgs::msg::Int32MultiArray trace_info;
      trace_info.data = {mode, dist, k_val};
      mode_pub_ -> publish(trace_info);

      // Image processing
      cv::Mat cv_image = image_processing(image_raw, depth_image_, mode, dist, centers3d_, cv_center_pub_, people);

      // Convert OpenCV Image to ROS Image
      cv_bridge::CvImage img_bridge = cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::BGR8, cv_image);
      sensor_msgs::msg::Image out_image; // >> message to be sent
      img_bridge.toImageMsg(out_image); // from cv_bridge to sensor_msgs::Image

      // Publish the data
      publisher_ -> publish(out_image);

      cv::waitKey(1);
    }

  /*void trace_bar_callback(const std_msgs::msg::UInt16MultiArray msg) const
    {
      for (size_t i = 0; i < msg.data.size(); i++) {
        pinks[i] = static_cast<int>(msg.data[i]);
      }
    } */

    void centers_3d_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg) const
    {
      centers3d_.clear();

      // Theres no Tf changes
      cv::Matx<double, 3, 4> I(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0);

      // Num of spheres * elements per sphere
      for (int i = 1 ; i <= msg->data[0]*4 ; i+=4) {
        // Sphere center
        cv::Mat p3d = (cv::Mat_<double>(4,1) << msg->data[i], msg->data[i+1], msg->data[i+2], 1);
        cv::Mat p2d = cv::Mat(camera_model.intrinsicMatrix()) * cv::Mat(I) * p3d;

        p2d.at<double>(0) = p2d.at<double>(0) / p2d.at<double>(2);
        p2d.at<double>(1) = p2d.at<double>(1) / p2d.at<double>(2);

        // 2d point of the radius
        cv::Mat pr3d = (cv::Mat_<double>(4,1) << msg->data[i] + msg->data[i+3], msg->data[i+1], msg->data[i+2], 1);
        cv::Mat pr2d = cv::Mat(camera_model.intrinsicMatrix()) * cv::Mat(I) * pr3d;

        pr2d.at<double>(0) = pr2d.at<double>(0) / pr2d.at<double>(2);
        pr2d.at<double>(1) = pr2d.at<double>(1) / pr2d.at<double>(2);

        double norma = cv::norm(pr2d.rowRange(0, 2) - p2d.rowRange(0, 2), cv::NORM_L2);

        centers3d_.push_back(cv::Vec3f(p2d.at<double>(0), p2d.at<double>(1), norma));
      }
    }

    // Publishers and subscribers
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr raw_img_sub_;

    // Comunication with pcl node
    rclcpp::Publisher<std_msgs::msg::Int32MultiArray>::SharedPtr mode_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr cv_center_pub_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr center_3d_sub_;

    //rclcpp::Subscription<std_msgs::msg::UInt16MultiArray>::SharedPtr trace_bar_sub_;

    // Tfs
    rclcpp::TimerBase::SharedPtr timer_{nullptr};
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};

    mutable cv::Mat depth_image_;

    mutable bool first_exe_ = true;
    mutable std::vector<cv::Vec3f> centers3d_;
    std::string TEXT[3] = {"Option", "Distance", "K-means"};
};






/************************************** PCL node*****************************************************/
pcl::PointCloud<pcl::PointXYZRGB> get_pink_spheres(pcl::PointCloud<pcl::PointXYZRGB> original_cloud) {
  pcl::PointCloud<pcl::PointXYZRGB> cloud;
  cloud = original_cloud;

  const int pink_min_r = 0;
  const int pink_max_r = 255;
  const int pink_min_g = 0;
  const int pink_max_g = 55;
  const int pink_min_b = 80;
  const int pink_max_b = 266;

  // Define the color range
  pcl::ConditionAnd<pcl::PointXYZRGB>::Ptr color_cond(new pcl::ConditionAnd<pcl::PointXYZRGB>());

  color_cond->addComparison(pcl::PackedRGBComparison<pcl::PointXYZRGB>::Ptr(
      new pcl::PackedRGBComparison<pcl::PointXYZRGB>("r", pcl::ComparisonOps::GE, pink_min_r)));
  color_cond->addComparison(pcl::PackedRGBComparison<pcl::PointXYZRGB>::Ptr(
      new pcl::PackedRGBComparison<pcl::PointXYZRGB>("r", pcl::ComparisonOps::LE, pink_max_r)));
  color_cond->addComparison(pcl::PackedRGBComparison<pcl::PointXYZRGB>::Ptr(
      new pcl::PackedRGBComparison<pcl::PointXYZRGB>("g", pcl::ComparisonOps::GE, pink_min_g)));
  color_cond->addComparison(pcl::PackedRGBComparison<pcl::PointXYZRGB>::Ptr(
      new pcl::PackedRGBComparison<pcl::PointXYZRGB>("g", pcl::ComparisonOps::LE, pink_max_g)));
  color_cond->addComparison(pcl::PackedRGBComparison<pcl::PointXYZRGB>::Ptr(
      new pcl::PackedRGBComparison<pcl::PointXYZRGB>("b", pcl::ComparisonOps::GE, pink_min_b)));
  color_cond->addComparison(pcl::PackedRGBComparison<pcl::PointXYZRGB>::Ptr(
      new pcl::PackedRGBComparison<pcl::PointXYZRGB>("b", pcl::ComparisonOps::LE, pink_max_b)));

  // Applys the color filter
  pcl::ConditionalRemoval<pcl::PointXYZRGB> color_filter;
  color_filter.setInputCloud(cloud.makeShared());
  color_filter.setCondition(color_cond);
  color_filter.filter(cloud);

  // Create the filtering object
  pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> statistRemove;
  statistRemove.setInputCloud (cloud.makeShared());
  statistRemove.setMeanK (70);
  statistRemove.setStddevMulThresh (1);
  statistRemove.filter (cloud);

  return cloud;
}


std::vector<std::vector<float>> get_sphere_centers(pcl::PointCloud<pcl::PointXYZRGB> original_cloud)
{
  std::vector<std::vector<float>> spheres;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::copyPointCloud(original_cloud, *cloud_filtered);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_p (new pcl::PointCloud<pcl::PointXYZ>), cloud_f (new pcl::PointCloud<pcl::PointXYZ>);

  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());

  // Create the segmentation object for sphere segmentation
  pcl::SACSegmentation<pcl::PointXYZ> seg_sphere;
  pcl::PointIndices::Ptr inliers_sphere (new pcl::PointIndices ());
  pcl::ModelCoefficients::Ptr coefficients_sphere (new pcl::ModelCoefficients ());

  // Configure the sphere segmentation parameters
  seg_sphere.setOptimizeCoefficients (true);
  seg_sphere.setModelType (pcl::SACMODEL_SPHERE);
  seg_sphere.setMethodType (pcl::SAC_RANSAC);
  seg_sphere.setMaxIterations (1000); 
  seg_sphere.setDistanceThreshold (0.01);

  pcl::ExtractIndices<pcl::PointXYZ> extract_sphere;

  while (cloud_filtered->size() > MIN_PCL) { 
    std::vector<float> new_ball;

    // Segment the sphere
    seg_sphere.setInputCloud(cloud_filtered);
    seg_sphere.segment(*inliers, *coefficients);
    if (inliers->indices.size() == 0) {
      std::cerr << "Could not estimate a spherical model for the given dataset." << std::endl;
      break;
    }

    // Extract the sphere
    extract_sphere.setInputCloud(cloud_filtered);
    extract_sphere.setIndices(inliers);
    extract_sphere.setNegative(false);
    extract_sphere.filter(*cloud_p);

    // Saves the coefficients
    new_ball.push_back(coefficients->values[0]);  // x val
    new_ball.push_back(coefficients->values[1]);  // y val
    new_ball.push_back(coefficients->values[2]);  // z val
    new_ball.push_back(coefficients->values[3]);  // radius

    // Remove the sphere from the cloud
    extract_sphere.setNegative(true);
    extract_sphere.filter(*cloud_f);
    cloud_filtered.swap(cloud_f);

    spheres.push_back(new_ball);
  }

  return spheres;
}


pcl::PointCloud<pcl::PointXYZRGB> draw_square(pcl::PointCloud<pcl::PointXYZRGB> original_cloud, 
  float x, float y, float z, std::array<int, 3> color) 
{
  pcl::PointCloud<pcl::PointXYZRGB> cloud;
  pcl::copyPointCloud(original_cloud, cloud);

  pcl::PointXYZRGB point;
  float dim = 0.1;
  float dist = 0.01;  // Distance between each point

  for (float i = x - dim/2; i < x + dim/2; i += dist) {
    for (float j = y - dim/2; j < y + dim/2; j += dist) {
      for (float k = z - dim/2; k < z + dim/2; k += dist) {
        
        // Creates the point
        point.r = color[0]; point.g = color[1]; point.b = color[2];
        point.x = i; point.y = j; point.z =  k;

        cloud.push_back(point);
      }
    }
  }
  return cloud;
}


pcl::PointCloud<pcl::PointXYZRGB> draw_dist_cubes(pcl::PointCloud<pcl::PointXYZRGB> original_cloud, 
  int min, int max, std::vector<std::array<int, 3>> colors)
{

  pcl::PointCloud<pcl::PointXYZRGB> cloud;
  pcl::copyPointCloud(original_cloud, cloud);

  for (int i = min; i <= max; i++) {

    Eigen::Matrix<float, 4, 1> point1, point2;
    point1 <<  i, R_WIDHT, 0, 1;
    point2 <<  i, -R_WIDHT, 0, 1;

    // Base_foot_print to camera tf
    Eigen::Matrix<float, 3, 1> t_p1 = T_matrix_pcl * point1;
    Eigen::Matrix<float, 3, 1> t_p2 = T_matrix_pcl * point2;

    cloud = draw_square(cloud, t_p1(0), t_p1(1), t_p1(2),  colors[i - min]);
    cloud = draw_square(cloud, t_p2(0), t_p2(1), t_p2(2),  colors[i - min]);
  }

  return cloud;  
}

void  publish_3d_centers(std::vector<std::vector<float>> spheres, 
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr center_publisher)  
{
  auto message = std_msgs::msg::Float32MultiArray();
  std::vector<float> pub_msg;

  pub_msg.push_back(spheres.size());

  for (uint i = 0; i < spheres.size(); i++) {
    pub_msg.push_back(spheres[i][0]);   // x
    pub_msg.push_back(spheres[i][1]);   // y
    pub_msg.push_back(spheres[i][2]);   // z
    pub_msg.push_back(spheres[i][3]);   // r
  }

  message.data = pub_msg;
  center_publisher->publish(message);
}

pcl::Kmeans::Centroids get_centers_k_mean(pcl::PointCloud<pcl::PointXYZRGB>::Ptr tempCloud, int n_balls) 
{
  pcl::Kmeans real(static_cast<int> (tempCloud->points.size()), 3);
  real.setClusterSize(n_balls); 

  for (size_t i = 0; i < tempCloud->points.size(); i++) {
    std::vector<float> data(3);
    data[0] = tempCloud->points[i].x;
    data[1] = tempCloud->points[i].y;
    data[2] = tempCloud->points[i].z;
    real.addDataPoint(data);
  }

  real.kMeans();
  
  // get the cluster centroids 
  pcl::Kmeans::Centroids centroids = real.get_centroids();

  return centroids;
}

pcl::PointCloud<pcl::PointXYZRGB> color_point_cloud_by_centroids(pcl::PointCloud<pcl::PointXYZRGB> original_cloud, 
  pcl::Kmeans::Centroids centroids) 
{
  pcl::PointCloud<pcl::PointXYZRGB> cloud;
  cloud = original_cloud;
  
  srand(0); 
  
  // Creates the colors  
  std::vector<pcl::RGB> colors(centroids.size());
  for (size_t i = 0; i < centroids.size(); i++) {
    colors[i].r = static_cast<uint8_t>(std::rand() % 256);
    colors[i].g = static_cast<uint8_t>(std::rand() % 256);
    colors[i].b = static_cast<uint8_t>(std::rand() % 256);
  }

  // Checks each point
  for (size_t i = 0; i < cloud.points.size(); i++) { 
    
    // pcl::PointXYZRGB to pcl::PointXYZ
    pcl::PointXYZ point;
    point.x = cloud.points[i].x;
    point.y = cloud.points[i].y;
    point.z = cloud.points[i].z;

    float dist = 1000;
    int nearest_centroid = 0;

    for (size_t j = 0; j < centroids.size(); j++) {
      pcl::PointXYZ centroid;
      centroid.x = centroids[j][0];
      centroid.y = centroids[j][1];
      centroid.z = centroids[j][2];

      float new_dist = pcl::euclideanDistance(point, centroid);

      if(new_dist < dist) {
        dist = new_dist;
        nearest_centroid = j;
      }
    }

    // Changes the point color
    cloud.points[i].r = colors[nearest_centroid].r;
    cloud.points[i].g = colors[nearest_centroid].g;
    cloud.points[i].b = colors[nearest_centroid].b;
  }

  return cloud;
}

pcl::PointCloud<pcl::PointXYZRGB> pcl_processing(const pcl::PointCloud<pcl::PointXYZRGB> in_pointcloud, int mode, int dist, int k_fil,
  std::vector<std::vector<float>> centers2d, rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr center_publisher)
{ 
  // Colors
  std::vector<std::array<int, 3>> colors{{255, 0, 0}, {134, 40, 40}, {135, 65, 60}, 
    {38, 104, 28}, {0, 255, 0}, {12, 91, 0}};

  std::array<int, 3> blue{0, 0, 255}; 
  std::array<int, 3> black{0, 0, 0};
  std::array<int, 3> purple{160, 32, 240};

  // Create output pointcloud
  pcl::PointCloud<pcl::PointXYZRGB> out_pointcloud;
  out_pointcloud = in_pointcloud;

  switch (mode) {

    case 1:
    {
      // Get the filtered pink spheres
      out_pointcloud = get_pink_spheres(out_pointcloud);

      if (out_pointcloud.size() < MIN_PCL) {
        std::cerr << "No spheres detected" << std::endl;
        out_pointcloud = draw_dist_cubes(in_pointcloud, MIN_DIST, MAX_DIST, colors);
        return out_pointcloud;
      }

      // Gets the center of each sphere
      std::vector<std::vector<float>> spheres = get_sphere_centers(out_pointcloud);
      publish_3d_centers(spheres, center_publisher);

      // Draws a square in each sphere center
      for (uint i = 0; i < spheres.size(); i++) {
        out_pointcloud = draw_square(out_pointcloud, 
          spheres[i][0], spheres[i][1], spheres[i][2],  blue);
      }

      // Draws 2d centers
      for (uint i = 0; i < centers2d.size(); i++) {
        out_pointcloud = draw_square(out_pointcloud, 
          centers2d[i][0], centers2d[i][1], centers2d[i][2],  black);
      }

      out_pointcloud = draw_dist_cubes(out_pointcloud, MIN_DIST, dist, colors); 
      break;
    }

    case 2:
    {
      out_pointcloud = get_pink_spheres(out_pointcloud);

      if (out_pointcloud.size() < MIN_PCL) {
        std::cerr << "No balls detected" << std::endl;
        out_pointcloud = draw_dist_cubes(in_pointcloud, MIN_DIST, MAX_DIST, colors);
        return out_pointcloud;
      }

      // Publish the spheres for the 3D to 2D extra
      std::vector<std::vector<float>> spheres = get_sphere_centers(out_pointcloud);
      publish_3d_centers(spheres, center_publisher);

      if (k_fil > 0) {
        // Gets the centers with the k mean method
        pcl::Kmeans::Centroids centroids = get_centers_k_mean(out_pointcloud.makeShared(), k_fil);

        out_pointcloud = color_point_cloud_by_centroids(out_pointcloud, centroids);

        // Draws the centroids
        for (size_t i = 0; i < centroids.size(); i++) {
          out_pointcloud = draw_square(out_pointcloud, 
            centroids[i][0], centroids[i][1], centroids[i][2],  purple);
        }
      }
    }
    break;

    default:
      break;

  }

  return out_pointcloud;
}



/************************************** CV node *****************************************************/
std::vector<double> get_point_dist(cv::Point point, cv::Mat depth_image) 
{
  // Intrinsic parametes
  double fx = cv::Mat(camera_model.intrinsicMatrix()).at<double>(0, 0);
  double fy = cv::Mat(camera_model.intrinsicMatrix()).at<double>(1, 1);
  double cx = cv::Mat(camera_model.intrinsicMatrix()).at<double>(0, 2);
  double cy = cv::Mat(camera_model.intrinsicMatrix()).at<double>(1, 2);

  float d = depth_image.at<float>((int)point.y, (int)point.x);

  // 2d to 3d
  double x = (point.x - cx) * d / fx;
  double y = (point.y - cy) * d / fy;

  std::vector<double> coords{x, y, d};
  return coords;
}

std::vector<cv::Point> draw_dist_lines(cv::Mat img, int dist) 
{
  std::vector<cv::Point> points;

  // Color array for each element
  std::vector<cv::Scalar> colors = {
    cv::Scalar(0, 0, 0), cv::Scalar(0, 0, 0), cv::Scalar(0, 0, 0),
    cv::Scalar(0, 0, 255), cv::Scalar(0, 0, 153), cv::Scalar(0, 102, 51), 
    cv::Scalar(0, 102, 0), cv::Scalar(0, 153, 0), cv::Scalar(0, 255, 0)};
  
  // Draws a line from 1 to 8
  for (int i = 0; i <= dist; i++) {
    cv::Mat p3d_left = (cv::Mat_<double>(4,1) << i, R_WIDHT, 0, 1);
    cv::Mat p3d_right = (cv::Mat_<double>(4,1) << i, -R_WIDHT, 0, 1);

    cv::Mat p2d_left = (cv::Mat(camera_model.intrinsicMatrix()) * cv::Mat(T_matrix_cv)) * p3d_left;
    cv::Mat p2d_right = (cv::Mat(camera_model.intrinsicMatrix()) * cv::Mat(T_matrix_cv)) * p3d_right;

    // Get the coords
    int x1 = p2d_left.at<double>(0, 0) /  p2d_left.at<double>(0, 2);
    int y1 = p2d_left.at<double>(0, 1) /  p2d_left.at<double>(0, 2);

    int x2 = p2d_right.at<double>(0, 0) /  p2d_right.at<double>(0, 2);
    int y2 = p2d_right.at<double>(0, 1) /  p2d_right.at<double>(0, 2);

    // Deletes the old points
    if ( ! points.empty()) {
      points.pop_back();
      points.pop_back();
    }

    // Adds the points to the return vector
    points.push_back(cv::Point(x1, y1));
    points.push_back(cv::Point(x2, y2));

    // Draws in the image
    cv::circle(img, cv::Point(x1, y1), CIRCLE_RAD, colors[i], -1);
    cv::circle(img, cv::Point(x2, y2), CIRCLE_RAD, colors[i], -1);
    cv::line(img, cv::Point(x1, y1), cv::Point(x2, y2), colors[i], 3, cv::LINE_AA);

    putText(img, std::to_string(i), cv::Point(x2 + TEXT_DIST, y2 + TEXT_DIST), 
      cv::FONT_HERSHEY_SIMPLEX, 0.5, colors[i]);

  }
  return points;
} // Returns the last 2 points


cv::Mat hsv_filter(cv::Mat input_image, cv::Scalar lower_color, cv::Scalar upper_color)
{
    // Convert input image to HSV color space
    cv::Mat hsv_image;
    cv::cvtColor(input_image, hsv_image, cv::COLOR_BGR2HSV);

    // Filter image to keep only green pixels in range
    cv::Mat mask;
    cv::inRange(hsv_image, lower_color, upper_color, mask);

    // Apply the mask to the input image to get the filtered image
    cv::Mat filtered_image;
    input_image.copyTo(filtered_image, mask);

    return filtered_image;
}

cv::Mat draw_circles(cv::Mat original, std::vector<cv::Vec3f> circles, cv::Scalar color_r, cv::Scalar color_c)
{
  cv::Mat circles_img;
  original.copyTo(circles_img);

  for( size_t i = 0; i < circles.size(); i++ ) {
    cv::Vec3i c = circles[i];

    cv::Point center = cv::Point(c[0], c[1]);
    cv::circle( circles_img, center, CV_CIRCLE_RADIUS, 
              color_c, -1);

    int radius = c[2];
    cv::circle( circles_img, center, radius, 
              color_r, THICKNESS, cv::LINE_AA);
    }
    return circles_img;
}

std::vector<cv::Vec3f> detect_circles_hough(cv::Mat color_filtered)
{
  cv::Mat gray;

  const int min_radius = 3;
  const int max_radius = 100;

  const int canny_threshold = 50;
  const int center_threshold = 10;

  cvtColor(color_filtered, gray, cv::COLOR_BGR2GRAY);
  medianBlur(gray, gray, 5);

  std::vector<cv::Vec3f> circles;
  HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1,
                gray.rows/6,  // Minimum distance between detected centers  
               canny_threshold, center_threshold, min_radius, max_radius);

  return circles;
}

void publish_centers(std::vector<cv::Vec3f> centers, cv::Mat depth_image, rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr publisher)
{
  auto message = std_msgs::msg::Float32MultiArray();
  std::vector<float> pub_msg_data; 

  pub_msg_data.push_back((float) centers.size());

  for (cv::Vec3f vect : centers) {
    cv::Point point = cv::Point(vect[0], vect[1]);
    std::vector<double> points3d = get_point_dist(point, depth_image);

    pub_msg_data.push_back((float)points3d[0]);  // x
    pub_msg_data.push_back((float)points3d[1]);  // y
    pub_msg_data.push_back((float)points3d[2]);  // z
  }

  message.data = pub_msg_data;
  publisher -> publish(message);
}

cv::Mat image_processing(cv::Mat in_image, cv::Mat depth_image, int mode, int dist, std::vector<cv::Vec3f> centers3d, 
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr center_publisher, std::vector<cv::Rect> people)
{
  cv::Mat out_image;
  in_image.copyTo(out_image);

  switch(mode) {

    case 0:
      break;
    
    case 1:
    {
      // Color filter 
      cv::Scalar pinkLower(140, 92, 66);
      cv::Scalar pinkUpper(177, 255, 255);

      cv::Mat pink_filter = hsv_filter(out_image, pinkLower, pinkUpper);
      std::vector<cv::Vec3f> centers = detect_circles_hough(pink_filter);

      publish_centers(centers, depth_image, center_publisher);
      out_image = draw_circles(out_image, centers, cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0));  // Draws 2d centers

      for (uint i = 0; i < centers3d.size(); i++) {
        cv::circle(out_image, cv::Point(centers3d[i][0], centers3d[i][1]), CV_CIRCLE_RADIUS, 
          cv::Scalar(255,255,255), -1);
      }
      
      // Prints the bounding box
      for (size_t i = 0; i < people.size(); i++) {
        cv::rectangle(out_image, people[i].tl(), people[i].br(), cv::Scalar(0, 255, 0), 2);
      }

      // Shows the lines
      if (dist > 0) {
        std::vector<cv::Point> line_points = draw_dist_lines(out_image, dist);
      }
    }  
      break;
    
    case 2:
      out_image = draw_circles(out_image, centers3d, cv::Scalar(0, 255, 255), cv::Scalar(0, 0, 255));
      break;
    
    default:
      
      break;
  }

  cv::imshow(WINDOW_NAME, out_image);

  return out_image;
}



int main(int argc, char * argv[]) 
{
  rclcpp::init(argc, argv);

  std::thread pcl_thread([&]() {
    rclcpp::spin(std::make_shared<pclNode>());
  });

  std::thread cv_thread([&]() {
    rclcpp::spin(std::make_shared<cvNode>());
  });

  pcl_thread.join();
  cv_thread.join();

  rclcpp::shutdown();

  return 0;
}