/*
# Copyright (c) 2022 Adrián Madinabeitia Portanova
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
*/

#include <chrono>
#include <functional>
#include <memory>
#include <string.h>
#include <stdlib.h>
#include <iostream>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

#include "std_msgs/msg/u_int16_multi_array.hpp"

#include "geometry_msgs/msg/transform_stamped.hpp"

#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

#include <image_transport/image_transport.hpp>


#include "cv_bridge/cv_bridge.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


#define WINDOW_NAME "Trace bars"

cv::Mat image_processing(const cv::Mat in_image);



class ComputerVisionSubscriber : public rclcpp::Node
{

  public:

    ComputerVisionSubscriber()
    : Node("opencv_subscriber")
    {
      auto qos = rclcpp::QoS( rclcpp::QoSInitialization( RMW_QOS_POLICY_HISTORY_KEEP_LAST, 5 ));
      qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
   
      subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/head_front_camera/rgb/image_raw", qos, std::bind(&ComputerVisionSubscriber::topic_callback, this, std::placeholders::_1));

      publisher_ = this->create_publisher<std_msgs::msg::UInt16MultiArray>("trace_bar", qos);
    }


  private:


    void topic_callback(const sensor_msgs::msg::Image::SharedPtr msg) const
    {     
      // Convert ROS Image to CV Image
      cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
      cv::Mat image_raw =  cv_ptr->image;

      // Sets the window
      if (first_exe_) {

        cv::namedWindow(WINDOW_NAME);    
        
        cv::createTrackbar(TEXT[0], WINDOW_NAME, nullptr, 255, 0); 
        cv::createTrackbar(TEXT[1], WINDOW_NAME, nullptr, 255, 0); 
        cv::createTrackbar(TEXT[2], WINDOW_NAME, nullptr, 255, 0); 

        cv::createTrackbar(TEXT[3], WINDOW_NAME, nullptr, 255, 0); 
        cv::createTrackbar(TEXT[4], WINDOW_NAME, nullptr, 255, 0); 
        cv::createTrackbar(TEXT[5], WINDOW_NAME, nullptr, 255, 0);             
        
        first_exe_ = false;
      }

      // Image processing
      cv::Mat cv_image = image_processing(image_raw);

      // Convert OpenCV Image to ROS Image
      cv_bridge::CvImage img_bridge = cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::BGR8, cv_image);
      sensor_msgs::msg::Image out_image; // >> message to be sent
      img_bridge.toImageMsg(out_image); // from cv_bridge to sensor_msgs::Image


      short unsigned int rh1 = cv::getTrackbarPos(TEXT[0], WINDOW_NAME); 
      short unsigned int rh2 = cv::getTrackbarPos(TEXT[1], WINDOW_NAME); 
      short unsigned int rh3 = cv::getTrackbarPos(TEXT[2], WINDOW_NAME); 

      short unsigned int rl1 = cv::getTrackbarPos(TEXT[3], WINDOW_NAME); 
      short unsigned int rl2 = cv::getTrackbarPos(TEXT[4], WINDOW_NAME); 
      short unsigned int rl3 = cv::getTrackbarPos(TEXT[5], WINDOW_NAME); 

      std_msgs::msg::UInt16MultiArray msg2;
      msg2.data = {rh1, rh2, rh3, rl1, rl2, rl3};
      // Publish the data
      publisher_ -> publish(msg2);

      cv::waitKey(1);
    }

    // Publishers and subscribers
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Publisher<std_msgs::msg::UInt16MultiArray>::SharedPtr publisher_;


    mutable bool first_exe_ = true;
    std::string TEXT[6] = {"Val 1", "Val 2", "Val 3", "Val 4", "Val 5", "Val 6"};
};



cv::Mat image_processing(const cv::Mat in_image)
{
  cv::Mat out_image;
  in_image.copyTo(out_image);
  return out_image;
}

int main(int argc, char * argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ComputerVisionSubscriber>());
  rclcpp::shutdown();
  return 0;
}