#ifndef DEPTH_POLICY_H
#define DEPTH_POLICY_H

#include <ros/ros.h>
#include <nodelet/nodelet.h>

#include <gr_depth_processing/depth_filters.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <std_msgs/Float32.h>
#include <std_msgs/String.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseStamped.h>

#include <sensor_msgs/CameraInfo.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <boost/thread/recursive_mutex.hpp>
#include <boost/function.hpp>
//#include <dynamic_reconfigure/server.h>

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video/background_segm.hpp>

#include <darknet_ros_msgs/BoundingBoxes.h>
#include <safety_msgs/FoundObjectsArray.h>

#include <common_detection_utils/common_detection_utils.h>
#include <common_detection_utils/math_functions.hpp>


//gr_detection::DetectionArray gr_detection::FusionDetection::DETECTIONSARRAY = gr_detection::DetectionArray();

namespace gr_depth_processing
{
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> ImagesSyncPolicy;
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, darknet_ros_msgs::BoundingBoxes> RegisteredSyncPolicy;

  template<typename T>
  T getOneMessage(std::string topic_name){
    boost::shared_ptr<T const> msg_pointer;
    msg_pointer =  ros::topic::waitForMessage<T>(topic_name);
    return *msg_pointer;
  }

  class MyNodeletClass : public nodelet::Nodelet, gr_detection::FusionDetection
  {
    public:
      virtual void onInit();
      void images_CB(const sensor_msgs::ImageConstPtr color_image,
                     const sensor_msgs::ImageConstPtr depth_image);

      void register_CB(const sensor_msgs::ImageConstPtr depth_image,
                     const darknet_ros_msgs::BoundingBoxesConstPtr bounding_boxes);

      boost::function<void(cv::Mat&)> filterImage;
      boost::function<double(darknet_ros_msgs::BoundingBox, cv::Mat&, sensor_msgs::CameraInfo)> registerImage;

    protected:
      bool convertROSImage2Mat(cv::Mat& frame,  const sensor_msgs::ImageConstPtr& ros_image);
      void publishOutput(cv::Mat frame, bool rotate = true);

    private:
      message_filters::Synchronizer<ImagesSyncPolicy>*images_syncronizer_;
      message_filters::Synchronizer<RegisteredSyncPolicy>*registered_syncronizer_;

      message_filters::Subscriber<sensor_msgs::Image>* color_image_sub_;
      message_filters::Subscriber<sensor_msgs::Image>* depth_image_sub_;
      message_filters::Subscriber<darknet_ros_msgs::BoundingBoxes>* bounding_boxes_sub_;

      ros::Publisher obstacle_pub_;
      ros::Publisher depth_image_pub_;
      ros::Publisher safety_pub_;

      geometry_msgs::PoseArray detected_objects_;
      safety_msgs::FoundObjectsArray objects_array_;

      //intrinsic params
      sensor_msgs::CameraInfo camera_color_info_;
      sensor_msgs::CameraInfo camera_depth_info_;

      tf2_ros::Buffer tf_buffer_;
      tf2_ros::TransformListener* tf2_listener_;

      boost::recursive_mutex mutex;
      double max_range_;
      std::string global_frame_;
  };

};

#endif
