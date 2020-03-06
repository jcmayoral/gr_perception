#include <gr_thermal_processing/thermal_processing.h>

namespace gr_thermal_processing
{
  ThermalProcessing::ThermalProcessing(){
    //Local NodeHandle
    ros::NodeHandle local_nh("~");
    filterImage = &cv_filter;
    ROS_INFO("Thermal Processing initialized");
  }

  bool ThermalProcessing::convertROSImage2Mat(cv::Mat& frame, const sensor_msgs::ImageConstPtr& ros_image){
    try{
      frame = cv_bridge::toCvShare(ros_image, sensor_msgs::image_encodings::TYPE_16UC1)->image; //realsense
      return true;
    }
    catch (cv_bridge::Exception& e){
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return false;
    }
  }

  void ThermalProcessing::images_CB(const sensor_msgs::ImageConstPtr thermal_image){
    boost::recursive_mutex::scoped_lock scoped_lock(mutex);
    cv::Mat process_frame;

    if (!convertROSImage2Mat(process_frame, thermal_image)){//DEPTH
    //if (!convertROSImage2Mat(process_frame, color_image)){//COLOR
      return;
    }

    filterImage(process_frame);
  }
}
