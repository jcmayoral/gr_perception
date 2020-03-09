#include <gr_thermal_processing/thermal_processing.h>

namespace gr_thermal_processing
{
  ThermalProcessing::ThermalProcessing(){
    //Local NodeHandle
    ros::NodeHandle local_nh("~");
    filterImage = &cv_filter;
    image_sub_ = local_nh.subscribe("/rtsp_camera_relay/image", 1, &ThermalProcessing::images_CB, this);
    image_pub_ = local_nh.advertise<sensor_msgs::Image>("output", 1);
    output_pub_ = local_nh.advertise<geometry_msgs::Accel>("results", 1);
    ROS_INFO("Thermal Processing initialized");
    ros::spin();
  }

  bool ThermalProcessing::convertROSImage2Mat(cv_bridge::CvImagePtr& frame, const sensor_msgs::ImageConstPtr& ros_image){
    try{
      frame = cv_bridge::toCvCopy(ros_image, sensor_msgs::image_encodings::BGR8);
      return true;
    }
    catch (cv_bridge::Exception& e){
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return false;
    }
  }

  void ThermalProcessing::images_CB(const sensor_msgs::ImageConstPtr thermal_image){
    boost::recursive_mutex::scoped_lock scoped_lock(mutex);
    cv_bridge::CvImagePtr process_frame;

    if (!convertROSImage2Mat(process_frame, thermal_image)){//DEPTH
    //if (!convertROSImage2Mat(process_frame, color_image)){//COLOR
      ROS_ERROR("NOT WORKING");
      return;
    }

    geometry_msgs::Accel out;
    filterImage(process_frame, out);
    output_pub_.publish(out);
    image_pub_.publish(process_frame->toImageMsg());
  }
}
