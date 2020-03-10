#include <gr_thermal_processing/thermal_processing.h>

namespace gr_thermal_processing
{
  ThermalProcessing::ThermalProcessing(): last_results_(){
    //Local NodeHandle
    ros::NodeHandle local_nh("~");
    filterImage = &cv_filter;
    image_subs_.emplace_back(local_nh.subscribe("/rtsp_camera_relay/image", 1, &ThermalProcessing::images_CB, this));
    image_subs_.emplace_back(local_nh.subscribe("/FlirA65/image_raw", 1, &ThermalProcessing::images_CB, this));
    image_pub_ = local_nh.advertise<sensor_msgs::Image>("output", 1);
    output_pub_ = local_nh.advertise<geometry_msgs::Accel>("results", 1);
    ROS_INFO("Thermal Processing initialized");
    ros::spin();
  }

  bool ThermalProcessing::convertROSImage2Mat(cv_bridge::CvImagePtr& frame, const sensor_msgs::ImageConstPtr& ros_image){
    try{
      frame = cv_bridge::toCvCopy(ros_image, sensor_msgs::image_encodings::MONO16);
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

    //initialize last result
    geometry_msgs::Accel out(last_results_);
    filterImage(process_frame, out);
    last_results_.linear.x = out.linear.x;
    last_results_.linear.y = out.linear.y;
    //last_results_.linear.y = (out.linear.y > 10000) ? out.linear.y : 1.0;
    last_results_.linear.z = out.linear.z;
    //last_results_.linear.z = (out.linear.z < 10000) ? out.linear.z : 0.001;
    last_results_.angular.x = out.angular.x;
    last_results_.angular.y = out.angular.y;
    last_results_.angular.z = out.angular.z;
    //last_results_.angular.z = (out.angular.z < 10000) ? out.angular.z : 0.001;
  
    output_pub_.publish(out);
    image_pub_.publish(process_frame->toImageMsg());
  }
}
