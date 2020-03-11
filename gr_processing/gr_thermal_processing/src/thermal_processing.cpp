#include <gr_thermal_processing/thermal_processing.h>

namespace gr_thermal_processing
{
  ThermalProcessing::ThermalProcessing(): last_results_(){
    //Local NodeHandle
    config_params_ = new ThermalFilterConfig();
    ros::NodeHandle local_nh("~");
    filterImage = &cv_filter;

   	dyn_server_cb_ = boost::bind(&ThermalProcessing::dyn_reconfigureCB, this, _1, _2);
   	dyn_server_.setCallback(dyn_server_cb_);

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

  void ThermalProcessing::dyn_reconfigureCB(ThermalConfig &config, uint32_t level){
    boost::mutex::scoped_lock lock(mtx_);
    //TODO mayne ThermalConfig can be copied and send it to the function instead
    config_params_->dilate_factor = config.dilate_factor;
    config_params_->erosion_factor = config.erosion_factor;
    config_params_->anchor_point = config.anchor_point;
    config_params_->ddepth = config.ddepth;
    config_params_->delta = config.delta;
    config_params_->kernel_size = config.kernel_size;
    config_params_->filter_iterations = config.filter_iterations;
    config_params_->threshold = config.threshold;
    config_params_->apply_threshold = config.apply_threshold;
    config_params_->norm_factor = config.norm_factor;
    config_params_->threshold_mode = config.threshold_mode;
  }

  void ThermalProcessing::images_CB(const sensor_msgs::ImageConstPtr thermal_image){
    //boost::mutex::scoped_lock lock(mtx_);
    cv_bridge::CvImagePtr process_frame;

    if (!convertROSImage2Mat(process_frame, thermal_image)){//DEPTH
    //if (!convertROSImage2Mat(process_frame, color_image)){//COLOR
      ROS_ERROR("NOT WORKING");
      return;
    }

    //initialize last result
    geometry_msgs::Accel out(last_results_);
    filterImage(process_frame, out, config_params_);
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
