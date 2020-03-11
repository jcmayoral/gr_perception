#ifndef THERMAL_MONITOR_H
#define THERMAL_MONITOR_H

#include <ros/ros.h>
#include <gr_thermal_processing/thermal_filters.h>

#include <std_msgs/Float32.h>
#include <std_msgs/String.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/Accel.h>

#include <boost/thread/recursive_mutex.hpp>
#include <boost/function.hpp>

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video/background_segm.hpp>

#include <safety_msgs/FoundObjectsArray.h>
#include <gr_thermal_processing/thermal_config.hpp>

#include <boost/thread/mutex.hpp>

#include <dynamic_reconfigure/server.h>
#include <gr_thermal_processing/ThermalConfig.h>

namespace gr_thermal_processing
{

  class ThermalProcessing
  {
    public:
      ThermalProcessing();
      void images_CB(const sensor_msgs::ImageConstPtr color_image);
      boost::function<void(cv_bridge::CvImagePtr&, geometry_msgs::Accel&, const ThermalFilterConfig*)> filterImage;
      void dyn_reconfigureCB(ThermalConfig &config, uint32_t level);

    protected:
      bool convertROSImage2Mat(cv_bridge::CvImagePtr& frame,  const sensor_msgs::ImageConstPtr& ros_image);

    private:
      boost::mutex mtx_;
      std::vector<ros::Subscriber> image_subs_;
      ros::Publisher image_pub_;
      ros::Publisher output_pub_;
      geometry_msgs::Accel last_results_;
      ThermalFilterConfig* config_params_;

      dynamic_reconfigure::Server<ThermalConfig> dyn_server_;
      dynamic_reconfigure::Server<ThermalConfig>::CallbackType dyn_server_cb_;
  };

};

#endif
