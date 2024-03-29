#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/statistical_outlier_removal.h>

// The GPU specific stuff here
#include <pcl/gpu/octree/octree.hpp>
#include <pcl/gpu/containers/device_array.hpp>
#include <pcl/gpu/segmentation/gpu_extract_clusters.h>
#include <pcl/gpu/segmentation/impl/gpu_extract_clusters.hpp>

#include <functional>
#include <iostream>
#include <mutex>

//#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>

//ROS stuff
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <geometry_msgs/PoseArray.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <pcl/filters/passthrough.h>
#include <gr_pointcloud_processing/PointCloudConfig.h>
#include <dynamic_reconfigure/server.h>

#include <boost/shared_ptr.hpp>
#include <boost/thread/mutex.hpp>

#include <pcl_cuda_tools/filters/filter_passthrough.h>
#include <pcl_cuda_tools/filters/pcl_filter_passthrough.h>

#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/utils.h>

#include <common_detection_utils/common_detection_utils.h>
#include <common_detection_utils/math_functions.hpp>
 
#include <safety_msgs/FoundObjectsArray.h>

#include <nodelet/nodelet.h>

//Save PCD
 #include <pcl/io/pcd_io.h>

//ACTIONLIB
#include <gr_action_msgs/GRPCProcessAction.h>
#include <actionlib/server/simple_action_server.h>

namespace gr_pointcloud_processing{
  
  template <typename T>
  void loadFromRemappings(std::map<std::string, std::string> args, std::string id, T& value){
    if(args.find(id) != args.end()){
      value = args[id];
    }
  };

  class PointCloudProcessor : public nodelet::Nodelet, gr_detection::FusionDetection{
  private:
    boost::mutex mutex_;
    ros::Subscriber pc_sub_;
    ros::Publisher pc_pub_;
    ros::Publisher cluster_pub_;
    ros::Publisher safety_pub_;
    ros::Publisher bb_pub_;
    pcl::gpu::Octree::PointCloud cloud_device;
    pcl::gpu::EuclideanClusterExtraction gec;
    pcl::ExtractIndices<pcl::PointXYZI> extraction_filter_;
    pcl::SACSegmentation<pcl::PointXYZI> segmentation_filter_;
    pcl::PassThrough<pcl::PointXYZI> pass_through_filter_;
    pcl::ConditionalRemoval<pcl::PointXYZI> condition_removal_;
    pcl::StatisticalOutlierRemoval<pcl::PointXYZI> outliers_filter_;
    pcl::PointCloud<pcl::PointXYZI> main_cloud_;
    ros::Timer timer_;
    geometry_msgs::TransformStamped to_odom_transform;

    //Testing
    pcl_gpu::FilterPassThrough radius_cuda_pass_;
    pcl_gpu::PCLFilterPassThrough pcl_cuda_pass_;

    //Dynamic Reconfigure
    dynamic_reconfigure::Server<gr_pointcloud_processing::PointCloudConfig> dyn_server_;
    dynamic_reconfigure::Server<gr_pointcloud_processing::PointCloudConfig>::CallbackType dyn_server_cb_;
    double dynamic_std_;
    double dynamic_std_z_;
    double distance_to_floor_;
    float static_dynamic_classifier_;

    bool output_publish_;
    bool remove_ground_;
    bool passthrough_enable_;
    bool is_processing_;
    bool is_timer_enable_;
    jsk_recognition_msgs::BoundingBoxArray bb;
    std::clock_t tStart;
    ros::Time last_detection_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener* tf2_listener_;
    std::string sensor_frame_;
    std::string global_frame_;

    //Testing
    //gr_detection::PersonArray persons_array_;
    //ACTIONLIB
    boost::shared_ptr<actionlib::SimpleActionServer<gr_action_msgs::GRPCProcessAction>> as_;
    gr_action_msgs::GRPCProcessFeedback feedback_;
    gr_action_msgs::GRPCProcessResult result_;
    safety_msgs::FoundObjectsArray safety_msg_;

  public:
    //ACTION LibCB
    void executeCB(const gr_action_msgs::GRPCProcessGoalConstPtr &goal);
    virtual void onInit();
    void pointcloud_cb(const sensor_msgs::PointCloud2ConstPtr msg);
    int run_filter(const boost::shared_ptr <pcl::PointCloud<pcl::PointXYZI>> cloud_filtered);
    template <class T> void publishPointCloud(T);
    void timer_cb(const ros::TimerEvent&);
    void cluster();
    void publishBoundingBoxes();
    void addBoundingBox(const geometry_msgs::Pose center, double v_x, double v_y, double v_z, double var_i, int label);
    void dyn_reconfigureCB(gr_pointcloud_processing::PointCloudConfig &config, uint32_t level);
    void removeGround(boost::shared_ptr <pcl::PointCloud<pcl::PointXYZI>> pc);
  };
};