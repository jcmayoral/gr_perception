//ROS
#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseArray.h>

//PCL
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>

#include <pcl_conversions/pcl_conversions.h>
//#include <pcl_ros/transforms.h>


//#include <pcl/visualization/cloud_viewer.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/extract_clusters.h>

//#include <pcl/cuda/filters/passthrough.h>

//dynamic_reconfigure
#include <dynamic_reconfigure/server.h>
#include <gr_pointcloud_filter/FiltersConfig.h>

//recursive_mutex
#include <boost/thread/recursive_mutex.hpp>

//C++
#include <string>

namespace gr_pointcloud_filter
{

    class MyNodeletClass : public nodelet::Nodelet
    {
    	private:
            ros::Subscriber pointcloud_sub_;
            ros::Publisher pointcloud_pub_;
            ros::Publisher obstacle_pub_;
            sensor_msgs::PointCloud2 output_pointcloud_;
            pcl::VoxelGrid<pcl::PointXYZ> voxel_filter_;
            pcl::SACSegmentation<pcl::PointXYZ> segmentation_filter_;
            pcl::ExtractIndices<pcl::PointXYZ> extraction_filter_;
            pcl::StatisticalOutlierRemoval<pcl::PointXYZ> outliers_filter_;
            pcl::RadiusOutlierRemoval<pcl::PointXYZ> radius_outliers_filter_;
            pcl::ConditionAnd<pcl::PointXYZ>::Ptr conditional_filter_;
            pcl::ConditionalRemoval<pcl::PointXYZ> condition_removal_;
            pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator_;
            pcl::PassThrough<pcl::PointXYZ> pass_through_filter_;
            pcl::EuclideanClusterExtraction<pcl::PointXYZ> euclidean_cluster_;

            //pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> region_growing_filter_;;
            ros::Time last_processing_time_;

            //Dynamic Reconfigure
            dynamic_reconfigure::Server<gr_pointcloud_filter::FiltersConfig> dyn_server_;
            dynamic_reconfigure::Server<gr_pointcloud_filter::FiltersConfig>::CallbackType dyn_server_cb_;
            bool filters_enablers_[6];
            bool enable_visualization_;

    	public:
            virtual void onInit();
            void pointcloud_cb(const sensor_msgs::PointCloud2ConstPtr msg);
            void applyFilters(const sensor_msgs::PointCloud2 msg);
            void setFiltersParams(gr_pointcloud_filter::FiltersConfig &config);
            void dyn_reconfigureCB(gr_pointcloud_filter::FiltersConfig &config, uint32_t level);
            template <class T> void publishPointCloud(T);
            boost::recursive_mutex mutex;

    };

}
