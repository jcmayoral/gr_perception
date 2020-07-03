#include <ros/ros.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <experimental/filesystem>
#include <common_detection_utils/math_functions.hpp>

namespace fs = ::boost::filesystem;

namespace persons_stuff{

    struct PersonInfo{
        double mean_x;
        double mean_y;
        double mean_z;
        double mean_i;
        double var_x=0;
        double var_y=0;
        double var_z=0;
        double var_i=0;
        double rangex;
        double rangey;
        double rangez;
        double rangei;
    };

    std::ostream& operator<<(std::ostream& os, const PersonInfo& pi){
        return os<<pi.mean_x << "," <<pi.mean_y << "," << pi.mean_z << ", " << pi.mean_i << ", " <<
        pi.var_x << "," <<pi.var_y << "," << pi.var_z << ", " << pi.var_i << ", " <<
        pi.rangex << "," <<pi.rangey << "," << pi.rangez << ", " << pi.rangei  << '\n';
    }

    class PersonsPCDReader{
        public:
        PersonsPCDReader();
        ~PersonsPCDReader();
        void readAllPCDFiles();
        void readBatchPCDFiles(int batch_size, int persons_per_batch=3);
        PersonInfo evaluateCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr pc);
        private:
        ros::NodeHandle nh_;
        ros::Publisher pc_pub_;
    };
}