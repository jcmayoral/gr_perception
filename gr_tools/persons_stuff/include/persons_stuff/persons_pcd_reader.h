#include <ros/ros.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <experimental/filesystem>

namespace fs = ::boost::filesystem;

namespace persons_stuff{
    class PersonsPCDReader{
        public:
        PersonsPCDReader();
        ~PersonsPCDReader();
        void readAllPCDFiles();
        void readBatchPCDFiles(int batch_size);
        private:
        ros::NodeHandle nh_;
        ros::Publisher pc_pub_;
    };
}