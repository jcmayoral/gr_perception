#include <ros/ros.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <experimental/filesystem>
#include <common_detection_utils/math_functions.hpp>
#include <safety_msgs/RiskIndexes.h>
#include <safety_msgs/FoundObjectsArray.h>

namespace fs = ::boost::filesystem;

namespace persons_stuff{
    template<typename T>
    T getOneMessage(std::string topic_name){
        boost::shared_ptr<T const> msg_pointer;
        msg_pointer =  ros::topic::waitForMessage<T>(topic_name);
        return *msg_pointer;
    }

    struct PersonInfo{
        std::string original_id;
        std::string stamp;
        std::string labeled_id;
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
        double safety_index;
    };

    std::ostream& operator<<(std::ostream& os, const PersonInfo& pi){
        return os<<pi.original_id << "," << pi.stamp << pi.labeled_id << "," << pi.mean_x << "," <<pi.mean_y << ","<< 
        pi.mean_z << "," << pi.mean_i << "," << pi.var_x << "," <<pi.var_y << "," << 
        pi.var_z << ", " << pi.var_i << ", " << pi.rangex << "," <<pi.rangey << "," << 
        pi.rangez << ", " << pi.rangei  << '\n';
    }

    class PersonsPCDReader{
        public:
        PersonsPCDReader();
        ~PersonsPCDReader();
        void readAllPCDFiles();
        void readBatchPCDFiles(int batch_size);
        void getHeaderInfo(const std::string stack, std::string& id, std::string& timestamp);
        PersonInfo evaluateCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr pc);
        void assignSafety(PersonInfo& i);
        private:
        ros::NodeHandle nh_;
        ros::Publisher pc_pub_;
        ros::Publisher safety_pub_;

    };
}