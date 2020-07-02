#include <persons_stuff/persons_pcd_reader.h>

using namespace persons_stuff;

PersonsPCDReader::PersonsPCDReader(): nh_{"~"}{
     pc_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("fake_pc", 1);
     ros::spinOnce();
}

PersonsPCDReader::~PersonsPCDReader(){
    
}

void PersonsPCDReader::readBatchPCDFiles(int batch_size){
    //TODO not start from begin()
    sensor_msgs::PointCloud2 output;
    fs::path p { "/media/datasets/persons_pcd/" };
    auto it = fs::directory_iterator(p);
    for ( int i=0 ; i<batch_size ;i++){
        it++;
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZI>);
        auto entry = *it;
        if (pcl::io::loadPCDFile<pcl::PointXYZI> (entry.path().string(), *cloud) == -1){
            PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
            continue;
        }
        std::cout << "PUBLISHING "<<std::endl;
        pcl::toROSMsg(*cloud, output);
        output.header.frame_id = "velodyne";
        pc_pub_.publish(output);
        ros::Duration(1).sleep();
    }
}

void PersonsPCDReader::readAllPCDFiles(){
    sensor_msgs::PointCloud2 output;
    fs::path p { "/media/datasets/persons_pcd/" };
    for (auto& entry : fs::directory_iterator(p)){
        std::cout << entry << std::endl;
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZI>);
        if (pcl::io::loadPCDFile<pcl::PointXYZI> (entry.path().string(), *cloud) == -1){
            PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
            continue;
        }
        std::cout << "PUBLISHING "<<std::endl;
        pcl::toROSMsg(*cloud, output);
        output.header.frame_id = "velodyne";
        pc_pub_.publish(output);
        ros::Duration(1).sleep();
    }
}