#include <persons_stuff/persons_pcd_reader.h>

using namespace persons_stuff;
using namespace gr_detection;

PersonsPCDReader::PersonsPCDReader(): nh_{"~"}{
     pc_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("fake_pc", 1);
     ros::spinOnce();
}

PersonsPCDReader::~PersonsPCDReader(){
    
}

PersonInfo PersonsPCDReader::evaluateCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr pc){

    PersonInfo info;
    std::vector<double> x_vector;
    std::vector<double> y_vector;
    std::vector<double> z_vector;
    std::vector<double> i_vector;

    for (int i = 0; i < pc->size() - 1; i++){
        x_vector.push_back(pc->points[i].x);
        y_vector.push_back(pc->points[i].y);
        z_vector.push_back(pc->points[i].z);
        i_vector.push_back(pc->points[i].intensity);
    }

    info.mean_x = calculateMean<double>(x_vector);
    info.mean_y = calculateMean<double>(y_vector);
    info.mean_z = calculateMean<double>(z_vector);
    info.mean_i = calculateMean<double>(i_vector);

    info.var_x = calculateVariance<double>(x_vector);
    info.var_y = calculateVariance<double>(y_vector);
    info.var_z = calculateVariance<double>(z_vector);
    info.var_i = calculateVariance<double>(i_vector);

    info.rangex = getAbsoluteRange<double>(x_vector);
    info.rangey = getAbsoluteRange<double>(y_vector);
    info.rangez = getAbsoluteRange<double>(z_vector);
    info.rangei = getAbsoluteRange<double>(i_vector);

    return info;
}

void PersonsPCDReader::readBatchPCDFiles(int batch_size, int persons_per_batch){
    //TODO not start from begin()
    sensor_msgs::PointCloud2 output;
    pcl::PointCloud<pcl::PointXYZI> ccloud;
    int counter =0 ;

    fs::path p { "/media/datasets/persons_pcd/" };
    auto it = fs::directory_iterator(p);
    //FOR HACK
    auto it2 = fs::directory_iterator();
    int filesnumber = std::distance(it,it2);
    std::cout << "FILES NUMBER? "<< filesnumber << std::endl;

    //WITHOUT NEXT LINE CODE CRASHES
    it = fs::directory_iterator(p);

    for ( int i=0 ; i<batch_size*persons_per_batch ;i++){
        it++;
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZI>);
        auto entry = *it;
        if (pcl::io::loadPCDFile<pcl::PointXYZI> (entry.path().string(), *cloud) == -1){
            ROS_ERROR_STREAM("Couldn't read file " << entry);
            continue;
        }

        PersonInfo info;
        info = evaluateCloud(cloud);
        std::cout << info << std::endl;

        if (counter<= persons_per_batch-1){
            ccloud += *cloud;
            counter++;
            //ROS_INFO("APPEND PC");
            continue;
        }
        counter = 0;
        std::cout << "PUBLISHING "<<std::endl;
        pcl::toROSMsg(ccloud, output);
        output.header.frame_id = "velodyne";
        pc_pub_.publish(output);
        ros::Duration(1).sleep();
        ccloud.points.clear();
    }
}

void PersonsPCDReader::readAllPCDFiles(){
    sensor_msgs::PointCloud2 output;
    fs::path p { "/media/datasets/persons_pcd/" };
    for (auto& entry : fs::directory_iterator(p)){
        std::cout << entry << std::endl;
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZI>);
        if (pcl::io::loadPCDFile<pcl::PointXYZI> (entry.path().string(), *cloud) == -1){
            ROS_ERROR_STREAM("Couldn't read file " << entry);
            continue;
        }

        PersonInfo info;
        info = evaluateCloud(cloud);
        std::cout << info << std::endl;

        std::cout << "PUBLISHING "<<std::endl;
        pcl::toROSMsg(*cloud, output);
        output.header.frame_id = "velodyne";
        pc_pub_.publish(output);
        ros::Duration(1).sleep();
    }
}