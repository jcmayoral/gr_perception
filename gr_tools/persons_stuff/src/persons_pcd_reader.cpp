#include <persons_stuff/persons_pcd_reader.h>

using namespace persons_stuff;
using namespace gr_detection;

PersonsPCDReader::PersonsPCDReader(): nh_{}{
     pc_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/velodyne_points", 1);
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

void PersonsPCDReader::getHeaderInfo(const std::string stack, std::string& id, std::string& timestamp){
        std::string help = stack;
        std::cout << help <<  std::endl;
        auto found = help.find_last_of("/");
		if (found != std::string::npos) { //if a match was found
			help.replace(help.begin(), help.begin()+found+1, "");            
		}

        auto found2 = help.find_last_of("_");
		if (found2 != std::string::npos) { //if a match was found
			//help= help.substr(help.begin()+found2, help.end());
			help.replace(help.begin()+found2, help.end(), "");            
        }

        auto found3 = help.find("*");
        id = help.substr(0,found3);
        timestamp = help.substr(found3+1);
        std::cout << help <<  std::endl;
        std::cout << id << ", " << timestamp << std::endl;
        //id = "id";
        //timestamp = "id";
}


void PersonsPCDReader::assignSafety(PersonInfo& i){
    safety_msgs::RiskIndexes msg = getOneMessage<safety_msgs::RiskIndexes>("/safety_indexes");
    ROS_INFO_STREAM(msg);
    std::cout << i.original_id << " , " << i.stamp << std::endl;
    //ID DOES NOT MATCH
    i.labeled_id = msg.objects_id[0];
    i.safety_index = msg.risk_indexes[0];
}

void PersonsPCDReader::readBatchPCDFiles(int batch_size){
    //TODO not start from begin()
    sensor_msgs::PointCloud2 output;
    fs::path p { "/media/datasets/persons_pcd/" };
    auto it = fs::directory_iterator(p);
    //FOR HACK
    auto it2 = fs::directory_iterator();
    int filesnumber = std::distance(it,it2);
    std::cout << "FILES NUMBER? "<< filesnumber << std::endl;

    //WITHOUT NEXT LINE CODE CRASHES
    it = fs::directory_iterator(p);

    for ( int i=0 ; i<batch_size ;i++){
        it++;
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZI>);
        auto entry = *it;
        if (pcl::io::loadPCDFile<pcl::PointXYZI> (entry.path().string(), *cloud) == -1){
            ROS_ERROR_STREAM("Couldn't read file " << entry);
            continue;
        }

        PersonInfo info;
        std::string file = entry.path().string();
        info = evaluateCloud(cloud);
        getHeaderInfo(file, info.original_id, info.stamp);
        pcl::toROSMsg(*cloud, output);
        output.header.frame_id = "velodyne";
        pc_pub_.publish(output);
        ros::Duration(0.05).sleep();
        assignSafety(info);
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
        std::string file = entry.path().string();
        info = evaluateCloud(cloud);
        getHeaderInfo(file, info.original_id, info.stamp);
        pcl::toROSMsg(*cloud, output);
        output.header.frame_id = "velodyne";
        pc_pub_.publish(output);
        ros::Duration(0.05).sleep();
        assignSafety(info);
    }
}