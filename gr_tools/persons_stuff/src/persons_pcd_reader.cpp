#include <persons_stuff/persons_pcd_reader.h>

using namespace persons_stuff;

PersonsPCDReader::PersonsPCDReader(){

}

PersonsPCDReader::~PersonsPCDReader(){
    
}

int PersonsPCDReader::readPCDFile(){
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZI>);
    if (pcl::io::loadPCDFile<pcl::PointXYZI> ("test_pcd.pcd", *cloud) == -1){
        PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
        return (-1);
    }
    for (size_t i = 0; i < cloud->points.size (); ++i)
    std::cout << "    " << cloud->points[i].x
              << " "    << cloud->points[i].y
              << " "    << cloud->points[i].z << std::endl;
}