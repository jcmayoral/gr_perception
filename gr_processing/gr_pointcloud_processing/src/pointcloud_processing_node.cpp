#include <gr_pointcloud_processing/pointcloud_processing.h>

int
main (int argc, char **argv)
{
  ros::init(argc, argv, "pointcloud_processing_node");
  PointCloudProcessor pc;
  ros::spin();
  return 0;
}
