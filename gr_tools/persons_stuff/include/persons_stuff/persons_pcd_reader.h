#include <ros/ros.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

namespace persons_stuff{
    class PersonsPCDReader{
        public:
        PersonsPCDReader();
        ~PersonsPCDReader();
        int readPCDFile();
    };
}