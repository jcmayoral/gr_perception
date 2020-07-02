#include <persons_stuff/persons_pcd_reader.h>

using namespace persons_stuff;

int main (int argc, char** argv){
    ros::init(argc, argv, "persons_pcd_reader");
    PersonsPCDReader* ppr = new PersonsPCDReader();
    ppr->readAllPCDFiles();
    return 0;
}