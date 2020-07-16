#include <persons_stuff/persons_pcd_reader.h>

using namespace persons_stuff;

int main (int argc, char** argv){
    ros::init(argc, argv, "persons_pcd_reader");
    PersonsPCDReader* ppr = new PersonsPCDReader();
    //
    if (argc==1){
        std::cout << "at least ONE argument required" << std::endl;
        return -1;
    }

    if (argc==2){
        ppr->readAllPCDFiles(argv[1]);
        return 1;
    }
    int cloudsnumber = std::stoi(argv[2]);
    std::cout << "Clouds number " << cloudsnumber << std::endl;
    ppr->readBatchPCDFiles(cloudsnumber,argv[1]);
    return 0;
}