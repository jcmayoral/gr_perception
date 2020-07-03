#include <persons_stuff/persons_pcd_reader.h>

using namespace persons_stuff;

int main (int argc, char** argv){
    ros::init(argc, argv, "persons_pcd_reader");
    PersonsPCDReader* ppr = new PersonsPCDReader();
    //
    if (argc==1){
        ppr->readAllPCDFiles();
        return 1;
    }
    int cloudsnumber = std::stoi(argv[1]);
    std::cout << "Clouds number " << cloudsnumber << std::endl;
    ppr->readBatchPCDFiles(cloudsnumber);
    return 0;
}