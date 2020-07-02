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
    int personspercloud = 3;
    int cloudsnumber = std::stoi(argv[1]);

    if (argc>2){
        personspercloud = std::stoi(argv[2]);
    }

    std::cout << "Clouds number " << cloudsnumber << std::endl;
    std::cout << "PPB " << personspercloud << std::endl;

    ppr->readBatchPCDFiles(cloudsnumber, personspercloud);
    return 0;
}