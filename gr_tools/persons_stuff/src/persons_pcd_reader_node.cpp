#include <persons_stuff/persons_pcd_reader.h>

using namespace persons_stuff;

int main (int argc, char** argv){
    PersonsPCDReader* ppr = new PersonsPCDReader();
    ppr->readPCDFile();
    return 0;
}