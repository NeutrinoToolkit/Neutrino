#include "nHolder.h"
#include "nPhysFormats.h"
std::vector<nPhysD*> nHolder::fileOpen(std::string fname) {
    
    std::vector<nPhysD*> retlist;
    
    for (auto &my_phys : phys_open(fname)) {
		nPhysD* pippo=new nPhysD(my_phys);
		retlist.push_back(pippo);
    }
    
    return retlist;
}
