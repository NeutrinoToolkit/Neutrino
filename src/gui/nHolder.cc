#include "nHolder.h"
#include "nPhysFormats.h"
std::vector<nPhysD*> nHolder::fileOpen(std::string fname) {
    
    std::vector<nPhysD*> retlist;
    
    for (auto &my_phys : phys_open(fname)) {
        retlist.push_back((nPhysD*)my_phys);
    }
    
    return retlist;
}
