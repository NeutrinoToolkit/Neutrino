#include "nHolder.h"

std::vector<nPhysD*> nHolder::fileOpen(std::string fname) {
    
    std::vector<nPhysD*> retlist;

//    retlist= phys_open(fname);
    
    insert(end(), retlist.begin(), retlist.end());
    
    return retlist;
}
