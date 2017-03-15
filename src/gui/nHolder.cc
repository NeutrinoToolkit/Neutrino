#include "nHolder.h"

std::list<nPhysD*> nHolder::fileOpen(std::string fname) {
    
    std::vector<nPhysImageF<double>> retlist = phys_open(fname);
    
    insert(end(), retlist.begin(), retlist.end());
}
