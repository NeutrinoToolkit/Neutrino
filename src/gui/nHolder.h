#ifndef __nHolder
#define __nHolder

#include "nPhysD.h"
#include <list>

class nHolder: public std::vector<nPhysD*>
{
public:
    static nHolder& getInstance()
    {
        static nHolder instance;
        return instance;
    }
private:
    nHolder() {
        DEBUG("here");
    }

public:
    nHolder(nHolder const&) = delete;
    void operator=(nHolder const&)  = delete;
    
	std::vector<nPhysD> fileOpen(std::string fname);
};


#endif
