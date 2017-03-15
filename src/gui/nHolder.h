#ifndef __nHolder
#define __nHolder

#include "nPhysD.h"
#include <list>

class nHolder: public std::list<nPhysD*>
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
    
    std::list<nPhysD*> fileOpen(std::string fname);
};


#endif
