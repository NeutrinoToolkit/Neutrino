#ifndef __nHolder
#define __nHolder

#include "nPhysImageF.h"
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

};


#endif
