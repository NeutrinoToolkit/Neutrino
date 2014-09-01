#include <string>
#include <iostream>
#include "panThread.h"
#include "nPhysImageF.h"


panThread::panThread() : 
params(NULL), 
calculation_function(NULL), 
n_iter(-1) {
    DEBUG("creator");   
}

void panThread::setThread(void *iparams, void (*ifunc)(void *, int &)) {
    DEBUG(">>>>>>>>>>>>>>> Setting thread param:" << iparams << " func:" << ifunc);
    params = iparams;
    calculation_function = ifunc;
}
void panThread::run() {
    if (calculation_function==NULL) {
        WARNING("Problems getting calculation_function");
        return;
    }
    std::cerr<<"[nGenericPan] pan thread running..."<<std::flush;
    (*calculation_function)(params, n_iter);
    std::cerr<<"finished!"<<std::endl;
}

void panThread::stop() {
    std::cerr<<"killed!"<<std::endl;
    n_iter = -1;
}
