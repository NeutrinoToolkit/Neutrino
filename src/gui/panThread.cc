#include <string>
#include <iostream>
#include "panThread.h"
#include "nPhysImageF.h"

#include <QMessageBox>
#include <QApplication>

panThread::panThread() : 
params(NULL), 
calculation_function(NULL), 
n_iter(-1),
err_message("Error in thread")
{
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
    DEBUG("[nGenericPan] pan thread running...");
    try {
        (*calculation_function)(params, n_iter);
    } catch (std::exception &e) {
        err_message=QString(e.what());
        stop();
    }
    DEBUG("[nGenericPan] pan thread finished");
}

void panThread::stop() {
    DEBUG("[nGenericPan] pan thread killed");
    n_iter = -1;
}
