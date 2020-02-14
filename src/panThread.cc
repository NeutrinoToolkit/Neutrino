#include <string>
#include <iostream>
#include "panThread.h"
#include "nPhysD.h"

#include <QMessageBox>
#include <QApplication>

panThread::panThread() : 
    n_iter(-1),
    err_message("Error in thread"),
    params(nullptr),
    calculation_function(nullptr)
{
    DEBUG("creator");
}

void panThread::setThread(void *iparams, void (*ifunc)(void *, int &)) {
    DEBUG(">>>>>>>>>>>>>>> Setting thread param:" << iparams << " func:" << ifunc);
    params = iparams;
    calculation_function = ifunc;
}

void panThread::run() {
    if (calculation_function==nullptr) {
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
    params=nullptr;
    calculation_function=nullptr;
}
