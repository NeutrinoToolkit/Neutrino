#ifndef __nPhysD
#define __nPhysD

#include "nPhysImageF.h"
#include <QtCore>
#include <list>

class nPhysD : public QObject, public nPhysImageF<double>  {
    Q_OBJECT

public:
    nPhysD(nPhysImageF<double> *ref) {
        DEBUG("------------------>>>>>>>>>>>>>" << ref->getName());
    }

    void TscanBrightness();
private:
    std::vector<nPhysD*> physChildren;
    std::vector<nPhysD*> physParents;

public slots:
    void addParent(nPhysD* my_phys);
    void addChildren(nPhysD* my_phys);
    void removeParent(nPhysD* my_phys);
    void removeChildren(nPhysD* my_phys);

    const int childCount();
    const int parentCount();

    nPhysD* childN(unsigned int num);
    nPhysD* parentN(unsigned int num);

signals:
    void physChanged(nPhysD*);

};

#endif
