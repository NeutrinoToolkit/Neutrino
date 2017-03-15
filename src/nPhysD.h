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
    std::list<nPhysD*> physChildren;
    std::list<nPhysD*> physParents;

public slots:
    void addParent(nPhysD* my_phys);
    void addChildren(nPhysD* my_phys) {physChildren.push_back(my_phys);}
    void removeParent(nPhysD* my_phys);
    void removeChildren(nPhysD* my_phys) {physChildren.push_back(my_phys);}

    int childCount() const;
    int parentCount() const;

    nPhysD* childN(int num);
    nPhysD* parentN(int num);

signals:
    void physChanged(nPhysD*);

};

#endif
