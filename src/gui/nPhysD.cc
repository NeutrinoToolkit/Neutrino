#include "nPhysD.h"


void nPhysD::addParent(nPhysD* my_phys) {
    physParents.push_back(my_phys);
    connect(my_phys, &nPhysD::destroyed, this, &nPhysD::removeParent);
}

void nPhysD::addChildren(nPhysD* my_phys) {
    physChildren.push_back(my_phys);
    connect(my_phys, &nPhysD::destroyed, this, &nPhysD::removeChildren);
}

void nPhysD::removeParent(nPhysD* my_phys) {
    physParents.remove(my_phys);
}

void nPhysD::removeChildren(nPhysD* my_phys) {
    physChildren.remove(my_phys);
}

int nPhysD::childCount() {
    return physChildren.size();
}

int nPhysD::parentCount() {
    return physParents.size();s
}

nPhysD* nPhysD::parentN(int num) {
    return (num<physParents.size() ? physParents[num] : nullptr);
}

nPhysD* nPhysD::childN(int num) {
    return (num<physChildren.size() ? physChildren[num] : nullptr);
}

void nPhysD::TscanBrightness() {
    nPhysImageF<double>::TscanBrightness();
    emit physChanged(this);
}
