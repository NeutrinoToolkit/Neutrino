#include "nPhysD.h"


void nPhysD::addParent(nPhysD* my_phys) {
    physParents.push_back(my_phys);
    connect(my_phys, SIGNAL(destroyed), this, SLOT(removeParent));
}

void nPhysD::addChildren(nPhysD* my_phys) {
    physChildren.push_back(my_phys);
    connect(my_phys, SIGNAL(destroyed), this, SLOT(removeChildren));
}

void nPhysD::removeParent(nPhysD* my_phys) {
    physParents.erase( std::remove( physParents.begin(), physParents.end(), my_phys ), physParents.end() );
}

void nPhysD::removeChildren(nPhysD* my_phys) {
    physChildren.erase( std::remove( physChildren.begin(), physChildren.end(), my_phys ), physChildren.end() );
}

const int nPhysD::childCount() {
    return physChildren.size();
}

const int nPhysD::parentCount() {
    return physParents.size();
}

nPhysD* nPhysD::parentN(unsigned int num) {
    return (num<physParents.size() ? physParents[num] : nullptr);
}

nPhysD* nPhysD::childN(unsigned int num) {
    return (num<physChildren.size() ? physChildren[num] : nullptr);
}

void nPhysD::TscanBrightness() {
    nPhysImageF<double>::TscanBrightness();
    emit physChanged(this);
}
