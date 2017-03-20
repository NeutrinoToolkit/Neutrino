#ifndef __nPhysD
#define __nPhysD

#include "nPhysImageF.h"
#include <QtCore>
#include <list>

class nPhysD : public QObject, public physD  {
    Q_OBJECT

public:
	nPhysD(physD *ref) ;

    void TscanBrightness();
	const unsigned char *to_uchar_palette(std::vector<unsigned char>  &my_palette, std::string palette_name);

    double gamma();

private:
    std::vector<nPhysD*> physChildren;
    std::vector<nPhysD*> physParents;
    std::vector<unsigned char> uchar_buf;


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

Q_DECLARE_METATYPE(physD*);

#endif
