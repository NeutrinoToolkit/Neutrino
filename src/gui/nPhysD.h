#ifndef __nPhysD
#define __nPhysD

#include "nPhysImageF.h"
#include <QtCore>
#include <list>

class nPhysD : public QObject, public physD  {
    Q_OBJECT

public:
	nPhysD(physD &ref, QObject*parent=nullptr) ;

    void TscanBrightness();
	const unsigned char *to_uchar_palette(std::vector<unsigned char>  &my_palette, std::string palette_name);

    double gamma();

private:
    std::vector<unsigned char> uchar_buf;

signals:
    void physChanged(nPhysD*);

};

Q_DECLARE_METATYPE(physD*);

#endif
