#include <vector>

#include "tridimvec.h"

#pragma once

#ifndef __tribox_h
#define __tribox_h

typedef tridimvec<double> fp;

//struct scalarMap {
//	std::vector<double> xmap;
//	std::vector<double> ymap;
//	std::vector<double> zmap;
//	double ***map;
//};


class tribox {

public:
    tribox()
        : myvertex1(), myvertex2(), scalarValue(0)
    { }
    tribox(fp vert1, fp vert2)
        : myvertex1(vert1), myvertex2(vert2)
    { }

    tribox(fp vert1, fp vert2, double value)
        : myvertex1(vert1), myvertex2(vert2), scalarValue(value)
    { }

    ~tribox()
    { }

    inline bool isInside(fp);
    bool isValid(void)
    { return (myvertex1 != myvertex2); }

    void setV1(fp p1)
    { myvertex1 = p1; }

    void setV2(fp p2)
    { myvertex2 = p2; }

    void setBoundaries(fp p1, fp p2)
    { myvertex1 = p1; myvertex2 = p2; }

    fp getCenter()
    { fp cc = 0.5*(myvertex1+myvertex2); return cc;}

    fp getSize()
    { fp si = (myvertex1-myvertex2).abs(); return si;}

    void setCenter(fp cp)
    {
            fp op1 = myvertex1, op2 = myvertex2;
        fp occ = 0.5*(op1+op2);
        fp dcc = cp-occ;
            myvertex1 = op1+dcc;
        myvertex2 = op2+dcc;

//        fp sz = getSize();
    }

    void setSize(fp si)
    { fp occ = 0.5*(myvertex1+myvertex2);
          myvertex1 = occ-0.5*si; myvertex2 = occ+0.5*si; }

    void assignScalarQuantity(double value)
    { scalarValue = value; }

    //void assignScalarQuantity(struct scalarMap *)
    //{ }

    double getField(fp look_point)
    { if (isInside(look_point)) return scalarValue; else return 0; }


    fp myvertex1, myvertex2;
    double scalarValue;

};

inline bool
tribox::isInside(fp cp)
{
    if (!isValid() )
        return false;

    if ( (((cp.x() > myvertex1.x()) && (cp.x() < myvertex2.x())) || ((cp.x() < myvertex1.x()) && (cp.x() > myvertex2.x()))) &&
    (((cp.y() > myvertex1.y()) && (cp.y() < myvertex2.y())) || ((cp.y() < myvertex1.y()) && (cp.y() > myvertex2.y()))) &&
            (((cp.z() > myvertex1.z()) && (cp.z() < myvertex2.z())) || ((cp.z() < myvertex1.z()) && (cp.z() > myvertex2.z()))) )
    { return true; }

    return false;
}

inline std::ostream &
operator<< (std::ostream &lhs, tribox &mybox)
{
    lhs<<"<"<<mybox.myvertex1<<"|"<<mybox.myvertex2<<">";
    return lhs;
}

#endif
