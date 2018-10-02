#ifndef __unwrap
#define __unwrap
#include "../nPhysImageF.h"

inline double grad(double p1, double p2) {
    double  r = p1 - p2;
    if (r > 0.5) return r-1.0;
    if (r < -0.5) return r+1.0;
    return r;
}

namespace unwrap {

void goldstein (physD &, physD &);
void miguel (physD &, physD &);
void miguel_quality (physD &, physD &, physD&);
void quality (physD &, physD &, physD &);
void simple_h (physD&, physD&);
void simple_v (physD&, physD&);

}

#endif
