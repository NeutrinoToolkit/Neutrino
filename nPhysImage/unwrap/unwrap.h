#ifndef unwrap_h_
#define unwrap_h_
#include "../nPhysImageF.h"

inline double grad(double p1, double p2) {
    double  r = p1 - p2;
    if (r > 0.5) return r-1.0;
    if (r < -0.5) return r+1.0;
    return r;
}

namespace unwrap {
  void goldstein (physD &, physD &);
  void goldstein (physD &, physD &, physD &); // quality: >0 order of the flood fill, <=0 (or NaN) are cuts
  void miguel (physD &, physD &);
  void miguel_quality (physD &, physD &, physD&);
  void quality (physD &, physD &, physD &);
  // from_uphase: each line starts from the value already in uphase (second pass of H+V or V+H)
  void simple_h (physD&, physD&, bool from_uphase=false);
  void simple_v (physD&, physD&, bool from_uphase=false);
}

#endif
