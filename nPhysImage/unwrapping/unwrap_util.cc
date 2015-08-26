#include <math.h>
#include "unwrap_util.h"

double grad(double p1, double p2) {
	double  r = p1 - p2;
	if (r > 0.5) return r-1.0;
	if (r < -0.5) return r+1.0;
	return r;
}

