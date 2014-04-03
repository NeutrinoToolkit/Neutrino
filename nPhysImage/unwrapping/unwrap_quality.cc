/*
 * mainqual.c -- phase unwrapping by quality-guided path following
 */
#include "unwrap_util.h"
#include "unwrap_quality.h"

void unwrap_quality (nPhysD *phase, nPhysD *soln, nPhysD *qual_map) {    
    unsigned int dx=phase->getW();
    unsigned int dy=phase->getH();
	
	nPhysBits bitflags(dx,dy,0,"bitflags");
	std::vector<unsigned int> index_list(dx*dy + 1);
	
	
    for (unsigned int j = 1; j<dy -1; ++j) {
        for (unsigned int i = 1; i<dx - 1; ++i) {
            double H = grad(phase->point(i-1,j), phase->point(i,j)) - grad(phase->point(i,j), phase->point(i+1,j));
            double V = grad(phase->point(i,j-1), phase->point(i,j)) - grad(phase->point(i,j), phase->point(i,j+1));
            double D1 = grad(phase->point(i-1,j-1), phase->point(i,j)) - grad(phase->point(i,j), phase->point(i+1,j+1));
            double D2 = grad(phase->point(i+1,j-1), phase->point(i,j)) - grad(phase->point(i,j), phase->point(i-1,j+1));
            qual_map->set(i,j, qual_map->point(i,j) / H*H + V*V + D1*D1 + D2*D2);
        }
    }
    qual_map->TscanBrightness();
	
	/* find starting point */
	unsigned int num_index = 0;
	for (unsigned int j=0; j<dy; j++) {
		for (unsigned int i=0; i<dx; i++) {
			if (!(bitflags.point(i,j) & (AVOID | UNWRAPPED))) {
				soln->set(i,j,phase->point(i,j));
				bitflags.set(i,j,bitflags.point(i,j) | UNWRAPPED);
				UpdateList(qual_map, i, j, phase->point(i,j), phase, soln, &bitflags, index_list, num_index);
				while (num_index > 0) {
					unsigned int a,b;
					if (GetNextOneToUnwrap(a, b, index_list, num_index, dx)) { 
                        bitflags.set(a,b,bitflags.point(a,b) | UNWRAPPED);
                        UpdateList(qual_map, a, b, soln->point(a,b), phase, soln, &bitflags, index_list, num_index);
                    }
				}
			}
		}
	}
}
