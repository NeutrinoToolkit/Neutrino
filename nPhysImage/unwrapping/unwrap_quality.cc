/*
 * mainqual.c -- phase unwrapping by quality-guided path following
 */
#include "unwrap_util.h"
#include "unwrap_quality.h"

void unwrap_quality (nPhysD *phase, nPhysD *soln, nPhysD *qual_map) {    
    unsigned int dx=phase->getW();
    unsigned int dy=phase->getH();
	
	nPhysBits bitflags(dx,dy,0,"bitflags");
	unsigned int *index_list = new unsigned int[dx*dy + 1]();
	
	/* find starting point */
	unsigned int num_index = 0;
	for (unsigned int j=0; j<dy; j++) {
		for (unsigned int i=0; i<dx; i++) {
			if (!(bitflags.point(i,j) & (AVOID | UNWRAPPED))) {
				soln->set(i,j,phase->point(i,j));
				bitflags.set(i,j,bitflags.point(i,j) | UNWRAPPED);
				UpdateList(qual_map, i, j, phase->point(i,j), phase, soln, &bitflags, index_list, &num_index);
				while (num_index > 0) {
					unsigned int a,b;
					if (GetNextOneToUnwrap(&a, &b, index_list, &num_index, dx)) { 
                        bitflags.set(a,b,bitflags.point(a,b) | UNWRAPPED);
                        UpdateList(qual_map, a, b, soln->point(a,b), phase, soln, &bitflags, index_list, &num_index);
                    }
				}
			}
		}
	}
	delete[] index_list;    
}
