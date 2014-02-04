#include <math.h>
#include "unwrap_util.h"

double grad(double p1, double p2) {
	double  r = p1 - p2;
	if (r > 0.5) return r-1.0;
	if (r < -0.5) return r+1.0;
	return r;
}

/* Returns false if no pixels left, true otherwise */
bool GetNextOneToUnwrap(unsigned int *a, unsigned int *b, unsigned int *index_list, unsigned int *num_index, unsigned int dx) {
	unsigned int index;
	if (*num_index ==0) 
		return false;
	index = index_list[*num_index - 1];
	*a = index%dx;
	*b = index/dx;
	--(*num_index);
	return true;
}

void InsertList(nPhysD *soln, double val, nPhysD *qual_map, nPhysBits *bits, unsigned int index, 
                unsigned int *index_list, unsigned int *num_index) {
    
	soln->set(index,val);
	/* otherwise, add to list */
    /* insert in list in order from lowest to highest quality */
    if (*num_index == 0) {
        index_list[0] = index;
    } else {
        if (qual_map->point(index) <= qual_map->point(index_list[0])) {
            /* insert at top of list */
            for (unsigned int i=*num_index; i>0; i--) 
                index_list[i] = index_list[i-1];
            index_list[0] = index;
        } else if (qual_map->point(index) > qual_map->point(index_list[*num_index - 1])) {
            /* insert at bottom */
            index_list[*num_index] = index;
        } else {   /* insert in middle */
            unsigned int top = 0;
            unsigned int bot = *num_index - 1;
            while (bot - top > 1) {
                unsigned int mid = (top + bot)/2;
                if (qual_map->point(index) <= qual_map->point(index_list[mid]))  bot = mid;
                else  top = mid;
            }
            for (unsigned int i=(*num_index); i>top+1; i--) 
                index_list[i] = index_list[i-1];
            index_list[top+1] = index;
        }
    }
    ++(*num_index);
	bits->set(index, bits->point(index) | UNWRAPPED);
}

/* Insert the four neighboring pixels of the given pixel */
/* (x,y) into the list.  The quality value of the given  */
/* pixel is "val".                                       */
void UpdateList(nPhysD *qual_map, unsigned int x, unsigned int y, double val, nPhysD *phase,
                nPhysD *soln, nPhysBits *bits, unsigned int *index_list, unsigned int *num_index) {
    unsigned int dx=phase->getW();
    unsigned int dy=phase->getH();

	double  my_val;
	
	if (x > 0 && !(bits->point(x-1,y) & (AVOID | UNWRAPPED))) {
		my_val = val + grad(phase->point(x-1,y), phase->point(x,y));
		InsertList(soln, my_val, qual_map, bits, y*dx+x-1, index_list, num_index);
	}
	
	if (x < dx-1  && !(bits->point(x+1,y) & (AVOID | UNWRAPPED))) {
		my_val = val - grad(phase->point(x,y), phase->point(x+1,y));
		InsertList(soln, my_val, qual_map, bits, y*dx+x+1, index_list, num_index);
	}
	
	if (y > 0 && !(bits->point(x,y-1) & (AVOID | UNWRAPPED))) {
		my_val = val + grad(phase->point(x,y-1), phase->point(x,y));
		InsertList(soln, my_val, qual_map, bits, (y-1)*dx+x, index_list, num_index);
	}
	
	if (y < dy-1 && !(bits->point(x,y+1) & (AVOID | UNWRAPPED))) {
		my_val = val - grad(phase->point(x,y), phase->point(x,y+1));
		InsertList(soln, my_val, qual_map, bits, (y+1)*dx+x, index_list, num_index);
	}
}
