/*
 * maingold.c - phase unwrapping by means of residues & branch cuts
 */
#include "unwrap_util.h"
#include "unwrap_goldstein.h"

#define POSITIVE    1
#define NEGATIVE    2
#define DONE        4
#define ACTIVE      8
#define BRANCH_CUT  16
#define BORDER      32
#define UNWRAPPED   64
#define RESIDUE     (POSITIVE | NEGATIVE)
#define AVOID       (BRANCH_CUT | BORDER)

typedef nPhysImageF<unsigned char> nPhysBits;

/* Returns false if no pixels left, true otherwise */
bool GetNextOneToUnwrap(unsigned int &a, unsigned int &b, std::vector<unsigned int> &index_list, unsigned int &num_index, unsigned int dx) {
	unsigned int index;
	if (num_index ==0) 
		return false;
	index = index_list[num_index - 1];
	a = index%dx;
	b = index/dx;
	num_index--;
	return true;
}

void InsertList(nPhysD *soln, double val, nPhysD *qual_map, nPhysBits *bits, unsigned int index, 
                std::vector<unsigned int> &index_list, unsigned int &num_index) {
    
	soln->set(index,val);
	/* otherwise, add to list */
    /* insert in list in order from lowest to highest quality */
    if (num_index == 0) {
        index_list[0] = index;
    } else {
        if (qual_map->point(index) <= qual_map->point(index_list[0])) {
            /* insert at top of list */
            for (unsigned int i=num_index; i>0; i--) 
                index_list[i] = index_list[i-1];
            index_list[0] = index;
        } else if (qual_map->point(index) > qual_map->point(index_list[num_index - 1])) {
            /* insert at bottom */
            index_list[num_index] = index;
        } else {   /* insert in middle */
            unsigned int top = 0;
            unsigned int bot = num_index - 1;
            while (bot - top > 1) {
                unsigned int mid = (top + bot)/2;
                if (qual_map->point(index) <= qual_map->point(index_list[mid]))  bot = mid;
                else  top = mid;
            }
            for (unsigned int i=(num_index); i>top+1; i--) 
                index_list[i] = index_list[i-1];
            index_list[top+1] = index;
        }
    }
    num_index++;
// 	DEBUG((int)bits->point(index) << " | " << (int)UNWRAPPED << " = " << (int) (bits->point(index) | UNWRAPPED));	
	bits->set(index, bits->point(index) | UNWRAPPED);
	
}

/* Insert the four neighboring pixels of the given pixel */
/* (x,y) into the list.  The quality value of the given  */
/* pixel is "val".                                       */
void UpdateList(nPhysD *qual_map, unsigned int x, unsigned int y, double val, nPhysD *phase,
                nPhysD *soln, nPhysBits *bits, std::vector<unsigned int> &index_list, unsigned int &num_index) {
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

/* Return the squared distance between the pixel (a,b) and the */
/* nearest border pixel.  The border pixels are encoded in the */
/* bits bits by the value of "border_code".               */
int DistToBorder(nPhysBits *bits, int a, int b, int *ra, int *rb) {
	unsigned int dx=bits->getW();
    unsigned int dy=bits->getH();
	int  besta=-1, bestb=-1, best_dist=-1;
	bool found;
	*ra = *rb = 0;
	for (unsigned int bs=0; bs<dx + dy; bs++) {
		found = false;
		best_dist = dx+dy;  /* initialize to large value */
		/* search boxes of increasing size until border pixel found */
		for (int j=b - bs; j<=b + (int)bs; j++) {
			for (int i=a - bs; i<=a + (int)bs; i++) {
				if (i<=0 || i>=(int)dx - 1 || j<=0 || j>=(int)dy - 1 || (bits->point(i,j) & BORDER)) {
					found = true;
					int dist2 = (j - b)*(j - b) + (i - a)*(i - a);
					if (dist2 < best_dist) {
						best_dist = dist2;
						besta = i;
						bestb = j;
					}           
				}
			}
		}
		if (found) {
			*ra = besta;
			*rb = bestb;
			break;
		}
	}
	return best_dist;
} 

/* Place a branch cut in the bits bits from pixel (a,b) */
/* to pixel (c,d).  The bit for the branch cut pixels is     */
/* given by the value of "code".                             */
void PlaceCut(nPhysBits *bits, int a, int b, int c, int d) {
	/* residue location is upper-left corner of 4-square */
	if (c > a && a > 0) a++;
	else if (c < a && c > 0) c++;
	if (d > b && b > 0) b++;
	else if (d < b && d > 0) d++;
	
	if (a==c && b==d) {
		bits->set(a,b,bits->point(a,b) | BRANCH_CUT);
		return;
	}
	int m = (a < c) ? c - a : a - c;
	int n = (b < d) ? d - b : b - d;
	if (m > n) {
		int istep = (a < c) ? +1 : -1;
		double r = ((double)(d - b))/((double)(c - a));
		for (int i=a; i!=c+istep; i+=istep) {
			int j = b + (i - a)*r + 0.5;
			bits->set(i,j,bits->point(i,j) | BRANCH_CUT);
		}
	} else {   /* n < m */
		int jstep = (b < d) ? +1 : -1;
		double r = ((double)(c - a))/((double)(d - b));
		for (int j=b; j!=d+jstep; j+=jstep) {
			int i = a + (j - b)*r + 0.5;
			bits->set(i,j,bits->point(i,j) | BRANCH_CUT);
		}
	}
}


void unwrap_goldstein (nPhysD *phase, nPhysD *soln) {
	unsigned int dx=phase->getW();
    unsigned int dy=phase->getH();

	nPhysD qual_map(dx,dy,0.0,"qual");
	nPhysBits bits(dx,dy,0,"bits");
	
	int NumRes=0;
	for (unsigned int j=0; j<dy - 1; j++) {
		for (unsigned int i=0; i<dx - 1; i++) {
			if (!((bits.point(i,j) & AVOID)   || (bits.point(i+1,j) & AVOID)
			 || (bits.point(i+1,j+1) & AVOID) || (bits.point(i,j+1) & AVOID))) {
				double r = grad(phase->point(i+1,j), phase->point(i,j))
						 + grad(phase->point(i+1,j+1), phase->point(i+1,j))
						 + grad(phase->point(i,j+1), phase->point(i+1,j+1))
						 + grad(phase->point(i,j), phase->point(i,j+1));
				if (r > 0.01) bits.set(i,j,bits.point(i,j) | POSITIVE);
				else if (r < -0.01) bits.set(i,j,bits.point(i,j) | NEGATIVE);
				if (r*r > 0.01) NumRes++;
			}
		}
	}

	DEBUG(NumRes << " Residues" << std::endl);
	
	for (unsigned int j=0; j<dy; j++) {
		for (unsigned int i=0; i<dx; i++) {
			int kk = 0;
			if ((bits.point(i,j) & POSITIVE)) {
				if (i<dx-1 && (bits.point(i+1,j) & NEGATIVE)) kk=i+1+j*dx;
				else if (j<dy-1) {
					if ((bits.point(i,j+1) & NEGATIVE)) kk=i+(j+1)*dx;
				}
			} else if ((bits.point(i,j) & NEGATIVE)) {
				if (i<dx-1 && (bits.point(i+1,j) & POSITIVE)) kk=i+1+j*dx;
				else if (j<dy-1) {
					if ((bits.point(i,j+1) & POSITIVE)) kk=i+(j+1)*dx;
				}
			}
			if (kk) {
				DEBUG("Connecting dipoles " << i << "," << j << " " << kk%dx << "," << kk/dx);
				PlaceCut(&bits, i, j, kk%dx, kk/dx);
				bits.set(i,j,bits.point(i,j) & (~(RESIDUE)));
				bits.set(kk,bits.point(kk) & (~(RESIDUE)));
			}
		}
	}
	DEBUG(NumRes << " Residues after Dipole" << std::endl);

	int MaxCutLen = dx + dy;
	
	int ri, rj;
	int rim_i=0, rim_j=0, near_i=0, near_j=0;

	int max_active = NumRes+10;
	
	std::vector<unsigned int> active_list(max_active + 1);
	int num_active = 0;
	
	/* branch cuts */
	for (unsigned int j=0; j<dy; j++) {
		for (unsigned int i=0; i<dx; i++) {
			if ((bits.point(i,j) & RESIDUE) && !(bits.point(i,j) & DONE)) {
				bits.set(i,j,(bits.point(i,j) | DONE) | ACTIVE);   /* turn on ACTIVE and DONE flag */
				int charge = (bits.point(i,j) & POSITIVE) ? 1 : -1;
				active_list[num_active++] = i+j*dx;
				if (num_active > max_active) num_active = max_active;
				for (int bodx = 3; bodx<=MaxCutLen && charge!=0; bodx += 2) {
					int bs2 = bodx/2;
					for (int ka=0; ka<num_active && charge!=0; ka++) {
						int boxctr_i = active_list[ka]%dx;
						int boxctr_j = active_list[ka]/dx;
						for (int jj=boxctr_j - bs2; jj<=boxctr_j + bs2 && charge!=0; jj++) {
							for (int ii=boxctr_i - bs2; ii<=boxctr_i + bs2 && charge!=0; ii++) {
								if (ii>=0 && ii<(int)dx && jj>=0 && jj<(int)dy) { 
									if (bits.point(ii,jj) & BORDER) {
										charge = 0;
										DistToBorder(&bits, boxctr_i, boxctr_j, &ri, &rj);
										PlaceCut(&bits, ri, rj, boxctr_i, boxctr_j);
									} else if ((bits.point(ii,jj) & RESIDUE) && !(bits.point(ii,jj) & ACTIVE)) {
										if (!(bits.point(ii,jj) & DONE)) {
											charge += (bits.point(ii,jj) & POSITIVE) ? 1 : -1;
											bits.set(ii,jj,bits.point(ii,jj) | DONE);
										}
										DEBUG("connected two");
										active_list[num_active++] = ii+dx*jj;
										if (num_active > max_active) num_active = max_active;
										bits.set(ii,jj,bits.point(ii,jj) | ACTIVE);
										PlaceCut(&bits, ii, jj, boxctr_i, boxctr_j);
									}
								}
							}
						}
					}
				}
				
				if (charge == 0) {   /* connect branch cuts to rim */
					/* mark all active pixels inactive */
					for (int ka=0; ka<num_active; ka++) 
						bits.set(active_list[ka],bits.point(active_list[ka]) & ~ACTIVE);  /* turn flag ACTIVE off */
				} else {
					int min_dist = dx + dy;  /* large value */
					for (int ka=0; ka<num_active; ka++) {
						int ii = active_list[ka]%dx;
						int jj = active_list[ka]/dx;
						int dist = DistToBorder(&bits, ii, jj, &ri, &rj);
						if (dist<min_dist) {
							min_dist = dist;
							near_i = ii;
							near_j = jj;
							rim_i = ri;
							rim_j = rj;
						}
					} 
					PlaceCut(&bits, near_i, near_j, rim_i, rim_j);
				}
			}
		}
	}
	
	/*  UNWRAP AROUND CUTS */
	unsigned int  a, b, num_pieces=0;
	double  value;
	unsigned int num_index=0;
	
	std::vector<unsigned int> index_list(dx*dy + 1 + 1);
		
	/* find starting point */
	int n = 0;
	for (unsigned int j=0; j<dy; j++) {
		for (unsigned int i=0; i<dx; i++) {
			if (!(bits.point(i,j) & (AVOID | UNWRAPPED))) {
				bits.set(i,j,bits.point(i,j) | UNWRAPPED);
				++num_pieces;
				value = phase->point(i,j);
				soln->set(i,j,value);
				UpdateList(&qual_map, i, j, value, phase, soln, &bits, index_list, num_index);
				while (num_index > 0) {
					++n;
					if (GetNextOneToUnwrap(a, b, index_list, num_index, dx)) {
						bits.set(a,b,bits.point(a,b) | UNWRAPPED);
						value = soln->point(a,b);        
						UpdateList(&qual_map, a, b, value, phase, soln, &bits, index_list, num_index);
					}
				}
			}
		}
	}
	/* unwrap branch cut pixels */
	for (unsigned int j=1; j<dy; j++) {
		for (unsigned int i=1; i<dx; i++) {
			if (bits.point(i,j) & AVOID) {
				if ((bits.point(i-1,j) & UNWRAPPED)) {
					soln->set(i,j,soln->point(i-1,j)+grad(phase->point(i,j),phase->point(i-1,j)));
					bits.set(i,j,bits.point(i,j) | UNWRAPPED);
				} else if ((bits.point(i,j-1) & UNWRAPPED)) {
					soln->set(i,j,soln->point(i,j-1)+grad(phase->point(i,j),phase->point(i,j-1)));
					bits.set(i,j,bits.point(i,j) | UNWRAPPED);
				}
			}
		}
	}
	
	DEBUG("Disconnected pieces : " << num_pieces << std::endl);
}
