// This program was written by Munther Gdeisat and Miguel Arevallilo Herraez to program the two-dimensional unwrapper
// entitled "Fast two-dimensional phase-unwrapping algorithm based on sorting by 
// quality following a noncontinuous path"
// by  Miguel Arevallilo Herraez, David R. Burton, Michael J. Lalor, and Munther A. Gdeisat
// published in the Journal Applied Optics, Vol. 41, No. 35, pp. 7437, 2002.
// This program was written by Munther Gdeisat, Liverpool John Moores University, United Kingdom.
// Date 26th August 2007
// The wrapped phase map is assumed to be of double point data type. The resultant unwrapped phase map is also of double point type.
// When the mask is 0 this means that the Pixel is invalid (noisy or corrupted Pixel)
// This program takes into consideration the image wrap around problem encountered in MRI imaging.

#include <iostream>
#include <limits>
#include <algorithm>
#include "unwrap_util.h"
#include "unwrap_miguel.h"

// Pixel information
struct Pixel {
	int jumps;						// # of 2*pi to add to the Pixel to unwrap it
	int nPixels;					// # of Pixel in the Pixel group
	double quality;					// quality based on reliability
	Pixel *first, *last, *next;		// pointer to the first, last and next Pixel in the group
};

//the Edge is the line that connects two Pixels.
struct Edge {
	Pixel *p1,*p2;					// pointer to the first and second Pixel
	double quality;					// quality of the Edge and it depends on the two Pixels
	int jumps;						// # of 2*pi to add to one of the Pixels tounwrap it with respect to the second
}; 

inline bool EdgeComp(Edge const & a, Edge const & b) {
    return a.quality<b.quality;
}

int jump(double lVal, double rVal) {
	double difference = lVal - rVal;
	if (difference > 0.5)	return -1;
	else if (difference<-0.5)	return 1;
	return 0;
} 

void unwrap_miguel(nPhysD* phase, nPhysD* unwrap) {
    unsigned int dx=phase->getW();
    unsigned int dy=phase->getH();
    nPhysD quality(dx,dy,1.0);
    unwrap_miguel_quality(phase, unwrap, &quality);
}

void unwrap_miguel_quality(nPhysD* phase, nPhysD* unwrap, nPhysD* quality) {
    unsigned int dx=phase->getW();
    unsigned int dy=phase->getH();
	// initialize
	std::vector<Pixel> px(dx*dy);
	for (std::vector<Pixel>::iterator pix = px.begin() ; pix != px.end(); ++pix) {
		pix->jumps = 0;
		pix->nPixels = 1;
		pix->quality = 1.0;
		pix->first = &*pix;
		pix->last = &*pix;
		pix->next = NULL;
	}	
	// calculate quality
	for (unsigned int j = 1; j<dy -1; ++j) {
		for (unsigned int i = 1; i<dx - 1; ++i) {
			double H = grad(phase->point(i-1,j), phase->point(i,j)) - grad(phase->point(i,j), phase->point(i+1,j));
			double V = grad(phase->point(i,j-1), phase->point(i,j)) - grad(phase->point(i,j), phase->point(i,j+1));
			double D1 = grad(phase->point(i-1,j-1), phase->point(i,j)) - grad(phase->point(i,j), phase->point(i+1,j+1));
			double D2 = grad(phase->point(i+1,j-1), phase->point(i,j)) - grad(phase->point(i,j), phase->point(i-1,j+1));
			px[i+j*dx].quality = H*H + V*V + D1*D1 + D2*D2;
		}
    }
	for (unsigned int i = 0; i<dx*dy; ++i) {
	    px[i].quality /= quality->point(i);
	}
	
	// calculate Edges
	std::vector<Edge> edge((dx-1)*dy+(dy-1)*dx); // look the 4 for below!
	int k=0;
	for (unsigned int j = 0; j<dy; j++) {
		for (unsigned int i = 0; i<dx - 1; i++)  {
			edge[k].p1 = &px[i+j*dx];
			edge[k].p2 = &px[i+1+j*dx];
			edge[k].quality = px[i+j*dx].quality + px[i+1+j*dx].quality;
			edge[k].jumps = jump(phase->point(i,j),phase->point(i+1,j));
			k++;
		}
    }
	for (unsigned int j = 0; j<dy - 1; j++) {
		for (unsigned int i = 0; i<dx; i++) {
			edge[k].p1 = &px[i+j*dx];
			edge[k].p2 = &px[i+(1+j)*dx];
			edge[k].quality = px[i+j*dx].quality + px[i+(1+j)*dx].quality;
			edge[k].jumps = jump(phase->point(i,j),phase->point(i,j+1));
			k++;
		}
    }
	    
	//sort the Edges depending on their reiability. The Pixels with higher relibility (small value) first
	std::sort(edge.begin(), edge.end(), EdgeComp);
    
	//gather Pixels into groups
	for (std::vector<Edge>::iterator ed = edge.begin() ; ed != edge.end(); ++ed) {
		if (ed->p2->first != ed->p1->first) {
			// Pixel 2 is alone in its group merge this Pixel with Pixel 1 group 
			// and find the number of 2 pi to add to or subtract to unwrap it
			if ((ed->p2->next == NULL) && (ed->p2->first == ed->p2)) {
				ed->p1->first->last->next = ed->p2;
				ed->p1->first->last = ed->p2;
				(ed->p1->first->nPixels)++;
				ed->p2->first=ed->p1->first;
				ed->p2->jumps = ed->p1->jumps-ed->jumps;
			} else if ((ed->p1->next == NULL) && (ed->p1->first == ed->p1)) {
				// Pixel 1 is alone in its group merge this Pixel with Pixel 2 group 
				// and find the number of 2 pi to add to or subtract to unwrap it
				ed->p2->first->last->next = ed->p1;
				ed->p2->first->last = ed->p1;
				(ed->p2->first->nPixels)++;
				ed->p1->first = ed->p2->first;
				ed->p1->jumps = ed->p2->jumps+ed->jumps;
			} else { //Pixel 1 and Pixel 2 both have groups
				Pixel *group1 = ed->p1->first;
				Pixel *group2 = ed->p2->first;
				// if the no. of Pixels in Pixel 1 group is larger than the no. of Pixels in Pixel 2 group.  
				// Merge Pixel 2 group to Pixel 1 group and find the number of wraps between Pixel 2
				// group and Pixel 1 group to unwrap Pixel 2 group with respect to Pixel 1 group.  
				// the no. of wraps will be added to Pixel 2 group in the future
				if (group1->nPixels > group2->nPixels) {
					//merge Pixel 2 with Pixel 1 group
					group1->last->next = group2;
					group1->last = group2->last;
					group1->nPixels = group1->nPixels + group2->nPixels;
					int incr = ed->p1->jumps-ed->jumps - ed->p2->jumps;
					//merge the other Pixels in Pixel 2 group to Pixel 1 group
					while (group2 != NULL) {
						group2->first = group1;
						group2->jumps += incr;
						group2 = group2->next;
					}
				} else {
					// if the no. of Pixels in Pixel 2 group is larger than the no. of Pixels in Pixel 1 group.  
					// Merge Pixel 1 group to Pixel 2 group and find the number of wraps between Pixel 2
					// group and Pixel 1 group to unwrap Pixel 1 group with respect to Pixel 2 group.
					// the no. of wraps will be added to Pixel 1 group in the future
					//merge Pixel 1 with Pixel 2 group
					group2->last->next = group1;
					group2->last = group1->last;
					group2->nPixels = group2->nPixels + group1->nPixels;
					int incr = ed->p2->jumps + ed->jumps - ed->p1->jumps;
					//merge the other Pixels in Pixel 2 group to Pixel 1 group
					while (group1 != NULL) {
						group1->first = group2;
						group1->jumps += incr;
						group1 = group1->next;
					} // while
					
                } // else
            } //else
        } //if
	}
	// fill the unwrapped map
	for (unsigned int i=0; i<dx*dy; i++) unwrap->set(i,phase->point(i)+px[i].jumps);
}
