#include <iostream>
#include <limits>
#include <algorithm>
#include "unwrap.h"

inline int jump(double lVal, double rVal) {
    double difference = rVal - lVal;
    if (difference > 0.5)	return 1;
    else if (difference<-0.5)	return -1;
	return 0;
} 

inline double reliability(physD &phase, int i, int j) {
    double H = grad(phase.point(i-1,j), phase.point(i,j)) - grad(phase.point(i,j), phase.point(i+1,j));
    double V = grad(phase.point(i,j-1), phase.point(i,j)) - grad(phase.point(i,j), phase.point(i,j+1));
    double D1 = grad(phase.point(i-1,j-1), phase.point(i,j)) - grad(phase.point(i,j), phase.point(i+1,j+1));
    double D2 = grad(phase.point(i+1,j-1), phase.point(i,j)) - grad(phase.point(i,j), phase.point(i-1,j+1));
    return 1.0/sqrt(H*H + V*V + 0.5*D1*D1 + 0.5*D2*D2);
}
// Pixel information
struct Pixel {
    Pixel() : jumps(0), nPixels(1), quality(0.0), first(this),  last(this),  next(nullptr) {}
	int jumps;						// # of 2*pi to add to the Pixel to unwrap it
	int nPixels;					// # of Pixel in the Pixel group
	double quality;					// quality based on reliability
	Pixel *first, *last, *next;		// pointer to the first, last and next Pixel in the group
};

//the Edge is the line that connects two Pixels.
struct Edge {
    Edge() : p1(nullptr), p2(nullptr), quality(0.0), jumps(0) {}
	Pixel *p1,*p2;					// pointer to the first and second Pixel
	double quality;					// quality of the Edge and it depends on the two Pixels
	int jumps;						// # of 2*pi to add to one of the Pixels tounwrap it with respect to the second
}; 

inline bool EdgeComp(Edge const & a, Edge const & b) {
    return a.quality>b.quality;
}

void unwrap::miguel(physD &phase, physD &unwrap) {
    unsigned int dx=phase.getW();
    unsigned int dy=phase.getH();
    physD quality_miguel(dx,dy,1.0);
    for (unsigned int j = 1; j<dy -1; ++j) {
        for (unsigned int i = 1; i<dx - 1; ++i) {
            quality_miguel.set(i,j,reliability(phase, i,j));
        }
    }
    unwrap::quality(phase, unwrap, quality_miguel);
}

void unwrap::miguel_quality(physD& phase, physD& unwrap, physD& quality) {
    unsigned int dx=phase.getW();
    unsigned int dy=phase.getH();
    physD quality_miguel(dx,dy,1.0);
    for (unsigned int j = 0; j<dy; ++j) {
        for (unsigned int i = 0; i<dx; ++i) {
            if (j>0 && j<dy-1 && i>0 && i < dx-1) {
                quality_miguel.set(i,j,quality.point(i,j)*reliability(phase, i,j));
            } else{
                quality_miguel.set(i,j,quality.point(i+j*dx));
            }
        }
    }
    unwrap::quality(phase, unwrap, quality_miguel);
}

void unwrap::quality(physD& phase, physD& unwrap, physD& quality) {
    unsigned int dx=phase.getW();
    unsigned int dy=phase.getH();
    // initialize
    std::vector<Pixel> px(dx*dy);
    for (unsigned int i = 0; i<dx*dy; ++i) {
        px[i].jumps = 0;
        px[i].nPixels = 1;
        px[i].quality = quality.point(i);
        px[i].first = &px[i];
        px[i].last = &px[i];
        px[i].next = NULL;
    }

    // calculate Edges
    std::vector<Edge> edge((dx-1)*dy+(dy-1)*dx); // look the 4 for below!
    int k=0;
    for (unsigned int j = 0; j<dy; j++) {
        for (unsigned int i = 0; i<dx - 1; i++)  {
            edge[k].p1 = &px[i+j*dx];
            edge[k].p2 = &px[i+1+j*dx];
            edge[k].quality = px[i+j*dx].quality + px[i+1+j*dx].quality;
            edge[k].jumps = jump(phase.point(i,j),phase.point(i+1,j));
            k++;
        }
    }
    for (unsigned int j = 0; j<dy - 1; j++) {
        for (unsigned int i = 0; i<dx; i++) {
            edge[k].p1 = &px[i+j*dx];
            edge[k].p2 = &px[i+(1+j)*dx];
            edge[k].quality = px[i+j*dx].quality + px[i+(1+j)*dx].quality;
            edge[k].jumps = jump(phase.point(i,j),phase.point(i,j+1));
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
    for (unsigned int i=0; i<dx*dy; i++) unwrap.set(i,phase.point(i)+px[i].jumps);
}
