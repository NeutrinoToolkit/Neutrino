#ifndef __LIST
#define __LIST
#include "../nPhysImageF.h"

#define POSITIVE    0x01   /* 1st bit */
#define NEGATIVE    0x02   /* 2nd bit */
#define DONE        0x04   /* 3rd bit */
#define ACTIVE      0x08   /* 4th bit */
#define BRANCH_CUT  0x10   /* 5th bit */
#define BORDER      0x20   /* 6th bit */
#define UNWRAPPED   0x40   /* 7th bit */
#define RESIDUE     (POSITIVE | NEGATIVE)
#define AVOID       (BRANCH_CUT | BORDER)

double grad(double, double);
typedef nPhysImageF<unsigned char> nPhysBits;


bool GetNextOneToUnwrap(unsigned int *, unsigned int *, unsigned int *, unsigned int *, unsigned int);

void UpdateList(nPhysD *, unsigned int, unsigned int, double, nPhysD *, nPhysD *, nPhysBits *, unsigned int *, unsigned int *);
    
void InsertList(nPhysD *, double, nPhysD *, nPhysBits *, unsigned int, unsigned int *, unsigned int *);
#endif
