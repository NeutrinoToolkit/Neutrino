#ifndef __unwrap_util
#define __unwrap_util
#include "../nPhysImageF.h"
#include <vector>

#define POSITIVE    1
#define NEGATIVE    2
#define DONE        4
#define ACTIVE      8
#define BRANCH_CUT  16
#define BORDER      32
#define UNWRAPPED   64
#define RESIDUE     (POSITIVE | NEGATIVE)
#define AVOID       (BRANCH_CUT | BORDER)

double grad(double, double);

int jump(double, double);

typedef nPhysImageF<unsigned char> nPhysBits;

bool GetNextOneToUnwrap(unsigned int &, unsigned int &, std::vector<unsigned int> &, unsigned int &, unsigned int);

void UpdateList(nPhysD *, unsigned int, unsigned int, double, nPhysD *, nPhysD *, nPhysBits *, std::vector<unsigned int> &, unsigned int &);

void InsertList(nPhysD *, double, nPhysD *, nPhysBits *, unsigned int, std::vector<unsigned int> &, unsigned int &);
#endif
