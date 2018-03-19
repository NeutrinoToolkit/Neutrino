/*
 * A class holding the definition for a thomson parabola (or any) generic system
 *
 * This file is NOT open source. 
 *
 * Any use is FORBIDDEN unless by written permission
 *
 * (C) Alessandro Flacco 2014
 *
 */

#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <sstream>

// gsl
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv.h>
#include <gsl/gsl_const_mksa.h>

// libconfig
//#include <libconfig.h++>

// memcpy
#include <string.h>

// non ricordo perche' mai l'ho levato...
#include "bidimvec.h"
#include "tridimvec.h"
#include "tribox.h"
#include "scalarMap.h"

#define PROTON_ION_MASS GSL_CONST_MKSA_MASS_PROTON
#define CARBON12_ION_MASS 12*GSL_CONST_MKSA_UNIFIED_ATOMIC_MASS
#define CARBON13_ION_MASS 13*GSL_CONST_MKSA_UNIFIED_ATOMIC_MASS
#define NITROGEN_ION_MASS 14*GSL_CONST_MKSA_UNIFIED_ATOMIC_MASS
#define OXYGEN_ION_MASS 16*GSL_CONST_MKSA_UNIFIED_ATOMIC_MASS

#define _cspeed GSL_CONST_MKSA_SPEED_OF_LIGHT

#define mm 1e-3
#define cm 1e-2
#define MeV ((double)1e6*GSL_CONST_MKSA_ELECTRON_CHARGE)
#define ns 1e-9
#define ps 1e-12

typedef tridimvec<double> f3point;
typedef std::vector<tridimvec<double> > f3vec;
typedef bidimvec<double> f2point;
typedef std::vector<bidimvec<double> > f2vec;




using namespace std;

#ifndef __tpSystem_h
#define __tpSystem_h

struct ionImpact {
	ionImpact()
		: traj_dump(false)
	{ }

	double charge;
	double mass;
	double energy;
	double gamma;
	f3point impact;

	// come cazzo la salvo la traiettoria? (qui in versione cingolato)
	bool traj_dump;
	std::vector<std::array<float, 7> > traj;
	
};

int lorentzStep(double, const double *, double *, void *);
int lorentzStepRelativistic(double, const double *, double *, void *);

class tpSystem {

public:
	tpSystem()
		: valid_system(false), valid_config(false), relativistic(false)
	{ sim_box = new tribox; Efield = new scalarMap; Bfield = new scalarMap; }

	~tpSystem()
	{ }

    //void parseConfig (const char *);
    //void writeConfig(const char *);

	void getImpact(struct ionImpact *);

	bool isValid()
	{ return valid_system; }
	bool isConfigured()
	{ return valid_config; }

	scalarMap *Efield, *Bfield;
	tribox *sim_box;

	ionImpact *my_ion;
	bool relativistic;

private:

	bool valid_system, valid_config;

};


#endif
