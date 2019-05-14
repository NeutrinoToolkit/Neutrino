/*
 *
 *    Copyright (C) 2013 Alessandro Flacco, Tommaso Vinci All Rights Reserved
 * 
 *    This file is part of nPhysImage library.
 *
 *    nPhysImage is free software: you can redistribute it and/or modndy
 *    it under the terms of the GNU Lesser General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    nPhysImage is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public License
 *    along with neutrino.  If not, see <http://www.gnu.org/licenses/>.
 *
 *    Contact Information: 
 *	Alessandro Flacco <alessandro.flacco@polytechnique.edu>
 *	Tommaso Vinci <tommaso.vinci@polytechnique.edu>
 *
 */
// functions for wavelet analysis
#include <iostream>
#include <cstring>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_const_mksa.h>
#include <gsl/gsl_dht.h>
#include <gsl/gsl_sf_bessel.h>


#include <list>

#include "nPhysImageF.h"
#include "nPhysMaths.h"
#include "mcomplex.h"
#include "bidimvec.h"

#ifdef HAVE_LIBCLFFT
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include "clFFT.h"
#endif

#ifndef __nPhysWave_h
#define __nPhysWave_h

namespace physWave {


#ifdef HAVE_LIBCLFFT
std::string CHECK_OPENCL_ERROR(cl_int err);
#define check_opencl_error(__err_num, __err_msg) if (__err_num != CL_SUCCESS) {WARNING(__err_num << " " << CHECK_OPENCL_ERROR(__err_num) << " " << __err_msg); throw phys_fileerror(__err_msg);};
#endif

enum unwrap_strategy {GOLDSTEIN, QUALITY, SIMPLE_HV, SIMPLE_VH, MIGUEL, MIGUEL_QUALITY};

class wavelet_params {
public:
    wavelet_params() :
    data(NULL),	opencl_unit(0), iter(0), iter_ptr(&iter) {
        DEBUG("wavelet_params created");
    }
    
    physD *data;

	double init_angle;
	double end_angle;
	size_t n_angles;

	double init_lambda;
	double end_lambda;
	size_t n_lambdas;

	int thickness;
	double damp;
	
    int opencl_unit;

	int iter;
	int *iter_ptr;

    std::map<std::string, physD*> olist;
};

void phys_wavelet_field_2D_morlet(wavelet_params &);

void phys_wavelet_field_2D_morlet_opencl(wavelet_params &);

// traslation functions
void phys_wavelet_trasl_cpu(void *, int &);

int openclEnabled();

unsigned int opencl_closest_size(unsigned int);

vec2i opencl_closest_size(vec2i);

#ifdef HAVE_LIBCLFFT
std::pair<cl_platform_id,cl_device_id> get_platform_device_opencl(int);
std::string get_platform_device_info_opencl(int);
#endif

void phys_wavelet_trasl_opencl(void *, int &);

// unwrap methods
void phys_phase_unwrap(nPhysImageF<double> &, nPhysImageF<double> &, enum unwrap_strategy, nPhysImageF<double>&);

// carrier subtraction
void phys_subtract_carrier (nPhysImageF<double> &, double, double);

// create a synthetic interferogram from phase and quality
nPhysImageF<double> phys_synthetic_interferogram (nPhysImageF<double> *, nPhysImageF<double> *);

bidimvec<double> phys_guess_carrier(nPhysImageF<double> &, double=1.0);

// integral inversions

// numerical approach for inversion
enum inversion_algo {ABEL = 10, ABEL_HF = 20};
enum inversion_physics { ABEL_GAS, ABEL_PLASMA, ABEL_NONE };
struct abel_params_str {
	abel_params_str()
		: iimage(NULL), oimage(NULL), iter_ptr(0)
	{ }

    physD *iimage;
    physD *oimage;
//     physD rimage;
    
    std::vector<vec2i> iaxis;
	phys_direction idir;
	inversion_algo ialgo;
	inversion_physics iphysics;
	
	int iter;
	int *iter_ptr;
};
typedef struct abel_params_str abel_params;

// main inversion function
void phys_invert_abel(abel_params &);

// inline inversion maths
inline void phys_invert_abel_1D(std::vector<double> &ivec, std::vector<double> &ovec)
{

	size_t size=ivec.size();
	// FIRST element of *ptr MUST BE on symmetry axis
// 	double integral;
	int size_dy = size-1;

	if (size==0)
		return;

	// 0. init 
	//memset(ovec, 0, size*sizeof(sizeof(double))); ??????
	std::fill(ovec.begin(), ovec.end(), 0);

	// 1. dy - mind that this shifts the function to semi-integer indexes
	std::vector<double> dy(size_dy), func_x(size_dy);
	for (int i=0; i<size_dy; i++)
		dy[i] = (ivec[i+1] - ivec[i]);
		//dy[i] = 2.*__ifg_pi*(y[i+1] - y[i]); taken care of elsewhere

	// 2. integral
	int R0_idx = size-1; 			// considering R=0 for the first index
	for (int R_idx=0; R_idx<R0_idx; R_idx++) {		// looping on function
		std::fill(func_x.begin(), func_x.end(), 0);
		for (int x_idx = (R_idx)+1; x_idx < R0_idx; x_idx++) {
			func_x[R_idx] = 0.5*(dy[x_idx]+dy[x_idx-1])/sqrt(pow(x_idx-0.5,2) - pow(R_idx,2));
			ovec[R_idx] += func_x[R_idx];
		}
		// first index is R_idx, last index is (R0_idx-2): trapezioid integral correction

		ovec[R_idx] -= 0.5*(func_x[R_idx]+func_x[R0_idx-1]);
		ovec[R_idx] /= -M_PI;

	}

	// shift of half index on the right
	double prev_p = ovec[0];
	for (size_t ii=1; ii<size; ii++) {
		ovec[ii] = .5*(prev_p+ovec[ii]);
		prev_p = ovec[ii];
	}
	
	// zero axis
	ovec[0] = 0;

	return;
}



//bessel_alloc_t *bessel_allocate();
//void bessel_free(bessel_alloc_t *);


inline void phys_invert_abelHF_1D(std::vector<double> &ivec, std::vector<double> &ovec, std::vector<double> &lut, std::vector<double> &Fivec, fftw_plan &r2rplan)
{
	if (ivec.size()!=ovec.size() || ivec.size()==0)
		return;

	// 0. init
	std::fill(ovec.begin(), ovec.end(), 0);

	// check Bessel LUT
	size_t N=ivec.size();
    if (N*N != lut.size()) {
		// recalculate lut
		DEBUG(10,"recalculate Bessel LUT for N="<<N<<"...");
        lut.resize(N*N);
		for (size_t j=0; j<N; j++) {
			for (size_t k=0; k<j+1; k++) {
				// direct cosine
				//lut.lut[k*j] = gsl_sf_bessel_J0(j*k/(2.*N+1.));

				// DCT 
				// (change in integration variable to respect FFTW-DCT definition)
                lut[k*j] = gsl_sf_bessel_J0(M_PI*j*k/N);
			}
		}
		DEBUG(10,"done");
	}

    if (Fivec.size()==0) {
        Fivec.resize(N);
        r2rplan = fftw_plan_r2r_1d(N, &ivec[0], &Fivec[0], FFTW_REDFT00, FFTW_ESTIMATE);
    }
	// sara' da trovare una soluzione rapida per il plan
	fftw_execute(r2rplan);

	for (size_t j=0; j<N; j++) {
	
	//	for (int i=-N; i<N; i++) { // this was for Cosinus transform
			for (size_t k=0; k<N; k++) {
				// direct cosinus
				//inv_y[j] += 2 * in[i] * cos(i*k/(2.*N+1.)) * k * gsl_sf_bessel_J0(j*k/(2.*N+1.));
				//ovec[j] += ivec[abs(i)] * cos(i*k/(2.*N+1.)) * k * lut.lut[j*k];
				//ovec[j] += ivec[abs(i)] * lut.cos_lut[abs(i)*k] * k * lut.lut[j*k];

				// fast DCT
				//ovec[j] += Fivec[k] * k * lut.lut[j*k];
                ovec[j] += Fivec[k] * (M_PI/N) * k * lut[j*k];
            }
	//	}
		//ovec[j] *= (1/(2*3.14*pow(2.*N+1.,2)));
		ovec[j] *= 1./(2.*N);
	}


/*	for (size_t j=0; j<size; j++) {
	
		for (int i=-N; i<N; i++) {
			for (int k=0; k<N; k++) {
				//inv_y[j] += 2 * in[i] * cos(i*k/(2.*N+1.)) * k * gsl_sf_bessel_J0(j*k/(2.*N+1.));
				//ovec[j] += ivec[abs(i)] * cos(i*k/(2.*N+1.)) * k * lut.lut[j*k];
				ovec[j] += ivec[abs(i)] * lut.cos_lut[abs(i)*k] * k * lut.lut[j*k];
			}
		}
		ovec[j] *= (1/(2*3.14*pow(2.*N+1.,2)));
	}*/

	return;
}


// inversion physics
void phys_apply_inversion_plasma(nPhysImageF<double> &, double, double);
void phys_apply_inversion_gas(nPhysImageF<double> &, double, double, double);
void phys_apply_inversion_protons(nPhysImageF<double> &, double, double, double, double);

}

#endif
