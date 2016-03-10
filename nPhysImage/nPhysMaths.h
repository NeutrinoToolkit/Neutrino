/*
 *
 *    Copyright (C) 2013 Alessandro Flacco, Tommaso Vinci All Rights Reserved
 * 
 *    This file is part of nPhysImage library.
 *
 *    nPhysImage is free software: you can redistribute it and/or modify
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
// collection of math functions
#include <iostream>

#ifdef HAVE_LIBGSL
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_multifit.h>
#endif

#include "nPhysImageF.h"
#include "nPhysTools.h"
#include "mcomplex.h"

#define _phys_deg (0.017453f)

//using namespace std;

#ifndef __physMaths_h
#define __physMaths_h



inline void planeFit(nPhysD *pi, double *coeffs);

// ------------------ general purpose functions for wavelet analysis ------------------------
inline mcomplex 
morlet(double lambda, double tau, double x);



inline mcomplex 
morlet(double lambda, double tau_l, double tau_v, double x, double y, double rotation);


inline void
phys_generate_meshgrid(int x1, int x2, int y1, int y2, nPhysImageF<int> &xx, nPhysImageF<int> &yy);

// meshgrids

typedef struct meshgrid_data_str meshgrid_data;
struct meshgrid_data_str {
	double x1, x2, y1, y2;
	int nx, ny;
};

inline void
phys_generate_meshgrid(meshgrid_data *mesh, nPhysD &xx, nPhysD &yy)
{
	if (mesh==NULL)
		return;

	xx.resize(mesh->nx, mesh->ny);
	yy.resize(mesh->nx, mesh->ny);

	double dx = (mesh->x2 - mesh->x1)/(mesh->nx-1);
	double dy = (mesh->y2 - mesh->y1)/(mesh->ny-1);


	for (int i=0; i<mesh->nx; i++) {
		for (int j=0; j<mesh->ny; j++) {
			//xx.Timg_matrix[j][i] = mesh->x1+i*dx;
			//yy.Timg_matrix[j][i] = mesh->y1+j*dy;
			xx.set(i, j, mesh->x1+i*dx);
			yy.set(i, j, mesh->y1+j*dy);
		}
	}

}


typedef struct morlet_data_str morlet_data;
struct morlet_data_str { double lambda, angle, thickness, damp; };

inline void 
phys_generate_morlet(morlet_data *md, nPhysD &xx, nPhysD &yy, nPhysC &zz);

inline void 
phys_generate_Fmorlet(morlet_data *md, nPhysD &xx, nPhysD &yy, nPhysC &zz)
{
	if ((xx.getW() != yy.getW()) || (xx.getH() != yy.getH())) {
		WARNING("size mismatch: op1 is: "<<xx.getW()<<"x"<<xx.getH()<<", op2 is: "<<yy.getW()<<"x"<<yy.getH());
		return;
	}
	
	double cr = cos(md->angle); double sr = sin(md->angle);

	//double damp, sigma, blur;

	//damp = md->damp;	// standard e' 1.
	//sigma = md->lambda;
	//blur = md->thickness;	// tipico l/2
	
	//tom
	double norm_damp=md->damp*M_PI;
	double norm_thickness=md->thickness*M_PI;
	zz.resize(xx.getW(), xx.getH());
	for (size_t i=0;i<zz.getW();i++) {
	    	for (size_t j=0;j<zz.getH();j++) {
			//int xc=(i+zz.getW()/2)%zz.getW()-zz.getW()/2; // swap and center in 0
			//int yc=(j+zz.getH()/2)%zz.getH()-zz.getH()/2;
			
			//double xc = xx.Timg_matrix[j][i];
			//double yc = yy.Timg_matrix[j][i];
			double xc = xx.point(i,j);
			double yc = yy.point(i,j);

			double xr = xc*cr - yc*sr; //rotate
			double yr = xc*sr + yc*cr;

			double e_x = pow(norm_damp*(xr*md->lambda/zz.getW()-1.0), 2.);
			double e_y = pow(yr*norm_thickness/zz.getH(), 2.);

			//double gauss = exp(-0.5*e_x)*exp(-0.5*e_y);
			double gauss = exp(-e_x)*exp(-e_y);

			//zz.Timg_matrix[j][i]=mcomplex(gauss, 0);
			zz.set(i,j,mcomplex(gauss, 0));
	    	}
	}
}


inline void phys_reverse_vector(double *buf, int size);

// some nice filters

void
phys_sin(nPhysD &);

void
phys_cos(nPhysD &);

void
phys_tan(nPhysD &);

void
phys_pow(nPhysD &, double);

void
phys_square(nPhysD &);

void
phys_sqrt(nPhysD &);

void
phys_abs(nPhysD &);

void
phys_log(nPhysD &);

void
phys_log10(nPhysD &);

void
phys_median_filter(nPhysD&, unsigned int);

void
phys_gaussian_blur(nPhysD &, double);

void
phys_fast_gaussian_blur(nPhysD &, double);

void
phys_fast_gaussian_blur(nPhysD &, double, double);

template<> void
nPhysC::TscanBrightness();

// constant operations
void phys_add(nPhysD &, double);
void phys_subtract(nPhysD &, double);
void phys_multiply(nPhysD &, double);
void phys_divide(nPhysD &, double);
void phys_divide(nPhysC &, double);

void phys_point_add(nPhysD &, nPhysD &);
void phys_point_subtract(nPhysD &, nPhysD &);
void phys_point_multiply(nPhysD &, nPhysD &);
void phys_point_divide(nPhysD &, nPhysD &);

void phys_add_noise(nPhysD &, double);

double phys_sum_points(nPhysD &);
double phys_sum_square_points(nPhysD &);
void phys_opposite(nPhysD &);
void phys_inverse(nPhysD &);

void phys_replace(nPhysD &, double, double);
void phys_replace_NaN(nPhysD &, double);
void phys_cutoff(nPhysD &, double, double);

std::pair<double, bidimvec<int> > phys_cross_correlate(nPhysD*, nPhysD*);


// crap functions
void phys_get_vec_brightness(const double *, size_t, double &, double &);


bidimvec<size_t>
phys_max_p(nPhysD &);

// complex functions
std::map<std::string, nPhysD > to_polar(nPhysC &iphys);
std::map<std::string, nPhysD > to_rect(const nPhysC &iphys);
std::map<std::string, nPhysD > to_powersp(nPhysC &iphys, bool);

// shift functions
template <class T>
nPhysImageF<T> ftshift2(nPhysImageF<T> &iimg) {
	nPhysImageF<T> oimg;
	oimg.resize(iimg.getW(), iimg.getH());
	std::vector<T> vec;
	for (register size_t jj=0; jj<iimg.getH(); jj++) {
		iimg.get_Trow(jj, floor(iimg.getW()/2), vec);
		oimg.set_Trow(jj+floor(iimg.getH()/2), 0, vec);
	}

	return oimg;
}

template <class T>
nPhysImageF<T> ftshift1(nPhysImageF<T> &iimg, enum phys_direction ftdir=PHYS_X) {
	nPhysImageF<T> oimg;
	oimg.resize(iimg.getW(), iimg.getH());
	std::vector<T> vec;

	if (ftdir == PHYS_X) {
		for (register size_t jj=0; jj<iimg.getH(); jj++) {
			iimg.get_Trow(jj, floor(iimg.getW()/2), vec);
			oimg.set_Trow(jj, 0, vec);
		}
	} else {
		for (register size_t jj=0; jj<iimg.getH(); jj++) {
			iimg.get_Trow(jj, 0, vec);
			oimg.set_Trow(jj+floor(iimg.getH()/2), 0, vec);
		}
	}

	return oimg;
}

template <class T>
nPhysImageF<T> resample(nPhysImageF<T> &iimg, vec2 new_size)
{
	nPhysImageF<T> oimg;

	oimg.resize(new_size.x(), new_size.y());

	double vscale = ((double)iimg.getH())/oimg.getH();
	double hscale = ((double)iimg.getW())/oimg.getW();

	DEBUG("Hscale: "<<hscale<<", Vscale: "<<vscale);

	for (register size_t ii=0; ii<oimg.getH(); ii++) {
		double frow = vscale*ii; 
		for (register size_t jj=0; jj<oimg.getW(); jj++) {
			double fcol = hscale*jj;
			oimg.set(jj, ii, iimg.getPoint(fcol, frow));
		}
	}

	return oimg;
}

nPhysC from_real_imaginary (nPhysD&, nPhysD&);

nPhysC from_real (nPhysD&, double=0.0);

#endif
