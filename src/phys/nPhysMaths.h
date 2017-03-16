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
#include <iterator>
#include <list>

#ifdef HAVE_LIBGSL
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_multifit.h>
#endif

#include "nPhysImageF.h"
#include "nPhysTools.h"
#include "mcomplex.h"

#define _phys_deg (0.017453f)


#ifndef __physMaths_h
#define __physMaths_h



inline void planeFit(physD *pi, double *coeffs);

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
phys_generate_meshgrid(meshgrid_data *mesh, physD &xx, physD &yy)
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
phys_generate_morlet(morlet_data *md, physD &xx, physD &yy, physC &zz);

inline void 
phys_generate_Fmorlet(morlet_data *md, physD &xx, physD &yy, physC &zz)
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
phys_sin(physD &);

void
phys_cos(physD &);

void
phys_tan(physD &);

void
phys_pow(physD &, double);

void
phys_square(physD &);

void
phys_sqrt(physD &);

void
phys_abs(physD &);

void
phys_log(physD &);

void
phys_log10(physD &);

void
phys_median_filter(physD&, unsigned int);

void
phys_gaussian_blur(physD &, double);

void
phys_fast_gaussian_blur(physD &, double);

void
phys_fast_gaussian_blur(physD &, double, double);

void
phys_laplace(physD&);

void
phys_gauss_laplace(physD&, double);

void
phys_sobel(physD&);

void
phys_gauss_sobel(physD&, double);

template<> void
physC::TscanBrightness();

// constant operations
void phys_add(physD &, double);
void phys_subtract(physD &, double);
void phys_multiply(physD &, double);
void phys_divide(physD &, double);
void phys_divide(physC &, double);

void phys_point_add(physD &, physD &);
void phys_point_subtract(physD &, physD &);
void phys_point_multiply(physD &, physD &);
void phys_point_divide(physD &, physD &);

void phys_add_noise(physD &, double);

double phys_sum_points(physD &);
double phys_sum_square_points(physD &);
void phys_opposite(physD &);
void phys_inverse(physD &);

void phys_replace(physD &, double, double);
void phys_replace_NaN(physD &, double);
void phys_cutoff(physD &, double, double);

std::pair<double, bidimvec<int> > phys_cross_correlate(physD*, physD*);


// crap functions
void phys_get_vec_brightness(const double *, size_t, double &, double &);


bidimvec<size_t>
phys_max_p(physD &);

// fourier
physC& ft2(physD&, enum phys_fft);
physC& ft2(physC&, enum phys_fft);

// complex functions
std::map<std::string, physD > to_polar(physC &iphys);
std::map<std::string, physD > to_rect(const physC &iphys);
std::map<std::string, physD > to_powersp(physC &iphys, bool);

// shift functions
template <class T>
nPhysImageF<T> ftshift2(nPhysImageF<T> &iimg) {
	nPhysImageF<T> oimg;
	oimg.resize(iimg.getW(), iimg.getH());
	std::vector<T> vec;
    for (size_t jj=0; jj<iimg.getH(); jj++) {
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
        for (size_t jj=0; jj<iimg.getH(); jj++) {
			iimg.get_Trow(jj, floor(iimg.getW()/2), vec);
			oimg.set_Trow(jj, 0, vec);
		}
	} else {
        for (size_t jj=0; jj<iimg.getH(); jj++) {
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

    for (size_t ii=0; ii<oimg.getH(); ii++) {
		double frow = vscale*ii; 
        for (size_t jj=0; jj<oimg.getW(); jj++) {
			double fcol = hscale*jj;
			oimg.set(jj, ii, iimg.getPoint(fcol, frow));
		}
	}

	return oimg;
}

physC from_real_imaginary (physD&, physD&);

physC from_real (physD&, double=0.0);

//! contour trace function
void contour_trace(physD &, std::list<vec2> &, float, bool blur=false, float blur_radius=10.);
std::list<double> contour_integrate(physD &, std::list<vec2> &, bool integrate_boundary=false);


#endif
