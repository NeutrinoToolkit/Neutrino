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
#include "mcomplex.h"

#define _phys_deg (0.017453f)


#ifndef __physMaths_h
#define __physMaths_h


namespace physMath {

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
phys_transpose(nPhysD &);

void
phys_median_filter(nPhysD&, unsigned int);

void
phys_gaussian_blur(nPhysD &, double);

void
phys_fast_gaussian_blur(nPhysD &, double);

void
phys_fast_gaussian_blur(nPhysD &, double, double);

void
phys_integratedNe(nPhysD &, double);

void
phys_laplace(nPhysD&);

void
phys_gauss_laplace(nPhysD&, double);

void
phys_sobel(nPhysD&);

void
phys_gauss_sobel(nPhysD&, double);

// constant operations
void phys_add(nPhysD &, double);
void phys_subtract(nPhysD &, double);
void phys_multiply(nPhysD &, double);
void phys_divide(nPhysD &, double);
void phys_divide(nPhysC &, double);

void phys_remainder(nPhysD &, double);

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
nPhysImageF<T> phys_resample(nPhysImageF<T> &iimg, vec2 new_size)
{
    nPhysImageF<T> oimg;

    oimg.resize(new_size.x(), new_size.y());

    double vscale = ((double)iimg.getH())/oimg.getH();
    double hscale = ((double)iimg.getW())/oimg.getW();

    DEBUG("Hscale: "<<hscale<<", Vscale: "<<vscale);

#pragma omp parallel for
    for (size_t ii=0; ii<oimg.getH(); ii++) {
        double frow = vscale*ii;
        for (size_t jj=0; jj<oimg.getW(); jj++) {
            double fcol = hscale*jj;
            oimg.set(jj, ii, iimg.getPoint(fcol, frow));
        }
    }

    return oimg;
}

nPhysC from_real_imaginary (nPhysD&, nPhysD&);

nPhysC from_real (nPhysD&, double=0.0);

//! contour trace function
void contour_trace(nPhysD &, std::list<vec2> &, float, bool blur=false, float blur_radius=10.);
nPhysImageF<char> contour_surface_map(nPhysD &iimage, std::list<vec2> &contour); // generate in/out/boundary map based on the contour
std::list<double> contour_integrate(nPhysD &, std::list<vec2> &, bool integrate_boundary=false);


template <class T>
inline void phys_flip_ud(nPhysImageF<T> &img)
{
    for (size_t i=0; i<img.getW(); i++) {
        for (size_t j=0; j<img.getH()/2; j++) {
            std::swap(img.Timg_matrix[j][i],img.Timg_matrix[img.getH()-j-1][i]);
        }
    }
    img.reset_display();
}

template <class T>
inline void phys_flip_lr(nPhysImageF<T> &img)
{
    for (size_t i=0; i<img.getW()/2; i++) {
        for (size_t j=0; j<img.getH(); j++) {
            std::swap(img.Timg_matrix[j][i],img.Timg_matrix[j][img.getW()-i-1]);
        }
    }
    img.reset_display();
}

template <class T>
inline void phys_rotate_left(nPhysImageF<T> &img)
{
    nPhysImageF<T> rotated=img;
    img.resize(img.getH(),img.getW());
    for (size_t j=0; j<img.getH(); j++) {
        for (size_t i=0; i<img.getW(); i++) {
            img.set(i,j,rotated.point(rotated.getW()-1-j,i));
        }
    }
    img.reset_display();
}

template <class T>
inline void phys_rotate_right(nPhysImageF<T> &img)
{
    nPhysImageF<T> rotated=img;
    img.resize(img.getH(),img.getW());
    for (size_t j=0; j<img.getH(); j++) {
        for (size_t i=0; i<img.getW(); i++) {
            img.set(i,j,rotated.point(j,rotated.getH()-1-i));
        }
    }
    img.reset_display();
}



template <class T1, class T2>
inline void phys_convolve(nPhysImageF<T1> &m1, nPhysImageF<T2> &m2, nPhysImageF<mcomplex> &conv_out)
{
    if ((m1.getW() != m2.getW()) || (m1.getH() != m2.getH())) {
        WARNING("size mismatch: op1 is: "<<m1.getW()<<"x"<<m1.getH()<<", op2 is: "<<m2.getW()<<"x"<<m2.getH());
        return;
    }

    int w = m1.getW();
    int h = m1.getH();

    nPhysImageF<mcomplex> *Fm1, *Fm2, *conv_ptr, buffer;

    conv_out.resize(w, h);

    Fm1 = m1.getFFT(1);
    Fm2 = m2.getFFT(1);

    buffer = (*Fm1 * *Fm2);

    conv_ptr = buffer.getFFT(-1);	// da non perdere (altrimenti leaks...)
    conv_out = *conv_ptr;

    delete Fm1;
    delete Fm2;
    delete conv_ptr;
}

// convolution, already start from FTs
template <class T1, class T2>
inline void phys_convolve_m1_Fm2(nPhysImageF<T1> &m1, nPhysImageF<T2> &Fm2, nPhysImageF<mcomplex> &conv_out)
{
    if ((m1.getW() != Fm2.getW()) || (m1.getH() != Fm2.getH())) {
        WARNING("size mismatch: op1 is: "<<m1.getW()<<"x"<<m1.getH()<<", op2 is: "<<Fm2.getW()<<"x"<<Fm2.getH());
        return;
    }

    int w = m1.getW();
    int h = m1.getH();

    nPhysImageF<mcomplex> *Fm1, *conv_ptr, buffer;

    conv_out.resize(w, h);

    Fm1 = m1.getFFT(1);

    buffer = (*Fm1 * Fm2);

    conv_ptr = buffer.getFFT(-1);	// da non perdere (altrimenti leaks...)
    conv_out = *conv_ptr;

    delete Fm1;
    delete conv_ptr;
}

template <class T1, class T2>
inline void phys_convolve_Fm1_Fm2(nPhysImageF<T1> &Fm1, nPhysImageF<T2> &Fm2, nPhysImageF<mcomplex> &conv_out)
{
    if ((Fm1.getW() != Fm2.getW()) || (Fm1.getH() != Fm2.getH())) {
        WARNING("size mismatch: op1 is: "<<Fm1.getW()<<"x"<<Fm1.getH()<<", op2 is: "<<Fm2.getW()<<"x"<<Fm2.getH());
        return;
    }

    int w = Fm1.getW();
    int h = Fm1.getH();

    nPhysImageF<mcomplex> *conv_ptr, buffer;

    conv_out.resize(w, h);

    buffer = (Fm1 * Fm2);

    conv_ptr = buffer.getFFT(-1);	// da non perdere (altrimenti leaks...)
    conv_out = *conv_ptr;

    delete conv_ptr;

}

// methods for matrix operations
template <class T>
void phys_get_bbox(nPhysImageF<T>& img1, nPhysImageF<T>& img2, vec2f& ul_corner, vec2f& lr_corner)
{
    //	using namespace vmath;
    //	max(vec2f(1,2), vec2f(3,4));
    //	vec2f MM(vmath::max(img1.property.origin, img2.property.origin));
}

template <class T> bidimvec<T> getColorPrecentPixels(nPhysImageF<T>& my_phys, vec2f val) {
    std::vector<T> tmp(my_phys.Timg_buffer,my_phys.Timg_buffer+my_phys.getSurf());
    typename std::vector<T>::iterator ptr  = std::partition(tmp.begin(), tmp.end(), [](T i){return !std::isnan(i);});

    std::sort(tmp.begin(),ptr);

    int notNaN = std::distance(tmp.begin(), ptr)-1;

    bidimvec<unsigned int> perc(notNaN*(val.first())/100.0,notNaN*(val.second())/100.0);

    bidimvec<T> retvec (tmp[perc.first()],tmp[perc.second()]);

    DEBUG(val << " " << perc << " " << retvec);
    return retvec;
}

template <class T> bidimvec<T> getColorPrecentPixels(nPhysImageF<T>& my_phys, double val) {
    if (val==100) {
        return my_phys.get_min_max();
    }
    return getColorPrecentPixels(my_phys,vec2f((100.0-val)/2.0, (100.0+val)/2.0));
}

template <class T> void phys_cutoff(nPhysImageF<T>& iimage, T minval, T maxval) {
    iimage.setShortName("IntensityCutoff");
#pragma omp parallel for
    for (size_t ii=0; ii<iimage.getSurf(); ii++) {
        T val=iimage.point(ii);
        if (std::isfinite(val)) iimage.set(ii,std::min(std::max(val,minval),maxval));
    }
    iimage.TscanBrightness();
    std::ostringstream ostr;
    ostr << "min_max(" << iimage.getName() << "," << minval << "," << maxval << ")";
    iimage.setName(ostr.str());
}

}



template<> void
nPhysC::TscanBrightness();


#endif
