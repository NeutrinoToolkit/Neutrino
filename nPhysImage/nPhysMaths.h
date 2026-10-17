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
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#endif

#include "nPhysImageF.h"
#include "mcomplex.h"

#ifndef nPhysMaths_h_
#define nPhysMaths_h_

constexpr double _phys_deg = 0.017453292519943295769;  // pi/180

namespace physMath {

std::array<double,6> getAffine(std::vector<vec2f>, std::vector<vec2f>);

inline void planeFit(physD *pi, vec2f &coeffs);

inline void phys_reverse_vector(double *buf, int size);

// some nice filters

void phys_sin(physD &);

void phys_cos(physD &);

void phys_tan(physD &);

void phys_pow(physD &, double);

void phys_square(physD &);

void phys_sqrt(physD &);

void phys_abs(physD &);

void phys_log(physD &);

void phys_log10(physD &);

void phys_transpose(physD &);

void phys_median_filter(physD&, unsigned int);

void phys_gaussian_blur(physD &, double);

void phys_fast_gaussian_blur(physD &, double);

void phys_fast_gaussian_blur(physD &, double, double);

void phys_hanning_window(physD &);

void phys_gauss_window(physD &, double sigmax=10, double sigmay=10, int power=2);

void phys_integratedNe(physD &, double);

void phys_laplace(physD&);

void phys_gauss_laplace(physD&, double);

void phys_sobel(physD&);

void phys_sobel_dir(physD&);

void phys_scharr(physD&);

void phys_gauss_sobel(physD&, double);

void phys_set_all(physD &, double);

void phys_crop(physD&, int, int, int, int);

// constant operations
void phys_add(physD &, double);
void phys_subtract(physD &, double);
void phys_multiply(physD &, double);
void phys_divide(physD &, double);
void phys_divide(physC &, double);

void phys_remainder(physD &, double);

void phys_point_add(physD &, physD &);
void phys_point_subtract(physD &, physD &);
void phys_point_multiply(physD &, physD &);
void phys_point_divide(physD &, physD &);

void add_noise(physD &, double);

double phys_sum_points(physD &);
double phys_sum_square_points(physD &);
void phys_opposite(physD &);
void phys_inverse(physD &);

void phys_replace(physD &, double, double);
void phys_replace_NaN(physD &, double);

std::pair<double, bidimvec<int> > phys_cross_correlate(physD*, physD*);

// crap functions
void phys_get_vec_brightness(const double *, size_t, double &, double &);


bidimvec<size_t>
phys_max_p(physD &);

// complex functions
std::map<std::string, physD > to_polar(physC &iphys);
std::map<std::string, physD > to_rect(const physC &iphys);
std::map<std::string, physD > to_powersp(physC &iphys, bool);

// shift functions
// template <class T>
// nPhysImageF<T> ftshift2(nPhysImageF<T> &iimg) {
//     nPhysImageF<T> oimg;
//     oimg.resize(iimg.getW(), iimg.getH());
//     std::vector<T> vec;
//     for (size_t jj=0; jj<iimg.getH(); jj++) {
//         iimg.get_Trow(jj, floor(iimg.getW()/2), vec);
//         oimg.set_Trow(jj+floor(iimg.getH()/2), 0, vec);
//     }
//
//     return oimg;
// }
//
// template <class T>
// nPhysImageF<T> ftshift1(nPhysImageF<T> &iimg, enum phys_direction ftdir=PHYS_X) {
//     nPhysImageF<T> oimg;
//     oimg.resize(iimg.getW(), iimg.getH());
//     std::vector<T> vec;
//
//     if (ftdir == PHYS_X) {
//         for (size_t jj=0; jj<iimg.getH(); jj++) {
//             iimg.get_Trow(jj, floor(iimg.getW()/2), vec);
//             oimg.set_Trow(jj, 0, vec);
//         }
//     } else {
//         for (size_t jj=0; jj<iimg.getH(); jj++) {
//             iimg.get_Trow(jj, 0, vec);
//             oimg.set_Trow(jj+floor(iimg.getH()/2), 0, vec);
//         }
//     }
//
//     return oimg;
// }
// (horizontally the row was rotated the other way: for an odd width the zero frequency went to W/2+1 (numpy
// ifftshift instead of fftshift), and the same function was used before the backward transform, which needs
// the inverse shift: forward -> shift -> backward did not give the image back for odd sizes)

//! circular shift: the pixel (x,y) goes to ((x+dx) mod W, (y+dy) mod H)
template <class T>
nPhysImageF<T> circshift(nPhysImageF<T> &iimg, int dx, int dy) {
    nPhysImageF<T> oimg;
    oimg.resize(iimg.getW(), iimg.getH());
    const int W=iimg.getW(), H=iimg.getH();
    if (W==0 || H==0) return oimg;
    dx=((dx%W)+W)%W;
    dy=((dy%H)+H)%H;
#pragma omp parallel for
    for (int y=0; y<H; y++) {
        int yo=(y+dy)%H;
        for (int x=0; x<W; x++) {
            oimg.Timg_matrix[yo][(x+dx)%W]=iimg.Timg_matrix[y][x];
        }
    }
    return oimg;
}

//! zero frequency from (0,0) to the centre (floor(W/2), floor(H/2)), as numpy fftshift
template <class T>
nPhysImageF<T> ftshift2(nPhysImageF<T> &iimg) {
    return circshift(iimg, iimg.getW()/2, iimg.getH()/2);
}

//! inverse of ftshift2 (numpy ifftshift): to be used before a backward transform
template <class T>
nPhysImageF<T> iftshift2(nPhysImageF<T> &iimg) {
    return circshift(iimg, -(int)(iimg.getW()/2), -(int)(iimg.getH()/2));
}

template <class T>
nPhysImageF<T> ftshift1(nPhysImageF<T> &iimg, enum phys_direction ftdir=PHYS_X) {
    return ftdir == PHYS_X ? circshift(iimg, iimg.getW()/2, 0) : circshift(iimg, 0, iimg.getH()/2);
}

template <class T>
nPhysImageF<T> iftshift1(nPhysImageF<T> &iimg, enum phys_direction ftdir=PHYS_X) {
    return ftdir == PHYS_X ? circshift(iimg, -(int)(iimg.getW()/2), 0) : circshift(iimg, 0, -(int)(iimg.getH()/2));
}

template <class T>
nPhysImageF<T> phys_resample(nPhysImageF<T> &iimg, vec2i new_size)
{
    nPhysImageF<T> oimg;

    oimg.resize(new_size.x(), new_size.y());

    double vscale = static_cast<double>(iimg.getH())/oimg.getH();
    double hscale = static_cast<double>(iimg.getW())/oimg.getW();

    DEBUG("Hscale: "<<hscale<<", Vscale: "<<vscale);

#pragma omp parallel for
    for (size_t ii=0; ii<oimg.getH(); ii++) {
        double frow = vscale*ii;
        for (size_t jj=0; jj<oimg.getW(); jj++) {
            double fcol = hscale*jj;
            oimg.set(jj, ii, iimg.getPoint(fcol, frow));
        }
    }
    std::ostringstream ostr;
    ostr << "resize(" << iimg.getName() << "," << new_size.x() << "," << new_size.y() << ")";
    oimg.setName(ostr.str());
    oimg.setShortName("scaled("+iimg.getShortName()+")");

    return oimg;
}

physC from_real_imaginary (physD&, physD&);

physC from_real (physD&, double=0.0);

//! closed iso-line at `level` around the image origin (its blurred maximum if the origin is 0:0).
//! Marching squares with sub-pixel points (pixel (x,y) is at coordinate (x,y)); empty if not found.
void contour_trace(physD &, std::list<vec2f> &, double level, bool blur=false, double blur_radius=10.);
nPhysImageF<char> contour_surface_map(physD &iimage, std::list<vec2i> &contour); // generate in/out/boundary map based on the contour

//! sums over the finite pixels whose centre is inside the closed polygon `contour`
//! (with integrate_boundary also the pixels crossed by the contour)
struct contour_integral {
    double integral=0, integral_sq=0;
    size_t count=0;
};
contour_integral contour_integrate(physD &, const std::list<vec2f> &contour, bool integrate_boundary=false);

//! fill the pixels with the harmonic interpolation (laplace equation) of their surroundings:
//! the pixels around them (finite values) are the boundary conditions. Red-black SOR.
//! iter_ptr (if given) counts the sweeps; set it to -1 to stop. Returns the sweeps done, -1 if nothing to interpolate from
int laplace_fill(physD &, const std::vector<vec2i> &pixels, double tol=1e-6, int max_iter=100000, int *iter_ptr=nullptr);


template <class T>
inline void phys_flip_ud(nPhysImageF<T> &img)
{
    for (size_t i=0; i<img.getW(); i++) {
        for (size_t j=0; j<img.getH()/2; j++) {
            std::swap(img.Timg_matrix[j][i],img.Timg_matrix[img.getH()-j-1][i]);
        }
    }
}

template <class T>
inline void phys_flip_lr(nPhysImageF<T> &img)
{
    for (size_t i=0; i<img.getW()/2; i++) {
        for (size_t j=0; j<img.getH(); j++) {
            std::swap(img.Timg_matrix[j][i],img.Timg_matrix[j][img.getW()-i-1]);
        }
    }
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
}

// methods for matrix operations

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
    if (val==100.) {
        return my_phys.get_min_max();
    }
    return getColorPrecentPixels(my_phys,vec2f((100.0-val)/2.0, (100.0+val)/2.0));
}

template <class T> void cutoff(nPhysImageF<T>& iimage, T minval, T maxval) {
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

template <class T> void cutoff(nPhysImageF<T>& iimage, bidimvec<T> minmax) {
    cutoff(iimage,minmax.x(),minmax.y());
}


template <class T> void padded(nPhysImageF<T>& iimage, vec2u newSize) {
    nPhysImageF<T> oldmatr=iimage;
    iimage.setShortName("Padded");
    iimage.resize(newSize);
    double mean=iimage.sum()/iimage.getSurf();
    phys_set_all(iimage,mean);

    bidimvec<int> offset=(newSize-oldmatr.get_size())/2;
    DEBUG("padding offset : " << offset);
    iimage.set_origin(oldmatr.get_origin()+offset);

#pragma omp parallel for collapse(2)
    for (size_t j=0; j<oldmatr.getH(); j++) {
        for (size_t i=0; i<oldmatr.getW(); i++) {
            iimage.set(i+offset.x(),j+offset.y(),oldmatr.getPoint(i,j));
        }
    }
    std::ostringstream ostr;
    ostr << "padded(" << iimage.getName() << "," << newSize.x() << "," << newSize.y() << ")";
    iimage.setName(ostr.str());
    iimage.TscanBrightness();
}



}



template<> void
physC::TscanBrightness();


#endif
