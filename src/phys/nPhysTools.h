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
#include <iostream>
#include <fstream>

#ifdef HAVE_LIBGSL
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_multifit.h>
#endif

#include "bidimvec.h"
#include "nPhysImageF.h"
#include <iterator>

#ifndef __nPhysTools
#define __nPhysTools


/*! \addtogroup nPhysTools
 * @{
 */
	
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
    typename std::vector<T>::iterator ptr  = std::partition(tmp.begin(), tmp.end(), [](T i){return !isnan(i);});

    std::sort(tmp.begin(),ptr);

    int notNaN = std::distance(tmp.begin(), ptr)-1;

    bidimvec<unsigned int> perc(notNaN*(val.first())/100.0,notNaN*(val.second())/100.0);

    bidimvec<T> retvec (tmp[perc.first()],tmp[perc.second()]);

    DEBUG(val << " " << perc << " " << retvec);
    return retvec;
}

template <class T> bidimvec<T> getColorPrecentPixels(nPhysImageF<T>& my_phys, double val) {
    return getColorPrecentPixels(my_phys,vec2f((100.0-val)/2.0, (100.0+val)/2.0));
}


#endif

/*!
 * @}
 */
