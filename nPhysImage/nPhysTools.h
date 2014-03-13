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

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_multifit.h>

#include "bidimvec.h"
#include "nPhysImageF.h"


//using namespace std;

#ifndef __nPhysTools
#define __nPhysTools


template <class T> inline void phys_flip_lr(nPhysImageF<T> &);
template <class T> inline void phys_flip_ud(nPhysImageF<T> &);
template <class T> inline void phys_rotate_left(nPhysImageF<T> &);
template <class T> inline void phys_rotate_right(nPhysImageF<T> &);


/*! \addtogroup nPhysTools
 * @{
 */
	
template <class T>
inline void phys_flip_ud(nPhysImageF<T> &img)
{
	for (size_t i=0; i<img.getW()/2; i++) {
		for (size_t j=0; j<img.getH(); j++) {
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
	for (register size_t j=0; j<img.getH(); j++) {
		for (register size_t i=0; i<img.getW(); i++) {
			img.set(i,j,rotated.point(rotated.getW()-1-j,i));
		}
	}
}

template <class T>
inline void phys_rotate_right(nPhysImageF<T> &img)
{
	nPhysImageF<T> rotated=img;
	img.resize(img.getH(),img.getW());
	for (register size_t j=0; j<img.getH(); j++) {
		for (register size_t i=0; i<img.getW(); i++) {
			img.set(i,j,rotated.point(j,rotated.getH()-1-i));
		}
	}
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


#endif

/*!
 * @}
 */
