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
// my very stupid complex<double> class (due to template conficts between
// physImageF and std::complex

#include <cmath>

#include <fftw3.h>

#pragma once

#ifndef __mcomplex_h
#define __mcomplex_h

class mcomplex {

public:
	mcomplex()
		: re(0), im(0)
	{ }

	mcomplex(double v)
		: re(v), im(0)
	{ }

	mcomplex(double v1, double v2)
		: re(v1), im(v2)
	{ }

	mcomplex(const mcomplex& oth)
	{ re = oth.re; im = oth.im; }

	inline double real()
	{ return re; }

	inline double imag()
	{ return im; }

	inline double mcabs()
	{ return re*re+im*im; }

	inline double mod()
	{ return sqrt(re*re+im*im); }

	inline double arg()
	{ return atan2(im, re); }

	void operator= (mcomplex oth)
	{ re = oth.real(); im = oth.imag(); }

	// wallerazz
	mcomplex operator/ (double val)
	{ return mcomplex(re/val, im/val); }

	mcomplex operator+ (mcomplex oth)
	{ return mcomplex(re+oth.real(), im+oth.imag()); }
	mcomplex operator- (mcomplex oth)
	{ return mcomplex(re-oth.real(), im-oth.imag()); }
	mcomplex operator* (mcomplex oth)
	{ return mcomplex(re*oth.real()-im*oth.imag(), im*oth.real()+re*oth.imag()); }
	mcomplex operator/ (mcomplex oth)
	{ double den = oth.re*oth.re - oth.im*oth.im; return mcomplex((re+oth.re)/den, (im+oth.im)/den); }

private:
	double re, im;
};

inline std::ostream &
operator<< (std::ostream &lhs, mcomplex &cc)
{
	lhs<<"("<<cc.real()<<":"<<cc.imag()<<")";
	return lhs;
}

inline double mabs(mcomplex cc)
{ return sqrt(pow(cc.real(),2)+pow(cc.imag(),2)); }

inline double marg(mcomplex cc)
{ return atan2(cc.imag(), cc.real()); }

// ma guarda te se uno deve essere obbligato a fare 'sti giri...
inline void assign_val_to_fftw_complex(double val, fftw_complex cc)
{ cc[0] = std::isnan(val) ? 0 : val; cc[1] = 0; }

inline void assign_val_to_fftw_complex(mcomplex cval, fftw_complex cc)
{ cc[0] = std::isnan(cval.real()) ? 0 : cval.real(); cc[1] = std::isnan(cval.imag()) ? 0 : cval.imag(); }

#endif
