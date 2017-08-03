/*
 *
 *	Copyright (C) 2013 Alessandro Flacco, Tommaso Vinci All Rights Reserved
 * 
 *	This file is part of nPhysImage library.
 *
 *	nPhysImage is free software: you can redistribute it and/or modify
 *	it under the terms of the GNU Lesser General Public License as published by
 *	the Free Software Foundation, either version 3 of the License, or
 *	(at your option) any later version.
 *
 *	nPhysImage is distributed in the hope that it will be useful,
 *	but WITHOUT ANY WARRANTY; without even the implied warranty of
 *	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *	GNU Lesser General Public License for more details.
 *
 *	You should have received a copy of the GNU Lesser General Public License
 *	along with neutrino.  If not, see <http://www.gnu.org/licenses/>.
 *
 *	Contact Information: 
 *	Alessandro Flacco <alessandro.flacco@polytechnique.edu>
 *	Tommaso Vinci <tommaso.vinci@polytechnique.edu>
 *
 */
// nPhysImageF.h -- template class -- former nPhysImageF

/*! \mainpage

  \section base_sec Basic stuff

  \section repr_sec Data representation and access

  \code{.cpp}

	T *Timg_buffer;
	T **Timg_matrix;
  \endcode

  \section templ_sec Template structure

 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include <string>
#include <map>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <map>
#include <typeinfo>
#include <limits>
#include <functional>

#include <assert.h>
#include <fftw3.h>

#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#include <time.h>

// conf variables from autoconf
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "mcomplex.h"
#include "anymap.h"
#include "tools.h"

#include "string.h"

#include "bidimvec.h"


#include <gsl/gsl_math.h>

#define _phys_cspeed (2.99e8)
#define _phys_echarge (1.602e-19)
#define _phys_emass (9.1e-31)
#define _phys_vacuum_eps (8.8e-12)
#define _phys_avogadro (6.022e23)


#pragma once
#ifndef __nPhysImageF_h
#define __nPhysImageF_h

//enum phys_direction { PHYS_HORIZONTAL, PHYS_VERTICAL, PHYS_X, PHYS_Y, PHYS_ROW, PHYS_COLUMN };

// direction defs moved to bidimvec.h
//#if (__GNUC_MINOR__ > 5)
//enum phys_direction : size_t { PHYS_HORIZONTAL = 0, PHYS_X = 0, PHYS_ROW = 0,  PHYS_VERTICAL = 1, PHYS_Y = 1, PHYS_COLUMN = 1 };
//#else
//enum phys_direction { PHYS_HORIZONTAL = 0, PHYS_X = 0, PHYS_ROW = 0,  PHYS_VERTICAL = 1, PHYS_Y = 1, PHYS_COLUMN = 1 };
//#endif

enum phys_way { PHYS_POS, PHYS_NEG };
enum phys_fft { PHYS_FORWARD, PHYS_BACKWARD };


class dException : public std::exception {
public:
	dException()
		: exception()
	{ }
	
	void set_msg(const char *descr)
	{ description = descr; }
    using std::exception::what;
    virtual const char *what()
	{ return description; }

	const char *description;
};

//! exception to be used on deprecated functions still in the sources
class phys_deprecated: public std::exception
{
  virtual const char* what() const throw()
  {
	return "FATAL: function declared UNSAFE!";
  }
};

//! exception to be used on wanna-be deprecated functions
class phys_trashable: public std::exception
{
  virtual const char* what() const throw()
  {
	return "FATAL: function will be REMOVED";
  }
};

//! exception in case of file read problems
class phys_fileerror: public std::exception
{

public:
    phys_fileerror(std::string str = std::string("(undefined file error"))
        : msg(str)
    { }

    ~phys_fileerror() throw()
    { }

    virtual const char* what() const throw()
    { return msg.c_str(); }

private:
    std::string msg;

};

typedef bidimvec<double> vec2f;
typedef bidimvec<int> vec2;

//! image source
//typedef enum phys_type_en {PHYS_FILE, PHYS_RFILE, PHYS_DYN} phys_type;

// now passing to std::string
typedef std::string phys_type;
#define PHYS_FILE "phys_file"
#define PHYS_RFILE "phys_rfile"
#define PHYS_DYN "phys_dyn"

// new version: using anymap instead of properties


//! image properties (transfered upon shallow copy)
/*struct phys_properties_str {

	phys_properties_str()
		: phys_name(std::string("")), phys_short_name(std::string("")), phys_from_name(std::string("")), phys_orig(PHYS_DYN), origin(0.0,0.0), scale(1.0,1.0)
	{ }
	
	//! image name
	std::string phys_name;

	//! image short name
	std::string phys_short_name;

	//! original file name
	std::string phys_from_name;

	//! image source
	phys_type phys_orig;

	//@{
	//! image origin and scale
	vec2f origin, scale;
	//@}
};
typedef struct phys_properties_str phys_properties;*/

// phys_properties should be initialised accordingly
class phys_properties : public anymap {

public:
	phys_properties()
		: anymap()
	{
		(*this)["origin"] = vec2f(0,0);
		(*this)["scale"] = vec2f(1,1);
	}

};

template<class T>
class nPhysImageF {
public:
	//! creates empty nPhys
	nPhysImageF();
	
	//! creates named empty nPhys
	nPhysImageF(std::string, phys_type=PHYS_DYN);
	
	//! creates named nPhys, performing a DEEP copy from the passed reference
	nPhysImageF(const nPhysImageF<T> &, std::string=std::string("copy")); // copy constructor --->> REALLOCATION!

	//! creates flat, named nPhys
	nPhysImageF(size_t, size_t, T, std::string = std::string());
	
	//! named copy from buffer (to be revisited)
	nPhysImageF(T *, size_t, size_t, std::string = std::string());
	
	//! memento mori
	~nPhysImageF();


	//! resize existing object. WARNING: existing data is deleted
	void resize(size_t new_w, size_t new_h)
	{ if ((getW() != new_w) || (getH() != new_h)) {width = new_w; height = new_h; matrix_points_aligned(); } }
	
	//! re-reads buffer for minimum/maximum value
	void TscanBrightness(void);
	
	//! 1D get functions for row/column access
	size_t get_Tvector(enum phys_direction, size_t, size_t, T*, size_t, phys_way orient=PHYS_POS);
	
	//! 1D set functions for row/column access
	void set_Tvector(enum phys_direction, size_t, size_t, T*, size_t, phys_way orient=PHYS_POS);

	//! get row specialized function
	void get_Trow(size_t, size_t, std::vector<T> &);
	
	//! set row specialized function
	void set_Trow(size_t, size_t, std::vector<T> &);

	//! old ft functions (DEPRECATED!)
	nPhysImageF<mcomplex> *getFFT(int);
	//! old ft functions (DEPRECATED!)
	nPhysImageF<mcomplex> *getSingletonFFT(enum phys_direction, enum phys_fft);
	//! old ft functions (DEPRECATED!)
	void fftshift();
	//! old ft functions (DEPRECATED!)
	void ifftshift();
	//! old ft functions (DEPRECATED!)
	T get_shifted(int, int);	

	//! 2D Complex FT
	nPhysImageF<mcomplex> ft2(enum phys_fft ftdir = PHYS_FORWARD);

	//! 1D Complex FT
	nPhysImageF<mcomplex> ft1(enum phys_direction, enum phys_fft ftdir = PHYS_FORWARD);

	//! ASCII output of the buffer
    inline void writeASC(const char *);

	//! RAW dump
	void writeRAW(const char *);

	std::string class_name ()
	{ return std::string(typeid(T).name()); }

	//! min/max point coordinates
    vec2 min_Tv, max_Tv;

	//phys_properties property; now on anymap
	//anymap property; specialized class: phys_properties
	phys_properties property;
    phys_properties display_property;

	// image derivation
	//nPhysImageF<T> *get_Tlast()
	//{ if (derived_pIF != NULL) return derived_pIF->get_Tlast(); else return this; }

	//! Exceptions (should use dException)
	std::exception E_alloc, E_access, E_unsafe;

	// --------------------- image points and statistics ----------------------------
	T sum() 
    { T sumTot=0; for (size_t i=0; i<getSurf(); i++) sumTot+=Timg_buffer[i]; return sumTot; }

		//! min/max values 
	bidimvec<T> get_min_max();
	T get_min();
	T get_max();

	// ------------------------------------------------------------------------------

	// operators, internal versions; still need crossed versions
	nPhysImageF<T> operator+ (const nPhysImageF<T> &) const;
	nPhysImageF<T> operator+ (T &) const;
	nPhysImageF<T> operator- (const nPhysImageF<T> &) const;
	nPhysImageF<T> operator- (T &) const;
	nPhysImageF<T> operator* (const nPhysImageF<T> &) const;
	nPhysImageF<T> operator/ (const nPhysImageF<T> &) const;

	// assignment operator -- SHALLOW COPY
	nPhysImageF<T> & operator= (const nPhysImageF<T> &rhs)
	{
		//std::cerr<<"shallow copy ------------------------------------"<<std::endl;

		// check if other instances are present (prevent leaks)
		if (_canResize())
			resize(0, 0);
		else {
			_init_temp_pointers();
			_trash_delete();
		}

		// copy everything
		property = rhs.property; // probably missing DEEP operator
		
		Tmaximum_value = rhs.Tmaximum_value;
		Tminimum_value = rhs.Tminimum_value;
		Timg_matrix = rhs.Timg_matrix;
		Timg_buffer = rhs.Timg_buffer;
		width = rhs.width;
		height = rhs.height;
		histogram=rhs.histogram;


		_n_inst = rhs._n_inst;
		_trash_new();
//		std::cerr<<"end shallow copy --------------------------------"<<std::endl;

		// not sure about this
		return *this;

	}


	// check for shallowness
	bool operator== (const nPhysImageF<T> &rhs)
	{ return (Timg_buffer == rhs.Timg_buffer); } 

	// .alex. -- I do ignore the sense of this..
	//bool operator== (const nPhysImageF<T> &rhs)
	//{ return (_n_inst == rhs._n_inst); } 

	template <class U> operator nPhysImageF<U>  () const
	{
		DEBUG(5,"cast constructor ------------------------------------");
		nPhysImageF<U> lhs;
//		lhs = new nPhysImageF<U>;
		lhs.resize(width, height);
#pragma omp parallel for
        for (size_t i=0; i<getSurf(); i++)
			lhs.Timg_buffer[i] = U(Timg_buffer[i]);	

        lhs.TscanBrightness();
		
		//lhs->object_name = object_name;
		//lhs->filename=filename;
		lhs.property = property;
		return lhs;
	}

	//! Returns a DEEP copy of the object 
	nPhysImageF<T> &copy()
	{
		nPhysImageF<T> *new_img = new nPhysImageF<T>(*this);
		new_img->_trash_return();
		return *new_img;
	}

	
	nPhysImageF<T> sub(size_t, size_t, size_t, size_t);

	// ! get interpolated stretched image
	nPhysImageF<T> * stretch(bidimvec<size_t> newSize) {
		nPhysImageF<T> *stretched = new nPhysImageF<T> (newSize.x(), newSize.y(), 0.);
		bidimvec<double> ratio=div_P<double>(newSize,get_size());
		DEBUG(5,"ratio" << newSize << " " << ratio);

 		stretched->set_origin(mul_P(get_origin(),ratio));
 		stretched->set_origin(mul_P(get_scale(),ratio));
		stretched->setType(PHYS_DYN);

        for (size_t j=0; j<stretched->getH(); j++) {
            for (size_t i=0; i<stretched->getW(); i++) {
				bidimvec<double> p=div_P<double>(bidimvec<double>(i,j),ratio);
				stretched->set(i,j,getPoint(p.x(),p.y()));
			}
		}
		return stretched;
	}
	nPhysImageF<T> * stretch(size_t newW, size_t newH) { return stretch(bidimvec<size_t>(newW,newH));}

	// get rotated matrix
	nPhysImageF<T> * rotated(double alphaDeg, T def_value=std::numeric_limits<T>::quiet_NaN()) {		
		double alpha=fmod(alphaDeg+360.0,360.0)/180.0* M_PI;
		nPhysImageF<T> *rotated=NULL;

		if (alphaDeg==0.0) {
			rotated = new nPhysImageF<T> (getW(), getH(), 0.0);
            for (size_t j=0; j<rotated->getH(); j++) {
                for (size_t i=0; i<rotated->getW(); i++) {
					rotated->set(i,j,point(i,j));
				}
			}
			rotated->set_origin(get_origin());
		} else if (alphaDeg==90.0) {
			rotated = new nPhysImageF<T> (getH(), getW(), 0.0);
            for (size_t j=0; j<rotated->getH(); j++) {
                for (size_t i=0; i<rotated->getW(); i++) {
					rotated->set(i,j,point(getW()-1-j,i));
				}
			}
			rotated->set_origin(get_origin().y(),getW()-1-get_origin().x());
		} else if (alphaDeg==180.0) {
			rotated = new nPhysImageF<T> (getW(), getH(), 0.0);
            for (size_t j=0; j<rotated->getH(); j++) {
                for (size_t i=0; i<rotated->getW(); i++) {
					rotated->set(i,j,point(getW()-1-i,getH()-1-j));
				}
			}
			rotated->set_origin(getW()-1-get_origin().x(),getH()-1-get_origin().y());
		} else if (alphaDeg==270.0) {
			rotated = new nPhysImageF<T> (getH(), getW(), 0.0);
            for (size_t j=0; j<rotated->getH(); j++) {
                for (size_t i=0; i<rotated->getW(); i++) {
					rotated->set(i,j,point(j,getH()-1-i));
				}
			}
			rotated->set_origin(get_origin().y(),getH()-1-get_origin().x());
		} else {		
			double sina=sin(alpha);
			double cosa=cos(alpha);
			double dx1=((double)(getH()-1))*sina;
			double dx2=((double)(getW()-1))*cosa;
			double dy1=-((double)(getW()-1))*sina;
			double dy2=((double)(getH()-1))*cosa;

            rotated = new nPhysImageF<T> (fabs(dx1)+fabs(dx2)+1,fabs(dy1)+fabs(dy2)+1, def_value);
			double shiftx=std::min(dx1,0.0)+std::min(dx2,0.0);
			double shifty=std::min(dy1,0.0)+std::min(dy2,0.0);
			size_t i,j;
#pragma omp parallel for collapse(2)
			for (j=0; j<rotated->getH(); j++) {
				for (i=0; i<rotated->getW(); i++) {
					double ir=(i+shiftx)*cos(alpha)-(j+shifty)*sin(alpha);
					double jr=(i+shiftx)*sin(alpha)+(j+shifty)*cos(alpha);
					rotated->set(i,j,getPoint(ir,jr,def_value));
				}
			}
			vec2f orig=get_origin();
			double ir=(orig.x())*cos(-alpha)-(orig.y())*sin(-alpha);
			double jr=(orig.x())*sin(-alpha)+(orig.y())*cos(-alpha);
			rotated->set_origin(ir-shiftx,jr-shifty);
		}
		//FIXME: this must be roto-translated
//		rotated->set_origin(get_origin());
		rotated->set_scale(get_scale());

		rotated->setType(PHYS_DYN);
		std::ostringstream my_name;
		my_name << getName() << ".rotate(" << alphaDeg << ")";
		rotated->setName(my_name.str());
		rotated->setShortName("rotated");
		rotated->setFromName(getFromName());

		rotated->TscanBrightness();
		return rotated;
	}
	
	nPhysImageF<T> * fast_rotated(double alphaDeg, T def_value=std::numeric_limits<T>::quiet_NaN()) {		
		double alpha=fmod(alphaDeg+360.0,360.0)/180.0* M_PI;
		nPhysImageF<T> *rotated= new nPhysImageF<T> (getW(),getH(), def_value);
		double dx_2=0.5*((double) getW());
		double dy_2=0.5*((double) getH());
		double cosa=cos(alpha);
		double sina=sin(alpha);
		for (size_t j=0; j<getH(); j++) {
			for (size_t i=0; i<getW(); i++) {
				double ir=dx_2+(i-dx_2)*cosa-(j-dy_2)*sina;
				double jr=dy_2+(i-dx_2)*sina+(j-dy_2)*cosa;
				rotated->set(i,j,getPoint(ir,jr,def_value));
			}
		}
		rotated->set_origin(get_origin());
		rotated->set_scale(get_scale());
		rotated->setType(PHYS_DYN);
		std::ostringstream my_name;
		my_name << getName() << ".fast_rotated(" << alphaDeg << ")";
		rotated->setName(my_name.str());
		rotated->setShortName("rotated");
		rotated->setFromName(getFromName());

		rotated->TscanBrightness();
		return rotated;	
	}

    // for simplicity we store it in an int which represents the power.
    // correspondence between int/power:
    // {neg,-1/(neg-2)} {-3,1/5} {-2,1/4} {-1,1/3} {0,1/2} {1,1} {2,2} {3,3} ...
    double gamma() {
        if (!property.have("gamma")) {
            property["gamma"]=(int)1;
        }
        int gamma_int= property["gamma"].get_i();
        return gamma_int < 1 ? -1.0/(gamma_int-2) : gamma_int;
    }

    inline void reset_display() {
        display_property.erase("display_range");
    }

    const unsigned char *to_uchar_palette(std::vector<unsigned char>  &palette, std::string palette_name) {
        if (getSurf()>0 && palette.size()==768) {

            if (uchar_buf.size() == getSurf()*3 &&
                    display_property.have("display_range") &&
                    display_property.have("palette_name") &&
                    display_property["palette_name"].get_str()==palette_name &&
                    display_property.have("gamma") &&
                    display_property["gamma"].get_i()==property["gamma"].get_i()) {

                vec2f old_display_range=display_property["display_range"];
                vec2f new_display_range=property["display_range"];

                if (old_display_range==new_display_range) {
                    DEBUG("reusing old uchar_buf");
                    return &uchar_buf[0];
                }
            }

            bidimvec<T> minmax=property.have("display_range") ? property["display_range"] : get_min_max();
            double mini=minmax.first();
            double maxi=minmax.second();
            double my_gamma=gamma();

            DEBUG(6,"8bit ["<<Tminimum_value<<":"<<Tmaximum_value << "] from [" << mini << ":" << maxi<<" , " << my_gamma << "]");
			uchar_buf.assign(getSurf()*3,255);
#pragma omp parallel for
            for (size_t i=0; i<getSurf(); i++) {
				//int val = mult*(Timg_buffer[i]-lower_cut);
                if (std::isfinite(Timg_buffer[i])) {
                    unsigned char val = std::max(0,std::min(255,(int) (255.0*pow((Timg_buffer[i]-mini)/(maxi-mini),my_gamma))));
					std::copy ( palette.begin()+val*3, palette.begin()+val*3+3, uchar_buf.begin()+3*i);
				}
			}
            display_property["palette_name"]=palette_name;
            display_property["gamma"]=property["gamma"].get_i();
            display_property["display_range"]=property["display_range"]=vec2f(mini,maxi);

            return &uchar_buf[0];
		}
		WARNING("asking for uchar buffer of empty image");

		return NULL;
	}

	const double *to_dvector(enum phys_direction direction, size_t index) {
        throw phys_deprecated();

//		std::cerr<<"[to_dvector] dir: "<<direction<<", index: "<<index<<std::endl;
		size_t size[2]={width,height};
			
		if (vector_buf[direction] == NULL)
			vector_buf[direction] = new double [size[direction]];

		size_t ndir = (direction+1)%2;	// normal direction
		index = std::min(index,size[ndir]-1);

		if (direction == PHYS_X) {
            for (size_t i=0; i<size[direction]; i++)
				vector_buf[direction][i] = Timg_matrix[index][i];
				//vector_buf[direction][i] = point(index, i);
		} else {
            for (size_t i=0; i<size[direction]; i++)
				vector_buf[direction][i] = Timg_matrix[i][index];
				//vector_buf[direction][i] = point(i, index);
		}

		return vector_buf[direction];

	}

	const double *to_axis(enum phys_direction direction) {
        throw phys_deprecated();
        size_t size= direction==PHYS_X ? width : height;
		
		if (axis_buf[direction]==NULL) axis_buf[direction] = new double [size];
		
        for (size_t i=0; i<size; i++)
			axis_buf[direction][i] = (i-get_origin(direction))*get_scale(direction);

		return axis_buf[direction];

	}

//	const double *to_evector(int x1, int y1, int x2, int y2) {
//		// here allocation is dynamic. calling object MUST cleanup

//		double *ovec;

//		if ((x1==x2) && (y1==y2)) {

//		}
//		return NULL;
//	}

	// interfacing methods

	// get point (to be used for accessing data - no overload)
	inline T getPoint(double x, double y, T nan_value=std::numeric_limits<T>::quiet_NaN()) {
		if (Timg_matrix != NULL) {
			if (x>=0 && y>=0) {
				size_t x1=(size_t)x;
				size_t y1=(size_t)y;
				if (x==x1 && y==y1) return point((size_t)x1,(size_t)y1);
				size_t x2=x1+1;
				size_t y2=y1+1;
				if (x2<getW() && y2<getH()) {
					T data11=Timg_matrix[y1][x1];
					T data12=Timg_matrix[y1][x2];
					T data21=Timg_matrix[y2][x1];
					T data22=Timg_matrix[y2][x2];			
					return (y2-y)*((x2-x)*data11+(x-x1)*data12)+(y-y1)*((x2-x)*data21+(x-x1)*data22);
				}
			}
		}
		return nan_value;
	}
	inline T getPoint(bidimvec<double> p, T nan_value=std::numeric_limits<T>::quiet_NaN()) {
		return getPoint(p.x(),p.y(),nan_value);
	}
	
	// get point (to be used for accessing data - no overload)
	inline T point(size_t x, size_t y, T nan_value=std::numeric_limits<T>::quiet_NaN()) const {
		if ((Timg_matrix != NULL) && (x<getW()) && (y<getH()))
			return Timg_matrix[y][x];
		else
			return nan_value;
	}

	inline T point(bidimvec<int> p, T nan_value=std::numeric_limits<T>::quiet_NaN()) const {
		if ((Timg_matrix != NULL) && (p.x()<(int)getW()) && (p.y()<(int)getH()) && (p.x()>=0) && (p.y()>=0))
			return Timg_matrix[p.y()][p.x()];
		else
			return nan_value;
	}


	// must check speed
	inline T clean_point(size_t x, size_t y, T nan_value=std::numeric_limits<T>::quiet_NaN()) {
		if ((Timg_matrix != NULL) && (x<getW()) && (y<getH())) {
			if (std::isfinite(Timg_matrix[y][x]))
				return Timg_matrix[y][x];
			return nan_value;
		} else
			return nan_value;
	}

	inline T point(size_t xy) const {
		if ((Timg_matrix != NULL) && (xy<getSurf()) )
			return Timg_buffer[xy];
		else
			return 0;
	}

	inline void set(size_t x, size_t y, T val) {
		if (Timg_matrix && (x<getW()) && (y<getH()))
			Timg_matrix[y][x] = val;
	}

	inline void set(size_t xy, T val) {
		if (Timg_matrix && (xy<getSurf()) )
			Timg_buffer[xy] = val;
	}

	inline void set(bidimvec<size_t> p, T val) {
		if (Timg_matrix && (p.x()<getW()) && (p.y()<getH()))
			Timg_matrix[p.y()][p.x()] = val;
	}

	inline void set(T val) { //! set a value allover the matrix
		DEBUG(PRINTVAR(val));
        for (size_t i=0; i<getSurf(); i++) {
			Timg_buffer[i]=val;
		}
		TscanBrightness();
	}

	const std::vector<double> &get_histogram()
	{
        if (getSurf() == 0) {
			histogram.resize(0);
			return histogram;
		}

		if (histogram.size() > 0)
			return histogram;


		if (Tmaximum_value == Tminimum_value)
			TscanBrightness();

		size_t nbins = std::max<int>(getW()*getH()/10000, 100lu);
		double binw = (double)(Tmaximum_value-Tminimum_value)/(nbins-1.);
		histogram.resize(nbins);

		DEBUG(5,"histogram has "<<nbins<<" bins, bin width: "<<binw);

        for (size_t i=0; i<getSurf(); i++) {
			size_t bin_n = (size_t)floor((Timg_buffer[i]-Tminimum_value)/(binw));
			histogram[bin_n]++;
		}

//        histogram.erase(std::remove(histogram.begin(), histogram.end(), 0.0), histogram.end());
		
		return histogram;
	}

	double count_colors() {
		std::map<T, int> img_colors;
        for (size_t i=0; i<getSurf(); i++)
			img_colors[Timg_buffer[i]]++;

		return img_colors.size();
	}

	inline bidimvec<size_t> get_size()
	{ return bidimvec<size_t>(width,height); }

	inline size_t getW() const
	{ return width; }

	inline size_t getH() const
	{ return height; }

	inline size_t getSurf() const
	{ return width*height; }

	inline vec2 getSize() {return vec2(width,height);}

	inline size_t getSizeByIndex(enum phys_direction dir)
	{ if (dir==PHYS_X) return getW(); if (dir == PHYS_Y) return getH(); return 0; }

	inline bool isInside(size_t x, size_t y) {
		if ((x < getW()) && (y < getH()))
			return true;
		return false;
	}

	// getting and setting properties
	inline std::string getName()
    { return property["phys_name"]; }
	void setName(std::string name)
    { property["phys_name"] = name; }

//tom
	inline std::string getShortName()
    { return property["phys_short_name"]; }
	
	inline void setShortName(std::string name)
    { property["phys_short_name"] = name; }

	inline std::string getFromName()
    { return property["phys_from_name"]; }
	
	inline void setFromName(std::string name)
    { property["phys_from_name"] = name; }

	inline vec2f get_origin()
    { return property["origin"]; }

	inline double get_origin(enum phys_direction direction)
    { return (direction==PHYS_X ? vec2f(property["origin"].get_str()).x() : vec2f(property["origin"].get_str()).y()); }

	inline void set_origin(T val_x, T val_y)
    { property["origin"] = vec2f(val_x,val_y); }
	
	inline void set_origin(vec2f val) 
    { property["origin"] = val; }

    inline vec2f get_scale()
    { return property["scale"]; }

	inline double get_scale(enum phys_direction direction)
    { return (direction==PHYS_X ? vec2f(property["scale"].get_str()).x() : vec2f(property["scale"].get_str()).y()); }

	inline void set_scale(T val_x, T val_y)
    { property["scale"] = vec2f(val_x,val_y); }
	
	inline void set_scale(vec2f val)
    { property["scale"] = val; }
	
	inline vec2f to_real(vec2f val) { 
		vec2f oo, ss; 
        oo = property["origin"]; ss = property["scale"];
		return vec2f((val.x()-oo.x())*ss.x(),(val.y()-oo.y())*ss.y()); 
	}

	inline vec2f to_pixel(vec2f val) { 
		vec2f oo, ss; 
        oo = property["origin"]; ss = property["scale"];
		return vec2f(val.x()/ss.x()+oo.x(),val.y()/ss.y()+oo.y()); 
	}

//end

	inline phys_type getType()
    { return property["phys_orig"]; }
	void setType(std::string orig)
	{ property["phys_orig"] = orig; }


	// (dovra' passare protected, prima o poi...)
	T *Timg_buffer;
	T **Timg_matrix;

protected:
    std::vector<unsigned char> uchar_buf;
	double **vector_buf;
	double **axis_buf;
	std::vector<double> histogram;

private:
	void init_Tvariables();
	void matrix_points_aligned();
	size_t width;
	size_t height;

	//! TODO: pass to bicimvec<T>
	T Tmaximum_value;
	T Tminimum_value;


	// gestione rifiuti
	void _trash_init()		// for constructors
	{ _n_inst = new int; *_n_inst = 1; }
	void _init_temp_pointers();	// for pointers needing refresh in shallow copy
	void _trash_new()		// for assignment operators
	{ (*_n_inst)++; }
	int _trash_delete()		// for destructors
	{ (*_n_inst)--; return *_n_inst; }
	void _trash_return()		// to be called before returning
	{ (*_n_inst)--; }

	bool _canResize()		// check for parallel instances
	{ return (*_n_inst == 1) ? true : false; }

    int *_n_inst;

};

typedef nPhysImageF<double> nPhysD;
typedef nPhysImageF<mcomplex> nPhysC;

typedef std::map<std::string, nPhysImageF<double> > nMapD;



// --------------------------------------------------------------------------------------------

template<class T>
nPhysImageF<T>::nPhysImageF()
{ init_Tvariables(); }

template<class T>
nPhysImageF<T>::nPhysImageF(std::string obj_name, phys_type pp)
{
	init_Tvariables();

	if (pp == PHYS_FILE) {
		// real file
		//setName(obj_name);
		setFromName(obj_name);
	} else if (pp == PHYS_DYN){
		//setName(obj_name);
		setFromName(std::string("(undefined)"));
	}
	setName(obj_name);
	setType(pp);
	std::string shortname=obj_name;
	if (pp==PHYS_FILE) {
		size_t last_idx = obj_name.find_last_of("\\/");
		if (std::string::npos != last_idx) {
			shortname.erase(0,last_idx + 1);
		}
	}
	DEBUG("shortname: "<<shortname);
	setShortName(shortname);	
}

// copy constructor
template<class T>
nPhysImageF<T>::nPhysImageF(const nPhysImageF<T> &oth, std::string sName)
{
//	std::cerr<<"copy constructor ------------------------------------"<<std::endl;
	init_Tvariables();
	resize(oth.width, oth.height);
	
//	memcpy(Timg_buffer, oth.Timg_buffer, getSurf()*sizeof(T));
    std::copy(oth.Timg_buffer, oth.Timg_buffer+getSurf(), Timg_buffer);
	property = oth.property;
	
	setShortName(sName);
	TscanBrightness();
//	std::cerr<<"end copy constructor ------------------------------------"<<std::endl;
}


template<class T>
nPhysImageF<T>::nPhysImageF(size_t w, size_t h, T val, std::string obj_name)
{
	init_Tvariables();
	setName(obj_name);

	resize(w, h);
    for (size_t i=0; i<getW()*getH(); i++)
		Timg_buffer[i] = val;
}



template<class T>
nPhysImageF<T>::nPhysImageF(T *o_buffer, size_t w, size_t h, std::string obj_name) {

	throw phys_trashable();
	init_Tvariables();

	setName(obj_name);
	resize(w, h);
	
	// not calling correspondent physImage constructor for matrix_points_aligned
	//
	// TODO: pass to std::copy
    memcpy(Timg_buffer, o_buffer, getSurf()*sizeof(T));
	
}

/*template<class T>
nPhysImageF<T>::nPhysImageF(T *o_buffer, size_t o_width, size_t o_height) {
	init_Tvariables();
	width = o_width;
	height = o_height;
	matrix_points_aligned();

	// not calling correspondent physImage constructor for matrix_points_aligned
    memcpy(Timg_buffer, o_buffer, getSurf()*sizeof(T));
	
	// rounding
}*/

template<class T>
nPhysImageF<T>::~nPhysImageF()
{
//	std::cerr<<"Destructor for "<<object_name<<std::endl;
	// check for copied instances

	int trashDelete=_trash_delete();
	if ( trashDelete == 0 ) {
        DEBUG(1,"["<<(void *)this<<"] "<<  getShortName() << " : " << getName() << " ALLOWING DELETE! " );
		if (Timg_buffer != NULL)
			delete Timg_buffer;
		
		//tom
		if (Timg_matrix != NULL) delete Timg_matrix;
	
		if (vector_buf != NULL) {
			if (vector_buf[0] != NULL)
				delete vector_buf[0];
			if (vector_buf[1] != NULL)
				delete vector_buf[1];
			delete vector_buf;
		}
	
		if (axis_buf != NULL) {
			if (axis_buf[0] != NULL)
				delete axis_buf[0];
			if (axis_buf[1] != NULL)
				delete axis_buf[1];
			delete axis_buf;
		}
		delete _n_inst; // FIXME alex: check this
		// end tom
	} else {
        DEBUG(1,"["<<(void *)this<<"] "<<  getShortName() << " : " << getName() << " NOT ALLOWING DELETE! " << trashDelete );
	}

}

// -----------------------------------------------------------------------

template<class T> void
nPhysImageF<T>::init_Tvariables()
{
	width = 0;
	height = 0;

	// donne! e' arrivato il monnezzaro
	vector_buf = axis_buf = NULL;
	_trash_init();
	_init_temp_pointers();

	//pIF_size.set_msg("size error");

	min_Tv=vec2(-1,-1);
	max_Tv=vec2(-1,-1);

	Tmaximum_value = 0;
	Tminimum_value = 0;


}

template<class T> void
nPhysImageF<T>::_init_temp_pointers()
{
	Timg_buffer = NULL;
	Timg_matrix = NULL;

	if (vector_buf)
		delete vector_buf;
	if (axis_buf)
		delete axis_buf;

	vector_buf = new double *[2];
	memset(vector_buf, 0, 2*sizeof(double *));
	axis_buf = new double *[2];
	memset(axis_buf, 0, 2*sizeof(double *));

	histogram.resize(0);
}


template<class T> void
nPhysImageF<T>::matrix_points_aligned()
{
    DEBUG(1,(void*) this << " allocate "<<width<<"x"<<height << " name: "<<getName() << " short: " << getShortName() << " from: " << getFromName());

	// check for other sessions
	if (! _canResize())
		throw; // (up)

	// clean up before new allocation	
	if (Timg_buffer != NULL) {
		delete Timg_buffer;
		Timg_buffer = NULL;
	}

	if (Timg_matrix != NULL) {
		delete Timg_matrix;
		Timg_matrix = NULL;
	}

    uchar_buf.clear();

	if (vector_buf != NULL) {
		if (vector_buf[0] != NULL)
			delete vector_buf[0];
		if (vector_buf[1] != NULL)
			delete vector_buf[1];
		vector_buf[0] = NULL;
		vector_buf[1] = NULL;
	}

	if (axis_buf != NULL) {
		if (axis_buf[0] != NULL)
			delete axis_buf[0];
		if (axis_buf[1] != NULL)
			delete axis_buf[1];
		axis_buf[0] = NULL;
		axis_buf[1] = NULL;
	}


    if (Timg_buffer == NULL && getSurf()>0) {
        assert( Timg_buffer = new T[getSurf()] );
		DEBUG(11,"[\t\t|--> ] template 32bit contiguous allocated");

		assert( Timg_matrix = new T* [height] );
        for (size_t i=0; i<height; i++)
			Timg_matrix[i] = Timg_buffer + i*width;
		DEBUG(11,"[\t\t|--> ] template matrix translation allocated");

	}


}

// previous implementation
//template<class T> void
//nPhysImageF<T>::get_Tvector(int direction, int index, int offset, T *ptr, int size)
//{
//	// copies a vector to an external buffer (useful for Abel inversion)
//	// vector is taken on direction (0=x, 1=y), starting from offset and for size points
//
//	int copy_len = size;
//	if (direction == 0) {
//		if ((size+offset > width)) 	// spem longam spatio brevi reseces ;-)
//			copy_len = width-offset;
//		for (size_t i=0; i<copy_len; i++)
//			ptr[i] = Timg_buffer[index*width+offset+i];
//	} else if (direction == 1) {
//		if ((size+offset > height))
//			copy_len = height-offset;
//		for (size_t i=0; i<copy_len; i++)
//			ptr[i] = Timg_buffer[(offset+i)*width+index];
//	}
//}
//
//template<class T> void
//nPhysImageF<T>::set_Tvector(int direction, int index, int offset, T *ptr, int size)
//{
//	int copy_len = size;
//	if (direction == 0) {
//		if ((size+offset > width)) 	// spem longam...
//			copy_len = width-offset;
//		for (size_t i=0; i<copy_len; i++)
//			Timg_buffer[index*width+offset+i] = ptr[i];
//	} else if (direction == 1) {
//		if ((size+offset > height)) 	
//			copy_len = height-offset;
//		for (size_t i=0; i<copy_len; i++)
//			Timg_buffer[(offset+i)*width+index] = ptr[i];
//	}
//
//}
//
//


// ----------------------- DATA ACCESS ----------------------------	

template<class T> size_t
nPhysImageF<T>::get_Tvector(enum phys_direction direction, size_t index, size_t offset, T *ptr, size_t size, phys_way orient)
{
	// copies a vector to an external buffer (useful for Abel inversion)
	// vector is taken on direction (0=x, 1=y), starting from offset and for size points

	size_t copy_len = size;
	if (direction == PHYS_X) {
		if (orient == PHYS_POS) {
			if ((size+offset > width)) 	// spem longam spatio brevi reseces ;-)
				copy_len = width-offset;
			for (size_t i=0; i<copy_len; i++)
				ptr[i] = clean_point(offset+i, index, 0.);
				//ptr[i] = Timg_matrix[index][offset+i];
				//ptr[i] = Timg_buffer[index*width+offset+i];
		} else {
			//if (((offset+1)-size < 0)) 	
			if (((offset+1) < size)) 	
				copy_len = offset+1;
			for (size_t i=0; i<copy_len; i++)
				ptr[i] = clean_point(offset-i, index, 0.);
				//ptr[i] = Timg_matrix[index][offset-i];
		}

	} else if (direction == PHYS_Y) {
		if (orient == PHYS_POS) {
			if ((size+offset > height))
				copy_len = height-offset;
			for (size_t i=0; i<copy_len; i++)
				ptr[i] = clean_point(index, offset+i, 0.);
				//ptr[i] = Timg_matrix[offset+i][index];
				//ptr[i] = Timg_buffer[(offset+i)*width+index];
		} else {
			//if (((offset+1)-size < 0)) 	
			if (((offset+1) < size)) 	
				copy_len = offset+1;
			for (size_t i=0; i<copy_len; i++)
				ptr[i] = clean_point(index, offset-i, 0.);
				//ptr[i] = Timg_matrix[offset-i][index];
		}
	}
	return copy_len;
}

template<class T> void
nPhysImageF<T>::set_Tvector(enum phys_direction direction, size_t index, size_t offset, T *ptr, size_t size, phys_way orient)
{
	// copies a vector to an external buffer (useful for Abel inversion)
	// vector is taken on direction (0=x, 1=y), starting from offset and for size points

	size_t copy_len = size;
	if (direction == PHYS_X) {
		if (orient == PHYS_POS) {
			if ((size+offset > width)) 	// spem longam spatio brevi reseces ;-)
				copy_len = width-offset;
			for (size_t i=0; i<copy_len; i++)
				Timg_matrix[index][offset+i] = ptr[i];
		} else {
			//if (((offset+1) -size < 0)) 	
			if (((offset+1) < size)) 	
				copy_len = offset+1;
			for (size_t i=0; i<copy_len; i++)
				Timg_matrix[index][offset-i] = ptr[i];
		}

	} else if (direction == PHYS_Y) {
		if (orient == PHYS_POS) {
			if ((size+offset > height))
				copy_len = height-offset;
			for (size_t i=0; i<copy_len; i++)
				Timg_matrix[offset+i][index] = ptr[i];
		} else {
			//if (((offset+1)-size < 0)) 	
			if (((offset+1) < size)) 	
				copy_len = offset+1;
			for (size_t i=0; i<copy_len; i++) {
				Timg_matrix[offset-i][index] = ptr[i];
			}
		}
	}
}

//! get_Trow specialization. Row copy/move can be faster for the use of bulk copy methods
//! WARNING: uses % on index and offset (for more interesting solutions)
template<class T> void
nPhysImageF<T>::get_Trow(size_t index, size_t offset, std::vector<T> &vec) {
	
	vec.resize(getW());
	
	typename std::vector<T>::iterator vitr;

	offset = offset%getW();

	vitr = std::copy(Timg_matrix[index%getH()] + offset, Timg_matrix[index%getH()] + getW(), vec.begin());
	if (offset > 0) 
		std::copy(Timg_matrix[index%getH()], Timg_matrix[index%getH()] + offset, vitr);

}

template<class T> void
nPhysImageF<T>::set_Trow(size_t index, size_t offset, std::vector<T> &vec) {

	T* optr;
	offset = offset%getW();
	
	optr = std::copy(vec.end()-offset, vec.end(), Timg_matrix[index%getH()]);
	optr = std::copy(vec.begin(), vec.end()-offset, optr);

}


//! get submatrix
template <class T> nPhysImageF<T> 
nPhysImageF<T>::sub(size_t x, size_t y, size_t Dx, size_t Dy) {

	nPhysImageF<T> subphys(Dx, Dy, 0.);
	subphys.set_origin(get_origin()-vec2f(x,y));
	subphys.set_scale(get_scale());

	subphys.setType(PHYS_DYN);

	if (isInside(x, y)) {
		size_t copy_w = std::min(x+Dx, (size_t)getW()); // FIXME
		size_t copy_h = std::min(y+Dy, (size_t)getH());
        for (size_t i=y; i<copy_h; i++) {
			std::copy(Timg_matrix[i]+x, Timg_matrix[i]+copy_w, subphys.Timg_matrix[i-y]);
		}
	}
	std::ostringstream my_name;
	
	my_name << "submatrix(" << getName() << "," << x << "," << y << "," << Dx << "," << Dy << ")";	
	subphys.setName(my_name.str());
	subphys.setShortName("submatrix("+getShortName()+")");
	subphys.setFromName(getFromName());

	subphys.TscanBrightness();
	return subphys;
}



template<class T> void
nPhysImageF<T>::TscanBrightness() {
    if (getSurf()>0) {
        bool found=false;

#pragma omp parallel for
        for (size_t i=0; i<getSurf(); i++) {
			if (std::isfinite(Timg_buffer[i])) {	
				if (!found) {
					Tminimum_value = Timg_buffer[i];
					Tmaximum_value = Tminimum_value;
					found=true;
				} else {
					 if ((Timg_buffer[i]) > Tmaximum_value) {
                         max_Tv=vec2(i%width, i/width);
                         Tmaximum_value = (Timg_buffer[i]);
					 } else if ((Timg_buffer[i]) < Tminimum_value) {
                         min_Tv=vec2(i%width, i/width);
						 Tminimum_value = (Timg_buffer[i]);
					 }
				}
			}
		}
		DEBUG(5,"[brightness scan] "<<Tminimum_value<<" -- "<<Tmaximum_value);
	}
}

// -----------------------------------------------------------  fft
//

template<class T> inline nPhysImageF<mcomplex>  
nPhysImageF<T>::ft2(enum phys_fft ftdir) {

	// 1. allocation
    fftw_complex *t = fftw_alloc_complex(getSurf());
    fftw_complex *Ft = fftw_alloc_complex(getSurf());
	
	nPhysImageF<mcomplex> ftbuf(getW(), getH(), mcomplex(0.,0.), "ftbuf");
	
	if (getSurf()>0) {
//        fftw_plan plan_t = fftw_plan_dft_2d(width, height, t, Ft, (ftdir == PHYS_FORWARD ? FFTW_FORWARD : FFTW_BACKWARD), FFTW_ESTIMATE);
        fftw_plan plan_t = fftw_plan_dft_2d(height, width, t, Ft, (ftdir == PHYS_FORWARD ? FFTW_FORWARD : FFTW_BACKWARD), FFTW_ESTIMATE);

        // 2. data copy
#pragma omp parallel for
        for (size_t i = 0; i < getSurf(); i++) {
            assign_val_to_fftw_complex(Timg_buffer[i], t[i]);
        }

//#pragma omp parallel for collapse(2)
//        for (size_t  j = 0; j < height; j++){
//            for (size_t i = 0; i < width; i++) {
//                assign_val_to_fftw_complex(Timg_matrix[j][i], t[i*height+j]);
//            }
//        }
		
		// 3. transform
		fftw_execute(plan_t);
		
		// 4. transplant
#pragma omp parallel for
        for (size_t i = 0; i < getSurf(); i++) {
            ftbuf.Timg_buffer[i]=mcomplex(Ft[i][0], Ft[i][1]);
        }

//#pragma omp parallel for collapse(2)
//        for (size_t  j = 0; j < height; j++){
//            for (size_t i = 0; i < width; i++) {
//                ftbuf.Timg_matrix[j][i] = mcomplex(Ft[i*height+j][0], Ft[i*height+j][1]);
//            }
//        }
		
		// 5. return
		fftw_free(t);
		fftw_free(Ft);
		fftw_destroy_plan(plan_t);
	}
	
	return ftbuf;
}

template<class T> inline nPhysImageF<mcomplex>  
nPhysImageF<T>::ft1(enum phys_direction imgdir, enum phys_fft ftdir)
{
	
	// che mi sta un po' sul culo lo switch...
	if (imgdir == PHYS_X) {
		// horizontal

		// 1. allocation
		fftw_complex *t = fftw_alloc_complex(getW());
		fftw_complex *Ft = fftw_alloc_complex(getW());
		nPhysImageF<mcomplex> ftbuf;
		ftbuf = nPhysImageF<mcomplex>(getW(), getH(), mcomplex(0.,0.));

		fftw_plan plan_t;
		
		if (ftdir == PHYS_FORWARD)
			plan_t = fftw_plan_dft_1d(getW(), t, Ft, FFTW_FORWARD, FFTW_ESTIMATE);
		else
			plan_t = fftw_plan_dft_1d(getW(), t, Ft, FFTW_BACKWARD, FFTW_ESTIMATE);

		// 2. data copy, transform and tralsplant
        for (size_t row_n = 0; row_n<getH(); row_n++) {
            for (size_t col_n = 0; col_n<getW(); col_n++)
				assign_val_to_fftw_complex(Timg_matrix[row_n][col_n], t[col_n]);
			fftw_execute(plan_t);
            for (size_t col_n = 0; col_n<getW(); col_n++)
				ftbuf.Timg_matrix[row_n][col_n] = mcomplex(Ft[col_n][0], Ft[col_n][1]);
			
		}

		// and don't bother to realign..

		fftw_destroy_plan(plan_t);
		fftw_free(t);
		fftw_free(Ft);

		// 5. return
		return ftbuf;

	} else {
		// vertical
	
		// 1. allocation
		fftw_complex *t = fftw_alloc_complex(getH());
		fftw_complex *Ft = fftw_alloc_complex(getH());
		nPhysImageF<mcomplex> ftbuf;
		ftbuf = nPhysImageF<mcomplex>(getW(), getH(), mcomplex(0.,0.));

		fftw_plan plan_t;
		
		if (ftdir == PHYS_FORWARD)
			plan_t = fftw_plan_dft_1d(getH(), t, Ft, FFTW_FORWARD, FFTW_ESTIMATE);
		else
			plan_t = fftw_plan_dft_1d(getH(), t, Ft, FFTW_BACKWARD, FFTW_ESTIMATE);

		// 2. data copy, transform and tralsplant
        for (size_t col_n = 0; col_n<getW(); col_n++) {
            for (size_t row_n = 0; row_n<getH(); row_n++)
				assign_val_to_fftw_complex(Timg_matrix[row_n][col_n], t[row_n]);
			fftw_execute(plan_t);
            for (size_t row_n = 0; row_n<getH(); row_n++)
				ftbuf.Timg_matrix[row_n][col_n] = mcomplex(Ft[row_n][0], Ft[row_n][1]);
			
		}

		// and don't bother to realign..

		fftw_destroy_plan(plan_t);
		fftw_free(t);
		fftw_free(Ft);

		// 5. return
		return ftbuf;
	}



}

// ------------------------------ deprecated functions -----------------------

template<class T> void
nPhysImageF<T>::fftshift() {


	// warning! : definitely a bad idea to back-transform a shifted spectrum
	T val;
	size_t hwidth = (width+1)/2;
	size_t hheight = (height+1)/2;
    for (size_t i=0; i<width/2; i++) {
        for (size_t j=0; j<height/2; j++) {
			val = Timg_matrix[j][i];
			Timg_matrix[j][i] = Timg_matrix[j+hheight][i+hwidth];
			Timg_matrix[j+hheight][i+hwidth] = val;
			
			val = Timg_matrix[j+hheight][i];
			Timg_matrix[j+hheight][i] = Timg_matrix[j][i+hwidth];
			Timg_matrix[j][i+hwidth] = val;
		}
	}
}

template<class T> void
nPhysImageF<T>::ifftshift() {


}


template<class T> inline nPhysImageF<mcomplex> * 
nPhysImageF<T>::getFFT(int direction) {
	throw phys_deprecated();
	// tu te la calcoli
	//if (direction == 0)
	//	throw pIF_FFT_error;
	//if (width == 0 || height == 0)
	//	throw pIF_FFT_error;

	// 1. allocation
    fftw_complex *t = fftw_alloc_complex(getSurf());
    fftw_complex *Ft = fftw_alloc_complex(getSurf());
	nPhysImageF<mcomplex> *ftbuf;

	ftbuf = new nPhysImageF<mcomplex>(width, height, mcomplex(0.,0.), "ftbuf");
	
	fftw_plan plan_t;
	if (direction > 0)
		plan_t = fftw_plan_dft_2d(width, height, t, Ft, FFTW_FORWARD, FFTW_ESTIMATE);
	else
		plan_t = fftw_plan_dft_2d(width, height, t, Ft, FFTW_BACKWARD, FFTW_ESTIMATE);
	
	// 2. data copy	
    for (size_t  j = 0; j < height; j++){
        for (size_t i = 0; i < width; i++) {
			//assign_val_to_fftw_complex(Timg_matrix[j][i], t[i+j*width]);
			assign_val_to_fftw_complex(Timg_matrix[j][i], t[i*height+j]);
			//t[i+j*width][0] = Timg_matrix[j][i];
			//t[i+j*width][1]=0.0;
		}
	}

	// 3. transform
	fftw_execute(plan_t);

	// 4. transplant
    for (size_t  j = 0; j < height; j++){
        for (size_t i = 0; i < width; i++) {
			//ftbuf->Timg_matrix[j][i] = mcomplex(Ft[i+j*width][0], Ft[i+j*width][1]);
			ftbuf->Timg_matrix[j][i] = mcomplex(Ft[i*height+j][0], Ft[i*height+j][1]);
		}
	}

	// 5. return
	fftw_free(t);
	fftw_free(Ft);
	fftw_destroy_plan(plan_t);
	
	DEBUG(10,"out of fft");

	return ftbuf;
}

template<class T> nPhysImageF<mcomplex> * 
nPhysImageF<T>::getSingletonFFT(enum phys_direction dir, enum phys_fft fftdir) {
	throw phys_deprecated();
	
	int mydir;
	switch (dir) {
        case 0:
			mydir = 0;
			break;

        case 1:
			mydir = 1;
			break;

		default:
			//throw pIF_FFT_error;
			return NULL;
			break;
	}
	
	// che mi sta un po' sul culo lo switch...
	if (mydir == 0) {
		// horizontal

		// 1. allocation
		fftw_complex *t = fftw_alloc_complex(width);
		fftw_complex *Ft = fftw_alloc_complex(width);
		nPhysImageF<mcomplex> *ftbuf;
		ftbuf = new nPhysImageF<mcomplex>(width, height, mcomplex(0.,0.));

		fftw_plan plan_t;
		
		if (fftdir == PHYS_FORWARD)
			plan_t = fftw_plan_dft_1d(width, t, Ft, FFTW_FORWARD, FFTW_ESTIMATE);
		else
			plan_t = fftw_plan_dft_1d(width, t, Ft, FFTW_BACKWARD, FFTW_ESTIMATE);

		// 2. data copy, transform and tralsplant
        for (size_t row_n = 0; row_n<height; row_n++) {
            for (size_t col_n = 0; col_n<width; col_n++)
				assign_val_to_fftw_complex(Timg_matrix[row_n][col_n], t[col_n]);
			fftw_execute(plan_t);
            for (size_t col_n = 0; col_n<width; col_n++)
				ftbuf->Timg_matrix[row_n][col_n] = mcomplex(Ft[col_n][0], Ft[col_n][1]);
			
		}

		// and don't bother to realign..

		fftw_destroy_plan(plan_t);
		fftw_free(t);
		fftw_free(Ft);

		// 5. return
		return ftbuf;

	} else if (mydir == 1) {
		// vertical
	
		// 1. allocation
		fftw_complex *t = fftw_alloc_complex(height);
		fftw_complex *Ft = fftw_alloc_complex(height);
		nPhysImageF<mcomplex> *ftbuf;
		ftbuf = new nPhysImageF<mcomplex>(width, height, mcomplex(0.,0.));

		fftw_plan plan_t;
		
		if (fftdir == PHYS_FORWARD)
			plan_t = fftw_plan_dft_1d(height, t, Ft, FFTW_FORWARD, FFTW_ESTIMATE);
		else
			plan_t = fftw_plan_dft_1d(height, t, Ft, FFTW_BACKWARD, FFTW_ESTIMATE);

		// 2. data copy, transform and tralsplant
        for (size_t col_n = 0; col_n<width; col_n++) {
            for (size_t row_n = 0; row_n<height; row_n++)
				assign_val_to_fftw_complex(Timg_matrix[row_n][col_n], t[row_n]);
			fftw_execute(plan_t);
            for (size_t row_n = 0; row_n<height; row_n++)
				ftbuf->Timg_matrix[row_n][col_n] = mcomplex(Ft[row_n][0], Ft[row_n][1]);
			
		}

		// and don't bother to realign..

		fftw_destroy_plan(plan_t);
		fftw_free(t);
		fftw_free(Ft);

		// 5. return
		return ftbuf;
	}

}

//template<> nPhysImageF<complex<double> > *
//nPhysImageF<complex<double> >::getFFT(int direction) {
	// tu pure te la calcoli
//}

// output
//
// --------------------------------------------------------------------------------------------------

template<class T> void
nPhysImageF<T>::writeASC(const char *ofilename) {
	// alla bruttissimo dio
	DEBUG(5,getName() << " Short: " << getShortName() << " from: " << getFromName());
	std::ofstream ofile(ofilename);
	if (ofile.good()) {
        for (size_t i=0; i<height; i++) {
            for (size_t j=0; j<width-1; j++)
				ofile<<std::setprecision(8)<<Timg_buffer[j+i*width]<<"\t";
			ofile<<std::setprecision(8)<<Timg_buffer[width-1+i*width] << "\n";
		}
		ofile.close();
    } else {
        throw phys_fileerror("ofstream error");
    }
}

// specializzazione per classe complex
template<> inline void
nPhysImageF<mcomplex>::writeASC(const char *ofilename) {
	std::ofstream r_ofile((std::string(ofilename)+".re").c_str());
	std::ofstream i_ofile((std::string(ofilename)+".im").c_str());	
	if (r_ofile.good() && i_ofile.good()) {

        for (size_t i=0; i<height; i++) {
            for (size_t j=0; j<width; j++) {
				r_ofile<<std::setprecision(8)<<Timg_buffer[j+i*width].real()<<"\t";
				i_ofile<<std::setprecision(8)<<Timg_buffer[j+i*width].imag()<<"\t";
			}
			r_ofile<<"\n";
			i_ofile<<"\n";
		}
		r_ofile.close();
		i_ofile.close();
    } else {
        throw phys_fileerror("ofstream error");
    }
}


template<class T> void
nPhysImageF<T>::writeRAW(const char *ofilename) {
    throw phys_deprecated();
    std::ofstream ofile(ofilename);

	ofile<<"ImagLab-RAW\t"<<width<<"\t"<<height<<"\t"<<typeid(*Timg_buffer).name()<<"\n";
	// alla bruttissimo dio
    ofile.write((char *)Timg_buffer, getSurf()*sizeof(T));
	ofile.close();
}

// ------------------------------ operators ------------------------------------

/*template<class T> nPhysImageF<T> &
nPhysImageF<T>::operator= (const nPhysImageF<T> &other) {
	resize(other.width, other.height);
    memcpy(Timg_buffer, other.Timg_buffer, getSurf()*sizeof(T));
	Tmaximum_value = other.Tmaximum_value;
	Tminimum_value = other.Tminimum_value;
	return *this;
}*/


template<class T> nPhysImageF<T>
nPhysImageF<T>::operator+ (const nPhysImageF<T> &other) const {

	if ( (width != other.width) || (height != other.height) )
		return *this;
	
	nPhysImageF<T> new_img;
	new_img.resize(width, height);

	new_img.set_origin(property.at("origin"));
	new_img.set_scale(property.at("scale"));
	new_img.setName("("+property.at("phys_name").get_str()+")+("+other.property.at("phys_name").get_str()+")");
	new_img.setShortName("Add");
    for (size_t i=0; i<height*width; i++)
		new_img.Timg_buffer[i] = (T)(Timg_buffer[i]) + (T)(other.Timg_buffer[i]);
		
	return(new_img);
}

template<class T> nPhysImageF<T>
nPhysImageF<T>::operator+ (T &val) const {
	
	nPhysImageF<T> new_img(*this);
	std::stringstream ss;
	ss<<val;
	
	new_img.setName("("+property.at("phys_name").get_str()+")+("+ss.str()+")");
	new_img.setShortName("Add "+ss.str());
    for (size_t i=0; i<getSurf(); i++)
		new_img.Timg_buffer[i] += val;
		
	return(new_img);
}

template<class T> nPhysImageF<T>
nPhysImageF<T>::operator- (const nPhysImageF<T> &other) const {

	if ( (width != other.width) || (height != other.height) )
		return *this;
	
	nPhysImageF<T> new_img;
	new_img.resize(width, height);


	new_img.set_origin(property.at("origin"));
	new_img.set_scale(property.at("scale"));
	new_img.setName("("+property.at("phys_name").get_str()+")-("+other.property.at("phys_name").get_str()+")");
	new_img.setShortName("Subtract");
	
    for (size_t i=0; i<height*width; i++)
		new_img.Timg_buffer[i] = Timg_buffer[i] - other.Timg_buffer[i];
		
	return(new_img);
}

template<class T> nPhysImageF<T>
nPhysImageF<T>::operator- (T &val) const {
	
	nPhysImageF<T> new_img(*this);
	std::stringstream ss;
	ss<<val;
	
	new_img.setName("("+property.at("phys_name").get_str()+")+("+ss.str()+")");
	new_img.setShortName("Add "+ss.str());
    for (size_t i=0; i<getSurf(); i++)
		new_img.Timg_buffer[i] -= val;
		
	return(new_img);
}

template<class T> nPhysImageF<T>
nPhysImageF<T>::operator* (const nPhysImageF<T> &other) const {

	if ( (width != other.width) || (height != other.height) )
		return *this;
	
	nPhysImageF<T> new_img;
	
	new_img.set_origin(property.at("origin"));
	new_img.set_scale(property.at("scale"));
	new_img.resize(width, height);
	new_img.setName("("+property.at("phys_name").get_str()+")*("+other.property.at("phys_name").get_str()+")");
	new_img.setShortName("Multiply");

    for (size_t i=0; i<height*width; i++)
		new_img.Timg_buffer[i] = Timg_buffer[i] * other.Timg_buffer[i];
		
	return(new_img);
}




template<class T> nPhysImageF<T>
nPhysImageF<T>::operator/ (const nPhysImageF<T> &other) const {

	if ( (width != other.width) || (height != other.height) )
		return *this;
	
	nPhysImageF<T> new_img;
	
	new_img.set_origin(property.at("origin"));
	new_img.set_scale(property.at("scale"));
	new_img.resize(width, height);
	new_img.setName("("+property.at("phys_name").get_str()+")/("+other.property.at("phys_name").get_str()+")");
	new_img.setShortName("Divide");
	
    for (size_t i=0; i<height*width; i++)
		new_img.Timg_buffer[i] = Timg_buffer[i] / other.Timg_buffer[i];
		
	return(new_img);
}



template<class T> inline T nPhysImageF<T>::get_min() {
	return Tminimum_value;
}

template<class T> inline T nPhysImageF<T>::get_max() {
	return Tmaximum_value;
}

template<class T> inline bidimvec<T> nPhysImageF<T>::get_min_max() {
	return bidimvec<T>(get_min(),get_max());
}



#endif
