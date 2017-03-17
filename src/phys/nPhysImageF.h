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
#include <memory>

#include <gsl/gsl_math.h>
#include <gsl/gsl_const_mksa.h>
#include <gsl/gsl_const_num.h>
#include <fftw3.h>
#include <time.h>
#include <assert.h>

#ifdef HAVE_OPENMP
#include <omp.h>
#endif


#include "mcomplex.h"
#include "anymap.h"
#include "debug-messages.h"

#include "physData.h"

#include "bidimvec.h"

#ifndef __nPhysImageF_h
#define __nPhysImageF_h

enum phys_way { PHYS_POS, PHYS_NEG };
enum phys_fft { PHYS_FORWARD, PHYS_BACKWARD };

typedef std::string phys_type;
#define PHYS_FILE "phys_file"
#define PHYS_RFILE "phys_rfile"
#define PHYS_DYN "phys_dyn"


class phys_properties : public anymap {

	public:
		phys_properties()
			: anymap()
		{
			(*this)["origin"] = vec2f(0,0);
			(*this)["scale"] = vec2f(1,1);
		}

};

struct phys_point_str {
        size_t x;
        size_t y;
};
typedef struct phys_point_str phys_point;


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
		void resize(size_t new_w, size_t new_h, T val=0)
		{ sh_data->resize(new_w, new_h, val); }

		//! re-reads buffer for minimum/maximum value
		void TscanBrightness(void);

		//! 1D get functions for row/column access
		size_t get_Tvector(enum phys_direction, size_t, size_t, T*, size_t, phys_way orient=PHYS_POS);

		//! 1D set functions for row/column access
		void set_Tvector(enum phys_direction, size_t, size_t, T*, size_t, phys_way orient=PHYS_POS);

		//! get row specialized function
		void get_Trow(size_t, size_t, std::vector<T> &);

		std::string class_name ()
		{ return std::string(typeid(T).name()); }

		//! min/max point coordinates
		vec2 min_Tv, max_Tv;

		//phys_properties property; now on anymap
		//anymap property; specialized class: phys_properties
		phys_properties prop;
		phys_properties display_property;

		//! Exceptions (should use dException)
		std::exception E_alloc, E_access, E_unsafe;

		// --------------------- image points and statistics ----------------------------

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

		// ------------------------------------------------------------------------
		// x-connections to physData
		inline size_t getW() const
		{ return sh_data->getW(); }
		inline size_t getH() const
		{ return sh_data->getH(); }
		inline size_t getSurf() const
		{ return sh_data->getSurf(); }
		T sum()
		{ return sh_data->sum(); }

		typename std::vector<T>::iterator buf_itr()
		{ return sh_data->buf_itr(); }
		const T *data_pointer()
		{ return sh_data->data_pointer(); }

		void swap_vector(size_t w, size_t h, std::vector<T> &vec) 
		{ sh_data->swap_vector(w, h, vec); }

		inline T point(size_t x, size_t y, T nan_value=std::numeric_limits<T>::signaling_NaN()) const
		{ return sh_data->point(x, y, nan_value); }

		inline T point(vec2 vv, T nan_value=std::numeric_limits<T>::signaling_NaN()) const
		{ return sh_data->point(vv.x(), vv.y(), nan_value); }

		inline T point(size_t xy, T nan_value=std::numeric_limits<T>::signaling_NaN()) const
		{ return sh_data->point(xy, nan_value); }

		inline void set(size_t x, size_t y, T val)
		{ sh_data->set(x, y, val); }
		inline void set(size_t xy, T val)
		{ sh_data->set(xy, val); }

                inline void set_Trow(size_t index, size_t offset, std::vector<T> &vec)
                { sh_data->set_Trow(index, offset, vec); }


		// ------------------------------------------------------------------------
		// ------------------------------------------------------------------------
		// ------------------------------------------------------------------------

		// assignment operator -- SHALLOW COPY
		nPhysImageF<T> & operator= (const nPhysImageF<T> &rhs)
		{
			DEBUG(10, "shallow copy");

			// copy everything
			prop = rhs.prop; // probably missing DEEP operator

			Tmaximum_value = rhs.Tmaximum_value;
			Tminimum_value = rhs.Tminimum_value;

			sh_data = rhs.sh_data;

			return *this;

		}


		// check for shallowness
		bool operator== (const nPhysImageF<T> &rhs)
		{ return (sh_data.get() == rhs.sh_data.get()); } 
		bool operator!= (const nPhysImageF<T> &rhs)
		{ return !(sh_data.get() == rhs.sh_data.get()); } 

		template <class U> operator nPhysImageF<U>  () const
		{
			DEBUG(5,"cast constructor ------------------------------------");
			nPhysImageF<U> lhs;
			//		lhs = new nPhysImageF<U>;
			lhs.resize(getW(), getH());
#pragma omp parallel for
			for (size_t ii=0; ii<getSurf(); ii++)
				lhs.set(ii, U(sh_data->point(ii)));	
			//lhs.Timg_buffer[i] = U(Timg_buffer[i]);	

			lhs.TscanBrightness();

			//lhs->object_name = object_name;
			//lhs->filename=filename;
			lhs.prop = prop;
			return lhs;
		}

		//! Returns a DEEP copy of the object 
		nPhysImageF<T> &copy()
		{
			// questo chissa' come funziona...
			//nPhysImageF<T> *new_img = new nPhysImageF<T>(*this);

			// a manina la generazione della copia equivarrebbe a:

			nPhysImageF<T> *new_img = new nPhysImageF<T>(*this); // questo dovr`a fare riferimento al deep-constructor di physData
			return *new_img;

			// non sono sicuro delle seguenti cose: 1. se questa roba vive fuori scope, 2. se davvero `e una copia deep
		}


		// get point (to be used for accessing data - no overload)
		inline T getPoint(double x, double y, T nan_value=std::numeric_limits<T>::quiet_NaN()) {
			if (x>=0 && y>=0) {
				size_t x1=(size_t)x;
				size_t y1=(size_t)y;
				if (x==x1 && y==y1) return sh_data->point((size_t)x1,(size_t)y1);
				size_t x2=x1+1;
				size_t y2=y1+1;
				if (x2<getW() && y2<getH()) {
					T data11=sh_data->point(x1, y1);
					T data12=sh_data->point(x2, y1);
					T data21=sh_data->point(x1, y2);
					T data22=sh_data->point(x2, y2);

					return (y2-y)*((x2-x)*data11+(x-x1)*data12)+(y-y1)*((x2-x)*data21+(x-x1)*data22);
				}
			}
			return nan_value;
		}
		inline T getPoint(bidimvec<double> p, T nan_value=std::numeric_limits<T>::quiet_NaN()) {
			return getPoint(p.x(),p.y(),nan_value);
		}

		inline void set(T val) { //! set a value allover the matrix
			DEBUG(PRINTVAR(val));
			for (size_t ii=0; ii<getSurf(); ii++) {
				sh_data->set(ii, val);
			}
			TscanBrightness();
		}


		inline size_t getSizeByIndex(enum phys_direction dir)
		{ if (dir==PHYS_X) return getW(); if (dir == PHYS_Y) return getH(); return 0; }

		inline bidimvec<size_t> getSize()
		{ return bidimvec<size_t>(getW(), getH()); }

		inline bool isInside(size_t x, size_t y) {
			if ((x < getW()) && (y < getH()))
				return true;
			return false;
		}


		nPhysImageF<T> sub(size_t, size_t, size_t, size_t);

		// getting and setting properties
		inline std::string getName()
		{ return prop["phys_name"]; }
		void setName(std::string name)
		{ prop["phys_name"] = name; }

		//tom
		inline std::string getShortName()
		{ return prop["phys_short_name"]; }

		inline void setShortName(std::string name)
		{ prop["phys_short_name"] = name; }

		inline std::string getFromName()
		{ return prop["phys_from_name"]; }

		inline void setFromName(std::string name)
		{ prop["phys_from_name"] = name; }

		inline vec2f get_origin()
		{ return prop["origin"]; }

		inline double get_origin(enum phys_direction direction)
		{ return (direction==PHYS_X ? vec2f(prop["origin"].get_str()).x() : vec2f(prop["origin"].get_str()).y()); }

		inline void set_origin(T val_x, T val_y)
		{ prop["origin"] = vec2f(val_x,val_y); }

		inline void set_origin(vec2f val) 
		{ prop["origin"] = val; }

		inline vec2f get_scale()
		{ return prop["scale"]; }

		inline double get_scale(enum phys_direction direction)
		{ return (direction==PHYS_X ? vec2f(prop["scale"].get_str()).x() : vec2f(prop["scale"].get_str()).y()); }

		inline void set_scale(T val_x, T val_y)
		{ prop["scale"] = vec2f(val_x,val_y); }

		inline void set_scale(vec2f val)
		{ prop["scale"] = val; }

		inline vec2f to_real(vec2f val) { 
			vec2f oo, ss; 
			oo = prop["origin"]; ss = prop["scale"];
			return vec2f((val.x()-oo.x())*ss.x(),(val.y()-oo.y())*ss.y()); 
		}

		inline vec2f to_pixel(vec2f val) { 
			vec2f oo, ss; 
			oo = prop["origin"]; ss = prop["scale"];
			return vec2f(val.x()/ss.x()+oo.x(),val.y()/ss.y()+oo.y()); 
		}

		//end

		inline phys_type getType()
		{ return prop["phys_orig"]; }

		void init_Tvariables();


	private:

		//! TODO: pass to bidimvec<T>
		T Tmaximum_value;
		T Tminimum_value;

		std::shared_ptr<physData<T> > sh_data;



};

typedef nPhysImageF<double> physD;
typedef nPhysImageF<mcomplex> physC;

// --------------------------------------------------------------------------------------------

	template<class T>
nPhysImageF<T>::nPhysImageF()
{ }

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
	//setType(pp);
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
	DEBUG(10, "copy constructor");
	init_Tvariables();
	resize(oth.getW(), oth.getH());

	//	memcpy(Timg_buffer, oth.Timg_buffer, getSurf()*sizeof(T));
	//std::copy(oth.Timg_buffer, oth.Timg_buffer+getSurf(), Timg_buffer);
	
	// this is probably MUCH MUCH slower. It has to be tested
	DEBUG(10, "my size: "<<getSurf());
	for (register size_t ii=0; ii<getSurf(); ii++) {
		set(ii, oth.point(ii));
	}
	prop = oth.prop;

	setShortName(sName);
	TscanBrightness();
	//	std::cerr<<"end copy constructor ------------------------------------"<<std::endl;
}


template<class T>
nPhysImageF<T>::nPhysImageF(size_t w, size_t h, T val, std::string obj_name)
{
	init_Tvariables();
	setName(obj_name);

	resize(w, h, val);
}



template<class T>
nPhysImageF<T>::nPhysImageF(T *o_buffer, size_t w, size_t h, std::string obj_name) {

	throw phys_trashable();
	init_Tvariables();

	setName(obj_name);
	resize(w, h);

	//memcpy(Timg_buffer, o_buffer, getSurf()*sizeof(T));
	for (register size_t ii=0; ii<getSurf(); ii++) {
		set(ii, o_buffer[ii]);
	}

}


template<class T>
nPhysImageF<T>::~nPhysImageF()
{
	DEBUG(1,"destructor: ["<<(void *)this<<", buffer "<<(void *)sh_data.get()<<"] "<<  getShortName() << " : " << getName() << " has "<<sh_data.use_count() << " instances. " );
	sh_data.reset();
}

// -----------------------------------------------------------------------

	template<class T> void
nPhysImageF<T>::init_Tvariables()
{

	sh_data = std::make_shared<physData<T> > ();

	min_Tv=vec2(-1,-1);
	max_Tv=vec2(-1,-1);

	Tmaximum_value = 0;
	Tminimum_value = 0;


}

//template<class T> void
//nPhysImageF<T>::_init_temp_pointers()
//{
//	Timg_buffer = NULL;
//	Timg_matrix = NULL;
//
//	if (vector_buf)
//		delete vector_buf;
//	if (axis_buf)
//		delete axis_buf;
//
//	vector_buf = new double *[2];
//	memset(vector_buf, 0, 2*sizeof(double *));
//	axis_buf = new double *[2];
//	memset(axis_buf, 0, 2*sizeof(double *));
//
//	histogram.resize(0);
//}




// ----------------------- DATA ACCESS ----------------------------	

	template<class T> size_t
nPhysImageF<T>::get_Tvector(enum phys_direction direction, size_t index, size_t offset, T *ptr, size_t size, phys_way orient)
{
	// copies a vector to an external buffer (useful for Abel inversion)
	// vector is taken on direction (0=x, 1=y), starting from offset and for size points

	size_t copy_len = size;
	if (direction == PHYS_X) {
		if (orient == PHYS_POS) {
			if ((size+offset > getW())) 	// spem longam spatio brevi reseces ;-)
				copy_len = getW()-offset;
			for (size_t i=0; i<copy_len; i++)
				ptr[i] = sh_data->point(offset+i, index, 0.);
			//ptr[i] = Timg_matrix[index][offset+i];
			//ptr[i] = Timg_buffer[index*width+offset+i];
		} else {
			//if (((offset+1)-size < 0)) 	
			if (((offset+1) < size)) 	
				copy_len = offset+1;
			for (size_t i=0; i<copy_len; i++)
				ptr[i] = sh_data->point(offset-i, index, 0.);
			//ptr[i] = Timg_matrix[index][offset-i];
		}

	} else if (direction == PHYS_Y) {
		if (orient == PHYS_POS) {
			if ((size+offset > getH()))
				copy_len = getH()-offset;
			for (size_t i=0; i<copy_len; i++)
				ptr[i] = sh_data->point(index, offset+i, 0.);
			//ptr[i] = Timg_matrix[offset+i][index];
			//ptr[i] = Timg_buffer[(offset+i)*width+index];
		} else {
			//if (((offset+1)-size < 0)) 	
			if (((offset+1) < size)) 	
				copy_len = offset+1;
			for (size_t i=0; i<copy_len; i++)
				ptr[i] = sh_data->point(index, offset-i, 0.);
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
			if ((size+offset > getW())) 	// spem longam spatio brevi reseces ;-)
				copy_len = getW()-offset;
			for (size_t i=0; i<copy_len; i++)
				sh_data->set(offset+i, index, ptr[i]);
				//Timg_matrix[index][offset+i] = ptr[i];
		} else {
			//if (((offset+1) -size < 0)) 	
			if (((offset+1) < size)) 	
				copy_len = offset+1;
			for (size_t i=0; i<copy_len; i++)
				sh_data->set(offset-i, index, ptr[i]);
				//Timg_matrix[index][offset-i] = ptr[i];
		}

	} else if (direction == PHYS_Y) {
		if (orient == PHYS_POS) {
			if ((size+offset > getH()))
				copy_len = getH()-offset;
			for (size_t i=0; i<copy_len; i++)
				sh_data->set(index, offset+i, ptr[i]);
				//Timg_matrix[offset+i][index] = ptr[i];
		} else {
			//if (((offset+1)-size < 0)) 	
			if (((offset+1) < size)) 	
				copy_len = offset+1;
			for (size_t i=0; i<copy_len; i++) {
				sh_data->set(index, offset-i, ptr[i]);
				//Timg_matrix[offset-i][index] = ptr[i];
                        }
		}
	}
}

//! get submatrix
template <class T> nPhysImageF<T> 
nPhysImageF<T>::sub(size_t x, size_t y, size_t Dx, size_t Dy) {

	nPhysImageF<T> subphys(Dx, Dy, 0.);
	subphys.set_origin(get_origin()-vec2f(x,y));
	subphys.set_scale(get_scale());

	//subphys.setType(PHYS_DYN);

	if (isInside(x, y)) {
		size_t copy_w = std::min(x+Dx, (size_t)getW()); // FIXME
		size_t copy_h = std::min(y+Dy, (size_t)getH());
		std::vector<T> copyvec;
		copyvec.resize(copy_w);
		for (size_t i=y; i<copy_h; i++) {
			std::copy(sh_data->data_pointer()+x, sh_data->data_pointer()+copy_w, copyvec.begin());
			subphys.set_Trow(i, 0, copyvec);
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
			if (std::isfinite(sh_data->point(i))) {	
				if (!found) {
					Tminimum_value = sh_data->point(i);
					Tmaximum_value = Tminimum_value;
					found=true;
				} else {
					if ((sh_data->point(i)) > Tmaximum_value) {
						max_Tv=vec2(i%getW(), i/getW());
						Tmaximum_value = (sh_data->point(i));
					} else if ((sh_data->point(i)) < Tminimum_value) {
						min_Tv=vec2(i%getW(), i/getW());
						Tminimum_value = (sh_data->point(i));
					}
				}
			}
		}
		DEBUG(5,"[brightness scan] "<<Tminimum_value<<" -- "<<Tmaximum_value);
	}
}

// ------------------------------ operators ------------------------------------


template<class T> nPhysImageF<T>
nPhysImageF<T>::operator+ (const nPhysImageF<T> &other) const {

	if ( (getW() != other.getW()) || (getH() != other.getH()) )
		return *this;

	nPhysImageF<T> new_img;
	new_img.resize(getW(), getH());

	new_img.set_origin(prop.at("origin"));
	new_img.set_scale(prop.at("scale"));
	new_img.setName("("+prop.at("phys_name").get_str()+")+("+other.prop.at("phys_name").get_str()+")");
	new_img.setShortName("Add");
	for (size_t i=0; i<getH()*getW(); i++)
		new_img.set(i, (T)(point(i)) + (T)(other.point(i)));

	return(new_img);
}

template<class T> nPhysImageF<T>
nPhysImageF<T>::operator+ (T &val) const {

	nPhysImageF<T> new_img(*this);
	std::stringstream ss;
	ss<<val;

	new_img.setName("("+prop.at("phys_name").get_str()+")+("+ss.str()+")");
	new_img.setShortName("Add "+ss.str());
	for (size_t i=0; i<getSurf(); i++)
		new_img.Timg_buffer[i] += val;

	return(new_img);
}

template<class T> nPhysImageF<T>
nPhysImageF<T>::operator- (const nPhysImageF<T> &other) const {

	if ( (getW() != other.getW()) || (getH() != other.getH()) )
		return *this;

	nPhysImageF<T> new_img;
	new_img.resize(getW(), getH());


	new_img.set_origin(prop.at("origin"));
	new_img.set_scale(prop.at("scale"));
	new_img.setName("("+prop.at("phys_name").get_str()+")-("+other.prop.at("phys_name").get_str()+")");
	new_img.setShortName("Subtract");

	for (size_t i=0; i<getH()*getW(); i++)
		new_img.set(i, point(i) - other.point(i));

	return(new_img);
}

template<class T> nPhysImageF<T>
nPhysImageF<T>::operator- (T &val) const {

	nPhysImageF<T> new_img(*this);
	std::stringstream ss;
	ss<<val;

	new_img.setName("("+prop.at("phys_name").get_str()+")+("+ss.str()+")");
	new_img.setShortName("Add "+ss.str());
	for (size_t i=0; i<getSurf(); i++)
		new_img.Timg_buffer[i] -= val;

	return(new_img);
}

template<class T> nPhysImageF<T>
nPhysImageF<T>::operator* (const nPhysImageF<T> &other) const {

	if ( (getW() != other.getW()) || (getH() != other.getH()) )
		return *this;

	nPhysImageF<T> new_img;

	new_img.set_origin(prop.at("origin"));
	new_img.set_scale(prop.at("scale"));
	new_img.resize(getW(), getH());
	new_img.setName("("+prop.at("phys_name").get_str()+")*("+other.prop.at("phys_name").get_str()+")");
	new_img.setShortName("Multiply");

	for (size_t i=0; i<getH()*getW(); i++)
		new_img.set(i, point(i) * other.point(i));

	return(new_img);
}




template<class T> nPhysImageF<T>
nPhysImageF<T>::operator/ (const nPhysImageF<T> &other) const {

	if ( (getW() != other.getW()) || (getH() != other.getH()) )
		return *this;

	nPhysImageF<T> new_img;

	new_img.set_origin(prop.at("origin"));
	new_img.set_scale(prop.at("scale"));
	new_img.resize(getW(), getH());
	new_img.setName("("+prop.at("phys_name").get_str()+")/("+other.prop.at("phys_name").get_str()+")");
	new_img.setShortName("Divide");

	for (size_t i=0; i<getH()*getW(); i++)
		new_img.set(i, point(i) / other.point(i));

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
