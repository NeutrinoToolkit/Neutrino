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
#include <string>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <vector>

#include <gsl/gsl_math.h>
#include "tools.h"

#ifndef __bidimvec_h
#define __bidimvec_h

#if (__GNUC_MINOR__ > 5)
enum phys_direction : size_t { PHYS_HORIZONTAL = 0, PHYS_X = 0, PHYS_ROW = 0,  PHYS_VERTICAL = 1, PHYS_Y = 1, PHYS_COLUMN = 1 };
#else
enum phys_direction { PHYS_HORIZONTAL = 0, PHYS_X = 0, PHYS_ROW = 0,  PHYS_VERTICAL = 1, PHYS_Y = 1, PHYS_COLUMN = 1 };
#endif

template <class T> bool
bd_from_string(T& t, const std::string& s)
{
  std::istringstream iss(s);
  return !(iss >> t).fail();
}


template <class T>
class bidimvec {

public:
	bidimvec(void);
	bidimvec(T v1, T v2);
	bidimvec(T &other);
	bidimvec(T *other);

	// pacchissimo...
	bidimvec(std::string);

	bidimvec<T> operator= (bidimvec<T>);
	bidimvec<T> operator+ (bidimvec<T>);
	bidimvec<T> operator- (bidimvec<T>);
	bidimvec<T> operator+= (bidimvec<T>);
	bidimvec<T> operator-= (bidimvec<T>);
	bidimvec<T> operator* (double);
	bidimvec<T> operator*= (double);
	bidimvec<T> operator/= (double);
	bidimvec<T> operator/ (double);
	bidimvec<T> operator- ();
    std::ostream& operator<< (std::ostream&);
	
	// products
	double operator| (bidimvec<T>);	// scalar product
	double operator^ (bidimvec<T>);	// angle between vectors

	inline bool operator== (bidimvec<T>);
	inline bool operator!= (bidimvec<T>);

	// cast
	template <class U> operator bidimvec<U> () const
	{ return bidimvec<U>(U(myval1), U(myval2)); }
	
	double mod();
	double mod2(); //square of mod (i.e. mod=sqrt(mod2))
	
	void norm();
	bidimvec<T> orthonorm(void);
	bidimvec<T> rotate(double);

	bidimvec<T> swap();

	//double intp(bidimvec<T>, bidimvec<T>);

	inline T x(void)
	{ return myval1; }
	inline T first(void)
	{ return myval1; }
	inline T y(void)
	{ return myval2; }
	inline T second(void)
	{ return myval2; }

	inline void set_first(T val)
	{ myval1= val;}
	inline void set_second(T val)
	{ myval2= val;}

    inline void set(bidimvec<T> val)
    {
        myval1= val.x();
        myval2= val.y();
    }

    inline T operator () (phys_direction dir) {
        return dir==0?myval1:myval2;
    }

private:
	T myval1, myval2;
};

template <class T>  
std::ostream& bidimvec<T>::operator<< (std::ostream& ss) {
        ss<<"("<<x()<<":"<<y()<<")";
        return ss;
}


template <class T> bidimvec<T>
bidimvec<T>::operator+= (bidimvec<T> rhs)
{ myval1+=rhs.x(); myval2+=rhs.y(); return *this; }

template <class T> bidimvec<T>
bidimvec<T>::operator-= (bidimvec<T> rhs)
{ myval1-=rhs.x(); myval2-=rhs.y(); return *this; }

template <class T> bidimvec<T>
bidimvec<T>::operator*= (double rhs)
{ myval1*=rhs; myval2*=rhs; return *this; }

template <class T> bidimvec<T>
bidimvec<T>::operator/= (double rhs)
{ myval1/=rhs; myval2/=rhs; return *this; }

template <class T> bool
bidimvec<T>::operator== (bidimvec<T> rhs)
{ return ( (myval1==rhs.x()) && (myval2==rhs.y())); }

template <class T> bool
bidimvec<T>::operator!= (bidimvec<T> rhs)
{ return ( (myval1!=rhs.x()) || (myval2!=rhs.y())); }

template <class T> double
bidimvec<T>::mod()
{ return sqrt(mod2()); }

template <class T> inline double
bidimvec<T>::mod2()
{ return myval1*myval1+myval2*myval2; }

template <class T> void
bidimvec<T>::norm(void)
{ double modulus = mod(); myval1/=modulus; myval2/=modulus; }

template <class T> bidimvec<T> 
bidimvec<T>::operator+ (bidimvec<T> rhs)
{ return bidimvec<T>(myval1+rhs.x(), myval2+rhs.y()); }

template <class T> bidimvec<T> 
bidimvec<T>::operator- (bidimvec<T> rhs)
{ return bidimvec<T>(myval1-rhs.x(), myval2-rhs.y()); }

template <class T> bidimvec<T>
operator+ (double lhs, bidimvec<T> rhs)
{ return rhs+lhs; }

template <class T> bidimvec<T>
bidimvec<T>::operator/ (double rhs)
{ return (*this * (1/rhs)); }

template <class T> bidimvec<T> 
bidimvec<T>::operator* (double rhs)
{ return bidimvec<T>(myval1*rhs, myval2*rhs); }

template <class T> bidimvec<T>
operator* (double lhs, bidimvec<T> rhs)
{ return rhs*lhs; }

template <class T> bidimvec<T>
bidimvec<T>::operator- ()
{ return bidimvec<T>(-myval1, -myval2); }




template <class T> bidimvec<T>
bidimvec<T>::operator= (bidimvec<T> rhs)
{
	myval1 = rhs.x();
	myval2 = rhs.y();
	return (*this);
}

// lascio la definizione piu' generica, tentando di stare attento a non fare cazzate
template <class T> double
bidimvec<T>::operator| (bidimvec<T> rhs)
{
	return (myval1*rhs.x()+myval2*rhs.y());
}


// sort of deviation of the cross product to the 2d case. Basically the normal component
template <class T> double
bidimvec<T>::operator^ (bidimvec<T> rhs)
{
	double theta1 = (myval1 == 0) ? M_PI_2 : atan(myval2/myval1);
	double theta2 = (rhs.x() == 0) ? M_PI_2 : atan(rhs.y()/rhs.x());
	std::cout<<*this<<", "<<rhs<<"th1 "<<theta1<<", th2 "<<theta2<<std::endl;
	return (theta1-theta2);
}


//creators

template <class T>
bidimvec<T>::bidimvec(void) 
	: myval1(0), myval2(0) 
{}



template <class T>
bidimvec<T>::bidimvec(T v1, T v2) 
	: myval1(v1), myval2(v2) 
{}

template <class T>
bidimvec<T>::bidimvec(std::string str)
{
	int ref1 = str.find("(",0), ref2 = str.find(":",0), ref3 = str.find(")",0);
	if (! bd_from_string<T>(myval1, str.substr(ref1+1, ref2-ref1-1) ))
        DEBUG("Vec conversion problems! string was: "<<str.substr(ref1+1, ref2-ref1-1)<<", val is "<<myval1);
	
	if (! bd_from_string<T>(myval2, str.substr(ref2+1, ref3-ref2-1) ))
        DEBUG("Vec conversion problems! string was: "<<str.substr(ref2+1, ref3-ref2-1)<<", val is "<<myval2);


	//std::cout<<"[bidimvec<T>] std::str constructor: "<<*this<<std::endl;
}

template <class T>
bidimvec<T>::bidimvec(T &other) 
	: myval1(other.x()), myval2(other.y()) 
{}


template <class T>
bidimvec<T>::bidimvec(T *other) 
	: myval1(other->x()), myval2(other->y()) 
{}


// -------------------------- external operators -----------------------------


template <class T> std::ostream &
operator<< (std::ostream &lhs, bidimvec<T> &myvec)
{
	lhs<<"("<<myvec.x()<<":"<<myvec.y()<<")";
	return lhs;
}

// ------------------ math methods -------------------

template <class T> bidimvec<T>
bidimvec<T>::orthonorm(void)
{ return bidimvec<T>( (myval2==0) ? 0 : 1, (myval2 == 0) ? myval1 : -myval1/myval2 ); }
//{ return bidimvec<T>( (myval2==0) ? 0 : -.5/myval2, (myval1 == 0) ? 0 : .5/myval1 ); }

template <class T> bidimvec<T>
bidimvec<T>::swap(void)
{ return bidimvec<T>(second(),first());}

template <class T> bidimvec<T>
bidimvec<T>::rotate(double theta)
{ return bidimvec<T>( myval1*cos(theta)-myval2*sin(theta), myval1*sin(theta)+myval2*cos(theta)); }

template <class T> bidimvec<T>
mul_P(bidimvec<T> v1, bidimvec<T> v2)
{ return bidimvec<T>(v1.x()*v2.x(), v1.y()*v2.y()); }

template <class T> bidimvec<T>
div_P(bidimvec<T> v1, bidimvec<T> v2)
{ return bidimvec<T>(v1.x()/v2.x(), v1.y()/v2.y()); }

template <class T> bool point_inside_poly(bidimvec<T> test, std::vector<bidimvec<T> > vert)
{
    bool c = false;
    unsigned int i,j;
    for (i = 0, j = vert.size()-1; i < vert.size(); j = i++) {
        if ( ((vert[i].y()>test.y()) != (vert[j].y()>test.y())) &&
            (test.x() < (vert[j].x()-vert[i].x()) * (test.y()-vert[i].y()) / (vert[j].y()-vert[i].y()) + vert[i].x()) )
            c = !c;
    }
    return c;
}


// ------------------ helper functions ------------------


	// ignoro cosa sia questo.. in ogni caso e' meglio non usarlo
	template <class T> 
	double intp(bidimvec<T> v1, bidimvec<T> v2)
	{ return sqrt(v1.x()*v2.x()+v1.y()*v2.y()); }
	
	template <class T> 
	double intp2(bidimvec<T> v1, bidimvec<T> v2)
	{ return (v1.x()*v2.x()+v1.y()*v2.y()); }

	template <class T>
	bidimvec<T> M2colmult(bidimvec<T> col1, bidimvec<T> col2, bidimvec<T> vec) 
	{ return bidimvec<T>( col1.x()*vec.x() + col2.x()*vec.y(), col1.y()*vec.x()+col2.y()*vec.y() );	}
	
	template <class T> double
	interp_alpha(bidimvec<T> ip, bidimvec<T> ep)
	{ return (ep.y()-ip.y())/(ep.x()-ip.x()); }
	
	template <class T> double
	interp_beta(bidimvec<T> ip, bidimvec<T> ep)
	{ return (ip.y()-interp_alpha(ip,ep)*ip.x()); }

namespace vmath {

	template <class T> bidimvec<T>
	min(bidimvec<T> v1, bidimvec<T> v2)
	{ return bidimvec<T>(std::min(v1.x(), v2.x()), std::min(v1.y(), v2.y())); }
	
	template <class T> bidimvec<T>
	max(bidimvec<T> v1, bidimvec<T> v2)
	{ return bidimvec<T>(std::max(v1.x(), v2.x()), std::max(v1.y(), v2.y())); }

	// diagonal of the tensor product
	template <class T> bidimvec<T>
	td(bidimvec<T> v1, bidimvec<T> v2)
	{ return bidimvec<T>(v1.x()*v2.x(), v1.y()*v2.y()); }
	
}

// -----------------------------------------------------------

typedef bidimvec<double> vec2f;
typedef bidimvec<int> vec2i;
typedef bidimvec<int> vec2u;


#endif
