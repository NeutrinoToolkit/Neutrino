
#include <iostream>
#include <string>
#include <sstream>
#include <cmath>

#ifndef __tridimvec_h
#define __tridimvec_h

#define __3v_x tridimvec<double>(1.0,0.0,0.0)
#define __3v_y tridimvec<double>(0.0,1.0,0.0)
#define __3v_z tridimvec<double>(0.0,0.0,1.0)

template <class T> bool
tri_from_string(T& t, const std::string& s)
{
  std::istringstream iss(s);
  return !(iss >> t).fail();
}


template <class T>
class tridimvec {

public:
	tridimvec(void);
	tridimvec(T v1, T v2, T v3);
	tridimvec(T &other);
	tridimvec(T *other);

	// pacchissimo...
	tridimvec(std::string);

	tridimvec<T> operator= (tridimvec<T>);
	tridimvec<T> operator+ (tridimvec<T>);
	tridimvec<T> operator- (tridimvec<T>);
	tridimvec<T> operator* (double);
	tridimvec<T> operator*= (double);
	tridimvec<T> operator/= (double);
	tridimvec<T> operator/ (double);
	tridimvec<T> operator- ();

	inline bool operator== (tridimvec<T>);
	inline bool operator!= (tridimvec<T>);

	double mod();
	void norm();
	tridimvec<T> normVec();
	tridimvec<T> abs();
//	tridimvec<T> orthonorm(void);

	std::string str()
	{ std::ostringstream ss; ss<<"("<<myval1<<":"<<myval2<<":"<<myval3<<")"; return ss.str(); }

//	double intp(tridimvec<T>, tridimvec<T>);

	T x(void);
	T y(void);
	T z(void);


private:
	T myval1, myval2, myval3;
};

//template <class T> double
//tridimvec<T>::intp(tridimvec<T> v1, tridimvec<T> v2)
//{ return sqrt(v1.x()*v2.x()+v1.y()*v2.y()+v1.z()*v2.z()); }

template <class T> tridimvec<T>
tridimvec<T>::abs()
{ return tridimvec<T>(std::abs(myval1), std::abs(myval2), std::abs(myval3)); }

template <class T> double
intp(tridimvec<T> v1, tridimvec<T> v2)
{ return (v1.x()*v2.x()+v1.y()*v2.y()+v1.z()*v2.z()); }

template <class T> tridimvec<T>
tridimvec<T>::operator*= (double rhs)
{ myval1*=rhs; myval2*=rhs; myval3*=rhs; return *this; }

template <class T> tridimvec<T>
tridimvec<T>::operator/= (double rhs)
{ myval1/=rhs; myval2/=rhs; myval3/=rhs; return *this; }

template <class T> bool
tridimvec<T>::operator== (tridimvec<T> rhs)
{ return ( (myval1==rhs.x()) && (myval2==rhs.y()) && (myval3==rhs.z())); }

template <class T> bool
tridimvec<T>::operator!= (tridimvec<T> rhs)
{ return ( (myval1!=rhs.x()) && (myval2!=rhs.y()) && (myval3!=rhs.z())); }

template <class T> double
tridimvec<T>::mod()
{ return sqrt(pow(myval1,2)+pow(myval2,2)+pow(myval3,2)); }

template <class T> void
tridimvec<T>::norm(void)
{ double modulus = mod(); myval1/=modulus; myval2/=modulus; myval3/=modulus; }

template <class T> tridimvec<T>
tridimvec<T>::normVec(void)
{ double modulus = mod(); return tridimvec<T>(myval1/=modulus, myval2/=modulus, myval3/=modulus); }

template <class T> tridimvec<T> 
tridimvec<T>::operator+ (tridimvec<T> rhs)
{ return tridimvec<T>(myval1+rhs.x(), myval2+rhs.y(), myval3+rhs.z()); }

template <class T> tridimvec<T> 
tridimvec<T>::operator- (tridimvec<T> rhs)
{ return tridimvec<T>(myval1-rhs.x(), myval2-rhs.y(), myval3-rhs.z()); }

template <class T> tridimvec<T>
operator+ (double lhs, tridimvec<T> rhs)
{ return rhs+lhs; }

template <class T> tridimvec<T>
tridimvec<T>::operator/ (double rhs)
{ return (*this * (1/rhs)); }

template <class T> tridimvec<T> 
tridimvec<T>::operator* (double rhs)
{ return tridimvec<T>(myval1*rhs, myval2*rhs, myval3*rhs); }

template <class T> tridimvec<T>
operator* (double lhs, tridimvec<T> rhs)
{ return rhs*lhs; }

template <class T> tridimvec<T>
tridimvec<T>::operator- ()
{ return tridimvec<T>(-myval1, -myval2, -myval3); }


template <class T> std::ostream &
operator<< (std::ostream &lhs, tridimvec<T> &myvec)
{
	lhs<<"("<<myvec.x()<<":"<<myvec.y()<<":"<<myvec.z()<<")";
	return lhs;
}


template <class T> tridimvec<T>
tridimvec<T>::operator= (tridimvec<T> rhs)
{
	myval1 = rhs.x();
	myval2 = rhs.y();
	myval3 = rhs.z();
	return (*this);
}


template <class T>
tridimvec<T>::tridimvec(void) 
	: myval1(0), myval2(0), myval3(0) 
{}



template <class T>
tridimvec<T>::tridimvec(T v1, T v2, T v3) 
	: myval1(v1), myval2(v2), myval3(v3)
{}

template <class T>
tridimvec<T>::tridimvec(std::string str)
{
	int ref1 = str.find("(",0), ref2 = str.find(":",0), ref3 = str.find(":",ref2+1), ref4 = str.find(")",0) ;
	tri_from_string<T>(myval1, str.substr(ref1+1, ref2-ref1-1) );
	tri_from_string<T>(myval2, str.substr(ref2+1, ref3-ref2-1) );
	tri_from_string<T>(myval3, str.substr(ref3+1, ref4-ref3-1) );
	//std::cout<<"[tridimvec<T>] std::str constructor: "<<*this<<std::endl;
}

template <class T>
tridimvec<T>::tridimvec(T &other) 
	: myval1(other.x()), myval2(other.y()), myval3(other.z()) 
{}


template <class T>
tridimvec<T>::tridimvec(T *other) 
	: myval1(other->x()), myval2(other->y()), myval3(other->z()) 
{}


template <class T> T
tridimvec<T>::x(void) 
{ return myval1; }


template <class T> T
tridimvec<T>::y(void) 
{ return myval2; }


template <class T> T
tridimvec<T>::z(void) 
{ return myval3; }


// ------------------ paccottiglia generica ------------------
//template <class T> double
//interp_alpha(tridimvec<T> ip, tridimvec<T> ep)
//{ return (ep.y()-ip.y())/(ep.x()-ip.x()); }
//
//template <class T> double
//interp_beta(tridimvec<T> ip, tridimvec<T> ep)
//{ return (ip.y()-interp_alpha(ip,ep)*ip.x()); }

//template <class T> tridimvec<T>
//tridimvec<T>::orthonorm(void)
//{ return tridimvec<T>( (myval2==0) ? 0 : 1, (myval2 == 0) ? myval1 : -myval1/myval2 ); }
//{ return tridimvec<T>( (myval2==0) ? 0 : -.5/myval2, (myval1 == 0) ? 0 : .5/myval1 ); }


template <class T> tridimvec<T>
min(tridimvec<T> v1, tridimvec<T> v2)
{ return tridimvec<T>(std::min(v1.x(), v2.x()), std::min(v1.y(), v2.y()), std::min(v1.z(), v2.z())); }

template <class T> tridimvec<T>
max(tridimvec<T> v1, tridimvec<T> v2)
{ return tridimvec<T>(std::max(v1.x(), v2.x()), std::max(v1.y(), v2.y()), std::max(v1.z(), v2.z())); }

// -----------------------------------------------------------

#endif
