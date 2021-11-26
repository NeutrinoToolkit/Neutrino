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
#include <fstream>
#include <ostream>
#include <sstream>
#include <map>
#include <regex>

#include "bidimvec.h"
#include "tools.h"

#ifndef anymap_h_
#define anymap_h_

#define __pp_init_str "@@ phys_properties"
#define __pp_end_str "@@ end"

std::string trim(const std::string&, const std::string&);
std::ostream & operator<< (std::ostream &, struct anydata &);
std::istream & operator>> (std::istream &, struct anydata &);
bool check_vec(const std::string &);

// generic data holder
struct anydata{
public:
	anydata()
	{ ddescr = any_none; }

	// insert ops
	anydata & operator= (int rhs) 
	{ i=rhs; ddescr = any_int; return *this; }

	anydata & operator= (double rhs) 
	{ d=rhs; ddescr = any_double; return *this; }
	
	template<class T> anydata & operator= (bidimvec<T> rhs) 
	{ 
		std::stringstream ss; ss<<rhs; 
		str = ss.str(); ddescr = any_vec; return *this; 
	}
	
	anydata & operator= (std::string rhs) 
	{
		// can be a vector though, lets check
		if (check_vec(rhs)) {
			DEBUG(10, "Got vector in string operator: "<<rhs);
			str = rhs; ddescr = any_vec; return *this;
		} else str=rhs; ddescr = any_str; return *this;
	}

	// extr. ops
	operator int() const { 
		return get_i();
	}

	operator double() const { 
		return get_d();
	}

	operator std::string() const {
		if (ddescr == any_none) {
			DEBUG("accessing uninitialized value");
		}
		return get_str();
	}

	template<class T> operator bidimvec<T>() const {
		if (!is_vec()) WARNING("wrong datatype (vec) required for map member!!");
		DEBUG(11, "string is: "<<get_str());
		bidimvec<T> vv(get_str());
		return vv;
	}	


	enum anydata_type {any_int, any_double, any_str, any_vec, any_none} ddescr;

	inline bool is_d() const { return ddescr == any_double; }
	inline bool is_i() const { return ddescr == any_int; }
	inline bool is_str() const { return ddescr == any_str; }
	inline bool is_vec() const { return ddescr == any_vec; }
	inline bool is_none() const {return ddescr==any_none;}

	double get_d() const { if (ddescr == any_double) return d; else if (ddescr == any_int) return i; else return 0; }

	int get_i() const { 
		if (ddescr == any_int) 
			return i;
		else if (ddescr == any_double) {
		       	WARNING("double to int conversion");
			return (int)d;
		} else return 0;
	}

	template<class T> bidimvec<T> get_vec() const { return (ddescr == any_vec) ? bidimvec<T>(str) : bidimvec<T>("(0:0)"); }
	std::string get_str() const { 
		if (ddescr == any_str || ddescr == any_vec) return str; 
		else {
			std::stringstream ss;
			if (is_d()) ss<<d;
			else if (is_i()) ss<<i;

			return ss.str();
		}
	}

private:
	double d;
	int i;
	std::string str;
};







// final data structure
class anymap : public std::map<std::string, anydata> {

public:
	anymap()
	{ }

	//anydata operator() (std::string val) const {
		/*if (find(val) != end())
		{
			return find(val)->second;
		}*/
	//	return at(val);
	//}	
	
	bool have(std::string search_me) {
	    return (find(search_me) != end());
	}

    void loader(std::istream &);

    void dumper(std::ostream &);

};

#endif

// TODO
//  - deep operator, probably
//  - initializers
//  - a working std::string () operator
//  - check std::string vec2f constructor
//  - purtroppo gestire gli enum sara' abbastanza complicato. per il momento passo a conversione a std::string (poi si vedra')
//  - in teoria i metodi di get/set saltano
