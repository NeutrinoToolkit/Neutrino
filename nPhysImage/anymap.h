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

#include "bidimvec.h"
#include "tools.h"

#ifndef __anymap
#define __anymap

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
		if (!is_i()) WARNING("wrong datatype (int) required for map member!!");
		return get_i();
	}

	operator double() const { 
		if (!is_d()) WARNING("wrong datatype (double) required for map member!!");
		return get_d();
	}

	operator std::string() const {
		if (ddescr == any_none) {
			DEBUG("accessing uninitialized value");
		} else if (!is_str()) WARNING("wrong datatype (string) required for map member!! String is ");
		return get_str();
	}

	template<class T> operator bidimvec<T>() const {
		if (!is_vec()) WARNING("wrong datatype (vec) required for map member!!");
		DEBUG(11, "string is: "<<get_str());
		bidimvec<T> vv(get_str());
		return vv;
	}	


	enum anydata_type {any_int, any_double, any_str, any_vec, any_none} ddescr;

	bool is_d() const { return ddescr == any_double; }
	bool is_i() const { return ddescr == any_int; }
	bool is_str() const { return ddescr == any_str; }
	bool is_vec() const { return ddescr == any_vec; }

	double get_d() const { return (ddescr == any_double) ? d : 0; }
	int get_i() const { return (ddescr == any_int) ? i : 0; }
	template<class T> bidimvec<T> get_vec() const { return (ddescr == any_vec) ? bidimvec<T>(str) : bidimvec<T>("(0:0)"); }
	std::string get_str() const { return (ddescr == any_str || ddescr == any_vec) ? str : std::string("(empty)"); } 

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

	void loader(std::istream &is) {
		std::string st;
		clear();

		getline(is, st);
		while (st.find(__pp_init_str) == std::string::npos && !is.eof()) {
			DEBUG("get");
			getline(is, st);
		}

		getline(is, st);
		while (st.find(__pp_end_str) == std::string::npos
				&& !is.eof()) {

			size_t eqpos = st.find("=");
			if (eqpos == std::string::npos) {
			    DEBUG(st<<": malformed line");
				continue;
			}
			std::string st_key = trim(st.substr(0, eqpos), "\t ");
			std::string st_arg = trim(st.substr(eqpos+1, std::string::npos), "\t ");
			DEBUG(10, "key: "<<st_key);
			DEBUG(10, "arg: "<<st_arg);
        
			// filling
			std::stringstream ss(st_arg);
			ss>>(*this)[st_key];
			
			getline(is, st);
		}
		DEBUG("[anydata] read "<<size()<<" keys");
	}


	void dumper(std::ostream &os) {
	    DEBUG("[anydata] Starting dump of "<<size()<<" elements");
		
		os<<__pp_init_str<<std::endl;
		
		// keys iterator
		std::map<std::string, anydata>::iterator itr;
		for (itr=begin(); itr != end(); ++itr) {
			DEBUG(5,"[anydata] Dumping "<<itr->first);

			// check if key was inserted by non-existent access
			// (strange std::map behaviour...)
			if (itr->second.ddescr != anydata::any_none)
				os<<itr->first<<" = "<<itr->second<<std::endl;
		}
		os<<__pp_end_str<<std::endl;
		
		DEBUG("[anydata] Dumping ended");
	}

};

#endif

// TODO
//  - deep operator, probably
//  - initializers
//  - a working std::string () operator
//  - check std::string vec2f constructor
//  - purtroppo gestire gli enum sara' abbastanza complicato. per il momento passo a conversione a std::string (poi si vedra')
//  - in teoria i metodi di get/set saltano
