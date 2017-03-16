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
#include "nanstream.h"

// override >> operator for float type
template<>
const NaNStream& NaNStream::operator>>(double& v) const 
{
//	std::cerr<<"ci entri qui?"<<std::endl;
  if (in >> v) {
    //  std::cerr<<"nan got num: "<<v<<std::endl;
      return *this;
  }


  in.clear();
  std::string str;
  if (!(in >> str)) {
  //  std::cerr<<"nan CAN'T get string!"<<std::endl;
    return *this;
  }

  //std::transform(str.begin(), str.end(), str.begin(), ::tolower);

  if (str == "nan") {
    v = std::numeric_limits<double>::quiet_NaN();
 //   std::cerr<<"nan got nan: "<<v<<std::endl;
  }  else {
 //   std::cerr<<"nan got error: "<<v<<std::endl;
    in.setstate(std::ios::badbit); // Whoops, we've still "stolen" the string
  }

  return *this;
}

