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
#include <limits>
#include <sstream>
#include <algorithm>

//using namespace std;

#ifndef __nanstream
#define __nanstream

// nan overloading for txt classes
class NaNStream 
{
public:
  NaNStream(std::ostream& _out, std::istream& _in):out(_out), in(_in){}
  template<typename T>
  const NaNStream& operator<<(const T& v) const {out << v;return *this;}
  template<typename T>
  const NaNStream& operator>>(T& v) const {std::cerr<<"sara' mica questa che chiami"<<std::endl; in >> v;return *this;}
protected:
  std::ostream& out;
  std::istream& in;
};

template<> const NaNStream& 
NaNStream::operator>>(double& ) const;


// override << operator for float type
//template <> const NaNStream& NaNStream::operator<<(const float& v) const 
//{
//  // test whether v is NaN 
//  if( v == v )
//    out << v;
//  else
//    out << "nan";
//  return *this;
//}

#endif
