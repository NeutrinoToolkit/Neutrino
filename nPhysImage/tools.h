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
/*!
 \page macros Code macros
 
 The c++ macros used in the code should be paced in the \ref Tools.h file.
 
 \section caveats Warning, Error and Debug
 All these macros will print to the standard error a tag , the name, line of the source file that caused the call 
 as well as the name of the function calling the macro. The arguments contained in the parenthesis will then be 
 appended. Arguments can be chained together in c++ stream style (using <tt><<</tt> operator)
 
 The macro <tt>WARNING("text")</tt> is the most basic and is itended for waringns that should always be present in the code. 
 
 The macro <tt>ERROR("text")</tt> is used to print an error and close the program.
 
 The macro <tt>DEBUG("text")</tt> can be used in two ways: using just an argument, it will display a debug message 
 (similar to <tt>WARNING("text")</tt> ) but it can be used in the form <tt>DEBUG(N,"text")</tt> in this case N is a number and 
 represents the debug level starting at which the dubug must be displayed. 
 The debug level can be changed int the namelist vie the key <tt>debug</tt>.
 
 
 */
 
#ifndef __AVAILABILITY_INTERNAL__MAC_10_13
#define __AVAILABILITY_INTERNAL__MAC_10_13 __AVAILABILITY_INTERNAL_WEAK_IMPORT
#endif

#ifndef Tools_h_
#define Tools_h_

#include <csignal>
#include <iostream>

#define PRINTVAR(x) #x"=" << x
#define PHYS__MSG  __FILE__ << ":" << __LINE__ << " (" << __FUNCTION__ << ") "
#define PHYS__HEADER(PHYS__MSG,PHYS__txt) std::cerr << "[" << PHYS__MSG << "] " << PHYS__MSG << PHYS__txt << std::endl

#define WARNING(__txt) PHYS__HEADER("WARNING", __txt)
#define ERROREXIT(__txt) {PHYS__HEADER("ERROR", __txt); std::raise(SIGSEGV);}


#ifdef  __phys_debug
#define DEBUG1(__txt) PHYS__HEADER("DEBUG", __txt)
#define DEBUG2(__val,__txt) if(__val<=__phys_debug) PHYS__HEADER("DEBUG "<<__val, __txt)
#define DEBUG3(arg1,arg2,arg3,...) arg3
#define DEBUG4(...) DEBUG3(__VA_ARGS__,DEBUG2,DEBUG1,)
#define DEBUG(...) DEBUG4(__VA_ARGS__)(__VA_ARGS__)
#else
#define DEBUG(...)
#define DEBUGVAR(...)
#endif

#endif
