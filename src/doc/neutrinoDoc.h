/*
 *
 *    Copyright (C) 2013 Alessandro Flacco, Tommaso Vinci All Rights Reserved
 * 
 *    This file is part of neutrino.
 *
 *    Neutrino is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU Lesser General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    Neutrino is distributed in the hope that it will be useful,
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
#ifndef _NEUTRINODOC_H
#define _NEUTRINODOC_H

/*!
  \if USE_GLOBAL_DOXYGEN_DOC
  \page NeutrinoPage Netrino Overview
  \else
  \mainpage notitle
  \endif

 \b


 Neutrino is a powerful tool to analyze exprimental images (strarted as a collection of algorithms use for plasma physics).

 - \ref Features
 - \ref GUI
 - \ref VISAR

 \page Features Features

 \section Builtin Built-in Features

 The following are the built-in features of Neutrino:

 - Open files of various formats (.img .sif .b16 .jpg .png .pgm) (contact us if yours is still not supported).
 - Modern GUI to analyze data.
 - Bleeding edge of algorithms all rewritten in fast C/C++ code usin parallel threading and OpenCL.

 \page GUI GUI features
 - \ref Keyboard
 - \ref ColorBar
 - \ref Measures

 \section Keyboard Keyboard Shortcuts

 Here is a list of most used keyboard shortcuts (pay attention to upper/lower case):
 
 - + : increase zoom
 - - : decrease zoom
 - = : toggle (adapt zoom to window) and (set zoom to 100%)

 - m : toggles the mouse cursor from small to large cross
 - r : show/hide origin ruler
 - O : set origin to mouse cursor position 
 - h : open the horizontal lineout
 - v : open the vertical lineout
 - Option-Up : go to previous image
 - Option-Down : go to next image

 - C : open colorbar window
 - a : rescale image colors to limits (relative values)
 - A : rescale image colors to limits (absolute values)
 - Option-Left : go to previous colorbar
 - Option-Right : go to next colorbar

 - i : opens the Measure Window

 \section ColorBar Colorbar window

 Here is the colorbar win

 \section Measures Measures window

 Here is the measures win
 
 \page VISAR VISAR Analisys
 
 Here we should explain the visar stuff.
 

*/
#endif
