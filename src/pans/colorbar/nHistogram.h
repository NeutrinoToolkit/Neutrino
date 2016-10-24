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
#include <iostream>

#include <QtGui>
#include <QWidget>
#include <QtSvg>

#ifndef __nhistogram_h
#define __nhistogram_h

#include "nPhysImageF.h"

class nColorBarWin;

class nHistogram : public QWidget
{
	Q_OBJECT
	
public:
	
	nHistogram (QWidget *parent=0);
	
	int offsety, offsetx, dyColorBar;
	double colorvalue;
	nColorBarWin *parentPan;
	
public slots:
	void paintEvent(QPaintEvent *);
    void mousePressEvent (QMouseEvent *);
    void mouseReleaseEvent (QMouseEvent *);
    void mouseMoveEvent (QMouseEvent *);

    void drawPicture (QPainter&);
	void colorValue(double);

};

#endif
