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
#ifndef __nPreferences_h
#define __nPreferences_h

#include <iostream>

#include <QtGui>
#include <QWidget>

#include "ui_nPreferences.h"
#include "nGenericPan.h"
#include "nPhysWave.h"

class nPreferences : public nGenericPan
{
	Q_OBJECT
	
public:	
	nPreferences(neutrino*,QString);
	Ui::nPreferences my_w;

public slots:
	void changeShowDimPixel();
	void changeIconSize(int);
    static void changeThreads(int);
	void changeFont();
	void hideEvent(QHideEvent*);
	void showEvent(QShowEvent*);
	void useDot();
    void askCloseUnsaved();
    void changephysNameLength(int);
};

#endif
