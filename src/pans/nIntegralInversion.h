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
#include <QtGui>
#include <QWidget>

#include <sstream>
#include <vector>

#include "nGenericPan.h"
#include "ui_nIntegralInversion.h"

#ifndef __nII
#define __nII
#include "nPhysWave.h"

class neutrino;
class nLine;


void phys_invert_abel_transl(void *params, int&);



class nIntegralInversion : public nGenericPan {
	Q_OBJECT

public:
	nIntegralInversion(neutrino *, QString);
	
	Ui::nIntegralInversion my_w;

	QPointer<nLine> axis;
	nPhysD *invertedPhys;
private:
	abel_params my_abel_params;

public slots:
    void physDel(nPhysD*);
	void sceneChanged();
	void refphase_checkbChanged(int);
	QVariant doInversion();
	
};

#endif
