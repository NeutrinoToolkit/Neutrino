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

#include "nGenericPan.h"
#include "ui_nWavelet.h"

#ifndef __nWavele
#define __nWavele
#include "nPhysWave.h"

class neutrino;
class nRect;
class plotZoomer;


std::list<nPhysD *> phys_wavelet_trasl_cuda(nPhysD *, void *, int &);
std::list<nPhysD *> phys_wavelet_trasl_nocuda(nPhysD *, void *, int &);

class nWavelet : public nGenericPan {
	Q_OBJECT

public:	
	nWavelet(neutrino *, QString);
	
	Ui::nWavelet my_w;

	QPointer<nRect> region;
	
	std::vector<nPhysD *> waveletPhys;
	nPhysD *origSubmatrix, *unwrapPhys, *referencePhys, *carrierPhys;

public slots:
	void guessCarrier();

	void doWavelet();
	void doUnwrap();
	void doRemove();
	void doRemoveCarrier();
	void doRemoveReference();
	
	void doAll();

	
};

#endif
