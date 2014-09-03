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
#include "ui_nInterferometry.h"
#include "ui_nInterferometry1.h"

#ifndef __nInterferometry
#define __nInterferometry
#include "nPhysWave.h"
#include "nLine.h"
#include "nRect.h"

class neutrino;

class nInterferometry : public nGenericPan {
	Q_OBJECT

public:	
	nInterferometry(neutrino *, QString);
	
	Ui::nInterferometry my_w;
	Ui::nInterferometry1 my_image[2];

	QPointer<nRect> region;	
	QPointer<nLine> linebarrier;
    QPointer<nLine> lineRegion;
	
	std::map<std::string, nPhysD *> waveletPhys[2];

    std::map<std::string, nPhysD *> localPhys;

private:
	wavelet_params my_params;
    std::vector<std::string> localPhysNames();

public slots:
		
    void physDel(nPhysD*);
	void useBarrierToggled(bool);
	void guessCarrier();

	void doWavelet();
	void doUnwrap();
	void doSubtract();
	void doMask();
	void doAbel();
	void getPosZero(bool);
	void setPosZero(QPointF);
	void getPosAbel(bool);
	void setPosAbel(QPointF);

	void bufferChanged(nPhysD*);
	void checkChangeCombo(QComboBox *);

	
};

#endif
