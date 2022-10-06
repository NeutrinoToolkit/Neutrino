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
#include "ui_Wavelet.h"

#ifndef __Wavelet
#define __Wavelet
#include "nPhysWave.h"
#include "nRect.h"
#include "nLine.h"

class neutrino;

class Wavelet : public nGenericPan, private Ui::Wavelet {
    Q_OBJECT

public:	
    Q_INVOKABLE Wavelet(neutrino *);

    nRect region;

    nLine linebarrier;

    std::map<std::string, nPhysD *> waveletPhys;
    nPhysD *origSubmatrix, *unwrapPhys, *referencePhys, *carrierPhys, *syntheticPhys;

private:
    physWave::wavelet_params my_params;

public slots:

    void useBarrierToggled(bool);
    void guessCarrier();

    void doWavelet();
    void doUnwrap();
    void doRemove();
    void doRemoveCarrier();
    void doRemoveReference();

    void bufferChanged(nPhysD*);
    void physDel(nPhysD*);

    void on_relative_toggled(bool);

    void doAll();


};

NEUTRINO_PLUGIN(Wavelet,Analysis);


#endif
