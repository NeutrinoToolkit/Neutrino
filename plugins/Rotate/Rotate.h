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
#include <QTimer>

#include "nGenericPan.h"
#include "ui_Rotate.h"

#ifndef Rotate_H_
#define Rotate_H_

class neutrino;

class Rotate : public nGenericPan, private Ui::Rotate {
    Q_OBJECT

public:	
    Q_INVOKABLE Rotate(neutrino *);

    nPhysD *rotated;

public slots:
    void doRotateLive();
    double getReplaceVal(nPhysD*);
    void angleChanged();
    void optionChanged();

private:
    QTimer updateTimer;             // live update, once the parameters stop changing
    // parameters of the last result: not computed again for the same ones
    nPhysD *lastSource=nullptr;
    double lastAngle=0;
    int lastFill=-1;
    bool lastSameSize=false;
};

NEUTRINO_PLUGIN(Rotate,Analysis);

#endif
