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
#include "ui_Interferometry.h"
#include "ui_Interferometry1.h"

#ifndef __Interferometry
#define __Interferometry
#include "nPhysWave.h"
#include "nLine.h"
#include "nRect.h"

class neutrino;

class Interferometry : public nGenericPan, private Ui::Interferometry {
    Q_OBJECT

public:	
    Q_INVOKABLE Interferometry(neutrino *);

    std::array<Ui::Interferometry1, 2> my_image;

    QPointer<nRect> region;
    QPointer<nLine> unwrapBarrier;
    QPointer<nLine> maskRegion;

    std::map<std::string, nPhysD *> localPhys;

    void loadSettings(QSettings&);
    void saveSettings(QSettings&);

private:

    std::map<QToolButton*, nLine *> my_shapes;

public slots:

    void on_actionDuplicate_triggered();
    void on_actionDelete_triggered();

    void physDel(nPhysD*);
    void useBarrierToggled(bool);
    void maskRegionToggled(bool);
    void interpolateToggled(bool);
    
    void guessCarrier();

    void doWavelet();
    void doWavelet(int);
    void doUnwrap();
    void doSubtract();
    void doMaskCutoff();
    void getPosZero(bool);
    void setPosZero(QPointF);

    void line_key_pressed(int);
    
    void addShape();
    void addShape(QString);
    void removeShape(QObject*);
    void doShape();

    void doCutoff();
    
    void bufferChanged(nPhysD*);

    void imagesTabBarClicked(int);
};

NEUTRINO_PLUGIN(Interferometry,Analysis);

#endif
