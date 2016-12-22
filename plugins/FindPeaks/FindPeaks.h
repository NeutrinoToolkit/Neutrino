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

#ifndef __FindPeaks
#define __FindPeaks

#include "nGenericPan.h"
#include "ui_FindPeaks.h"
#include "neutrino.h"

class nRect;

class FindPeaks : public nGenericPan, private Ui::FindPeaks {
    Q_OBJECT

public:

    Q_INVOKABLE FindPeaks(neutrino *);

public slots:

    void mouseAtMatrix(QPointF);
    void updatePlot();

    void set_origin();
    void set_scale();
    void on_actionClipboard_triggered();
    void on_actionTxt_triggered();

    QString getPoints();

private:
    QPointer<nRect> box;
};


NEUTRINO_PLUGIN(FindPeaks,Analysis);

#endif
