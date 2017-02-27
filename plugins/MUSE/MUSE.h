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
#include "ui_MUSE.h"

#ifndef __MUSE
#define __MUSE

#include "neutrino.h"

class MUSE : public nGenericPan, private Ui::MUSE {
    Q_OBJECT

public:
    Q_INVOKABLE MUSE(neutrino *);

public slots:

    void loadCube();

    void doSpectrum(QPointF p);

    void on_actionMode_toggled();
//    void on_actionFFT_triggered();
    void on_actionMovie_triggered();
    void on_actionMean_triggered();
    void on_actionExportTxt_triggered();
    void on_percentMin_valueChanged(int);
    void on_percentMax_valueChanged(int);

    void showImagePlane(int z);

    void updateLastPoint();

    void plotClick(QMouseEvent*);

    void horzScrollBarChanged(int value);
    void xAxisChanged(QCPRange range);

    void keyPressEvent (QKeyEvent *e);

    void nextPlane();


private:
    QPointF my_offset;
    QPointF my_offset_val;
    QPointF my_scale;

    QVector<double> xvals;
    QVector<double> yvals;
    QVector<double> ymean;

    std::vector<double> cubevect;
    std::vector<unsigned int> cubesize;

    nPhysD *cubeSlice;
    nPhysD *meanSlice;

    phys_properties cube_prop;
    vec2f wavelen;
    QPoint lastpoint;

    QVariant extractData(QString key, QStringList values);

    QString to_min_sec(double val);

    QTimer my_timer;

};

NEUTRINO_PLUGIN(MUSE,Analysis);

#endif
