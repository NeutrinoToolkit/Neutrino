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
#ifndef nCustomPlots_H
#define nCustomPlots_H

#include "qcustomplot.h"


class nCustomPlot : public QCustomPlot {
    Q_OBJECT

public:
    nCustomPlot(QWidget*);

public slots:
    void my_axisClick(QCPAxis*,QCPAxis::SelectablePart,QMouseEvent*);

};

//plot with mouse (vertical)
class nCustomPlotMouseX : public nCustomPlot {
    Q_OBJECT

public:
    nCustomPlotMouseX(QWidget*);

private:
    QCPItemLine mouseMarker;

public slots:
    void setMousePosition(double);

};


//plot with mouse (horizontal and vertical)
class nCustomPlotMouseXY : public nCustomPlot {
    Q_OBJECT

public:
    nCustomPlotMouseXY(QWidget*);

private:
    QCPItemLine mouseMarkerX;
    QCPItemLine mouseMarkerY;

public slots:
    void setMousePosition(double,double);

};


//plot with mouse (horizontal and vertical) and two curves (x,y) and (x2,y2)
class nCustomDoublePlot : public nCustomPlotMouseXY {
    Q_OBJECT

public:
    nCustomDoublePlot(QWidget*);

};


#endif // nCustomPlots_H

