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
#include <QMenu>

class nCustomPlot : public QCustomPlot {
    Q_OBJECT

private:
    QPointer<QCPTextElement> title;

public:
    nCustomPlot(QWidget*);

public slots:
    void myAxisDoubleClick(QCPAxis*,QCPAxis::SelectablePart,QMouseEvent*e);
    void get_data(QTextStream &, QObject *obj=nullptr);
    void get_data_graph(QTextStream &out, QCPGraph *graph);

    void save_data();
    void copy_data();
    void export_image();

    void contextMenuEvent (QContextMenuEvent*) override;

    //SETTINGS
    void loadSettings(QSettings *);
    void saveSettings(QSettings *);

    void setLabel(QString);
    void showGrid(bool val);
    void setLog(bool val);
    void setColor();
    void setTitle(QString);
    void changeTitleFont();
    void changeAxisFont();

};

//plot with mouse (vertical)
class nCustomPlotMouseX : public nCustomPlot {
    Q_OBJECT

public:
    nCustomPlotMouseX(QWidget*);

private:
    QPointer<QCPItemLine> mouseMarker;

public slots:
    void setMousePosition(double);

};


//plot with mouse (horizontal and vertical)
class nCustomPlotMouseXY : public nCustomPlot {
    Q_OBJECT

public:
    nCustomPlotMouseXY(QWidget*);

private:
    QPointer<QCPItemLine> mouseMarkerX;
    QPointer<QCPItemLine> mouseMarkerY;

public slots:
    void setMousePosition(double,double);

};


//plot with mouse (horizontal and vertical) and two curves (x,y) and (x2,y2)
class nCustomDoublePlot : public nCustomPlotMouseXY {
    Q_OBJECT

public:
    nCustomDoublePlot(QWidget*);

};

class nCustomPlotMouseX2Y : public nCustomPlotMouseX {
    Q_OBJECT
public:
    nCustomPlotMouseX2Y(QWidget*);
};


class nCustomPlotMouseX3Y : public nCustomPlotMouseX2Y {
    Q_OBJECT
public:
    nCustomPlotMouseX3Y(QWidget*);
    QPointer<QCPAxis> yAxis3;
};


#endif // nCustomPlots_H

