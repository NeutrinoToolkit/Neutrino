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
#include <QSettings>

class nCustomRangeLineEdit : public QWidget {
    Q_OBJECT
public:
    nCustomRangeLineEdit(QCPAxis*);

private:
    QPointer<QLineEdit> my_min, my_max;
    QPointer<QCPAxis> my_axis;

public slots:
    void rangeChanged(const QCPRange& );
    void setRange(QString minmax_str);
    void setLock(bool check);
};

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
    QString get_data(int g=-1);

    void copy_data();
    void save_data();
    virtual void copy_image();
    virtual void export_image();

    void contextMenuEvent (QContextMenuEvent*) override;

    //SETTINGS
    void loadSettings();
    void saveSettings();
    void loadSettings(QSettings&);
    void saveSettings(QSettings&);

    void setLabel(QString);
    void showGrid(int val);
    void setLog(bool val);
    void setColor();
    void setTitle(QString);
    void setTitleFont(QFont);
    void changeTitleFont();
    void changeAxisFont();
    void showAxis(bool);
    inline QString getTitle() {if (title) {return title->text();} else {return QString();} };

    void changeAllFonts();

    void showGraph(bool);
    void changeGraphThickness(double);
    void rescaleAxes (bool onlyVisiblePlottables = false);

};

//plot with mouse (vertical)
class nCustomPlotMouseX : public nCustomPlot {
    Q_OBJECT

public:
    nCustomPlotMouseX(QWidget*);

private:
    QPointer<QCPItemStraightLine> mouseMarker;

public slots:
    void setMousePosition(double);
    void copy_image() override {
        if(mouseMarker) {
            mouseMarker->setVisible(false);
        }
        nCustomPlot::copy_image();
        mouseMarker->setVisible(false);
    }
    void export_image() override {
        if(mouseMarker) {
            mouseMarker->setVisible(false);
        }
        nCustomPlot::export_image();
        if(mouseMarker) {
            mouseMarker->setVisible(true);
        }
    }
};


//plot with mouse (horizontal and vertical)
class nCustomPlotMouseXY : public nCustomPlot {
    Q_OBJECT

public:
    nCustomPlotMouseXY(QWidget*);

private:
    QPointer<QCPItemStraightLine> mouseMarkerX;
    QPointer<QCPItemStraightLine> mouseMarkerY;

public slots:
    void setMousePosition(double,double);
    void copy_image() override {
        if(mouseMarkerX) {
            mouseMarkerX->setVisible(false);
        }
        if(mouseMarkerY) {
            mouseMarkerY->setVisible(false);
        }
        nCustomPlot::copy_image();
        if(mouseMarkerX) {
            mouseMarkerX->setVisible(true);
        }
        if(mouseMarkerY) {
            mouseMarkerY->setVisible(true);
        }
    }
    void export_image() override {
        if(mouseMarkerX) {
            mouseMarkerX->setVisible(false);
        }
        if(mouseMarkerY) {
            mouseMarkerY->setVisible(false);
        }
        nCustomPlot::export_image();
        if(mouseMarkerX) {
            mouseMarkerX->setVisible(true);
        }
        if(mouseMarkerY) {
            mouseMarkerY->setVisible(true);
        }
    }

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


//Q_DECLARE_METATYPE(nCustomPlot*);
//Q_DECLARE_METATYPE(nCustomPlotMouseX*);
//Q_DECLARE_METATYPE(nCustomPlotMouseXY*);
//Q_DECLARE_METATYPE(nCustomDoublePlot*);
//Q_DECLARE_METATYPE(nCustomPlotMouseX2Y*);
//Q_DECLARE_METATYPE(nCustomPlotMouseX3Y*);

#endif // nCustomPlots_H

