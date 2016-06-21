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

#include "nCustomPlots.h"
#include "neutrino.h"

nCustomPlot::nCustomPlot(QWidget* parent):
    QCustomPlot(parent)
{
    connect(this, SIGNAL(axisClick(QCPAxis*,QCPAxis::SelectablePart,QMouseEvent*)), this, SLOT(my_axisClick(QCPAxis*,QCPAxis::SelectablePart,QMouseEvent*)));
    setInteractions(QCP::iRangeDrag | QCP::iRangeZoom);

    QList<QCPAxis*> all_axis({xAxis,xAxis2,yAxis,yAxis2});

    foreach(QCPAxis* axis, all_axis) {
        axis->setLabelPadding(-1);
    }

    QSettings settings("neutrino","");
    settings.beginGroup("Preferences");
    QVariant fontString=settings.value("defaultFont");
    if (fontString.isValid()) {
        QFont fontTmp;
        if (fontTmp.fromString(fontString.toString())) {
            foreach(QCPAxis* axis, all_axis) {
                axis->setTickLabelFont(fontTmp);
                axis->setLabelFont(fontTmp);
            }
            legend->setFont(fontTmp);
        }
    }

    axisRect()->setRangeDrag(0);
    axisRect()->setRangeZoom(0);

}

void nCustomPlot::my_axisClick(QCPAxis*ax,QCPAxis::SelectablePart,QMouseEvent*) {
    DEBUG("Here");
    axisRect()->setRangeDragAxes(ax,ax);
    axisRect()->setRangeDrag(ax->orientation());
    axisRect()->setRangeZoomAxes(ax,ax);
    axisRect()->setRangeZoom(ax->orientation());
}


// plot as nCustomPlot but with x mouse line
nCustomPlotMouseX::nCustomPlotMouseX(QWidget* parent):
    nCustomPlot(parent),
    mouseMarker(this) {
}

void nCustomPlotMouseX::setMousePosition(double position) {
    mouseMarker.start->setCoords(position, QCPRange::minRange);
    mouseMarker.end->setCoords(position, QCPRange::maxRange);
    replot();
}

// plot as nCustomPlot but with x and y mouse lines
nCustomPlotMouseXY::nCustomPlotMouseXY(QWidget* parent):
    nCustomPlot(parent),
    mouseMarkerX(this),
    mouseMarkerY(this) {
}

void nCustomPlotMouseXY::setMousePosition(double positionX, double positionY) {
    mouseMarkerX.start->setCoords(positionX, QCPRange::minRange);
    mouseMarkerX.end->setCoords(positionX, QCPRange::maxRange);
    mouseMarkerY.start->setCoords(QCPRange::minRange,positionY);
    mouseMarkerY.end->setCoords(QCPRange::maxRange,positionY);
    replot();
}


// plot as nCustomPlot but with x and y mouse lines
nCustomDoublePlot::nCustomDoublePlot(QWidget* parent):
    nCustomPlotMouseXY(parent){

    addGraph(xAxis, yAxis2);
    graph(0)->setPen(QPen(Qt::red));
    addGraph(yAxis, xAxis2);
    graph(1)->setPen(QPen(Qt::blue));

    xAxis2->setVisible(true);
    yAxis2->setVisible(true);

    yAxis->setRangeReversed(true);

    xAxis->setLabel(tr("X"));
    xAxis->setLabelColor(Qt::red);
    xAxis->setTickLabelColor(Qt::red);
    yAxis2->setLabel(tr("X value"));
    yAxis2->setLabelColor(Qt::red);
    yAxis2->setTickLabelColor(Qt::red);
    yAxis->setLabel(tr("Y"));
    yAxis->setLabelColor(Qt::blue);
    yAxis->setTickLabelColor(Qt::blue);
    xAxis2->setLabel(tr("Y value"));
    xAxis2->setLabelColor(Qt::blue);
    xAxis2->setTickLabelColor(Qt::blue);

}

