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
    QCustomPlot(parent),
    my_menu(new QMenu(this))
{

    QFont f = my_menu->font();
    f.setPointSize(10);
    my_menu->setFont(f);

    setProperty("fileTxt", "data.txt");
    setProperty("fileExport", "data.pdf");

//    my_menu=new QMenu(this);
    QAction *my_act;

    my_act = new QAction("Copy data",my_menu);
    connect(my_act, SIGNAL(triggered()), this, SLOT(copy_data()));
    my_menu->addAction(my_act);

    my_act = new QAction("Save data",my_menu);
    connect(my_act, SIGNAL(triggered()), this, SLOT(save_data()));
    my_menu->addAction(my_act);

    my_act = new QAction("Export image",my_menu);
    connect(my_act, SIGNAL(triggered()), this, SLOT(export_image()));
    my_menu->addAction(my_act);

    setProperty("panName", "orphan graph");

    QObject *this_widget = qobject_cast<QObject*>(this);
    do{
        nGenericPan* this_pan = qobject_cast<nGenericPan*>(this_widget);
        if (this_pan) {
            setProperty("panName", this_pan->panName);
            break;
        }
    } while ((this_widget=this_widget->parent())!=nullptr);

    connect(this, SIGNAL(axisClick(QCPAxis*,QCPAxis::SelectablePart,QMouseEvent*)), this, SLOT(my_axisClick(QCPAxis*,QCPAxis::SelectablePart,QMouseEvent*)));
    setInteractions(QCP::iRangeDrag | QCP::iRangeZoom);

    QList<QCPAxis*> all_axis({xAxis,xAxis2,yAxis,yAxis2});

    foreach(QCPAxis* axis, all_axis) {
        axis->setLabelPadding(-1);
    }

    QSettings settings("neutrino","");
    settings.beginGroup("Preferences");
    QVariant fontString=settings.value("defaultFont");
    settings.endGroup();
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

void nCustomPlot::contextMenuEvent (QContextMenuEvent *ev) {
    my_menu->exec(ev->globalPos());
}


void nCustomPlot::get_data(QTextStream &out) {
    out << "# " <<  property("panName").toString() << " " << graphCount() << endl;
    for (int g=0; g<graphCount(); g++) {
        out << "# " << g << " name: " << graph(g)->name() << endl;
        for (QCPGraphDataContainer::const_iterator it=graph(g)->data()->begin(); it!=graph(g)->data()->end(); ++it) {
            out << it->key << " " << it->value << endl;
        }
        out << endl << endl;
    }
}

void nCustomPlot::save_data(){
    QString fnametmp=QFileDialog::getSaveFileName(this,tr("Save data in text"),property("fileTxt").toString(),tr("Text files (*.txt *.csv);;Any files (*)"));
    if (!fnametmp.isEmpty()) {
        setProperty("fileTxt", fnametmp);
        QFile t(fnametmp);
        t.open(QIODevice::WriteOnly| QIODevice::Text);
        QTextStream out(&t);
        get_data(out);
        t.close();
    }
}

void nCustomPlot::copy_data(){
    QString t;
    QTextStream out(&t);
    get_data(out);
    QApplication::clipboard()->setText(t);
}

void nCustomPlot::export_image(){
    QString fnametmp = QFileDialog::getSaveFileName(this,tr("Save Drawing"),property("fileExport").toString(),"Vector files (*.pdf,*.svg)");
    if (!fnametmp.isEmpty()) {
        setProperty("fileExport", fnametmp);
        savePdf(fnametmp, 0, 0, QCP::epAllowCosmetic, "Neutrino", property("panName").toString());
    }
}

void nCustomPlot::my_axisClick(QCPAxis*ax,QCPAxis::SelectablePart,QMouseEvent*) {
    axisRect()->setRangeDragAxes(ax,ax);
    axisRect()->setRangeDrag(ax->orientation());
    axisRect()->setRangeZoomAxes(ax,ax);
    axisRect()->setRangeZoom(ax->orientation());
}


// plot as nCustomPlot but with x mouse line
nCustomPlotMouseX::nCustomPlotMouseX(QWidget* parent):
    nCustomPlot(parent),
    mouseMarker(new QCPItemLine(this)) {
}

void nCustomPlotMouseX::setMousePosition(double position) {
    if (mouseMarker) {
        mouseMarker->start->setCoords(position, QCPRange::minRange);
        mouseMarker->end->setCoords(position, QCPRange::maxRange);
    }
    replot();
}

// plot as nCustomPlot but with x and y mouse lines
nCustomPlotMouseXY::nCustomPlotMouseXY(QWidget* parent):
    nCustomPlot(parent),
    mouseMarkerX(new QCPItemLine(this)),
    mouseMarkerY(new QCPItemLine(this)) {

}

void nCustomPlotMouseXY::setMousePosition(double positionX, double positionY) {
    if (mouseMarkerX && mouseMarkerY) {
        mouseMarkerX->start->setCoords(positionX, QCPRange::minRange);
        mouseMarkerX->end->setCoords(positionX, QCPRange::maxRange);
        mouseMarkerY->start->setCoords(QCPRange::minRange,positionY);
        mouseMarkerY->end->setCoords(QCPRange::maxRange,positionY);
    }
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

