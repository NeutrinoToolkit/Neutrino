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
    setContentsMargins(0,0,0,0);
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

    connect(this, SIGNAL(axisDoubleClick(QCPAxis*,QCPAxis::SelectablePart,QMouseEvent*)), this, SLOT(my_axisDoubleClick(QCPAxis*,QCPAxis::SelectablePart,QMouseEvent*)));
    connect(this, SIGNAL(axisClick(QCPAxis*,QCPAxis::SelectablePart,QMouseEvent*)), this, SLOT(my_axisDoubleClick(QCPAxis*,QCPAxis::SelectablePart,QMouseEvent*)));
    connect(this, SIGNAL(plottableDoubleClick(QCPAbstractPlottable*,int,QMouseEvent*)), this, SLOT(my_plottableDoubleClick(QCPAbstractPlottable*,int,QMouseEvent*)));

    setInteractions(QCP::iRangeDrag | QCP::iRangeZoom);

    axisRect()->setRangeDrag(0);
    axisRect()->setRangeZoom(0);
    axisRect()->setMargins(QMargins(0,0,0,0));
    setCursor(QCursor(Qt::CrossCursor));
    repaint();
}

void nCustomPlot::keyPressEvent(QKeyEvent *e) {
    QCustomPlot::keyPressEvent(e);
    if (e->modifiers() & Qt::ControlModifier) {
        switch (e->key()) {
        case Qt::Key_C: copy_data(); break;
        case Qt::Key_S: save_data(); break;
        case Qt::Key_E: export_image(); break;
        }
    }
}

void nCustomPlot::contextMenuEvent (QContextMenuEvent *ev) {
    my_menu->exec(ev->globalPos());
}

void nCustomPlot::get_data(QTextStream &out, QCPGraph *graph) {
    out << "# " << graph->name() << endl;
    for (QCPGraphDataContainer::const_iterator it=graph->data()->begin(); it!=graph->data()->end(); ++it) {
        out << it->key << " " << it->value << endl;
    }
}

void nCustomPlot::get_data(QTextStream &out) {
    out << "# " <<  property("panName").toString() << " (" << graphCount() << " graphs)" << endl;
    for (int g=0; g<graphCount(); g++) {
        out << "# " << g <<" ";
        get_data(out,graph(g));
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
        if (QFileInfo(fnametmp).suffix().toLower()==QString("pdf")) {
            savePdf(fnametmp, 0, 0, QCP::epAllowCosmetic, "Neutrino", property("panName").toString());
        } else if (QFileInfo(fnametmp).suffix().toLower()==QString("svg")) {
            QSvgGenerator printer;
            printer.setFileName(fnametmp);
            QCPPainter qcpPainter;
            qcpPainter.begin(&printer);
            toPainter(&qcpPainter, 0,0);
            qcpPainter.end();
        }
    }
}

void nCustomPlot::my_axisDoubleClick(QCPAxis*ax,QCPAxis::SelectablePart,QMouseEvent*) {
    DEBUG("here " << (void*)ax);
    axisRect()->setRangeDragAxes(ax,ax);
    axisRect()->setRangeDrag(ax->orientation());
    axisRect()->setRangeZoomAxes(ax,ax);
    axisRect()->setRangeZoom(ax->orientation());
}

void nCustomPlot::my_plottableDoubleClick(QCPAbstractPlottable* plottable, int dataIndex, QMouseEvent *e) {
    QCPGraph *graph = qobject_cast<QCPGraph *>(plottable);
    if(graph) {
        QString t;
        QTextStream out(&t);
        get_data(out, graph);
        QApplication::clipboard()->setText(t);
    }
    axisRect()->setRangeDragAxes(nullptr,nullptr);
    axisRect()->setRangeZoomAxes(nullptr,nullptr);
}


// SETTINGS

void
nCustomPlot::loadSettings(QSettings *settings) {
    DEBUG(">>>>>>>>>>>>>>>>>>"<<objectName().toStdString());
    DEBUG(">>>>>>>>>>>>>>>>>>"<<toolTip().toStdString());
    settings->beginGroup(objectName());
    QStringList labels = settings->value("labels").toStringList();
    QList<QCPAxis *> axis=findChildren<QCPAxis *>();
    QMutableListIterator<QCPAxis *> iter(axis);
    while (iter.hasNext()) {
        if (!iter.next()->visible()) {
            iter.remove();
        }
    }
    if (labels.size() == axis.size()) {
        DEBUG(">>>>>>>>>>"<<axis.size());
        for (int i=0; i< labels.size(); i++) {
            if(axis[i]->visible()) {
                axis[i]->setLabel(labels[i]);
            }
        }
    }
    settings->endGroup();
}

void
nCustomPlot::saveSettings(QSettings *settings) {
    DEBUG(">>>>>>>>>>>>>>>>>>"<<objectName().toStdString());
    DEBUG(">>>>>>>>>>>>>>>>>>"<<toolTip().toStdString());

    settings->beginGroup(objectName());
    QStringList labels;
    foreach (QCPAxis *axis, findChildren<QCPAxis *>()) {
        if(axis->visible()) labels << axis->label();
    }
    settings->setValue("labels",labels);
    settings->endGroup();
}


// plot as nCustomPlot but with x mouse line
nCustomPlotMouseX::nCustomPlotMouseX(QWidget* parent):
    nCustomPlot(parent),
    mouseMarker(new QCPItemLine(this)) {
    setMousePosition(0);
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
    setMousePosition(0,0);
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


nCustomPlotMouseX2Y::nCustomPlotMouseX2Y(QWidget* parent):
    nCustomPlotMouseX(parent)
{
    yAxis2->setVisible(true);

    yAxis->setLabelColor(Qt::red);
    yAxis->setTickLabelColor(Qt::red);

    yAxis2->setLabelColor(Qt::blue);
    yAxis2->setTickLabelColor(Qt::blue);

    show();
};

nCustomPlotMouseX3Y::nCustomPlotMouseX3Y(QWidget* parent):
    nCustomPlotMouseX2Y(parent),
    yAxis3(axisRect(0)->addAxis(QCPAxis::atRight,0))
{
    yAxis3->setLabelColor(Qt::darkCyan);
    yAxis3->setTickLabelColor(Qt::darkCyan);

    show();
};


