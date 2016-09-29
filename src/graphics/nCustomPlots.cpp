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
#include "ui_nCustomPlot.h"

nCustomPlot::nCustomPlot(QWidget* parent):
    QCustomPlot(parent)
{
    setContentsMargins(0,0,0,0);

    connect(this, SIGNAL(axisDoubleClick(QCPAxis*,QCPAxis::SelectablePart,QMouseEvent*)), this, SLOT(myAxisDoubleClick(QCPAxis*,QCPAxis::SelectablePart,QMouseEvent*)));
    connect(this, SIGNAL(plottableDoubleClick(QCPAbstractPlottable*,int,QMouseEvent*)), this, SLOT(myPlottableDoubleClick(QCPAbstractPlottable*,int,QMouseEvent*)));

    setInteractions(QCP::iRangeDrag | QCP::iRangeZoom);

    axisRect()->setRangeDrag(0);
    axisRect()->setRangeZoom(0);
    axisRect()->setMargins(QMargins(0,0,0,0));
    setCursor(QCursor(Qt::CrossCursor));
    repaint();
}

void nCustomPlot::contextMenuEvent (QContextMenuEvent *ev) {
    QMainWindow *my_pad=nullptr;
    foreach (my_pad, findChildren<QMainWindow *>()) {
        if(my_pad->windowTitle()=="Plot Preferences") break;
    }
    if (!my_pad) {
        my_pad=new QMainWindow(this);
        Ui::nCustomPlot my_w;
        my_w.setupUi(my_pad);
        connect(my_w.actionCopy, SIGNAL(triggered()), this, SLOT(copy_data()));
        connect(my_w.actionSave, SIGNAL(triggered()), this, SLOT(save_data()));
        connect(my_w.actionExport, SIGNAL(triggered()), this, SLOT(export_image()));
        foreach (QCPAxis *axis, findChildren<QCPAxis *>()) {
            if (axis->visible()) {
                int row=my_w.labels_layout->rowCount();
                QLabel *la = new QLabel(this);
                QFont f = la->font();
                f.setPointSize(10);
                la->setFont(f);
                QString ax_pos;
                switch (axis->axisType()) {
                case QCPAxis::atLeft : ax_pos="Left"; break;
                case QCPAxis::atRight : ax_pos="Right"; break;
                case QCPAxis::atTop : ax_pos="Top"; break;
                case QCPAxis::atBottom : ax_pos="Bottom"; break;
                }

                la->setText(ax_pos);
                my_w.labels_layout->addWidget(la,row,0);

                QLineEdit *le = new QLineEdit(this);
                le->setFont(f);
                le->setText(axis->label());
                le->setProperty("axis",qVariantFromValue((void *) axis));
                connect(le,SIGNAL(textChanged(QString)),this,SLOT(setLabel(QString)));
                my_w.labels_layout->addWidget(le,row,1);

                QCheckBox *cb_grid = new QCheckBox("Grid", this);
                QCPGrid *grid = axis->grid();
                cb_grid->setChecked(grid->visible());
                cb_grid->setFont(f);
                cb_grid->setProperty("grid",qVariantFromValue((void *) grid));
                connect(cb_grid,SIGNAL(toggled(bool)),this,SLOT(showGrid(bool)));
                my_w.labels_layout->addWidget(cb_grid,row,2);

                QCheckBox *cb_log = new QCheckBox("Log", this);
                cb_log->setChecked(axis->scaleType()==QCPAxis::stLogarithmic);
                cb_log->setFont(f);
                cb_log->setProperty("axis",qVariantFromValue((void *) axis));
                connect(cb_log,SIGNAL(toggled(bool)),this,SLOT(setLog(bool)));
                my_w.labels_layout->addWidget(cb_log,row,3);

//                QLineEdit *minmax = new QLineEdit(this);
//                minmax->setFont(f);
//                QString minmaxstr;

//                minmax->setText(minmaxstr);
//                minmax->setProperty("axis",qVariantFromValue((void *) axis));
//                my_w.labels_layout->addWidget(minmax,row,2);

            }
        }
        for (int g=0; g<plottableCount(); g++) {
            QCPGraph *graph = qobject_cast<QCPGraph *>(plottable(g));
            if(graph) {
                my_w.plotName->addItem(graph->name(), qVariantFromValue((void *) graph));
            }
            my_w.copyData->setProperty("combo",qVariantFromValue((void *) my_w.plotName));
            my_w.saveData->setProperty("combo",qVariantFromValue((void *) my_w.plotName));
        }
        connect(my_w.copyData,SIGNAL(released()),this,SLOT(copy_data()));
        connect(my_w.saveData,SIGNAL(released()),this,SLOT(save_data()));
    }
    if (my_pad) my_pad->show();
}

void nCustomPlot::get_data_graph(QTextStream &out, QCPGraph *graph) {
    out << "# " << graph->name() << endl;
    for (QCPGraphDataContainer::const_iterator it=graph->data()->begin(); it!=graph->data()->end(); ++it) {
        out << it->key << " " << it->value << endl;
    }
}

void nCustomPlot::get_data(QTextStream &out, QObject *obj) {
    if (obj) {
        QCPGraph *graph = qobject_cast<QCPGraph *>(obj);
        if (graph) {
            get_data_graph(out,graph);
        } else {
            if (obj->property("combo").isValid()) {
                QComboBox *combo = (QComboBox *) obj->property("combo").value<void *>();
                if (combo) {
                    QCPGraph *graph = (QCPGraph *) combo->currentData().value<void *>();
                    if (graph) {
                        get_data_graph(out, graph);
                    } else {
                        get_data(out);
                    }
                }
            } else {
                get_data(out);
            }
        }
    } else {
        out << "# " <<  property("panName").toString() << " (" << graphCount() << " graphs)" << endl;
        for (int g=0; g<graphCount(); g++) {
            out << "# " << g <<" ";
            get_data_graph(out,graph(g));
            out << endl << endl;
        }
    }
}

void nCustomPlot::save_data(){
    DEBUG("here");
    QString fnametmp=QFileDialog::getSaveFileName(this,tr("Save data in text"),property("fileTxt").toString(),tr("Text files (*.txt *.csv);;Any files (*)"));
    if (!fnametmp.isEmpty()) {
        setProperty("fileTxt", fnametmp);
        QFile t(fnametmp);
        t.open(QIODevice::WriteOnly| QIODevice::Text);
        QTextStream out(&t);
        get_data(out, sender());
        t.close();
    }
}

void nCustomPlot::copy_data(){
    DEBUG("here");
    QString t;
    QTextStream out(&t);
    get_data(out,sender());
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

void nCustomPlot::showGrid(bool val) {
    if (sender() && sender()->property("grid").isValid()) {
        QCPGrid *grid = (QCPGrid *) sender()->property("grid").value<void *>();
        if (grid) {
            grid->setVisible(val);
            replot();
        }
    }
}

void nCustomPlot::setLog(bool val) {
    if (sender() && sender()->property("axis").isValid()) {
        QCPAxis *axis = (QCPAxis *) sender()->property("axis").value<void *>();
        if (axis) {
            axis->setScaleType(val?QCPAxis::stLogarithmic:QCPAxis::stLinear);
            replot();
        }
    }
}

void nCustomPlot::setLabel(QString name) {
    if (sender() && sender()->property("axis").isValid()) {
        QCPAxis *axis = (QCPAxis *) sender()->property("axis").value<void *>();
        if (axis) {
            axis->setLabel(name);
            replot();
        }
    }
}

void nCustomPlot::myAxisDoubleClick(QCPAxis*ax,QCPAxis::SelectablePart,QMouseEvent*) {
    DEBUG("here " << (void*)ax);
    axisRect()->setRangeDragAxes(ax,ax);
    axisRect()->setRangeDrag(ax->orientation());
    axisRect()->setRangeZoomAxes(ax,ax);
    axisRect()->setRangeZoom(ax->orientation());
}

// SETTINGS
void
nCustomPlot::loadSettings(QSettings *settings) {
    settings->beginGroup(objectName());
    QStringList labels = settings->value("labels").toStringList();
    QList<QVariant> grids = settings->value("grids").toList();
    QList<QVariant> logs = settings->value("logs").toList();
    QList<QCPAxis *> axis=findChildren<QCPAxis *>();
    QMutableListIterator<QCPAxis *> iter(axis);
    while (iter.hasNext()) {
        if (!iter.next()->visible()) {
            iter.remove();
        }
    }
    if (labels.size() == axis.size() && grids.size() == axis.size() && logs.size() == axis.size() ) {
        DEBUG(">>>>>>>>>>"<<axis.size());
        for (int i=0; i< labels.size(); i++) {
            if(axis[i]->visible()) {
                axis[i]->setLabel(labels[i]);
                axis[i]->grid()->setVisible(grids[i].toBool());
                axis[i]->setScaleType(logs[i].toBool()?QCPAxis::stLogarithmic:QCPAxis::stLinear);
            }
        }
    }
    settings->endGroup();
}

void
nCustomPlot::saveSettings(QSettings *settings) {
    settings->beginGroup(objectName());
    QStringList labels;
    QList<QVariant> grids, logs;
    foreach (QCPAxis *axis, findChildren<QCPAxis *>()) {
        if(axis->visible()) {
            labels << axis->label();
            grids << axis->grid()->visible();
            logs << QVariant::fromValue(axis->scaleType()==QCPAxis::stLogarithmic);
        }
    }
    settings->setValue("labels",labels);
    settings->setValue("grids",grids);
    settings->setValue("logs",logs);
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


