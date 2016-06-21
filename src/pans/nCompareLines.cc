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
#include "nCompareLines.h"

nCompareLines::nCompareLines(neutrino *nparent, QString winname)
    : nGenericPan(nparent, winname)
{
    my_w.setupUi(this);

    // signals
    line =  new nLine(nparent);
    line->setParentPan(panName,1);
    QPolygonF poly;
    poly << QPointF(0,0) << QPointF(100,100);
    line->setPoints(poly);

    connect(my_w.actionLoadPref, SIGNAL(triggered()), this, SLOT(loadSettings()));
    connect(my_w.actionSavePref, SIGNAL(triggered()), this, SLOT(saveSettings()));
    connect(my_w.actionSaveClipboard, SIGNAL(triggered()), this, SLOT(copy_clip()));
    connect(my_w.actionSaveTxt, SIGNAL(triggered()), this, SLOT(export_txt()));
    connect(my_w.actionSavePDF, SIGNAL(triggered()), this, SLOT(export_pdf()));
    connect(my_w.actionAddAll, SIGNAL(triggered()), this, SLOT(addImages()));
    connect(my_w.actionRemoveAll, SIGNAL(triggered()), this, SLOT(removeImages()));


    connect(my_w.actionLine, SIGNAL(triggered()), line, SLOT(togglePadella()));
    connect(my_w.addImage, SIGNAL(released()), this, SLOT(addImage()));
    connect(my_w.removeImage, SIGNAL(released()), this, SLOT(removeImage()));

    connect(my_w.current, SIGNAL(released()), this, SLOT(updatePlot()));

    connect(nparent, SIGNAL(physDel(nPhysD*)), this, SLOT(physDel(nPhysD*)));
    connect(nparent, SIGNAL(physMod(std::pair<nPhysD*,nPhysD*>)), this, SLOT(physMod(std::pair<nPhysD*,nPhysD*>)));

    my_w.plot->xAxis->setLabel(tr("Distance"));
    my_w.plot->yAxis->setLabel(tr("Value"));

    decorate();
    loadDefaults();
    connect(line, SIGNAL(sceneChanged()), this, SLOT(sceneChanged()));
    connect(nparent, SIGNAL(bufferChanged(nPhysD*)), this, SLOT(updatePlot()));
    updatePlot();
}


void nCompareLines::physDel(nPhysD* my_phys) {
    images.removeAll(my_phys);
    updatePlot();
}

void nCompareLines::physMod(std::pair<nPhysD*,nPhysD*> my_mod) {
    images.removeAll(my_mod.first);
    images.append(my_mod.second);
    updatePlot();
}

void nCompareLines::addImage() {
    nPhysD *my_phys=nGenericPan::getPhysFromCombo(my_w.image);
    if(!images.contains(my_phys)) {
        images.append(my_phys);
    }
    updatePlot();
}

void nCompareLines::removeImage() {
    images.removeAll(nGenericPan::getPhysFromCombo(my_w.image));
    updatePlot();
}

void nCompareLines::addImages() {
    foreach (nPhysD *my_phys, nparent->getBufferList()) {
        if(!images.contains(my_phys)) {
            images.append(my_phys);
        }
    }
    updatePlot();
}

void nCompareLines::removeImages() {
    images.clear();
    updatePlot();
}

void nCompareLines::sceneChanged() {
    if (sender()==line) updatePlot();
}

void nCompareLines::updatePlot() {
    if (currentBuffer && isVisible()) {

        my_w.plot->clearGraphs();

        QPolygonF my_poly=line->poly(line->numPoints);
        for (int i=0; i<nparent->getBufferList().size(); i++) {
            nPhysD *phys=nparent->getBufferList().at(i);

            if (images.contains(phys) || (my_w.current->isChecked() && phys==currentBuffer)) {
                QVector<double> toPlotx;
                QVector<double> toPloty;

                double dist=0.0;
                double my_val=0.0;
                vec2f scale=phys->get_scale();
                for(int ii=0;ii<my_poly.size()-1;ii++) {
                    QPointF p=my_poly.at(ii);
                    my_val=phys->getPoint(p.x()-phys->get_origin().x(),p.y()-phys->get_origin().y());
                    if (std::isfinite(my_val)) {
                        toPlotx << dist;
                        toPloty << my_val;
                    }
                    QPointF deltaP=my_poly.at(ii+1)-my_poly.at(ii);
                    dist+=sqrt(pow(deltaP.x()*scale.x(),2)+pow(deltaP.y()*scale.y(),2));
                }
                QPointF p=my_poly.last();
                my_val=phys->getPoint(p.x(),p.y());
                if (std::isfinite(my_val)) {
                    toPlotx << dist;
                    toPloty << my_val;
                }
                QCPGraph* graph=my_w.plot->addGraph(my_w.plot->xAxis, my_w.plot->yAxis);
                graph->setName(QString::fromStdString(phys->getName()));
                graph->setPen(QPen((phys==currentBuffer?Qt::blue:Qt::red)));
                graph->setData(toPlotx,toPloty);
            }

        }
        my_w.plot->rescaleAxes();
        my_w.plot->replot();
    }
}

void nCompareLines::copy_clip() {
    QString t;
    QTextStream out(&t);
    getText(out);
    QApplication::clipboard()->setText(t);
    showMessage(tr("Points copied to clipboard"));
}

void nCompareLines::getText(QTextStream &out) {
    out << "# " << panName << " " << my_w.plot->graphCount() << endl;
    for (int g=0; g<my_w.plot->graphCount(); g++) {
        out << "# " << g << " " << my_w.plot->graph(g)->name() << endl;
        const QCPDataMap *dataMap = my_w.plot->graph(g)->data();
        QMap<double, QCPData>::const_iterator i = dataMap->constBegin();
        while (i != dataMap->constEnd()) {
            out << i.value().key << " " << i.value().value << endl;
            ++i;
        }
        out << endl << endl;
    }
}

void nCompareLines::export_txt() {
    QString fnametmp=QFileDialog::getSaveFileName(this,tr("Save data in text"),property("fileTxt").toString(),tr("Text files (*.txt *.csv);;Any files (*)"));
    if (!fnametmp.isEmpty()) {
        setProperty("fileTxt", fnametmp);
        QFile t(fnametmp);
        t.open(QIODevice::WriteOnly| QIODevice::Text);
        QTextStream out(&t);
        getText(out);
        t.close();
        showMessage(tr("Export in file:")+fnametmp,2000);
    }
}

void
nCompareLines::export_pdf() {
    QString fnametmp = QFileDialog::getSaveFileName(this,tr("Save Drawing"),property("fileExport").toString(),"Vector files (*.pdf,*.svg)");
    if (!fnametmp.isEmpty()) {
        setProperty("fileExport", fnametmp);
        my_w.plot->savePdf(fnametmp,true,0,0,"Neutrino", panName);
    }
}

