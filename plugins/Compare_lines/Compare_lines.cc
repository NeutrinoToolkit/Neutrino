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
#include "Compare_lines.h"

Compare_lines::Compare_lines(neutrino *nparent) : nGenericPan(nparent),
     line(this,1)
{
    my_w.setupUi(this);

    // signals
    QPolygonF poly;
    poly << QPointF(0,0) << QPointF(100,100);
    line.setPoints(poly);

    connect(my_w.actionSaveClipboard, SIGNAL(triggered()), my_w.plot, SLOT(copy_data()));
    connect(my_w.actionSaveTxt      , SIGNAL(triggered()), my_w.plot, SLOT(save_data()));
    connect(my_w.actionSavePDF      , SIGNAL(triggered()), my_w.plot, SLOT(export_image()));

    connect(my_w.actionAddAll, SIGNAL(triggered()), this, SLOT(addImages()));
    connect(my_w.actionRemoveAll, SIGNAL(triggered()), this, SLOT(removeImages()));


    connect(my_w.addImage, SIGNAL(released()), this, SLOT(addImage()));
    connect(my_w.removeImage, SIGNAL(released()), this, SLOT(removeImage()));

    connect(my_w.current, SIGNAL(released()), this, SLOT(updatePlot()));

    connect(nparent, SIGNAL(physDel(nPhysD*)), this, SLOT(physDel(nPhysD*)));
    connect(nparent, SIGNAL(physReplace(std::pair<nPhysD*,nPhysD*>)), this, SLOT(physReplace(std::pair<nPhysD*,nPhysD*>)));

    my_w.plot->xAxis->setLabel(tr("Distance"));
    my_w.plot->yAxis->setLabel(tr("Value"));

    my_w.plot->addGraph(my_w.plot->xAxis, my_w.plot->yAxis);
    my_w.plot->graph(0)->setName("Compare Lines");

    show();
    connect(my_w.actionLine, SIGNAL(triggered()), &line, SLOT(togglePadella()));
    connect(&line, SIGNAL(sceneChanged()), this, SLOT(sceneChanged()));
    connect(nparent, SIGNAL(bufferChanged(nPhysD*)), this, SLOT(updatePlot()));
    updatePlot();
}


void Compare_lines::physDel(nPhysD* my_phys) {
    images.removeAll(my_phys);
    updatePlot();
}

void Compare_lines::physReplace(std::pair<nPhysD*,nPhysD*> my_mod) {
    images.removeAll(my_mod.first);
    images.append(my_mod.second);
    updatePlot();
}

void Compare_lines::addImage() {
    nPhysD *my_phys=nGenericPan::getPhysFromCombo(my_w.image);
    if(!images.contains(my_phys)) {
        images.append(my_phys);
    }
    updatePlot();
}

void Compare_lines::removeImage() {
    images.removeAll(nGenericPan::getPhysFromCombo(my_w.image));
    updatePlot();
}

void Compare_lines::addImages() {
    foreach (nPhysD *my_phys, nparent->getBufferList()) {
        if(!images.contains(my_phys)) {
            images.append(my_phys);
        }
    }
    updatePlot();
}

void Compare_lines::removeImages() {
    images.clear();
    updatePlot();
}

void Compare_lines::sceneChanged() {
    if (sender()== &line) updatePlot();
}

void Compare_lines::updatePlot() {
    if (currentBuffer && isVisible()) {

        my_w.plot->clearGraphs();

        QPolygonF my_poly=line.poly(line.numPoints);
		qDebug() << my_poly;

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
					my_val=phys->getPoint(p.x()+phys->get_origin().x(),p.y()+phys->get_origin().y());
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
                graph->setPen(QPen((phys==currentBuffer?my_w.plot->yAxis->labelColor():Qt::red)));
                graph->setData(toPlotx,toPloty);
            }

        }
        my_w.plot->rescaleAxes();
        my_w.plot->replot();
    }
}

