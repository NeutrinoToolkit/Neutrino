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
    setupUi(this);

    // signals
    QPolygonF poly;
    poly << QPointF(0,0) << QPointF(100,100);
    line.setPoints(poly);

    connect(actionSaveClipboard, &QAction::triggered, plot, &nCustomPlot::copy_data);
    connect(actionSaveTxt, &QAction::triggered, plot, &nCustomPlot::save_data);
    connect(actionSavePDF, &QAction::triggered, plot, &nCustomPlot::export_image);

    connect(actionAddAll, &QAction::triggered, this, &Compare_lines::addImages);
    connect(actionRemoveAll, &QAction::triggered, this, &Compare_lines::removeImages);


    connect(addImageB, &QToolButton::released, this, &Compare_lines::addImage);
    connect(removeImageB, &QToolButton::released, this, &Compare_lines::removeImage);

    connect(current, &QCheckBox::released, this, &Compare_lines::updatePlot);

    connect(nparent, &neutrino::physDel, this, &Compare_lines::physDel);
    connect(nparent, &neutrino::physReplace, this, &Compare_lines::physReplace);

    plot->xAxis->setLabel(tr("Distance"));
    plot->yAxis->setLabel(tr("Value"));

    plot->addGraph(plot->xAxis, plot->yAxis);
    plot->graph(0)->setName("Compare Lines");

    show();
    connect(actionLine, &QAction::triggered, &line, &nLine::togglePadella);
    connect(&line, &nLine::sceneChanged, this, &Compare_lines::sceneChanged);
    connect(nparent, &neutrino::bufferChanged, this, &Compare_lines::updatePlot);
    updatePlot();
}


void Compare_lines::physDel(nPhysD* my_phys) {
    images.removeAll(my_phys);
    updatePlot();
}

void Compare_lines::physReplace(std::pair<nPhysD*,nPhysD*> my_mod) {
    // follow only the images we are comparing
    if (images.contains(my_mod.first) && my_mod.first!=my_mod.second) {
        images.replace(images.indexOf(my_mod.first), my_mod.second);
        images.removeAll(my_mod.first);
    }
    updatePlot();
}

void Compare_lines::addImage() {
    nPhysD *my_phys=nGenericPan::getPhysFromCombo(image);
    if(my_phys && !images.contains(my_phys)) {
        images.append(my_phys);
    }
    updatePlot();
}

void Compare_lines::removeImage() {
    images.removeAll(nGenericPan::getPhysFromCombo(image));
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

// refreshed while hidden is skipped: catch up when shown
void Compare_lines::showEvent(QShowEvent *e) {
    nGenericPan::showEvent(e);
    updatePlot();
}

void Compare_lines::updatePlot() {
    if (!isVisible()) return;

    plot->clearGraphs();
    if (!currentBuffer) { // nothing displayed: no stale curves
        plot->replot();
        return;
    }

    // Qt's 10 standard chart colors; the displayed image is drawn thicker
    static const std::array<QColor,10> colors={{QColor(0x1f77b4),QColor(0xff7f0e),QColor(0x2ca02c),QColor(0xd62728),QColor(0x9467bd),
                                                QColor(0x8c564b),QColor(0xe377c2),QColor(0x7f7f7f),QColor(0xbcbd22),QColor(0x17becf)}};

    QPolygonF my_poly=line.poly(line.numPoints);
    // qDebug() << my_poly;

    const QList<nPhysD*> buffers=nparent->getBufferList();
    int ncurve=0;
    for (nPhysD *phys : buffers) {
        if (images.contains(phys) || (current->isChecked() && phys==currentBuffer)) {
            QVector<double> toPlotx;
            QVector<double> toPloty;

            // line points are relative to the image origin: pixel of this image = point + its origin
            vec2f shift=phys->get_origin();
            // vec2f scale=phys->get_scale();
            double dist=0.0;
            for(int ii=0;ii<my_poly.size();ii++) {
                const QPointF p=my_poly.at(ii);
                double my_val=phys->getPoint(p.x()+shift.x(),p.y()+shift.y());
                if (std::isfinite(my_val)) {
                    toPlotx << dist;
                    toPloty << my_val;
                }
                if (ii+1<my_poly.size()) {
                    QPointF deltaP=my_poly.at(ii+1)-p;
                    // dist+=sqrt(pow(deltaP.x()*scale.x(),2)+pow(deltaP.y()*scale.y(),2));
                    // in world units, also along non-linear axes (the same with linear ones)
                    const QPointF a=p+QPointF(shift.x(),shift.y()), b=a+deltaP;
                    dist+=std::hypot(phys->to_world(PHYS_X,b.x())-phys->to_world(PHYS_X,a.x()),
                                     phys->to_world(PHYS_Y,b.y())-phys->to_world(PHYS_Y,a.y()));
                }
            }
            QCPGraph* graph=plot->addGraph(plot->xAxis, plot->yAxis);
            graph->setName(QString::fromStdString(phys->getName()));
            graph->setPen(QPen(colors[ncurve++ % colors.size()], phys==currentBuffer ? 2.5 : 1.0));
            graph->setData(toPlotx,toPloty);
        }
    }
    plot->rescaleAxes();
    plot->replot();
}
