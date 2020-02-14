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
#include "Box_lineout.h"
#include <functional>

Box_lineout::Box_lineout(neutrino *nparent) : nGenericPan(nparent)
{
    my_w.setupUi(this);

    // signals
    box =  new nRect(this,1);
    box->setRect(QRectF(0,0,100,100));
    connect(my_w.actionRect, SIGNAL(triggered()), box, SLOT(togglePadella()));

    connect(my_w.actionSaveClipboard, SIGNAL(triggered()), my_w.plot, SLOT(copy_data()));
    connect(my_w.actionSaveTxt      , SIGNAL(triggered()), my_w.plot, SLOT(save_data()));
    connect(my_w.actionSavePDF      , SIGNAL(triggered()), my_w.plot, SLOT(export_image()));

    my_w.plot->graph(0)->setName("Horizontal");
    my_w.plot->graph(1)->setName("Vertical");

    show();
    connect(nparent, SIGNAL(bufferChanged(nPhysD *)), this, SLOT(updatePlot()));
    connect(box, SIGNAL(sceneChanged()), this, SLOT(sceneChanged()));
    connect(my_w.tabWidget, SIGNAL(currentChanged(int)), this, SLOT(updatePlot()));

    updatePlot();
}

void Box_lineout::sceneChanged() {
    if (sender()==box) updatePlot();
}

void Box_lineout::mouseAtWorld(QPointF p) {
    if (currentBuffer) {
        my_w.plot->setMousePosition(p.x(),p.y());
    }
}

void Box_lineout::updatePlot() {
    DEBUG( my_w.tabWidget->currentIndex());

    if (currentBuffer) {
        QRect geomBox=box->getRect(currentBuffer);
        if (geomBox.isEmpty()) {
            my_w.statusBar->showMessage(tr("Attention: the region is outside the image!"),2000);
            return;
        }
        int dx=geomBox.width();
        int dy=geomBox.height();

        vec2f orig=currentBuffer->get_origin();
        vec2f scal=currentBuffer->get_scale();

        if (my_w.tabWidget->currentIndex()==0) {
            QVector<double> xd(dx);
            QVector<double> yd(dy);
            for (int j=0;j<dy;j++){
                for (int i=0;i<dx; i++) {
                    double val=currentBuffer->point(i+geomBox.x(),j+geomBox.y(),0.0);
                    xd[i]+=val;
                    yd[j]+=val;
                }
            }

            std::transform(xd.begin(), xd.end(), xd.begin(),std::bind(std::divides<double>(), std::placeholders::_1, dy));
            std::transform(yd.begin(), yd.end(), yd.begin(),std::bind(std::divides<double>(), std::placeholders::_1, dx));

            QVector <double> xdata(dx);
            QVector <double> ydata(dy);


            for (int i=0;i<dx;i++) xdata[i]=(geomBox.x()+i-orig.x())*scal.x();
            for (int j=0;j<dy;j++) ydata[j]=(geomBox.y()+j-orig.y())*scal.y();

            my_w.plot->graph(0)->setData(xdata,xd);
            my_w.plot->graph(1)->setData(ydata,yd);

            my_w.plot->rescaleAxes();
            my_w.plot->replot();
        } else {
            double sum = 0;
            double mean = 0;
            double variance = 0;
            double surface = dx*dy*scal.x()*scal.y();

            for (int j=0;j<dy;j++){
                for (int i=0;i<dx; i++) {
                    double val=currentBuffer->point(i+geomBox.x(),j+geomBox.y(),0.0);
                    sum+=val;
                }
            }
            mean=sum/(dx*dy);
            for (int j=0;j<dy;j++){
                for (int i=0;i<dx; i++) {
                    double val=currentBuffer->point(i+geomBox.x(),j+geomBox.y(),0.0);
                    variance+=pow(val-mean,2);
                }
            }
            variance = sqrt(variance/(dx*dy-1));


            my_w.sum->setText(QLocale().toString(sum));
            my_w.mean->setText(QLocale().toString(mean));
            my_w.variance->setText(QLocale().toString(variance));
            my_w.surface->setText(QLocale().toString(surface));

        }
    }
}
