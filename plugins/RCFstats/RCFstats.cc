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
#include "RCFstats.h"
#include <functional>

RCFstats::RCFstats(neutrino *nparent) : nGenericPan(nparent)
{
    my_w.setupUi(this);

    // signals
    circ =  new nCircle(this, 1);

    circ->setRect(QRectF(0,0,100,100));

    //box->setRect(QRectF(0,0,100,100));
    connect(my_w.actionCircle, SIGNAL(triggered()), circ, SLOT(togglePadella()));

    //connect(my_w.actionSaveClipboard, SIGNAL(triggered()), my_w.plot, SLOT(copy_data()));
    //connect(my_w.actionSaveTxt      , SIGNAL(triggered()), my_w.plot, SLOT(save_data()));
    //connect(my_w.actionSavePDF      , SIGNAL(triggered()), my_w.plot, SLOT(export_image()));

    //my_w.plot->graph(0)->setName("Horizontal");
    //my_w.plot->graph(1)->setName("Vertical");

    show();
    connect(nparent, SIGNAL(bufferChanged(nPhysD *)), this, SLOT(updateStats()));

    connect(circ, SIGNAL(sceneChanged()), this, SLOT(sceneChanged()));
    //connect(my_w.tabWidget, SIGNAL(currentChanged(int)), this, SLOT(updatePlot()));

    updateStats();
}

void RCFstats::sceneChanged() {
    if (sender()==circ) updateStats();
    else DEBUG("got signal from "<<sender()->objectName().constData());
}

void RCFstats::mouseAtWorld(QPointF p) {
    if (currentBuffer) {
        //my_w.plot->setMousePosition(p.x(),p.y());
    }
}

void RCFstats::updateStats() {
    DEBUG( my_w.tabWidget->currentIndex());


    if (currentBuffer) {
        QRect geomBox=circ->getRect(currentBuffer);
        if (geomBox.isEmpty()) {
            my_w.statusBar->showMessage(tr("Attention: the region is outside the image!"),2000);
            return;
        }
        int dx=geomBox.width();
        int dy=geomBox.height();

        //DEBUG( "rect circ x "<<(circ->getRect(currentBuffer).x()) );
        //DEBUG( "rect circ y "<<(circ->getRect(currentBuffer).y()) );
        //DEBUG( "rect circ dx "<<dx<<", dy "<<dy );

        vec2f orig=currentBuffer->get_origin();
        vec2f scal=currentBuffer->get_scale();

        circ->setCenter(QPointF(orig.x(), orig.y()));



        // plot tab
        if (my_w.tabWidget->currentIndex()==0) {
            /*QVector<double> xd(dx);
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
            my_w.plot->replot();*/
        } else {
            double sum = 0;
            double sumsq = 0;
            double pt_count = 0;
            double tot_points = 0;

            double mean = 0;
            double variance = 0;

            //QPolygonF pg = circ->shape().toFillPolygon();

            for (int j=0;j<dy;j++){
                for (int i=0;i<dx; i++) {
                    QPointF pt(i+geomBox.x(),j+geomBox.y());

                    double cc_x = geomBox.x()+dx/2.;
                    double cc_y = geomBox.y()+dy/2.;

                    double xx = i+geomBox.x();
                    double yy = j+geomBox.y();

                    //if (circ->contains(pt)) {
                    if (sqrt( pow(xx-cc_x, 2)+pow(yy-cc_y, 2)) < dx/2 ) {
                        double val=currentBuffer->point(i+geomBox.x(),j+geomBox.y(),0.0);
                        sum+=val;
                        sumsq+=val*val;
                        pt_count++;
                    }
                    tot_points++;
                }
            }

            mean=sum/pt_count;
            variance = sqrt((sumsq/pt_count) - pow(mean, 2));

            my_w.sum->setText(QLocale().toString(sum));
            my_w.mean->setText(QLocale().toString(mean));

            my_w.variance->setText(QLocale().toString(variance));
            my_w.surface->setText(QLocale().toString(pt_count));

        }
    }
}
