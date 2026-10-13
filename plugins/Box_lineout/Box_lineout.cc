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
    setupUi(this);

    // signals
    box =  new nRect(this,1);
    box->setRect(QRectF(0,0,100,100));
    connect(actionRect, SIGNAL(triggered()), box, SLOT(togglePadella()));

    connect(actionSaveClipboard, SIGNAL(triggered()), plot, SLOT(copy_data()));
    connect(actionSaveTxt      , SIGNAL(triggered()), plot, SLOT(save_data()));
    connect(actionSavePDF      , SIGNAL(triggered()), plot, SLOT(export_image()));

    plot->graph(0)->setName("Horizontal");
    plot->graph(1)->setName("Vertical");

    show();
    connect(nparent, SIGNAL(bufferChanged(nPhysD *)), this, SLOT(updatePlot()));
    connect(box, SIGNAL(sceneChanged()), this, SLOT(sceneChanged()));
    connect(tabWidget, SIGNAL(currentChanged(int)), this, SLOT(updatePlot()));
    connect(plot, SIGNAL(mouseMove(QMouseEvent*)), this, SLOT(mouseMove(QMouseEvent*)));
    updatePlot();
}

void Box_lineout::mouseMove(QMouseEvent* e) {
    double x1 = plot->xAxis->pixelToCoord(e->pos().x());
    double y1 = plot->yAxis->pixelToCoord(e->pos().y());
    double x2 = plot->xAxis2->pixelToCoord(e->pos().x());
    double y2 = plot->yAxis2->pixelToCoord(e->pos().y());

    QString msg= plot->xAxis->label() + ": " + QString::number(x1) + " , " + plot->yAxis->label() + ": " + QString::number(y1) + " ; " +\
                 plot->xAxis2->label() + ": " + QString::number(x2) + " , " + plot->yAxis2->label() + ": " + QString::number(y2);
    statusBar()->showMessage(msg);
}

void Box_lineout::on_actionExpand_triggered() {
    if (currentBuffer) {
        box->expandX();
        box->expandY();
    }
}

void Box_lineout::sceneChanged() {
    if (sender()==box) updatePlot();
}

void Box_lineout::mouseAtWorld(QPointF p) {
    if (currentBuffer) {
        plot->setMousePosition(p.x(),p.y());
        QString msg= QString::number(p.x()) + " , " + QString::number(p.y());
        statusBar()->showMessage(msg);
    }
}

void Box_lineout::updatePlot() {
    // DEBUG( tabWidget->currentIndex());

    QRect geomBox;
    if (currentBuffer) {
        geomBox=box->getRect(currentBuffer);
        if (geomBox.isEmpty()) {
            statusBar()->showMessage(tr("Attention: the region is outside the image!"),2000);
        }
    }
    if (!currentBuffer || geomBox.isEmpty()) {
        // don't leave the results of another image/region on screen
        plot->graph(0)->data()->clear();
        plot->graph(1)->data()->clear();
        plot->replot();
        for (QLineEdit *le : {sumLE, meanLE, varianceLE, surfaceLE}) le->clear();
        return;
    }
    int dx=geomBox.width();
    int dy=geomBox.height();

    vec2f orig=currentBuffer->get_origin();
    vec2f scal=currentBuffer->get_scale();

    // only finite pixels count (NaN are common: masks, transformations...)
    if (tabWidget->currentIndex()==0) {
        QVector<double> xd(dx), yd(dy);
        QVector<int> xn(dx), yn(dy);
        for (int j=0;j<dy;j++){
            for (int i=0;i<dx; i++) {
                double val=currentBuffer->point(i+geomBox.x(),j+geomBox.y(),0.0);
                if (std::isfinite(val)) {
                    xd[i]+=val; xn[i]++;
                    yd[j]+=val; yn[j]++;
                }
            }
        }
        for (int i=0;i<dx;i++) xd[i] = xn[i] ? xd[i]/xn[i] : std::numeric_limits<double>::quiet_NaN();
        for (int j=0;j<dy;j++) yd[j] = yn[j] ? yd[j]/yn[j] : std::numeric_limits<double>::quiet_NaN();

        QVector <double> xdata(dx);
        QVector <double> ydata(dy);

        for (int i=0;i<dx;i++) xdata[i]=(geomBox.x()+i-orig.x())*scal.x();
        for (int j=0;j<dy;j++) ydata[j]=(geomBox.y()+j-orig.y())*scal.y();

        plot->graph(0)->setData(xdata,xd);
        plot->graph(1)->setData(ydata,yd);

        plot->rescaleAxes();
        plot->replot();
    } else {
        double sum = 0;
        long long count = 0;
        for (int j=0;j<dy;j++){
            for (int i=0;i<dx; i++) {
                double val=currentBuffer->point(i+geomBox.x(),j+geomBox.y(),0.0);
                if (std::isfinite(val)) {
                    sum+=val;
                    count++;
                }
            }
        }
        double mean = count ? sum/count : std::numeric_limits<double>::quiet_NaN();
        double sumsq = 0;
        for (int j=0;j<dy;j++){
            for (int i=0;i<dx; i++) {
                double val=currentBuffer->point(i+geomBox.x(),j+geomBox.y(),0.0);
                if (std::isfinite(val)) sumsq+=pow(val-mean,2);
            }
        }
        double stddev = count>1 ? sqrt(sumsq/(count-1)) : 0.0;
        double surface = std::abs(dx*dy*scal.x()*scal.y());

        sumLE->setText(QLocale().toString(sum));
        meanLE->setText(QLocale().toString(mean));
        varianceLE->setText(QLocale().toString(stddev));
        surfaceLE->setText(QLocale().toString(surface));
    }
}
