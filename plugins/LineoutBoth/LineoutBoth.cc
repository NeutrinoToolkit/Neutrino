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
#include "LineoutBoth.h"
#include "neutrino.h"

LineoutBoth::LineoutBoth(neutrino *parent) : nGenericPan(parent)
{
	my_w.setupUi(this);

    my_w.statusBar->addPermanentWidget(my_w.lockColors, 0);
    my_w.statusBar->addPermanentWidget(my_w.lockClick, 0);

    connect(parent, SIGNAL(bufferChanged(nPhysD*)), this, SLOT(updateLastPoint(void)));

    connect(parent, SIGNAL(nZoom(double)), this, SLOT(updateLastPoint(void)));

    connect(my_w.lockClick,SIGNAL(released()), this, SLOT(setBehaviour()));

    for (int k=0;k<2;k++) {
        if (currentBuffer) {
            my_w.plot->graph(k)->keyAxis()->setRange(currentBuffer->getW(),currentBuffer->getH());
            my_w.plot->graph(k)->valueAxis()->setRange(currentBuffer->get_min(),currentBuffer->get_max());
        }
    }
    my_w.plot->graph(0)->setName("Horizontal");
    my_w.plot->graph(1)->setName("Vertical");

    show();
    setBehaviour();
	updateLastPoint();
    
}

void LineoutBoth::setBehaviour() {
    if (my_w.lockClick->isChecked()) {
        disconnect(nparent->my_w->my_view, SIGNAL(mouseposition(QPointF)), this, SLOT(updatePlot(QPointF)));
        connect(nparent->my_w->my_view, SIGNAL(mousePressEvent_sig(QPointF)), this, SLOT(updatePlot(QPointF)));
    } else {
        disconnect(nparent->my_w->my_view, SIGNAL(mousePressEvent_sig(QPointF)), this, SLOT(updatePlot(QPointF)));
        connect(nparent->my_w->my_view, SIGNAL(mouseposition(QPointF)), this, SLOT(updatePlot(QPointF)));
    }
}

void LineoutBoth::updatePlot(QPointF p) {

    if (currentBuffer != NULL) {

        vec2 b_p(p.x(),p.y());

        //get bounds from view
        QPointF orig = nparent->my_w->my_view->mapToScene(QPoint(0,0));
        QPointF corner = nparent->my_w->my_view->mapToScene(QPoint(nparent->my_w->my_view->width(), nparent->my_w->my_view->height()));

        vec2 b_o((int)orig.x(),(int)orig.y());
        vec2 b_c((int)corner.x(),(int)corner.y());

        for (int k=0;k<2;k++) {
            phys_direction cut_dir=k==0?PHYS_HORIZONTAL:PHYS_VERTICAL;
            phys_direction oth_dir=k==0?PHYS_VERTICAL:PHYS_HORIZONTAL;

            size_t lat_skip = std::max(b_o(cut_dir), 0);
            size_t z_size = std::min(b_c(cut_dir)-lat_skip, currentBuffer->getSizeByIndex(cut_dir)-lat_skip);

            QVector<double> x(z_size);
            for (unsigned int i=0;i<z_size;i++){
                x[i]=(i+lat_skip-currentBuffer->get_origin(cut_dir))*currentBuffer->get_scale(cut_dir);
            }
            QVector<double> y(z_size);
            if (k==0) {
                for (unsigned int i=0;i<z_size;i++){
                    y[i]=currentBuffer->point(i+lat_skip,b_p(oth_dir));
                }
            } else {
                for (unsigned int i=0;i<z_size;i++){
                    y[i]=currentBuffer->point(b_p(oth_dir),i+lat_skip);
                }
            }
            my_w.plot->graph(k)->setData(x,y);

            my_w.plot->graph(k)->keyAxis()->setRange(x.first(), x.last());


            my_w.plot->graph(k)->valueAxis()->setProperty("lock",my_w.lockColors->isChecked());

            if(my_w.lockColors->isChecked()) {
                vec2f rang=currentBuffer->property["display_range"];
                my_w.plot->graph(k)->valueAxis()->setRange(rang.x(),rang.y());
            }

            vec2f phys_origin=currentBuffer->get_origin();
            my_w.plot->setMousePosition((p.x()-phys_origin.x())*currentBuffer->get_scale(cut_dir),(p.y()-phys_origin.y())*currentBuffer->get_scale(cut_dir));
        }


        statusBar()->showMessage(tr("Point (")+QString::number(p.x())+","+QString::number(p.y())+")="+QString::number(currentBuffer->point(p.x(),p.y())));
        my_w.plot->rescaleAxes();
        my_w.plot->replot();
    }
}

void LineoutBoth::updateLastPoint() {
    if (!my_w.lockClick->isChecked()) {
        updatePlot(nparent->my_w->my_view->my_mouse.pos());
    }
}



