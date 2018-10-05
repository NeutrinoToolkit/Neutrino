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
#include "Lineout_H_V.h"
#include "neutrino.h"

Lineout_H_V::Lineout_H_V(neutrino *parent) : nGenericPan(parent)
{
	my_w.setupUi(this);

    my_w.statusBar->addPermanentWidget(my_w.lockColors, 0);
    my_w.statusBar->addPermanentWidget(my_w.lockClick, 0);

    connect(parent, SIGNAL(bufferChanged(nPhysD*)), this, SLOT(updatePlot()));

    connect(parent, SIGNAL(nZoom(double)), this, SLOT(updatePlot()));

    connect(my_w.lockClick,SIGNAL(released()), this, SLOT(setBehaviour()));

    for (int k=0;k<2;k++) {
        if (currentBuffer) {
            my_w.plot->graph(k)->keyAxis()->setRange(currentBuffer->getW(),currentBuffer->getH());
            my_w.plot->graph(k)->valueAxis()->setRange(currentBuffer->get_min(),currentBuffer->get_max());
        }
        my_w.plot->graph(0)->setAntialiased(false);
    }
    my_w.plot->graph(0)->setName("Horizontal");
    my_w.plot->graph(1)->setName("Vertical");


    show();
    setBehaviour();
    updatePlot();
    
}

void Lineout_H_V::setBehaviour() {
    if (my_w.lockClick->isChecked()) {
        disconnect(nparent->my_w->my_view, SIGNAL(mouseposition(QPointF)), this, SLOT(updatePlot(QPointF)));
        connect(nparent->my_w->my_view, SIGNAL(mousePressEvent_sig(QPointF)), this, SLOT(updatePlot(QPointF)));
    } else {
        disconnect(nparent->my_w->my_view, SIGNAL(mousePressEvent_sig(QPointF)), this, SLOT(updatePlot(QPointF)));
        connect(nparent->my_w->my_view, SIGNAL(mouseposition(QPointF)), this, SLOT(updatePlot(QPointF)));
    }
}

void Lineout_H_V::updatePlot(QPointF p) {

    if (currentBuffer != NULL) {
        if (p.isNull())
            p=nparent->my_w->my_view->my_mouse.pos();

        vec2i b_p(p.x(),p.y());

        //get bounds from view
        QPointF orig = nparent->my_w->my_view->mapToScene(QPoint(0,0));
        QPointF corner = nparent->my_w->my_view->mapToScene(QPoint(nparent->my_w->my_view->width(), nparent->my_w->my_view->height()));

        vec2i b_o((int)orig.x(),(int)orig.y());
        vec2i b_c((int)corner.x(),(int)corner.y());

        std::array<phys_direction,2> cut_dir={{PHYS_HORIZONTAL,PHYS_VERTICAL}};
        std::array<phys_direction,2> oth_dir={{PHYS_VERTICAL,PHYS_HORIZONTAL}};

        vec2f origin=currentBuffer->get_origin();
        vec2f scale=currentBuffer->get_scale();
        vec2i size=currentBuffer->get_size();

        for (int k=0;k<2;k++) {

            size_t lat_skip = std::max(b_o(cut_dir[k]), 0);
            size_t z_size = std::min(b_c(cut_dir[k])-lat_skip, size(cut_dir[k])-lat_skip);

            QVector<double> x(z_size);
            for (unsigned int i=0;i<z_size;i++){
                x[i]=(i+lat_skip-origin(cut_dir[k]))*scale(cut_dir[k]);
            }
            QVector<double> y(z_size);
            if (k==0) {
                for (unsigned int i=0;i<z_size;i++){
                    y[i]=currentBuffer->point(i+lat_skip,b_p(oth_dir[k]));
                }
            } else {
                for (unsigned int i=0;i<z_size;i++){
                    y[i]=currentBuffer->point(b_p(oth_dir[k]),i+lat_skip);
                }
            }
            my_w.plot->graph(k)->setData(x,y,true);

            my_w.plot->graph(k)->keyAxis()->setRange(x.first(), x.last());

            if(my_w.lockColors->isChecked()) {
                vec2f rang=currentBuffer->prop["display_range"];
                my_w.plot->graph(k)->valueAxis()->setRange(rang.x(),rang.y());
            }
            my_w.plot->graph(k)->valueAxis()->setProperty("lock",my_w.lockColors->isChecked());

            my_w.plot->setMousePosition((p.x()-origin.x())*scale(cut_dir[k]),(p.y()-origin.y())*scale(cut_dir[k]));
        }


        statusBar()->showMessage(tr("Point (")+QString::number(p.x())+","+QString::number(p.y())+")="+QString::number(currentBuffer->point(p.x(),p.y())));
        my_w.plot->rescaleAxes();
        my_w.plot->replot();
    }
}



