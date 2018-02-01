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
#include "nLineout.h"
#include "neutrino.h"

nLineout::nLineout(neutrino *parent, enum phys_direction plot_dir) : nGenericPan(parent),
  cut_dir(plot_dir)
{
	my_w.setupUi(this);

    connect(my_w.actionToggleZoom,SIGNAL(triggered()), this, SLOT(updateLastPoint()));

    connect(parent, SIGNAL(bufferChanged(nPhysD*)), this, SLOT(updateLastPoint()));

    connect(my_w.actionLockClick,SIGNAL(triggered()), this, SLOT(setBehaviour()));

    connect(my_w.actionLockColors, SIGNAL(toggled(bool)), this, SLOT(updateLastPoint()));

    my_w.plot->addGraph(my_w.plot->xAxis, my_w.plot->yAxis);
    my_w.plot->graph(0)->setPen(QPen(Qt::black));
    QString namedirection(plot_dir==PHYS_HORIZONTAL?"Horizontal":"Vertical");
    my_w.plot->graph(0)->setName(namedirection);

    show();
    setWindowTitle(windowTitle()+" "+namedirection);
    setBehaviour();
    updateLastPoint();
}

void nLineout::setBehaviour() {
    if (my_w.actionLockClick->isChecked()) {
        disconnect(nparent->my_w->my_view, SIGNAL(mouseposition(QPointF)), this, SLOT(updatePlot(QPointF)));
        connect(nparent->my_w->my_view, SIGNAL(mousePressEvent_sig(QPointF)), this, SLOT(updatePlot(QPointF)));
    } else {
        disconnect(nparent->my_w->my_view, SIGNAL(mousePressEvent_sig(QPointF)), this, SLOT(updatePlot(QPointF)));
        connect(nparent->my_w->my_view, SIGNAL(mouseposition(QPointF)), this, SLOT(updatePlot(QPointF)));
    }
    updateLastPoint();
}


// mouse movement
void
nLineout::updatePlot(QPointF my_point) {

    if (currentBuffer != NULL) {

        vec2 b_p(my_point.x(),my_point.y());

        //get bounds from view
        QPointF orig,corner;
        if (my_w.actionToggleZoom->isChecked()) {
            orig = nparent->my_w->my_view->mapToScene(QPoint(0,0));
            corner = nparent->my_w->my_view->mapToScene(QPoint(nparent->my_w->my_view->width(), nparent->my_w->my_view->height()));
        } else {
            orig = QPoint(0,0);
            corner = QPoint(currentBuffer->getW(),currentBuffer->getH());
        }

        vec2 b_o((int)orig.x(),(int)orig.y());
        vec2 b_c((int)corner.x(),(int)corner.y());

        phys_direction other_dir=(cut_dir==PHYS_HORIZONTAL?PHYS_VERTICAL:PHYS_HORIZONTAL);

        size_t lat_skip = std::max(b_o(cut_dir), 0);
        size_t z_size = std::min(b_c(cut_dir)-lat_skip, currentBuffer->getSizeByIndex(cut_dir)-lat_skip);

        QVector<double> x(z_size);
        for (unsigned int i=0;i<z_size;i++){
            x[i]=(i+lat_skip-currentBuffer->get_origin(cut_dir))*currentBuffer->get_scale(cut_dir);
        }
        QVector<double> y(z_size);
        if (cut_dir==PHYS_HORIZONTAL) {
            for (unsigned int i=0;i<z_size;i++){
                y[i]=currentBuffer->point(i+lat_skip,b_p(other_dir));
            }
        } else {
            for (unsigned int i=0;i<z_size;i++){
                y[i]=currentBuffer->point(b_p(other_dir),i+lat_skip);
            }
        }

        my_w.plot->graph(0)->valueAxis()->setProperty("lock",my_w.actionLockColors->isChecked());

        if(my_w.actionLockColors->isChecked()) {
            vec2f rang=currentBuffer->property["display_range"];
            my_w.plot->graph(0)->valueAxis()->setRange(rang.x(),rang.y());
        } else {
        }
        my_w.plot->rescaleAxes();

        statusBar()->showMessage(tr("Point (")+QString::number(my_point.x())+","+QString::number(my_point.y())+")="+QString::number(currentBuffer->point(my_point.x(),my_point.y())));
		double pos_mouse=(b_p(cut_dir)-currentBuffer->get_origin(cut_dir))*currentBuffer->get_scale(cut_dir);
        my_w.plot->setMousePosition(pos_mouse);

        my_w.plot->graph(0)->setData(x,y,true);

    }

}

void nLineout::nZoom(double d) {
    updateLastPoint();
}

void nLineout::updateLastPoint() {
    updatePlot(nparent->my_w->my_view->my_mouse.pos());
}



