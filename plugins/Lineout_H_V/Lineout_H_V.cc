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
#include <QScrollBar>
#include <algorithm>

Lineout_H_V::Lineout_H_V(neutrino *parent) : nGenericPan(parent)
{
	my_w.setupUi(this);

    my_w.statusBar->addPermanentWidget(my_w.lockColors, 0);
    my_w.statusBar->addPermanentWidget(my_w.lockClick, 0);

    // connect(parent, SIGNAL(bufferChanged(nPhysD*)), this, SLOT(updatePlot()));
    // (same signal as the nparent one below: every buffer change replotted twice)

    connect(parent, SIGNAL(nZoom(double)), this, SLOT(updatePlot()));
    connect(nparent, SIGNAL(bufferChanged(nPhysD *)), this, SLOT(updatePlot()));

    // the plot covers the visible part of the image: follow the scrollbars too
    connect(nparent->my_view->horizontalScrollBar(), SIGNAL(valueChanged(int)), this, SLOT(updatePlot()));
    connect(nparent->my_view->verticalScrollBar(), SIGNAL(valueChanged(int)), this, SLOT(updatePlot()));

    // connect(my_w.lockClick,SIGNAL(released()), this, SLOT(setBehaviour()));
    // toggled: also emitted when the state is restored from the settings (setChecked)
    connect(my_w.lockClick,SIGNAL(toggled(bool)), this, SLOT(setBehaviour()));
    connect(my_w.lockColors,SIGNAL(toggled(bool)), this, SLOT(lockColors(bool)));

    for (int k=0;k<2;k++) {
        if (currentBuffer) {
            // my_w.plot->graph(k)->keyAxis()->setRange(currentBuffer->getW(),currentBuffer->getH());
            // (from W to H: replaced anyway by updatePlot)
            my_w.plot->graph(k)->valueAxis()->setRange(currentBuffer->get_min(),currentBuffer->get_max());
        }
        // my_w.plot->graph(0)->setAntialiased(false);
        my_w.plot->graph(k)->setAntialiased(false);
    }
    my_w.plot->graph(0)->setName("Horizontal");
    my_w.plot->graph(1)->setName("Vertical");


    show();
    setBehaviour();
    updatePlot();
    
}

void Lineout_H_V::setBehaviour() {
    if (my_w.lockClick->isChecked()) {
        disconnect(nparent->my_view, SIGNAL(mouseposition(QPointF)), this, SLOT(updatePlot(QPointF)));
        connect(nparent->my_view, SIGNAL(mousePressEvent_sig(QPointF)), this, SLOT(updatePlot(QPointF)), Qt::UniqueConnection);
    } else {
        disconnect(nparent->my_view, SIGNAL(mousePressEvent_sig(QPointF)), this, SLOT(updatePlot(QPointF)));
        connect(nparent->my_view, SIGNAL(mouseposition(QPointF)), this, SLOT(updatePlot(QPointF)), Qt::UniqueConnection);
    }
}

void Lineout_H_V::lockColors(bool lock) {
    for (int k=0;k<2;k++) {
        my_w.plot->graph(k)->valueAxis()->setProperty("lock",lock);
    }
    updatePlot();
}

void Lineout_H_V::updatePlot(QPointF p) {

    if (currentBuffer) {
        // if (p.isNull())
        //     p=nparent->my_view->my_mouse.pos();
        // without a point (buffer changed, zoom...) keep the clicked point in lock-click mode, otherwise follow the mouse
        if (p.isNull())
            p=(my_w.lockClick->isChecked() && !lastPoint.isNull()) ? lastPoint : nparent->my_view->my_mouse.pos();
        lastPoint=p;

        // vec2i b_p(p.x(),p.y());
        // floor: (int) truncates -0.5 to row 0
        vec2i b_p((int)floor(p.x()),(int)floor(p.y()));

        //get bounds from view
        QPointF orig = nparent->my_view->mapToScene(QPoint(0,0));
        QPointF corner = nparent->my_view->mapToScene(QPoint(nparent->my_view->width(), nparent->my_view->height()));

        vec2i b_o((int)orig.x(),(int)orig.y());
        vec2i b_c((int)corner.x(),(int)corner.y());

        std::array<phys_direction,2> cut_dir={{PHYS_HORIZONTAL,PHYS_VERTICAL}};
        std::array<phys_direction,2> oth_dir={{PHYS_VERTICAL,PHYS_HORIZONTAL}};

        vec2f origin=currentBuffer->get_origin();
        vec2f scale=currentBuffer->get_scale();
        vec2i size=currentBuffer->get_size();

        for (int k=0;k<2;k++) {

            // size_t lat_skip = std::max(b_o(cut_dir[k]), 0);
            // size_t z_size = std::min(b_c(cut_dir[k])-lat_skip, size(cut_dir[k])-lat_skip);
            // in int and clamped to the image: the size_t version wrapped around when the view is outside the image
            int lat_skip = std::clamp(b_o(cut_dir[k]), 0, size(cut_dir[k]));
            int z_size = std::clamp(b_c(cut_dir[k]), lat_skip, size(cut_dir[k])) - lat_skip;

            QVector<double> x(z_size);
            for (int i=0;i<z_size;i++){
                x[i]=(i+lat_skip-origin(cut_dir[k]))*scale(cut_dir[k]);
            }
            QVector<double> y(z_size);
            if (k==0) {
                for (int i=0;i<z_size;i++){
                    y[i]=currentBuffer->point(i+lat_skip,b_p(oth_dir[k]));
                }
            } else {
                for (int i=0;i<z_size;i++){
                    y[i]=currentBuffer->point(b_p(oth_dir[k]),i+lat_skip);
                }
            }
            my_w.plot->graph(k)->setData(x,y,true);

            // my_w.plot->graph(k)->keyAxis()->setRange(x.first(), x.last());
            // (undefined on an empty x, and rescaleAxes below sets the same range)

            if(my_w.lockColors->isChecked()) {
                vec2f rang=currentBuffer->displayRange();
                my_w.plot->graph(k)->valueAxis()->setRange(rang.x(),rang.y());
                my_w.plot->graph(k)->valueAxis()->setProperty("lock",true);
            }
            // my_w.plot->graph(k)->valueAxis()->setProperty("lock",my_w.lockColors->isChecked());
            // (set in lockColors: here it overrode the lock button of the plot axis at every update)

            // my_w.plot->setMousePosition((p.x()-origin.x())*scale(cut_dir[k]),(p.y()-origin.y())*scale(cut_dir[k]));
            // (inside the loop with the scale of one direction for both coordinates: moved after the loop)
        }


        // statusBar()->showMessage(tr("Point (")+QLocale().toString(p.x())+","+QLocale().toString(p.y())+")="+QLocale().toString(currentBuffer->point(p.x(),p.y())));
        // point() takes unsigned: negative coordinates are passed as int (outside the image -> NaN)
        statusBar()->showMessage(tr("Point (")+QLocale().toString(p.x())+","+QLocale().toString(p.y())+")="+QLocale().toString(currentBuffer->point(b_p)));
        my_w.plot->rescaleAxes();
        // my_w.plot->replot();
        // setMousePosition replots
        my_w.plot->setMousePosition((p.x()-origin.x())*scale.x(),(p.y()-origin.y())*scale.y());
    }
}



