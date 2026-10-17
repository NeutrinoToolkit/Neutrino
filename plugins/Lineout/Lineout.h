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
#ifndef nlineout_H_
#define nlineout_H_

#include <QtGui>
#include <QWidget>
#include <QScrollBar>
#include <algorithm>

#include "nGenericPan.h"
#include "ui_Lineout.h"

// #include <iostream>
#include "nPhysImageF.h"
#include "nPhysMaths.h"
#include "neutrino.h"


class nLineout : public nGenericPan {
    Q_OBJECT
public:
    Q_INVOKABLE nLineout(neutrino *parent, enum phys_direction plot_dir) : nGenericPan(parent),
        cut_dir(plot_dir)
    {
        my_w.setupUi(this);

        connect(my_w.actionToggleZoom,SIGNAL(triggered()), this, SLOT(updatePlot()));

        // connect(parent, SIGNAL(bufferChanged(nPhysD*)), this, SLOT(updatePlot()));
        // (same signal as the nparent one below: every buffer change replotted twice)

        // connect(my_w.actionLockClick,SIGNAL(triggered()), this, SLOT(setBehaviour()));
        // toggled: also emitted when the state is restored from the settings (setChecked)
        connect(my_w.actionLockClick,SIGNAL(toggled(bool)), this, SLOT(setBehaviour()));

        // connect(my_w.actionLockColors, SIGNAL(toggled(bool)), this, SLOT(updatePlot()));
        connect(my_w.actionLockColors, SIGNAL(toggled(bool)), this, SLOT(lockColors(bool)));

        connect(nparent, SIGNAL(bufferChanged(nPhysD *)), this, SLOT(updatePlot()));

        // in zoom mode the plot covers the visible part of the image: follow the scrollbars too
        connect(nparent->my_view->horizontalScrollBar(), SIGNAL(valueChanged(int)), this, SLOT(viewChanged()));
        connect(nparent->my_view->verticalScrollBar(), SIGNAL(valueChanged(int)), this, SLOT(viewChanged()));

        my_w.plot->addGraph(my_w.plot->xAxis, my_w.plot->yAxis);
        my_w.plot->graph(0)->setPen(QPen(Qt::black));
        QString namedirection(plot_dir==PHYS_HORIZONTAL?"Horizontal":"Vertical");
        my_w.plot->graph(0)->setName(namedirection);

        setProperty("panName",namedirection);
        show(true);
        setBehaviour();
        updatePlot();
    }


public slots:

    void updatePlot(QPointF my_point=QPointF()) {
        // if (my_point.isNull())
        //     my_point=nparent->my_view->my_mouse.pos();
        // without a point (buffer changed, zoom...) keep the clicked point in lock-click mode, otherwise follow the mouse
        if (my_point.isNull())
            my_point=(my_w.actionLockClick->isChecked() && !lastPoint.isNull()) ? lastPoint : nparent->my_view->my_mouse.pos();
        lastPoint=my_point;

        if (nparent->getCurrentBuffer() && nparent->nPhysExists(currentBuffer)) {
            vec2f curbuf_or=currentBuffer->get_origin();
            vec2f curbuf_sc=currentBuffer->get_scale();
            vec2i curbuf_si=currentBuffer->get_size();

            // vec2i b_p(my_point.x(),my_point.y());
            // floor: (int) truncates -0.5 to row 0
            vec2i b_p((int)floor(my_point.x()),(int)floor(my_point.y()));

            //get bounds from view
            QPointF orig,corner;
            if (my_w.actionToggleZoom->isChecked()) {
                orig = nparent->my_view->mapToScene(QPoint(0,0));
                corner = nparent->my_view->mapToScene(QPoint(nparent->my_view->width(), nparent->my_view->height()));
            } else {
                orig = QPoint(0,0);
                corner = QPoint(curbuf_si.x(),curbuf_si.y());
            }

            vec2i b_o((int)orig.x(),(int)orig.y());
            vec2i b_c((int)corner.x(),(int)corner.y());

            phys_direction other_dir=(cut_dir==PHYS_HORIZONTAL?PHYS_VERTICAL:PHYS_HORIZONTAL);

            // size_t lat_skip = std::max(b_o(cut_dir), 0);
            // size_t z_size = std::min(b_c(cut_dir)-lat_skip, curbuf_si(cut_dir)-lat_skip);
            // in int and clamped to the image: the size_t version wrapped around when the view is outside the image
            int lat_skip = std::clamp(b_o(cut_dir), 0, curbuf_si(cut_dir));
            int z_size = std::clamp(b_c(cut_dir), lat_skip, curbuf_si(cut_dir)) - lat_skip;

            QVector<double> x(z_size);
            for (int i=0;i<z_size;i++){
                x[i]=(i+lat_skip-curbuf_or(cut_dir))*curbuf_sc(cut_dir);
            }
            QVector<double> y(z_size);
            if (cut_dir==PHYS_HORIZONTAL) {
                for (int i=0;i<z_size;i++){
                    y[i]=currentBuffer->point(i+lat_skip,b_p(other_dir));
                }
            } else {
                for (int i=0;i<z_size;i++){
                    y[i]=currentBuffer->point(b_p(other_dir),i+lat_skip);
                }
            }

            // the data first: rescaleAxes and the replot (in setMousePosition) used the previous data
            my_w.plot->graph(0)->setData(x,y,true);

            if(my_w.actionLockColors->isChecked()) {
                vec2f rang=currentBuffer->displayRange();
                my_w.plot->graph(0)->valueAxis()->setRange(rang.x(),rang.y());
                my_w.plot->graph(0)->valueAxis()->setProperty("lock",true);
            }

            // my_w.plot->graph(0)->valueAxis()->setProperty("lock",my_w.actionLockColors->isChecked());
            // (set in lockColors: here it overrode the lock button of the plot axis at every mouse move)
            my_w.plot->rescaleAxes();

            // statusBar()->showMessage(tr("Point (")+QLocale().toString(my_point.x())+","+QLocale().toString(my_point.y())+")="+QLocale().toString(currentBuffer->point(my_point.x(),my_point.y())));
            // point() takes unsigned: negative coordinates are passed as int (outside the image -> NaN)
            statusBar()->showMessage(tr("Point (")+QLocale().toString(my_point.x())+","+QLocale().toString(my_point.y())+")="+QLocale().toString(currentBuffer->point(b_p)));
            double pos_mouse=(b_p(cut_dir)-curbuf_or(cut_dir))*curbuf_sc(cut_dir);
            my_w.plot->setMousePosition(pos_mouse);

            // my_w.plot->graph(0)->setData(x,y,true);

        } else {
            my_w.plot->graph(0)->data()->clear();
            my_w.plot->replot();
        }
    }

    void nZoom(double){
        updatePlot();
    }

    void viewChanged() {
        if (my_w.actionToggleZoom->isChecked())
            updatePlot();
    }

    void lockColors(bool lock) {
        my_w.plot->graph(0)->valueAxis()->setProperty("lock",lock);
        updatePlot();
    }

    void setBehaviour() {
        if (my_w.actionLockClick->isChecked()) {
            disconnect(nparent->my_view, SIGNAL(mouseposition(QPointF)), this, SLOT(updatePlot(QPointF)));
            connect(nparent->my_view, SIGNAL(mousePressEvent_sig(QPointF)), this, SLOT(updatePlot(QPointF)), Qt::UniqueConnection);
        } else {
            disconnect(nparent->my_view, SIGNAL(mousePressEvent_sig(QPointF)), this, SLOT(updatePlot(QPointF)));
            connect(nparent->my_view, SIGNAL(mouseposition(QPointF)), this, SLOT(updatePlot(QPointF)), Qt::UniqueConnection);
        }
        updatePlot();
    };

private:
    Ui::nLineout my_w;

    enum phys_direction cut_dir;

    QPointF lastPoint;

};


#endif
