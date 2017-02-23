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
#include "nZoomWin.h"
#include "neutrino.h"
#include "ui_neutrino.h"

// physGhosts

nZoomWin::nZoomWin(neutrino *nparent) : nGenericPan(nparent)
{
    my_w.setupUi(this);

    my_w.toolBar->addWidget(my_w.my_scale);
    connect(my_w.actionLockClick, SIGNAL(triggered()), this, SLOT(setBehaviour()));
    my_w.my_view->setInteractive(true);
    connect(my_w.my_scale, SIGNAL(valueChanged(double)), this, SLOT(changeZoom(double)));

    show();
    setBehaviour();

    my_w.my_view->setScene(&(nparent->getScene()));
    my_w.my_view->scale(2,2);

}

void nZoomWin::updatePlot(QPointF p) {
    my_w.my_view->centerOn(p);
}

void nZoomWin::changeZoom(double val) {
    my_w.my_view->resetTransform();
    my_w.my_view->scale(val,val);
}


void nZoomWin::setBehaviour() {
    if (my_w.actionLockClick->isChecked()) {
        disconnect(nparent->my_w->my_view, SIGNAL(mouseposition(QPointF)), this, SLOT(updatePlot(QPointF)));
        connect(nparent->my_w->my_view, SIGNAL(mousePressEvent_sig(QPointF)), this, SLOT(updatePlot(QPointF)));
    } else {
        disconnect(nparent->my_w->my_view, SIGNAL(mousePressEvent_sig(QPointF)), this, SLOT(updatePlot(QPointF)));
        connect(nparent->my_w->my_view, SIGNAL(mouseposition(QPointF)), this, SLOT(updatePlot(QPointF)));
    }
}
