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
#include "ZoomWin.h"
#include "neutrino.h"

ZoomWin::ZoomWin(neutrino *nparent) : nGenericPan(nparent)
{
    setupUi(this);

    toolBar->addWidget(my_scale);
    // connect(actionLockClick, SIGNAL(triggered()), this, SLOT(setBehaviour()));
    // toggled: also emitted when the state is restored from the settings (setChecked)
    connect(actionLockClick, SIGNAL(toggled(bool)), this, SLOT(setBehaviour()));
    my_view->setInteractive(true);
    connect(my_scale, SIGNAL(valueChanged(double)), this, SLOT(changeZoom(double)));

    show();
    setBehaviour();

    my_view->setScene(&(nparent->getScene()));
    // my_view->scale(2,2);
    // (multiplied the zoom restored by show(): the view was at twice the value shown)
    changeZoom(my_scale->value());
    // centred on the mouse from the start (it showed the corner of the scene until the mouse moved)
    updatePlot(nparent->my_view->my_mouse.pos());

}

void ZoomWin::updatePlot(QPointF p) {
    lastPoint=p;
    my_view->centerOn(p);
}

void ZoomWin::changeZoom(double val) {
    my_view->resetTransform();
    my_view->scale(val,val);
    my_view->centerOn(lastPoint); // explicitly (it relied on the anchor of the transformation)
}


void ZoomWin::setBehaviour() {
    if (actionLockClick->isChecked()) {
        disconnect(nparent->my_view, SIGNAL(mouseposition(QPointF)), this, SLOT(updatePlot(QPointF)));
        connect(nparent->my_view, SIGNAL(mousePressEvent_sig(QPointF)), this, SLOT(updatePlot(QPointF)), Qt::UniqueConnection);
    } else {
        disconnect(nparent->my_view, SIGNAL(mousePressEvent_sig(QPointF)), this, SLOT(updatePlot(QPointF)));
        connect(nparent->my_view, SIGNAL(mouseposition(QPointF)), this, SLOT(updatePlot(QPointF)), Qt::UniqueConnection);
    }
}
