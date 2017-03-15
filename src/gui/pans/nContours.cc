/*
 *
 *    Copyright (C) 2014 Alessandro Flacco, Tommaso Vinci All Rights Reserved
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


#include "nContours.h"
#include "neutrino.h"
#include "ui_neutrino.h"

nContours::nContours(neutrino *nparent) : nGenericPan(nparent)
{
    my_w.setupUi(this);
    my_c = new nLine(this,3);
    my_c->setPoints(QPolygonF()<<QPointF(0,0)<<QPointF(0,0));

    connect(my_w.actionLoadPref, SIGNAL(triggered()), this, SLOT(loadSettings()));
    connect(my_w.actionSavePref, SIGNAL(triggered()), this, SLOT(saveSettings()));
    connect(my_w.actionLine, SIGNAL(triggered()), my_c, SLOT(togglePadella()));

    show();
    on_percent_released();
    //connect(nparent, SIGNAL(bufferChanged(nPhysD*)), this, SLOT(calculate_stats()));
    //connect(my_w.zero_dsb, SIGNAL(editingFinished()), this, SLOT(calculate_stats()));
    connect(my_w.draw_pb, SIGNAL(released()), this, SLOT(draw()));

    //calculate_stats();
}

void
nContours::on_percent_released() {
    my_w.level_dsb->setSuffix(my_w.percent->isChecked()?"%":"");
}

void
nContours::on_actionCenter_toggled(bool check) {
    if (check) {
        connect(nparent->my_w->my_view, SIGNAL(mouseDoubleClickEvent_sig(QPointF)), this, SLOT(setOrigin(QPointF)));
    } else {
        disconnect(nparent->my_w->my_view, SIGNAL(mouseDoubleClickEvent_sig(QPointF)), this, SLOT(setOrigin(QPointF)));
    }
}

void
nContours::setOrigin(QPointF p) {
    if (currentBuffer) {
        currentBuffer->set_origin(p.x(),p.y());
        nparent->createQimage();
    }
    my_w.actionCenter->setChecked(false);
}

void
nContours::draw()
{
    saveDefaults();
    if (currentBuffer)
        return;

    // 0. build decimated
    decimated = nPhysD(*currentBuffer);
    if(my_w.blur_radius_sb->value()>0) {
        phys_fast_gaussian_blur(decimated, my_w.blur_radius_sb->value());
    }

    // 1. find centroid
    vec2 centr;
    if (currentBuffer->get_origin() == vec2(0,0)) {
        centr = decimated.max_Tv;
    } else {
        centr = currentBuffer->get_origin();
    }
    decimated.set_origin(centr);

    std::list<vec2> contour;
    double cutoff=my_w.level_dsb->value();
    if (my_w.percent->isChecked()) {
        cutoff = decimated.get_min() + (decimated.get_max()-decimated.get_min())*(cutoff/100.0) ;
    }

    DEBUG("cutoff" << cutoff);

    contour_trace(decimated, contour, cutoff);


    my_c->setPoints(QPolygonF());
    if (contour.size() > 0) {

        // set polygon
        my_c->setPoints(QPolygonF());
        QPolygonF myp;
        for (auto &p : contour) {
            myp<<QPointF(p.x(), p.y());
        }
        my_c->setPoints(myp);
        currentBuffer->set_origin(centr);
        //my_w.statusBar->showMessage("Contour ok");
        my_w.statusBar->showMessage(QString::number(cutoff) + " : " + QString::number(contour.size())+" "+tr("points"),5000);
    } else {
        my_w.statusBar->showMessage(QString::number(cutoff) + " : "+tr(" cannot trace contour"),5000);
    }

}

