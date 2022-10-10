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


#include "Contours.h"
#include "neutrino.h"

Contours::Contours(neutrino *nparent) : nGenericPan(nparent)
{
    setupUi(this);
	my_c = new nLine(this,3);
	my_c->setPoints(QPolygonF()<<QPointF(0,0)<<QPointF(0,0));

    connect(actionLine, SIGNAL(triggered()), my_c, SLOT(togglePadella()));

	show();
	on_percent_released();
	//connect(nparent, SIGNAL(bufferChanged(nPhysD*)), this, SLOT(calculate_stats()));
    //connect(zero_dsb, SIGNAL(editingFinished()), this, SLOT(calculate_stats()));
    connect(draw_pb, SIGNAL(released()), this, SLOT(draw()));

	//calculate_stats();
}

void
Contours::on_percent_released() {
    level_dsb->setSuffix(percent->isChecked()?"%":"");
}

void
Contours::on_actionCenter_toggled(bool check) {
	if (check) {
        connect(nparent->my_view, SIGNAL(mouseDoubleClickEvent_sig(QPointF)), this, SLOT(setOrigin(QPointF)));
	} else {
        disconnect(nparent->my_view, SIGNAL(mouseDoubleClickEvent_sig(QPointF)), this, SLOT(setOrigin(QPointF)));
	}
}

void
Contours::setOrigin(QPointF p) {
	if (currentBuffer) {
		currentBuffer->set_origin(p.x(),p.y());
        nparent->showPhys();
	}
    actionCenter->setChecked(false);
}

void
Contours::on_draw_pb_released()
{
	saveDefaults();
	if (currentBuffer) {

		// 0. build decimated
		decimated = nPhysD(*currentBuffer);
        if(blur_radius_sb->value()>0) {
            physMath::phys_fast_gaussian_blur(decimated, blur_radius_sb->value());
		}

		// 1. find centroid
		vec2i centr;
		if (currentBuffer->get_origin() == vec2i(0,0)) {
			centr = decimated.max_Tv;
		} else {
			centr = currentBuffer->get_origin();
		}
		decimated.set_origin(centr);

		std::list<vec2i> contour;
        double cutoff=level_dsb->value();
        if (percent->isChecked()) {
			cutoff = decimated.get_min() + (decimated.get_max()-decimated.get_min())*(cutoff/100.0) ;
		}

		DEBUG("cutoff" << cutoff);

        physMath::contour_trace(decimated, contour, cutoff);


		my_c->setPoints(QPolygonF());
		if (contour.size() > 0) {
			qDebug() << contour.size();

			// set polygon
            QPolygonF myp;
			for (auto &p : contour) {
				myp<<QPointF(p.x(), p.y());
			}
			my_c->setPoints(myp);
			currentBuffer->set_origin(centr);
            statusBar()->showMessage(QLocale().toString(cutoff) + " : " + QLocale().toString((unsigned int)contour.size())+" "+tr("points"),5000);
		} else {
            statusBar()->showMessage(QLocale().toString(cutoff) + " : "+tr("cannot trace contour"),5000);
		}
	}
}

