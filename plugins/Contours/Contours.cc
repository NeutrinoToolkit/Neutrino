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

    connect(actionLine, &QAction::triggered, my_c, &nLine::togglePadella);

	show();
	on_percent_released();
	//connect(nparent, SIGNAL(bufferChanged(nPhysD*)), this, SLOT(calculate_stats()));
    //connect(zero_dsb, SIGNAL(editingFinished()), this, SLOT(calculate_stats()));

	//calculate_stats();
}

void
Contours::on_percent_released() {
    level_dsb->setSuffix(percent->isChecked()?"%":"");
}

void
Contours::on_actionCenter_toggled(bool check) {
	if (check) {
        connect(nparent->my_view, &nView::mouseDoubleClickEvent_sig, this, &Contours::setOrigin);
	} else {
        disconnect(nparent->my_view, &nView::mouseDoubleClickEvent_sig, this, &Contours::setOrigin);
	}
}

void
Contours::setOrigin(QPointF p) {
	if (currentBuffer) {
		currentBuffer->set_origin(p.x(),p.y());
        nparent->emitPhysChanged(currentBuffer); // Image_list, other pans and the view
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

		std::list<vec2f> contour;
        double cutoff=level_dsb->value();
        if (percent->isChecked()) {
			cutoff = decimated.get_min() + (decimated.get_max()-decimated.get_min())*(cutoff/100.0) ;
		}

		// DEBUG("cutoff" << cutoff);

        physMath::contour_trace(decimated, contour, cutoff);


		// keep the previous contour if tracing fails (an nLine can't be empty)
		if (contour.size() > 1) {
			// qDebug() << contour.size();

			// // set polygon
            // QPolygonF myp;
			// for (auto &p : contour) {
			// 	myp<<QPointF(p.x(), p.y());
			// }
			// my_c->setPoints(myp);
			// my_c->toggleClosedLine(true);
			if (moveOrigin->isChecked() && currentBuffer->get_origin()!=vec2f(centr.x(),centr.y())) {
				currentBuffer->set_origin(centr);
				nparent->emitPhysChanged(currentBuffer);
			}
			// set polygon (after the origin change): the line sits at the image origin, its points are
			// relative to it; the contour is in pixel coordinates, pixel (i,j) has its centre at (i+0.5,j+0.5)
			vec2f orig=currentBuffer->get_origin();
            QPolygonF myp;
			for (auto &p : contour) {
				myp<<QPointF(p.x()+0.5-orig.x(), p.y()+0.5-orig.y());
			}
			my_c->setPoints(myp);
			my_c->toggleClosedLine(true);
            statusBar()->showMessage(QLocale().toString(cutoff) + " : " + QLocale().toString((unsigned int)contour.size())+" "+tr("points"));
		} else {
            statusBar()->showMessage(QLocale().toString(cutoff) + " : "+tr("cannot trace contour"));
		}
	}
}

