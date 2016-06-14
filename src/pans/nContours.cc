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

nContours::nContours(neutrino *nparent, QString winname)
	: nGenericPan(nparent, winname)
{
	my_w.setupUi(this);
	my_c = new nLine(nparent);
	my_c->setParentPan(panName,3);
	
	decorate();
	//connect(nparent, SIGNAL(bufferChanged(nPhysD*)), this, SLOT(calculate_stats()));
	//connect(my_w.zero_dsb, SIGNAL(editingFinished()), this, SLOT(calculate_stats()));
	connect(my_w.draw_pb, SIGNAL(clicked()), this, SLOT(draw()));

	//calculate_stats();
}

void
nContours::draw()
{
	nPhysD *cur = nparent->getBuffer(-1);
	if (!cur) 
		return;

	// 0. build decimated
	decimated = nPhysD(*cur);
	phys_fast_gaussian_blur(decimated, my_w.blur_radius_sb->value());
	decimated.TscanBrightness();

	// 1. find centroid
	vec2 centr;
	if (cur->get_origin() == vec2(0,0)) {
		centr = vec2(decimated.max_Tv_x, decimated.max_Tv_y);
		cur->set_origin(centr);
	} else
		centr = cur->get_origin();
	decimated.set_origin(centr);

	std::list<vec2> contour;
	contour_trace(decimated, contour, my_w.level_dsb->value());
	std::list<vec2>::iterator itr = contour.begin(), itr_last = contour.end();

	DEBUG(5, "got contour of "<<contour.size()<<" points");

	my_c->setPoints(QPolygonF());
	if (contour.size() > 0) {

		// set polygon
		my_c->setPoints(QPolygonF());
		QPolygonF myp;
		for (itr = contour.begin(); itr != itr_last; ++itr) {
			myp<<QPointF((*itr).x(), (*itr).y());
			//std::cerr<<*itr<<std::endl;
		}

		my_c->setPoints(myp);
		//my_w.statusBar->showMessage("Contour ok");
	}

}

