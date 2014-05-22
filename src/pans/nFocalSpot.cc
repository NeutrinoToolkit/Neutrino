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

#include <iterator>

#include "nFocalSpot.h"
#include "neutrino.h"

nFocalSpot::nFocalSpot(neutrino *nparent, QString winname)
	: nGenericPan(nparent, winname)
{
	my_w.setupUi(this);

	decorate();

	calculate_stats();
	connect(nparent, SIGNAL(buffer_changed(nPhysD*)), this, SLOT(calculate_stats()));
	connect(my_w.zero_dsb, SIGNAL(valueChanged(double)), this, SLOT(calculate_stats()));
	connect(my_w.check_dsb, SIGNAL(valueChanged(double)), this, SLOT(calculate_stats()));

	nContour = new nLine(nparent);
}


void
nFocalSpot::calculate_stats()
{
	nPhysD *cur = nparent->getBuffer(-1);
	if (!cur) 
		return;

	if (cur->property.count("nFocalSpotDynamic") > 0) {
		std::cerr<<"nFocal dynamic image --- skip"<<std::endl;
		return;
	}

	// 1. find centroid
	vec2f centr;
	if (cur->get_origin() == vec2f(0,0)) {
		nPhysD decimated(*cur);
		phys_fast_gaussian_blur(decimated, 10);
		decimated.TscanBrightness();
	
		centr = vec2f(decimated.max_Tv_x, decimated.max_Tv_y);
		cur->set_origin(centr);
	} else
		centr = cur->get_origin();

	my_w.centroid_lbl->setText(QString("%1:%2").arg(centr.x()).arg(centr.y()));


	// 2. calculate integrals
	double c_value = cur->point(centr.x(),centr.y());
	double total_energy = cur->sum()-cur->getSurf()*my_w.zero_dsb->value();
	double above_th_energy = 0;
	int point_count = 0;
	double th = my_w.check_dsb->value()*(c_value-my_w.zero_dsb->value()) +my_w.zero_dsb->value() ;
	for (register size_t ii=0; ii<cur->getSurf(); ii++) 
		if (cur->point(ii) > th) {
			above_th_energy += cur->point(ii);
			point_count++;
		}

	above_th_energy -= point_count*my_w.zero_dsb->value();

	double energy_ratio;
	if (total_energy != 0)
		energy_ratio = 100*(above_th_energy/total_energy);
	else
		energy_ratio = 0;

	my_w.integral_lbl->setText(QString("%1/%2\n(%3\%)").arg(above_th_energy).arg(total_energy).arg(energy_ratio));

	//std::cerr<<"min/max: "<<cur->get_min()<<"/"<<cur->get_max()<<", surf: "<<cur->getSurf()<<", point_count: "<<point_count<<std::endl;
	
	find_contour();

}

void
nFocalSpot::find_contour(void)
{
	// marching squares algorithm
	
	bool contour_ok = false;

	nPhysD *cur = nparent->getBuffer(-1);
	if (!cur) 
		return;

	// 1. generate boolean map
	vec2f orig = cur->get_origin();
	double c_value = cur->point(orig.x(),orig.y());
	double th = my_w.check_dsb->value()*(c_value-my_w.zero_dsb->value()) +my_w.zero_dsb->value();

	nPhysImageF<short> bmap(cur->getW(), cur->getH(), 0);
	for (register size_t ii=0; ii<cur->getSurf(); ii++)
		if (cur->point(ii) > th) 
			bmap.set(ii, 1);

	// 2. cell map
	nPhysImageF<short> cmap(cur->getW()-1, cur->getH()-1, 0);
	for (register size_t ii=0; ii<cmap.getSurf(); ii++) {
		int xx = ii%cmap.getW();
		int yy = ii/cmap.getW();

		short cval = (bmap.point(xx,yy)<<3) + (bmap.point(xx+1,yy)<<2) + (bmap.point(xx+1, yy+1)<<1) + bmap.point(xx,yy+1);
		cmap.set(ii, cval);
	}
	// close boundary
	for (size_t ii=0; ii<cmap.getW(); ii++) {
		cmap.set(ii, 0, cmap.point(ii, 0) &~ 12);
		cmap.set(ii, 0, cmap.point(ii, cmap.getH()-1) &~ 3);
	}
	for (size_t ii=0; ii<cmap.getH(); ii++) {
		cmap.set(0, ii, cmap.point(0, ii) &~ 9);
		cmap.set(cmap.getW()-1, ii, cmap.point(cmap.getW()-1, ii) &~ 6);
	}

	if (my_w.show_cmap_cb->isChecked()) {
		nPhysD *mycmap = new nPhysD;
		*mycmap = cmap;
		mycmap->property["nFocalSpotDynamic"] = 1;
		nparent->addPhys(mycmap);
	}

	// 3. now find contours
	int stats[16];
	for (int i=0; i<16; i++)
		stats[i] = 0;
	for (register size_t ii=0; ii<cmap.getSurf(); ii++)
		stats[cmap.point(ii)] ++;

	int b_points = 0;
	for (int ii=1; ii<15; ii++) b_points+=stats[ii];

	std::cerr<<"[walker] There are "<<stats[0]<<" points under threshold, "<<stats[15]<<" points over threshold and "<<b_points<<" boundary points"<<std::endl;

	
	// find only main contour
	if (stats[0] == 0 || stats[15] == 0) {
		//std::cerr<<"no contour possible"<<std::endl;
		my_w.statusBar->showMessage("No contour possible");
	}

	int ls_x = orig.x();
	while (ls_x > -1 && cmap.point(ls_x, orig.y()) == 15)
		ls_x--;
	stats[cmap.point(ls_x, orig.y())]--;


	std::list<vec2f> contour(b_points);
	std::list<vec2f>::iterator itr = contour.begin(), itr_last = contour.begin();
	*itr = vec2f(ls_x, orig.y());

	while (itr != contour.end()) {
		short xx = (*itr).x();
		short yy = (*itr).y();
		short val = cmap.point(xx, yy);

		//std::cerr<<"[walker] I'm at "<<xx<<":"<<yy<<" with val "<<val<<std::endl;

		if (val==0 || val==15) {
			//std::cerr<<"Walker got sadly lost..."<<std::endl;
			my_w.statusBar->showMessage("Contour walk failed");
			break;
		}
		
		stats[val]--;

		// saddles: check central value and last movement
		if (val==5 || val==10) {
			short central = ((.25*cur->point(xx,yy) + cur->point(xx+1,yy+1) + cur->point(xx+1,yy) + cur->point(xx,yy+1)) > th) ? 1 : -1;
			short saddle_type = (val == 5) ? 1 : -1;

			vec2f last = *itr- *itr_last; // let's hope we're not starting with a saddle...

			//std::cerr<<"[Walker] Saddle point! central: "<<central<<std::endl;

			short xadd, yadd;

			if (last.x() > 0) {
				xadd = 1;
				if (saddle_type < 0)
					xadd*= -1*central;
			} else if (last.x() < 0){
				xadd = -1;
				if (saddle_type < 0)
					xadd*= -1*central;

			} else {
				xadd = last.y();
				if (saddle_type < 0) {
					xadd *= central;
				}
			}

			if (last.y() > 0) {
				yadd = 1;
				if (saddle_type > 0) 
					yadd *= -1*central;
			} else if (last.y() < 0){
				yadd = -1;
				if (saddle_type > 0)
					yadd*= -1*central;

			} else {
				yadd = -last.x();
				if (saddle_type > 0) {
					yadd *= central;
				}
			}

			xx+=xadd;
			yy+=yadd;



		} else if ((val&4) && !(val&8)) {
			yy--;
		} else if ((val&2) && !(val&4)) {
			xx++;
		} else if ((val&1) && !(val&2)) {
			yy++;
		} else if ((val&8) && !(val&1)) {
			xx--;
		}


		itr_last = itr;
		itr++;
		*itr = vec2f(xx,yy);

		if (*itr == *contour.begin()) {
			std::cerr<<"Closed contour!!"<<std::endl;
			contour_ok = true;
			break;
		}

	}


	nContour->setPoints(QPolygonF());
	if (contour_ok) {
		nContour->setPoints(QPolygonF());
		QPolygonF myp;
		for (itr = contour.begin(); itr != itr_last; ++itr) {
			myp<<QPointF((*itr).x(), (*itr).y());
		}

		nContour->setPoints(myp);
		my_w.statusBar->showMessage("Contour ok");
	}

}

void
nFocalSpot::bufferChanged(nPhysD *buf)
{
	nGenericPan::bufferChanged(buf);
	calculate_stats();
}
