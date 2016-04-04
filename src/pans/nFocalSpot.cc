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
	nContour = new nLine(nparent);
	nContour->setParentPan(panName,3);
	
	decorate();
	connect(nparent, SIGNAL(bufferChanged(nPhysD*)), this, SLOT(calculate_stats()));
	connect(my_w.zero_dsb, SIGNAL(editingFinished()), this, SLOT(calculate_stats()));
	connect(my_w.check_dsb, SIGNAL(editingFinished()), this, SLOT(calculate_stats()));

	calculate_stats();
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
	vec2 centr;
	if (cur->get_origin() == vec2(0,0)) {
		nPhysD decimated(*cur);
		phys_fast_gaussian_blur(decimated, 10);
		decimated.TscanBrightness();
	
		centr = vec2(decimated.max_Tv_x, decimated.max_Tv_y);
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
    for (size_t ii=0; ii<cur->getSurf(); ii++)
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

	my_w.integral_lbl->setText(QString("Threshold integral %:\n%1").arg(energy_ratio));

	//std::cerr<<"min/max: "<<cur->get_min()<<"/"<<cur->get_max()<<", surf: "<<cur->getSurf()<<", point_count: "<<point_count<<std::endl;
	
	double c_integral = find_contour();
	
	//double contour_ratio = contour_integral();
	my_w.integral_lbl->setText(my_w.integral_lbl->text()+QString("\nContour integral %:\n%1\n(total: %2)").arg(100*c_integral/total_energy).arg(c_integral));

}

double
nFocalSpot::find_contour(void)
{
	// marching squares algorithm
	
	bool contour_ok = false;

	nPhysD *cur = nparent->getBuffer(-1);
	if (!cur) 
		return -1;

	// 1. generate boolean map
	vec2 orig = cur->get_origin();
	double c_value = cur->point(orig.x(),orig.y());
	double th = my_w.check_dsb->value()*(c_value-my_w.zero_dsb->value()) +my_w.zero_dsb->value();

	nPhysImageF<short> bmap(cur->getW(), cur->getH(), 0);
    for (size_t ii=0; ii<cur->getSurf(); ii++)
		if (cur->point(ii) > th) 
			bmap.set(ii, 1);

	// 2. cell map
	nPhysImageF<short> cmap(cur->getW()-1, cur->getH()-1, 0);
    for (size_t ii=0; ii<cmap.getSurf(); ii++) {
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
    for (size_t ii=0; ii<cmap.getSurf(); ii++)
		stats[cmap.point(ii)] ++;

	int b_points = 0;
	for (int ii=1; ii<15; ii++) b_points+=stats[ii];

	DEBUG(5,"[walker] There are "<<stats[0]<<" points under threshold, "<<stats[15]<<" points over threshold and "<<b_points<<" boundary points"<<std::endl);

	
	// find only main contour
	if (stats[0] == 0 || stats[15] == 0) {
		//std::cerr<<"no contour possible"<<std::endl;
		my_w.statusBar->showMessage("No contour possible");
	}

	int ls_x = orig.x();
	while (ls_x > -1 && cmap.point(ls_x, orig.y()) == 15)
		ls_x--;
	stats[cmap.point(ls_x, orig.y())]--;


	std::list<vec2> contour(b_points);
	std::list<vec2>::iterator itr = contour.begin(), itr_last = contour.begin();
	*itr = vec2(ls_x, orig.y());

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

			vec2 last = *itr- *itr_last; // let's hope we're not starting with a saddle...

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
		*itr = vec2(xx,yy);

		if (*itr == *contour.begin()) {
			//std::cerr<<"Closed contour!!"<<std::endl;
			contour_ok = true;
			itr_last++;
			break;
		}

	}


	nContour->setPoints(QPolygonF());
	if (contour_ok) {

		// set polygon
		nContour->setPoints(QPolygonF());
		QPolygonF myp;
		for (itr = contour.begin(); itr != itr_last; ++itr) {
			myp<<QPointF((*itr).x(), (*itr).y());
			//std::cerr<<*itr<<std::endl;
		}

		// get stats
		vec2f c_center = cur->get_origin();
		vec2f c_scale = cur->get_scale();
		double min_r = vmath::td<double>(contour.front()-c_center, c_scale).mod();
		double max_r = min_r;
		for (itr = contour.begin(); itr != itr_last; ++itr) {
			double dd = vmath::td<double>((*itr)-c_center, c_scale).mod();
			if (dd > max_r) max_r = dd;
			if (dd < min_r) min_r = dd;
		}

		my_w.contour_lbl->setText(QString("min. radius: %1\nmax. radius: %2").arg(min_r).arg(max_r));

		nContour->setPoints(myp);
		my_w.statusBar->showMessage("Contour ok");

		double intg_contour = contour_integral(contour, itr_last);
		return intg_contour;
	}
    return -1;
}


double
nFocalSpot::contour_integral(std::list<vec2> &contour, std::list<vec2>::iterator &itr_last)
{
	nPhysD *cur = nparent->getBuffer(-1);
	if (!cur) 
		return -1;

	vec2 bbox_inf = contour.front(), bbox_sup = contour.front();
	nPhysD check_image(*cur);
	check_image.TscanBrightness();
	double check_val = check_image.get_min() - 1;
	//double c_integral = 0;

	for (std::list<vec2>::iterator itr = contour.begin(); itr != itr_last; ++itr) {
		bbox_inf = vmath::min(bbox_inf, *itr);
		bbox_sup = vmath::max(bbox_sup, *itr);
		check_image.set((*itr).x(), (*itr).y(), check_val);
	}

	// coutour bbox subimage (to perform integral on)
	nPhysD intg_image = check_image.sub(bbox_inf.x(), bbox_inf.y(), bbox_sup.x()-bbox_inf.x()+1, bbox_sup.y()-bbox_inf.y()+1);
	intg_image.set_origin(cur->get_origin()-bbox_inf);
	intg_image.property["nFocalSpotDynamic"] = 1;

	// integrate by scanline fill
	double intg=0;
	std::list<vec2> up_pl, scan_pl, tmplist;

	vec2 starting_point = intg_image.get_origin();

	for (int xx=starting_point.x(); intg_image.point(xx, starting_point.y()) != check_val; xx++) {
		up_pl.push_back(vec2(xx, starting_point.y()));
	}
	
	for (int xx=starting_point.x(); intg_image.point(xx, starting_point.y()) != check_val; xx--) {
		up_pl.push_front(vec2(xx, starting_point.y()));
	}
	

	//std::cerr<<"walk starting from "<<up_pl.front()<<" to "<<up_pl.back()<<std::endl;

	int line_check =0;


	while (!up_pl.empty()) {
		
		//scan_pl = up_pl;
		//up_pl.clear();

		tmplist.clear();
		scan_pl.clear();
		std::list<vec2>::iterator itr = up_pl.begin(), itrf = up_pl.begin();
		itrf++;

		while (itrf != up_pl.end()) {
			if (((*itrf).x()-(*itr).x()) > 1) {
				//std::cerr<<"separation at "<<*itr<<" -- "<<*itrf<<std::endl;
				scan_pl.push_back(*itr);
				tmplist.clear();
				tmplist.push_back(*itrf);


				//std::cerr<<"line "<<line_check<<": sep/walk starting from "<<*itr<<" to "<<*itrf<<std::endl;
				vec2 ref_sx = *itr, ref_dx = *itrf;
				while (intg_image.point(scan_pl.back(), check_val) != check_val) {
					ref_sx+=vec2(1,0);
					scan_pl.push_back(ref_sx);
				}
				//std::cerr<<"line "<<line_check<<": sep/walk starting from "<<scan_pl.back()<<" to "<<tmplist.front()<<std::endl;
				while (intg_image.point(tmplist.front(), check_val) != check_val) {
					ref_dx -= vec2(1,0);
					tmplist.push_front(ref_dx);
					//std::cerr<<"\t\tsep/walking to "<<tmplist.front()<<" - "<<intg_image.point(tmplist.front())<<std::endl;
				}
				//std::cerr<<"line "<<line_check<<": sep/walk starting from "<<scan_pl.back()<<" to "<<tmplist.front()<<std::endl;
				scan_pl.splice(scan_pl.end(), tmplist);

				itr++;
				itrf++;


			} else scan_pl.push_back(*itr); // ovvero se linea senza separazioni somma tutti i punti

			itr++;
			itrf++;
		}
		
		//std::cerr<<"line "<<line_check<<": walk starting from "<<scan_pl.front()<<" to "<<scan_pl.back()<<std::endl;

		
		while (intg_image.point(scan_pl.front(), check_val) != check_val) {
			scan_pl.push_front(scan_pl.front()+vec2(-1, 0));
			//std::cerr<<"--------------"<<intg_image.getPoint(scan_pl.front())<<std::endl;
		}

		while (intg_image.point(scan_pl.back(), check_val) != check_val) {
			scan_pl.push_back(scan_pl.back()+vec2(1, 0));
			//std::cerr<<"--------------"<<intg_image.point(scan_pl.back(), check_val)<<std::endl;
		}

		//std::cerr<<"line "<<line_check<<": walk starting from "<<scan_pl.front()<<" to "<<scan_pl.back()<<std::endl;
		//if (line_check == 38)
		//	break;

		up_pl.clear();

		while (!scan_pl.empty()) {
			vec2 pp = scan_pl.front();
			scan_pl.pop_front();
			if (intg_image.point(pp, check_val) != check_val) {
				intg+=intg_image.point(pp);
				intg_image.set(pp.x(), pp.y(), check_val);

				up_pl.push_back(vec2(pp.x(), pp.y()+1));
				up_pl.push_back(vec2(pp.x(), pp.y()-1));
				//std::cerr<<"point read: "<<pp<<std::endl;
			} 
			//else std::cerr<<"--------------- cippacazzo ------------------"<<pp<<std::endl;
		}

		line_check++;

	}

	//nparent->addPhys(intg_image);
	return intg;

//	bool isUp = false, isIn = false, isDown = false;
//	int point_count = 0;
//	for (size_t yy=0; yy<intg_image.getH(); yy++) {
//		size_t xx = 0;
//		isIn = false; isUp = false, isDown = false;
//		while (xx<intg_image.getW()) {
//			double the_point = intg_image.point(xx,yy);
//			if (the_point == check_val && !isUp) {
//				isUp = true;
//				xx++;
//				continue;
//			} else if (the_point != check_val && isUp) {
//				isIn = true;
//			}
//
//			if (the_point == check_val && isIn) {
//				isDown = true;
//				isIn = false;
//				isUp = false;
//				xx++;
//				continue;
//			} else if (the_point != check_val && isDown) {
//				isDown = false;
//				xx++;
//				continue;
//			}
//			
//			
//
//			if (isIn) {
//				c_integral += the_point;
//				point_count++;
//			}
//			xx++;
//
//		}
//
//		std::cerr<<"point_count: "<<point_count<<std::endl;
//		if (isIn) {
//			std::cerr<<"WARNING: boundary problem"<<std::endl;
//			return -1;
//		}
//	}
//	return c_integral-point_count*my_w.zero_dsb->value();
}



void
nFocalSpot::bufferChanged(nPhysD *buf)
{
	nGenericPan::bufferChanged(buf);
	calculate_stats();
}
