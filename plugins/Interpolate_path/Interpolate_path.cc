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
#include "Interpolate_path.h"
#include "neutrino.h"


Interpolate_path::Interpolate_path(neutrino *nparent) : nGenericPan(nparent)
{

	my_w.setupUi(this);

    region =  new nLine(this,1);
	// TODO: create something better to avoid line removal
	region->setPoints(QPolygonF()<<QPointF(10, 10)<<QPointF(10, 50)<<QPointF(50, 50));
	region->toggleClosedLine(true);

    show();

    connect(my_w.actionRegion, SIGNAL(triggered()), region, SLOT(togglePadella()));
	connect(my_w.actionBezier, SIGNAL(triggered()), region, SLOT(toggleBezier()));

	connect(my_w.doIt, SIGNAL(clicked()), SLOT(doIt()));
	connect(my_w.duplicate, SIGNAL(clicked()), SLOT(duplicate()));
    interpolatePhys=nullptr;
}

void Interpolate_path::duplicate () {
    if (interpolatePhys==nullptr) {
        doIt();
    }
    interpolatePhys=nullptr;
}

void Interpolate_path::doIt() {
    saveDefaults();
    nPhysD *image=getPhysFromCombo(my_w.image);
    if (image) {
        QPolygonF regionPoly=region->poly(1).translated(image->get_origin().x(),image->get_origin().y());

        std::vector<vec2f> vecPoints(regionPoly.size());
        for(int k=0;k<regionPoly.size();k++) {
            vecPoints[k]=vec2f(regionPoly[k].x(),regionPoly[k].y());
        }
        
        nPhysD *regionPath = new nPhysD(*image);

        QPolygonF regionPoly2=region->poly(20).translated(image->get_origin().x(),image->get_origin().y());

        std::vector<std::pair<vec2f, double> > vals;
        for(int k=0;k<regionPoly2.size();k++) {
            vec2f p(regionPoly2[k].x(),regionPoly2[k].y());
            double pval=regionPath->point(p);
            if (std::isfinite(pval)) {
                vals.push_back(std::make_pair(p, pval));
            }
        }
        
        QRect rectRegion=regionPoly2.boundingRect().toRect();
        
        double ex=my_w.weight->value();
        
        regionPath->setShortName("Region path");
        regionPath->setName("path");
        QProgressDialog progress("Interpolate", "Stop", 0, rectRegion.width(), this);
        progress.setWindowModality(Qt::WindowModal);
        progress.show();
        for (int i=rectRegion.left(); i<=rectRegion.right(); i++) {
            if (progress.wasCanceled()) break;
            QApplication::processEvents();
            for (int j=rectRegion.top(); j<=rectRegion.bottom(); j++) {
                vec2f pp(i,j);
                if (point_inside_poly(pp,vecPoints)) {
                    double mean=0;
                    double weight=0;
                    for(std::vector<std::pair<vec2f,double> >::iterator it=vals.begin();it!=vals.end();++it){
                        vec2f p=it->first;
                        double wi=1.0/(pow(std::abs((pp-p).x()),ex)+pow(std::abs((pp-p).y()),ex));
                        if (std::isfinite(wi) &&std::isfinite(it->second)) {
                            mean+=wi*it->second;
                            weight+=wi;
                        }
                    }
                    regionPath->set(pp,mean/weight);
                }
            }
            progress.setValue(i-rectRegion.left());
        }
        regionPath->reset_display();
        interpolatePhys=nparent->replacePhys(regionPath,interpolatePhys);
    }
}
