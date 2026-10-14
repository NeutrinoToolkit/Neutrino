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
#include "nPhysMaths.h"

// laplace filling in the pan thread (Stop button of runThread)
struct laplace_params {
    physD *img;
    std::vector<vec2i> pixels;
    int result=-1;
};

void laplace_fill_transl(void *params, int &iter) {
    laplace_params *my_params=static_cast<laplace_params*>(params);
    iter=0;
    my_params->result=physMath::laplace_fill(*my_params->img, my_params->pixels, 1e-6, 100000, &iter);
    if (my_params->result<0) iter=1; // nothing to interpolate from: not a thread error (handled by doIt)
}


Interpolate_path::Interpolate_path(neutrino *nparent) : nGenericPan(nparent)
{

    setupUi(this);

    region =  new nLine(this,1);
	// TODO: create something better to avoid line removal
	region->setPoints(QPolygonF()<<QPointF(10, 10)<<QPointF(10, 50)<<QPointF(50, 50));
	region->toggleClosedLine(true);

    show();

    connect(actionRegion, SIGNAL(triggered()), region, SLOT(togglePadella()));
    connect(actionBezier, SIGNAL(triggered()), region, SLOT(toggleBezier()));

    connect(doInterpolate, SIGNAL(clicked()), SLOT(doIt()));
    // the weight is the exponent of the inverse distance method only
    connect(method, &QComboBox::currentIndexChanged, this, [this](int) { weight->setEnabled(method->currentText()=="Inverse distance"); });
    weight->setEnabled(method->currentText()=="Inverse distance");
    interpolatePhys=nullptr;
    // the result was closed by the user: never replacePhys on a stale pointer
    connect(nparent, &neutrino::physDel, this, [this](nPhysD *buf) { if (buf==interpolatePhys) interpolatePhys=nullptr; });
}

void Interpolate_path::doIt() {
    saveDefaults();
    nPhysD *my_phys=getPhysFromCombo(image);
    if (my_phys) {
        QPolygonF regionPoly=region->poly(1).translated(my_phys->get_origin().x(),my_phys->get_origin().y());

        std::vector<vec2f> vecPoints(regionPoly.size());
        for(int k=0;k<regionPoly.size();k++) {
            vecPoints[k]=vec2f(regionPoly[k].x(),regionPoly[k].y());
        }
        
        QPolygonF regionPoly2=region->poly(20).translated(my_phys->get_origin().x(),my_phys->get_origin().y());

        std::vector<std::pair<vec2f, double> > vals;
        for(int k=0;k<regionPoly2.size();k++) {
            vec2f p(regionPoly2[k].x(),regionPoly2[k].y());
            double pval=my_phys->point(p);
            if (std::isfinite(pval)) {
                vals.push_back(std::make_pair(p, pval));
            }
        }
        if (vals.size()==0) {
            showMessage(tr("No valid values along the path"), 5000);
            return;
        }

        QRect rectRegion=regionPoly2.boundingRect().toRect().intersected(QRect(0,0,my_phys->getW(),my_phys->getH()));

        nPhysD *regionPath = new nPhysD(*my_phys);
        
        double ex=weight->value();
        
        regionPath->setShortName("Region path");
        regionPath->setName("path");

        if (method->currentText()=="Laplace") {
            // harmonic interpolation: the pixels around the region are the boundary conditions
            laplace_params my_params;
            my_params.img=regionPath;
            for (int i=rectRegion.left(); i<=rectRegion.right(); i++) {
                for (int j=rectRegion.top(); j<=rectRegion.bottom(); j++) {
                    vec2f pp(i,j);
                    if (point_inside_poly(pp,vecPoints)) {
                        my_params.pixels.push_back(vec2i(i,j));
                    }
                }
            }
            // progress: about 2 sweeps per pixel of the region size
            runThread(&my_params, laplace_fill_transl, "Laplace", 2*std::max(rectRegion.width(),rectRegion.height()));
            if (nThread.n_iter==-1) {
                delete regionPath;
                showMessage(tr("Interpolation stopped"), 5000);
                return;
            }
            if (my_params.result<0) {
                delete regionPath;
                showMessage(tr("No valid values around the region"), 5000);
                return;
            }
            regionPath->TscanBrightness();
            erasePrevious->setEnabled(true);
            if (erasePrevious->isChecked()) {
                interpolatePhys=nparent->replacePhys(regionPath,interpolatePhys,true);
            } else {
                nparent->addShowPhys(regionPath);
                interpolatePhys=regionPath;
            }
            return;
        }

        // inverse distance weighting of the values along the path
        QProgressDialog progress("Interpolate", "Stop", 0, rectRegion.width(), this);
        progress.setWindowModality(Qt::WindowModal);
        progress.show();
        bool canceled=false;
        for (int i=rectRegion.left(); i<=rectRegion.right(); i++) {
            if (progress.wasCanceled()) {
                canceled=true;
                break;
            }
            QApplication::processEvents();
            for (int j=rectRegion.top(); j<=rectRegion.bottom(); j++) {
                vec2f pp(i,j);
                if (point_inside_poly(pp,vecPoints)) {
                    double mean=0;
                    double weight=0;
                    bool onSample=false;
                    for(std::vector<std::pair<vec2f,double> >::iterator it=vals.begin();it!=vals.end();++it){
                        vec2f p=it->first;
                        double d2=(pp-p).mod2();
                        if (d2==0) { // on a sample point: take its value (1/0 weight)
                            mean=it->second;
                            onSample=true;
                            break;
                        }
                        // double wi=1.0/(pow(std::abs((pp-p).x()),ex)+pow(std::abs((pp-p).y()),ex));
                        double wi=1.0/pow(d2,ex/2.0); // inverse distance weighting 1/r^ex
                        mean+=wi*it->second;
                        weight+=wi;
                    }
                    regionPath->set(pp,onSample ? mean : mean/weight);
                }
            }
            progress.setValue(i-rectRegion.left());
        }
        if (canceled) {
            delete regionPath;
            showMessage(tr("Interpolation stopped"), 5000);
            return;
        }
        regionPath->TscanBrightness();

        erasePrevious->setEnabled(true);
        if (erasePrevious->isChecked()) {
            interpolatePhys=nparent->replacePhys(regionPath,interpolatePhys,true);
        } else {
            nparent->addShowPhys(regionPath);
            interpolatePhys=regionPath;
        }

    }
}
