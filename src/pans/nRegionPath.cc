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
#include "nRegionPath.h"
#include "neutrino.h"

nRegionPath::nRegionPath(neutrino *nparent, QString winname)
: nGenericPan(nparent, winname)
{

	my_w.setupUi(this);

	region =  new nLine(nparent);
	region->setParentPan(panName,1);
	// TODO: create something better to avoid line removal
	region->setPoints(QPolygonF()<<QPointF(10, 10)<<QPointF(10, 50)<<QPointF(50, 50));
	region->toggleClosedLine(true);

	decorate();

	connect(my_w.actionLoadPref, SIGNAL(triggered()), this, SLOT(loadSettings()));
	connect(my_w.actionSavePref, SIGNAL(triggered()), this, SLOT(saveSettings()));
	connect(my_w.actionLine, SIGNAL(triggered()), region, SLOT(togglePadella()));
	connect(my_w.actionBezier, SIGNAL(triggered()), region, SLOT(toggleBezier()));

	connect(my_w.doIt, SIGNAL(clicked()), SLOT(doIt()));
	connect(my_w.doMask, SIGNAL(clicked()), SLOT(doMask()));
	regionPhys=NULL;
}

void
nRegionPath::doIt() {
	if (currentBuffer) {
		saveDefaults();
		nPhysD *image=getPhysFromCombo(my_w.image);
		if (image) {
            double replaceVal=getReplaceVal();
            
			QPolygon regionPoly=region->getLine().toPolygon();
			QRect rectRegion=region->boundingRect().toRect().intersected(QRect(0,0,image->getW(),image->getH()));
            
			nPhysD *regionPath = new nPhysD();
            bidimvec<int> offset(0,0);
            if (my_w.crop->isChecked()) {
                regionPath = new nPhysD();
                *regionPath=image->sub(rectRegion.x(), rectRegion.y(), rectRegion.width(), rectRegion.height());
                offset+=bidimvec<int>(rectRegion.x(), rectRegion.y());
                regionPath->set_origin(image->get_origin()-offset);
            } else {
                if (my_w.negative->isChecked()) {
                    regionPath = new nPhysD(*image);
                } else {
                    regionPath = new nPhysD(image->getW(),image->getH(),replaceVal);
                } 
            }
			regionPath->setShortName("Region mask");
			regionPath->setName("mask");
			QProgressDialog progress("Extracting", "Stop", 0, rectRegion.width(), this);
			progress.setWindowModality(Qt::WindowModal);
			progress.show();
			for (int i=rectRegion.left(); i<rectRegion.right(); i++) {
				if (progress.wasCanceled()) break;
				QApplication::processEvents();
				for (int j=rectRegion.top(); j<rectRegion.bottom(); j++) {
					if (regionPoly.containsPoint(QPoint(i,j),Qt::OddEvenFill)==my_w.negative->isChecked()) {
						regionPath->set(bidimvec<int>(i,j)-offset,replaceVal);
					} else {
						regionPath->set(bidimvec<int>(i,j)-offset,image->point(i,j));
                    }

				}
				progress.setValue(i-rectRegion.left());
			}
            regionPath->TscanBrightness();
			regionPhys=nparent->replacePhys(regionPath,regionPhys);
		}
	}
}

void nRegionPath::doMask() {
	if (currentBuffer) {
		saveDefaults();
		nPhysD *image=getPhysFromCombo(my_w.image);
		if (image) {
			QPolygon regionPoly=region->poly(region->numPoints).toPolygon();
			QRect rectRegion=region->boundingRect().toRect().intersected(QRect(0,0,image->getW(),image->getH()));
            double defValue=my_w.negative->isChecked()? 1.0: 0.0;
			nPhysD *regionPath = new nPhysD();
            bidimvec<int> offset(0,0);
            if (my_w.crop->isChecked()) {
                regionPath = new nPhysD(rectRegion.width(), rectRegion.height(),defValue);
                offset+=bidimvec<int>(rectRegion.x(), rectRegion.y());
                regionPath->set_origin(image->get_origin()-offset);
            } else {
                regionPath = new nPhysD(image->getW(),image->getH(),defValue);
            }
			regionPath->setShortName("Region mask");
			regionPath->setName("mask");
			QProgressDialog progress("Extracting", "Stop", 0, rectRegion.width(), this);
			progress.setWindowModality(Qt::WindowModal);
			progress.show();
			for (int i=rectRegion.left(); i<rectRegion.right(); i++) {
				if (progress.wasCanceled()) break;
				QApplication::processEvents();
				for (int j=rectRegion.top(); j<rectRegion.bottom(); j++) {
					if (regionPoly.containsPoint(QPoint(i,j),Qt::OddEvenFill)==my_w.negative->isChecked()) {
						regionPath->set(bidimvec<int>(i,j)-offset,0.0);
					} else {
						regionPath->set(bidimvec<int>(i,j)-offset,1.0);
                    }
                    
				}
				progress.setValue(i-rectRegion.left());
			}
            regionPath->TscanBrightness();
			regionPhys=nparent->replacePhys(regionPath,regionPhys);
		}
	}
}

double nRegionPath::getReplaceVal() {
	double val=0.0;
	nPhysD *image=getPhysFromCombo(my_w.image);
	if (image) {
		switch (my_w.defaultValue->currentIndex()) {
			case 0:
				val=std::numeric_limits<double>::quiet_NaN();
				break;
			case 1:
				val=image->Tminimum_value;
				break;
			case 2:
				val=image->Tmaximum_value;
				break;
			case 3:
				val=0.5*(image->Tminimum_value+image->Tmaximum_value);
				break;
			case 4:
				val=0.0;
				break;
			case 5:
				val=my_w.replace->text().toDouble();
				break;
			default:
				WARNING("something is broken here");
				break;
		}
	}
	return val;
}
