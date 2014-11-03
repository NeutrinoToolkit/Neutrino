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
	connect(my_w.duplicate, SIGNAL(clicked()), SLOT(duplicate()));
	regionPhys=NULL;
}

void nRegionPath::duplicate () {
    if (regionPhys==NULL) {
        doIt();
    }
	regionPhys=NULL;
}

void
nRegionPath::doIt() {
	if (currentBuffer) {
		saveDefaults();
		nPhysD *image=getPhysFromCombo(my_w.image);
		if (image) {
            double replaceVal=getReplaceVal();
            QPolygonF regionPoly=region->poly(1);
            regionPoly=regionPoly.translated(image->get_origin().x(),image->get_origin().y());
            regionPoly=regionPoly.intersected(QPolygonF(QRectF(0,0,image->getW(),image->getH())));
            qDebug() << PRINTVAR(regionPoly);
            
            QRect rectRegion=regionPoly.boundingRect().toRect();
            
            vec2f my_offset(0,0);
            DEBUG(PRINTVAR(my_offset));
            
            nPhysD *regionPath = new nPhysD();

            if (my_w.crop->isChecked()) {
                regionPath = new nPhysD();
                *regionPath=image->sub(rectRegion.x(), rectRegion.y(), rectRegion.width(), rectRegion.height());
                my_offset+=vec2f(rectRegion.left(),rectRegion.top());
                DEBUG(PRINTVAR(my_offset));
            } else {
                regionPath = new nPhysD(*image);
            }
            if (!my_w.inverse->isChecked()) {
                regionPath->set(replaceVal);
            } 
            
			regionPath->setShortName("Region path");
			regionPath->setName("path");
			QProgressDialog progress("Extracting", "Stop", 0, rectRegion.width(), this);
			progress.setWindowModality(Qt::WindowModal);
			progress.show();
			for (int i=rectRegion.left(); i<=rectRegion.right(); i++) {
				if (progress.wasCanceled()) break;
				QApplication::processEvents();
				for (int j=rectRegion.top(); j<=rectRegion.bottom(); j++) {
					if (regionPoly.containsPoint(QPoint(i,j),Qt::OddEvenFill)==my_w.inverse->isChecked()) {
						regionPath->set(bidimvec<int>(i,j)-my_offset,replaceVal);
					} else {
						regionPath->set(bidimvec<int>(i,j)-my_offset,image->point(i,j));
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
			default:
				val=my_w.replace->text().toDouble();
				break;
		}
	}
	return val;
}
