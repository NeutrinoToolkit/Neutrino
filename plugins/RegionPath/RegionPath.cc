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
#include "RegionPath.h"
#include "neutrino.h"

RegionPath::RegionPath(neutrino *nparent) : nGenericPan(nparent),
    region(this,1),
    regionPhys(nullptr)
{

    setupUi(this);

	// TODO: create something better to avoid line removal
    region.setPoints(QPolygonF()<<QPointF(10, 10)<<QPointF(10, 50)<<QPointF(50, 50));
    region.toggleClosedLine(true);

    show();

    connect(actionRegion, SIGNAL(triggered()), &region, SLOT(togglePadella()));
    connect(actionBezier, SIGNAL(triggered()), &region, SLOT(toggleBezier()));

    connect(doItB, SIGNAL(clicked()), SLOT(doIt()));
}

void RegionPath::doIt() {
    saveDefaults();
    nPhysD *my_phys=getPhysFromCombo(image);
    if (!my_phys) {
        statusBar()->showMessage(tr("Select an image"),5000);
        return;
    }
    if (my_phys) {
        double replaceVal=getReplaceVal();
        // the options read once (they were read by the OpenMP threads)
        const bool inv=inverse->isChecked();
        QPolygonF regPoly=region.poly(1);
        regPoly=regPoly.translated(my_phys->get_origin().x(),my_phys->get_origin().y());
        
        std::vector<vec2f> vecPoints(regPoly.size());
        for(int k=0;k<regPoly.size();k++) {
            vecPoints[k]=vec2f(regPoly[k].x(),regPoly[k].y());
        }
        
        // QRect regRect=regPoly.boundingRect().toRect();
        // inside the image: a region going out of it wrote with negative coordinates
        // (cast to unsigned: undefined, on ARM 0, i.e. the first row/column overwritten)
        QRect regRect=regPoly.boundingRect().toAlignedRect().intersected(QRect(0,0,my_phys->getW(),my_phys->getH()));
        
        nPhysD *regPath = new nPhysD(*my_phys);
        
        if (!inv) {
            regPath->set(replaceVal);
        } 
        
        // regPath->setShortName("Region path");
        // regPath->setName("path");
        regPath->setShortName("Region path");
        regPath->setName("RegionPath("+my_phys->getName()+")");
        // QProgressDialog progress("Extracting", "Stop", 0, regRect.width(), this);
        // progress.setWindowModality(Qt::WindowModal);
        // progress.show();
        // for (int i=regRect.left(); i<=regRect.right(); i++) {
        //     if (progress.wasCanceled()) break;
        //     QApplication::processEvents();
// #pragma omp parallel for
        //     for (int j=regRect.top(); j<=regRect.bottom(); j++) {
        //         vec2f pp(i,j);
        //         if (point_inside_poly(pp,vecPoints)==inverse->isChecked()) {
        //             regPath->set(pp,replaceVal);
        //         } else {
        //             regPath->set(pp,my_phys->point(i,j));
        //         }
        //     }
        //     progress.setValue(i-regRect.left());
        // }
        // blocks of rows in parallel (threads were started for every column), pixel centres tested
        // (the corner (i,j) shifted the mask by half a pixel); Stop discards the result
        if (!regRect.isEmpty()) {
            const int block=64;
            QProgressDialog progress("Extracting", "Stop", 0, regRect.height(), this);
            progress.setWindowModality(Qt::WindowModal);
            progress.show();
            for (int j0=regRect.top(); j0<=regRect.bottom(); j0+=block) {
                QApplication::processEvents();
                if (progress.wasCanceled()) {
                    delete regPath;
                    statusBar()->showMessage(tr("Stopped"),5000);
                    return;
                }
                const int j1=std::min(j0+block-1, regRect.bottom());
#pragma omp parallel for
                for (int j=j0; j<=j1; j++) {
                    for (int i=regRect.left(); i<=regRect.right(); i++) {
                        if (point_inside_poly(vec2f(i+0.5,j+0.5),vecPoints)==inv) {
                            regPath->set(i,j,replaceVal);
                        } else {
                            regPath->set(i,j,my_phys->point(i,j));
                        }
                    }
                }
                progress.setValue(j1-regRect.top()+1);
            }
        }
        if (crop->isChecked()) {
            // regRect is inside the image: the crop is not padded with 0 any more
            if (regRect.isEmpty()) {
                delete regPath;
                statusBar()->showMessage(tr("The region is outside the image"),5000);
                return;
            }
            *regPath=regPath->sub(regRect);
            regPath->setShortName("Region path");
            regPath->setName("RegionPath("+my_phys->getName()+")");
        }
        regPath->TscanBrightness();
        erasePrevious->setEnabled(true);
        if (erasePrevious->isChecked()) {
            regionPhys=nparent->replacePhys(regPath,regionPhys, true);
        } else {
            nparent->addShowPhys(regPath);
            regionPhys=regPath;
        }

    }
}

double RegionPath::getReplaceVal() {
	double val=0.0;
    nPhysD *my_phys=getPhysFromCombo(image);
	// if (image) {
	if (my_phys) {
        switch (defaultValue->currentIndex()) {
			case 0:
				val=std::numeric_limits<double>::quiet_NaN();
				break;
			case 1:
                val=my_phys->get_min();
				break;
			case 2:
                val=my_phys->get_max();
				break;
			case 3:
                val=0.5*(my_phys->get_min()+my_phys->get_max());
				break;
			case 4:
				val=0.0;
				break;
			default:
                // val=replace->text().toDouble();
                {
                    // the user locale, or the C one (toDouble alone: "1,5" was 0)
                    bool ok;
                    val=QLocale().toDouble(replace->text(),&ok);
                    if (!ok) val=QLocale::c().toDouble(replace->text(),&ok);
                    if (!ok) statusBar()->showMessage(tr("Cannot read the value ")+replace->text(),5000);
                }
				break;
		}
	}
	return val;
}
