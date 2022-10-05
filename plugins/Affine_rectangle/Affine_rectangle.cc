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
#include "Affine_rectangle.h"
#include "neutrino.h"

Affine_rectangle::Affine_rectangle(neutrino *nparent) : nGenericPan(nparent),
    affined(nullptr), l1(this,1)
{
	my_w.setupUi(this);
    l1.changeToolTip(panName()+"Points");
    l1.changeColorHolder("red");
    resetPoints();

    connect(my_w.actionReset, SIGNAL(triggered()), this, SLOT(resetPoints()));
    connect(my_w.actionLine, SIGNAL(triggered()), &l1, SLOT(togglePadella()));
    connect(my_w.transform,SIGNAL(pressed()),this,SLOT(affine()));

    show();
}

void Affine_rectangle::resetPoints() {
    QPolygonF poly;
    poly << QPointF(100,0) << QPointF(0,0) << QPointF(0,100);
    l1.setPoints(poly);
}

void Affine_rectangle::bufferChanged(nPhysD* buf) {
    nGenericPan::bufferChanged(buf);
    if (buf && buf==getPhysFromCombo(my_w.image1)) {
        l1.show();
    } else {
        l1.hide();
    }
}

void Affine_rectangle::affine() {
    std::vector<vec2f> rectPoly={ vec2f(my_w.newWidth->value(),0), vec2f(0,0),  vec2f(0,my_w.newHeight->value())};

    std::array<double,6> transVect=physMath::getAffine(rectPoly,l1.getPointsVec2f());

    nPhysD *my_phys=getPhysFromCombo(my_w.image1);
    if (my_phys) {
		
		double replaceVal=0.0;
		switch (my_w.defaultValue->currentIndex()) {
			case 0:
				replaceVal=std::numeric_limits<double>::quiet_NaN();
				break;
			case 1:
				replaceVal=my_phys->get_min();
				break;
			case 2:
				replaceVal=my_phys->get_max();
				break;
			case 3:
				replaceVal=0.5*(my_phys->get_min()+my_phys->get_max());
				break;
			case 4:
				replaceVal=0.0;
				break;
			default:
				WARNING("something is broken here");
				break;
		}

        unsigned int dx=my_w.newWidth->value();
        unsigned int dy=my_w.newHeight->value();
		
        DEBUG(affine(vec2f(0,0),transVect).x() << " " << affine(vec2f(0,0),transVect).y());

        nPhysD affinePhys(dx,dy,0.0,"affine");

        QProgressDialog progress("", "Cancel", 0, dx, this);
        progress.setCancelButton(0);
        progress.setWindowModality(Qt::WindowModal);
        progress.setValue(0);
        progress.show();

		for (unsigned int i=0; i<dx; i++) {
            progress.setValue(i);
			for (unsigned int j=0; j<dy; j++) {
                affinePhys.set(i,j,my_phys->getPoint(affine(vec2f(i,j),transVect),replaceVal));
			}
		}
        affinePhys.TscanBrightness();

        if (my_w.erasePrevious->isChecked()) {
            affined=nparent->replacePhys(new nPhysD(affinePhys),affined,true);
        } else {
            affined=new nPhysD(affinePhys);
            nparent->addShowPhys(affined);
        }
	}
}

vec2f Affine_rectangle::affine(vec2f in, std::array<double,6>& vec){
	return vec2f(in.x()*vec[0]+in.y()*vec[1]+vec[2],in.x()*vec[3]+in.y()*vec[4]+vec[5]);
}


