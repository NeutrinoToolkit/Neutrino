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
#include "Rotate.h"
#include "neutrino.h"

Rotate::Rotate(neutrino *nparent) : nGenericPan(nparent)
{
	my_w.setupUi(this);

    show();

	connect(my_w.valueAngle, SIGNAL(valueChanged(double)), this, SLOT(doRotateLive()));
    connect(my_w.doIt,SIGNAL(pressed()),this,SLOT(keepCopy()));
	connect(my_w.image,SIGNAL(activated(int)),this,SLOT(doRotateLive()));
    rotated=nullptr;
	doRotateLive();
}

void Rotate::doRotateLive () {
	double alpha=my_w.valueAngle->value();
	nPhysD *image=getPhysFromCombo(my_w.image);
	if (image) {
		if (image!=rotated) {
            if (my_w.sameSize->isChecked()) {
                rotated=nparent->replacePhys(new nPhysD(image->fast_rotated(alpha,getReplaceVal(image))),rotated, true);
            } else {
                rotated=nparent->replacePhys(new nPhysD(image->rotated(alpha,getReplaceVal(image))),rotated, true);
            }
		} else {
			my_w.statusbar->showMessage("Can't work on this image",5000);
		}
	}
}

void Rotate::keepCopy () {
	doRotateLive();
	if (rotated) {
        nPhysD *newRotated=new nPhysD(*rotated);
		nparent->addPhys(newRotated);
	}
}

double Rotate::getReplaceVal(nPhysD* image) {
	double val=0.0;
    switch (my_w.defaultValue->currentIndex()) {
        case 0:
            val=std::numeric_limits<double>::quiet_NaN();
            break;
        case 1:
            val=image->get_min();
            break;
        case 2:
            val=image->get_max();
            break;
        case 3:
            val=0.5*(image->get_min()+image->get_max());
            break;
        case 4:
            val=0.0;
            break;
        default:
            WARNING("something is broken here");
            break;
    }
    return val;
}
